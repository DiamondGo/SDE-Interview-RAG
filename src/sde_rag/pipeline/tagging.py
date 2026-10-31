"""Tagging: rules and source hints first (free, immediate), then the LLM tagger refines within a per-run budget.

Rule tags come from alias matches in the question and its section path (and repeated matches in the answer);
they are made again when the taxonomy version changes.
Source hints are the category/tags configured for the repository the question came from; they are trusted
and kept. LLM tags replace rule tags; unknown keywords the LLM proposes go to tag_candidates for review.
"""

import json
import logging
from collections import Counter

from pydantic import BaseModel, field_validator

from sde_rag import db
from sde_rag.agents import AgentError, get_registry
from sde_rag.settings import section
from sde_rag.taxonomy import taxonomy

log = logging.getLogger("sde_rag.tagging")


def _hints(t, qid: int) -> tuple[Counter, set[str]]:
    cats, tags = Counter(), set()
    for (h,) in t.execute("SELECT d.hints FROM question_mentions m JOIN documents d ON d.id = m.doc_id "
                          "WHERE m.question_id = ?", (qid,)):
        hint = json.loads(h or "{}")
        if hint.get("category"):
            cats[hint["category"]] += 1
        for tag in hint.get("tags", []):
            resolved = taxonomy().resolve(tag)
            if resolved:
                tags.add(resolved)
    return cats, tags


def rule_tag(question: str, section_path: str, answer: str) -> tuple[list[str], str | None]:
    tax = taxonomy()
    head = tax.match(f"{question}\n{section_path}")
    body = tax.match(answer[:800])
    tags = [t for t, _ in head.most_common(5)]
    tags += [t for t, n in body.most_common(5) if n >= 2 and t not in tags]
    return tags[:6], tax.category_of([t for t, _ in head.most_common(3)])


def _set_tags(t, qid: int, tags: dict[str, str], category: str | None, status: str, difficulty: str | None = None):
    t.execute("DELETE FROM question_tags WHERE question_id = ?", (qid,))
    for tag, origin in tags.items():
        t.execute("INSERT INTO question_tags (question_id, tag, origin) VALUES (?, ?, ?)", (qid, tag, origin))
    t.execute("UPDATE questions SET category = ?, difficulty = COALESCE(?, difficulty), tag_status = ?, "
              "tag_version = ? WHERE id = ?", (category, difficulty, status, taxonomy().version, qid))


def rule_tag_stage(limit: int = 100000) -> dict:
    c = db.conn()
    # New questions, and rule tags made with an older taxonomy (LLM tags stay until the LLM tags again).
    rows = c.execute("SELECT id, text, section_path, answer FROM questions WHERE tag_status IS NULL OR "
                     "(tag_status = 'rule' AND COALESCE(tag_version, 0) < ?) LIMIT ?",
                     (taxonomy().version, limit)).fetchall()
    changed = []
    with db.tx() as t:
        for r in rows:
            tags, category = rule_tag(r["text"], r["section_path"], r["answer"])
            hint_cats, hint_tags = _hints(t, r["id"])
            hint_cat = hint_cats.most_common(1)[0][0] if hint_cats else None
            all_tags = {tag: "rule" for tag in tags} | {tag: "hint" for tag in hint_tags}
            category = category or hint_cat or taxonomy().category_of(all_tags)
            _set_tags(t, r["id"], all_tags, category, "rule")
            changed.append(r["id"])
    refresh_chunk_tags(changed)
    return {"rule_tagged": len(changed)}


class TagItem(BaseModel):
    idx: int
    category: str
    tags: list[str] = []
    difficulty: str | None = None


class TagBatch(BaseModel):
    items: list[TagItem]

    @field_validator("items", mode="before")
    @classmethod
    def _decode(cls, v):
        # Some local models emit the array as a JSON-encoded string.
        return json.loads(v) if isinstance(v, str) else v


def tag_prompt(questions: list[dict]) -> list[tuple[str, str]]:
    tax = taxonomy()
    cats = "\n".join(f"- {k}: {v}" for k, v in tax.categories.items())
    tags = "\n".join(f"- {cat}: {', '.join(tax.tags_in(cat))}" for cat in tax.categories)
    items = "\n".join(f"[{i}] {q['text']}" + (f"  (section: {q['section_path']})" if q["section_path"] else "")
                      for i, q in enumerate(questions))
    system = ("You label software-engineering interview questions for a study index. "
              "Answer only with the structured output.")
    user = f"""Label every question below.

For each question give:
- idx: the number in brackets
- category: exactly one of these keys
{cats}
- tags: 1 to 5 tags, preferably from this list (any category's tags may be used):
{tags}
  If an important topic is missing from the list, you may add one short lowercase keyword (e.g. "graphql").
- difficulty: easy, medium or hard (for an interview at a mid-level software engineer position)

Questions:
{items}"""
    return [("system", system), ("user", user)]


def llm_tag_stage(limit: int, *, enabled: bool = True, heartbeat=lambda: None) -> dict:
    stats = {"llm_tagged": 0, "llm_failed": 0, "candidates": 0}
    reg = get_registry()
    if not enabled or not reg.usable("tagger") or limit <= 0:
        stats["skipped"] = True
        return stats
    cfg = section("ingest")
    batch_size, min_batch = int(cfg.get("tag_batch", 20)), int(cfg.get("tag_min_batch", 5))
    rows = [dict(r) for r in db.conn().execute(
        "SELECT id, text, section_path FROM questions WHERE tag_status = 'rule' ORDER BY heat DESC, id LIMIT ?",
        (limit,))]
    pending = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    tax = taxonomy()
    retried: set[int] = set()
    while pending:
        batch = pending.pop(0)
        try:
            result: TagBatch = reg.structured("tagger", TagBatch, tag_prompt(batch))
        except AgentError as e:
            if len(batch) > min_batch:   # malformed output is usually batch-specific: split and retry
                half = len(batch) // 2
                pending[:0] = [batch[:half], batch[half:]]
            else:
                stats["llm_failed"] += len(batch)
                log.warning("tagging batch of %d failed: %s", len(batch), str(e)[:300])
            continue
        by_idx = {it.idx: it for it in result.items}
        changed = []
        with db.tx() as t:
            for i, q in enumerate(batch):
                it = by_idx.get(i)
                if it is None or it.category not in tax.categories:
                    continue
                _, hint_tags = _hints(t, q["id"])
                tags = {tag: "hint" for tag in hint_tags}
                for raw in it.tags[:5]:
                    resolved = tax.resolve(raw)
                    if resolved:
                        tags.setdefault(resolved, "llm")
                    elif raw.strip():
                        kw = raw.strip().lower()[:40]
                        t.execute("INSERT INTO tag_candidates (keyword, count, first_seen, last_seen) "
                                  "VALUES (?, 1, ?, ?) ON CONFLICT(keyword) DO UPDATE SET count = count + 1, "
                                  "last_seen = excluded.last_seen", (kw, db.now_iso(), db.now_iso()))
                        stats["candidates"] += 1
                difficulty = it.difficulty if it.difficulty in ("easy", "medium", "hard") else None
                _set_tags(t, q["id"], tags, it.category, "llm", difficulty)
                changed.append(q["id"])
        refresh_chunk_tags(changed)
        stats["llm_tagged"] += len(changed)
        # Models sometimes drop items from a long list: give the missing ones one more, smaller batch.
        done = set(changed)
        missing = [q for q in batch if q["id"] not in done and q["id"] not in retried]
        if missing:
            retried.update(q["id"] for q in missing)
            pending.append(missing)
        heartbeat()
    return stats


def question_tags(qid: int) -> list[str]:
    return [r[0] for r in db.conn().execute("SELECT tag FROM question_tags WHERE question_id = ? ORDER BY tag", (qid,))]


def refresh_chunk_tags(qids: list[int]) -> None:
    """Tags live in chunk metadata and the FTS tags column; changing them needs no re-embedding."""
    if not qids:
        return
    with db.tx() as t:
        for qid in qids:
            q = t.execute("SELECT category FROM questions WHERE id = ?", (qid,)).fetchone()
            if q is None:
                continue
            tags = " ".join(question_tags(qid))
            for (chunk_id,) in t.execute("SELECT id FROM chunks WHERE kind = 'question' AND question_id = ?",
                                         (qid,)).fetchall():
                t.execute("UPDATE chunks SET category = ?, tags = ? WHERE id = ?", (q["category"], tags, chunk_id))
                t.execute("UPDATE chunk_fts SET tags = ? WHERE rowid = ?", (f"{q['category'] or ''} {tags}", chunk_id))
        db.set_meta("index_dirty", "1")
