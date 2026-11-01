"""LLM question extraction from interview-experience posts.

A separate track next to parse/index: a post is indexed as text whether or not the extractor has read it,
and extraction waits in the queue while the model is unavailable. Every extracted question must quote the
post (`original_text`); a quote that does not match the post (similarity below `min_quote_similarity`) means
the question was made up, and it is discarded and counted (the fabrication rate).

Reposts and lightly edited copies (SimHash within `simhash_bits` bits) are extracted once.
"""

import json
import logging
import re

from pydantic import BaseModel, field_validator

from sde_rag import db
from sde_rag.agents import AgentError, get_registry
from sde_rag.pipeline.indexing import split_paragraphs
from sde_rag.pipeline.stages import drop_mentions, upsert_question
from sde_rag.settings import section
from sde_rag.text import clean_question, hamming, question_hash, quote_similarity, sha1, simhash

log = logging.getLogger("sde_rag.extraction")
ROUNDS = ("oa", "phone", "onsite", "behavioral", "unknown")
_ROUND_WORDS = {"online assessment": "oa", "screen": "phone", "phone screen": "phone", "on-site": "onsite",
                "virtual onsite": "onsite", "technical": "onsite", "coding": "onsite", "system design": "onsite",
                "hr": "behavioral", "culture": "behavioral"}
_VAGUE_COMPANY = re.compile(r"^(?:an?|the|one|some)?\s*(?:big|large|small|top|major|faang|maang|tech|"
                            r"well[- ]known|fortune \d+|startup|company|firm|unicorn|mnc|\s)+$", re.I)


class ExtractedQuestion(BaseModel):
    text: str
    original_text: str = ""
    answer_text: str = ""
    round: str = "unknown"

    @field_validator("round", mode="before")
    @classmethod
    def _round(cls, v):
        v = str(v or "unknown").strip().lower()
        return v if v in ROUNDS else _ROUND_WORDS.get(v, "unknown")

    @field_validator("answer_text", "original_text", mode="before")
    @classmethod
    def _text(cls, v):
        return "" if v is None else str(v)


class Extraction(BaseModel):
    is_interview_content: bool = True
    company: str | None = None
    role: str | None = None
    level: str | None = None
    year: int | None = None
    questions: list[ExtractedQuestion] = []

    @field_validator("questions", mode="before")
    @classmethod
    def _decode(cls, v):
        # Some local models emit the array as a JSON-encoded string.
        return json.loads(v) if isinstance(v, str) else (v or [])

    @field_validator("year", mode="before")
    @classmethod
    def _year(cls, v):
        m = re.search(r"(?:19|20)\d\d", str(v or ""))
        return int(m.group()) if m else None


def extract_prompt(title: str, part: str, n: int, total: int) -> list[tuple[str, str]]:
    system = ("You extract interview questions from posts about software-engineering job interviews, for a study "
              "index. Answer only with the structured output.")
    where = f" (part {n} of {total})" if total > 1 else ""
    user = f"""Post title: {title}

Read the post{where} below and report:
- is_interview_content: true if the post is about job interviews (an interview experience, a list of
  interview questions, interview preparation); false for a tutorial, news or opinion piece that only
  mentions interviews in passing. When false, return no questions.
- company, role, level, year: as the post states them (company name only, e.g. "Amazon"; null for vague
  phrases such as "a big tech company"); null when not stated.
- questions: every question the post says was asked in an interview, or presents as an interview question.
  Not rhetorical questions, not questions the author asks the reader, not the author's own wonderings.
  For each:
  - text: the question as an interviewer would ask it, self-contained, one sentence, in English (translate
    if the post is in another language). For a coding problem, name the task, e.g. "Implement an LRU cache
    with O(1) get and put."
  - original_text: the exact words of the post where this question appears, copied verbatim (up to about
    200 characters, in the post's language). Never paraphrase or translate this field.
  - answer_text: the answer or approach the post gives, in at most three sentences; empty if none.
  - round: one of oa, phone, onsite, behavioral, unknown.

Post{where}:
<<<
{part}
>>>"""
    return [("system", system), ("user", user)]


_NONE = {"none", "null", "n/a", "na", "unknown", "not stated", "not specified", "unspecified", "-"}


def _stated(v) -> str | int | None:
    """Models write "None" or "unknown" for fields the post does not state."""
    if isinstance(v, str):
        v = v.strip().strip(".")
        return None if not v or v.lower() in _NONE or len(v) > 60 else v
    return v


def _company(v: str | None) -> str | None:
    v = _stated(v)
    return None if not v or _VAGUE_COMPANY.match(v) else v


def _call(reg, title: str, parts: list[str], min_chars: int, heartbeat, trace: list | None = None
          ) -> list[tuple[str, Extraction]]:
    """One call per part; a part the model cannot answer for is split in half and tried again."""
    out, queue, total = [], [(p, i + 1) for i, p in enumerate(parts)], len(parts)
    while queue:
        part, n = queue.pop(0)
        try:
            out.append((part, reg.structured("extractor", Extraction, extract_prompt(title, part, n, total),
                                             trace=trace)))
        except AgentError:
            if len(part) < 2 * min_chars:
                raise
            cut = part.rfind("\n\n", 0, len(part) // 2 + 500)
            cut = cut if cut > min_chars else len(part) // 2
            queue[:0] = [(part[:cut], n), (part[cut:], n)]
        heartbeat()
    return out


def _near_duplicate(doc, bits: int) -> int | None:
    rows = db.conn().execute("SELECT id, simhash FROM documents WHERE kind = 'experience' AND id != ? AND "
                             "simhash IS NOT NULL AND extract_status IN ('done', 'not_interview')", (doc["id"],))
    for r in rows:
        if hamming(r["simhash"], doc["simhash"]) <= bits:
            return r["id"]
    return None


def extract_stage(limit: int, *, enabled: bool = True, heartbeat=lambda: None) -> dict:
    cfg = section("extract")
    stats = {"docs": 0, "not_interview": 0, "duplicates": 0, "failed": 0, "proposed": 0, "kept": 0,
             "unquoted": 0, "new_questions": 0}
    reg = get_registry()
    if not enabled or not reg.usable("extractor") or limit <= 0:
        stats["skipped"] = True
        return stats
    max_attempts = int(cfg.get("max_attempts", 3))
    docs = db.conn().execute(
        "SELECT id, title, clean_text, simhash FROM documents WHERE kind = 'experience' "
        "AND status IN ('parsed', 'indexed') AND (extract_status IS NULL OR extract_status = 'failed') "
        "ORDER BY extract_status IS NOT NULL, id LIMIT ?", (limit,)).fetchall()
    fingerprint = f"llm:{reg.fingerprint('extractor')}"
    for doc in docs:
        doc = dict(doc)
        text = doc["clean_text"] or ""
        if doc["simhash"] is None:
            doc["simhash"] = simhash(text)
            db.conn().execute("UPDATE documents SET simhash = ? WHERE id = ?", (doc["simhash"], doc["id"]))
        dup = _near_duplicate(doc, int(cfg.get("simhash_bits", 3)))
        if dup is not None:
            with db.tx() as t:
                drop_mentions(t, doc["id"], llm=True)
                t.execute("UPDATE documents SET extract_status = 'duplicate', dup_of = ? WHERE id = ?", (dup, doc["id"]))
            stats["duplicates"] += 1
            continue
        parts = split_paragraphs(text, int(cfg.get("part_chars", 6000)))[:int(cfg.get("max_parts", 8))]
        try:
            results = _call(reg, doc["title"], parts, int(cfg.get("min_part_chars", 1500)), heartbeat)
        except AgentError as e:
            attempts = db.conn().execute("SELECT extract_attempts FROM documents WHERE id = ?",
                                         (doc["id"],)).fetchone()[0] + 1
            db.conn().execute("UPDATE documents SET extract_status = ?, extract_attempts = ?, extract_error = ? "
                              "WHERE id = ?", ("dead" if attempts >= max_attempts else "failed", attempts,
                                               str(e)[:2000], doc["id"]))
            log.warning("extraction of document %s failed (attempt %d): %s", doc["id"], attempts, str(e)[:200])
            stats["failed"] += 1
            continue
        doc_stats = _store(doc, results, fingerprint, float(cfg.get("min_quote_similarity", 0.8)))
        for k in ("proposed", "kept", "unquoted", "new_questions"):
            stats[k] += doc_stats[k]
        stats["docs"] += 1
        stats["not_interview"] += doc_stats["status"] == "not_interview"
    stats["fabrication_rate"] = round(stats["unquoted"] / stats["proposed"], 3) if stats["proposed"] else 0.0
    return stats


def _store(doc: dict, results: list[tuple[str, Extraction]], fingerprint: str, min_sim: float) -> dict:
    interview = any(x.is_interview_content for _, x in results)
    meta = {}
    for _, x in results:
        for k in ("company", "role", "level", "year"):
            if meta.get(k) is None:
                meta[k] = _company(x.company) if k == "company" else _stated(getattr(x, k))
    st = {"proposed": 0, "kept": 0, "unquoted": 0, "new_questions": 0, "unquoted_examples": []}
    with db.tx() as t:
        drop_mentions(t, doc["id"], llm=True)
        seen: set[str] = set()
        ordinal = 0
        for part, x in results if interview else []:
            for q in x.questions:
                text = clean_question(q.text)[:500]
                if len(text) < 8:
                    continue
                st["proposed"] += 1
                # The quote must come from the post; the part it was read from first, then the whole post.
                if quote_similarity(q.original_text, part) < min_sim and \
                        quote_similarity(q.original_text, doc["clean_text"] or "") < min_sim:
                    st["unquoted"] += 1
                    if len(st["unquoted_examples"]) < 3:
                        st["unquoted_examples"].append(q.original_text[:120] or text[:120])
                    continue
                h = question_hash(text)
                if h in seen:
                    continue
                seen.add(h)
                qid, created = upsert_question(t, text, q.answer_text.strip()[:3000])
                st["new_questions"] += created
                t.execute(
                    "INSERT INTO question_mentions (question_id, doc_id, mention_hash, text, answer, section_path, "
                    "ordinal, company, role, level, round, year, extractor, created_at) "
                    "VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(mention_hash) DO UPDATE SET "
                    "company = excluded.company, role = excluded.role, level = excluded.level, "
                    "round = excluded.round, year = excluded.year",
                    (qid, doc["id"], sha1(f"{doc['id']}:{h}"), text, q.answer_text.strip()[:3000], ordinal,
                     meta.get("company"), meta.get("role"), meta.get("level"), q.round, meta.get("year"),
                     fingerprint, db.now_iso()))
                t.execute("UPDATE questions SET stats_dirty = 1 WHERE id = ?", (qid,))
                st["kept"] += 1
                ordinal += 1
        status = "done" if interview else "not_interview"
        t.execute("UPDATE documents SET extract_status = ?, extractor = ?, extract_stats = ?, extract_error = NULL "
                  "WHERE id = ?", (status, fingerprint, json.dumps({**st, **meta}, ensure_ascii=False), doc["id"]))
    st["status"] = status
    return st
