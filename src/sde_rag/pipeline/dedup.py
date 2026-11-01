"""Near-duplicate questions: the same question in different words becomes one canonical question.

Each question's text alone (no answer) is embedded once, as an e5 "query:" (the symmetric-similarity form),
and kept in question_vectors. A question not yet checked is compared with every other question:
  similarity >= merge_at and the same content words -> merged right away
  otherwise similarity >= judge_from               -> a candidate pair for the dedup_judge role (per-run budget)
Similarity alone cannot merge: measured on the M1 data, pairs above 0.97 still include "private IP" vs
"public IP" and "WeakSet" vs "WeakMap", one word apart. Below 0.92 almost every pair was a different question.
Merging moves the mentions to the kept question, leaves the dropped wording as an alias (parsing it again
finds the kept question) and records the merge in question_merges.
"""

import difflib
import json
import logging
import re

import numpy as np
from pydantic import BaseModel, field_validator

from sde_rag import db
from sde_rag.agents import AgentError, get_registry
from sde_rag.index.embedder import get_embedder
from sde_rag.pipeline.stages import delete_question
from sde_rag.pipeline.tagging import refresh_chunk_tags
from sde_rag.settings import section
from sde_rag.text import sha1, tokens

log = logging.getLogger("sde_rag.dedup")


def embed_questions(heartbeat=lambda: None, batch: int = 256) -> int:
    emb = get_embedder()
    have = {r[0]: (r[1], r[2]) for r in db.conn().execute("SELECT question_id, model, text_hash FROM question_vectors")}
    todo = [(r[0], r[1]) for r in db.conn().execute("SELECT id, text FROM questions ORDER BY id")
            if have.get(r[0]) != (emb.name, sha1(r[1]))]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        vecs = emb.embed_queries([t for _, t in part]).astype(np.float32)
        with db.tx() as t:
            for (qid, text), v in zip(part, vecs):
                t.execute("INSERT INTO question_vectors (question_id, model, text_hash, vector) VALUES (?, ?, ?, ?) "
                          "ON CONFLICT(question_id) DO UPDATE SET model = excluded.model, "
                          "text_hash = excluded.text_hash, vector = excluded.vector",
                          (qid, emb.name, sha1(text), v.tobytes()))
        heartbeat()
    return len(todo)


_FILLER = set("""a an the of in on at for to from with by about using via and or vs versus is are was were be been
being do does did done can could would should will shall may might must you your yours i me my we our us they
their it its this that these those there here some any how what why when where which who whom whose explain
describe tell define definition mean means meaning give example examples please briefly list name way ways
""".split())


# "What is the output of the code below?" means a different snippet in every source: never a duplicate.
_POINTS_AT_CONTEXT = re.compile(
    r"\b(?:below|following|above|given|this|these|attached|preceding)\s+(?:\w+\s+){0,2}?"
    r"(?:code|snippet|program|query|queries|spec|specs|command|commands|script|function|example|output|"
    r"file|config|configuration|manifest|statement|expression|fqdn|diagram|table|schema)\b"
    r"|\b(?:code|snippet|program|output|query)\s+(?:below|above)\b", re.I)


def points_at_context(text: str) -> bool:
    return bool(_POINTS_AT_CONTEXT.search(text))


def content_words(text: str) -> tuple[tuple[str, ...], list[str]]:
    """The words that carry the question, as (distinct words in order, all in order): no question words or
    filler, plural -s dropped."""
    seq = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
           for w in tokens(text) if w not in _FILLER]
    return tuple(dict.fromkeys(seq)), seq


def _load() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """(question ids, unit vectors, texts, checked flags), in id order."""
    rows = db.conn().execute(
        "SELECT q.id, q.text, q.dedup_checked, v.vector FROM questions q JOIN question_vectors v "
        "ON v.question_id = q.id WHERE v.model = ? ORDER BY q.id", (get_embedder().name,)).fetchall()
    if not rows:
        return np.zeros(0, np.int64), np.zeros((0, 0), np.float32), [], np.zeros(0, bool)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    return ids, mat, [r[1] for r in rows], np.array([bool(r[2]) for r in rows])


def _keep_order(t, a: int, b: int) -> tuple[int, int]:
    """(keep, drop): more mentions wins, then having an answer, then the older question."""
    def key(qid):
        n = t.execute("SELECT COUNT(*) FROM question_mentions WHERE question_id = ?", (qid,)).fetchone()[0]
        has_answer = t.execute("SELECT answer != '' FROM questions WHERE id = ?", (qid,)).fetchone()[0]
        return (-n, -has_answer, qid)
    return (a, b) if key(a) <= key(b) else (b, a)


def merge_questions(t, a: int, b: int, *, how: str, similarity: float | None) -> int | None:
    """Fold one of a, b into the other; returns the kept id (None when either is gone already)."""
    rows = {r["id"]: r for r in t.execute("SELECT * FROM questions WHERE id IN (?, ?)", (a, b))}
    if len(rows) < 2:
        return None
    keep, drop = _keep_order(t, a, b)
    k, d = rows[keep], rows[drop]
    t.execute("UPDATE question_mentions SET question_id = ? WHERE question_id = ?", (keep, drop))
    t.execute("UPDATE question_aliases SET question_id = ? WHERE question_id = ?", (keep, drop))
    t.execute("INSERT INTO question_aliases (norm_hash, question_id, created_at) VALUES (?, ?, ?) "
              "ON CONFLICT(norm_hash) DO UPDATE SET question_id = excluded.question_id", (d["norm_hash"], keep, db.now_iso()))
    if len(d["answer"] or "") > len(k["answer"] or ""):
        t.execute("UPDATE questions SET answer = ?, section_path = ?, answer_verdict = ?, answer_review_id = ? "
                  "WHERE id = ?", (d["answer"], d["section_path"], d["answer_verdict"], d["answer_review_id"], keep))
    t.execute("UPDATE questions SET difficulty = COALESCE(difficulty, ?), category = COALESCE(category, ?), "
              "stats_dirty = 1, updated_at = ? WHERE id = ?", (d["difficulty"], d["category"], db.now_iso(), keep))
    t.execute("INSERT INTO question_tags (question_id, tag, origin) SELECT ?, tag, origin FROM question_tags "
              "WHERE question_id = ? ON CONFLICT DO NOTHING", (keep, drop))
    t.execute("INSERT INTO question_merges (kept_id, dropped_id, dropped_text, similarity, how, created_at) "
              "VALUES (?, ?, ?, ?, ?, ?)", (keep, drop, d["text"], similarity, how, db.now_iso()))
    delete_question(t, drop)
    return keep


def find_duplicates(heartbeat=lambda: None) -> dict:
    cfg = section("dedup")
    merge_at, judge_from = float(cfg.get("merge_at", 0.95)), float(cfg.get("judge_from", 0.92))
    per_question = int(cfg.get("pairs_per_question", 2))
    ids, mat, texts, checked = _load()
    stats = {"checked": 0, "merged": 0, "pairs": 0}
    pending = np.flatnonzero(~checked)
    if len(pending) == 0:
        return stats
    alive = np.ones(len(ids), dtype=bool)
    eligible = np.array([not points_at_context(x) for x in texts], dtype=bool)
    words: dict[int, tuple] = {}

    def same_words(i: int, j: int) -> bool:
        """The same content words, in about the same order ("generators from iterators" is not
        "iterators from generators")."""
        for k in (i, j):
            if k not in words:
                words[k] = content_words(texts[k])
        a, b = words[i], words[j]
        return bool(a[0]) and set(a[0]) == set(b[0]) and \
            difflib.SequenceMatcher(None, a[1], b[1], autojunk=False).ratio() >= 0.75

    kept: set[int] = set()
    for start in range(0, len(pending), 512):
        block = pending[start:start + 512]
        sims = mat[block] @ mat.T
        with db.tx() as t:
            for row, i in enumerate(block):
                if not alive[i]:
                    continue
                s = sims[row]
                s[i] = -1
                cand = np.flatnonzero((s >= judge_from) & alive & eligible & eligible[i])
                cand = cand[np.argsort(-s[cand])][:per_question]
                for j in cand:
                    if not (alive[i] and alive[j]):
                        continue
                    if s[j] >= merge_at and same_words(i, j):
                        keep = merge_questions(t, int(ids[i]), int(ids[j]), how="auto", similarity=float(s[j]))
                        if keep is not None:
                            alive[i if keep == ids[j] else j] = False
                            kept.add(keep)
                            stats["merged"] += 1
                    else:
                        a, b = sorted((int(ids[i]), int(ids[j])))
                        stats["pairs"] += t.execute(
                            "INSERT INTO dedup_pairs (a, b, similarity, created_at) VALUES (?, ?, ?, ?) "
                            "ON CONFLICT DO NOTHING", (a, b, float(s[j]), db.now_iso())).rowcount
                if alive[i]:
                    t.execute("UPDATE questions SET dedup_checked = 1 WHERE id = ?", (int(ids[i]),))
                stats["checked"] += 1
        heartbeat()
    refresh_chunk_tags(sorted(kept))
    return stats


class Verdicts(BaseModel):
    """Flat lists of pair numbers: local models break JSON for lists of objects more often."""
    same: list[int] = []
    different: list[int] = []

    @field_validator("same", "different", mode="before")
    @classmethod
    def _decode(cls, v):
        if isinstance(v, str):
            v = json.loads(v) if v.strip().startswith("[") else [x for x in v.replace(",", " ").split() if x]
        return [int(x) for x in v or []]


def judge_prompt(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    items = "\n".join(f"[{i}]\nA: {a}\nB: {b}" for i, (a, b) in enumerate(pairs))
    system = ("You decide whether two interview questions are the same question, for merging duplicates in a "
              "study index. Answer only with the structured output.")
    user = f"""Put the number of every pair below into exactly one of two lists.

same: an interviewer asking either one would expect the same answer; only the wording, language or small
details differ (e.g. "What is the GIL?" and "Explain Python's global interpreter lock").
different: one asks about something else, is clearly narrower or broader, or compares different things
(e.g. "What is a process?" and "What is the difference between a process and a thread?"; "private key" and
"public key").

{items}"""
    return [("system", system), ("user", user)]


def judge_pairs(limit: int, *, enabled: bool = True, heartbeat=lambda: None) -> dict:
    stats = {"judged": 0, "same": 0, "failed": 0}
    reg = get_registry()
    if not enabled or not reg.usable("dedup_judge") or limit <= 0:
        stats["skipped"] = True
        return stats
    batch = int(section("dedup").get("judge_batch", 10))
    rows = db.conn().execute(
        "SELECT p.a, p.b, p.similarity, qa.text AS ta, qb.text AS tb FROM dedup_pairs p "
        "JOIN questions qa ON qa.id = p.a JOIN questions qb ON qb.id = p.b WHERE p.verdict IS NULL "
        "ORDER BY p.similarity DESC LIMIT ?", (limit,)).fetchall()
    who = f"llm:{reg.fingerprint('dedup_judge')}"
    kept: set[int] = set()
    pending = [rows[i:i + batch] for i in range(0, len(rows), batch)]
    while pending:
        part = pending.pop(0)
        try:
            result: Verdicts = reg.structured("dedup_judge", Verdicts, judge_prompt([(r["ta"], r["tb"]) for r in part]))
        except AgentError as e:
            if len(part) > 2:            # malformed output is usually batch-specific: split and retry
                pending[:0] = [part[:len(part) // 2], part[len(part) // 2:]]
            else:
                stats["failed"] += len(part)
                log.warning("dedup judging of %d pairs failed: %s", len(part), str(e)[:200])
            continue
        both = set(result.same) & set(result.different)    # contradictory: ask again another time
        verdicts = {i: True for i in result.same if i not in both} | {i: False for i in result.different
                                                                      if i not in both}
        with db.tx() as t:
            for n, r in enumerate(part):
                if n not in verdicts:
                    continue
                same = verdicts[n]
                t.execute("UPDATE dedup_pairs SET verdict = ?, judged_by = ?, judged_at = ? WHERE a = ? AND b = ?",
                          ("same" if same else "different", who, db.now_iso(), r["a"], r["b"]))
                stats["judged"] += 1
                if same:
                    keep = merge_questions(t, r["a"], r["b"], how="judge", similarity=r["similarity"])
                    if keep is not None:
                        kept.add(keep)
                        stats["same"] += 1
        heartbeat()
    refresh_chunk_tags(sorted(kept))
    return stats


def dedup_stage(*, judge_enabled: bool, judge_limit: int, heartbeat=lambda: None) -> dict:
    return {"embedded": embed_questions(heartbeat), **find_duplicates(heartbeat),
            **{f"judge_{k}": v for k, v in judge_pairs(judge_limit, enabled=judge_enabled, heartbeat=heartbeat).items()}}
