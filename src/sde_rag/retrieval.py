"""Hybrid retrieval: BM25 and vector search in parallel lists, fused with reciprocal rank fusion (RRF).

Access control is a pre-filter in both lists (auth.AccessPolicy.visible_filter) and is checked again with
can_read before anything reaches a prompt.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from sde_rag import db
from sde_rag.auth import Principal, access_policy, safe_can_read
from sde_rag.index import fts
from sde_rag.index.embedder import get_embedder
from sde_rag.index.vectors import get_index
from sde_rag.index.versions import check_model
from sde_rag.settings import section
from sde_rag.text import truncate

log = logging.getLogger("sde_rag.retrieval")


@dataclass
class Hit:
    chunk_id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


def rrf(rankings: dict[str, list[int]], k: int = 60) -> list[Hit]:
    scores: dict[int, float] = defaultdict(float)
    ranks: dict[int, dict[str, int]] = defaultdict(dict)
    for name, ids in rankings.items():
        for rank, cid in enumerate(ids, 1):
            scores[cid] += 1.0 / (k + rank)
            ranks[cid][name] = rank
    return [Hit(cid, s, ranks[cid]) for cid, s in sorted(scores.items(), key=lambda kv: -kv[1])]


def hybrid(principal: Principal, queries: list[str], *, kinds: list[str] | None = None, category: str | None = None,
           tags: list[str] | None = None, k: int | None = None, trace: dict | None = None) -> list[Hit]:
    cfg = section("query")
    k = k or int(cfg.get("retrieve_k", 50))
    version = check_model()
    chunk_filter = access_policy().visible_filter(principal)
    queries = [q for q in queries if q and q.strip()] or [""]
    rankings: dict[str, list[int]] = {}
    for i, q in enumerate(queries):
        rankings[f"bm25:{i}"] = [cid for cid, _ in fts.search(version, [q], k, kinds=kinds, category=category,
                                                               tags=tags, chunk_filter=chunk_filter)]
    try:
        ix = get_index(version)
        mask = ix.mask(kinds=kinds, category=category, tags=tags, chunk_filter=chunk_filter)
        qvecs = get_embedder().embed_queries(queries)
        for i, v in enumerate(qvecs):
            rankings[f"vec:{i}"] = [cid for cid, _ in ix.search(v, k, mask)]
    except Exception as e:  # degrade to keyword search only
        log.warning("vector search failed, using BM25 only: %s", e)
        if trace is not None:
            trace["degraded"] = f"vector search failed: {type(e).__name__}: {str(e)[:200]}"
    hits = rrf(rankings, int(cfg.get("rrf_k", 60)))
    if trace is not None:
        trace["lists"] = {name: len(ids) for name, ids in rankings.items()}
    return hits[:k]


def _chunks(ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {r["id"]: dict(r) for r in db.conn().execute(
        f"SELECT id, kind, question_id, doc_id, ordinal, embed_text, category, tags, tenant_id, visibility, acl "
        f"FROM chunks WHERE id IN ({marks})", ids)}


def _question_source(qid: int) -> dict:
    r = db.conn().execute(
        "SELECT d.url, d.origin, d.title FROM question_mentions m JOIN documents d ON d.id = m.doc_id "
        "WHERE m.question_id = ? ORDER BY d.id LIMIT 1", (qid,)).fetchone()
    return dict(r) if r else {"url": "", "origin": "", "title": ""}


def load_context(principal: Principal, hits: list[Hit], limit: int, *, answer_chars: int = 1200,
                 trace: dict | None = None) -> list[dict]:
    """Hits -> prompt-ready items with stable labels ([Q12], [D34#2]). Unreadable chunks are dropped."""
    chunks = _chunks([h.chunk_id for h in hits])
    ordered = [chunks[h.chunk_id] for h in hits if h.chunk_id in chunks]
    allowed = safe_can_read(principal, ordered)
    denied = sum(1 for ok in allowed if not ok)
    if trace is not None and denied:
        trace["denied_by_policy"] = denied
    items, seen_q = [], set()
    for ch, ok in zip(ordered, allowed):
        if not ok:
            continue
        if ch["kind"] == "question":
            if ch["question_id"] in seen_q:
                continue
            seen_q.add(ch["question_id"])
            q = db.conn().execute("SELECT * FROM questions WHERE id = ?", (ch["question_id"],)).fetchone()
            if q is None:
                continue
            src = _question_source(q["id"])
            items.append({
                "label": f"Q{q['id']}", "kind": "question", "chunk_id": ch["id"], "question_id": q["id"],
                "title": q["text"], "text": f"Question: {q['text']}\n"
                + (f"Source answer: {truncate(q['answer'], answer_chars)}" if q["answer"] else "Source answer: (none)"),
                "url": src["url"], "source": src["origin"], "category": q["category"], "tags": ch["tags"].split(),
                "mention_count": q["mention_count"], "distinct_sources": q["distinct_sources"], "heat": q["heat"]})
        else:
            d = db.conn().execute("SELECT title, url, origin, published_at FROM documents WHERE id = ?",
                                  (ch["doc_id"],)).fetchone()
            if d is None:
                continue
            items.append({
                "label": f"D{ch['doc_id']}#{ch['ordinal']}", "kind": "doc", "chunk_id": ch["id"],
                "doc_id": ch["doc_id"], "title": d["title"], "text": truncate(ch["embed_text"], answer_chars * 2),
                "url": d["url"], "source": d["origin"], "published_at": d["published_at"],
                "category": ch["category"], "tags": ch["tags"].split()})
        if len(items) >= limit:
            break
    return items


def topic_questions(principal: Principal, queries: list[str], *, category: str | None, tags: list[str] | None,
                    k: int, trace: dict | None = None) -> list[dict]:
    """Common questions of a topic: relevance to the request fused with how often the question shows up."""
    relevant = hybrid(principal, queries, kinds=["question"], category=category, tags=tags, k=max(k * 4, 100),
                      trace=trace)
    rankings = {"relevance": [h.chunk_id for h in relevant]}
    if category or tags:
        version = check_model()
        where, args = fts.filter_sql(kinds=["question"], category=category,
                                     chunk_filter=access_policy().visible_filter(principal))
        tag_clause = ""
        if tags:
            tag_clause = (f" AND EXISTS (SELECT 1 FROM question_tags qt WHERE qt.question_id = c.question_id AND "
                          f"qt.tag IN ({','.join('?' * len(tags))}))")
            args += list(tags)
        rows = db.conn().execute(
            f"SELECT c.id FROM chunks c JOIN questions q ON q.id = c.question_id WHERE c.index_version = ?{where}"
            f"{tag_clause} ORDER BY q.heat DESC, q.mention_count DESC LIMIT ?", [version, *args, k * 4]).fetchall()
        rankings["heat"] = [r[0] for r in rows]
    else:
        heat = {r["id"]: r["heat"] for r in db.conn().execute(
            f"SELECT c.id, q.heat FROM chunks c JOIN questions q ON q.id = c.question_id "
            f"WHERE c.id IN ({','.join('?' * len(rankings['relevance'])) or 'NULL'})", rankings["relevance"])}
        rankings["heat"] = sorted(heat, key=lambda cid: -heat[cid])
    fused = rrf(rankings, int(section("query").get("rrf_k", 60)))
    return load_context(principal, fused, k, answer_chars=300, trace=trace)


def describe_filters(filters: dict) -> str:
    return json.dumps({k: v for k, v in filters.items() if v}, ensure_ascii=False)
