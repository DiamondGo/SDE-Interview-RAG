"""Chunks, FTS rows and embeddings for the active index version.

Question chunks: one per canonical question (question + representative answer). Document chunks: paragraphs
of interview-experience documents. Embeddings are committed in small batches, so an interrupted run keeps
what it finished; the vector snapshot is refreshed at the end.
"""

import logging

import numpy as np

from sde_rag import db
from sde_rag.index.embedder import get_embedder
from sde_rag.index.vectors import load_vectors
from sde_rag.index.versions import check_model
from sde_rag.pipeline.stages import delete_chunk, due_documents, mark_failed
from sde_rag.pipeline.tagging import question_tags
from sde_rag.settings import section
from sde_rag.taxonomy import rule_tags, taxonomy
from sde_rag.text import sha1

log = logging.getLogger("sde_rag.indexing")


def question_embed_text(q, answer_chars: int) -> str:
    head = f"{q['section_path']}\n" if q["section_path"] else ""
    answer = (q["answer"] or "")[:answer_chars]
    return f"{head}{q['text']}\n{answer}".strip()


MAX_CODE_BLOCK_CHARS = 8000


def split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Paragraph-aligned chunks. A fenced code block stays whole unless it exceeds MAX_CODE_BLOCK_CHARS."""
    blocks, buf, in_code = [], [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        buf.append(line)
        if not in_code and not line.strip():
            blocks.append("\n".join(buf).strip())
            buf = []
    if buf:
        blocks.append("\n".join(buf).strip())
    chunks, cur = [], ""
    for b in filter(None, blocks):
        if cur and len(cur) + len(b) + 2 > max_chars:
            chunks.append(cur)
            cur = ""
        limit = MAX_CODE_BLOCK_CHARS if b.startswith("```") else max_chars * 2
        while len(b) > limit:                  # a single huge paragraph
            chunks.append(b[:max_chars])
            b = b[max_chars:]
        cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        chunks.append(cur)
    return chunks


def _upsert_question_chunks(version: str) -> dict:
    cfg = section("index")
    answer_chars = int(cfg.get("question_answer_chars", 1500))
    c = db.conn()
    rows = c.execute(
        "SELECT q.*, ch.id AS chunk_id, ch.content_hash AS chunk_hash FROM questions q "
        "LEFT JOIN chunks ch ON ch.kind = 'question' AND ch.question_id = q.id AND ch.index_version = ? "
        "WHERE q.tag_status IS NOT NULL", (version,)).fetchall()
    added = changed = 0
    with db.tx() as t:
        for q in rows:
            text = question_embed_text(q, answer_chars)
            h = sha1(text)
            if q["chunk_id"] and q["chunk_hash"] == h:
                continue
            tags = " ".join(question_tags(q["id"]))
            body = f"{q['text']}\n{q['section_path']}\n{(q['answer'] or '')[:4000]}"
            fts_tags = f"{q['category'] or ''} {tags}"
            if q["chunk_id"]:
                t.execute("UPDATE chunks SET embed_text = ?, content_hash = ?, category = ?, tags = ?, embedded = 0 "
                          "WHERE id = ?", (text, h, q["category"], tags, q["chunk_id"]))
                t.execute("DELETE FROM embeddings WHERE chunk_id = ?", (q["chunk_id"],))
                t.execute("DELETE FROM chunk_fts WHERE rowid = ?", (q["chunk_id"],))
                chunk_id = q["chunk_id"]
                changed += 1
            else:
                chunk_id = t.execute(
                    "INSERT INTO chunks (index_version, kind, question_id, embed_text, content_hash, category, tags,"
                    " created_at) VALUES (?, 'question', ?, ?, ?, ?, ?, ?)",
                    (version, q["id"], text, h, q["category"], tags, db.now_iso())).lastrowid
                added += 1
            t.execute("INSERT INTO chunk_fts (rowid, body, tags) VALUES (?, ?, ?)", (chunk_id, body, fts_tags))
    return {"question_chunks_added": added, "question_chunks_changed": changed}


def _index_documents(version: str, limit: int, heartbeat=lambda: None) -> dict:
    stats = {"docs_indexed": 0, "doc_chunks": 0, "doc_chunks_kept": 0, "failed": 0}
    for doc in due_documents("parsed", "index", limit):
        try:
            _index_document(doc, version, stats)
            stats["docs_indexed"] += 1
        except Exception as e:  # rolled back; the document is retried later
            mark_failed(doc["id"], "index", f"{type(e).__name__}: {e}")
            stats["failed"] += 1
        heartbeat()
    return stats


def _index_document(doc, version: str, stats: dict) -> None:
    max_chars = int(section("index").get("doc_chunk_chars", 1800))
    tax = taxonomy()
    added = kept = 0
    with db.tx() as t:
        # Chunks whose text is unchanged keep their row and embedding; only new text is embedded.
        old: dict[str, list[int]] = {}
        for r in t.execute("SELECT id, content_hash FROM chunks WHERE kind = 'doc' AND doc_id = ? "
                           "AND index_version = ? ORDER BY id", (doc["id"], version)).fetchall():
            old.setdefault(r["content_hash"], []).append(r["id"])
        # Out of the way of the UNIQUE ordinal while chunks are renumbered.
        t.execute("UPDATE chunks SET ordinal = -id WHERE kind = 'doc' AND doc_id = ? AND index_version = ?",
                  (doc["id"], version))
        stale = [cid for cid, in t.execute("SELECT id FROM chunks WHERE kind = 'doc' AND doc_id = ? "
                                           "AND index_version != ?", (doc["id"], version)).fetchall()]
        if doc["kind"] == "experience" and doc["clean_text"]:
            for n, part in enumerate(split_paragraphs(doc["clean_text"], max_chars)):
                tags = rule_tags(f"{doc['title']}\n{part}", limit=6)
                category = tax.category_of(tags)
                text = f"{doc['title']}\n{part}" if n else part
                reuse = old.get(sha1(text))
                if reuse:
                    chunk_id = reuse.pop(0)
                    t.execute("UPDATE chunks SET ordinal = ?, category = ?, tags = ? WHERE id = ?",
                              (n, category, " ".join(tags), chunk_id))
                    t.execute("UPDATE chunk_fts SET tags = ? WHERE rowid = ?",
                              (f"{category or ''} {' '.join(tags)}", chunk_id))
                    kept += 1
                    continue
                chunk_id = t.execute(
                    "INSERT INTO chunks (index_version, kind, doc_id, ordinal, embed_text, content_hash, category,"
                    " tags, tenant_id, visibility, acl, created_at) VALUES (?, 'doc', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (version, doc["id"], n, text, sha1(text), category, " ".join(tags), doc["tenant_id"],
                     doc["visibility"], doc["acl"], db.now_iso())).lastrowid
                t.execute("INSERT INTO chunk_fts (rowid, body, tags) VALUES (?, ?, ?)",
                          (chunk_id, f"{doc['title']}\n{part}", f"{category or ''} {' '.join(tags)}"))
                added += 1
        for chunk_id in stale + [cid for ids in old.values() for cid in ids]:
            delete_chunk(t, chunk_id)
        t.execute("UPDATE documents SET status = 'indexed', stage = NULL, updated_at = ? WHERE id = ?",
                  (db.now_iso(), doc["id"]))
    stats["doc_chunks"] += added
    stats["doc_chunks_kept"] += kept


def embed_pending(version: str, heartbeat=lambda: None) -> dict:
    emb = get_embedder()
    every = int(section("index").get("embed_commit_every", 64))
    total = 0
    while True:
        rows = db.conn().execute("SELECT id, embed_text FROM chunks WHERE index_version = ? AND embedded = 0 "
                                 "ORDER BY id LIMIT ?", (version, every)).fetchall()
        if not rows:
            break
        # Similar lengths per model batch: every text is padded to the longest one in its batch.
        rows = sorted(rows, key=lambda r: len(r["embed_text"]))
        vecs = emb.embed_passages([r["embed_text"] for r in rows]).astype(np.float32)
        with db.tx() as t:
            for r, v in zip(rows, vecs):
                t.execute("INSERT INTO embeddings (chunk_id, model, dim, vector) VALUES (?, ?, ?, ?) "
                          "ON CONFLICT(chunk_id) DO UPDATE SET model = excluded.model, dim = excluded.dim, "
                          "vector = excluded.vector", (r["id"], emb.name, len(v), v.tobytes()))
                t.execute("UPDATE chunks SET embedded = 1 WHERE id = ?", (r["id"],))
            t.execute("UPDATE index_versions SET dim = ? WHERE version = ? AND dim IS NULL", (len(vecs[0]), version))
        total += len(rows)
        heartbeat()
        log.info("embedded %d chunks so far", total)
    return {"embedded": total}


def index_stage(doc_limit: int, heartbeat=lambda: None) -> dict:
    version = check_model()
    stats = {**_upsert_question_chunks(version), **_index_documents(version, doc_limit, heartbeat)}
    stats.update(embed_pending(version, heartbeat))
    ids, _, how = load_vectors(version, write=True)
    stats.update(vectors=len(ids), snapshot=how)
    db.set_meta("index_epoch", db.now_iso() + f"#{len(ids)}")
    db.set_meta("index_dirty", "0")
    return stats
