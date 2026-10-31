"""Document stages after fetching. Each stage takes the documents waiting for it (or due for a retry),
so an interrupted or failed run is picked up by the next one.

  fetched --clean--> cleaned --parse--> parsed --index--> indexed
  any stage failure -> failed (retried with backoff) -> dead after `max_attempts`
"""

import json
import logging
import math
from datetime import datetime, timedelta, timezone

from sde_rag import db
from sde_rag.parse.clean import CLEANER_VERSION, clean
from sde_rag.parse.markdown_qa import PARSER_VERSION, parse_markdown_qa
from sde_rag.pipeline.fetch import read_raw
from sde_rag.settings import section
from sde_rag.text import detect_lang, question_hash, sha1

log = logging.getLogger("sde_rag.stages")
MIN_CLEAN_CHARS = 200


def due_documents(input_status: str, stage: str, limit: int) -> list:
    return db.conn().execute(
        "SELECT * FROM documents WHERE status = ? OR (status = 'failed' AND stage = ? AND next_retry_at <= ?) "
        "ORDER BY id LIMIT ?", (input_status, stage, db.now_iso(), limit)).fetchall()


def mark_failed(doc_id: int, stage: str, error: str) -> str:
    max_attempts = int(section("ingest").get("max_attempts", 5))
    c = db.conn()
    attempts = c.execute("SELECT attempts FROM documents WHERE id = ?", (doc_id,)).fetchone()[0] + 1
    status = "dead" if attempts >= max_attempts else "failed"
    retry = datetime.now(timezone.utc) + timedelta(hours=min(48, 2 ** (attempts - 1)))
    c.execute("UPDATE documents SET status = ?, stage = ?, attempts = ?, next_retry_at = ?, last_error = ?, "
              "updated_at = ? WHERE id = ?",
              (status, stage, attempts, retry.isoformat(timespec="seconds"), error[:2000], db.now_iso(), doc_id))
    log.warning("document %s failed at %s (attempt %d, now %s): %s", doc_id, stage, attempts, status, error[:200])
    return status


def clean_stage(limit: int, heartbeat=lambda: None) -> dict:
    stats = {"cleaned": 0, "skipped": 0, "failed": 0}
    for doc in due_documents("fetched", "clean", limit):
        try:
            text = clean(read_raw(doc["raw_path"]), doc["content_type"], doc["url"])
            status, reason = ("cleaned", None) if len(text) >= MIN_CLEAN_CHARS else ("skipped", "too little text")
            db.conn().execute(
                "UPDATE documents SET clean_text = ?, clean_hash = ?, cleaner_version = ?, lang = ?, status = ?, "
                "skip_reason = ?, stage = NULL, updated_at = ? WHERE id = ?",
                (text, sha1(text), CLEANER_VERSION, detect_lang(text), status, reason, db.now_iso(), doc["id"]))
            stats["cleaned" if status == "cleaned" else "skipped"] += 1
        except Exception as e:
            mark_failed(doc["id"], "clean", f"{type(e).__name__}: {e}")
            stats["failed"] += 1
        heartbeat()
    return stats


def _drop_mentions(t, doc_id: int) -> None:
    t.execute("UPDATE questions SET stats_dirty = 1 WHERE id IN "
              "(SELECT question_id FROM question_mentions WHERE doc_id = ?)", (doc_id,))
    t.execute("DELETE FROM question_mentions WHERE doc_id = ?", (doc_id,))


def requeue_outdated_parses() -> int:
    """Documents parsed by an older parser go back to `cleaned`; their stored clean text is parsed again."""
    return db.conn().execute(
        "UPDATE documents SET status = 'cleaned', updated_at = ? WHERE status IN ('parsed', 'indexed') "
        "AND COALESCE(parser_version, '') != ?", (db.now_iso(), PARSER_VERSION)).rowcount


def parse_stage(limit: int, heartbeat=lambda: None) -> dict:
    stats = {"parsed": 0, "questions": 0, "new_questions": 0, "no_questions": 0, "failed": 0,
             "reparse_queued": requeue_outdated_parses()}
    for doc in due_documents("cleaned", "parse", limit):
        try:
            with db.tx() as t:
                _drop_mentions(t, doc["id"])
                # Experience posts are also searched for Q&A headings; their bare list items and titles are not
                # questions (discussion threads are full of rhetorical ones). LLM extraction is milestone 2.
                prep = doc["kind"] == "prep"
                qas = parse_markdown_qa(doc["clean_text"] or "", bare_list_questions=prep, title_questions=prep)
                if prep and not qas:
                    stats["no_questions"] += 1
                if qas:
                    now = db.now_iso()
                    for qa in qas:
                        h = question_hash(qa.question)
                        row = t.execute("SELECT id, answer FROM questions WHERE norm_hash = ?", (h,)).fetchone()
                        if row is None:
                            qid = t.execute(
                                "INSERT INTO questions (norm_hash, text, answer, section_path, difficulty, "
                                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (h, qa.question, qa.answer, qa.section_path, qa.difficulty, now, now)).lastrowid
                            stats["new_questions"] += 1
                        else:
                            qid = row["id"]
                            if qa.difficulty:
                                t.execute("UPDATE questions SET difficulty = COALESCE(difficulty, ?) WHERE id = ?",
                                          (qa.difficulty, qid))
                            if qa.answer and len(qa.answer) > len(row["answer"] or ""):
                                t.execute("UPDATE questions SET answer = ?, section_path = ?, updated_at = ? "
                                          "WHERE id = ?", (qa.answer, qa.section_path, now, qid))
                        t.execute(
                            "INSERT INTO question_mentions (question_id, doc_id, mention_hash, text, answer, "
                            "section_path, ordinal, extractor, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                            "ON CONFLICT(mention_hash) DO NOTHING",
                            (qid, doc["id"], sha1(f"{doc['id']}:{h}"), qa.question, qa.answer, qa.section_path,
                             qa.ordinal, PARSER_VERSION, now))
                        t.execute("UPDATE questions SET stats_dirty = 1 WHERE id = ?", (qid,))
                    stats["questions"] += len(qas)
                t.execute("UPDATE documents SET status = 'parsed', stage = NULL, parser_version = ?, updated_at = ? "
                          "WHERE id = ?", (PARSER_VERSION, db.now_iso(), doc["id"]))
            stats["parsed"] += 1
        except Exception as e:
            mark_failed(doc["id"], "parse", f"{type(e).__name__}: {e}")
            stats["failed"] += 1
        heartbeat()
    return stats


def heat(distinct_sources: int, distinct_companies: int, mentions: int, days_since_seen: float) -> float:
    base = math.log1p(distinct_sources) + 0.5 * math.log1p(distinct_companies) + 0.3 * math.log1p(mentions)
    return base * 0.5 ** (max(days_since_seen, 0) / 365)


def merge_stage() -> dict:
    """Refresh counts and heat of questions whose mentions changed; drop questions nobody mentions any more."""
    c = db.conn()
    dirty = [r[0] for r in c.execute("SELECT id FROM questions WHERE stats_dirty = 1")]
    removed = updated = 0
    now = datetime.now(timezone.utc)
    for i in range(0, len(dirty), 500):
        batch = dirty[i:i + 500]
        marks = ",".join("?" * len(batch))
        rows = c.execute(
            f"SELECT m.question_id AS qid, COUNT(*) AS mentions, COUNT(DISTINCT d.origin) AS origins, "
            f"COUNT(DISTINCT m.company) AS companies, json_group_array(DISTINCT m.company) AS company_list, "
            f"MIN(COALESCE(d.published_at, d.fetched_at)) AS first_seen, "
            f"MAX(COALESCE(d.published_at, d.fetched_at)) AS last_seen "
            f"FROM question_mentions m JOIN documents d ON d.id = m.doc_id "
            f"WHERE m.question_id IN ({marks}) GROUP BY m.question_id", batch).fetchall()
        stats = {r["qid"]: r for r in rows}
        with db.tx() as t:
            for qid in batch:
                r = stats.get(qid)
                if r is None:
                    delete_question(t, qid)
                    removed += 1
                    continue
                days = (now - datetime.fromisoformat(r["last_seen"])).total_seconds() / 86400
                companies = [x for x in json.loads(r["company_list"]) if x]
                t.execute("UPDATE questions SET mention_count = ?, distinct_sources = ?, companies = ?, first_seen = ?,"
                          " last_seen = ?, heat = ?, stats_dirty = 0 WHERE id = ?",
                          (r["mentions"], r["origins"], json.dumps(companies), r["first_seen"], r["last_seen"],
                           heat(r["origins"], r["companies"], r["mentions"], days), qid))
                updated += 1
    return {"updated": updated, "removed": removed}


def delete_question(t, qid: int) -> None:
    for (chunk_id,) in t.execute("SELECT id FROM chunks WHERE kind = 'question' AND question_id = ?", (qid,)).fetchall():
        delete_chunk(t, chunk_id)
    t.execute("DELETE FROM question_tags WHERE question_id = ?", (qid,))
    t.execute("DELETE FROM questions WHERE id = ?", (qid,))


def delete_chunk(t, chunk_id: int) -> None:
    t.execute("DELETE FROM chunk_fts WHERE rowid = ?", (chunk_id,))
    t.execute("DELETE FROM embeddings WHERE chunk_id = ?", (chunk_id,))
    t.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    db.set_meta("index_dirty", "1")
