"""Nightly backup and weekly reconciliation (run by the systemd timers, or by hand)."""

import gzip
import json
import logging
import os
import shutil
import socket
import sqlite3
from datetime import datetime, timedelta, timezone

from sde_rag import db, notify
from sde_rag.graphs import ingest
from sde_rag.index.vectors import snapshot_dir
from sde_rag.pipeline import stages
from sde_rag.settings import data_dir, section
from sde_rag.sources import CONNECTOR_TYPES, source_configs

log = logging.getLogger("sde_rag.maintenance")


def backup(keep: int | None = None) -> dict:
    """A consistent copy of rag.db (SQLite online backup: safe while other processes write), gzipped.
    Vector snapshots are not backed up: they are rebuilt from the database."""
    keep = max(1, int(keep or section("maintenance").get("backup_keep", 7)))
    d = data_dir() / "backups"
    d.mkdir(parents=True, exist_ok=True)
    stamp, n = f"{datetime.now():%Y%m%d-%H%M%S}", 0
    name = f"rag-{stamp}.db"
    while (d / f"{name}.gz").exists():
        n += 1
        name = f"rag-{stamp}-{n}.db"
    tmp = d / f"{name}.tmp"
    dst = sqlite3.connect(tmp)
    try:
        db.conn().backup(dst)
    finally:
        dst.close()
    out = d / f"{name}.gz"
    with open(tmp, "rb") as f, gzip.open(out.with_suffix(".gz.tmp"), "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g, 1 << 20)
    out.with_suffix(".gz.tmp").replace(out)
    tmp.unlink()
    backups = sorted(d.glob("rag-*.db.gz"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    removed = [p.name for p in backups[:-keep]]
    for p in backups[:-keep]:
        p.unlink()
    return {"backup": str(out), "bytes": out.stat().st_size, "kept": min(len(backups), keep), "removed": removed}


def _fix_references(t) -> dict:
    return {
        "seen_items_unlinked": t.execute(
            "UPDATE seen_items SET doc_id = NULL WHERE doc_id IS NOT NULL "
            "AND doc_id NOT IN (SELECT id FROM documents)").rowcount,
        "mentions_orphaned": t.execute(
            "DELETE FROM question_mentions WHERE doc_id NOT IN (SELECT id FROM documents)").rowcount,
        "embeddings_orphaned": t.execute(
            "DELETE FROM embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)").rowcount,
        "fts_rows_orphaned": t.execute(
            "DELETE FROM chunk_fts WHERE rowid NOT IN (SELECT id FROM chunks)").rowcount,
    }


def _orphan_chunks(t) -> int:
    ids = [r[0] for r in t.execute(
        "SELECT id FROM chunks WHERE (kind = 'question' AND question_id NOT IN (SELECT id FROM questions)) "
        "OR (kind = 'doc' AND doc_id NOT IN (SELECT id FROM documents))")]
    for cid in ids:
        stages.delete_chunk(t, cid)
    return len(ids)


def _retire_old_versions(t) -> list[str]:
    """Index versions retired more than `index_retire_days` ago lose their chunks and snapshot."""
    days = float(section("maintenance").get("index_retire_days", 7))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    active = db.get_meta("active_index_version")
    gone = []
    for (version,) in t.execute("SELECT version FROM index_versions WHERE status = 'retired' AND created_at < ? "
                                "AND version != ?", (cutoff, active or "")).fetchall():
        for (cid,) in t.execute("SELECT id FROM chunks WHERE index_version = ?", (version,)).fetchall():
            stages.delete_chunk(t, cid)
        t.execute("UPDATE index_versions SET status = 'deleted' WHERE version = ?", (version,))
        shutil.rmtree(snapshot_dir(version), ignore_errors=True)
        gone.append(version)
    return gone


def reconcile() -> dict:
    """Weekly: repair references between tables, list every GitHub repo in full on the next run, recompute
    heat (its time decay moves even when nothing else changes), drop expired index versions, check FTS."""
    owner = f"{socket.gethostname()}:{os.getpid()}:reconcile"
    if not ingest._acquire_lock(owner):
        return {"skipped": "an ingest run holds the lock"}
    try:
        out: dict = {}
        with db.tx() as t:
            out.update(_fix_references(t))
            out["chunks_orphaned"] = _orphan_chunks(t)
            out["index_versions_deleted"] = _retire_old_versions(t)
            out["full_listing_next_run"] = []
            for name, cfg in source_configs().items():
                full = getattr(CONNECTOR_TYPES.get(cfg.get("type")), "full_listing", None)
                row = t.execute("SELECT state FROM sources WHERE name = ?", (name,)).fetchone()
                if full and row:
                    t.execute("UPDATE sources SET state = ? WHERE name = ?",
                              (json.dumps(full(json.loads(row["state"]))), name))
                    out["full_listing_next_run"].append(name)
            out["missing_raw_files"] = sum(
                1 for (p,) in t.execute("SELECT raw_path FROM documents WHERE raw_path IS NOT NULL")
                if not (data_dir() / p).is_file())
            t.execute("UPDATE questions SET stats_dirty = 1")
        out["heat"] = stages.merge_stage()
        try:
            db.conn().execute("INSERT INTO chunk_fts (chunk_fts) VALUES ('integrity-check')")
            out["fts"] = "ok"
            notify.resolve("db:fts", "Full-text index consistent again")
        except sqlite3.DatabaseError as e:
            out["fts"] = f"{type(e).__name__}: {e}"
            notify.alert("db:fts", "Full-text index failed its integrity check", str(e))
        if out["missing_raw_files"]:
            notify.alert("data:raw", "Raw files missing",
                         f"{out['missing_raw_files']} documents cannot be re-cleaned without downloading again")
        return out
    finally:
        ingest.release_lock(owner)
