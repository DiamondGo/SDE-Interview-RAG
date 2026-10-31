"""Exact vector search over the active index version.

The database holds the authoritative vectors (embeddings table). A snapshot in data/index/<version>/
(vectors.npy, chunk_ids.npy, manifest.json) makes startup fast: it is memory-mapped when it matches the
database, extended with only the new rows when the database grew, and rebuilt from the database when it
does not match (never re-embedded).
"""

import json
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sde_rag import db
from sde_rag.auth import ChunkFilter
from sde_rag.settings import data_dir

log = logging.getLogger("sde_rag.vectors")


def snapshot_dir(version: str) -> Path:
    return data_dir() / "index" / version


def _db_summary(version: str) -> dict:
    r = db.conn().execute(
        "SELECT COUNT(*) AS n, COALESCE(MAX(e.chunk_id), 0) AS max_id, COALESCE(SUM(e.chunk_id), 0) AS sum_ids, "
        "MAX(e.dim) AS dim FROM embeddings e JOIN chunks c ON c.id = e.chunk_id WHERE c.index_version = ?",
        (version,)).fetchone()
    return {"count": r["n"], "max_id": r["max_id"], "sum_ids": r["sum_ids"], "dim": r["dim"]}


def _load_rows(version: str, after_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rows = db.conn().execute(
        "SELECT e.chunk_id, e.dim, e.vector FROM embeddings e JOIN chunks c ON c.id = e.chunk_id "
        "WHERE c.index_version = ? AND e.chunk_id > ? ORDER BY e.chunk_id", (version, after_id)).fetchall()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    mat = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    return ids, mat


def _write_snapshot(version: str, ids: np.ndarray, mat: np.ndarray, summary: dict) -> None:
    d = snapshot_dir(version)
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "vectors.npy", np.ascontiguousarray(mat, dtype=np.float32))
    np.save(tmp / "chunk_ids.npy", ids)
    (tmp / "manifest.json").write_text(json.dumps({**summary, "written_at": db.now_iso()}))
    old = d.with_name(d.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if d.exists():
        d.rename(old)
    tmp.rename(d)
    shutil.rmtree(old, ignore_errors=True)


def load_vectors(version: str, *, write: bool = True) -> tuple[np.ndarray, np.ndarray, str]:
    """(chunk_ids, matrix, how) where how is one of snapshot | extended | rebuilt | empty."""
    summary = _db_summary(version)
    if summary["count"] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32), "empty"
    d = snapshot_dir(version)
    try:
        manifest = json.loads((d / "manifest.json").read_text())
        ids = np.load(d / "chunk_ids.npy")
        mat = np.load(d / "vectors.npy", mmap_mode="r")
        consistent = (len(ids) == manifest["count"] == mat.shape[0] and int(ids.sum()) == manifest["sum_ids"]
                      and (len(ids) == 0 or int(ids.max()) == manifest["max_id"]))
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        consistent, manifest = False, None
    if consistent and manifest == {**summary, "written_at": manifest.get("written_at")}:
        return ids, mat, "snapshot"
    if consistent:
        # Did the database only grow past the snapshot? Then load just the new rows.
        r = db.conn().execute(
            "SELECT COUNT(*), COALESCE(SUM(e.chunk_id), 0) FROM embeddings e JOIN chunks c ON c.id = e.chunk_id "
            "WHERE c.index_version = ? AND e.chunk_id <= ?", (version, manifest["max_id"])).fetchone()
        if r[0] == manifest["count"] and r[1] == manifest["sum_ids"] and summary["dim"] == mat.shape[1]:
            new_ids, new_mat = _load_rows(version, manifest["max_id"])
            ids = np.concatenate([ids, new_ids])
            mat = np.concatenate([np.asarray(mat), new_mat]) if len(new_ids) else np.asarray(mat)
            if write:
                _write_snapshot(version, ids, mat, summary)
            return ids, mat, "extended"
    ids, mat = _load_rows(version)
    if write:
        _write_snapshot(version, ids, mat, summary)
    log.info("vector snapshot for %s rebuilt from the database (%d vectors)", version, len(ids))
    return ids, mat, "rebuilt"


@dataclass
class _Meta:
    kind: np.ndarray
    category: np.ndarray
    tenant: np.ndarray
    visibility: np.ndarray
    acl: list
    tags: list


class VectorIndex:
    def __init__(self, version: str):
        self.version = version
        self.ids = np.zeros(0, dtype=np.int64)
        self.mat = np.zeros((0, 0), dtype=np.float32)
        self.pos: dict[int, int] = {}
        self.meta: _Meta | None = None
        self.loaded_from = ""
        self.epoch = None
        self._lock = threading.Lock()

    def load(self, *, write: bool = False) -> "VectorIndex":
        """Readers (query processes) do not write snapshots; the ingest run that changed the index does."""
        with self._lock:
            self.epoch = db.get_meta("index_epoch")
            self.ids, self.mat, self.loaded_from = load_vectors(self.version, write=write)
            self.pos = {int(i): n for n, i in enumerate(self.ids)}
            rows = {r["id"]: r for r in db.conn().execute(
                "SELECT id, kind, category, tags, tenant_id, visibility, acl FROM chunks WHERE index_version = ?",
                (self.version,))}
            ordered = [rows.get(int(i)) for i in self.ids]
            self.meta = _Meta(
                kind=np.array([r["kind"] if r else "" for r in ordered], dtype=object),
                category=np.array([(r["category"] or "") if r else "" for r in ordered], dtype=object),
                tenant=np.array([r["tenant_id"] if r else "" for r in ordered], dtype=object),
                visibility=np.array([r["visibility"] if r else "" for r in ordered], dtype=object),
                acl=[r["acl"] if r else None for r in ordered],
                tags=[set((r["tags"] or "").split()) if r else set() for r in ordered])
        return self

    def stale(self) -> bool:
        return db.get_meta("index_epoch") != self.epoch

    def mask(self, *, kinds: list[str] | None = None, category: str | None = None, tags: list[str] | None = None,
             chunk_filter: ChunkFilter | None = None) -> np.ndarray | None:
        if self.meta is None or len(self.ids) == 0:
            return None
        m = np.ones(len(self.ids), dtype=bool)
        if kinds:
            m &= np.isin(self.meta.kind, kinds)
        if category:
            m &= self.meta.category == category
        if tags:
            want = set(tags)
            m &= np.fromiter((bool(want & t) for t in self.meta.tags), dtype=bool, count=len(self.ids))
        if chunk_filter is not None:
            m &= np.fromiter((chunk_filter.allows(t, v, a) for t, v, a in
                              zip(self.meta.tenant, self.meta.visibility, self.meta.acl)), dtype=bool, count=len(self.ids))
        return m

    def search(self, qvec: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        if len(self.ids) == 0:
            return []
        if mask is None:
            idx = np.arange(len(self.ids))
        else:
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                return []
        scores = np.asarray(self.mat[idx]) @ np.asarray(qvec, dtype=np.float32)
        top = np.argpartition(-scores, min(k, len(idx)) - 1)[:k] if len(idx) > k else np.arange(len(idx))
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[idx[i]]), float(scores[i])) for i in top]


_indexes: dict[str, VectorIndex] = {}
_lock = threading.Lock()


def get_index(version: str) -> VectorIndex:
    """Loaded once per process; reloaded (incrementally) when an ingest run bumped the index epoch."""
    with _lock:
        ix = _indexes.get(version)
        if ix is None or ix.stale():
            ix = _indexes[version] = VectorIndex(version).load()
        return ix


def reset_indexes() -> None:
    with _lock:
        _indexes.clear()
