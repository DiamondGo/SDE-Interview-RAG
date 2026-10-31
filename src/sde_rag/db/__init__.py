"""SQLite access: one connection per thread, WAL mode, versioned migrations applied on first use."""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Iterator

from sde_rag.settings import data_dir

_local = threading.local()
_migrated: set[str] = set()
_migrate_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    return data_dir() / "rag.db"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # autocommit; transactions are explicit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def conn() -> sqlite3.Connection:
    path = db_path()
    key = str(path)
    cache = getattr(_local, "conns", None)
    if cache is None:
        cache = _local.conns = {}
    c = cache.get(key)
    if c is None:
        c = cache[key] = _connect(path)
        if key not in _migrated:
            with _migrate_lock:
                if key not in _migrated:
                    migrate(c)
                    _migrated.add(key)
    return c


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    """A write transaction. BEGIN IMMEDIATE takes the write lock up front, so it never deadlocks on upgrade."""
    c = conn()
    if c.in_transaction:  # nested: join the outer transaction
        yield c
        return
    c.execute("BEGIN IMMEDIATE")
    try:
        yield c
    except BaseException:
        c.execute("ROLLBACK")
        raise
    c.execute("COMMIT")


def migrate(c: sqlite3.Connection) -> list[str]:
    c.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
    done = {r[0] for r in c.execute("SELECT version FROM schema_migrations")}
    applied = []
    files = sorted(p for p in resources.files("sde_rag.db").joinpath("migrations").iterdir()
                   if p.name.endswith(".sql"))
    for f in files:
        version = f.name.split("_", 1)[0]
        if version in done:
            continue
        c.execute("BEGIN IMMEDIATE")
        try:
            # executescript would COMMIT our transaction; run statements one by one instead.
            for stmt in _split_sql(f.read_text()):
                c.execute(stmt)
            c.execute("INSERT INTO schema_migrations VALUES (?, ?)", (version, now_iso()))
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
        applied.append(f.name)
    return applied


def _split_sql(script: str) -> list[str]:
    stmts, buf = [], []
    for line in script.splitlines():
        if line.strip().startswith("--"):
            continue
        buf.append(line)
        if sqlite3.complete_statement("\n".join(buf)):
            stmt = "\n".join(buf).strip()
            if stmt.rstrip(";").strip():
                stmts.append(stmt)
            buf = []
    return stmts


def close_all() -> None:
    """Close this thread's connections and forget migrations (tests switch databases)."""
    for c in getattr(_local, "conns", {}).values():
        c.close()
    _local.conns = {}
    _migrated.clear()


def get_meta(key: str, default: str | None = None) -> str | None:
    row = conn().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(key: str, value: str) -> None:
    conn().execute("INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                   (key, value))
