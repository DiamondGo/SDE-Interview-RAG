"""Incremental fetching for one source: list, skip what was already seen, download only new or changed content."""

import gzip
import json
import logging
from datetime import datetime, timedelta, timezone

from sde_rag import db, notify
from sde_rag.http import NotModified, Validators, normalize_url
from sde_rag.settings import data_dir, section
from sde_rag.sources.base import Connector, DocRef, RawDoc
from sde_rag.text import sha1

log = logging.getLogger("sde_rag.fetch")
FAILURES_BEFORE_ALERT = 3


def source_due(name: str, interval_hours: float, *, slack_minutes: float = 30) -> bool:
    """Due one interval after the last success. A failing source waits longer after each failure in a row
    (one interval, two, four, ...) up to `source_backoff_max_hours`, so a broken site is not hammered."""
    row = db.conn().execute("SELECT last_success_at, last_run_at, consecutive_failures FROM sources WHERE name = ?",
                            (name,)).fetchone()
    if not row or not row["last_run_at"]:
        return True
    now, slack = datetime.now(timezone.utc), timedelta(minutes=slack_minutes)
    failures = row["consecutive_failures"] or 0
    if failures:
        cap = float(section("ingest").get("source_backoff_max_hours", 48))
        wait = timedelta(hours=min(cap, interval_hours * 2 ** (failures - 1)))
        return now - datetime.fromisoformat(row["last_run_at"]) >= wait - slack
    if not row["last_success_at"]:
        return True
    return now - datetime.fromisoformat(row["last_success_at"]) >= timedelta(hours=interval_hours) - slack


def _load_state(name: str) -> dict:
    row = db.conn().execute("SELECT state FROM sources WHERE name = ?", (name,)).fetchone()
    return json.loads(row["state"]) if row else {}


def _seen(source: str, external_id: str):
    return db.conn().execute("SELECT * FROM seen_items WHERE source = ? AND external_id = ?",
                             (source, external_id)).fetchone()


def _doc_by_url(url_norm: str):
    return db.conn().execute(
        "SELECT id, source FROM documents WHERE url_norm = ? OR alt_url_norm = ? ORDER BY id LIMIT 1",
        (url_norm, url_norm)).fetchone()


def _upsert_seen(t, ref: DocRef, *, url_norm: str, doc_id: int | None, content_hash: str | None = None,
                 etag: str | None = None, last_modified: str | None = None) -> None:
    now = db.now_iso()
    t.execute(
        "INSERT INTO seen_items (source, external_id, url_norm, title, version, etag, last_modified, content_hash,"
        " doc_id, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(source, external_id) DO UPDATE SET url_norm = excluded.url_norm, title = excluded.title, "
        "version = excluded.version, etag = COALESCE(excluded.etag, etag), "
        "last_modified = COALESCE(excluded.last_modified, last_modified), "
        "content_hash = COALESCE(excluded.content_hash, content_hash), doc_id = COALESCE(excluded.doc_id, doc_id), "
        "last_seen_at = excluded.last_seen_at",
        (ref.source, ref.external_id, url_norm, ref.title, ref.version, etag, last_modified, content_hash, doc_id,
         now, now))


def _save_raw(source: str, content_hash: str, content: str) -> str:
    d = data_dir() / "raw" / source
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{content_hash}.gz"
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(content)
        tmp.replace(p)
    return str(p.relative_to(data_dir()))


def read_raw(raw_path: str) -> str:
    with gzip.open(data_dir() / raw_path, "rt", encoding="utf-8") as f:
        return f.read()


def _store_document(raw: RawDoc, content_hash: str, raw_path: str) -> int:
    ref = raw.ref
    now = db.now_iso()
    url_norm = normalize_url(ref.url)
    alt = normalize_url(raw.canonical_url) if raw.canonical_url else None
    alt = alt if alt and alt != url_norm else None
    with db.tx() as t:
        t.execute(
            "INSERT INTO documents (source, origin, external_id, url, url_norm, alt_url_norm, title, author,"
            " published_at, fetched_at, kind, content_type, raw_path, raw_hash, hints, status, attempts, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'fetched', 0, ?) "
            "ON CONFLICT(source, external_id) DO UPDATE SET origin = excluded.origin, url = excluded.url, "
            "url_norm = excluded.url_norm, alt_url_norm = excluded.alt_url_norm, title = excluded.title, "
            "author = excluded.author, published_at = excluded.published_at, fetched_at = excluded.fetched_at, "
            "kind = excluded.kind, content_type = excluded.content_type, raw_path = excluded.raw_path, "
            "raw_hash = excluded.raw_hash, hints = excluded.hints, status = 'fetched', stage = NULL, attempts = 0, "
            "next_retry_at = NULL, last_error = NULL, skip_reason = NULL, extract_status = NULL, "
            "extract_attempts = 0, simhash = NULL, dup_of = NULL, updated_at = excluded.updated_at",
            (ref.source, raw.origin, ref.external_id, ref.url, url_norm, alt, raw.title or ref.title, raw.author,
             raw.published_at or ref.published_at, now, raw.kind, raw.content_type, raw_path, content_hash,
             json.dumps(raw.hints), now))
        doc_id = t.execute("SELECT id FROM documents WHERE source = ? AND external_id = ?",
                           (ref.source, ref.external_id)).fetchone()[0]
        _upsert_seen(t, ref, url_norm=url_norm, doc_id=doc_id, content_hash=content_hash, etag=raw.etag,
                     last_modified=raw.last_modified)
    return doc_id


def sync_source(name: str, connector: Connector, heartbeat=lambda: None) -> dict:
    """Returns counters for the run report."""
    stats = {"listed": 0, "seen": 0, "cross_source": 0, "not_modified": 0, "downloaded": 0, "new": 0,
             "changed": 0, "duplicate_downloads": 0, "failed": 0, "errors": []}
    c = db.conn()
    c.execute("INSERT INTO sources (name) VALUES (?) ON CONFLICT(name) DO NOTHING", (name,))
    c.execute("UPDATE sources SET last_run_at = ? WHERE name = ?", (db.now_iso(), name))
    state = _load_state(name)
    try:
        listing = connector.list_updates(state)
    except Exception as e:
        _record_failure(name, f"listing failed: {type(e).__name__}: {e}")
        stats["errors"].append(f"{name}: listing failed: {type(e).__name__}: {str(e)[:300]}")
        return stats
    if listing.skipped:
        stats["skipped"] = listing.skipped
        log.info("source %s skipped this run: %s", name, listing.skipped)
        return stats
    stats["listed"] = len(listing.refs)
    stats["errors"] += [f"{name}: {e}" for e in listing.errors]
    failed: list[DocRef] = []
    cap = (getattr(connector, "cfg", None) or {}).get("max_downloads_per_run")
    heartbeat()
    for ref in listing.refs:
        heartbeat()
        seen = _seen(ref.source, ref.external_id)
        if seen and seen["doc_id"] and ref.version is not None and seen["version"] == ref.version:
            stats["seen"] += 1
            c.execute("UPDATE seen_items SET last_seen_at = ? WHERE id = ?", (db.now_iso(), seen["id"]))
            continue
        if cap is not None and stats["downloaded"] + stats["failed"] >= int(cap):
            failed.append(ref)                   # for the next run: not a failure, but listed again
            stats["deferred"] = stats.get("deferred", 0) + 1
            continue
        if ref.outbound_url:
            known = _doc_by_url(normalize_url(ref.outbound_url))
            if known and known["source"] != ref.source:
                ref.meta["outbound_known"] = known["id"]
                if not (seen and seen["doc_id"]):
                    c.execute("UPDATE documents SET citations = citations + 1 WHERE id = ?", (known["id"],))
                    stats["cross_source"] += 1
        validators = Validators(seen["etag"], seen["last_modified"]) if seen else None
        try:
            raw = connector.fetch_doc(ref, validators)
        except NotModified:
            stats["not_modified"] += 1
            with db.tx() as t:
                _upsert_seen(t, ref, url_norm=normalize_url(ref.url), doc_id=seen["doc_id"] if seen else None)
            continue
        except Exception as e:  # one bad document must not stop the source
            failed.append(ref)
            stats["failed"] += 1
            stats["errors"].append(f"{name}: {ref.external_id}: {type(e).__name__}: {str(e)[:200]}")
            continue
        stats["downloaded"] += 1
        content_hash = sha1(raw.content)
        existing = c.execute("SELECT id, raw_hash FROM documents WHERE source = ? AND external_id = ?",
                             (ref.source, ref.external_id)).fetchone()
        same_elsewhere = c.execute("SELECT id FROM documents WHERE raw_hash = ? AND NOT (source = ? AND external_id = ?)"
                                   " LIMIT 1", (content_hash, ref.source, ref.external_id)).fetchone()
        if (existing and existing["raw_hash"] == content_hash) or (same_elsewhere and not existing):
            stats["duplicate_downloads"] += 1
            doc_id = existing["id"] if existing else same_elsewhere["id"]
            with db.tx() as t:
                _upsert_seen(t, ref, url_norm=normalize_url(ref.url), doc_id=doc_id, content_hash=content_hash,
                             etag=raw.etag, last_modified=raw.last_modified)
            continue
        _store_document(raw, content_hash, _save_raw(ref.source, content_hash, raw.content))
        stats["changed" if existing else "new"] += 1
    new_state = connector.finalize(listing.state, failed)
    ok = not listing.errors or bool(listing.refs)
    c.execute("UPDATE sources SET state = ? WHERE name = ?", (json.dumps(new_state), name))
    if ok:
        c.execute("UPDATE sources SET last_success_at = ?, consecutive_failures = 0, last_error = NULL WHERE name = ?",
                  (db.now_iso(), name))
        notify.resolve(f"source:{name}", f"Source {name} recovered")
    else:
        _record_failure(name, "; ".join(listing.errors)[:1000])
    return stats


def _record_failure(name: str, error: str) -> None:
    c = db.conn()
    c.execute("UPDATE sources SET consecutive_failures = consecutive_failures + 1, last_error = ? WHERE name = ?",
              (error[:2000], name))
    n = c.execute("SELECT consecutive_failures FROM sources WHERE name = ?", (name,)).fetchone()[0]
    log.warning("source %s failed (%d in a row): %s", name, n, error[:300])
    if n >= FAILURES_BEFORE_ALERT:
        notify.alert(f"source:{name}", f"Source {name} failing", f"{n} runs in a row: {error[:300]}")
