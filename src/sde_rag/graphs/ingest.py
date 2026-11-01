"""Ingest graph: fetch new interview content, clean, extract questions, tag, merge and index.

start -> preflight -> plan -(Send per due source)-> fetch_source -> clean -> parse -> extract -> tag -> dedup
      -> merge -> index -> report

Idempotent per run_key (default: today's half-day, e.g. 2026-10-31-PM): a finished run_key returns its summary
unless force=True. A database lock keeps two schedulers from running ingest at the same time.
State holds only ids and counters; the documents themselves live in SQLite, so any stage can resume.
"""

import json
import logging
import operator
import os
import socket
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, Send
from typing_extensions import TypedDict

from sde_rag import db, notify, watchdog
from sde_rag.agents import get_registry
from sde_rag.pipeline import dedup, extraction, stages, tagging
from sde_rag.pipeline.fetch import source_due, sync_source
from sde_rag.pipeline.indexing import index_stage
from sde_rag.settings import section
from sde_rag.sources import build_connector, source_configs

log = logging.getLogger("sde_rag.ingest")
LOCK = "ingest"


@dataclass
class IngestContext:
    doc_limit: int | None = None      # documents per stage this run (default from settings)
    skip_llm: bool = False            # rules only: no LLM tagging


class IngestInput(TypedDict, total=False):
    run_key: str
    sources: list[str]                # only these sources (default: every enabled source that is due)
    force: bool


class IngestOutput(TypedDict, total=False):
    summary: dict


def _merge(old: dict | None, new: dict | None) -> dict:
    return {**(old or {}), **(new or {})}


class IngestState(TypedDict, total=False):
    run_key: str
    sources: list[str]
    force: bool
    run_id: int
    owner: str
    done: bool
    llm: dict                         # role -> usable this run (tagger, extractor, dedup_judge)
    stats: Annotated[dict, _merge]
    errors: Annotated[list[str], operator.add]
    summary: dict


def default_run_key(now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    return f"{now:%Y-%m-%d}-{'AM' if now.hour < 12 else 'PM'}"


def _owner_gone(owner: str) -> bool:
    """A lock owned by a process on this host that no longer exists (killed run) is stale."""
    host, _, rest = owner.partition(":")
    pid = rest.partition(":")[0]
    if host != socket.gethostname() or not pid.isdigit():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _acquire_lock(owner: str) -> bool:
    ttl = int(section("ingest").get("lock_ttl_s", 21600))
    now = datetime.now(timezone.utc)
    with db.tx() as t:
        row = t.execute("SELECT owner, expires_at FROM run_locks WHERE name = ?", (LOCK,)).fetchone()
        if (row and row["owner"] != owner and datetime.fromisoformat(row["expires_at"]) > now
                and not _owner_gone(row["owner"])):
            return False
        t.execute("INSERT INTO run_locks (name, owner, expires_at) VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
                  "owner = excluded.owner, expires_at = excluded.expires_at",
                  (LOCK, owner, (now + timedelta(seconds=ttl)).isoformat()))
    return True


def release_lock(owner: str) -> None:
    db.conn().execute("DELETE FROM run_locks WHERE name = ? AND owner = ?", (LOCK, owner))


class Heartbeat:
    """Marks the run as alive (and extends the lock) at most every `every_s` seconds."""

    def __init__(self, run_id: int, owner: str, every_s: float = 30):
        self.run_id, self.owner, self.every_s, self.last = run_id, owner, every_s, 0.0

    def __call__(self) -> None:
        watchdog.progress()
        if time.monotonic() - self.last < self.every_s:
            return
        self.last = time.monotonic()
        now = datetime.now(timezone.utc)
        ttl = int(section("ingest").get("lock_ttl_s", 21600))
        c = db.conn()
        c.execute("UPDATE ingest_runs SET heartbeat_at = ? WHERE id = ?", (now.isoformat(timespec="seconds"), self.run_id))
        c.execute("UPDATE run_locks SET expires_at = ? WHERE name = ? AND owner = ?",
                  ((now + timedelta(seconds=ttl)).isoformat(), LOCK, self.owner))


def _hb(state: IngestState) -> Heartbeat:
    return Heartbeat(state["run_id"], state["owner"])


def _limit(runtime: Runtime[IngestContext] | None) -> int:
    ctx = runtime.context if runtime else None
    return int((ctx.doc_limit if ctx and ctx.doc_limit else None) or section("ingest").get("doc_limit_per_run", 300))


# ---- nodes -------------------------------------------------------------------------------

def start(state: IngestState) -> dict:
    run_key = state.get("run_key") or default_run_key()
    c = db.conn()
    if not state.get("force"):
        done = c.execute("SELECT summary FROM ingest_runs WHERE run_key = ? AND status = 'done' ORDER BY id DESC "
                         "LIMIT 1", (run_key,)).fetchone()
        if done:
            return {"run_key": run_key, "done": True, "summary": {**json.loads(done["summary"] or "{}"),
                                                                   "note": "already done for this run key"}}
    owner = f"{socket.gethostname()}:{os.getpid()}:{run_key}"
    if not _acquire_lock(owner):
        return {"run_key": run_key, "done": True, "summary": {"run_key": run_key, "skipped": "another ingest run holds the lock"}}
    # Holding the lock, any run still marked running belongs to a process that died without saying so.
    c.execute("UPDATE ingest_runs SET status = 'abandoned', finished_at = ? WHERE status = 'running'", (db.now_iso(),))
    run_id = c.execute("INSERT INTO ingest_runs (run_key, started_at, heartbeat_at, status) VALUES (?, ?, ?, 'running')",
                       (run_key, db.now_iso(), db.now_iso())).lastrowid
    return {"run_key": run_key, "run_id": run_id, "owner": owner, "done": False, "stats": {}, "errors": []}


LLM_ROLES = ("tagger", "extractor", "dedup_judge")


def preflight(state: IngestState, runtime: Runtime[IngestContext]) -> dict:
    """Which LLM roles can work right now? A role that cannot leaves its work queued: the run still fetches,
    parses, rule-tags, merges exact and obvious duplicates, and indexes."""
    if runtime.context and runtime.context.skip_llm:
        return {"llm": {}, "stats": {"preflight": "LLM skipped by request"}}
    reg = get_registry()
    llm, notes, probed = {}, {}, {}
    for role in LLM_ROLES:
        if role not in reg.roles:
            notes[role] = "not configured"
            continue
        problems = [m for lvl, m in reg.problems().get(role, []) if lvl == "error"]
        if problems:
            notes[role] = f"unusable: {problems}"
            continue
        backend = reg.backends[reg.role(role).attempts[0].backend]
        if backend.name not in probed:
            probed[backend.name] = backend.probe()
            if probed[backend.name]:
                notify.resolve(f"llm:{backend.name}", f"Model backend {backend.name} responding again")
            else:
                notify.alert(f"llm:{backend.name}", f"Model backend {backend.name} not responding",
                             "this ingest run leaves its LLM work (tagging, extraction, duplicate judging) queued")
        llm[role] = probed[backend.name]
        notes[role] = "ok" if llm[role] else f"{backend.name} probe failed"
    return {"llm": llm, "stats": {"preflight": notes}}


def plan(state: IngestState):
    cfgs = source_configs()
    wanted = state.get("sources")
    hour = datetime.now().astimezone().hour
    due = []
    for name, cfg in cfgs.items():
        if wanted:
            if name in wanted:
                due.append(name)
        elif cfg.get("enabled", True) and (state.get("force") or (
                source_due(name, float(cfg.get("interval_hours", 12)))
                and hour in cfg.get("run_hours", range(24)))):   # e.g. only the evening run
            due.append(name)
    if not due:
        return "clean"
    return [Send("fetch_source", {"source": name, "run_id": state["run_id"], "owner": state["owner"]}) for name in due]


def fetch_source(payload: dict) -> dict:
    name = payload["source"]
    cfg = source_configs()[name]
    connector = build_connector(name, cfg)
    try:
        st = sync_source(name, connector, heartbeat=Heartbeat(payload["run_id"], payload["owner"]))
    finally:
        (getattr(connector, "close", None) or connector.http.close)()
    errors = st.pop("errors")
    log.info("source %s: %s", name, st)
    return {"stats": {f"fetch:{name}": st}, "errors": errors[:50]}


def clean(state: IngestState, runtime: Runtime[IngestContext]) -> dict:
    return {"stats": {"clean": stages.clean_stage(_limit(runtime), _hb(state))}}


def parse(state: IngestState, runtime: Runtime[IngestContext]) -> dict:
    return {"stats": {"parse": stages.parse_stage(_limit(runtime), _hb(state))}}


def extract(state: IngestState) -> dict:
    return {"stats": {"extract": extraction.extract_stage(
        int(section("extract").get("docs_per_run", 40)), enabled=bool((state.get("llm") or {}).get("extractor")),
        heartbeat=_hb(state))}}


def tag(state: IngestState) -> dict:
    rule = tagging.rule_tag_stage()
    llm = tagging.llm_tag_stage(int(section("ingest").get("llm_tag_limit_per_run", 600)),
                                enabled=bool((state.get("llm") or {}).get("tagger")), heartbeat=_hb(state))
    return {"stats": {"tag": {**rule, **llm}}}


def dedup_node(state: IngestState) -> dict:
    return {"stats": {"dedup": dedup.dedup_stage(
        judge_enabled=bool((state.get("llm") or {}).get("dedup_judge")),
        judge_limit=int(section("dedup").get("judge_pairs_per_run", 600)), heartbeat=_hb(state))}}


def merge(state: IngestState) -> dict:
    return {"stats": {"merge": stages.merge_stage()}}


def index(state: IngestState, runtime: Runtime[IngestContext]) -> dict:
    return {"stats": {"index": index_stage(_limit(runtime), _hb(state))}}


FINISHED = ("indexed", "unchanged", "skipped", "dead")


def _previous_summaries(run_id: int, n: int) -> list[dict]:
    rows = db.conn().execute("SELECT summary FROM ingest_runs WHERE status = 'done' AND id < ? AND summary IS NOT NULL "
                             "ORDER BY id DESC LIMIT ?", (run_id, n)).fetchall()
    return [json.loads(r[0]) for r in rows]


def report(state: IngestState) -> dict:
    cfg = section("ingest")
    stats, errors = state.get("stats") or {}, state.get("errors") or []
    fetch = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("fetch:")}
    downloaded = sum(v.get("downloaded", 0) for v in fetch.values())
    dups = sum(v.get("duplicate_downloads", 0) for v in fetch.values())
    counts = {r["status"]: r["n"] for r in db.conn().execute("SELECT status, COUNT(*) AS n FROM documents GROUP BY status")}
    stalls = get_registry().take_stalls()
    summary = {
        "run_key": state["run_key"], "sources": fetch,
        "duplicate_download_rate": round(dups / downloaded, 4) if downloaded else 0.0,
        **{k: v for k, v in stats.items() if not k.startswith("fetch:")},
        "documents_by_status": counts,
        "backlog": sum(n for s, n in counts.items() if s not in FINISHED),
        "stalls": stalls, "stall_count": sum(stalls.values()),
        "questions": db.conn().execute("SELECT COUNT(*) FROM questions").fetchone()[0],
        "errors": errors[:100], "error_count": len(errors)}
    db.conn().execute("UPDATE ingest_runs SET finished_at = ?, status = 'done', summary = ? WHERE id = ?",
                      (db.now_iso(), json.dumps(summary, ensure_ascii=False, default=str), state["run_id"]))
    release_lock(state["owner"])
    _alerts(summary, _previous_summaries(state["run_id"], 3), cfg, downloaded, dups)
    return {"summary": summary}


def _alerts(summary: dict, previous: list[dict], cfg: dict, downloaded: int, dups: int) -> None:
    if summary["duplicate_download_rate"] > 0.05:
        notify.alert("ingest:duplicates", "Duplicate downloads",
                     f"{dups} of {downloaded} downloads were content already stored: an incremental connector is broken")
    dead = summary["documents_by_status"].get("dead", 0)
    dead_before = previous[0].get("documents_by_status", {}).get("dead", 0) if previous else 0
    if dead - dead_before > int(cfg.get("dead_alert_increase", 30)):
        notify.alert("ingest:dead", "Dead letters piling up",
                     f"{dead - dead_before} more documents failed {cfg.get('max_attempts', 5)} times "
                     f"({dead} in total); see `sde-rag replay --list`")
    backlogs = [summary["backlog"]] + [p["backlog"] for p in previous if "backlog" in p]
    if len(backlogs) >= 4 and all(a > b for a, b in zip(backlogs, backlogs[1:])):
        notify.alert("ingest:backlog", "Backlog keeps growing",
                     f"documents waiting for a stage, newest run first: {backlogs}")
    elif len(backlogs) >= 2 and backlogs[0] <= backlogs[1]:
        notify.resolve("ingest:backlog", "Backlog no longer growing", f"{backlogs[0]} documents waiting")
    ext = summary.get("extract") or {}
    if ext.get("proposed", 0) >= 20:
        limit = float(section("extract").get("fabrication_alert", 0.15))
        if ext.get("fabrication_rate", 0) > limit:
            notify.alert("extract:fabrication", "Extractor making up questions",
                         f"{ext['unquoted']} of {ext['proposed']} extracted questions quote nothing in their post "
                         f"(more than {limit:.0%}); they were discarded")
        else:
            notify.resolve("extract:fabrication", "Extractor quoting its posts again")
    if summary["stall_count"] >= int(cfg.get("stall_alert", 3)):
        notify.alert("ingest:stalls", "Model calls stalling",
                     f"{summary['stall_count']} stalled calls this run: {summary['stalls']}")


def build_graph() -> StateGraph:
    b = StateGraph(IngestState, context_schema=IngestContext, input_schema=IngestInput, output_schema=IngestOutput)
    # Stages only take work from SQLite and record it there, so re-running a node after "database is locked"
    # (another process holding the write lock past busy_timeout) is safe.
    busy = RetryPolicy(max_attempts=3, initial_interval=5.0, retry_on=sqlite3.OperationalError)
    for name, fn in (("start", start), ("preflight", preflight), ("fetch_source", fetch_source), ("clean", clean),
                     ("parse", parse), ("extract", extract), ("tag", tag), ("dedup", dedup_node), ("merge", merge),
                     ("index", index), ("report", report)):
        b.add_node(name, fn, retry_policy=None if name == "start" else busy)
    b.add_edge(START, "start")
    b.add_conditional_edges("start", lambda s: END if s.get("done") else "preflight", ["preflight", END])
    b.add_conditional_edges("preflight", plan, ["fetch_source", "clean"])
    b.add_edge("fetch_source", "clean")
    b.add_edge("clean", "parse")
    b.add_edge("parse", "extract")
    b.add_edge("extract", "tag")
    b.add_edge("tag", "dedup")
    b.add_edge("dedup", "merge")
    b.add_edge("merge", "index")
    b.add_edge("index", "report")
    b.add_edge("report", END)
    return b


graph = build_graph().compile()
