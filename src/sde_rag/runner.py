"""Running the graphs from this project's own processes (CLI, UI) with a SQLite checkpointer.

Under a LangGraph server the server supplies persistence instead; the graphs themselves do not depend on it.
"""

import os
import socket
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from sde_rag import db
from sde_rag.graphs import evaluate, ingest, query
from sde_rag.settings import data_dir


@contextmanager
def checkpointer() -> Iterator[SqliteSaver]:
    with SqliteSaver.from_conn_string(str(data_dir() / "checkpoints.db")) as cp:
        yield cp


def run_ingest(*, run_key: str | None = None, sources: list[str] | None = None, force: bool = False,
               doc_limit: int | None = None, skip_llm: bool = False, on_update=None) -> dict:
    run_key = run_key or ingest.default_run_key()
    thread = f"ingest:{run_key}" + (f":{int(time.time())}" if force else "")
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 100, "max_concurrency": 4}
    ctx = ingest.IngestContext(doc_limit=doc_limit, skip_llm=skip_llm)
    with checkpointer() as cp:
        app = ingest.build_graph().compile(checkpointer=cp)
        snap = app.get_state(cfg)
        payload = None if snap.next else {"run_key": run_key, "sources": sources or [], "force": force}
        if payload is None and snap.values.get("owner"):
            # Resuming an interrupted run: this process takes over its lock (stale if the old process died).
            owner = f"{socket.gethostname()}:{os.getpid()}:{run_key}"
            if not ingest._acquire_lock(owner):
                return {"run_key": run_key, "skipped": "another ingest run holds the lock"}
            app.update_state(cfg, {"owner": owner})
        try:
            for update in app.stream(payload, cfg, context=ctx, stream_mode="updates"):
                if on_update:
                    on_update(update)
        except BaseException:
            state = app.get_state(cfg).values
            if state.get("run_id"):
                db.conn().execute("UPDATE ingest_runs SET status = 'failed', finished_at = ? WHERE id = ?",
                                  (db.now_iso(), state["run_id"]))
            if state.get("owner"):
                ingest.release_lock(state["owner"])
            raise
        return app.get_state(cfg).values.get("summary") or {}


def run_eval(suite: str = "retrieval_self", *, sample_size: int = 100, compare_to: int | None = None) -> dict:
    return evaluate.graph.invoke({"suite": suite, "compare_to": compare_to},
                                 context=evaluate.EvalContext(sample_size=sample_size))


class QueryService:
    """The query graph with one long-lived checkpointer; each conversation is a thread in it."""

    def __init__(self):
        conn = sqlite3.connect(data_dir() / "checkpoints.db", check_same_thread=False)
        self._app = query.build_graph().compile(checkpointer=SqliteSaver(conn))

    @staticmethod
    def new_thread(user_id: str = "local") -> str:
        return f"{user_id}:{uuid.uuid4().hex[:12]}"

    def stream(self, thread_id: str, question: str, filters: dict | None = None,
               user_id: str = "local") -> Iterator[tuple[str, dict]]:
        """Yields ("event", custom_event) while running and ("result", output) at the end."""
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
        for mode, chunk in self._app.stream({"question": question, "filters": filters or {}}, cfg,
                                            context=query.QueryContext(user_id=user_id),
                                            stream_mode=["custom", "values"]):
            if mode == "custom":
                yield "event", chunk
        values = self._app.get_state(cfg).values
        yield "result", {k: values.get(k) for k in ("answer", "citations", "intent", "degraded", "review_job_id")}


_service: QueryService | None = None
_service_lock = threading.Lock()


def query_service() -> QueryService:
    global _service
    with _service_lock:
        if _service is None:
            _service = QueryService()
        return _service
