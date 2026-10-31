"""The query graph inside a real LangGraph runtime (`langgraph dev`), called through the SDK.

Run with: uv run pytest -m integration tests/test_langgraph_server.py
Models are pointed at an undefined backend, so the run exercises the degraded paths deterministically.
"""

import asyncio
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from sde_rag import db
from sde_rag.pipeline import stages, tagging
from sde_rag.pipeline.fetch import sync_source
from sde_rag.pipeline.indexing import index_stage
from tests.fakes import GO_MD, PREP_MD, FakeConnector

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(isolated):
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD},
                                         "go": {"version": "1", "content": GO_MD}}))
    stages.clean_stage(10)
    stages.parse_stage(10)
    tagging.rule_tag_stage()
    stages.merge_stage()
    index_stage(10)
    db.close_all()
    port = _free_port()
    env = {**os.environ, "SDE_ROLE__ROUTER__BACKEND": "undefined", "SDE_ROLE__GENERATOR__BACKEND": "undefined",
           "SDE_ROLE__COMPOSER__BACKEND": "undefined", "SDE_HOME": str(ROOT)}
    proc = subprocess.Popen(["uv", "run", "langgraph", "dev", "--no-browser", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{url}/ok", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None:
                pytest.fail(proc.stdout.read())
            time.sleep(1)
        else:
            pytest.fail("langgraph dev did not start")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_query_graph_runs_in_langgraph_dev(server):
    from langgraph_sdk import get_client

    async def run():
        client = get_client(url=server)
        assistants = await client.assistants.search()
        assert {a["graph_id"] for a in assistants} >= {"ingest", "query", "evaluate"}
        thread = await client.threads.create()
        events, final = [], None
        async for part in client.runs.stream(thread["thread_id"], "query",
                                             input={"question": "How do goroutines communicate?"},
                                             context={"user_id": "integration"},
                                             stream_mode=["custom", "values"]):
            if part.event == "custom":
                events.append(part.data)
            elif part.event == "values":
                final = part.data
        return events, final

    events, final = asyncio.run(run())
    assert final["intent"] == "solve" and final["degraded"] is True
    assert "goroutine" in final["answer"].lower()
    assert any(e.get("type") == "status" for e in events)
    row = db.conn().execute("SELECT user_id FROM query_logs ORDER BY id DESC LIMIT 1").fetchone()
    assert row["user_id"] == "integration"
