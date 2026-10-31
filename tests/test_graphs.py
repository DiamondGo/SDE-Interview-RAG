import importlib.util
import json
from pathlib import Path

import pytest

from sde_rag import db
from sde_rag.graphs import ingest as ingest_graph
from sde_rag.graphs import query as query_graph
from sde_rag.pipeline import stages, tagging
from sde_rag.pipeline.fetch import sync_source
from sde_rag.pipeline.indexing import index_stage
from tests.conftest import FakeBackend, make_registry
from tests.fakes import EXPERIENCE_MD, GO_MD, PREP_MD, FakeConnector

ROOT = Path(__file__).resolve().parents[1]


def _load(spec: str):
    path, attr = spec.split(":")
    s = importlib.util.spec_from_file_location(f"graph_{Path(path).stem}", ROOT / path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return getattr(mod, attr)


def test_langgraph_json_graphs_load_with_declared_io():
    cfg = json.loads((ROOT / "langgraph.json").read_text())
    expected = {"ingest": ({"run_key", "sources", "force"}, {"summary"}),
                "query": ({"question", "filters"}, {"answer", "citations", "intent", "degraded", "review_job_id"}),
                "evaluate": ({"suite", "compare_to"}, {"metrics", "baseline", "deltas", "passed", "run_id"})}
    assert set(cfg["graphs"]) == set(expected)
    for name, spec in cfg["graphs"].items():
        g = _load(spec)
        ins, outs = expected[name]
        assert set(g.get_input_jsonschema()["properties"]) == ins, name
        assert set(g.get_output_jsonschema()["properties"]) == outs, name
        assert g.checkpointer is None, "exported graphs must leave persistence to the runtime"


def _index():
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD}}))
    sync_source("b", FakeConnector("b", {"go": {"version": "1", "content": GO_MD},
                                         "exp": {"version": "1", "content": EXPERIENCE_MD, "kind": "experience"}}))
    stages.clean_stage(10)
    stages.parse_stage(10)
    tagging.rule_tag_stage()
    stages.merge_stage()
    index_stage(10)


def _router(intent, queries, tags=()):
    return lambda schema, messages: {"intent": intent, "language": "en", "tags": list(tags),
                                     "standalone_question": queries[0], "search_queries": queries}


def _gil_label():
    qid = db.conn().execute("SELECT id FROM questions WHERE text LIKE '%GIL%'").fetchone()[0]
    return f"Q{qid}"


def test_solve_flow_streams_and_keeps_only_valid_citations():
    _index()
    label = _gil_label()
    gen = FakeBackend("gen", stream_script=[f"The GIL serialises bytecode [{label}]. ", "Also see [Q99999]."])
    make_registry({"router": {"backend": "r"}, "generator": {"backend": "gen", "stream": True}},
                  {"r": FakeBackend("r", structured=_router("solve", ["python GIL threads"])), "gen": gen})
    events = []
    out = None
    for mode, chunk in query_graph.graph.stream({"question": "Explain the GIL"}, stream_mode=["custom", "values"]):
        if mode == "custom":
            events.append(chunk)
        else:
            out = chunk
    assert "".join(e["text"] for e in events if e["type"] == "token").startswith("The GIL")
    assert "[Q99999]" not in out["answer"] and f"[{label}]" in out["answer"]
    assert [c["label"] for c in out["citations"]] == [label]
    log = db.conn().execute("SELECT intent, degraded, trace FROM query_logs").fetchone()
    assert log["intent"] == "solve" and json.loads(log["trace"])["invalid_citations"] == ["Q99999"]


def test_topic_flow_lists_questions_and_falls_back_to_a_table():
    _index()
    make_registry({"router": {"backend": "r"}, "composer": {"backend": "bad", "stream": True}},
                  {"r": FakeBackend("r", structured=_router("topic", ["python interview questions"], ["python"])),
                   "bad": FakeBackend("bad", error=RuntimeError("down"))})
    out = query_graph.graph.invoke({"question": "Common Python questions"})
    assert out["intent"] == "topic" and out["degraded"]
    assert "GIL" in out["answer"] and "|" in out["answer"]


def test_router_failure_uses_rules_and_generator_failure_degrades():
    _index()
    make_registry({"router": {"backend": "bad"}, "generator": {"backend": "bad", "stream": True}},
                  {"bad": FakeBackend("bad", error=RuntimeError("down"))})
    out = query_graph.graph.invoke({"question": "How do goroutines communicate?"})
    assert out["intent"] == "solve" and out["degraded"]
    assert "goroutine" in out["answer"].lower()


def test_invalid_input_never_reaches_models():
    make_registry({}, {})
    out = query_graph.graph.invoke({"question": "   "})
    assert out["intent"] == "invalid" and out["answer"]


def test_ingest_graph_is_idempotent_per_run_key_and_respects_the_lock(monkeypatch):
    conn = FakeConnector("fake", {"py": {"version": "1", "content": PREP_MD}})
    monkeypatch.setattr(ingest_graph, "source_configs", lambda: {"fake": {"type": "fake", "enabled": True}})
    monkeypatch.setattr(ingest_graph, "build_connector", lambda name, cfg: conn)
    make_registry({}, {})
    out = ingest_graph.graph.invoke({"run_key": "2026-10-31-PM"})
    s = out["summary"]
    assert s["sources"]["fake"]["new"] == 1 and s["questions"] == 3 and s["duplicate_download_rate"] == 0
    assert "tagger" in s["preflight"]                       # no tagger configured: rules only, run still completes
    again = ingest_graph.graph.invoke({"run_key": "2026-10-31-PM"})
    assert again["summary"]["note"] == "already done for this run key"
    assert ingest_graph._acquire_lock("someone-else")
    blocked = ingest_graph.graph.invoke({"run_key": "2026-11-01-AM"})
    assert "lock" in blocked["summary"]["skipped"]


def test_lock_of_a_dead_process_on_this_host_is_taken_over():
    import socket
    dead = f"{socket.gethostname()}:999999999:2026-10-31-PM"     # no such pid
    live_elsewhere = "other-host:1:2026-10-31-PM"
    assert ingest_graph._acquire_lock(dead)
    assert ingest_graph._acquire_lock("me")                       # stale: taken over
    assert ingest_graph._acquire_lock(live_elsewhere) is False    # held by "me"? "me" has no pid: kept
    ingest_graph.release_lock("me")
    assert ingest_graph._acquire_lock(live_elsewhere)
    assert ingest_graph._acquire_lock("me") is False              # another host: cannot tell, wait for expiry
