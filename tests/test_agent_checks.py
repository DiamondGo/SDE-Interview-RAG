"""The capability-check machinery with fake backends (the checks themselves call real models: -m agent)."""

import json

import pytest

from sde_rag import agent_checks as ac
from sde_rag import db, review, settings
from tests.conftest import FakeBackend, make_registry


def test_every_golden_set_loads_and_has_its_threshold_section():
    for role in ac.ROLES:
        cases = ac.load_cases(role)
        assert cases, role
        assert ac.thresholds(role).get("quick"), role
        assert len(ac.load_cases(role, "quick")) == min(len(cases), ac.thresholds(role)["quick"])


def test_golden_sets_are_consistent_with_the_taxonomy_and_the_planner():
    from sde_rag.graphs import planner
    from sde_rag.taxonomy import taxonomy
    tax = taxonomy()
    for c in ac.load_cases("tagger"):
        assert c["category"] in tax.categories and all(tax.resolve(t) for t in c["tags"]), c
    for c in ac.load_cases("router"):
        assert c["intent"] in ("topic", "solve", "experience", "out_of_scope")
        assert c.get("category") in (None, *tax.categories) and all(tax.resolve(t) for t in c.get("tags", [])), c
    for c in ac.load_cases("planner"):
        if c["kind"] == "replan":
            plan = planner.Plan(**c["plan"])
            assert planner.validate(plan) is None, c["question"]
            assert set(c["unmet"]) <= {s.id for s in plan.subqueries}


def test_thresholds_compare_and_local_relaxation_is_reported(tmp_path, monkeypatch):
    checks = ac.compare("router", {"intent_accuracy": 0.95, "filter_f1": 0.5, "stalls": 0})
    assert {c["metric"]: c["ok"] for c in checks} == {"intent_accuracy": True, "filter_f1": False, "stalls": True}
    local = tmp_path / "relaxed.toml"
    local.write_text("[router]\nfilter_f1 = { min = 0.4 }\n")
    monkeypatch.setenv("SDE_AGENT_THRESHOLDS_CONFIG", str(local))
    checks = {c["metric"]: c for c in ac.compare("router", {"filter_f1": 0.5})}
    assert checks["filter_f1"]["ok"] and checks["filter_f1"]["relaxed"]
    assert ac.relaxed("router") == {"filter_f1": ({"min": 0.8}, {"min": 0.4})}


def _router_answers(cases):
    by_q = {c["question"]: c for c in cases}

    def answer(schema, messages):
        q = messages[1][1].rsplit("Latest message:\n", 1)[1].strip()
        c = by_q[q]
        return {"intent": c["intent"], "language": c["lang"], "category": c.get("category"), "tags": c.get("tags", []),
                "standalone_question": q, "search_queries": [" ".join(k.split("|")[0] for k in c.get("key_terms", [])) or q]}
    return answer


def test_a_router_that_answers_like_the_labels_passes_and_is_stored():
    cases = ac.load_cases("router", "quick")
    make_registry({"router": {"backend": "r"}}, {"r": FakeBackend("r", structured=_router_answers(cases))})
    out = ac.run("router", "quick")
    assert out["passed"], out
    assert out["metrics"]["intent_accuracy"] == 1.0 and out["metrics"]["key_terms"] == 1.0
    rec = ac.latest("router")
    assert rec["passed"] == 1 and rec["level"] == "quick" and json.loads(rec["metrics"])["intent_accuracy"] == 1.0


def test_a_failing_role_alerts_and_its_gate_closes(monkeypatch):
    monkeypatch.setenv("SDE_SETTING__GATES__ENABLED", "true")
    settings.reset_caches()
    make_registry({"router": {"backend": "r"}}, {"r": FakeBackend("r", structured={
        "intent": "out_of_scope", "language": "en", "search_queries": ["x"]})})
    ok, why = ac.gate("router")                                # never checked: runs the quick check now
    assert not ok and "just now" in why
    assert db.conn().execute("SELECT resolved_at FROM alerts WHERE key = 'agent:router'").fetchone()[0] is None
    calls = len(db.conn().execute("SELECT id FROM agent_checks").fetchall())
    assert ac.gate("router") == (False, ac.gate("router")[1]) and "failed at" in ac.gate("router")[1]
    assert len(db.conn().execute("SELECT id FROM agent_checks").fetchall()) == calls     # fresh: not run again
    assert ac.allowed("router") is False


def test_a_new_configuration_has_no_passing_record():
    cases = ac.load_cases("router", "quick")
    make_registry({"router": {"backend": "r"}}, {"r": FakeBackend("r", structured=_router_answers(cases))})
    ac.run("router", "quick")
    make_registry({"router": {"backend": "r", "thinking": True}},
                  {"r": FakeBackend("r", structured=_router_answers(cases))})
    assert ac.latest("router") is None
    assert ac.gate("router", run_if_stale=False) == (False, "never checked")


def test_the_reviewer_pauses_while_it_has_not_passed(monkeypatch):
    make_registry({"reviewer": {"backend": "rev"}}, {"rev": FakeBackend("rev", structured={"verdict": "correct"})})
    assert review.enabled()                                    # gates off (the unit-test default)
    monkeypatch.setenv("SDE_SETTING__GATES__ENABLED", "true")
    settings.reset_caches()
    ac._allowed_cache.clear()
    assert not review.enabled()                                # never checked: paused


def test_a_gated_generator_gives_retrieval_only_answers(monkeypatch):
    from sde_rag.graphs import query as query_graph
    from tests.test_graphs import _index, _router
    _index()
    make_registry({"router": {"backend": "r"}, "generator": {"backend": "gen", "stream": True}},
                  {"r": FakeBackend("r", structured=_router("solve", ["python GIL threads"])),
                   "gen": FakeBackend("gen", stream_script=["A generated answer."])})
    monkeypatch.setenv("SDE_SETTING__GATES__ENABLED", "true")
    settings.reset_caches()
    ac._allowed_cache.clear()
    out = query_graph.graph.invoke({"question": "Explain the GIL"})
    assert out["degraded"] and "A generated answer." not in out["answer"] and "GIL" in out["answer"]


def test_reviewer_scoring_counts_detection_false_positives_and_pinpointing():
    cases = [{"question": "q1", "answer": "a", "has_error": True, "error_keywords": ["log n"]},
             {"question": "q2", "answer": "a", "has_error": True, "error_keywords": ["partition"]},
             {"question": "q3", "answer": "a", "has_error": False, "error_keywords": []}]
    replies = iter([
        {"verdict": "incorrect", "issues": [{"claim": "O(n)", "problem": "it is O(log n)", "correction": ""}]},
        {"verdict": "correct", "issues": []},
        {"verdict": "incorrect", "issues": [{"claim": "x", "problem": "y"}]}])
    make_registry({"reviewer": {"backend": "rev"}}, {"rev": FakeBackend("rev", structured=lambda s, m: next(replies))})
    metrics, fails = ac.check_reviewer(cases, ac.Calls())
    assert metrics == {"valid_output": 1.0, "error_detection": 0.5, "false_positive": 1.0, "pinpoint": 0.5}
    assert len(fails) == 2


@pytest.mark.parametrize("kw, text, ok", [(["lru"], "Implement an LRU cache", True),
                                          (["island|grid"], "number of islands", True),
                                          (["reverse", "linked list"], "Reverse a list", False)])
def test_keyword_matching(kw, text, ok):
    assert ac.kw_match(kw, text) is ok
