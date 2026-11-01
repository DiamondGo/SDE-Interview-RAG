import json

import pytest

from sde_rag import db
from sde_rag.graphs import planner, query as query_graph
from tests.conftest import FakeBackend, make_registry
from tests.test_graphs import _index


def _router(intent="solve", complexity="complex"):
    return lambda schema, messages: {"intent": intent, "language": "en", "tags": [], "complexity": complexity,
                                     "complexity_reason": "two things", "standalone_question": "",
                                     "search_queries": ["python GIL goroutines"]}


def _sub(sid, goal, queries, criteria, **kw):
    return {"id": sid, "goal": goal, "intent": kw.pop("intent", "solve"), "search_queries": queries,
            "criteria": [{"kind": k, "value": v} for k, v in criteria], **kw}


GIL = _sub("s1", "Python's GIL", ["python GIL"], [("has_kind", "question"), ("mentions", "GIL")])
GO = _sub("s2", "How goroutines communicate", ["goroutines communicate channels"], [("mentions", "channel")])
META = _sub("s2", "Goroutines at Meta", ["goroutines meta"], [("mentions", "Meta")])


def _plans(*plans):
    calls = []

    def planner_fn(schema, messages):
        calls.append(messages[1][1])
        return plans[min(len(calls) - 1, len(plans) - 1)]
    return planner_fn, calls


def _setup(plans, *, grader=None, generator_script=("An answer.",), router=None):
    _index()
    fn, calls = _plans(*plans)
    gen = FakeBackend("gen", stream_script=list(generator_script))
    backends = {"r": FakeBackend("r", structured=router or _router()), "p": FakeBackend("p", structured=fn), "gen": gen}
    roles = {"router": {"backend": "r"}, "planner": {"backend": "p"}, "generator": {"backend": "gen", "stream": True}}
    if grader is not None:
        backends["g"] = grader
        roles["grader"] = {"backend": "g"}
    make_registry(roles, backends)
    return calls, gen


def _trace():
    return json.loads(db.conn().execute("SELECT trace FROM query_logs ORDER BY id DESC").fetchone()[0])


def test_a_complex_question_is_split_checked_and_answered_by_the_outline():
    calls, gen = _setup([{"subqueries": [GIL, GO], "outline": ["The GIL (s1)", "Goroutines (s2)"]}])
    out = query_graph.graph.invoke({"question": "What is the GIL, and how do goroutines communicate?"})
    t = _trace()
    assert len(calls) == 1 and out["answer"] == "An answer." and out["intent"] == "solve"
    assert {k: v["status"] for k, v in t["plan_result"].items()} == {"s1": "met", "s2": "met"}
    assert t["plan_done"] == "all met" and t["replans"] == 0 and "retrieval_0" not in t
    ctx = t["context"]
    assert any(l in ctx for l in t["plan_result"]["s1"]["items"]) and any(l in ctx for l in t["plan_result"]["s2"]["items"])
    assert out["review_job_id"] is None or out["review_job_id"]          # queued only when a reviewer is configured


def test_an_unmet_part_is_retried_then_replanned_and_only_the_changed_part_runs_again(monkeypatch):
    seen = []
    real = planner._retrieve
    monkeypatch.setattr(planner, "_retrieve", lambda sub, q, p: seen.append((sub.id, list(q))) or real(sub, q, p))
    grader = FakeBackend("g", structured={"sufficient": False, "queries": ["goroutine channels at meta"]})
    calls, _ = _setup([{"subqueries": [GIL, META], "outline": []},
                       {"subqueries": [GIL, {**META, "goal": "How goroutines communicate", "search_queries":
                                             ["goroutines communicate"], "criteria": [{"kind": "mentions",
                                                                                       "value": "goroutine"}]}]}],
                      grader=grader)
    query_graph.graph.invoke({"question": "How do goroutines communicate at Meta, and what is the GIL?"})
    t = _trace()
    assert len(calls) == 2 and "s2 " not in calls[0] and '"status": "unmet"' in calls[1]
    assert [s for s, _ in seen] == ["s1", "s2", "s2", "s2"]                 # s2: try, retry, after the replan
    assert seen[2][1] == ["goroutines meta", "goroutine channels at meta"]
    assert t["replan_1"]["changed"] == ["s2"] and t["plan_result"]["s2"]["status"] == "met"
    assert t["plan_result"]["s1"]["attempts"] == 1


def test_a_replan_that_changes_nothing_stops_and_the_answer_says_what_was_not_found():
    plan = {"subqueries": [GIL, META], "outline": []}
    calls, gen = _setup([plan, plan], grader=FakeBackend("g", structured={"queries": []}))
    seen = {}
    real = gen.stream_text
    gen.stream_text = lambda messages, params: seen.setdefault("prompt", messages[1][1]) and real(messages, params)
    query_graph.graph.invoke({"question": "Goroutines at Meta and the GIL?"})
    t = _trace()
    assert t["replan_1"] == "unchanged" and t["plan_result"]["s2"]["status"] == "unmet"
    assert "s2: Goroutines at Meta — NO SOURCE FOUND" in seen["prompt"]


def test_no_source_parts_are_not_searched():
    calls, _ = _setup([{"subqueries": [GIL, {**META, "no_source": True}], "outline": []}])
    query_graph.graph.invoke({"question": "GIL and goroutines at Meta?"})
    t = _trace()
    assert t["plan_result"]["s2"]["status"] == "no_source" and t["plan_result"]["s2"]["attempts"] == 0


@pytest.mark.parametrize("bad, why", [
    ({"subqueries": [GIL], "outline": []}, "not needed"),
    ({"subqueries": [GIL, {**GO, "criteria": [{"kind": "covers", "value": "x"}]}]}, "rejected"),
    ({"subqueries": [GIL, {**GO, "search_queries": ["协程怎么通信"]}]}, "rejected"),
    ({"subqueries": [{**GIL, "depends_on": ["s2"]}, {**GO, "depends_on": ["s1"]}]}, "rejected"),
])
def test_plans_that_cannot_be_used_take_the_simple_path(bad, why):
    _setup([bad])
    out = query_graph.graph.invoke({"question": "GIL and goroutines?"})
    t = _trace()
    assert t["plan"] == why and "retrieval_0" in t and out["answer"] == "An answer."


def test_a_failing_planner_takes_the_simple_path():
    _index()
    make_registry({"router": {"backend": "r"}, "planner": {"backend": "bad"}, "generator": {"backend": "gen", "stream": True}},
                  {"r": FakeBackend("r", structured=_router()), "bad": FakeBackend("bad", error=RuntimeError("down")),
                   "gen": FakeBackend("gen", stream_script=["An answer."])})
    out = query_graph.graph.invoke({"question": "GIL and goroutines?"})
    assert _trace()["plan"].startswith("planner unavailable") and out["answer"] == "An answer."


def test_parts_of_a_question_that_wants_an_answer_are_answered_not_listed():
    calls, _ = _setup([{"subqueries": [{**GIL, "intent": "topic"}, {**GO, "intent": "topic"}]}])
    query_graph.graph.invoke({"question": "What does the GIL do to threads, and how do goroutines communicate?"})
    t = _trace()
    assert t["plan_topic_to_solve"] == ["s1", "s2"] and all(s["intent"] == "solve" for s in t["plan"]["subqueries"])
    calls, _ = _setup([{"subqueries": [{**GIL, "intent": "topic"}, {**GO, "intent": "topic"}]}],
                      router=_router(intent="topic"))
    query_graph.graph.invoke({"question": "Common GIL questions, and common goroutine questions?"})
    assert _trace()["plan_topic_to_solve"] == []


def test_simple_questions_never_reach_the_planner():
    calls, _ = _setup([{"subqueries": [GIL, GO]}], router=_router(complexity="simple"))
    query_graph.graph.invoke({"question": "Explain the GIL"})
    assert calls == [] and "plan" not in _trace()


def test_several_questions_are_planned_even_when_the_router_calls_them_simple():
    calls, _ = _setup([{"subqueries": [GIL, GO]}], router=_router(complexity="simple"))
    query_graph.graph.invoke({"question": "What is the GIL? How do goroutines communicate?"})
    assert len(calls) == 1


def test_when_the_router_fails_rules_decide_the_complexity():
    _index()
    fn, calls = _plans({"subqueries": [GIL, GO]})
    make_registry({"router": {"backend": "bad"}, "planner": {"backend": "p"}, "generator": {"backend": "gen", "stream": True}},
                  {"bad": FakeBackend("bad", error=RuntimeError("down")), "p": FakeBackend("p", structured=fn),
                   "gen": FakeBackend("gen", stream_script=["An answer."])})
    query_graph.graph.invoke({"question": "Kafka 和 RabbitMQ 有什么区别？各自适合什么场景？"})
    assert len(calls) == 1
    assert planner.rule_complexity("什么是进程") == ("simple", "")


def test_a_dependent_part_searches_with_what_the_first_part_found(monkeypatch):
    seen = []
    real = planner._retrieve
    monkeypatch.setattr(planner, "_retrieve", lambda sub, q, p: seen.append((sub.id, list(q))) or real(sub, q, p))
    dep = {**GO, "search_queries": ["answers to {s1}"], "depends_on": ["s1"]}
    _setup([{"subqueries": [GIL, dep]}])
    query_graph.graph.invoke({"question": "Common GIL questions, and how to answer them?"})
    assert seen[0][0] == "s1" and seen[1][0] == "s2" and "GIL" in seen[1][1][0] and "{s1}" not in seen[1][1][0]


def test_covers_criteria_are_checked_by_the_grader_and_count_as_met_when_it_fails():
    sub = planner.SubQuery(**_sub("s1", "GIL", ["gil"], [("has_kind", "question"), ("covers", "explains the GIL"),
                                                         ("covers", "gives a benchmark")]))
    items = [{"label": "Q1", "kind": "question", "title": "What is the GIL?", "text": "a lock"}]
    make_registry({"grader": {"backend": "g"}}, {"g": FakeBackend("g", structured={
        "supported": [0], "unsupported": [1], "queries": ["gil benchmark"]})})
    checks, calls, new = planner.check_criteria(sub, items, ["gil"])
    assert [c["met"] for c in checks] == [True, True, False] and calls == 1 and new == ["gil benchmark"]
    make_registry({"grader": {"backend": "g"}}, {"g": FakeBackend("g", error=RuntimeError("down"))})
    checks, _, _ = planner.check_criteria(sub, items, ["gil"])
    assert all(c["met"] for c in checks) and checks[1]["why"].startswith("unchecked")


def test_code_criteria():
    items = [{"label": "D1#0", "kind": "doc", "title": "My Meta onsite", "text": "they asked about Kafka",
              "rerank_score": 1.2},
             {"label": "Q2", "kind": "question", "title": "Explain consistent hashing", "text": "", "rerank_score": -2.0}]
    c = planner.Criterion
    assert planner.check_code(c(kind="min_items", value="2"), items, 0.0) == (False, "1 relevant items, 2 needed")
    assert planner.check_code(c(kind="has_kind", value="experience"), items, 0.0)[0]
    assert planner.check_code(c(kind="mentions", value="meta"), items, 0.0)[0]
    assert not planner.check_code(c(kind="mentions", value="Google"), items, 0.0)[0]


def test_plan_context_takes_items_from_every_part_in_turns():
    plan = planner.Plan(subqueries=[planner.SubQuery(**GIL), planner.SubQuery(**GO)])
    results = {"s1": {"items": [{"label": f"Q{n}", "title": "", "kind": "question"} for n in range(10)]},
               "s2": {"items": [{"label": "Q100", "title": "", "kind": "question"}, {"label": "Q1", "title": "",
                                                                                     "kind": "question"}]}}
    labels = [i["label"] for i in planner.plan_context(plan, results, 4)]
    assert labels == ["Q0", "Q100", "Q1", "Q2"]
