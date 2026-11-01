import json

import numpy as np

from sde_rag import db, review
from sde_rag.graphs import query as query_graph
from sde_rag.index import reranker as reranker_mod
from tests.conftest import FakeBackend, make_registry
from tests.test_graphs import _index, _router

REVIEW_OK = {"verdict": "correct", "domain_accuracy": 5, "solution_correctness": 5, "issues": [], "confidence": 0.9}
REVIEW_BAD = {"verdict": "incorrect", "domain_accuracy": 2, "solution_correctness": 2, "confidence": 0.8,
              "issues": [{"claim": "the GIL makes threads useless", "problem": "I/O-bound threads still help",
                          "correction": "threads overlap I/O; use processes for CPU-bound work", "severity": "major"}]}


class KeywordReranker:
    """Scores by how many of `words` an item contains: the test decides the order."""
    name = "keyword"

    def __init__(self, words):
        self.words = words

    def score(self, query, texts):
        return np.array([sum(w in t.lower() for w in self.words) for t in texts], dtype=np.float32)


def _qid(pattern):
    return db.conn().execute("SELECT id FROM questions WHERE text LIKE ?", (pattern,)).fetchone()[0]


def _registry(**roles):
    backends = {"r": FakeBackend("r", structured=_router("solve", ["python GIL threads"])),
                "gen": FakeBackend("gen", stream_script=["An answer."])}
    cfg = {"router": {"backend": "r"}, "generator": {"backend": "gen", "stream": True}}
    for role, backend in roles.items():
        backends[role] = backend
        cfg[role] = {"backend": role}
    return make_registry(cfg, backends)


def test_rerank_orders_by_the_cross_encoder_and_moves_answers_judged_wrong_down():
    _index()
    gil = _qid("%GIL%")
    reranker_mod.set_reranker(KeywordReranker(["gil", "thread", "python"]))
    _registry()
    out = query_graph.graph.invoke({"question": "Explain the GIL"})
    trace = json.loads(db.conn().execute("SELECT trace FROM query_logs").fetchone()[0])
    assert trace["context"][0] == f"Q{gil}" and trace["rerank_0"] == "keyword"
    # The reviewer judged the GIL answer wrong: it drops below the others, and its corrections travel with it.
    job = review.enqueue("source_answer", gil, "What is the GIL?", "x", priority=4)
    with db.tx() as t:
        review._finish(t, job, review.Review(**REVIEW_BAD), "test")
    db.conn().execute("DELETE FROM query_logs")
    seen = {}
    real = query_graph.generate

    def spy(state):
        seen["context"] = state["context"]
        return real(state)

    query_graph.generate = spy                       # build_graph looks the node functions up when it runs
    try:
        query_graph.build_graph().compile().invoke({"question": "Explain the GIL"})
    finally:
        query_graph.generate = real
    labels = [i["label"] for i in seen["context"]]
    assert labels.index(f"Q{gil}") > 0
    note = next(i for i in seen["context"] if i["label"] == f"Q{gil}")["text"]
    assert "Reviewer: this source answer has errors" in note and "threads overlap I/O" in note
    assert out["answer"] == "An answer."


def test_one_long_post_cannot_fill_the_whole_context(monkeypatch):
    items = [{"label": f"D1#{n}", "kind": "doc", "doc_id": 1, "title": "Long post", "text": "rate limiter " * 5}
             for n in range(6)] + [{"label": "Q9", "kind": "question", "title": "Design a rate limiter", "text": "x"}]
    reranker_mod.set_reranker(KeywordReranker(["rate limiter"]))
    out = query_graph.rerank({"question": "rate limiter", "context": items, "rounds": 0})
    labels = [i["label"] for i in out["context"]]
    assert sum(lbl.startswith("D1#") for lbl in labels) == 3 and "Q9" in labels


def test_grader_asks_for_another_search_at_most_twice():
    _index()
    calls = []

    def grader(schema, messages):
        calls.append(messages[1][1])
        return {"sufficient": False, "missing": "more", "queries": [f"goroutine channels {len(calls)}"]}

    _registry(grader=FakeBackend("grader", structured=grader))
    query_graph.graph.invoke({"question": "How do goroutines communicate?"})
    trace = json.loads(db.conn().execute("SELECT trace FROM query_logs").fetchone()[0])
    assert len(calls) == 2                                   # rounds 0 and 1 graded; round 2 hits the limit
    assert "retrieval_2" in trace and "grade_2" not in trace
    assert trace["grade_0"]["queries"] == ["goroutine channels 1"]


def test_a_broken_grader_counts_as_enough():
    _index()
    _registry(grader=FakeBackend("grader", error=RuntimeError("down")))
    out = query_graph.graph.invoke({"question": "How do goroutines communicate?"})
    trace = json.loads(db.conn().execute("SELECT trace FROM query_logs").fetchone()[0])
    assert out["answer"] == "An answer." and trace["grade_0"] == "unavailable" and "retrieval_1" not in trace


def test_a_solve_answer_is_queued_reviewed_and_the_verdict_kept():
    _index()
    reviewer = FakeBackend("rev", structured=REVIEW_BAD)
    _registry(reviewer=reviewer)
    out = query_graph.graph.invoke({"question": "Explain the GIL"})
    job_id = int(out["review_job_id"])
    assert review.job(job_id)["status"] == "pending"
    log = db.conn().execute("SELECT id, review_job_id FROM query_logs").fetchone()
    assert log["review_job_id"] == job_id
    assert review.run_one() == {"id": job_id, "status": "done", "verdict": "incorrect"}
    j = review.job(job_id)
    assert j["result"]["issues"][0]["severity"] == "major" and review.run_one() is None
    make_registry({"router": {"backend": "r"}, "composer": {"backend": "c", "stream": True},
                   "reviewer": {"backend": "rev"}},
                  {"r": FakeBackend("r", structured=_router("topic", ["python interview questions"], ["python"])),
                   "c": FakeBackend("c", stream_script=["- What is the GIL?"]), "rev": reviewer})
    topic = query_graph.graph.invoke({"question": "Common Python questions"})
    assert topic["intent"] == "topic" and topic["review_job_id"] is None     # lists are not reviewed


def test_quota_retries_and_alerts(monkeypatch):
    reviewer = FakeBackend("rev", error=RuntimeError("cli down"))
    _registry(reviewer=reviewer)
    monkeypatch.setenv("SDE_SETTING__REVIEW__DAILY_QUOTA", "5")
    from sde_rag import settings
    settings.reset_caches()
    jobs = [review.enqueue("query_answer", i, "q", "a", priority=2) for i in range(3)]
    for _ in range(3):
        assert review.run_one()["status"] == "error"
    assert review.job(jobs[0])["status"] == "failed" and "cli down" in review.job(jobs[0])["error"]
    assert review.job(jobs[1])["status"] == "pending"
    assert db.conn().execute("SELECT resolved_at FROM alerts WHERE key = 'reviewer'").fetchone()[0] is None
    assert review.quota_left() == 2
    reviewer.error = None
    reviewer.structured = REVIEW_OK
    review.run_one()
    review.run_one()
    assert review.quota_left() == 0 and review.run_one() is None             # quota used up
    assert db.conn().execute("SELECT resolved_at FROM alerts WHERE key = 'reviewer'").fetchone()[0] is not None


def test_user_request_jumps_the_queue():
    _registry(reviewer=FakeBackend("rev", structured=REVIEW_OK))
    first = review.enqueue("query_answer", 1, "q1", "a1", priority=2)
    second = review.enqueue("query_answer", 2, "q2", "a2", priority=2)
    review.request_review(second)
    assert review.run_one()["id"] == second and review.run_one()["id"] == first


def test_source_answers_are_reviewed_in_batches_within_the_reserve(monkeypatch):
    _index()
    seen = []

    def batch(schema, messages):
        n = messages[1][1].count("] Question: ")
        seen.append(n)
        return {"items": [{**(REVIEW_BAD if i == 0 else REVIEW_OK), "idx": i} for i in range(n)]}

    _registry(reviewer=FakeBackend("rev", structured=batch))
    monkeypatch.setenv("SDE_SETTING__REVIEW__SOURCE_BATCH", "2")
    monkeypatch.setenv("SDE_SETTING__REVIEW__MIN_ANSWER_CHARS", "10")
    from sde_rag import settings
    settings.reset_caches()
    st = review.review_source_answers(4)
    assert st["calls"] == 2 and st["reviewed"] == 4 and st["incorrect"] == 2 and seen == [2, 2]
    verdicts = [r[0] for r in db.conn().execute("SELECT answer_verdict FROM questions WHERE answer_verdict IS NOT NULL")]
    assert sorted(verdicts) == ["correct", "correct", "incorrect", "incorrect"]
    assert review.calls_today() == 2
    monkeypatch.setenv("SDE_SETTING__REVIEW__INTERACTIVE_RESERVE", "1000")      # what is left is kept for people
    settings.reset_caches()
    assert review.review_source_answers(4).get("stopped") == "quota reserve reached" and seen == [2, 2]
    monkeypatch.setenv("SDE_SETTING__REVIEW__INTERACTIVE_RESERVE", "0")
    settings.reset_caches()
    again = review.review_source_answers(4)
    assert again["reviewed"] == sum(seen[2:]) > 0               # the questions not reviewed yet, nothing twice
    assert db.conn().execute("SELECT COUNT(DISTINCT subject_id) FROM review_jobs WHERE kind = 'source_answer'"
                             ).fetchone()[0] == 4 + again["reviewed"]


def test_review_marker_text():
    from sde_rag.ui.app import review_md
    _registry(reviewer=FakeBackend("rev", structured=REVIEW_BAD))
    job = review.enqueue("query_answer", 1, "q", "a", priority=2)
    assert "评审中" in review_md(job)
    review.run_one()
    md = review_md(job)
    assert md.startswith("**评审员：有错误**") and "更正：threads overlap I/O" in md and "[较严重]" in md
