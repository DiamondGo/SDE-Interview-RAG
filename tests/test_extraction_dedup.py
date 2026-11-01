import numpy as np
import pytest

from sde_rag import db
from sde_rag.index import embedder as embedder_mod
from sde_rag.pipeline import dedup, extraction, stages, tagging
from sde_rag.pipeline.fetch import sync_source
from sde_rag.text import hamming, quote_similarity, simhash
from tests.conftest import FakeBackend, make_registry
from tests.fakes import EXPERIENCE_MD, FakeConnector

POST = """# My Amazon onsite

The loop had four rounds. In the first round the interviewer asked me to implement an LRU cache with O(1)
get and put, then how to make it thread-safe.

The system design round: "Design a rate limiter for a public API". We talked about sliding windows and Redis.

Behavioral: tell me about a conflict with a coworker. Overall it went fine and I got the offer.
""" + "\nFiller paragraph about preparation and mock interviews.\n" * 10


def test_quote_similarity_accepts_verbatim_and_light_edits_but_not_inventions():
    assert quote_similarity("implement an LRU cache with O(1) get and put", POST) == 1.0
    assert quote_similarity("Implement an **LRU cache** with O(1) get and put!", POST) == 1.0
    assert quote_similarity("implement an LRU cache with O(1) get & put", POST) > 0.8
    assert quote_similarity("asked me to implement a LRU cache with O(1) get and put", POST) > 0.8
    assert quote_similarity("Reverse a linked list in place without extra memory", POST) < 0.5
    assert quote_similarity("", POST) == 0.0
    zh = "一面问了如何设计一个限流器，然后聊了滑动窗口。"
    assert quote_similarity("如何设计一个限流器", zh) == 1.0
    assert quote_similarity("如何实现分布式锁", zh) < 0.5


def test_simhash_is_close_for_light_edits_and_far_for_different_text():
    a, b = simhash(POST), simhash(POST.replace("fine", "well") + "\nEdit: typo fixed.\n")
    assert hamming(a, b) <= 6
    assert hamming(a, simhash(EXPERIENCE_MD)) > 12
    assert -(1 << 63) <= a < (1 << 63)


def _extraction(questions, interview=True, **meta):
    return {"is_interview_content": interview, "company": meta.get("company"), "level": meta.get("level"),
            "role": None, "year": meta.get("year"), "questions": questions}


def _posts(docs: dict):
    sync_source("devto", FakeConnector("devto", {k: {"version": "1", "content": v, "kind": "experience"}
                                                 for k, v in docs.items()}))
    stages.clean_stage(10)
    stages.parse_stage(10)


def _doc(eid: str):
    return db.conn().execute("SELECT * FROM documents WHERE external_id = ?", (eid,)).fetchone()


def test_extraction_keeps_quoted_questions_drops_invented_ones_and_records_the_company():
    _posts({"amz": POST})
    reply = _extraction([
        {"text": "Implement an LRU cache with O(1) get and put.", "round": "Onsite",
         "original_text": "implement an LRU cache with O(1) get and put", "answer_text": "Hash map plus a list."},
        {"text": "Design a rate limiter for a public API.", "round": "system design",
         "original_text": "\"Design a rate limiter for a public API\""},
        {"text": "Reverse a linked list.", "original_text": "reverse a linked list in place"},   # not in the post
    ], company="Amazon", level="None", year="2025")
    ext = FakeBackend("ext", structured=reply)
    make_registry({"extractor": {"backend": "ext"}}, {"ext": ext})
    st = extraction.extract_stage(10)
    assert (st["docs"], st["proposed"], st["kept"], st["unquoted"]) == (1, 3, 2, 1)
    assert st["fabrication_rate"] == pytest.approx(0.333, abs=0.001)
    rows = db.conn().execute("SELECT m.company, m.level, m.round, m.year, m.extractor, q.text, q.answer "
                             "FROM question_mentions m JOIN questions q ON q.id = m.question_id "
                             "WHERE m.extractor LIKE 'llm:%' ORDER BY m.ordinal").fetchall()
    assert [r["text"] for r in rows] == ["Implement an LRU cache with O(1) get and put.",
                                         "Design a rate limiter for a public API."]
    assert {(r["company"], r["level"], r["round"], r["year"]) for r in rows} == {
        ("Amazon", None, "onsite", 2025)}
    assert rows[0]["answer"] == "Hash map plus a list."
    assert _doc("amz")["extract_status"] == "done"
    calls = ext.calls
    assert extraction.extract_stage(10)["docs"] == 0 and ext.calls == calls        # read once


def test_a_rule_reparse_keeps_the_extracted_questions():
    _posts({"amz": POST})
    make_registry({"extractor": {"backend": "ext"}}, {"ext": FakeBackend("ext", structured=_extraction([
        {"text": "Implement an LRU cache.", "original_text": "implement an LRU cache with O(1) get and put"}]))})
    extraction.extract_stage(10)
    db.conn().execute("UPDATE documents SET status = 'cleaned'")
    stages.parse_stage(10)
    assert db.conn().execute("SELECT COUNT(*) FROM question_mentions WHERE extractor LIKE 'llm:%'").fetchone()[0] == 1


def test_not_interview_posts_and_reposts_are_not_extracted_twice():
    _posts({"tut": "# Saga pattern\n" + "A saga coordinates local transactions across services. " * 40,
            "orig": POST})
    make_registry({"extractor": {"backend": "ext"}}, {"ext": FakeBackend("ext", structured=lambda schema, m: (
        _extraction([], interview=False) if "Saga" in m[1][1] else _extraction([
            {"text": "Implement an LRU cache.", "original_text": "implement an LRU cache with O(1) get and put"}])))})
    st = extraction.extract_stage(10)
    assert st["not_interview"] == 1 and st["kept"] == 1
    assert _doc("tut")["extract_status"] == "not_interview"
    sync_source("hn", FakeConnector("hn", {"copy": {"version": "1", "kind": "experience",
                                                   "content": POST.replace("fine", "well")}}))
    stages.clean_stage(10)
    stages.parse_stage(10)
    assert extraction.extract_stage(10)["duplicates"] == 1
    assert _doc("copy")["dup_of"] == _doc("orig")["id"]


def test_failed_extraction_is_retried_then_dead_then_replayed():
    _posts({"amz": POST})
    make_registry({"extractor": {"backend": "bad"}}, {"bad": FakeBackend("bad", error=RuntimeError("down"))})
    for expected in ("failed", "failed", "dead"):
        assert extraction.extract_stage(10)["failed"] == 1
        assert _doc("amz")["extract_status"] == expected
    assert extraction.extract_stage(10)["failed"] == 0
    assert [r["stage"] for r in stages.dead_letters("extract")] == ["extract"]
    assert stages.replay("extract") == {"extract": 1}
    assert _doc("amz")["extract_status"] is None and _doc("amz")["extract_attempts"] == 0


def test_extraction_waits_while_the_extractor_is_unavailable():
    _posts({"amz": POST})
    make_registry({}, {})
    assert extraction.extract_stage(10)["skipped"]
    assert _doc("amz")["extract_status"] is None


class FakeEmbedder:
    """Vectors chosen per text: pairs of questions get exactly the similarity a test needs."""
    name, dim = "fake-emb", 3

    def __init__(self, sims: dict[str, tuple[str, float]]):
        self.vecs: dict[str, np.ndarray] = {}
        n = 0
        for a, (b, s) in sims.items():
            base = np.zeros(3 + 2 * len(sims))
            base[3 + n] = 1.0
            other = np.zeros_like(base)
            other[3 + n], other[4 + n] = s, np.sqrt(1 - s * s)
            self.vecs[a], self.vecs[b] = base, other
            n += 2
        self.dim = 3 + 2 * len(sims)

    def embed_queries(self, texts):
        out = []
        for i, t in enumerate(texts):
            v = self.vecs.get(t)
            if v is None:        # unrelated to everything
                v = np.zeros(self.dim)
                v[i % 3] = 1.0
            out.append(v)
        return np.array(out, dtype=np.float32)

    embed_passages = embed_queries


PAIRS = {
    "How to create components in React?": ("How do you create a component in React?", 0.99),   # same words
    "What is a private IP address?": ("What is a public IP address?", 0.97),                   # one word apart
    "What is memory management in Python?": ("How does Python manage memory?", 0.93),          # judge says same
    "What is the output of the code below?": ("What will be the output of the code below?", 0.99),
    "Can a generator be used to create iterators?": ("Can iterators be used to create generators?", 0.97),
}


def _questions(monkeypatch):
    md = "# Bank\n" + "".join(f"## {q}\nAnswer to {q}\n" for pair in PAIRS.items() for q in (pair[0], pair[1][0]))
    sync_source("a", FakeConnector("a", {"bank": {"version": "1", "content": md}}))
    stages.clean_stage(10)
    stages.parse_stage(10)
    tagging.rule_tag_stage()
    embedder_mod.set_embedder(FakeEmbedder(PAIRS))


def _qid(text):
    row = db.conn().execute("SELECT id FROM questions WHERE text = ?", (text,)).fetchone()
    return row[0] if row else None


def test_near_duplicates_merge_only_when_safe_and_the_rest_goes_to_the_judge(monkeypatch):
    _questions(monkeypatch)
    def verdicts(schema, m):
        firsts = [line for line in m[1][1].splitlines() if line.startswith("A: ")]
        return {"same": [i for i, a in enumerate(firsts) if "memory" in a],
                "different": [i for i, a in enumerate(firsts) if "memory" not in a]}

    judge = FakeBackend("j", structured=verdicts)
    make_registry({"dedup_judge": {"backend": "j"}}, {"j": judge})
    st = dedup.dedup_stage(judge_enabled=True, judge_limit=100)
    assert st["embedded"] == 10 and st["merged"] == 1
    assert _qid("How to create components in React?") and not _qid("How do you create a component in React?")
    assert st["judge_judged"] == 3 and st["judge_same"] == 1
    assert bool(_qid("What is memory management in Python?")) != bool(_qid("How does Python manage memory?"))
    assert _qid("What is a private IP address?") and _qid("What is a public IP address?")
    assert _qid("Can a generator be used to create iterators?") and _qid("Can iterators be used to create generators?")
    assert _qid("What is the output of the code below?") and _qid("What will be the output of the code below?")
    verdicts = dict(((r[0], r[1]), r[2]) for r in db.conn().execute("SELECT a, b, verdict FROM dedup_pairs"))
    assert sorted(verdicts.values()) == ["different", "different"]
    merges = db.conn().execute("SELECT how, dropped_text FROM question_merges ORDER BY id").fetchall()
    assert [m["how"] for m in merges] == ["auto", "judge"]
    stages.merge_stage()
    kept = _qid("How to create components in React?")
    assert db.conn().execute("SELECT mention_count FROM questions WHERE id = ?", (kept,)).fetchone()[0] == 2
    # Nothing new: the next run neither re-embeds nor re-judges.
    again = dedup.dedup_stage(judge_enabled=True, judge_limit=100)
    assert again["embedded"] == 0 and again["checked"] == 0 and again["judge_judged"] == 0


def test_a_merged_wording_parsed_again_attaches_to_the_kept_question(monkeypatch):
    _questions(monkeypatch)
    make_registry({}, {})
    dedup.dedup_stage(judge_enabled=False, judge_limit=0)
    kept = _qid("How to create components in React?")
    n = db.conn().execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    db.conn().execute("UPDATE documents SET status = 'cleaned', parser_version = 'old'")
    stages.parse_stage(10)
    assert db.conn().execute("SELECT COUNT(*) FROM questions").fetchone()[0] == n
    assert not _qid("How do you create a component in React?")
    assert db.conn().execute("SELECT COUNT(*) FROM question_mentions WHERE question_id = ?", (kept,)).fetchone()[0] == 2


def test_content_words_ignore_question_words_and_plurals():
    a = dedup.content_words("How to create components in React?")
    b = dedup.content_words("How do you create a component in React?")
    assert a[0] == b[0] == ("create", "component", "react")
    assert dedup.points_at_context("What is the output of the following code?")
    assert not dedup.points_at_context("What is the output layer of a neural network?")
