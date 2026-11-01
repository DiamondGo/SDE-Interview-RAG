"""The reviewer (DESIGN 7.6): is an answer's domain knowledge accurate and its solution correct?

Different from faithfulness: a faithful summary of a wrong source answer is still wrong. Jobs wait in
review_jobs and run one at a time (the reviewer is a CLI agent with one process) by priority:
  1  the user asked for a review           2  a `solve` answer, reviewed after it was returned
  4  source answers of high-heat questions (the ingest run, in batches)
within a daily quota of calls. Below `interactive_reserve` calls left, only priorities 1 and 2 run.
A failed call goes back to the queue; three failures in a row raise an alert.
"""

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, field_validator

from sde_rag import db, notify
from sde_rag.agents import AgentError, get_registry
from sde_rag.settings import section

log = logging.getLogger("sde_rag.review")
ROLE = "reviewer"
VERDICTS = ("correct", "minor_issues", "incorrect", "unverifiable")


class ReviewIssue(BaseModel):
    claim: str
    problem: str
    correction: str = ""
    severity: Literal["minor", "major", "critical"] = "minor"


class Review(BaseModel):
    verdict: Literal["correct", "minor_issues", "incorrect", "unverifiable"]
    domain_accuracy: int = 3          # 1 to 5
    solution_correctness: int = 3     # 1 to 5, including edge cases
    complexity_correct: bool | None = None
    issues: list[ReviewIssue] = []
    confidence: float = 0.5

    @field_validator("issues", mode="before")
    @classmethod
    def _decode(cls, v):
        return json.loads(v) if isinstance(v, str) else (v or [])


class ItemReview(Review):
    idx: int


class BatchReview(BaseModel):
    items: list[ItemReview]

    @field_validator("items", mode="before")
    @classmethod
    def _decode(cls, v):
        return json.loads(v) if isinstance(v, str) else v


_CHECKS = """Check:
- knowledge: behaviour of languages and frameworks (including version differences), definitions, outdated claims;
- algorithm questions: whether the solution is correct, edge cases, time and space complexity, obvious bugs in code;
- system design: there is no single right answer, so check whether the claims hold, whether key trade-offs or the
  main bottleneck are missing, and whether order-of-magnitude estimates are reasonable.
Report every problem as an issue (the claim, what is wrong, the correct statement, minor/major/critical).
verdict: correct (no issues that matter), minor_issues, incorrect (a major or critical issue), or unverifiable
when you cannot tell; never force a verdict. domain_accuracy and solution_correctness: 1 to 5.
complexity_correct: true/false for algorithm answers that state a complexity, null otherwise.
confidence: 0 to 1. Answer only with the structured output."""


def review_prompt(question: str, answer: str) -> list[tuple[str, str]]:
    system = "You are a senior software engineer reviewing answers to interview questions for correctness."
    user = f"""Is this answer to an interview question correct?

{_CHECKS}

Judge the answer itself, not whether it cites sources. A section titled "模型补充" or "Model supplement" is the
answerer's own knowledge: review it like the rest.

Question:
{question}

Answer:
<<<
{answer}
>>>"""
    return [("system", system), ("user", user)]


def batch_prompt(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    system = "You are a senior software engineer reviewing source answers in an interview question bank."
    blocks = "\n\n".join(f"[{i}] Question: {q}\nAnswer:\n<<<\n{a}\n>>>" for i, (q, a) in enumerate(items))
    user = f"""Review each numbered question's answer, giving idx and the review fields for every item.

{_CHECKS}

{blocks}"""
    return [("system", system), ("user", user)]


# ---- queue -------------------------------------------------------------------------------

def _cfg() -> dict:
    return section("review")


def enabled() -> bool:
    """Below its capability thresholds (or never checked) the reviewer pauses: unreliable verdicts mislead."""
    from sde_rag import agent_checks
    return bool(_cfg().get("enabled", True)) and get_registry().usable(ROLE) and agent_checks.allowed(ROLE)


def enqueue(kind: str, subject_id: int, question: str, answer: str, *, priority: int,
            context: str | None = None) -> int:
    return db.conn().execute(
        "INSERT INTO review_jobs (kind, subject_id, priority, status, question, answer, context, created_at) "
        "VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)",
        (kind, subject_id, priority, question[:4000], answer[:20000], (context or "")[:20000], db.now_iso())).lastrowid


def request_review(job_id: int) -> dict:
    """The user asked: move the job to the front (or queue it again when it already finished)."""
    row = db.conn().execute("SELECT status FROM review_jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise LookupError(f"no review job {job_id}")
    if row["status"] in ("pending", "failed"):
        db.conn().execute("UPDATE review_jobs SET priority = 1, status = 'pending' WHERE id = ?", (job_id,))
    return job(job_id)


def job(job_id: int) -> dict:
    r = db.conn().execute("SELECT id, status, verdict, result, error, priority FROM review_jobs WHERE id = ?",
                          (job_id,)).fetchone()
    if r is None:
        return {"id": job_id, "status": "missing"}
    out = dict(r)
    out["result"] = json.loads(r["result"]) if r["result"] else None
    return out


def _counter_key() -> str:
    return f"review_calls:{datetime.now().astimezone():%Y-%m-%d}"


def _count_call() -> None:
    """Every call to the reviewer, failed or not, retried or not (a batch is one call)."""
    db.conn().execute("INSERT INTO meta (key, value) VALUES (?, '1') ON CONFLICT(key) DO UPDATE SET "
                      "value = CAST(value AS INTEGER) + 1", (_counter_key(),))


def calls_today() -> int:
    return int(db.get_meta(_counter_key(), "0"))


def quota_left() -> int:
    return max(0, int(_cfg().get("daily_quota", 120)) - calls_today())


def _claim(priorities: tuple[int, ...]) -> dict | None:
    marks = ",".join("?" * len(priorities))
    with db.tx() as t:
        r = t.execute(f"SELECT * FROM review_jobs WHERE status = 'pending' AND priority IN ({marks}) "
                      f"ORDER BY priority, id LIMIT 1", priorities).fetchone()
        if r is None:
            return None
        t.execute("UPDATE review_jobs SET status = 'running', started_at = ?, attempts = attempts + 1 WHERE id = ?",
                  (db.now_iso(), r["id"]))
    return dict(r)


def _finish(t, job_id: int, review: Review, who: str) -> None:
    t.execute("UPDATE review_jobs SET status = 'done', verdict = ?, result = ?, reviewer = ?, error = NULL, "
              "finished_at = ? WHERE id = ?",
              (review.verdict, review.model_dump_json(), who, db.now_iso(), job_id))
    row = t.execute("SELECT kind, subject_id FROM review_jobs WHERE id = ?", (job_id,)).fetchone()
    if row["kind"] == "source_answer":
        t.execute("UPDATE questions SET answer_verdict = ?, answer_review_id = ? WHERE id = ?",
                  (review.verdict, job_id, row["subject_id"]))


def _failed(job_ids: list[int], error: str) -> None:
    max_attempts = int(_cfg().get("max_attempts", 3))
    with db.tx() as t:
        for jid in job_ids:
            t.execute("UPDATE review_jobs SET status = CASE WHEN attempts >= ? THEN 'failed' ELSE 'pending' END, "
                      "error = ? WHERE id = ?", (max_attempts, error[:2000], jid))
    n = int(db.get_meta("review_failures_in_a_row", "0")) + 1
    db.set_meta("review_failures_in_a_row", str(n))
    if n >= 3:
        notify.alert("reviewer", "Reviewer failing", f"{n} review calls in a row failed: {error[:300]}")


def _ok() -> None:
    db.set_meta("review_failures_in_a_row", "0")
    notify.resolve("reviewer", "Reviewer working again")


def run_one(priorities: tuple[int, ...] = (1, 2)) -> dict | None:
    """Review the next queued answer. None when there is nothing to do, no quota, or no usable reviewer."""
    if not enabled() or quota_left() <= 0:
        return None
    j = _claim(priorities)
    if j is None:
        return None
    reg = get_registry()
    _count_call()
    try:
        review: Review = reg.structured(ROLE, Review, review_prompt(j["question"], j["answer"]))
    except AgentError as e:
        _failed([j["id"]], str(e))
        log.warning("review of job %s failed: %s", j["id"], str(e)[:200])
        return {"id": j["id"], "status": "error", "error": str(e)[:300]}
    with db.tx() as t:
        _finish(t, j["id"], review, f"llm:{reg.fingerprint(ROLE)}")
    _ok()
    return {"id": j["id"], "status": "done", "verdict": review.verdict}


def review_source_answers(limit: int, *, enabled_now: bool = True, heartbeat=lambda: None) -> dict:
    """The ingest run's share: source answers of the hottest questions not reviewed yet, in batches, only while
    more than `interactive_reserve` calls are left today (the rest is kept for answers people are reading)."""
    cfg = _cfg()
    stats = {"reviewed": 0, "incorrect": 0, "minor_issues": 0, "calls": 0, "failed": 0}
    if not enabled_now or not enabled() or limit <= 0:
        return {**stats, "skipped": True}
    batch = int(cfg.get("source_batch", 5))
    reserve = int(cfg.get("interactive_reserve", 40))
    rows = db.conn().execute(
        "SELECT q.id, q.text, q.answer FROM questions q WHERE q.answer_verdict IS NULL AND length(q.answer) >= ? "
        "AND NOT EXISTS (SELECT 1 FROM review_jobs j WHERE j.kind = 'source_answer' AND j.subject_id = q.id "
        "AND j.status IN ('pending', 'running', 'failed')) ORDER BY q.heat DESC, q.id LIMIT ?",
        (int(cfg.get("min_answer_chars", 80)), limit)).fetchall()
    reg = get_registry()
    who = f"llm:{reg.fingerprint(ROLE)}"
    for i in range(0, len(rows), batch):
        if quota_left() <= reserve:
            stats["stopped"] = "quota reserve reached"
            break
        part = rows[i:i + batch]
        now = db.now_iso()
        ids = [enqueue("source_answer", r["id"], r["text"], r["answer"], priority=4) for r in part]
        with db.tx() as t:
            for jid in ids:
                t.execute("UPDATE review_jobs SET status = 'running', started_at = ?, attempts = 1 WHERE id = ?",
                          (now, jid))
        stats["calls"] += 1
        _count_call()
        try:
            result: BatchReview = reg.structured(ROLE, BatchReview,
                                                 batch_prompt([(r["text"], r["answer"][:3000]) for r in part]))
        except AgentError as e:
            _failed(ids, str(e))
            stats["failed"] += len(part)
            continue
        by_idx = {it.idx: it for it in result.items}
        with db.tx() as t:
            for n, jid in enumerate(ids):
                it = by_idx.get(n)
                if it is None:
                    t.execute("UPDATE review_jobs SET status = 'pending', error = 'missing from the batch' WHERE id = ?",
                              (jid,))
                    continue
                _finish(t, jid, Review(**it.model_dump(exclude={"idx"})), who)
                stats["reviewed"] += 1
                stats[it.verdict] = stats.get(it.verdict, 0) + 1
        _ok()
        heartbeat()
    stats["quota_left"] = quota_left()
    return stats


def requeue_stale(older_than_s: int = 900) -> int:
    """Jobs left `running` by a process that died."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=older_than_s)).isoformat(timespec="seconds")
    return db.conn().execute("UPDATE review_jobs SET status = 'pending' WHERE status = 'running' AND started_at < ?",
                             (cutoff,)).rowcount


class ReviewWorker:
    """In the UI process: reviews answers people asked about, one at a time, shortly after they were given."""

    def __init__(self, every_s: float = 3.0, failure_wait_s: float = 60.0):
        self.every_s, self.failure_wait_s = every_s, failure_wait_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="review-worker")

    def start(self) -> "ReviewWorker":
        requeue_stale()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            wait = self.every_s
            try:
                out = run_one((1, 2))
                if out is not None and out["status"] == "done":
                    continue
                if out is not None:          # a failing reviewer: do not spend the quota in a tight loop
                    wait = self.failure_wait_s
            except Exception:
                log.exception("review worker")
                wait = self.failure_wait_s
            self._stop.wait(wait)
