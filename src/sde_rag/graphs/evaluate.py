"""Evaluate graph. Milestone 1 has one suite; generation, reviewer and agent suites come in later milestones.

retrieval_self: sample canonical questions, search with each question's own text (no LLM involved) and check
that its chunk comes back. A sanity floor for the index (catches broken embeddings, FTS or snapshots), not a
measure of real retrieval quality, which needs paraphrased and human-written queries (milestone 3).
"""

import json
import random
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from sde_rag import db, retrieval
from sde_rag.auth import Principal
from sde_rag.index.versions import check_model

MAX_DROP = 0.03


@dataclass
class EvalContext:
    sample_size: int = 100
    seed: int = 7


class EvalInput(TypedDict, total=False):
    suite: str
    compare_to: int | None     # eval_runs.id of the baseline; default: the previous run of the same suite


class EvalOutput(TypedDict, total=False):
    metrics: dict
    baseline: dict
    deltas: dict
    passed: bool
    run_id: int


class EvalState(EvalInput, EvalOutput, total=False):
    pass


def retrieval_self(sample_size: int, seed: int) -> dict:
    version = check_model()
    rows = db.conn().execute(
        "SELECT c.id AS chunk_id, q.text, q.category FROM chunks c JOIN questions q ON q.id = c.question_id "
        "WHERE c.kind = 'question' AND c.index_version = ? AND c.embedded = 1 ORDER BY c.id", (version,)).fetchall()
    if not rows:
        return {"n": 0}
    sample = random.Random(seed).sample(list(rows), min(sample_size, len(rows)))
    p = Principal()
    r5 = r20 = rr = 0.0
    per_cat: dict[str, list[int]] = {}
    for row in sample:
        ids = [h.chunk_id for h in retrieval.hybrid(p, [row["text"]], kinds=["question"], k=20)]
        rank = ids.index(row["chunk_id"]) + 1 if row["chunk_id"] in ids else None
        r5 += bool(rank and rank <= 5)
        r20 += bool(rank)
        rr += 1 / rank if rank else 0
        per_cat.setdefault(row["category"] or "none", []).append(1 if rank and rank <= 5 else 0)
    n = len(sample)
    return {"n": n, "recall@5": round(r5 / n, 4), "recall@20": round(r20 / n, 4), "mrr": round(rr / n, 4),
            "recall@5_by_category": {k: round(sum(v) / len(v), 3) for k, v in sorted(per_cat.items())}}


SUITES = {"retrieval_self": retrieval_self}


def run_suite(state: EvalState, runtime: Runtime[EvalContext]) -> dict:
    suite = state.get("suite") or "retrieval_self"
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; available: {', '.join(SUITES)}")
    ctx = runtime.context or EvalContext()
    return {"suite": suite, "metrics": SUITES[suite](ctx.sample_size, ctx.seed)}


def compare(state: EvalState) -> dict:
    c = db.conn()
    suite, metrics = state["suite"], state["metrics"]
    if state.get("compare_to"):
        base = c.execute("SELECT metrics FROM eval_runs WHERE id = ?", (state["compare_to"],)).fetchone()
    else:
        base = c.execute("SELECT metrics FROM eval_runs WHERE suite = ? ORDER BY id DESC LIMIT 1", (suite,)).fetchone()
    baseline = json.loads(base["metrics"]) if base else {}
    deltas = {k: round(metrics[k] - baseline[k], 4) for k in metrics
              if isinstance(metrics[k], (int, float)) and k != "n" and isinstance(baseline.get(k), (int, float))}
    passed = metrics.get("n", 0) > 0 and all(d >= -MAX_DROP for d in deltas.values())
    run_id = c.execute("INSERT INTO eval_runs (suite, created_at, metrics, passed) VALUES (?, ?, ?, ?)",
                       (suite, db.now_iso(), json.dumps(metrics), int(passed))).lastrowid
    return {"baseline": baseline, "deltas": deltas, "passed": passed, "run_id": run_id}


def build_graph() -> StateGraph:
    b = StateGraph(EvalState, context_schema=EvalContext, input_schema=EvalInput, output_schema=EvalOutput)
    b.add_node("run_suite", run_suite)
    b.add_node("compare", compare)
    b.add_edge(START, "run_suite")
    b.add_edge("run_suite", "compare")
    b.add_edge("compare", END)
    return b


graph = build_graph().compile()
