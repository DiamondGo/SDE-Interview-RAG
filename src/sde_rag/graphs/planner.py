"""Planner for complex questions (DESIGN 5.4.1).

plan -> subqueries (one Send branch each; dependent ones in a later wave) -> check_plan
     -> all met, or budget spent: synthesize
     -> unmet left: replan (met subqueries keep their results; only changed ones run again) -> subqueries ...

Every subquery carries success criteria. Three kinds are checked by code (min_items, has_kind, mentions);
`covers` statements are checked by the grader in one call per subquery. An unmet subquery is retried once
with new search terms before the planner is asked again. Every failure falls back: no usable plan -> the
simple path; a failed check -> code criteria decide and `covers` counts as met; a failed replan -> answer
with what was found.
"""

import json
import re
import time
from typing import Literal

from langgraph.types import Send
from pydantic import BaseModel, field_validator

from sde_rag import prompts, retrieval
from sde_rag.agents import AgentError, get_registry
from sde_rag.auth import Principal
from sde_rag.graphs.streaming import emit, stream_role
from sde_rag.settings import section
from sde_rag.taxonomy import taxonomy
from sde_rag.text import truncate

CODE_KINDS = ("min_items", "has_kind", "mentions")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def _decode(v):
    """Local models sometimes emit nested values as JSON-encoded strings."""
    if isinstance(v, str) and v.strip()[:1] in "[{":
        return json.loads(v)
    return v


class Criterion(BaseModel):
    kind: Literal["min_items", "has_kind", "mentions", "covers"]
    value: str

    @field_validator("value", mode="before")
    @classmethod
    def _str(cls, v):
        return str(v)


class SubQuery(BaseModel):
    id: str
    goal: str
    intent: Literal["topic", "solve", "experience"] = "solve"
    search_queries: list[str] = []
    filters: dict = {}
    criteria: list[Criterion] = []
    depends_on: list[str] = []
    no_source: bool = False           # set on replanning: no material is likely; answered from model knowledge

    @field_validator("search_queries", "criteria", "depends_on", "filters", mode="before")
    @classmethod
    def _nested(cls, v):
        return _decode(v) if v is not None else v


class Plan(BaseModel):
    subqueries: list[SubQuery]
    outline: list[str] = []

    @field_validator("subqueries", "outline", mode="before")
    @classmethod
    def _nested(cls, v):
        return _decode(v)


class CoverCheck(BaseModel):
    """Flat lists of statement numbers (lists of objects break local models' JSON more often)."""
    supported: list[int] = []
    unsupported: list[int] = []
    queries: list[str] = []

    @field_validator("supported", "unsupported", "queries", mode="before")
    @classmethod
    def _nested(cls, v):
        return _decode(v) or []


class NewQueries(BaseModel):
    sufficient: bool = True
    missing: str = ""
    queries: list[str] = []


def cfg() -> dict:
    return section("planner")


# ---- deciding to plan ----------------------------------------------------------------------

_SEVERAL = re.compile(r"[?？]")
_COMPLEX = re.compile(r"分别|以及|并且|而且|同时|各自|对比|区别|包括[^。？?]*[和、与及]|\bvs\.?\b|\bversus\b|\bcompare\b|"
                      r"\bas well as\b|\band also\b|\b(?:and|then) (?:how|why|what|when|which)\b", re.I)


def several_questions(q: str) -> bool:
    """Two or more questions in one message: planned even when the router calls it simple."""
    return len(_SEVERAL.findall(q)) >= 2


def rule_complexity(q: str) -> tuple[str, str]:
    """When the router failed: (complexity, reason)."""
    if several_questions(q):
        return "complex", "several questions"
    m = _COMPLEX.search(q)
    return ("complex", f"enumeration or comparison: {m.group(0)!r}") if m else ("simple", "")


# ---- the plan ------------------------------------------------------------------------------

def validate(plan: Plan) -> str | None:
    """Why the plan cannot be used, or None. One subquery is not an error: the question needs no plan."""
    subs, limit = plan.subqueries, int(cfg().get("max_subqueries", 4))
    if len(subs) < 2:
        return "not needed"
    if len(subs) > limit:
        return f"{len(subs)} subqueries (at most {limit})"
    ids = [s.id for s in subs]
    if len(set(ids)) != len(ids):
        return "duplicate subquery ids"
    for s in subs:
        if s.no_source:
            continue
        if not 1 <= len(s.criteria) <= 3:
            return f"{s.id}: {len(s.criteria)} criteria"
        if not any(c.kind in CODE_KINDS for c in s.criteria):
            return f"{s.id}: only covers criteria (one must be checked by code)"
        for c in s.criteria:
            if c.kind == "min_items" and not c.value.strip().isdigit():
                return f"{s.id}: min_items needs a number, got {c.value!r}"
            if c.kind == "has_kind" and c.value.strip().lower() not in ("question", "experience"):
                return f"{s.id}: has_kind must be question or experience, got {c.value!r}"
        queries = [q for q in s.search_queries if q.strip()]
        if not 1 <= len(queries) <= 3:
            return f"{s.id}: {len(queries)} search queries"
        if any(_CJK.search(q.replace("{", "").replace("}", "")) for q in queries):
            return f"{s.id}: search queries must be English"
        unknown = [d for d in s.depends_on if d not in ids or d == s.id]
        if unknown:
            return f"{s.id}: depends on unknown {unknown}"
    seen, visiting = set(), set()

    def cyclic(sid: str) -> bool:
        if sid in visiting:
            return True
        if sid in seen:
            return False
        visiting.add(sid)
        found = any(cyclic(d) for d in next(s for s in subs if s.id == sid).depends_on)
        visiting.discard(sid)
        seen.add(sid)
        return found

    return "dependency cycle" if any(cyclic(s.id) for s in subs) else None


def runnable(plan: Plan, results: dict) -> list[SubQuery]:
    return [s for s in plan.subqueries
            if s.id not in results and not s.no_source and all(d in results for d in s.depends_on)]


def sends(state: dict) -> list[Send]:
    plan, results = Plan(**state["plan"]), state.get("sub_results") or {}
    out = []
    for s in runnable(plan, results):
        deps = {d: [i["title"] for i in (results[d].get("items") or [])[:3]] for d in s.depends_on}
        out.append(Send("subquery", {"sub": s.model_dump(), "deps": deps, "user_id": state.get("user_id") or "local",
                                     "started": state.get("plan_started") or time.time(),
                                     "language": state.get("language", "en")}))
    return out


def _no_source_results(plan: Plan) -> dict:
    return {s.id: {"id": s.id, "goal": s.goal, "status": "no_source", "items": [], "criteria": [], "calls": 0,
                   "attempts": 0, "queries": []} for s in plan.subqueries if s.no_source}


def plan_node(state: dict) -> dict:
    emit({"type": "status", "text": "splitting the question into parts"})
    question = state.get("standalone") or state["question"]
    calls: list = []
    try:
        p: Plan = get_registry().structured("planner", Plan, prompts.plan_messages(
            question, state.get("language", "en"), int(cfg().get("max_subqueries", 4))), trace=calls)
    except AgentError as e:
        return {"plan": None, "trace": {"plan": f"planner unavailable: {str(e)[:200]}", "plan_calls_trace": calls}}
    why = validate(p)
    if why:
        return {"plan": None, "trace": {"plan": "not needed" if why == "not needed" else "rejected",
                                        "plan_rejected": why, "plan_raw": p.model_dump(), "plan_calls_trace": calls}}
    coerced = _answers_not_lists(p, state.get("intent"))
    emit({"type": "status", "text": "parts: " + " | ".join(f"{s.id}: {s.goal}" for s in p.subqueries)})
    return {"plan": p.model_dump(), "sub_results": {"_replace": True, **_no_source_results(p)}, "replans": 0,
            "plan_calls": 1, "plan_started": time.time(), "plan_next": None,
            "trace": {"plan": p.model_dump(), "plan_calls_trace": calls, "plan_topic_to_solve": coerced}}


def _answers_not_lists(p: Plan, intent: str | None) -> list[str]:
    """A question that wants an answer gets answers for its parts too: planners like to label parts "topic",
    which would turn an explanation into lists of interview questions."""
    if intent == "topic":
        return []
    changed = [s.id for s in p.subqueries if s.intent == "topic"]
    for s in p.subqueries:
        if s.intent == "topic":
            s.intent = "solve"
    return changed


# ---- one subquery --------------------------------------------------------------------------

def _fill(queries: list[str], deps: dict[str, list[str]]) -> list[str]:
    """"{s1}" -> the titles s1 found (dropped when it found nothing)."""
    out = []
    for q in queries:
        for d, titles in deps.items():
            q = q.replace(f"{{{d}}}", "; ".join(t[:80] for t in titles))
        q = re.sub(r"\{s\d+\}", "", q).strip(" ;,")
        if q:
            out.append(q)
    return out


def _filters(sub: SubQuery) -> tuple[str | None, list[str] | None]:
    tax = taxonomy()
    cat = sub.filters.get("category")
    tags = [t for t in (tax.resolve(str(x)) for x in sub.filters.get("tags") or []) if t]
    return (cat if cat in tax.categories else None), (tags or None)


def _retrieve(sub: SubQuery, queries: list[str], principal: Principal) -> list[dict]:
    c, category, tags = cfg(), *_filters(sub)
    company = (sub.filters.get("company") or "").strip()
    if company and not any(company.lower() in q.lower() for q in queries):
        queries = queries + [f"{company} {queries[0]}"]
    if sub.intent == "topic":
        return retrieval.topic_questions(principal, queries, category=category, tags=tags,
                                         k=int(c.get("topic_items", 10)))
    hits = retrieval.hybrid(principal, queries, kinds=["doc"] if sub.intent == "experience" else None,
                            category=category, tags=tags)
    cands = retrieval.load_context(principal, hits, int(section("query").get("rerank_candidates", 30)))
    # Against the English search query, not the goal: the goal is in the user's language, and a Chinese goal
    # scored the same relevant English items about 1.4 lower, below the min_items threshold.
    items, _ = retrieval.rerank_items(queries[0], cands, int(c.get("items_per_subquery", 4)))
    return items


def check_code(c: Criterion, items: list[dict], threshold: float) -> tuple[bool, str]:
    if c.kind == "min_items":
        n = int(c.value)
        relevant = [i for i in items if i.get("rerank_score") is None or i["rerank_score"] >= threshold]
        return len(relevant) >= n, f"{len(relevant)} relevant items, {n} needed"
    if c.kind == "has_kind":
        want = "doc" if c.value.strip().lower() == "experience" else "question"
        ok = any(i["kind"] == want for i in items)
        return ok, "" if ok else f"no {c.value.strip().lower()} among the items"
    term = c.value.strip().lower()
    tag = taxonomy().resolve(term)
    for i in items:
        text = f"{i['title']}\n{i['text']}"
        if term in text.lower() or (tag and tag in taxonomy().match(text)):
            return True, ""
    return False, f"no item mentions {c.value!r}"


def check_criteria(sub: SubQuery, items: list[dict], queries: list[str]) -> tuple[list[dict], int, list[str]]:
    """[{kind, value, met, why}], grader calls made, new search queries the grader proposed."""
    threshold = float(cfg().get("min_relevance", 0.0))
    out = []
    for c in sub.criteria:
        if c.kind in CODE_KINDS:
            ok, why = check_code(c, items, threshold)
            out.append({"kind": c.kind, "value": c.value, "met": ok, "why": why})
    statements = [c for c in sub.criteria if c.kind == "covers"]
    if not statements:
        return out, 0, []
    try:
        r: CoverCheck = get_registry().structured("grader", CoverCheck, prompts.cover_messages(
            sub.goal, items, [c.value for c in statements], queries))
    except AgentError as e:   # as `grade` does: a failed check counts as enough
        return out + [{"kind": "covers", "value": c.value, "met": True, "why": f"unchecked: {str(e)[:80]}"}
                      for c in statements], 1, []
    for n, c in enumerate(statements):
        ok = n not in r.unsupported or n in r.supported
        out.append({"kind": "covers", "value": c.value, "met": ok, "why": "" if ok else "the items do not support it"})
    return out, 1, [q for q in r.queries if q.strip() and q.strip() not in queries][:3]


def run_subquery(payload: dict) -> dict:
    c = cfg()
    sub = SubQuery(**payload["sub"])
    principal = Principal(user_id=payload.get("user_id") or "local")
    deadline = float(payload.get("started") or time.time()) + float(c.get("plan_budget_s", 120))
    queries = _fill(sub.search_queries, payload.get("deps") or {}) or [sub.goal]
    result = {"id": sub.id, "goal": sub.goal, "intent": sub.intent, "attempts": 0, "calls": 0}
    items, checks = [], []
    for attempt in range(1 + int(c.get("subquery_retries", 1))):
        emit({"type": "status", "text": f"part {sub.id}: {sub.goal}" + (" (searching again)" if attempt else "")})
        items = _retrieve(sub, queries, principal)
        checks, calls, proposed = check_criteria(sub, items, queries)
        result["attempts"], result["calls"] = attempt + 1, result["calls"] + calls
        if all(x["met"] for x in checks) or attempt >= int(c.get("subquery_retries", 1)) or time.time() > deadline:
            break
        if not proposed:
            try:
                g: NewQueries = get_registry().structured("grader", NewQueries,
                                                          prompts.grade_messages(sub.goal, items, queries))
                proposed = [q for q in g.queries if q.strip() and q.strip() not in queries][:3]
            except AgentError:
                proposed = []
            result["calls"] += 1
        if not proposed:
            break
        queries = list(dict.fromkeys(queries + proposed))
    met = all(x["met"] for x in checks)
    result.update(status="met" if met else "unmet", criteria=checks, items=items, queries=queries)
    return {"sub_results": {sub.id: result}}


# ---- after a wave ----------------------------------------------------------------------------

def _calls_used(state: dict) -> int:
    return int(state.get("plan_calls") or 0) + sum(int(r.get("calls") or 0) for r in (state.get("sub_results") or {}).values())


def check_plan(state: dict) -> dict:
    c = cfg()
    plan, results = Plan(**state["plan"]), state.get("sub_results") or {}
    if runnable(plan, results):
        return {"plan_next": "run"}
    unmet = [s.id for s in plan.subqueries if (results.get(s.id) or {}).get("status") == "unmet"]
    over_time = time.time() - float(state.get("plan_started") or time.time()) > float(c.get("plan_budget_s", 120))
    over_calls = _calls_used(state) >= int(c.get("max_plan_calls", 10))
    if unmet and int(state.get("replans") or 0) < int(c.get("max_replans", 1)) and not (over_time or over_calls):
        return {"plan_next": "replan"}
    why = "all met" if not unmet else "budget spent" if (over_time or over_calls) else "replans used up"
    return {"plan_next": "done", "trace": {"plan_done": why}}


def after_plan(state: dict):
    if not state.get("plan"):
        return "browse" if state.get("intent") == "topic" else "retrieve"
    return sends(state) or "synthesize"


def after_check(state: dict):
    nxt = state.get("plan_next")
    if nxt == "run":
        return sends(state) or "synthesize"
    return "replan" if nxt == "replan" else "synthesize"


def after_replan(state: dict):
    if state.get("plan_next") == "done":
        return "synthesize"
    return sends(state) or "synthesize"


def _same(a: SubQuery, b: SubQuery) -> bool:
    keys = ("goal", "intent", "search_queries", "filters", "criteria", "no_source")
    return all(getattr(a, k) == getattr(b, k) for k in keys)


def replan(state: dict) -> dict:
    plan, results = Plan(**state["plan"]), state.get("sub_results") or {}
    unmet = [s for s in plan.subqueries if (results.get(s.id) or {}).get("status") == "unmet"]
    for s in unmet:
        why = "; ".join(x["why"] for x in results[s.id].get("criteria", []) if not x["met"])[:120]
        emit({"type": "status", "text": f"part {s.id} not met ({why}): planning again"})
    question = state.get("standalone") or state["question"]
    used = {"plan_calls": int(state.get("plan_calls") or 0) + 1, "replans": int(state.get("replans") or 0) + 1}
    calls: list = []
    try:
        new: Plan = get_registry().structured("planner", Plan, prompts.replan_messages(
            question, state.get("language", "en"), plan.model_dump(), _outcomes(plan, results),
            int(cfg().get("max_subqueries", 4))), trace=calls)
    except AgentError as e:
        return {**used, "plan_next": "done", "trace": {f"replan_{used['replans']}": f"failed: {str(e)[:200]}"}}
    why = validate(new)
    if why:
        return {**used, "plan_next": "done", "trace": {f"replan_{used['replans']}": f"rejected: {why}"}}
    _answers_not_lists(new, state.get("intent"))
    old = {s.id: s for s in plan.subqueries}
    changed = [s.id for s in new.subqueries if s.id not in old or not _same(s, old[s.id])]
    dropped = [s.id for s in unmet if s.id not in {x.id for x in new.subqueries}]
    if not changed and not dropped:          # the same plan again: stop instead of repeating it
        return {**used, "plan_next": "done", "trace": {f"replan_{used['replans']}": "unchanged"}}
    # Whatever the planner did not change keeps its result (met or not); only changed subqueries run again.
    kept = {s.id: results[s.id] for s in new.subqueries if s.id in results and s.id not in changed}
    return {**used, "plan": new.model_dump(), "plan_next": None,
            "sub_results": {"_replace": True, **kept, **_no_source_results(new)},
            "trace": {f"replan_{used['replans']}": {"plan": new.model_dump(), "changed": changed,
                                                    "outcomes": _outcomes(plan, results)}}}


def _outcomes(plan: Plan, results: dict) -> list[dict]:
    out = []
    for s in plan.subqueries:
        r = results.get(s.id) or {}
        out.append({"id": s.id, "goal": s.goal, "status": r.get("status", "not run"), "queries": r.get("queries", []),
                    "criteria": r.get("criteria", []),
                    "found": [truncate(i["title"], 100) for i in (r.get("items") or [])[:5]]})
    return out


# ---- the answer ------------------------------------------------------------------------------

def plan_context(plan: Plan, results: dict, limit: int) -> list[dict]:
    """Items of all subqueries, deduplicated, taken in turns so that no part crowds out the others."""
    queues = [list((results.get(s.id) or {}).get("items") or []) for s in plan.subqueries]
    ids = [s.id for s in plan.subqueries]
    out, seen = [], set()
    while any(queues) and len(out) < limit:
        for sid, q in zip(ids, queues):
            while q:
                it = q.pop(0)
                if it["label"] not in seen:
                    seen.add(it["label"])
                    out.append({**it, "subquery": sid})
                    break
            if len(out) >= limit:
                break
    return out


def synthesize(state: dict) -> dict:
    plan, results = Plan(**state["plan"]), state.get("sub_results") or {}
    items = plan_context(plan, results, int(cfg().get("context_max", 12)))
    lang = state.get("language", "en")
    question = state.get("standalone") or state["question"]
    lists = all(s.intent == "topic" for s in plan.subqueries)
    summary = {s.id: {"status": (results.get(s.id) or {}).get("status", "not run"),
                      "attempts": (results.get(s.id) or {}).get("attempts", 0),
                      "queries": (results.get(s.id) or {}).get("queries", []),
                      "criteria": (results.get(s.id) or {}).get("criteria", []),
                      "items": [i["label"] for i in (results.get(s.id) or {}).get("items") or []]}
               for s in plan.subqueries}
    trace = {"plan_result": summary, "replans": int(state.get("replans") or 0), "context": [i["label"] for i in items],
             "plan_calls": _calls_used(state)}
    text, calls, err = stream_role("composer" if lists else "generator",
                                   prompts.planned_messages(question, plan.model_dump(), summary, items, lang, lists))
    if text:
        return {"answer": text, "context": items, "trace": {**trace, "generate_calls": calls}}
    head = ("生成模型暂时不可用。以下是按子问题整理的检索结果：" if lang == "zh"
            else "The answer model is unavailable. These are the items found for each part of the question:")
    parts = []
    for s in plan.subqueries:
        found = [i for i in items if i.get("subquery") == s.id]
        body = "\n".join(f"- [{i['label']}] {truncate(i['title'], 140)}" for i in found) or "-"
        parts.append(f"**{s.goal}**\n{body}")
    answer = head + "\n\n" + "\n\n".join(parts)
    emit({"type": "token", "text": answer})
    return {"answer": answer, "context": items, "degraded": True,
            "trace": {**trace, "generate_calls": calls, "generate_error": err}}
