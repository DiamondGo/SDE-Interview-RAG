"""Query graph: answer an interview-prep question with cited sources.

auth -> guard -> understand -> topic: browse -> compose
                            -> solve/experience: retrieve -> rerank -> grade -(not enough, at most twice)-> rewrite
                               -> retrieve ...; enough -> generate
      -> verify -> log (a solve answer is queued for the reviewer)

Streaming: tokens and progress are emitted as custom stream events
  {"type": "token", "text": ...}   append to the answer being shown
  {"type": "reset"}               the model failed mid-answer: discard what was shown so far
  {"type": "status", "text": ...} progress ("searching", "model not responding, switching to ...")
Consume them with stream_mode="custom" (in-process or through the LangGraph SDK).
"""

import json
import operator
import re
import time
from dataclasses import dataclass
from typing import Annotated, Literal

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel
from typing_extensions import TypedDict

from sde_rag import db, prompts, retrieval, review
from sde_rag.agents import AgentError, get_registry
from sde_rag.auth import Principal, auth_provider, safe_can_read
from sde_rag.index.reranker import get_reranker
from sde_rag.parse.markdown_qa import is_question
from sde_rag.settings import section
from sde_rag.taxonomy import taxonomy
from sde_rag.text import detect_lang, truncate


@dataclass
class QueryContext:
    user_id: str = "local"


class QueryInput(TypedDict, total=False):
    question: str
    filters: dict          # optional: category, tags (list), company


class QueryOutput(TypedDict, total=False):
    answer: str
    citations: list[dict]
    intent: str
    degraded: bool
    review_job_id: str | None


def _merge_trace(old: dict | None, new: dict | None) -> dict:
    if not new:
        return old or {}
    if new.get("_reset"):
        return {k: v for k, v in new.items() if k != "_reset"}
    return {**(old or {}), **new}


class QueryState(TypedDict, total=False):
    question: str
    filters: dict
    history: Annotated[list[dict], operator.add]
    user_id: str
    started: float
    intent: str
    language: str
    standalone: str
    search_queries: list[str]
    category: str | None
    tags: list[str]
    context: list[dict]
    rounds: int                 # retrieval rounds after the first (rewrites)
    grade: dict
    answer: str
    citations: list[dict]
    degraded: bool
    review_job_id: str | None
    trace: Annotated[dict, _merge_trace]


class Understanding(BaseModel):
    intent: Literal["topic", "solve", "experience", "out_of_scope"]
    language: Literal["zh", "en"] = "en"
    category: str | None = None
    tags: list[str] = []
    company: str | None = None
    standalone_question: str = ""
    search_queries: list[str] = []


def _emit(event: dict) -> None:
    try:
        get_stream_writer()(event)
    except Exception:  # not streaming (plain invoke): nothing to emit to
        pass


def _principal(state: QueryState) -> Principal:
    return Principal(user_id=state.get("user_id") or "local")


# ---- nodes -------------------------------------------------------------------------------

def auth(state: QueryState, runtime: Runtime[QueryContext]) -> dict:
    p = auth_provider().authenticate(runtime.context)
    # A thread keeps state across turns: clear everything the previous turn produced.
    return {"user_id": p.user_id, "started": time.time(), "intent": "", "degraded": False, "answer": "",
            "citations": [], "context": [], "rounds": 0, "grade": {}, "review_job_id": None,
            "trace": {"_reset": True, "user": p.scope_key()}}


_INJECTION = re.compile(r"ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts)|system prompt", re.I)


def guard(state: QueryState) -> dict:
    q = (state.get("question") or "").strip()
    limit = int(section("query").get("max_question_chars", 2000))
    if not q:
        return {"intent": "invalid", "answer": "请输入一个问题。/ Please enter a question."}
    if len(q) > limit:
        return {"intent": "invalid", "answer": f"问题太长了（上限 {limit} 字符）。/ The question is too long."}
    return {"question": q, "trace": {"injection_suspected": bool(_INJECTION.search(q))}}


_TOPIC_RULE = re.compile(r"常见|常问|会问|问什么|考什么|考察|面试题|有哪些|哪些问题|题目|准备|复习|学什么|"
                         r"common|frequent|typical|top \d+|list of|"
                         r"questions (for|about|on)|prepare for|study|interviews\b", re.I)


def _one_question(q: str) -> bool:
    q = q.strip()
    return is_question(q) or (len(q) >= 8 and q.endswith(("?", "？")))
_EXPERIENCE_RULE = re.compile(r"面经|面试经历|面试流程|几轮|interview (experience|process|loop)|onsite at|interviewing at", re.I)


def rule_understanding(question: str) -> Understanding:
    tax = taxonomy()
    tags = [t for t, _ in tax.match(question).most_common(4)]
    intent = "experience" if _EXPERIENCE_RULE.search(question) else "topic" if _TOPIC_RULE.search(question) else "solve"
    return Understanding(intent=intent, language=detect_lang(question), category=tax.category_of(tags), tags=tags,
                         standalone_question=question, search_queries=[question])


def understand(state: QueryState) -> dict:
    q = state["question"]
    calls: list = []
    try:
        u = get_registry().structured("router", Understanding,
                                      prompts.router_messages(q, state.get("history") or []), trace=calls)
        source = "router"
    except AgentError as e:
        u = rule_understanding(q)
        source = f"rules ({str(e)[:120]})"
    corrected = None
    if u.intent == "topic" and _one_question(q) and not _TOPIC_RULE.search(q):
        # One specific question asks for an answer, not a list (small routers mix the two up).
        u.intent, corrected = "solve", "topic->solve"
    tax = taxonomy()
    filters = state.get("filters") or {}
    category = filters.get("category") or (u.category if u.category in tax.categories else None)
    tags = [t for t in (tax.resolve(x) for x in (filters.get("tags") or u.tags)) if t]
    queries = [s for s in u.search_queries if s.strip()][:3] or [u.standalone_question or q]
    if u.standalone_question and u.standalone_question not in queries:
        queries.append(u.standalone_question)
    lang = u.language if u.language in ("zh", "en") else detect_lang(q)
    return {"intent": u.intent, "language": lang, "standalone": u.standalone_question or q,
            "search_queries": queries, "category": category, "tags": tags,
            "trace": {"understanding": source, "calls": calls, "intent": u.intent, "queries": queries,
                      "category": category, "tags": tags, "intent_corrected": corrected}}


def route(state: QueryState) -> str:
    intent = state.get("intent")
    if intent == "invalid":
        return "log"
    if intent == "out_of_scope":
        return "out_of_scope"
    return "browse" if intent == "topic" else "retrieve"


def browse(state: QueryState) -> dict:
    _emit({"type": "status", "text": "searching the question bank"})
    trace: dict = {}
    items = retrieval.topic_questions(
        _principal(state), state["search_queries"], category=state.get("category"), tags=state.get("tags") or None,
        k=int(section("query").get("topic_k", 30)), trace=trace)
    if not items and (state.get("category") or state.get("tags")):
        # The model's guess at a category or tag can be too narrow: retry with the words alone.
        items = retrieval.topic_questions(_principal(state), state["search_queries"], category=None, tags=None,
                                          k=int(section("query").get("topic_k", 30)), trace=trace)
    return {"context": items, "trace": {"retrieval": trace, "context": [i["label"] for i in items]}}


def retrieve(state: QueryState) -> dict:
    _emit({"type": "status", "text": "searching sources" if not state.get("rounds") else "searching again"})
    cfg = section("query")
    trace: dict = {}
    kinds = ["doc"] if state["intent"] == "experience" else None
    filters = state.get("filters") or {}
    hits = retrieval.hybrid(_principal(state), state["search_queries"], kinds=kinds,
                            category=filters.get("category"), tags=filters.get("tags"), trace=trace)
    items = retrieval.load_context(_principal(state), hits, int(cfg.get("rerank_candidates", 30)), trace=trace)
    if trace.get("degraded"):
        _emit({"type": "status", "text": "vector search unavailable, using keyword search"})
    return {"context": items, "degraded": bool(state.get("degraded")) or bool(trace.get("degraded")),
            "trace": {f"retrieval_{state.get('rounds', 0)}": trace}}


def rerank(state: QueryState) -> dict:
    """A cross-encoder orders the candidates; source answers the reviewer judged wrong move down."""
    cfg = section("query")
    items, k = state.get("context") or [], int(cfg.get("context_k", 6))
    question = state.get("standalone") or state["question"]
    how = "retrieval order"
    try:
        r = get_reranker()
        if r is not None and items:
            _emit({"type": "status", "text": "ranking what was found"})
            scores = r.score(question, [f"{i['title']}\n{i['text']}"[:2000] for i in items])
            penalty = float(cfg.get("incorrect_answer_penalty", 3.0))
            adjusted = [float(s) - (penalty if i.get("answer_verdict") == "incorrect" else 0.0)
                        for s, i in zip(scores, items)]
            order = sorted(range(len(items)), key=lambda n: -adjusted[n])
            items = [{**items[n], "rerank_score": round(adjusted[n], 3)} for n in order]
            how = r.name
    except Exception as e:  # keep the retrieval order
        how = f"retrieval order (reranker failed: {type(e).__name__}: {str(e)[:120]})"
    # One long article can fill every slot (each of its paragraphs matches): keep room for other sources.
    per_doc, kept, taken = int(cfg.get("max_per_document", 3)), [], {}
    for i in items:
        if i["kind"] == "doc":
            if taken.get(i["doc_id"], 0) >= per_doc:
                continue
            taken[i["doc_id"]] = taken.get(i["doc_id"], 0) + 1
        kept.append(i)
        if len(kept) >= k:
            break
    return {"context": kept, "trace": {f"rerank_{state.get('rounds', 0)}": how,
                                       "context": [i["label"] for i in kept]}}


class Grade(BaseModel):
    sufficient: bool = True
    same_question: bool = False
    missing: str = ""
    queries: list[str] = []


def grade(state: QueryState) -> dict:
    """Does the context cover the question? When not, the grader proposes new search queries."""
    if state.get("rounds", 0) >= int(section("query").get("max_rewrites", 2)):
        return {"grade": {"sufficient": True, "why": "rewrite limit reached"}}
    _emit({"type": "status", "text": "checking whether the sources cover the question"})
    question = state.get("standalone") or state["question"]
    calls: list = []
    try:
        g: Grade = get_registry().structured("grader", Grade, prompts.grade_messages(
            question, state.get("context") or [], state["search_queries"]), trace=calls)
    except AgentError as e:  # treat as enough: answering beats a loop on a broken model
        return {"grade": {"sufficient": True, "why": f"grader unavailable: {str(e)[:120]}"},
                "trace": {f"grade_{state.get('rounds', 0)}": "unavailable"}}
    new = [q.strip() for q in g.queries if q.strip() and q.strip() not in state["search_queries"]][:3]
    out = g.model_dump() | {"queries": new}
    if not g.sufficient and not new:
        out["sufficient"] = True          # nothing new to search for
    return {"grade": out, "trace": {f"grade_{state.get('rounds', 0)}": out, "grade_calls": calls}}


def after_grade(state: QueryState) -> str:
    return "generate" if (state.get("grade") or {}).get("sufficient", True) else "rewrite"


def rewrite(state: QueryState) -> dict:
    new = state["grade"]["queries"]
    _emit({"type": "status", "text": f"searching again: {'; '.join(new)}"})
    return {"search_queries": list(dict.fromkeys(state["search_queries"] + new)), "rounds": state.get("rounds", 0) + 1}


def _stream_role(role: str, messages) -> tuple[str | None, list, str | None]:
    """Stream a role's answer as custom events. Returns (text or None when every backend failed, calls, error)."""
    calls: list = []
    parts: list[str] = []
    try:
        for ev in get_registry().stream(role, messages, trace=calls):
            if ev["type"] == "token":
                parts.append(ev["text"])
                _emit({"type": "token", "text": ev["text"]})
            elif ev["type"] == "reset":
                parts = []
                _emit({"type": "reset"})
                _emit({"type": "status", "text": "model stopped responding, switching to a fallback"})
            elif ev["type"] == "fallback":
                _emit({"type": "status", "text": "model unavailable, switching to a fallback"})
            elif ev["type"] == "attempt":
                _emit({"type": "status", "text": f"writing the answer ({ev['backend']})"})
        return "".join(parts), calls, None
    except AgentError as e:
        if parts:
            _emit({"type": "reset"})
        return None, calls, str(e)


def _sources_list(items: list[dict], lang: str) -> str:
    lines = [f"- [{i['label']}] {truncate(i['title'], 120)}" + (f" — {i['url']}" if i.get("url") else "")
             for i in items]
    return "\n".join(lines)


def compose(state: QueryState) -> dict:
    items, lang = state.get("context") or [], state.get("language", "en")
    if not items:
        msg = ("题库里没有找到相关的题目。可以换个关键词，或者直接问一道具体的题。" if lang == "zh"
               else "No matching questions in the bank yet. Try other keywords or ask a specific question.")
        return {"answer": msg}
    text, calls, err = _stream_role("composer", prompts.topic_messages(state["question"], items, lang))
    if text:
        return {"answer": text, "trace": {"compose_calls": calls}}
    head = "模型暂时不可用，以下是按出现频率排序的相关题目：" if lang == "zh" else \
        "The model is unavailable; here are the matching questions, most frequent first:"
    rows = [f"| {n} | {truncate(i['title'], 140)} [{i['label']}] | {i.get('mention_count', 0)} | "
            f"{', '.join(i.get('tags') or [])} |" for n, i in enumerate(items, 1)]
    table = "| # | 题目 / Question | 出现次数 / Seen | 标签 / Tags |\n|---|---|---|---|\n" + "\n".join(rows)
    answer = f"{head}\n\n{table}"
    _emit({"type": "token", "text": answer})
    return {"answer": answer, "degraded": True, "trace": {"compose_calls": calls, "compose_error": err}}


def generate(state: QueryState) -> dict:
    items, lang = state.get("context") or [], state.get("language", "en")
    question = state.get("standalone") or state["question"]
    build = prompts.experience_messages if state["intent"] == "experience" else prompts.solve_messages
    text, calls, err = _stream_role("generator", build(question, items, lang))
    if text:
        return {"answer": text, "trace": {"generate_calls": calls}}
    head = ("生成模型暂时不可用。以下是检索到的相关题目和来源，可以先参考：" if lang == "zh"
            else "The answer model is unavailable. These are the most relevant items found:")
    body = "\n\n".join(f"**[{i['label']}] {truncate(i['title'], 160)}**\n\n{truncate(i['text'], 800)}" for i in items)
    answer = f"{head}\n\n{body}" if items else head
    _emit({"type": "token", "text": answer})
    return {"answer": answer, "degraded": True, "trace": {"generate_calls": calls, "generate_error": err}}


def out_of_scope(state: QueryState) -> dict:
    zh = state.get("language") == "zh"
    return {"answer": "这个问题和软件工程师面试准备无关，我只能回答面试相关的问题。" if zh else
            "That is outside interview preparation for software engineers, which is all I can help with."}


_LABEL = re.compile(r"\[((?:Q\d+)|(?:D\d+#\d+))\]")
_LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|\|)\s")


def verify(state: QueryState) -> dict:
    """Keep only citations that point at context items the user may read; drop the rest from the text.
    A list item or table row citing nothing but invalid labels is an invented entry and is dropped whole."""
    items = {i["label"]: i for i in state.get("context") or []}
    allowed = dict(zip(items, safe_can_read(_principal(state), list(items.values()))))
    answer = state.get("answer") or ""
    invalid = sorted({m for m in _LABEL.findall(answer) if not allowed.get(m)})
    if invalid:
        kept, dropped = [], 0
        for line in answer.split("\n"):
            labels = _LABEL.findall(line)
            if labels and _LIST_LINE.match(line) and not any(allowed.get(m) for m in labels):
                dropped += 1
                continue
            for label in labels:
                if not allowed.get(label):
                    line = re.sub(rf"[ \t]*\[{re.escape(label)}\]", "", line)
            kept.append(line)
        answer = "\n".join(kept)
        if dropped:
            _emit({"type": "status", "text": f"removed {dropped} uncited item(s)"})
    order: list[str] = []
    for m in _LABEL.findall(answer):
        if m not in order:
            order.append(m)
    citations = [{"label": l, "title": items[l]["title"], "url": items[l].get("url", ""),
                  "source": items[l].get("source", ""), "kind": items[l]["kind"]} for l in order]
    return {"answer": answer, "citations": citations, "trace": {"invalid_citations": invalid}}


def log(state: QueryState, runtime: Runtime[QueryContext]) -> dict:
    latency = int((time.time() - state.get("started", time.time())) * 1000)
    trace = {**(state.get("trace") or {}), "latency_ms": latency}
    job_id = None
    try:
        log_id = db.conn().execute(
            "INSERT INTO query_logs (user_id, created_at, question, intent, filters, answer, citations, degraded, "
            "latency_ms, trace) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (state.get("user_id") or "local", db.now_iso(), state.get("question", ""), state.get("intent"),
             json.dumps(state.get("filters") or {}, ensure_ascii=False), state.get("answer", ""),
             json.dumps(state.get("citations") or [], ensure_ascii=False), int(bool(state.get("degraded"))),
             latency, json.dumps(trace, ensure_ascii=False, default=str))).lastrowid
        job_id = _queue_review(state, log_id)
        if job_id:
            db.conn().execute("UPDATE query_logs SET review_job_id = ? WHERE id = ?", (job_id, log_id))
    except Exception:  # logging must never break the answer
        pass
    turn = []
    if state.get("intent") != "invalid":
        turn = [{"role": "user", "content": state.get("question", "")},
                {"role": "assistant", "content": truncate(state.get("answer", ""), 1500)}]
    return {"history": turn, "review_job_id": str(job_id) if job_id else None, "trace": {"latency_ms": latency}}


def _queue_review(state: QueryState, log_id: int) -> int | None:
    """A `solve` answer the generator wrote goes to the reviewer, who checks it after it was returned."""
    if state.get("intent") != "solve" or not state.get("answer") or "generate_error" in (state.get("trace") or {}):
        return None
    if not section("review").get("review_solve_answers", True) or not review.enabled():
        return None
    return review.enqueue("query_answer", log_id, state.get("standalone") or state["question"], state["answer"],
                          priority=2, context=prompts.render_context(state.get("context") or []))


def build_graph() -> StateGraph:
    b = StateGraph(QueryState, context_schema=QueryContext, input_schema=QueryInput, output_schema=QueryOutput)
    for name, fn in (("auth", auth), ("guard", guard), ("understand", understand), ("browse", browse),
                     ("retrieve", retrieve), ("rerank", rerank), ("grade", grade), ("rewrite", rewrite),
                     ("compose", compose), ("generate", generate), ("out_of_scope", out_of_scope),
                     ("verify", verify), ("log", log)):
        b.add_node(name, fn)
    b.add_edge(START, "auth")
    b.add_edge("auth", "guard")
    b.add_conditional_edges("guard", lambda s: "log" if s.get("intent") == "invalid" else "understand",
                            ["log", "understand"])
    b.add_conditional_edges("understand", route, ["browse", "retrieve", "out_of_scope", "log"])
    b.add_edge("browse", "compose")
    b.add_edge("retrieve", "rerank")
    b.add_edge("rerank", "grade")
    b.add_conditional_edges("grade", after_grade, ["generate", "rewrite"])
    b.add_edge("rewrite", "retrieve")
    b.add_edge("compose", "verify")
    b.add_edge("generate", "verify")
    b.add_edge("out_of_scope", "log")
    b.add_edge("verify", "log")
    b.add_edge("log", END)
    return b


graph = build_graph().compile()
