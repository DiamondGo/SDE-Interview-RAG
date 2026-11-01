"""Agent capability checks (DESIGN 8): with this machine's configuration, can each LLM role still do its job?

The golden sets are evals/agent/<role>.jsonl; thresholds come from config/agent_thresholds.toml (relaxed per
machine in agent_thresholds.local.toml). The checks go through the same prompts, schemas and parsing as the
pipeline, so they measure what runs. A result is stored with the role's configuration fingerprint: a changed
model or parameter has no passing record until it is checked again, and the gates treat that as below threshold.

  quick: the first `quick` cases of a role (minutes; the ingest gate and the UI run this)
  full:  every case (`sde-rag agent-check --level full`, `pytest -m agent --agent-level full`)
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from sde_rag import db, notify
from sde_rag.agents import AgentError, get_registry
from sde_rag.settings import ROOT, load_layered

log = logging.getLogger("sde_rag.agent_checks")
EVALS = ROOT / "evals" / "agent"
ROLES = ("extractor", "tagger", "router", "grader", "planner", "generator", "reviewer", "e2e")
E2E_ROLES = ("router", "planner", "grader", "generator", "composer")


# ---- cases and thresholds ------------------------------------------------------------------

def load_cases(role: str, level: str = "full") -> list[dict]:
    path = EVALS / f"{role}.jsonl"
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip() and not line.startswith("//")]
    if level == "quick":
        cases = cases[:int(thresholds(role).get("quick", 5))]
    return cases


def _threshold_layers() -> tuple[dict, dict]:
    base_path = ROOT / "config" / "agent_thresholds.toml"
    import tomllib
    base = tomllib.loads(base_path.read_text()) if base_path.is_file() else {}
    return base, load_layered("agent_thresholds")


def thresholds(role: str) -> dict:
    return _threshold_layers()[1].get(role, {})


def gate_config() -> dict:
    return _threshold_layers()[1].get("gate", {})


def relaxed(role: str) -> dict[str, tuple]:
    """Metrics whose threshold this machine changed: metric -> (committed, local)."""
    base, merged = (layer.get(role, {}) for layer in _threshold_layers())
    return {k: (base.get(k), v) for k, v in merged.items() if isinstance(v, dict) and base.get(k) != v}


def compare(role: str, metrics: dict) -> list[dict]:
    out, loosened = [], relaxed(role)
    for metric, rule in thresholds(role).items():
        if not isinstance(rule, dict) or metric not in metrics:
            continue
        value = metrics[metric]
        if "min" in rule:
            op, limit, ok = ">=", rule["min"], value >= rule["min"]
        else:
            op, limit, ok = "<=", rule["max"], value <= rule["max"]
        out.append({"metric": metric, "value": value, "op": op, "threshold": limit, "ok": bool(ok),
                    "relaxed": metric in loosened})
    return out


# ---- measuring -----------------------------------------------------------------------------

class Calls:
    """The registry's trace of every model call made by a check: latency and stalls."""

    def __init__(self):
        self.trace: list[dict] = []

    def p95(self) -> float:
        ms = sorted(t["ms"] for t in self.trace if "ms" in t)
        return round(ms[min(len(ms) - 1, int(0.95 * len(ms)))] / 1000, 1) if ms else 0.0

    def stalls(self) -> int:
        return sum(1 for t in self.trace if t.get("stalled"))


def kw_match(keywords: list[str], text: str) -> bool:
    """Every keyword present; "a|b" accepts either."""
    low = text.lower()
    return all(any(alt.strip().lower() in low for alt in kw.split("|")) for kw in keywords)


def _ratio(a: float, b: float, empty: float = 1.0) -> float:
    return round(a / b, 3) if b else empty


# ---- one scorer per role -------------------------------------------------------------------

def check_extractor(cases: list[dict], calls: Calls) -> tuple[dict, list]:
    from sde_rag.pipeline.extraction import _call
    from sde_rag.pipeline.indexing import split_paragraphs
    from sde_rag.text import quote_similarity
    reg, fails = get_registry(), []
    invalid = right = proposed = unquoted = expected = found = 0
    for c in cases:
        try:
            results = _call(reg, c["title"], split_paragraphs(c["text"], 6000)[:8], 1500, lambda: None,
                            trace=calls.trace)
        except AgentError as e:
            invalid += 1
            fails.append({"id": c["id"], "error": str(e)[:200]})
            continue
        interview = any(x.is_interview_content for _, x in results)
        right += interview == c["is_interview"]
        qs = [q for _, x in results for q in x.questions] if interview else []
        proposed += len(qs)
        bad = [q.original_text for q in qs if quote_similarity(q.original_text, c["text"]) < 0.8]
        unquoted += len(bad)
        missing = [e["keywords"] for e in c.get("questions", []) if not any(kw_match(e["keywords"], q.text) for q in qs)]
        expected += len(c.get("questions", []))
        found += len(c.get("questions", [])) - len(missing)
        if missing or bad or interview != c["is_interview"]:
            fails.append({"id": c["id"], "interview": interview, "missing": missing, "unquoted": bad[:3],
                          "extracted": [q.text for q in qs][:12]})
    n = len(cases)
    return {"valid_output": _ratio(n - invalid, n), "recall": _ratio(found, expected),
            "fabrication": _ratio(unquoted, proposed, 0.0), "interview_accuracy": _ratio(right, n - invalid, 0.0)}, fails


def check_tagger(cases: list[dict], calls: Calls) -> tuple[dict, list]:
    from sde_rag.pipeline.tagging import TagBatch, tag_prompt
    from sde_rag.taxonomy import taxonomy
    reg, tax, fails = get_registry(), taxonomy(), []
    right = tp = fp = fn = proposed = outside = 0
    for i in range(0, len(cases), 20):
        batch = cases[i:i + 20]
        try:
            res: TagBatch = reg.structured("tagger", TagBatch, tag_prompt(
                [{"text": c["question"], "section_path": c.get("section_path", "")} for c in batch]), trace=calls.trace)
            by_idx = {it.idx: it for it in res.items}
        except AgentError as e:
            by_idx = {}
            fails.append({"batch": i, "error": str(e)[:200]})
        for n, c in enumerate(batch):
            it = by_idx.get(n)
            want = {t for t in (tax.resolve(x) for x in c["tags"]) if t}
            got_raw = list(it.tags[:5]) if it else []
            got = {t for t in (tax.resolve(x) for x in got_raw) if t}
            proposed += len(got_raw)
            outside += sum(1 for x in got_raw if not tax.resolve(x))
            ok = it is not None and it.category == c["category"]
            right += ok
            tp, fp, fn = tp + len(got & want), fp + len(got - want), fn + len(want - got)
            if not ok or got != want:
                fails.append({"question": c["question"][:100], "want": [c["category"], sorted(want)],
                              "got": [it.category if it else None, sorted(got)]})
    prec, rec = _ratio(tp, tp + fp, 0.0), _ratio(tp, tp + fn, 0.0)
    f1 = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0
    return {"category_accuracy": _ratio(right, len(cases)), "tag_f1": f1,
            "outside_taxonomy": _ratio(outside, proposed, 0.0)}, fails


def _understand(question: str, calls: Calls):
    """The router's answer as the pipeline uses it (with its topic->solve correction), or None on failure."""
    from sde_rag import prompts
    from sde_rag.graphs.query import _TOPIC_RULE, Understanding, _one_question
    try:
        u = get_registry().structured("router", Understanding, prompts.router_messages(question, []), trace=calls.trace)
    except AgentError:
        return None
    if u.intent == "topic" and _one_question(question) and not _TOPIC_RULE.search(question):
        u.intent = "solve"
    from sde_rag.graphs.query import resolve_filters
    u.category, u.tags = resolve_filters(u, {})
    return u


def check_router(cases: list[dict], calls: Calls) -> tuple[dict, list]:
    from sde_rag.taxonomy import taxonomy
    tax, fails = taxonomy(), []
    right = tp = fp = fn = terms_ok = terms_n = 0
    for c in cases:
        u = _understand(c["question"], calls)
        if u is None:
            fails.append({"question": c["question"], "error": "router failed"})
            fn += int(bool(c.get("category"))) + len(c.get("tags", []))
            terms_n += bool(c.get("key_terms"))
            continue
        right += u.intent == c["intent"]
        want = ({("category", c["category"])} if c.get("category") else set()) | \
            {("tag", t) for t in (tax.resolve(x) for x in c.get("tags", [])) if t}
        if want:
            got = ({("category", u.category)} if c.get("category") and u.category in tax.categories else set()) | \
                {("tag", t) for t in (tax.resolve(x) for x in u.tags) if t and any(w[0] == "tag" for w in want)}
            tp, fp, fn = tp + len(got & want), fp + len(got - want), fn + len(want - got)
        ok_terms = True
        if c.get("key_terms"):
            terms_n += 1
            ok_terms = kw_match(c["key_terms"], " ".join(u.search_queries))
            terms_ok += ok_terms
        if u.intent != c["intent"] or not ok_terms:
            fails.append({"question": c["question"], "want": c["intent"], "got": u.intent,
                          "queries": u.search_queries, "key_terms": c.get("key_terms")})
    prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0
    return {"intent_accuracy": _ratio(right, len(cases)), "filter_f1": f1, "key_terms": _ratio(terms_ok, terms_n)}, fails


def check_grader(cases: list[dict], calls: Calls) -> tuple[dict, list]:
    from sde_rag import prompts
    from sde_rag.graphs.planner import CoverCheck
    from sde_rag.graphs.query import Grade
    reg, fails = get_registry(), []
    suff = [c for c in cases if c["kind"] == "sufficiency"]
    crit = [c for c in cases if c["kind"] == "criterion"]
    s_right = c_right = 0
    for c in suff:
        try:
            g: Grade = reg.structured("grader", Grade, prompts.grade_messages(c["question"], c["items"], c["queries"]),
                                      trace=calls.trace)
            ok = g.sufficient == c["sufficient"]
        except AgentError as e:
            ok, g = False, None
        s_right += ok
        if not ok:
            fails.append({"question": c["question"], "want": c["sufficient"], "got": g.sufficient if g else "error"})
    for c in crit:
        try:
            r: CoverCheck = reg.structured("grader", CoverCheck, prompts.cover_messages(
                c["goal"], c["items"], [c["statement"]], [c["goal"]]), trace=calls.trace)
            supported = 0 not in r.unsupported or 0 in r.supported
            ok = supported == c["supported"]
        except AgentError:
            ok, supported = False, "error"
        c_right += ok
        if not ok:
            fails.append({"statement": c["statement"], "want": c["supported"], "got": supported})
    return {"sufficiency_accuracy": _ratio(s_right, len(suff)), "criterion_accuracy": _ratio(c_right, len(crit))}, fails


def check_planner(cases: list[dict], calls: Calls) -> tuple[dict, list]:
    from sde_rag import prompts
    from sde_rag.graphs import planner as P
    reg, fails = get_registry(), []
    comp = [c for c in cases if c["kind"] == "complexity"]
    repl = [c for c in cases if c["kind"] == "replan"]
    c_right = simple_n = simple_ok = valid = planned = aspects_n = aspects_ok = changed_ok = 0
    limit = int(P.cfg().get("max_subqueries", 4))
    for c in comp:
        u = _understand(c["question"], calls)
        complexity = (u.complexity if u else P.rule_complexity(c["question"])[0])
        if complexity == "simple" and P.several_questions(c["question"]):
            complexity = "complex"
        want = "complex" if c["complex"] else "simple"
        c_right += complexity == want
        if not c["complex"]:
            simple_n += 1
            simple_ok += complexity == "simple"
            if complexity != "simple":
                fails.append({"question": c["question"], "want": "simple", "got": complexity})
            continue
        planned += 1
        try:
            p = reg.structured("planner", P.Plan, prompts.plan_messages(c["question"], c.get("lang", "en"), limit),
                               trace=calls.trace)
            why = P.validate(p)
        except AgentError as e:
            p, why = None, f"error: {str(e)[:120]}"
        if why is None:
            valid += 1
            texts = [f"{s.goal} {' '.join(s.search_queries)}" for s in p.subqueries]
            missing = [a["keywords"] for a in c.get("aspects", []) if not any(kw_match(a["keywords"], t) for t in texts)]
            aspects_n += len(c.get("aspects", []))
            aspects_ok += len(c.get("aspects", [])) - len(missing)
            if missing or complexity != want:
                fails.append({"question": c["question"], "complexity": complexity, "missing_aspects": missing,
                              "subqueries": texts})
        else:
            aspects_n += len(c.get("aspects", []))
            fails.append({"question": c["question"], "complexity": complexity, "plan_rejected": why})
    for c in repl:
        try:
            new = reg.structured("planner", P.Plan, prompts.replan_messages(
                c["question"], c.get("lang", "en"), c["plan"], c["outcomes"], limit), trace=calls.trace)
            why = P.validate(new)
        except AgentError as e:
            new, why = None, f"error: {str(e)[:120]}"
        ok = False
        if why is None:
            old = {s["id"]: P.SubQuery(**s) for s in c["plan"]["subqueries"]}
            now = {s.id: s for s in new.subqueries}
            ok = all(sid not in now or now[sid].no_source or not P._same(now[sid], old[sid]) for sid in c["unmet"])
        changed_ok += ok
        if not ok:
            fails.append({"replan": c["question"], "why": why or "unmet subquery left unchanged",
                          "plan": new.model_dump() if new else None})
    return {"complexity_accuracy": _ratio(c_right, len(comp)), "simple_unplanned": _ratio(simple_ok, simple_n),
            "valid_plans": _ratio(valid, planned), "aspect_coverage": _ratio(aspects_ok, aspects_n),
            "replan_changes_unmet": _ratio(changed_ok, len(repl))}, fails


_LABEL = re.compile(r"\[((?:Q\d+)|(?:D\d+#\d+))\]")


def check_generator(cases: list[dict], calls: Calls) -> tuple[dict, list]:
    from sde_rag import prompts
    from sde_rag.text import detect_lang
    reg, fails = get_registry(), []
    cited = valid = lang_ok = nosrc_n = nosrc_ok = 0
    for c in cases:
        try:
            text = reg.text("generator", prompts.solve_messages(c["question"], c["items"], c["lang"]), trace=calls.trace)
        except AgentError as e:
            fails.append({"question": c["question"], "error": str(e)[:200]})
            nosrc_n += not c["items"]
            continue
        labels = _LABEL.findall(text)
        allowed = {i["label"] for i in c["items"]}
        cited += len(labels)
        valid += sum(1 for x in labels if x in allowed)
        ok_lang = detect_lang(text) == c["lang"]
        lang_ok += ok_lang
        ok_nosrc = True
        if not c["items"]:
            nosrc_n += 1
            ok_nosrc = prompts.SUPPLEMENT_HEADING[c["lang"]] in text and not labels
            nosrc_ok += ok_nosrc
        if not ok_lang or not ok_nosrc or any(x not in allowed for x in labels):
            fails.append({"question": c["question"], "lang_ok": ok_lang, "no_source_ok": ok_nosrc,
                          "invalid_labels": [x for x in labels if x not in allowed], "answer": text[:300]})
    return {"citation_validity": _ratio(valid, cited), "no_source_labeled": _ratio(nosrc_ok, nosrc_n),
            "language": _ratio(lang_ok, len(cases))}, fails


def check_reviewer(cases: list[dict], calls: Calls) -> tuple[dict, list]:
    from sde_rag import review
    reg, fails = get_registry(), []
    invalid = detected = errors = fpos = corrects = pinned = 0
    for c in cases:
        review._count_call()                       # the reviewer's quota pays for its checks too
        try:
            r = reg.structured("reviewer", review.Review, review.review_prompt(c["question"], c["answer"]),
                               trace=calls.trace)
        except AgentError as e:
            invalid += 1
            fails.append({"question": c["question"], "error": str(e)[:200]})
            continue
        issues = " ".join(f"{i.claim} {i.problem} {i.correction}" for i in r.issues)
        if c["has_error"]:
            errors += 1
            hit = r.verdict in ("incorrect", "minor_issues")
            detected += hit
            pin = any(kw_match([k], issues) for k in c["error_keywords"])
            pinned += pin
            if not (hit and pin):
                fails.append({"question": c["question"], "planted": c["error_keywords"], "verdict": r.verdict,
                              "issues": issues[:300]})
        else:
            corrects += 1
            fpos += r.verdict == "incorrect"
            if r.verdict == "incorrect":
                fails.append({"question": c["question"], "false_positive": issues[:300]})
    return {"valid_output": _ratio(len(cases) - invalid, len(cases)), "error_detection": _ratio(detected, errors),
            "false_positive": _ratio(fpos, corrects, 0.0), "pinpoint": _ratio(pinned, errors)}, fails


def check_e2e(cases: list[dict], calls: Calls) -> tuple[dict, list]:
    from sde_rag.graphs import query
    app, fails, done = query.build_graph().compile(), [], 0
    for c in cases:
        t0 = time.time()
        try:
            out = app.invoke({"question": c["question"]}, {"recursion_limit": 40},
                             context=query.QueryContext(user_id="agent-check", review=False))
            ok = bool(out.get("answer"))
        except Exception as e:  # the smoke test is about exceptions too
            out, ok = {}, False
            fails.append({"question": c["question"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
        row = db.conn().execute("SELECT trace FROM query_logs WHERE user_id = 'agent-check' ORDER BY id DESC LIMIT 1"
                                ).fetchone()
        trace = json.loads(row[0]) if row else {}
        stalled = json.dumps(trace).count('"stalled": true')
        calls.trace.append({"ms": int((time.time() - t0) * 1000), "stalled": bool(stalled)})
        done += ok
        if ok and c.get("intent") and out.get("intent") != c["intent"]:
            fails.append({"question": c["question"], "intent": out.get("intent"), "expected": c["intent"]})
    return {"completed": _ratio(done, len(cases))}, fails


CHECKS: dict[str, Callable[[list[dict], Calls], tuple[dict, list]]] = {
    "extractor": check_extractor, "tagger": check_tagger, "router": check_router, "grader": check_grader,
    "planner": check_planner, "generator": check_generator, "reviewer": check_reviewer, "e2e": check_e2e}


# ---- running, storing, gating --------------------------------------------------------------

def fingerprint(role: str) -> str:
    reg = get_registry()
    if role == "e2e":
        return "+".join(reg.fingerprint(r) for r in E2E_ROLES if r in reg.roles)
    return reg.fingerprint(role)


def run(role: str, level: str = "quick", *, store: bool = True) -> dict:
    if role not in CHECKS:
        raise ValueError(f"unknown role {role!r} (checked roles: {', '.join(CHECKS)})")
    reg = get_registry()
    if role != "e2e" and not reg.usable(role):
        return {"role": role, "level": level, "passed": False, "skipped": f"role {role!r} is not usable",
                "checks": [], "metrics": {}}
    cases = load_cases(role, level)
    calls, t0 = Calls(), time.time()
    metrics, failures = CHECKS[role](cases, calls)
    metrics.update(stalls=calls.stalls(), p95_latency_s=calls.p95())
    checks = compare(role, metrics)
    out = {"role": role, "level": level, "fingerprint": fingerprint(role), "cases": len(cases), "metrics": metrics,
           "checks": checks, "passed": all(c["ok"] for c in checks), "failures": failures[:20],
           "duration_s": round(time.time() - t0, 1)}
    if store:
        db.conn().execute(
            "INSERT INTO agent_checks (role, fingerprint, level, cases, metrics, checks, passed, failures, duration_s, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (role, out["fingerprint"], level, len(cases), json.dumps(metrics), json.dumps(checks), int(out["passed"]),
             json.dumps(out["failures"], ensure_ascii=False, default=str), out["duration_s"], db.now_iso()))
        if out["passed"]:
            notify.resolve(f"agent:{role}", f"Role {role} passes its capability check again")
        else:
            bad = ", ".join(f"{c['metric']} {c['value']} (needs {c['op']} {c['threshold']})" for c in checks if not c["ok"])
            notify.alert(f"agent:{role}", f"Role {role} below its capability thresholds", f"{level} check: {bad}")
    return out


def latest(role: str) -> dict | None:
    """The newest result for the role's current configuration."""
    row = db.conn().execute("SELECT * FROM agent_checks WHERE role = ? AND fingerprint = ? ORDER BY id DESC LIMIT 1",
                            (role, fingerprint(role))).fetchone()
    return dict(row) if row else None


_allowed_cache: dict[str, tuple[float, bool]] = {}


def gates_enabled() -> bool:
    from sde_rag.settings import section
    return bool(section("gates").get("enabled", True))


def allowed(role: str, ttl_s: float = 60) -> bool:
    """For hot paths (every query, the review worker): the last result, cached for a minute; never runs a check."""
    if not gates_enabled():
        return True
    hit = _allowed_cache.get(role)
    if hit and time.monotonic() - hit[0] < ttl_s:
        return hit[1]
    ok = gate(role, run_if_stale=False)[0]
    _allowed_cache[role] = (time.monotonic(), ok)
    return ok


def gate(role: str, *, run_if_stale: bool = True) -> tuple[bool, str]:
    """(may the role be used, why). Without a recent passing record for this configuration the quick check runs
    (when allowed); a role that has never passed is treated as below threshold (DESIGN 8.4)."""
    max_age = timedelta(hours=float(_threshold_layers()[1].get("gate", {}).get("max_age_hours", 24)))
    rec = latest(role)
    fresh = rec and datetime.now(timezone.utc) - datetime.fromisoformat(rec["created_at"]) < max_age
    if fresh:
        return bool(rec["passed"]), f"{rec['level']} check {'passed' if rec['passed'] else 'failed'} at {rec['created_at']}"
    if not run_if_stale:
        return (bool(rec["passed"]), "last check is old") if rec else (False, "never checked")
    out = run(role, "quick")
    _allowed_cache.pop(role, None)
    return out["passed"], f"quick check {'passed' if out['passed'] else 'failed'} just now"
