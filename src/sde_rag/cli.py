"""sde-rag: command line for the SDE interview RAG."""

import argparse
import json
import logging
import sys


def _print_table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def cmd_migrate(args) -> int:
    from sde_rag import db
    print("database:", db.db_path())
    print("applied:", db.migrate(db.conn()) or "nothing new")
    return 0


def cmd_ingest(args) -> int:
    from sde_rag.runner import run_ingest

    def show(update):
        for node, delta in update.items():
            if node == "__interrupt__" or not isinstance(delta, dict):
                continue
            stats = delta.get("stats") or {}
            brief = json.dumps(stats, ensure_ascii=False, default=str)[:400] if stats else ""
            print(f"[{node}] {brief}", flush=True)

    summary = run_ingest(run_key=args.run_key, sources=args.sources.split(",") if args.sources else None,
                         force=args.force, doc_limit=args.doc_limit, skip_llm=args.skip_llm, on_update=show)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    return 0


def cmd_query(args) -> int:
    from sde_rag.runner import QueryService, query_service
    filters = {"category": args.category, "tags": args.tags.split(",") if args.tags else None}
    stream = query_service().stream(QueryService.new_thread(), args.question, {k: v for k, v in filters.items() if v})
    streamed = ""
    for kind, payload in stream:
        if kind == "event":
            if payload["type"] == "token":
                streamed += payload["text"]
                print(payload["text"], end="", flush=True)
            elif payload["type"] == "reset":
                streamed = ""
                print("\n[--- discarded partial answer ---]", flush=True)
            elif payload["type"] == "status":
                print(f"[{payload['text']}]", file=sys.stderr, flush=True)
        else:
            print("\n")
            final = payload.get("answer") or ""
            if final.strip() != streamed.strip():
                print(f"[--- verified answer ---]\n{final}\n")
            if payload.get("degraded"):
                print("[degraded answer]", file=sys.stderr)
            for c in payload.get("citations") or []:
                print(f"[{c['label']}] {c['title'][:100]} {c['url']}")
            job_id = payload.get("review_job_id")
            if job_id and args.review:
                from sde_rag import review
                review.request_review(int(job_id))
                review.run_one((1,))
                print("\n[reviewer]", json.dumps(review.job(int(job_id)).get("result"), ensure_ascii=False, indent=1))
            elif job_id:
                print(f"\n[review queued as job {job_id}: `sde-rag review --run 1`, or --review to wait for it]")
    return 0


def cmd_review(args) -> int:
    from sde_rag import db, review
    if args.list:
        rows = [dict(r) for r in db.conn().execute(
            "SELECT id, kind, priority, status, verdict, substr(replace(question, char(10), ' '), 1, 60) AS question, "
            "finished_at FROM review_jobs ORDER BY id DESC LIMIT ?", (args.limit,))]
        if rows:
            _print_table(rows, ["id", "kind", "priority", "status", "verdict", "finished_at", "question"])
        print(f"\ncalls left today: {review.quota_left()}")
        return 0
    if args.show:
        print(json.dumps(review.job(args.show), ensure_ascii=False, indent=1))
        return 0
    done = 0
    while done < args.run:
        out = review.run_one((1, 2))
        if out is None:
            break
        print(json.dumps(out, ensure_ascii=False))
        done += 1
    if args.sources:
        print(json.dumps(review.review_source_answers(args.sources), ensure_ascii=False, indent=1))
    print(f"calls left today: {review.quota_left()}")
    return 0


def cmd_eval(args) -> int:
    from sde_rag.runner import run_eval
    out = run_eval(args.suite, sample_size=args.sample, compare_to=args.compare_to)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out.get("passed") else 1


def cmd_agents(args) -> int:
    from sde_rag.agents import get_registry
    reg = get_registry()
    _print_table(reg.describe(), ["role", "order", "backend", "type", "model", "base_url", "thinking", "stream",
                                  "fingerprint"])
    probs = reg.problems()
    if probs:
        print("\nproblems:")
        for role, items in sorted(probs.items()):
            for level, msg in items:
                print(f"  {level:<7} {role}: {msg}")
    return 0


def cmd_doctor(args) -> int:
    import time

    from sde_rag import db
    from sde_rag.agents import get_registry
    from sde_rag.index.embedder import configured_model_name
    from sde_rag.index.versions import IndexModelMismatch, check_model
    reg = get_registry()
    failures = 0
    print("database:", db.db_path())
    try:
        print("index:", check_model(), "embedding model:", configured_model_name())
    except IndexModelMismatch as e:
        print("index: ERROR", e)
        failures += 1
    for role, items in sorted(reg.problems().items()):
        for level, msg in items:
            print(f"{level.upper():<7} role {role}: {msg}")
            failures += level == "error"
    used = sorted({a.backend for r in reg.roles.values() for a in r.attempts})
    for name in used:
        b = reg.backends.get(name)
        if b is None or b.problems():
            continue
        t0 = time.monotonic()
        try:
            out = b.invoke_text("Reply with exactly: OK", {"thinking": False, "max_tokens": 50, "timeout_s": 120})
            print(f"OK      backend {name:<14} {time.monotonic() - t0:5.1f}s  {out.strip()[:40]!r}")
        except Exception as e:
            failures += 1
            print(f"FAIL    backend {name:<14} {time.monotonic() - t0:5.1f}s  {type(e).__name__}: {str(e)[:200]}")
    print("\nAgent capability tests (pytest -m agent) arrive in milestone 2.")
    return 1 if failures else 0


def cmd_stats(args) -> int:
    from sde_rag import db
    c = db.conn()
    out = {
        "documents": {r[0]: r[1] for r in c.execute("SELECT status, COUNT(*) FROM documents GROUP BY status")},
        "documents_by_source": {r[0]: r[1] for r in c.execute("SELECT source, COUNT(*) FROM documents GROUP BY source")},
        "questions": c.execute("SELECT COUNT(*) FROM questions").fetchone()[0],
        "questions_by_category": {r[0] or "none": r[1] for r in c.execute(
            "SELECT category, COUNT(*) FROM questions GROUP BY category ORDER BY 2 DESC")},
        "questions_by_tag_status": {r[0] or "untagged": r[1] for r in c.execute(
            "SELECT tag_status, COUNT(*) FROM questions GROUP BY tag_status")},
        "chunks": {f"{r[0]}/{'embedded' if r[1] else 'pending'}": r[2] for r in c.execute(
            "SELECT kind, embedded, COUNT(*) FROM chunks GROUP BY kind, embedded")},
        "sources": [dict(r) for r in c.execute(
            "SELECT name, last_success_at, consecutive_failures, last_error FROM sources")],
        "last_runs": [dict(r) for r in c.execute(
            "SELECT run_key, started_at, finished_at, status FROM ingest_runs ORDER BY id DESC LIMIT 5")],
        "dead_letters": {r[0] or "?": r[1] for r in c.execute(
            "SELECT stage, COUNT(*) FROM documents WHERE status = 'dead' GROUP BY stage")},
        "open_alerts": [dict(r) for r in c.execute(
            "SELECT key, message, first_at, last_notified_at FROM alerts WHERE resolved_at IS NULL ORDER BY first_at")],
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def cmd_publish_data(args) -> int:
    from sde_rag.pipeline.dataset import publish_data
    out = publish_data(upload=not args.no_upload)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if out["database"]["release"]:
        print(f"\nUploaded release {out['database']['release']}. Next: git add dataset && git commit -m "
              f"'Update dataset' && git push  (MANIFEST.json names the release; raw files use Git LFS)")
    else:
        print("\nLocal only: MANIFEST.json names no release. Do not commit it; run without --no-upload to publish.")
    return 0


def cmd_import_data(args) -> int:
    from sde_rag.pipeline.dataset import import_data
    print(json.dumps(import_data(force=args.force), ensure_ascii=False, indent=1))
    return 0


def cmd_replay(args) -> int:
    from sde_rag.pipeline import stages
    if args.list:
        rows = stages.dead_letters(args.stage, args.status, args.limit)
        for r in rows:
            r["last_error"] = (r["last_error"] or "").replace("\n", " ")[:80]
            r["title"] = (r["title"] or "")[:50]
        if rows:
            _print_table(rows, ["id", "source", "stage", "attempts", "updated_at", "title", "last_error"])
        else:
            print(f"no {args.status} documents" + (f" at stage {args.stage}" if args.stage else ""))
        return 0
    ids = [int(x) for x in args.ids.split(",")] if args.ids else None
    out = stages.replay(args.stage, args.status, ids)
    print(json.dumps({"requeued": out, "total": sum(out.values())}, indent=1))
    if out:
        print("The next `sde-rag ingest` run processes them.")
    return 0


def cmd_backup(args) -> int:
    from sde_rag.maintenance import backup
    print(json.dumps(backup(args.keep), indent=1))
    return 0


def cmd_reconcile(args) -> int:
    from sde_rag.maintenance import reconcile
    out = reconcile()
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 75 if out.get("skipped") else 0      # EX_TEMPFAIL: try again later


def cmd_watchdog_ui(args) -> int:
    from sde_rag.watchdog import check_ui
    out = check_ui(args.url, restart=not args.no_restart)
    print(json.dumps(out, ensure_ascii=False, default=str))
    return 0 if out["ok"] else 1


def cmd_install_systemd(args) -> int:
    from sde_rag import deploy
    out = deploy.uninstall() if args.remove else deploy.install(enable=args.enable, ui=args.ui)
    print(json.dumps(out, indent=1))
    if not args.remove and not args.enable:
        print("\nUnits linked but not enabled. Enable the timers with --enable (and the UI service with --ui).")
    return 1 if out.get("conflicts") else 0


def cmd_ui(args) -> int:
    from sde_rag.ui.app import serve
    serve(host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="sde-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="apply database migrations").set_defaults(fn=cmd_migrate)
    s = sub.add_parser("ingest", help="fetch, parse, tag and index new content")
    s.add_argument("--run-key")
    s.add_argument("--sources", help="comma-separated source names (default: all due)")
    s.add_argument("--force", action="store_true", help="run even if this run key is done; ignore intervals")
    s.add_argument("--doc-limit", type=int)
    s.add_argument("--skip-llm", action="store_true", help="rule tags only")
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("query", help="ask one question on the command line")
    s.add_argument("question")
    s.add_argument("--category")
    s.add_argument("--tags")
    s.add_argument("--review", action="store_true", help="wait for the reviewer's verdict on the answer")
    s.set_defaults(fn=cmd_query)
    s = sub.add_parser("review", help="the reviewer's queue: run queued reviews, list them, show one")
    s.add_argument("--run", type=int, default=0, help="review up to N queued answers (the UI does this itself)")
    s.add_argument("--sources", type=int, default=0, help="review up to N source answers of high-heat questions")
    s.add_argument("--list", action="store_true")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--show", type=int, help="print one review job")
    s.set_defaults(fn=cmd_review)
    s = sub.add_parser("eval", help="run an evaluation suite")
    s.add_argument("--suite", default="retrieval_self")
    s.add_argument("--sample", type=int, default=100)
    s.add_argument("--compare-to", type=int)
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("agents", help="show which backend serves each role")
    s.add_argument("action", choices=["show"], nargs="?", default="show")
    s.set_defaults(fn=cmd_agents)
    sub.add_parser("doctor", help="check configuration and that every backend answers").set_defaults(fn=cmd_doctor)
    sub.add_parser("stats", help="counts of documents, questions and chunks").set_defaults(fn=cmd_stats)
    s = sub.add_parser("publish-data", help="publish the dataset (no local usage data, no unpublishable sources): "
                       "the database to a GitHub release, raw files to dataset/")
    s.add_argument("--no-upload", action="store_true", help="only write dataset/ locally")
    s.set_defaults(fn=cmd_publish_data)
    s = sub.add_parser("import-data", help="start data/ from the published dataset (downloads the database)")
    s.add_argument("--force", action="store_true", help="replace an existing working database")
    s.set_defaults(fn=cmd_import_data)
    s = sub.add_parser("replay", help="send failed or dead-letter documents through their stage again")
    s.add_argument("--stage", choices=["clean", "parse", "index", "extract"], help="default: every stage")
    s.add_argument("--status", choices=["dead", "failed"], default="dead",
                   help="dead: gave up after max_attempts; failed: waiting for a retry (skip the wait)")
    s.add_argument("--ids", help="comma-separated document ids (default: all matching)")
    s.add_argument("--list", action="store_true", help="only list them")
    s.add_argument("--limit", type=int, default=100, help="rows shown by --list")
    s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("backup", help="gzipped online backup of the database into data/backups/")
    s.add_argument("--keep", type=int, help="backups kept (default [maintenance] backup_keep)")
    s.set_defaults(fn=cmd_backup)
    sub.add_parser("reconcile", help="weekly consistency repair, full GitHub listing next run, heat refresh"
                   ).set_defaults(fn=cmd_reconcile)
    s = sub.add_parser("watchdog-ui", help="check the UI's /health; restart sde-ui.service after repeated failures")
    s.add_argument("--url", help="default: http://<[ui] host>:<[ui] port>/health")
    s.add_argument("--no-restart", action="store_true")
    s.set_defaults(fn=cmd_watchdog_ui)
    s = sub.add_parser("install-systemd", help="write systemd user units to deploy/systemd/ and link them")
    s.add_argument("--enable", action="store_true", help="also enable and start the timers")
    s.add_argument("--ui", action="store_true", help="also enable and start sde-ui.service")
    s.add_argument("--remove", action="store_true", help="disable the units and remove the links")
    s.set_defaults(fn=cmd_install_systemd)
    s = sub.add_parser("ui", help="start the web UI")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_ui)
    args = p.parse_args(argv)
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
