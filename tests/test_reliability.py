import gzip
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

from sde_rag import db, maintenance, watchdog
from sde_rag.agents import AllAttemptsFailed
from sde_rag.agents.backends import Breaker
from sde_rag.deploy import install
from sde_rag.graphs import ingest as ingest_graph
from sde_rag.pipeline import indexing, stages, tagging
from sde_rag.pipeline.fetch import source_due, sync_source
from tests.conftest import FakeBackend, make_registry
from tests.fakes import EXPERIENCE_MD, PREP_MD, FakeConnector


def _ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")


def _alert(key: str):
    return db.conn().execute("SELECT * FROM alerts WHERE key = ?", (key,)).fetchone()


def test_failing_source_backs_off_one_two_four_intervals_up_to_the_cap():
    c = db.conn()
    c.execute("INSERT INTO sources (name, last_run_at, last_success_at, consecutive_failures) VALUES "
              "('s', ?, ?, 0)", (_ago(13), _ago(13)))
    assert source_due("s", 12)
    for failures, hours_since_try, due in ((1, 11, False), (1, 12, True), (2, 20, False), (2, 24, True),
                                           (3, 47, False), (5, 47.6, True)):          # 12 * 2**4 capped at 48
        c.execute("UPDATE sources SET consecutive_failures = ?, last_run_at = ? WHERE name = 's'",
                  (failures, _ago(hours_since_try)))
        assert source_due("s", 12) is due, (failures, hours_since_try)


def test_fetch_reports_progress_for_every_listed_item():
    beats = []
    docs = {f"d{i}": {"version": "1", "content": PREP_MD + f"\n## Extra {i}?\nx\n"} for i in range(3)}
    sync_source("repo", FakeConnector("repo", docs), heartbeat=lambda: beats.append(1))
    assert len(beats) >= 3


def test_dead_letters_are_listed_and_replayed_through_their_stage():
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD}}))
    doc_id = db.conn().execute("SELECT id FROM documents").fetchone()[0]
    for _ in range(5):
        stages.mark_failed(doc_id, "clean", "boom")
    assert [r["id"] for r in stages.dead_letters("clean")] == [doc_id]
    assert stages.clean_stage(10)["cleaned"] == 0            # dead letters are not retried by themselves
    assert stages.replay("parse") == {}
    assert stages.replay("clean") == {"clean": 1}
    row = db.conn().execute("SELECT status, attempts, next_retry_at FROM documents").fetchone()
    assert (row["status"], row["attempts"], row["next_retry_at"]) == ("fetched", 0, None)
    assert stages.clean_stage(10)["cleaned"] == 1
    with pytest.raises(ValueError):
        stages.replay("extract-typo")


def test_replay_failed_skips_the_retry_wait():
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD}}))
    doc_id = db.conn().execute("SELECT id FROM documents").fetchone()[0]
    stages.mark_failed(doc_id, "clean", "boom")
    assert stages.clean_stage(10)["cleaned"] == 0            # waiting an hour for the retry
    assert stages.replay(status="failed") == {"clean": 1}
    assert stages.clean_stage(10)["cleaned"] == 1


def test_one_bad_document_does_not_stop_the_index_stage(monkeypatch):
    sync_source("a", FakeConnector("a", {"e1": {"version": "1", "content": EXPERIENCE_MD, "kind": "experience"},
                                         "e2": {"version": "1", "content": EXPERIENCE_MD + "\nMore.\n" * 30,
                                                "kind": "experience"}}))
    stages.clean_stage(10)
    stages.parse_stage(10)
    tagging.rule_tag_stage()
    stages.merge_stage()
    real = indexing.split_paragraphs
    monkeypatch.setattr(indexing, "split_paragraphs",
                        lambda text, n: (_ for _ in ()).throw(RuntimeError("bad")) if "More." in text else real(text, n))
    st = indexing.index_stage(10)
    assert st["docs_indexed"] == 1 and st["failed"] == 1
    rows = dict(db.conn().execute("SELECT external_id, status FROM documents").fetchall())
    assert rows == {"e1": "indexed", "e2": "failed"}
    assert db.conn().execute("SELECT COUNT(*) FROM chunks WHERE doc_id = (SELECT id FROM documents "
                             "WHERE external_id = 'e2')").fetchone()[0] == 0      # rolled back


def test_watchdog_dumps_stacks_marks_and_exits_when_nothing_progresses():
    stalled, exits = [], []
    with watchdog.RunWatchdog(0.2, stalled.append, tag="t", check_s=0.05, exit=exits.append):
        time.sleep(0.6)
    assert exits == [watchdog.EXIT_STALLED]
    assert len(stalled) == 1 and "Thread" in open(stalled[0]).read()


def test_watchdog_leaves_a_slow_run_that_keeps_progressing_alone():
    exits = []
    with watchdog.RunWatchdog(0.3, lambda p: None, tag="t", check_s=0.05, exit=exits.append):
        for _ in range(10):
            time.sleep(0.08)
            watchdog.progress()
    assert exits == []


def test_hung_model_is_detected_falls_back_alerts_and_recovers():
    hung = FakeBackend("local", delay=1.0)
    hung.max_concurrency = 4
    probe_ok = {"v": False}
    hung.breaker = Breaker(lambda: probe_ok["v"], enabled=True, open_s=600)
    spare = FakeBackend("spare", structured={"items": []})
    reg = make_registry({"tagger": {"backend": "local", "timeout_s": 0.1, "fallbacks": ["spare"]}},
                        {"local": hung, "spare": spare})
    for _ in range(2):                                        # two stalls, two failed probes
        assert reg.structured("tagger", tagging.TagBatch, "x").items == []
    assert hung.breaker.is_open() and _alert("llm:local")["resolved_at"] is None
    calls = hung.calls
    reg.structured("tagger", tagging.TagBatch, "x")           # circuit open: straight to the fallback
    assert hung.calls == calls
    assert reg.take_stalls() == {"tagger@local": 2} and reg.take_stalls() == {}
    hung.breaker.open_until, hung.delay, hung.structured = 0, 0, {"items": []}
    reg.structured("tagger", tagging.TagBatch, "x")
    assert _alert("llm:local")["resolved_at"] is not None


def test_hung_model_without_fallback_fails_fast_once_the_circuit_opens():
    hung = FakeBackend("local", delay=1.0)
    hung.max_concurrency = 4
    hung.breaker = Breaker(lambda: False, enabled=True)
    reg = make_registry({"tagger": {"backend": "local", "timeout_s": 0.1}}, {"local": hung})
    for _ in range(2):
        with pytest.raises(AllAttemptsFailed):
            reg.structured("tagger", tagging.TagBatch, "x")
    t0 = time.monotonic()
    with pytest.raises(AllAttemptsFailed, match="circuit open"):
        reg.structured("tagger", tagging.TagBatch, "x")
    assert time.monotonic() - t0 < 0.1


def test_retry_on_a_hung_single_slot_backend_probes_instead_of_queuing_behind_the_stalled_call():
    hung = FakeBackend("local", delay=2.0)                    # one slot, like the local server
    hung.breaker = Breaker(lambda: False, enabled=True)
    reg = make_registry({"tagger": {"backend": "local", "timeout_s": 0.1, "fallbacks": ["local"]}}, {"local": hung})
    t0 = time.monotonic()
    with pytest.raises(AllAttemptsFailed, match="failed its health probe again"):
        reg.structured("tagger", tagging.TagBatch, "x")
    assert time.monotonic() - t0 < 1.0
    assert hung.calls == 1 and hung.breaker.is_open()
    assert _alert("llm:local")["resolved_at"] is None


def _summary(backlog: int, dead: int = 0, stalls: int = 0) -> dict:
    return {"duplicate_download_rate": 0.0, "documents_by_status": {"dead": dead}, "backlog": backlog,
            "stalls": {"tagger@local": stalls} if stalls else {}, "stall_count": stalls}


def test_run_report_alerts_on_dead_letters_backlog_growth_and_stalls():
    cfg = {"dead_alert_increase": 30, "stall_alert": 3}
    ingest_graph._alerts(_summary(10, dead=40), [_summary(10, dead=5)], cfg, 0, 0)
    assert _alert("ingest:dead") and _alert("ingest:backlog") is None
    ingest_graph._alerts(_summary(40), [_summary(30), _summary(20), _summary(10)], cfg, 0, 0)
    assert _alert("ingest:backlog")["resolved_at"] is None
    ingest_graph._alerts(_summary(35), [_summary(40), _summary(30)], cfg, 0, 0)
    assert _alert("ingest:backlog")["resolved_at"] is not None
    ingest_graph._alerts(_summary(0, stalls=3), [], cfg, 0, 0)
    assert _alert("ingest:stalls")


def test_ingest_run_marks_runs_of_dead_processes_abandoned(monkeypatch):
    monkeypatch.setattr(ingest_graph, "source_configs", lambda: {})
    make_registry({}, {})
    db.conn().execute("INSERT INTO ingest_runs (run_key, started_at, status) VALUES ('old', ?, 'running')",
                      (db.now_iso(),))
    out = ingest_graph.graph.invoke({"run_key": "2026-11-01-AM"})
    assert "stall_count" in out["summary"] and "backlog" in out["summary"]
    assert db.conn().execute("SELECT status FROM ingest_runs WHERE run_key = 'old'").fetchone()[0] == "abandoned"


def test_backup_is_a_consistent_gzipped_copy_and_keeps_the_newest(tmp_path):
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD}}))
    outs = [maintenance.backup(keep=2) for _ in range(3)]
    assert outs[-1]["kept"] == 2 and len(outs[-1]["removed"]) == 1
    restored = tmp_path / "restored.db"
    with gzip.open(outs[-1]["backup"], "rb") as f:
        restored.write_bytes(f.read())
    assert sqlite3.connect(restored).execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_reconcile_repairs_references_requests_full_github_listing_and_refreshes_heat(monkeypatch):
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD}}))
    stages.clean_stage(10)
    stages.parse_stage(10)
    stages.merge_stage()
    c = db.conn()
    c.execute("INSERT INTO seen_items (source, external_id, doc_id, first_seen_at, last_seen_at) "
              "VALUES ('a', 'gone', 99999, ?, ?)", (db.now_iso(), db.now_iso()))
    c.execute("INSERT INTO sources (name, state) VALUES ('gh', ?)", (json.dumps({"o/r": {"sha": "x"}}),))
    monkeypatch.setattr(maintenance, "source_configs", lambda: {"gh": {"type": "github_repos"}, "a": {"type": "devto"}})
    out = maintenance.reconcile()
    assert out["seen_items_unlinked"] == 1 and out["full_listing_next_run"] == ["gh"]
    assert out["heat"]["updated"] == 3 and out["fts"] == "ok" and out["missing_raw_files"] == 0
    assert json.loads(c.execute("SELECT state FROM sources WHERE name = 'gh'").fetchone()[0])["o/r"]["complete"] is False
    assert ingest_graph._acquire_lock("someone-else")
    assert "lock" in maintenance.reconcile()["skipped"]


def test_ui_watchdog_restarts_the_unit_after_two_failures(monkeypatch):
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "enabled\n", ""

    monkeypatch.setattr(watchdog.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    monkeypatch.setattr(watchdog.subprocess, "run", lambda cmd, **k: calls.append(cmd) or Done())
    first = watchdog.check_ui("http://x/health")
    assert first["failures_in_a_row"] == 1 and not first["restarted"] and calls == []
    second = watchdog.check_ui("http://x/health")
    assert second["restarted"] and ["systemctl", "--user", "restart", "sde-ui.service"] in calls
    assert _alert("ui:health")["resolved_at"] is None

    class Ok:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(watchdog.httpx, "get", lambda *a, **k: Ok())
    assert watchdog.check_ui("http://x/health")["ok"]
    assert _alert("ui:health")["resolved_at"] is not None


def test_a_run_started_by_hand_does_not_take_the_scheduled_slot(monkeypatch):
    from sde_rag import cli, runner
    keys = []
    monkeypatch.setattr(runner, "run_ingest", lambda **kw: keys.append(kw["run_key"]) or {})
    for argv in (["ingest"], ["ingest", "--scheduled"], ["ingest", "--run-key", "x"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    assert keys[0].startswith("manual-") and keys[1] is None and keys[2] == "x"


def test_install_systemd_writes_links_and_leaves_foreign_units_alone(tmp_path):
    out_dir, unit_dir = tmp_path / "deploy", tmp_path / "user"
    unit_dir.mkdir()
    (unit_dir / "sde-ui.service").write_text("[Service]\nExecStart=/bin/true\n")   # someone else's unit
    calls = []
    sysctl = lambda *a: calls.append(a) or type("R", (), {"returncode": 0, "stderr": ""})()
    rep = install(enable=True, out_dir=out_dir, unit_dir=unit_dir, systemctl=sysctl)
    assert len(rep["written"]) == 9 and len(rep["linked"]) == 8 and "sde-ui.service" in rep["conflicts"][0]
    assert (unit_dir / "sde-ingest.timer").resolve() == (out_dir / "sde-ingest.timer").resolve()
    timer = (out_dir / "sde-ingest.timer").read_text()
    assert "OnCalendar=*-*-* 06,18:30:00" in timer and "Persistent=true" in timer
    assert "TimeoutStartSec=4h" in (out_dir / "sde-ingest.service").read_text()
    assert "sde-rag ingest --scheduled" in (out_dir / "sde-ingest.service").read_text()
    nightly = (out_dir / "sde-eval.service").read_text()
    assert "ExecStart=-" in nightly and "agent-check --level full --roles reviewer --max-age-hours 160" in nightly
    assert ("enable", "--now", "sde-ingest.timer") in calls and ("enable", "--now", "sde-ui.service") not in calls
    again = install(out_dir=out_dir, unit_dir=unit_dir, systemctl=sysctl)
    assert again["linked"] == []
