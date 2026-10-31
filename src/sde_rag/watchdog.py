"""Watchdogs. Slow is fine; no progress at all is not.

RunWatchdog: a thread inside an ingest process. Every finished record calls `progress()`; when nothing has
progressed for `idle_s`, it dumps every thread's stack, lets the caller mark the run stalled, and exits the
process with EXIT_STALLED. The next scheduled run picks up the queued work.

check_ui: the systemd watchdog timer's probe of the UI's /health endpoint; two failures in a row restart it.
"""

import faulthandler
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Callable

import httpx

from sde_rag import db, notify
from sde_rag.settings import data_dir, section

log = logging.getLogger("sde_rag.watchdog")
EXIT_STALLED = 3
UI_UNIT = "sde-ui.service"

_last_progress = time.monotonic()


def progress() -> None:
    global _last_progress
    _last_progress = time.monotonic()


def idle_for() -> float:
    return time.monotonic() - _last_progress


def dump_stacks(tag: str) -> str:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"stall-{tag}-{datetime.now():%Y%m%d-%H%M%S}.txt"
    with open(path, "w") as f:
        faulthandler.dump_traceback(f, all_threads=True)
    try:
        faulthandler.dump_traceback(sys.stderr, all_threads=True)
    except (AttributeError, ValueError, OSError):  # stderr without a file descriptor
        pass
    return str(path)


class RunWatchdog:
    def __init__(self, idle_s: float, on_stall: Callable[[str], None], *, tag: str = "ingest",
                 check_s: float = 30, exit: Callable[[int], None] = os._exit):
        self.idle_s, self.on_stall, self.tag, self.check_s, self._exit = idle_s, on_stall, tag, check_s, exit
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"watchdog:{tag}")

    def __enter__(self) -> "RunWatchdog":
        progress()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.check_s):
            idle = idle_for()
            if idle < self.idle_s:
                continue
            path = dump_stacks(self.tag)
            log.error("%s: no progress for %.0fs; stacks written to %s", self.tag, idle, path)
            try:
                self.on_stall(path)
            except Exception:
                log.exception("%s: marking the stalled run failed", self.tag)
            self._exit(EXIT_STALLED)
            return


def check_ui(url: str | None = None, *, restart: bool = True) -> dict:
    """Probe /health; after `failures_before_restart` failures in a row restart the UI unit and alert."""
    cfg = section("ui")
    url = url or f"http://{cfg.get('host', '127.0.0.1')}:{cfg.get('port', 7860)}/health"
    need = int(section("watchdog").get("failures_before_restart", 2))
    try:
        r = httpx.get(url, timeout=float(section("watchdog").get("health_timeout_s", 30)))
        r.raise_for_status()
        ok, detail = bool(r.json().get("ok")), r.json()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {str(e)[:300]}"
    if ok:
        db.set_meta("ui_health_failures", "0")
        notify.resolve("ui:health", "UI healthy again", url)
        return {"ok": True, "url": url, "health": detail}
    failures = int(db.get_meta("ui_health_failures", "0")) + 1
    db.set_meta("ui_health_failures", str(failures))
    out = {"ok": False, "url": url, "error": detail, "failures_in_a_row": failures, "restarted": False}
    if failures >= need and restart and _unit_enabled(UI_UNIT):
        res = subprocess.run(["systemctl", "--user", "restart", UI_UNIT], capture_output=True, text=True, timeout=60)
        out["restarted"] = res.returncode == 0
        db.set_meta("ui_health_failures", "0")
        notify.alert("ui:health", "UI not healthy",
                     f"{url}: {detail}; {failures} checks in a row failed; "
                     f"{'restarted ' + UI_UNIT if out['restarted'] else 'restart failed: ' + res.stderr[:200]}")
    return out


def _unit_enabled(unit: str) -> bool:
    try:
        r = subprocess.run(["systemctl", "--user", "is-enabled", unit], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.stdout.strip() == "enabled"
