"""systemd user units for this machine: written to deploy/systemd/ (machine paths, not committed) and linked
into ~/.config/systemd/user. Schedule: DESIGN.md section 10."""

import os
import shutil
import subprocess
from pathlib import Path

from sde_rag.settings import ROOT

TIMERS = ("sde-ingest.timer", "sde-eval.timer", "sde-reconcile.timer", "sde-watchdog.timer")
SERVICES = ("sde-ui.service",)


def _units(uv: str) -> dict[str, str]:
    path = ":".join(dict.fromkeys([str(Path(uv).parent), str(Path.home() / ".local/bin"), "/usr/local/bin",
                                   "/usr/bin", "/bin"]))
    run = f"{uv} run --frozen sde-rag"
    service = (f"[Service]\nType=oneshot\nWorkingDirectory={ROOT}\nEnvironment=PYTHONUNBUFFERED=1\n"
               f"Environment=PATH={path}\nNice=10\n")
    net = "After=network-online.target\nWants=network-online.target\n"
    return {
        "sde-ingest.service": f"[Unit]\nDescription=SDE Interview RAG: fetch, parse, tag and index new content\n{net}\n"
                              f"{service}ExecStart={run} ingest\nTimeoutStartSec=4h\n",
        "sde-ingest.timer": "[Unit]\nDescription=SDE Interview RAG ingest at 06:30 and 18:30\n\n"
                            "[Timer]\nOnCalendar=*-*-* 06,18:30:00\nRandomizedDelaySec=120\nPersistent=true\n\n"
                            "[Install]\nWantedBy=timers.target\n",
        "sde-eval.service": "[Unit]\nDescription=SDE Interview RAG: database backup and nightly evaluation\n\n"
                            f"{service}ExecStart={run} backup\nExecStart={run} eval\nTimeoutStartSec=2h\n",
        "sde-eval.timer": "[Unit]\nDescription=SDE Interview RAG backup and evaluation at 03:30\n\n"
                          "[Timer]\nOnCalendar=*-*-* 03:30:00\nRandomizedDelaySec=120\nPersistent=true\n\n"
                          "[Install]\nWantedBy=timers.target\n",
        "sde-reconcile.service": f"[Unit]\nDescription=SDE Interview RAG: weekly reconciliation\n\n"
                                 f"{service}ExecStart={run} reconcile\nTimeoutStartSec=1h\n",
        "sde-reconcile.timer": "[Unit]\nDescription=SDE Interview RAG reconciliation on Sundays at 04:30\n\n"
                               "[Timer]\nOnCalendar=Sun *-*-* 04:30:00\nPersistent=true\n\n"
                               "[Install]\nWantedBy=timers.target\n",
        "sde-watchdog.service": f"[Unit]\nDescription=SDE Interview RAG: UI health check (restarts it when stuck)\n\n"
                                f"{service}ExecStart={run} watchdog-ui\nTimeoutStartSec=5min\n",
        "sde-watchdog.timer": "[Unit]\nDescription=SDE Interview RAG UI health check every 5 minutes\n\n"
                              "[Timer]\nOnBootSec=5min\nOnUnitActiveSec=5min\nAccuracySec=30s\n\n"
                              "[Install]\nWantedBy=timers.target\n",
        "sde-ui.service": f"[Unit]\nDescription=SDE Interview RAG web UI\n{net}\n"
                          f"[Service]\nType=simple\nWorkingDirectory={ROOT}\nEnvironment=PYTHONUNBUFFERED=1\n"
                          f"Environment=PATH={path}\nExecStart={run} ui\nRestart=on-failure\nRestartSec=30\n\n"
                          "[Install]\nWantedBy=default.target\n",
    }


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=120)


def install(*, enable: bool = False, ui: bool = False, out_dir: Path | None = None,
            unit_dir: Path | None = None, systemctl=_systemctl) -> dict:
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    out_dir = out_dir or ROOT / "deploy" / "systemd"
    unit_dir = unit_dir or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd" / "user"
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"written": [], "linked": [], "conflicts": [], "enabled": []}
    for name, text in _units(uv).items():
        src = out_dir / name
        src.write_text(text)
        report["written"].append(str(src))
        link = unit_dir / name
        if link.is_symlink() and link.resolve() == src.resolve():
            continue
        if link.exists() or link.is_symlink():
            report["conflicts"].append(f"{link} exists and is not a link to {src}: left alone")
            continue
        link.symlink_to(src)
        report["linked"].append(str(link))
    systemctl("daemon-reload")
    # The UI first: enabling the watchdog timer runs its first health check right away.
    wanted = (SERVICES if ui else ()) + (TIMERS if enable else ())
    for unit in wanted:
        r = systemctl("enable", "--now", unit)
        (report["enabled"] if r.returncode == 0 else report["conflicts"]).append(
            unit if r.returncode == 0 else f"{unit}: {r.stderr.strip()[:200]}")
    return report


def uninstall(*, unit_dir: Path | None = None, systemctl=_systemctl) -> dict:
    unit_dir = unit_dir or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd" / "user"
    out_dir = ROOT / "deploy" / "systemd"
    removed = []
    for unit in TIMERS + SERVICES:
        systemctl("disable", "--now", unit)
    for name in _units("uv"):
        link = unit_dir / name
        if link.is_symlink() and link.resolve().parent == out_dir.resolve():
            link.unlink()
            removed.append(str(link))
    systemctl("daemon-reload")
    return {"unlinked": removed}
