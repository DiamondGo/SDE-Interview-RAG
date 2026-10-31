"""Alerts: `alert(key, ...)` announces a problem once, and again every `repeat_hours` while it lasts;
`resolve(key, ...)` announces the recovery once. Channels: log, desktop (notify-send), telegram."""

import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import httpx

from sde_rag import db
from sde_rag.settings import section

log = logging.getLogger("sde_rag.notify")


def _send(title: str, body: str) -> None:
    cfg = section("notify")
    for ch in cfg.get("channels", ["log"]):
        try:
            if ch == "log":
                log.warning("%s: %s", title, body)
            elif ch == "desktop" and shutil.which("notify-send"):
                subprocess.run(["notify-send", "-u", "critical", "-a", "SDE Interview RAG", title, body[:500]],
                               timeout=10, check=False)
            elif ch == "telegram":
                token = os.environ.get(cfg.get("telegram_token_env", "TELEGRAM_BOT_TOKEN"))
                chat = os.environ.get(cfg.get("telegram_chat_env", "TELEGRAM_CHAT_ID"))
                if token and chat:
                    httpx.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=15,
                               json={"chat_id": chat, "text": f"{title}\n{body}"[:4000]})
        except Exception as e:  # an alert channel failing must never break the caller
            log.error("notify channel %s failed: %s", ch, e)


def alert(key: str, title: str, body: str = "") -> bool:
    """Returns True when a notification was actually sent."""
    now = datetime.now(timezone.utc)
    repeat = timedelta(hours=float(section("notify").get("repeat_hours", 24)))
    c = db.conn()
    row = c.execute("SELECT last_notified_at, resolved_at FROM alerts WHERE key = ?", (key,)).fetchone()
    if row and row["resolved_at"] is None and now - datetime.fromisoformat(row["last_notified_at"]) < repeat:
        return False
    with db.tx() as t:
        if row is None or row["resolved_at"] is not None:
            t.execute("INSERT INTO alerts (key, message, first_at, last_notified_at) VALUES (?, ?, ?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET message = excluded.message, first_at = excluded.first_at, "
                      "last_notified_at = excluded.last_notified_at, resolved_at = NULL",
                      (key, f"{title}: {body}", now.isoformat(), now.isoformat()))
        else:
            t.execute("UPDATE alerts SET message = ?, last_notified_at = ? WHERE key = ?",
                      (f"{title}: {body}", now.isoformat(), key))
    _send(title, body)
    return True


def resolve(key: str, title: str, body: str = "") -> bool:
    c = db.conn()
    row = c.execute("SELECT resolved_at FROM alerts WHERE key = ?", (key,)).fetchone()
    if row is None or row["resolved_at"] is not None:
        return False
    c.execute("UPDATE alerts SET resolved_at = ? WHERE key = ?", (db.now_iso(), key))
    _send(title, body)
    return True
