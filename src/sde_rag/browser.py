"""Headless Chrome on a persistent, logged-in profile, for sites that only work in a browser.

The launch flags must match the ones used when the profile was logged in by hand: `--password-store=basic`
keeps the cookies readable by systemd jobs (the desktop keyring is not reachable there).
A profile can be open in one Chrome only; another job (or a window opened by hand) holding it means this
run skips the source, it is not a failure. Only the Chrome processes this session started are ever killed.
"""

import os
import shutil
import signal
import subprocess
import time
from functools import cache
from pathlib import Path

from sde_rag import watchdog

CHROME_ARGS = ["--password-store=basic", "--no-first-run", "--no-default-browser-check"]
# Headless Chrome announces itself in the user agent and navigator.webdriver; sites behind Cloudflare block both.
STEALTH_ARGS = ["--disable-blink-features=AutomationControlled"]
VIEWPORT = {"width": 1280, "height": 900}


class ProfileInUse(Exception):
    pass


@cache
def user_agent() -> str:
    exe = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    if not exe:
        raise RuntimeError("google-chrome is not installed")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True).stdout.split()[-1]
    return f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"


def profile_in_use(profile: Path) -> bool:
    return os.path.lexists(profile / "SingletonLock")


def _chrome_pids(profile: Path) -> set[int]:
    """Processes started with this profile (Chrome passes --user-data-dir to all of its processes)."""
    want = f"--user-data-dir={profile}".encode()
    pids = set()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if want in (d / "cmdline").read_bytes():
                pids.add(int(d.name))
        except OSError:
            continue
    return pids


class BrowserSession:
    """One headless Chrome for a connector's whole run: list pages and posts share it."""

    def __init__(self, profile: Path, *, page_timeout_s: float = 60, session_limit_s: float = 900,
                 locale: str | None = None):
        self.profile = Path(profile).expanduser()
        self.page_timeout_ms = int(page_timeout_s * 1000)
        self.deadline = time.monotonic() + session_limit_s
        self.locale = locale
        self._pw = self._ctx = self.page = None
        self._pids: set[int] = set()
        self._cleanup_key: int | None = None

    def open(self) -> "BrowserSession":
        if not self.profile.is_dir():
            raise FileNotFoundError(f"browser profile {self.profile} does not exist")
        if profile_in_use(self.profile):
            raise ProfileInUse(f"{self.profile} is open in another Chrome")
        from playwright.sync_api import Error, sync_playwright
        before = _chrome_pids(self.profile)
        self._pw = sync_playwright().start()
        try:
            self._ctx = self._pw.chromium.launch_persistent_context(
                str(self.profile), channel="chrome", headless=True, args=CHROME_ARGS + STEALTH_ARGS,
                ignore_default_args=["--enable-automation"], user_agent=user_agent(), locale=self.locale,
                viewport=VIEWPORT)
        except Error as e:
            self._pw.stop()
            self._pw = None
            if "ProcessSingleton" in str(e) or "SingletonLock" in str(e) or "already in use" in str(e):
                raise ProfileInUse(str(e)[:300]) from e
            raise
        self._pids = _chrome_pids(self.profile) - before
        self._cleanup_key = watchdog.on_kill(self._kill)
        self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self.page.set_default_timeout(self.page_timeout_ms)
        return self

    def goto(self, url: str, *, settle_ms: int = 2000):
        if time.monotonic() > self.deadline:
            raise TimeoutError("browser session time limit reached")
        r = self.page.goto(url, wait_until="domcontentloaded", timeout=self.page_timeout_ms)
        self.page.wait_for_timeout(settle_ms)
        return r

    def _kill(self) -> None:
        for pid in self._pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def close(self) -> None:
        try:
            if self._ctx is not None:
                self._ctx.close()
        except Exception:
            self._kill()
        finally:
            if self._pw is not None:
                self._pw.stop()
            if self._cleanup_key is not None:
                watchdog.cancel_on_kill(self._cleanup_key)
            self._ctx = self._pw = self.page = None
