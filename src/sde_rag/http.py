"""HTTP for connectors: per-host spacing, retries with backoff (honouring Retry-After), conditional requests,
and robots.txt for web pages (official APIs are not crawled pages and skip it)."""

import random
import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from sde_rag.settings import section

RETRY_STATUS = {429, 500, 502, 503, 504}
_TRACKING = ("utm_", "fbclid", "gclid", "ref", "source", "mc_cid", "mc_eid")


def normalize_url(url: str) -> str:
    """Same page -> same string: lower-case host, no www, no tracking params, no fragment, no trailing slash."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not any(k.lower() == t or k.lower().startswith(t) for t in _TRACKING)]
    path = parts.path.rstrip("/") or ""
    return urlunsplit(("https" if parts.scheme in ("http", "https") else parts.scheme, host, path,
                       urlencode(sorted(query)), ""))


@dataclass
class Validators:
    etag: str | None = None
    last_modified: str | None = None


class NotModified(Exception):
    pass


class RobotsDisallowed(Exception):
    pass


class HttpClient:
    def __init__(self, *, delay_s: float | None = None, user_agent: str | None = None,
                 transport: httpx.BaseTransport | None = None):
        cfg = section("http")
        self.delay_s = float(cfg.get("default_delay_s", 2.0) if delay_s is None else delay_s)
        self.max_attempts = int(cfg.get("max_attempts", 4))
        self.user_agent = user_agent or cfg.get("user_agent", "SDE-Interview-RAG/0.1")
        self.client = httpx.Client(timeout=float(cfg.get("timeout_s", 30)), follow_redirects=True,
                                   headers={"User-Agent": self.user_agent}, transport=transport)
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _space(self, host: str) -> None:
        with self._lock:
            wait = self._last.get(host, 0) + self.delay_s - time.monotonic()
            self._last[host] = time.monotonic() + max(wait, 0)
        if wait > 0:
            time.sleep(wait)

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.client.get(f"{origin}/robots.txt", timeout=10)
                if r.status_code >= 400:
                    rp = None  # no robots.txt: everything allowed
                else:
                    rp.parse(r.text.splitlines())
            except httpx.HTTPError:
                rp = None
            self._robots[origin] = rp
        rp = self._robots[origin]
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    def get(self, url: str, *, params: dict | None = None, headers: dict | None = None,
            validators: Validators | None = None, robots: bool = False) -> httpx.Response:
        if robots and not self.allowed(url):
            raise RobotsDisallowed(url)
        hdrs = dict(headers or {})
        if validators and validators.etag:
            hdrs["If-None-Match"] = validators.etag
        if validators and validators.last_modified:
            hdrs["If-Modified-Since"] = validators.last_modified
        host = urlsplit(url).netloc
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._space(host)
            try:
                r = self.client.get(url, params=params, headers=hdrs)
            except httpx.TransportError as e:
                last_exc = e
                self._backoff(attempt, None)
                continue
            if r.status_code == 304:
                raise NotModified(url)
            if r.status_code in RETRY_STATUS and attempt < self.max_attempts:
                self._backoff(attempt, r.headers.get("Retry-After"))
                continue
            r.raise_for_status()
            return r
        raise last_exc or httpx.HTTPError(f"GET {url} failed")

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = min(60.0, 2 ** attempt) + random.uniform(0, 1)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                try:
                    delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                except (TypeError, ValueError):
                    pass
        time.sleep(min(delay, 300))

    def close(self) -> None:
        self.client.close()
