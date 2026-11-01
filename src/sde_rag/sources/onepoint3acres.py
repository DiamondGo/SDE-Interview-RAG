"""1Point3Acres (一亩三分地) interview-experience board, read in headless Chrome with a logged-in profile.

Direct requests get 403, so pages are opened in a browser (sde_rag.browser) on the profile named by
`chrome_profile` in [onepoint3acres] of settings.toml (without one the source is skipped). Rules (DESIGN 3.4):
- low frequency: only in the runs listed in `run_hours`, at most `max_downloads_per_run` posts, a random
  10 to 20 seconds before every page;
- read only: list pages and the first post of a thread; nothing is clicked, posted or replied to;
- a login page, a CAPTCHA or a permission prompt stops the run and raises an alert; it is never worked around.
  Parts of a post hidden behind a points threshold stay hidden: the visible text is stored, marked partial;
- the profile in use by another Chrome: this run is skipped, not failed.
"""

import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from sde_rag import notify
from sde_rag.browser import BrowserSession, ProfileInUse
from sde_rag.http import HttpClient, Validators
from sde_rag.settings import section
from sde_rag.sources.base import DocRef, ListResult, RawDoc, SkipDoc

BASE = "https://www.1point3acres.com/bbs"
HIDDEN_MARK = "[部分内容需要更高积分才能查看，未读取]"
_HIDDEN = re.compile(r"(?:\S*，)?您好！\s*本帖隐藏的内容需要积分高于[\s\S]*?(?:获取积分的方式|Curated Interview Questions from Top Companies)")
_PROMO = re.compile(r"^(?:Unlock interview details and practice with AI|Curated Interview Questions from Top Companies|"
                    r"VIP即刻解锁阅读权限.*|使用VIP即刻解锁.*)$", re.M)
# Judged from the page's structure, never from its text: posts talk about logins and verification codes too.
_BLOCKED_URL = re.compile(r"member\.php\?mod=logging|/login|captcha|/verify", re.I)
_CHALLENGE_TITLE = re.compile(r"Just a moment|Attention Required|安全验证|Verify", re.I)
LOGGED_IN = "a[href*='mod=spacecp']"          # the account's settings link, only shown to a logged-in user
PROMPT = "#messagetext"                         # Discuz's box for "please log in", "no permission" and the like
# Prompts about one thread (a read-permission level it needs, deleted, hidden): skip it, the run goes on.
_THREAD_ONLY = re.compile(r"本帖要求阅读权限|阅读权限高于|指定的主题不存在|已被删除|主题.*屏蔽|仅作者可见|需要购买|该帖被")


class Blocked(RuntimeError):
    """A login page, CAPTCHA or permission prompt: a person has to look at it."""


def parse_when(s: str, now: datetime | None = None) -> str | None:
    """Discuz times: "16 分钟前", "2 小时前", "昨天 23:02", "前天 08:10", "3 天前", "2026-10-30 21:15", "2026-10-30"."""
    now = now or datetime.now().astimezone()
    s = (s or "").strip()
    m = re.search(r"(\d+)\s*(秒|分钟|小时|天)前", s)
    if m:
        unit = {"秒": "seconds", "分钟": "minutes", "小时": "hours", "天": "days"}[m.group(2)]
        return (now - timedelta(**{unit: int(m.group(1))})).isoformat(timespec="seconds")
    m = re.search(r"(昨天|前天)\s*(\d{1,2}):(\d{2})", s)
    if m:
        day = now - timedelta(days=1 if m.group(1) == "昨天" else 2)
        return day.replace(hour=int(m.group(2)), minute=int(m.group(3)), second=0, microsecond=0).isoformat()
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2}):(\d{2}))?", s)
    if m:
        y, mo, d = (int(x) for x in m.group(1, 2, 3))
        hh, mm = int(m.group(4) or 0), int(m.group(5) or 0)
        return now.replace(year=y, month=mo, day=d, hour=hh, minute=mm, second=0, microsecond=0).isoformat()
    return None


def visible_text(post: str) -> tuple[str, bool]:
    """The post without the "hidden content" notices; True when part of it was hidden."""
    hidden = "本帖隐藏的内容" in post
    t = _PROMO.sub("", _HIDDEN.sub(f"\n{HIDDEN_MARK}\n", post))
    t = re.sub(rf"(?:{re.escape(HIDDEN_MARK)}\s*)+", HIDDEN_MARK + "\n\n", t)   # the site shows the notice twice
    return re.sub(r"\n{3,}", "\n\n", t).strip(), hidden


class OnePoint3Acres:
    def __init__(self, name: str, cfg: dict, http: HttpClient):
        self.name, self.cfg, self.http = name, cfg, http
        profile = section("onepoint3acres").get("chrome_profile")
        self.profile = Path(profile).expanduser() if profile else None
        self.include = re.compile(cfg["include_pattern"]) if cfg.get("include_pattern") else None
        self.session: BrowserSession | None = None
        self.blocked = False
        self._last_page = 0.0

    # ---- browser ------------------------------------------------------------------------

    def _pause(self) -> None:
        lo, hi = self.cfg.get("page_delay_s", [10, 20])
        wait = self._last_page + random.uniform(float(lo), float(hi)) - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _open(self, url: str, *, thread: bool = False) -> None:
        self._pause()
        try:
            r = self.session.goto(url)
        finally:
            self._last_page = time.monotonic()
        page = self.session.page
        why = None
        if r is not None and r.status >= 400:
            why = f"HTTP {r.status}"
        elif _BLOCKED_URL.search(page.url):
            why = "redirected to a login or verification page"
        elif _CHALLENGE_TITLE.search(page.title() or ""):
            why = f"browser check ({page.title()!r})"
        elif page.query_selector(PROMPT) is not None:
            prompt = page.inner_text(PROMPT).strip()
            if thread and _THREAD_ONLY.search(prompt):
                raise SkipDoc(prompt[:120])        # this thread only: the run goes on, the thread is not retried
            why = f"site prompt: {prompt[:120]!r}"
        elif page.query_selector(LOGGED_IN) is None:
            why = "not logged in any more"
        if why:
            raise Blocked(f"{page.url}: {why}")

    def _blocked(self, e: Blocked) -> None:
        notify.alert(f"source:{self.name}", "1Point3Acres needs attention",
                     f"{e}. Open the profile by hand (same flags, --password-store=basic) and check; "
                     f"the connector never logs in or answers checks itself.")

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None
        self.http.close()

    # ---- connector ----------------------------------------------------------------------

    def list_updates(self, state: dict) -> ListResult:
        if self.profile is None:
            return ListResult([], dict(state), skipped="no [onepoint3acres] chrome_profile configured")
        try:
            self.session = BrowserSession(self.profile, page_timeout_s=float(self.cfg.get("page_timeout_s", 60)),
                                          session_limit_s=float(self.cfg.get("session_limit_s", 900))).open()
        except ProfileInUse as e:
            return ListResult([], dict(state), skipped=f"browser profile in use: {e}")
        refs, errors, seen = [], [], set()
        fid = int(self.cfg.get("forum_id", 145))
        try:
            for page in range(1, int(self.cfg.get("max_pages", 2)) + 1):
                self._open(f"{BASE}/forum.php?mod=forumdisplay&fid={fid}&orderby=dateline&filter=author&page={page}")
                rows = self.session.page.eval_on_selector_all("tbody[id^='normalthread_']", """els => els.map(e => {
                    const a = e.querySelector('a.s.xst') || e.querySelector('th a[href*="thread-"]');
                    const by = e.querySelector('td.by');
                    const th = e.querySelector('th');
                    return {id: e.id.replace('normalthread_', ''), href: a && a.href, title: a && a.innerText.trim(),
                            by: by ? by.innerText.replace(/\\s+/g, ' ').trim() : '',
                            line: th ? th.innerText.replace(/\\s+/g, ' ').trim() : ''};
                })""")
                for r in rows:
                    if not r["href"] or r["id"] in seen or (self.include and not self.include.search(r["line"])):
                        continue
                    seen.add(r["id"])
                    labels = r["line"].replace(r["title"], "", 1).strip()
                    refs.append(DocRef(source=self.name, external_id=r["id"], url=f"{BASE}/thread-{r['id']}-1-1.html",
                                       title=r["title"], version="1", published_at=parse_when(r["by"]),
                                       meta={"labels": labels, "author": r["by"].split(" ")[0] if r["by"] else None}))
        except Blocked as e:
            self.blocked = True
            self._blocked(e)
            errors.append(str(e))
        except Exception as e:
            errors.append(f"list: {type(e).__name__}: {str(e)[:200]}")
        if refs or not errors:
            notify.resolve(f"source:{self.name}", "1Point3Acres readable again")
        return ListResult(refs, dict(state), errors)

    def fetch_doc(self, ref: DocRef, validators: Validators | None) -> RawDoc:
        if self.blocked:                       # one block stops the run: no more pages until a person looked
            raise Blocked("stopped after an earlier block in this run")
        try:
            self._open(ref.url, thread=True)
        except Blocked as e:
            self.blocked = True
            self._blocked(e)
            raise
        posts = self.session.page.eval_on_selector_all("td.t_f, div.t_f", "els => els.map(e => e.innerText.trim())")
        if not posts:
            raise LookupError(f"thread {ref.external_id}: no post text on the page")
        text, partial = visible_text(posts[0])
        labels = ref.meta.get("labels") or ""
        content = f"# {ref.title}\n\n{labels}\n\n{text}" if labels else f"# {ref.title}\n\n{text}"
        return RawDoc(ref=ref, content=content, content_type="markdown", kind="experience", origin="1point3acres",
                      title=ref.title, author=ref.meta.get("author"), published_at=ref.published_at,
                      hints={"partial": True} if partial else {})

    def finalize(self, state: dict, failed: list[DocRef]) -> dict:
        return state                       # posts not fetched are simply listed again while they are recent
