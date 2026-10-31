"""Hacker News stories about interviews (Algolia API), with their top comments and the linked article.

Stories are taken once they are `settle_hours` old, so points and comments have mostly settled; each
story is then fetched once (version "1"). The high-water mark only advances past stories that were fetched.
"""

import html
import re
import time

from sde_rag.http import HttpClient, RobotsDisallowed, Validators, normalize_url
from sde_rag.parse.clean import html_to_markdown
from sde_rag.sources.base import DocRef, ListResult, RawDoc, iso

API = "https://hn.algolia.com/api/v1"
_TAG = re.compile(r"<[^>]+>")


def comment_to_text(h: str) -> str:
    t = re.sub(r"<pre><code>(.*?)</code></pre>", lambda m: "\n```\n" + m.group(1) + "\n```\n", h or "", flags=re.S)
    t = re.sub(r"</?p>", "\n\n", t)
    t = re.sub(r"<i>(.*?)</i>", r"*\1*", t, flags=re.S)
    return html.unescape(_TAG.sub("", t)).strip()


class HackerNews:
    def __init__(self, name: str, cfg: dict, http: HttpClient):
        self.name = name
        self.cfg = cfg
        self.http = http
        self.title_re = re.compile(cfg.get("title_pattern", "(?i)interview"))
        self.title_re2 = re.compile(cfg["title_pattern_also"]) if cfg.get("title_pattern_also") else None

    def _relevant(self, title: str) -> bool:
        return bool(self.title_re.search(title)) and (self.title_re2 is None or bool(self.title_re2.search(title)))

    def list_updates(self, state: dict) -> ListResult:
        now = time.time()
        hw = float(state.get("hw") or now - float(self.cfg.get("backfill_days", 365)) * 86400)
        start = hw - float(self.cfg.get("overlap_hours", 24)) * 3600
        end = now - float(self.cfg.get("settle_hours", 48)) * 3600
        if end <= start:
            return ListResult([], dict(state))
        hits: dict[str, dict] = {}
        errors = []
        for q in self.cfg.get("queries", ["interview"]):
            for page in range(10):
                try:
                    r = self.http.get(f"{API}/search_by_date", params={
                        "query": q, "tags": "story", "hitsPerPage": 100, "page": page,
                        "numericFilters": f"created_at_i>{int(start)},created_at_i<{int(end)},"
                                          f"points>={int(self.cfg.get('min_points', 10))}"}).json()
                except Exception as e:
                    errors.append(f"search {q!r} page {page}: {type(e).__name__}: {str(e)[:200]}")
                    break
                for h in r.get("hits", []):
                    if h.get("title") and self._relevant(h["title"]):
                        hits[h["objectID"]] = h
                if page + 1 >= r.get("nbPages", 0):
                    break
        ordered = sorted(hits.values(), key=lambda h: h["created_at_i"])
        limit = int(self.cfg.get("per_run_limit", 150))
        taken = ordered[:limit]
        new_hw = end if len(ordered) <= limit else float(taken[-1]["created_at_i"])
        refs = [DocRef(source=self.name, external_id=h["objectID"],
                       url=f"https://news.ycombinator.com/item?id={h['objectID']}", title=h["title"],
                       version="1", published_at=iso(h["created_at_i"]),
                       outbound_url=h.get("url") or None,
                       meta={"points": h.get("points"), "num_comments": h.get("num_comments"),
                             "created_at_i": h["created_at_i"]})
                for h in taken]
        if errors and not refs:
            new_hw = hw  # nothing listed: do not skip this window
        return ListResult(refs, {**state, "hw": new_hw}, errors)

    def fetch_doc(self, ref: DocRef, validators: Validators | None) -> RawDoc:
        item = self.http.get(f"{API}/items/{ref.external_id}").json()
        parts = [f"# {ref.title}"]
        if item.get("text"):
            parts.append(comment_to_text(item["text"]))
        if ref.outbound_url and self.cfg.get("fetch_articles", True) and not ref.meta.get("outbound_known"):
            try:
                page = self.http.get(ref.outbound_url, robots=True)
                if "html" in page.headers.get("content-type", ""):
                    article = html_to_markdown(page.text, ref.outbound_url)
                    if article:
                        parts.append(f"## Linked article\n\nSource: {normalize_url(ref.outbound_url)}\n\n"
                                     f"{article[:12000]}")
            except (RobotsDisallowed, Exception):
                pass  # the discussion is still worth keeping without the article
        comments, total = [], 0
        budget = int(self.cfg.get("comment_chars", 15000))
        for c in (item.get("children") or [])[: int(self.cfg.get("max_comments", 40))]:
            text = comment_to_text(c.get("text") or "")
            if not text or total + len(text) > budget:
                continue
            total += len(text)
            comments.append(f"- **{c.get('author') or 'anon'}**: {text}")
        if comments:
            parts.append("## Discussion\n\n" + "\n\n".join(comments))
        return RawDoc(ref=ref, content="\n\n".join(parts), content_type="markdown", kind="experience",
                      origin="hn", title=ref.title, author=item.get("author"), published_at=ref.published_at)

    def finalize(self, state: dict, failed: list[DocRef]) -> dict:
        if failed:
            earliest = min(float(r.meta["created_at_i"]) for r in failed)
            state["hw"] = min(float(state.get("hw") or earliest), earliest)
        return state
