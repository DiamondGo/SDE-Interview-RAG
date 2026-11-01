"""Reddit posts about interviews: each subreddit's search RSS, newest first (the JSON API answers 403).

The feed carries the post's text, so a post costs no request beyond the listing. Comments are not read.
A link post's target is its outbound URL: content another source already has is counted, not fetched.
"""

import re
from urllib.parse import quote_plus

from sde_rag.http import HttpClient, Validators
from sde_rag.parse.clean import fragment_to_markdown
from sde_rag.sources.base import DocRef, ListResult, RawDoc
from sde_rag.sources.feeds import parse_feed

_LINK = re.compile(r'<a href="([^"]+)">\[link\]</a>')


class Reddit:
    def __init__(self, name: str, cfg: dict, http: HttpClient):
        self.name, self.cfg, self.http = name, cfg, http

    def list_updates(self, state: dict) -> ListResult:
        refs, errors, seen = [], [], set()
        query = self.cfg.get("query", "title:interview")
        for sub in self.cfg.get("subreddits", ["cscareerquestions"]):
            url = (f"https://www.reddit.com/r/{sub}/search.rss?q={quote_plus(query)}&restrict_sr=1&sort=new"
                   f"&t={self.cfg.get('window', 'month')}&limit={int(self.cfg.get('per_feed', 100))}")
            try:
                entries = parse_feed(self.http.get(url).text)
            except Exception as e:
                errors.append(f"r/{sub}: {type(e).__name__}: {str(e)[:200]}")
                continue
            for e in entries:
                pid = e.id.rsplit("/", 1)[-1]
                if pid in seen:
                    continue
                seen.add(pid)
                link = _LINK.search(e.content)
                outbound = link.group(1) if link and "reddit.com" not in link.group(1) else None
                refs.append(DocRef(source=self.name, external_id=pid, url=e.link, title=e.title,
                                   version=e.updated or e.published, published_at=e.published or e.updated,
                                   outbound_url=outbound,
                                   meta={"sub": sub, "html": e.content, "author": e.author}))
        return ListResult(refs, dict(state), errors)

    def fetch_doc(self, ref: DocRef, validators: Validators | None) -> RawDoc:
        html = ref.meta.get("html") or ""
        cut = html.rfind("submitted by")               # the feed's footer: author, [link], [comments]
        body = fragment_to_markdown(html[:cut] if cut >= 0 else html)
        return RawDoc(ref=ref, content=f"# {ref.title}\n\n{body}", content_type="markdown", kind="experience",
                      origin=f"reddit/r/{ref.meta['sub']}", title=ref.title,
                      author=(ref.meta.get("author") or "").removeprefix("/u/") or None,
                      published_at=ref.published_at)

    def finalize(self, state: dict, failed: list[DocRef]) -> dict:
        return state                          # nothing to rewind: the feed is read in full every run
