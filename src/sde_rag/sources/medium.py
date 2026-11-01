"""Medium posts by tag.

Tag feeds carry only a one-line snippet, and article pages answer plain requests with 403. The author's
feed (medium.com/feed/@user, or user.medium.com/feed) carries the full text of their latest posts, so a
post found in a tag feed is read from its author's feed, each feed fetched at most once a run. Publication
feeds often carry only snippets; such posts keep the snippet and the cleaning stage skips them as too short.
"""

from urllib.parse import urlsplit, urlunsplit

from sde_rag.http import HttpClient, Validators
from sde_rag.parse.clean import fragment_to_markdown
from sde_rag.sources.base import DocRef, ListResult, RawDoc
from sde_rag.sources.feeds import Entry, parse_feed


def author_feed(link: str) -> str:
    parts = urlsplit(link)
    host, path = parts.hostname or "", [p for p in parts.path.split("/") if p]
    if host in ("medium.com", "www.medium.com"):
        return f"https://medium.com/feed/{path[0]}" if path else "https://medium.com/feed"
    return f"https://{host}/feed"            # user.medium.com, or a custom domain


def _clean_link(link: str) -> str:
    p = urlsplit(link)
    return urlunsplit((p.scheme, p.netloc, p.path, "", ""))


class Medium:
    def __init__(self, name: str, cfg: dict, http: HttpClient):
        self.name, self.cfg, self.http = name, cfg, http
        self._feeds: dict[str, dict[str, Entry]] = {}

    def list_updates(self, state: dict) -> ListResult:
        refs, errors, seen = [], [], set()
        for tag in self.cfg.get("tags", ["interview-experience"]):
            try:
                entries = parse_feed(self.http.get(f"https://medium.com/feed/tag/{tag}").text)
            except Exception as e:
                errors.append(f"tag {tag}: {type(e).__name__}: {str(e)[:200]}")
                continue
            for e in entries:
                pid = e.id.rsplit("/", 1)[-1]
                if pid in seen:
                    continue
                seen.add(pid)
                link = _clean_link(e.link)
                refs.append(DocRef(source=self.name, external_id=pid, url=link, title=e.title,
                                   version=e.updated or e.published, published_at=e.published,
                                   meta={"guid": e.id, "feed": author_feed(link), "snippet": e.summary,
                                         "author": e.author, "tags": e.categories}))
        return ListResult(refs, dict(state), errors)

    def _author_entries(self, feed: str) -> dict[str, Entry]:
        if feed not in self._feeds:
            self._feeds[feed] = {e.id: e for e in parse_feed(self.http.get(feed).text)}
        return self._feeds[feed]

    def fetch_doc(self, ref: DocRef, validators: Validators | None) -> RawDoc:
        entry = self._author_entries(ref.meta["feed"]).get(ref.meta["guid"])
        html = entry.content if entry and entry.content else ref.meta.get("snippet") or ""
        body = fragment_to_markdown(html)
        return RawDoc(ref=ref, content=f"# {ref.title}\n\n{body}", content_type="markdown", kind="experience",
                      origin="medium", title=ref.title, author=ref.meta.get("author"),
                      published_at=ref.published_at, canonical_url=ref.url)

    def finalize(self, state: dict, failed: list[DocRef]) -> dict:
        return state
