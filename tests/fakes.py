from sde_rag.http import NotModified
from sde_rag.sources.base import DocRef, ListResult, RawDoc

PREP_MD = """# Python Interview Questions
## 1. What is the GIL in Python?
The global interpreter lock lets one thread run Python bytecode at a time; use multiprocessing for CPU-bound work.
## 2. How do Python generators work?
A generator function yields values lazily and keeps its frame between calls.
## 3. Explain decorators in Python.
A decorator wraps a function to add behaviour, for example caching or logging.
"""

GO_MD = """# Go Questions
## What is a goroutine?
A lightweight thread managed by the Go runtime, started with the go keyword.
## How do goroutines communicate?
Through channels: typed conduits that synchronise send and receive.
## What is the GIL in Python?
Asked here too, to test merging across sources.
"""

EXPERIENCE_MD = """# My onsite at a big tech company

The loop had four rounds. The first was a coding round about an LRU cache and its eviction policy; the
interviewer asked about the time complexity of get and put and how to make it thread-safe.

The system design round asked me to design a rate limiter for a public API, with follow-ups on sliding windows,
Redis, and what happens when the limiter itself is down.

The behavioral round was about a conflict with a coworker. Overall, practise talking while you code.
"""


class FakeHttp:
    def close(self):
        pass


class FakeConnector:
    """docs: external_id -> {version, content, kind, origin, outbound_url, fail, not_modified, hints}."""

    def __init__(self, name: str, docs: dict):
        self.name, self.docs, self.http = name, docs, FakeHttp()
        self.fetched: list[str] = []
        self.finalized_failed: list[str] = []

    def list_updates(self, state: dict) -> ListResult:
        refs = [DocRef(source=self.name, external_id=eid, url=d.get("url", f"https://example.com/{self.name}/{eid}"),
                       title=d.get("title", eid), version=d.get("version"), outbound_url=d.get("outbound_url"))
                for eid, d in self.docs.items()]
        return ListResult(refs, {"runs": state.get("runs", 0) + 1})

    def fetch_doc(self, ref: DocRef, validators) -> RawDoc:
        d = self.docs[ref.external_id]
        self.fetched.append(ref.external_id)
        if d.get("fail"):
            raise RuntimeError("fetch failed")
        if d.get("not_modified"):
            raise NotModified(ref.url)
        return RawDoc(ref=ref, content=d["content"], content_type="markdown", kind=d.get("kind", "prep"),
                      origin=d.get("origin", self.name), title=ref.title, hints=d.get("hints", {}),
                      etag=d.get("etag"))

    def finalize(self, state: dict, failed: list[DocRef]) -> dict:
        self.finalized_failed = [r.external_id for r in failed]
        return state
