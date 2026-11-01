"""Connector interface. A connector lists what changed since its saved state, then fetches single documents.

The pipeline (sde_rag.pipeline.fetch) decides what to download: anything whose `version` matches what was
already seen is skipped without a request, links to content another source already has are recorded as
citations, and validators (ETag / Last-Modified) are sent for conditional requests.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from sde_rag.http import Validators


@dataclass
class DocRef:
    source: str
    external_id: str
    url: str
    title: str
    version: str | None = None           # equal version => unchanged, no download
    published_at: str | None = None
    outbound_url: str | None = None      # a link post's target article
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawDoc:
    ref: DocRef
    content: str
    content_type: str                    # markdown | html
    kind: str                            # prep | experience
    origin: str                          # e.g. "owner/repo", "hn", "devto"
    title: str = ""
    author: str | None = None
    published_at: str | None = None
    canonical_url: str | None = None
    hints: dict[str, Any] = field(default_factory=dict)
    etag: str | None = None
    last_modified: str | None = None


class SkipDoc(Exception):
    """This one document cannot be read (a permission it needs, deleted): remembered as seen, never retried,
    and not a failure of the source."""


@dataclass
class ListResult:
    refs: list[DocRef]
    state: dict
    errors: list[str] = field(default_factory=list)
    skipped: str | None = None           # why this run could not look (e.g. the browser profile is in use):
                                         # neither a success nor a failure, try again next run


class Connector(Protocol):
    name: str

    def list_updates(self, state: dict) -> ListResult: ...

    def fetch_doc(self, ref: DocRef, validators: Validators | None) -> RawDoc: ...

    def finalize(self, state: dict, failed: list[DocRef]) -> dict:
        """Adjust the new state when some documents failed (or were left for the next run by the per-run
        download cap), so they are listed again next time."""
        ...

    # Optional: def close(self) -> None, for connectors holding more than `http` (a browser).


def iso(ts: float | int | str | None) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat(timespec="seconds")
    from datetime import timezone
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat(timespec="seconds")
