"""RSS 2.0 and Atom feeds -> plain entries (standard library XML parser)."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

from sde_rag.sources.base import iso

ATOM = "{http://www.w3.org/2005/Atom}"
CONTENT = "{http://purl.org/rss/1.0/modules/content/}encoded"
DC_CREATOR = "{http://purl.org/dc/elements/1.1/}creator"


@dataclass
class Entry:
    id: str
    title: str
    link: str
    published: str | None
    updated: str | None
    content: str                    # HTML: the full text when the feed carries it, else ""
    summary: str = ""
    author: str | None = None
    categories: list[str] = field(default_factory=list)


def _date(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    try:
        return iso(s)
    except ValueError:
        try:
            return parsedate_to_datetime(s).isoformat(timespec="seconds")
        except (TypeError, ValueError):
            return None


def parse_feed(xml: str) -> list[Entry]:
    root = ET.fromstring(xml)
    out = []
    for it in root.iter("item"):                                   # RSS 2.0
        out.append(Entry(
            id=(it.findtext("guid") or it.findtext("link") or "").strip(),
            title=(it.findtext("title") or "").strip(), link=(it.findtext("link") or "").strip(),
            published=_date(it.findtext("pubDate")), updated=_date(it.findtext(f"{ATOM}updated")),
            content=it.findtext(CONTENT) or "", summary=it.findtext("description") or "",
            author=it.findtext(DC_CREATOR) or it.findtext("author"),
            categories=[c.text.strip() for c in it.findall("category") if c.text]))
    for e in root.iter(f"{ATOM}entry"):                            # Atom
        link = next((ln.get("href") for ln in e.findall(f"{ATOM}link") if ln.get("rel", "alternate") == "alternate"),
                    None) or ""
        out.append(Entry(
            id=(e.findtext(f"{ATOM}id") or link).strip(), title=(e.findtext(f"{ATOM}title") or "").strip(),
            link=link, published=_date(e.findtext(f"{ATOM}published")), updated=_date(e.findtext(f"{ATOM}updated")),
            content=e.findtext(f"{ATOM}content") or "", summary=e.findtext(f"{ATOM}summary") or "",
            author=e.findtext(f"{ATOM}author/{ATOM}name"),
            categories=[c.get("term") or c.get("label") or "" for c in e.findall(f"{ATOM}category")]))
    return out
