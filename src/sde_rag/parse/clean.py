"""Raw content -> cleaned Markdown. Keeps headings, lists, code blocks, tables and <details> Q&A blocks."""

import re
import unicodedata

import trafilatura

CLEANER_VERSION = 1

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_BADGE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_IMG = re.compile(r"<img\b[^>]*>", re.I)
_LIQUID = re.compile(r"{%.*?%}", re.S)                  # dev.to embeds
_BACK_TO_TOP = re.compile(r"^.*\bback to top\b.*$", re.I | re.M)
_BLANKS = re.compile(r"\n{3,}")


def clean_markdown(md: str) -> str:
    t = unicodedata.normalize("NFC", md.replace("\r\n", "\n").replace("\r", "\n"))
    t = _FRONT_MATTER.sub("", t)
    t = _HTML_COMMENT.sub("", t)
    t = _BADGE.sub("", t)
    t = _IMAGE.sub("", t)
    t = _HTML_IMG.sub("", t)
    t = _LIQUID.sub("", t)
    t = _BACK_TO_TOP.sub("", t)
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    return _BLANKS.sub("\n\n", t).strip()


def html_to_markdown(html: str, url: str | None = None) -> str:
    out = trafilatura.extract(html, url=url, output_format="markdown", include_tables=True,
                              include_links=False, include_images=False, include_comments=False,
                              favor_recall=True) or ""
    return clean_markdown(out)


def clean(content: str, content_type: str, url: str | None = None) -> str:
    return html_to_markdown(content, url) if content_type == "html" else clean_markdown(content)
