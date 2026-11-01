"""Raw content -> cleaned Markdown. Keeps headings, lists, code blocks, tables and <details> Q&A blocks."""

import html
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


_FRAG_RULES = [
    (re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I), ""),
    (re.compile(r"<pre\b[^>]*>(.*?)</pre>", re.S | re.I),
     lambda m: "\n\n```\n" + re.sub(r"<[^>]+>", "", re.sub(r"<br\s*/?>", "\n", m.group(1))) + "\n```\n\n"),
    (re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.S | re.I), lambda m: f"\n\n{'#' * int(m.group(1))} {m.group(2)}\n\n"),
    (re.compile(r"<li\b[^>]*>", re.I), "\n- "),
    (re.compile(r"<br\s*/?>", re.I), "\n"),
    (re.compile(r"</?(p|div|ul|ol|blockquote|figure|section|table|tr)\b[^>]*>", re.I), "\n\n"),
    (re.compile(r"</t[dh]>", re.I), " | "),
    (re.compile(r"<code\b[^>]*>(.*?)</code>", re.S | re.I), r"`\1`"),
    (re.compile(r"<(strong|b)\b[^>]*>(.*?)</\1>", re.S | re.I), r"**\2**"),
    (re.compile(r"<[^>]+>"), ""),
]


def fragment_to_markdown(fragment: str) -> str:
    """An HTML fragment from a feed (no page around it, so nothing for trafilatura to strip) -> Markdown."""
    t = fragment
    for pattern, repl in _FRAG_RULES:
        t = pattern.sub(repl, t)
    t = html.unescape(t).replace("\xa0", " ")
    t = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in t.split("\n"))
    return clean_markdown(t)


def clean(content: str, content_type: str, url: str | None = None) -> str:
    return html_to_markdown(content, url) if content_type == "html" else clean_markdown(content)
