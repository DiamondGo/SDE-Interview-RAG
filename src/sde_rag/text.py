import hashlib
import html
import re
import unicodedata

_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_EMPHASIS = re.compile(r"(?<![\w*])[_*]([^_*\n]+?)[_*](?![\w*])")   # _italic_ / *italic*, not snake_case
_LEADING_MARKERS = re.compile(
    r"^(?:[\W_]*?)(?:(?:q(?:uestion)?\s*\.?\s*\d*\s*[:.)\-]|\d+\s*[.):](?=\s)|\[[a-z]\])\s*)+", re.I)
_EMOJI_SHORTCODE = re.compile(r":[a-z0-9_+\-]+:")
_NON_WORD = re.compile(r"[^\w\s+#]")
_SPACE = re.compile(r"\s+")


def sha1(text: str | bytes) -> str:
    return hashlib.sha1(text.encode() if isinstance(text, str) else text).hexdigest()


def strip_markup(text: str) -> str:
    """Markdown/HTML inline markup -> plain text (links keep their label)."""
    t = _MD_LINK.sub(r"\1", text)
    t = _HTML_TAG.sub(" ", t)
    t = html.unescape(t)
    t = t.replace("**", "").replace("__", "").replace("`", "")
    t = _EMPHASIS.sub(r"\1", t)
    return _SPACE.sub(" ", t).strip()


def clean_question(text: str) -> str:
    """A heading or list item -> the question text: no markup, emoji, "Q1:", "12." or "[E]" prefixes."""
    t = _EMOJI_SHORTCODE.sub(" ", strip_markup(text))
    t = "".join(ch for ch in t if unicodedata.category(ch)[0] != "S" or ch in "+#<>=^|~$%&*")
    t = _SPACE.sub(" ", t).strip()
    t = _LEADING_MARKERS.sub("", t).strip()
    t = re.sub(r"^\*+|\*+$", "", t).strip()
    return t


def normalize_question(text: str) -> str:
    t = unicodedata.normalize("NFKC", clean_question(text)).lower()
    t = _NON_WORD.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def question_hash(text: str) -> str:
    return sha1(normalize_question(text))


def detect_lang(text: str) -> str:
    sample = text[:4000]
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in sample if ch.isalpha())
    return "zh" if letters and cjk / letters > 0.2 else "en"


def truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"
