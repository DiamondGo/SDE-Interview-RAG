import difflib
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


# Words, and CJK characters one by one (Chinese has no spaces between words).
_TOKEN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]|[^\W_]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", strip_markup(text)).lower())


def quote_similarity(quote: str, text: str, *, max_tokens: int = 60) -> float:
    """How well `quote` matches some passage of `text` (1.0 = verbatim), ignoring markup, case and punctuation.

    Windows of the quote's length are compared wherever one of the quote's first or last tokens occurs."""
    q = tokens(quote)[:max_tokens]
    if not q:
        return 0.0
    t = tokens(text)
    if f" {' '.join(q)} " in f" {' '.join(t)} ":
        return 1.0
    anchors = {}
    for j in list(range(min(3, len(q)))) + list(range(max(0, len(q) - 3), len(q))):
        anchors.setdefault(q[j], j)
    best, n, tried = 0.0, len(q), set()
    for i, tok in enumerate(t):
        j = anchors.get(tok)
        if j is None or (start := max(0, i - j)) in tried:
            continue
        tried.add(start)
        best = max(best, difflib.SequenceMatcher(None, q, t[start:start + n], autojunk=False).ratio())
        if best >= 0.999:
            break
    return best


def simhash(text: str, *, shingle: int = 3) -> int:
    """64-bit SimHash over word 3-grams, as a signed int (SQLite INTEGER). Reposts and light edits land a few
    bits apart; unrelated texts about 32."""
    toks = tokens(text)
    grams = [" ".join(toks[i:i + shingle]) for i in range(max(1, len(toks) - shingle + 1))]
    acc = [0] * 64
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
        for b in range(64):
            acc[b] += 1 if h >> b & 1 else -1
    v = sum(1 << b for b in range(64) if acc[b] > 0)
    return v - (1 << 64) if v >= 1 << 63 else v


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & ((1 << 64) - 1)).bit_count()


def truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"
