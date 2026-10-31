"""Rule-based question/answer extraction from interview-question Markdown.

Recognised layouts (all seen in popular question repos):
  1. A heading that is a question; its answer runs until the next heading of the same or a higher level
     ("## 1. What is X?", "## Q. ...", "## 📌 Q1: ...", sub-headings such as "### ✅ Answer" belong to the answer).
  2. The same inside a numbered list ("1. ### What is X" with an indented answer).
  3. <details><summary>Question</summary> answer </details>.
  4. List items that are questions, without answers ("* What is CSS specificity?").
Headings that are not questions become the section path of the questions below them.
Lines inside code fences are never treated as structure.
"""

import re
import textwrap
from dataclasses import dataclass

from sde_rag.text import clean_question, strip_markup

PARSER_VERSION = "rules-v1"

# At most 3 spaces of indentation: deeper "# ..." lines are comments inside indented code blocks.
_HEADING = re.compile(r"^ {0,3}(?:\d+\.\s+)?(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# "73. ### What ..." never occurs inside real code; seeing it means an earlier fence was left unclosed.
_NUMBERED_HEADING = re.compile(r"^\d+\.\s+#{1,6}\s+\S")
_INTERROGATIVE = re.compile(
    r"^(what|what's|whats|why|how|when|where|which|who|whom|whose|is|are|was|were|do|does|did|can|could|"
    r"should|would|will|shall|may|have|has)\b", re.I)
# Verbs that only start questions ("Explain ...").
_IMPERATIVE = re.compile(
    r"^(explain|describe|define|compare|contrast|differentiate|distinguish|discuss|elaborate|summarize|"
    r"illustrate|demonstrate|outline|walk me through|talk about|tell me|tell us)\b", re.I)
# Verbs that also start section titles ("Build, test and run", "State management"): a question only when
# followed by a determiner, or for "design" by a product name ("Design Pastebin.com", "Design Amazon's ...").
_WEAK_IMPERATIVE = re.compile(
    r"^(list|name|give|find|write|implement|design|build|create|calculate|estimate|identify|suggest|propose|"
    r"mention|state|sort|reverse|merge|convert|count|check|print|return|solve)\s+"
    r"(a|an|the|all|some|any|me|us|your|two|three|four|five|\d+|out|down)\b", re.I)
_DESIGN_PRODUCT = re.compile(r"^design\s+\S*(\.\w{2,}|'s)\b", re.I)
_ANSWER_LABEL = re.compile(r"^\s*#{1,6}\s*\W*\s*(answer|solution|explanation)s?\s*:?\s*$", re.I)
_META = re.compile(
    r"\b(contribut\w*|license|table of contents|installation|getting started|sponsor\w*|acknowledg\w*|"
    r"disclaimer|translations?|how to use this|support this|star this|follow me|subscribe|newsletter|"
    r"download|buy me)\b", re.I)
_TAGS = re.compile(r"</?(?:b|strong|i|em|br|p|code)\s*/?>", re.I)


@dataclass
class QA:
    question: str
    answer: str
    section_path: str
    ordinal: int


def is_question(text: str) -> bool:
    t = text.strip()
    words = t.split()
    if len(t) < 8 or len(t) > 400 or len(words) < 2:
        return False
    if _META.search(t):
        return False
    if t.endswith("?"):
        return True
    if len(words) >= 3 and (_INTERROGATIVE.match(t) or _IMPERATIVE.match(t) or _WEAK_IMPERATIVE.match(t)
                            or _DESIGN_PRODUCT.match(t)):
        return True
    return bool(re.search(r"\bdifferences? between\b", t, re.I))


def _clean_answer(lines: list[str]) -> str:
    text = textwrap.dedent("\n".join(l for l in lines if not _ANSWER_LABEL.match(l)))
    text = _TAGS.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class _Parser:
    def __init__(self):
        self.out: list[QA] = []
        self.sections: list[tuple[int, str]] = []
        self.cur: dict | None = None   # {"q", "kind": heading|details, "level", "lines", "section"}
        self.title_questions = True

    def section_path(self) -> str:
        return " > ".join(t for _, t in self.sections[-3:])

    def close(self):
        if self.cur:
            q = self.cur["q"]
            self.out.append(QA(q, _clean_answer(self.cur["lines"])[:8000], self.cur["section"], len(self.out)))
        self.cur = None

    def emit_bare(self, q: str):
        self.out.append(QA(q, "", self.section_path(), len(self.out)))

    def heading(self, level: int, title: str):
        self.close()
        # Sections at this level or deeper end here, whether the new heading is a question or a section.
        while self.sections and self.sections[-1][0] >= level:
            self.sections.pop()
        q = clean_question(title)
        if is_question(q) and (self.title_questions or level > 1):
            self.cur = {"q": q, "kind": "heading", "level": level, "lines": [], "section": self.section_path()}
            return
        name = strip_markup(title)
        if name:
            self.sections.append((level, clean_question(name) or name))


def parse_markdown_qa(md: str, *, bare_list_questions: bool = True, title_questions: bool = True) -> list[QA]:
    """`bare_list_questions=False` ignores question-like list items (discussion threads are full of them);
    `title_questions=False` never takes the level-1 heading (a post title) as a question."""
    p = _Parser()
    p.title_questions = title_questions
    lines = md.split("\n")
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if _FENCE.match(line):
            in_fence = not in_fence
            if p.cur:
                p.cur["lines"].append(line)
            continue
        if in_fence and _NUMBERED_HEADING.match(line):
            in_fence = False
        if in_fence:
            if p.cur:
                p.cur["lines"].append(line)
            continue

        cur = p.cur
        if cur and cur["kind"] == "details":
            if "</details>" in line.lower():
                before = line[:line.lower().index("</details>")]
                if before.strip():
                    cur["lines"].append(before)
                p.close()
            else:
                cur["lines"].append(line)
            continue

        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            if cur and cur["kind"] == "heading" and level > cur["level"]:
                cur["lines"].append(line)
            else:
                p.heading(level, m.group(2))
            continue

        low = line.lower()
        if "<summary" in low and not (cur and cur["kind"] == "heading"):
            # The summary may span several lines.
            buf = line
            while "</summary>" not in buf.lower() and i < len(lines):
                buf += "\n" + lines[i]
                i += 1
            start = buf.lower().index("<summary")
            start = buf.index(">", start) + 1
            end = buf.lower().find("</summary>")
            summary = buf[start:end if end >= 0 else len(buf)]
            rest = buf[end + len("</summary>"):] if end >= 0 else ""
            q = clean_question(summary)
            p.close()
            if is_question(q):
                p.cur = {"q": q, "kind": "details", "level": 99, "lines": [rest] if rest.strip() else [],
                         "section": p.section_path()}
            else:
                # Not a question (e.g. "Show answer"): keep the block's text out of every answer.
                p.cur = {"q": "", "kind": "details", "level": 99, "lines": [], "section": ""}
            continue

        if cur and cur["kind"] == "heading":
            cur["lines"].append(line)
            continue

        lm = _LIST_ITEM.match(line) if bare_list_questions else None
        if lm and "](#" not in line:
            q = clean_question(lm.group(1))
            if is_question(q):
                p.emit_bare(q)
    p.close()
    return [qa for qa in p.out if qa.question]
