"""The tag taxonomy (config/taxonomy.toml) and rule-based tag matching."""

import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

from sde_rag.settings import load_layered

_BOUNDARY_L = r"(?<![A-Za-z0-9_])"
_BOUNDARY_R = r"(?![A-Za-z0-9_])"


@dataclass(frozen=True)
class Taxonomy:
    version: int
    categories: dict[str, str]           # key -> display name
    tag_category: dict[str, str]          # tag -> category
    alias_tag: dict[str, str]             # lower-cased alias -> tag
    pattern: re.Pattern
    synonyms: dict[str, str] = None       # resolve-only names (ordinary words such as "go")

    def resolve(self, name: str) -> str | None:
        """A tag key, alias or synonym (any case) -> the tag key."""
        n = name.strip().lower()
        if n in self.tag_category:
            return n
        for cand in (n, n.replace("_", "-"), n.replace("-", " "), n.replace(" ", "-")):
            if cand in self.tag_category:
                return cand
            if cand in self.alias_tag:
                return self.alias_tag[cand]
            if self.synonyms and cand in self.synonyms:
                return self.synonyms[cand]
        return None

    def match(self, text: str) -> Counter:
        """Tags mentioned in `text`, with hit counts."""
        return Counter(self.alias_tag[m.group(0).lower()] for m in self.pattern.finditer(text))

    def category_of(self, tags) -> str | None:
        votes = Counter(self.tag_category[t] for t in tags if t in self.tag_category)
        return votes.most_common(1)[0][0] if votes else None

    def tags_in(self, category: str) -> list[str]:
        return sorted(t for t, c in self.tag_category.items() if c == category)


def build(cfg: dict) -> Taxonomy:
    tag_category: dict[str, str] = {}
    alias_tag: dict[str, str] = {}
    for category, tags in (cfg.get("tags") or {}).items():
        for tag, aliases in tags.items():
            tag_category[tag] = category
            for a in [tag, *aliases]:
                alias_tag.setdefault(a.lower(), tag)
    alternation = "|".join(re.escape(a) for a in sorted(alias_tag, key=len, reverse=True))
    pattern = re.compile(f"{_BOUNDARY_L}(?:{alternation}){_BOUNDARY_R}", re.I)
    synonyms = {k.lower(): v for k, v in (cfg.get("synonyms") or {}).items() if v in tag_category}
    return Taxonomy(int(cfg.get("version", 1)), dict(cfg.get("categories") or {}), tag_category, alias_tag, pattern,
                    synonyms)


@cache
def taxonomy() -> Taxonomy:
    return build(load_layered("taxonomy"))


def rule_tags(text: str, *, min_hits: int = 1, limit: int = 6) -> list[str]:
    hits = taxonomy().match(text)
    return [t for t, n in hits.most_common(limit) if n >= min_hits]
