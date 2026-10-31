"""dev.to articles by tag (public API). The listing is roughly newest-first; paging stops at the high-water mark."""

import time
from datetime import datetime

from sde_rag.http import HttpClient, Validators
from sde_rag.sources.base import DocRef, ListResult, RawDoc, iso

API = "https://dev.to/api"


def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class DevTo:
    def __init__(self, name: str, cfg: dict, http: HttpClient):
        self.name = name
        self.cfg = cfg
        self.http = http

    def list_updates(self, state: dict) -> ListResult:
        now = time.time()
        settle = now - float(self.cfg.get("settle_hours", 72)) * 3600
        overlap = float(self.cfg.get("overlap_hours", 24)) * 3600
        min_reactions = int(self.cfg.get("min_reactions", 2))
        new_state = dict(state)
        found: dict[str, dict] = {}
        errors = []
        for tag in self.cfg.get("tags", ["interview"]):
            hw = state.get(f"hw:{tag}")
            pages = int(self.cfg.get("backfill_pages", 10)) if hw is None else 30
            failed = False
            for page in range(1, pages + 1):
                try:
                    batch = self.http.get(f"{API}/articles",
                                          params={"tag": tag, "per_page": 100, "page": page}).json()
                except Exception as e:
                    errors.append(f"tag {tag} page {page}: {type(e).__name__}: {str(e)[:200]}")
                    failed = True
                    break
                if not batch:
                    break
                for a in batch:
                    ts = _ts(a["published_at"])
                    if ts > settle or (hw is not None and ts < float(hw) - overlap):
                        continue
                    if int(a.get("public_reactions_count") or 0) < min_reactions:
                        continue
                    found[str(a["id"])] = a
                if hw is not None and all(_ts(a["published_at"]) < float(hw) - overlap for a in batch):
                    break
            if not failed:  # everything up to the settle line has been seen
                new_state[f"hw:{tag}"] = settle
        ordered = sorted(found.values(), key=lambda a: a["published_at"])
        limit = int(self.cfg.get("per_run_limit", 150))
        refs = [DocRef(source=self.name, external_id=str(a["id"]), url=a["url"], title=a["title"],
                       version=a.get("edited_at") or a["published_at"], published_at=iso(a["published_at"]),
                       meta={"tags": a.get("tag_list") or [], "reactions": a.get("public_reactions_count"),
                             "published_ts": _ts(a["published_at"])})
                for a in ordered[:limit]]
        if len(ordered) > limit:  # come back for the rest
            cutoff = refs[-1].meta["published_ts"]
            for tag in self.cfg.get("tags", ["interview"]):
                if new_state.get(f"hw:{tag}"):
                    new_state[f"hw:{tag}"] = min(float(new_state[f"hw:{tag}"]), cutoff)
        return ListResult(refs, new_state, errors)

    def fetch_doc(self, ref: DocRef, validators: Validators | None) -> RawDoc:
        r = self.http.get(f"{API}/articles/{ref.external_id}", validators=validators)
        a = r.json()
        body = a.get("body_markdown") or ""
        return RawDoc(ref=ref, content=f"# {a.get('title', ref.title)}\n\n{body}", content_type="markdown",
                      kind="experience", origin="devto", title=a.get("title", ref.title),
                      author=(a.get("user") or {}).get("name"), published_at=iso(a.get("published_at")),
                      canonical_url=a.get("canonical_url"), etag=r.headers.get("ETag"),
                      last_modified=r.headers.get("Last-Modified"))

    def finalize(self, state: dict, failed: list[DocRef]) -> dict:
        if failed:
            earliest = min(r.meta["published_ts"] for r in failed)
            for k in list(state):
                if k.startswith("hw:") and state[k]:
                    state[k] = min(float(state[k]), earliest)
        return state
