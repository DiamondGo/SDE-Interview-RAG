"""LeetCode Discuss posts by tag, through the site's GraphQL API (unofficial: read-only queries, low rate).

Listing is newest first and stops at the high-water mark (with an overlap); a post's `updatedAt` is its
version, so an edited post is fetched again.
"""

from sde_rag.http import HttpClient, Validators
from sde_rag.sources.base import DocRef, ListResult, RawDoc, iso

API = "https://leetcode.com/graphql"
LIST_QUERY = """query list($tags: [String!], $skip: Int!, $first: Int!) {
  ugcArticleDiscussionArticles(orderBy: MOST_RECENT, keywords: [""], tagSlugs: $tags, skip: $skip, first: $first) {
    edges { node { topicId slug title createdAt updatedAt } } } }"""
POST_QUERY = """query post($topicId: ID!) {
  ugcArticleDiscussionArticle(topicId: $topicId) { title content createdAt updatedAt author { userName } tags { slug } } }"""


def _ts(s: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class LeetCodeDiscuss:
    def __init__(self, name: str, cfg: dict, http: HttpClient):
        self.name, self.cfg, self.http = name, cfg, http
        self.headers = {"Referer": "https://leetcode.com/discuss/", "Content-Type": "application/json"}

    def _query(self, query: str, variables: dict) -> dict:
        r = self.http.post(API, json={"query": query, "variables": variables}, headers=self.headers).json()
        if r.get("errors"):
            raise RuntimeError(f"GraphQL: {r['errors'][0].get('message', r['errors'])}"[:300])
        return r["data"]

    def list_updates(self, state: dict) -> ListResult:
        overlap = float(self.cfg.get("overlap_hours", 24)) * 3600
        size, pages = int(self.cfg.get("page_size", 50)), int(self.cfg.get("max_pages", 6))
        found: dict[str, dict] = {}
        errors, new_state = [], dict(state)
        for tag in self.cfg.get("tags", ["interview"]):
            hw = state.get(f"hw:{tag}")
            newest, complete = None, True
            for page in range(pages):
                try:
                    edges = self._query(LIST_QUERY, {"tags": [tag], "skip": page * size, "first": size}
                                        )["ugcArticleDiscussionArticles"]["edges"]
                except Exception as e:
                    errors.append(f"tag {tag} page {page}: {type(e).__name__}: {str(e)[:200]}")
                    complete = False
                    break
                if not edges:
                    break
                for n in (e["node"] for e in edges):
                    newest = max(newest or 0, _ts(n["createdAt"]))
                    if hw is None or _ts(n["updatedAt"]) >= float(hw) - overlap:
                        found[str(n["topicId"])] = n
                if hw is not None and all(_ts(e["node"]["createdAt"]) < float(hw) - overlap for e in edges):
                    break
            if complete and newest:
                new_state[f"hw:{tag}"] = max(newest, float(hw or 0))
        refs = [DocRef(source=self.name, external_id=tid, url=f"https://leetcode.com/discuss/post/{tid}/{n['slug']}/",
                       title=n["title"], version=n["updatedAt"], published_at=iso(n["createdAt"]),
                       meta={"created_ts": _ts(n["createdAt"])})
                for tid, n in sorted(found.items(), key=lambda kv: kv[1]["createdAt"])]
        return ListResult(refs, new_state, errors)

    def fetch_doc(self, ref: DocRef, validators: Validators | None) -> RawDoc:
        a = self._query(POST_QUERY, {"topicId": ref.external_id})["ugcArticleDiscussionArticle"]
        if not a:
            raise LookupError(f"post {ref.external_id} not found")
        tags = ", ".join(t["slug"] for t in a.get("tags") or [])
        return RawDoc(ref=ref, content=f"# {a['title']}\n\n{a.get('content') or ''}\n\nTags: {tags}",
                      content_type="markdown", kind="experience", origin="leetcode", title=a["title"],
                      author=(a.get("author") or {}).get("userName"), published_at=iso(a["createdAt"]))

    def finalize(self, state: dict, failed: list[DocRef]) -> dict:
        if failed:                   # rewind so the failed (or deferred) posts are listed again
            earliest = min(r.meta["created_ts"] for r in failed)
            for k in [k for k in state if k.startswith("hw:")]:
                state[k] = min(float(state[k]), earliest)
        return state
