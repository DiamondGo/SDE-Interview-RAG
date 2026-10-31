"""Question repositories on GitHub.

Per repo, one cheap request asks for the branch's commit SHA (with ETag, so an unchanged repo costs a 304).
Only when it moved is the tree listed; each file's blob SHA is its version, so only changed files download.
"""

import fnmatch
import os
import shutil
import subprocess
from functools import cache
from urllib.parse import quote

from sde_rag.http import HttpClient, NotModified, Validators
from sde_rag.sources.base import DocRef, ListResult, RawDoc

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"


@cache
def _gh_token() -> str | None:
    if not shutil.which("gh"):
        return None
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


class GitHubRepos:
    def __init__(self, name: str, cfg: dict, http: HttpClient):
        self.name = name
        self.cfg = cfg
        self.http = http
        self.repos = {r["repo"]: r for r in cfg.get("repos", [])}
        token = os.environ.get(cfg.get("token_env", "GITHUB_TOKEN")) or _gh_token()
        self.headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _api(self, path: str, *, accept: str | None = None, validators: Validators | None = None):
        headers = dict(self.headers)
        if accept:
            headers["Accept"] = accept
        return self.http.get(f"{API}{path}", headers=headers, validators=validators)

    @staticmethod
    def _wanted(path: str, spec: dict) -> bool:
        inc = spec.get("include") or ["*.md"]
        exc = spec.get("exclude") or []
        return any(fnmatch.fnmatch(path, p) for p in inc) and not any(fnmatch.fnmatch(path, p) for p in exc)

    def list_updates(self, state: dict) -> ListResult:
        new_state = {k: dict(v) for k, v in state.items()}
        refs, errors = [], []
        for repo, spec in self.repos.items():
            st = new_state.setdefault(repo, {})
            try:
                if not st.get("branch"):
                    st["branch"] = self._api(f"/repos/{repo}").json()["default_branch"]
                try:
                    r = self._api(f"/repos/{repo}/commits/{st['branch']}", accept="application/vnd.github.sha",
                                  validators=Validators(etag=st.get("etag")))
                    sha, etag = r.text.strip(), r.headers.get("ETag")
                except NotModified:
                    sha, etag = st.get("sha"), st.get("etag")
                if sha == st.get("sha") and st.get("complete", True):
                    continue
                tree = self._api(f"/repos/{repo}/git/trees/{sha}?recursive=1").json()
                if tree.get("truncated"):
                    errors.append(f"{repo}: tree listing truncated; some files may be missing")
                for item in tree.get("tree", []):
                    if item.get("type") != "blob" or not self._wanted(item["path"], spec):
                        continue
                    path = item["path"]
                    refs.append(DocRef(
                        source=self.name, external_id=f"{repo}:{path}",
                        url=f"https://github.com/{repo}/blob/{st['branch']}/{quote(path)}",
                        title=f"{repo}/{path}", version=item["sha"],
                        meta={"repo": repo, "path": path, "commit": sha}))
                st.update(sha=sha, etag=etag, complete=True)
            except Exception as e:  # one broken repo must not stop the others
                errors.append(f"{repo}: {type(e).__name__}: {str(e)[:200]}")
        return ListResult(refs, new_state, errors)

    def fetch_doc(self, ref: DocRef, validators: Validators | None) -> RawDoc:
        repo, path, commit = ref.meta["repo"], ref.meta["path"], ref.meta["commit"]
        text = self.http.get(f"{RAW}/{repo}/{commit}/{quote(path)}").text
        spec = self.repos.get(repo, {})
        hints = {k: spec[k] for k in ("category", "tags") if k in spec}
        return RawDoc(ref=ref, content=text, content_type="markdown", kind="prep", origin=repo,
                      title=ref.title, hints=hints)

    @staticmethod
    def full_listing(state: dict) -> dict:
        """Reconciliation: list every repo's tree again next run (unchanged blobs still are not downloaded)."""
        return {repo: {**st, "complete": False} for repo, st in state.items()}

    def finalize(self, state: dict, failed: list[DocRef]) -> dict:
        for ref in failed:
            state.get(ref.meta["repo"], {})["complete"] = False   # list the repo again next run
        return state
