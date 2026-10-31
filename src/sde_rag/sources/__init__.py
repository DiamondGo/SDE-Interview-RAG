from sde_rag.http import HttpClient
from sde_rag.settings import apply_env_overrides, load_env, load_layered
from sde_rag.sources.base import Connector, DocRef, ListResult, RawDoc
from sde_rag.sources.devto import DevTo
from sde_rag.sources.github_repos import GitHubRepos
from sde_rag.sources.hn import HackerNews

CONNECTOR_TYPES = {
    "github_repos": GitHubRepos,
    "hn_algolia": HackerNews,
    "devto": DevTo,
}


def source_configs() -> dict[str, dict]:
    """config/sources.toml + sources.local.toml + SDE_SOURCE__<NAME>__<FIELD> overrides."""
    load_env()
    cfg = apply_env_overrides(load_layered("sources"), {"SDE_SOURCE__": "sources"})
    return dict(cfg.get("sources") or {})


def build_connector(name: str, cfg: dict, http: HttpClient | None = None) -> Connector:
    if cfg.get("type") not in CONNECTOR_TYPES:
        raise ValueError(f"source {name!r}: unknown type {cfg.get('type')!r} (known: {', '.join(CONNECTOR_TYPES)})")
    http = http or HttpClient(delay_s=cfg.get("delay_s"))
    return CONNECTOR_TYPES[cfg["type"]](name, cfg, http)


__all__ = ["CONNECTOR_TYPES", "Connector", "DocRef", "ListResult", "RawDoc", "build_connector", "source_configs"]
