"""Cross-encoder reranking, chosen by [models.reranker] in config/agents.toml.

provider = "fastembed": an ONNX cross-encoder on CPU (default BAAI/bge-reranker-v2-m3-int8, multilingual).
provider = "overlap": shared-word counting, for tests and offline smoke runs only.
provider = "none": no reranking (retrieval order is kept).
"""

import re
import threading
from typing import Protocol

import numpy as np

from sde_rag.agents.registry import load_agents_config
from sde_rag.settings import data_dir


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


class FastEmbedReranker:
    def __init__(self, cfg: dict):
        self.name = cfg.get("name", "BAAI/bge-reranker-v2-m3-int8")
        self.batch_size = int(cfg.get("batch_size", 8))
        self.threads = cfg.get("threads")
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                cache = data_dir() / "models"
                cache.mkdir(parents=True, exist_ok=True)
                self._model = TextCrossEncoder(self.name, cache_dir=str(cache), threads=self.threads)
            return self._model

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(list(self._get().rerank(query, texts, batch_size=self.batch_size)), dtype=np.float32)


class OverlapReranker:
    name = "overlap"

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        q = set(re.findall(r"\w+", query.lower()))
        return np.array([len(q & set(re.findall(r"\w+", t.lower()))) for t in texts], dtype=np.float32)


_reranker: Reranker | None = None
_configured = False
_lock = threading.Lock()


def get_reranker() -> Reranker | None:
    global _reranker, _configured
    with _lock:
        if not _configured:
            cfg = (load_agents_config().get("models") or {}).get("reranker") or {"provider": "none"}
            provider = cfg.get("provider", "fastembed")
            _reranker = {"fastembed": FastEmbedReranker, "overlap": lambda c: OverlapReranker()}.get(
                provider, lambda c: None)(cfg)
            _configured = True
        return _reranker


def set_reranker(r: Reranker | None, *, configured: bool = True) -> None:
    """Tests inject one; set_reranker(None, configured=False) goes back to the configuration."""
    global _reranker, _configured
    with _lock:
        _reranker, _configured = r, configured
