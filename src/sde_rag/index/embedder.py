"""Embedding models, chosen by [models.embedder] in config/agents.toml.

provider = "fastembed": ONNX models on CPU (the model name comes from the config).
provider = "hash": a deterministic bag-of-words hash, for tests and offline smoke runs only.
"""

import hashlib
import re
import threading
from typing import Protocol

import numpy as np

from sde_rag.agents.errors import ConfigError
from sde_rag.agents.registry import load_agents_config
from sde_rag.settings import data_dir


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class FastEmbedEmbedder:
    def __init__(self, cfg: dict):
        self.name = _model_name(cfg)
        self.query_prefix = cfg.get("query_prefix", "")
        self.passage_prefix = cfg.get("passage_prefix", "")
        self.batch_size = int(cfg.get("batch_size", 16))
        self.threads = cfg.get("threads")
        self._model = None
        self._lock = threading.Lock()
        self.dim = 0

    def _get(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding
                cache = data_dir() / "models"
                cache.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(self.name, cache_dir=str(cache), threads=self.threads)
                self.dim = len(next(iter(self._model.embed(["dimension probe"]))))
            return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack(list(model.embed(texts, batch_size=self.batch_size))))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.passage_prefix + t for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.query_prefix + t for t in texts])


class HashEmbedder:
    """Lexical overlap as cosine similarity. Not a semantic model: never use it for real retrieval."""

    def __init__(self, cfg: dict):
        self.dim = int(cfg.get("dim", 256))
        self.name = f"hash-{self.dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self.dim] += 1
        return v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)

    embed_queries = embed_passages


PROVIDERS = {"fastembed": FastEmbedEmbedder, "hash": HashEmbedder}
_embedder: Embedder | None = None
_lock = threading.Lock()


def _model_name(cfg: dict) -> str:
    if not cfg.get("name"):
        raise ConfigError("[models.embedder] needs a model name (see config/agents.example.toml)")
    return cfg["name"]


def embedder_config() -> dict:
    cfg = (load_agents_config().get("models") or {}).get("embedder")
    if not cfg:
        raise ConfigError("no [models.embedder] in config/agents.toml (see config/agents.example.toml)")
    return cfg


def get_embedder() -> Embedder:
    global _embedder
    with _lock:
        if _embedder is None:
            cfg = embedder_config()
            _embedder = PROVIDERS[cfg.get("provider", "fastembed")](cfg)
        return _embedder


def set_embedder(e: Embedder | None) -> None:
    global _embedder
    with _lock:
        _embedder = e


def configured_model_name() -> str:
    cfg = embedder_config()
    if cfg.get("provider") == "hash":
        return f"hash-{int(cfg.get('dim', 256))}"
    return _model_name(cfg)
