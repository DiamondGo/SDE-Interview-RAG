from __future__ import annotations  # FakeBackend.type would shadow the builtin in annotations

import time
from typing import Iterator

import pytest
from pydantic import BaseModel

from sde_rag import auth, db, settings
from sde_rag.agents import AgentRegistry, set_registry
from sde_rag.agents.backends import Breaker
from sde_rag.index import embedder, vectors
from sde_rag import taxonomy


def pytest_configure(config):
    # Process-wide fallback, set outside monkeypatch: even if a test undoes its monkeypatching, the data
    # directory falls back to a throwaway directory, never to the project's real data/.
    import os
    import tempfile
    os.environ["SDE_DATA_DIR"] = tempfile.mkdtemp(prefix="sde-rag-tests-")
    os.environ["SDE_MODEL__EMBEDDER__PROVIDER"] = "hash"
    for name in ("SETTINGS", "AGENTS", "SOURCES", "TAXONOMY"):
        os.environ[f"SDE_{name}_CONFIG"] = os.path.join(os.environ["SDE_DATA_DIR"], "no-local-override.toml")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets its own data dir, the hash embedder, no machine-local overrides and fresh caches."""
    monkeypatch.setenv("SDE_DATA_DIR", str(tmp_path / "data"))
    for name in ("SETTINGS", "AGENTS", "SOURCES", "TAXONOMY"):
        monkeypatch.setenv(f"SDE_{name}_CONFIG", str(tmp_path / "no-local-override.toml"))
    monkeypatch.setenv("SDE_MODEL__EMBEDDER__PROVIDER", "hash")
    monkeypatch.setenv("SDE_MODEL__EMBEDDER__DIM", "64")
    monkeypatch.setenv("SDE_SETTING__HTTP__DEFAULT_DELAY_S", "0")
    _reset()
    yield tmp_path
    _reset()


def _reset():
    db.close_all()
    settings.reset_caches()
    taxonomy.taxonomy.cache_clear()
    set_registry(None)
    embedder.set_embedder(None)
    vectors.reset_indexes()
    auth.reset()


class FakeBackend:
    """Scripted backend. `stream_script` items: str = chunk, float = sleep that long, "HANG" = never finish."""
    type = "fake"
    capabilities = frozenset({"text", "structured_output", "streaming"})

    def __init__(self, name="fake", *, text="ok", structured=None, stream_script=None, error=None,
                 delay=0.0, caps=None):
        self.name, self.model, self.max_concurrency = name, "fake-model", 1
        self.text, self.structured, self.stream_script = text, structured, stream_script
        self.error, self.delay = error, delay
        if caps is not None:
            self.capabilities = frozenset(caps)
        self.breaker = Breaker(lambda: True, enabled=False)
        self.calls = 0

    def problems(self):
        return []

    def describe(self):
        return {"type": self.type, "model": self.model, "base_url": "fake://"}

    def fingerprint_parts(self, params):
        return {"type": "fake", "name": self.name}

    def _maybe_fail(self):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error

    def invoke_text(self, messages, params):
        self._maybe_fail()
        return self.text

    def invoke_structured(self, messages, schema: type[BaseModel], params):
        self._maybe_fail()
        value = self.structured(schema, messages) if callable(self.structured) else self.structured
        return value if isinstance(value, schema) else schema.model_validate(value)

    def stream_text(self, messages, params) -> Iterator[str]:
        self._maybe_fail()
        for item in self.stream_script or [self.text]:
            if item == "HANG":
                time.sleep(3600)
            elif isinstance(item, float):
                time.sleep(item)
            else:
                yield item

    def probe(self):
        return True


def make_registry(roles: dict, backends: dict) -> AgentRegistry:
    reg = AgentRegistry(cfg={"roles": roles, "backends": {}}, backends=backends)
    set_registry(reg)
    return reg


@pytest.fixture
def fake_backend():
    return FakeBackend
