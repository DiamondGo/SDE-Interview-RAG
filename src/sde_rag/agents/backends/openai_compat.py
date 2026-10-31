"""Any OpenAI-compatible chat endpoint (llama.cpp, vLLM, Ollama, LiteLLM, OpenAI, ...)."""

from __future__ import annotations  # the class attribute `type` would shadow the builtin in annotations

import os
from typing import Any, Iterator

import httpx
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from sde_rag.agents.backends import Breaker, Messages, as_pairs

# Constructor arguments taken from backend config as-is when present.
_PASSTHROUGH = ("temperature", "top_p", "frequency_penalty", "presence_penalty", "reasoning_effort",
                "extra_body", "model_kwargs", "default_headers")


class OpenAICompatBackend:
    type = "openai_compat"
    capabilities = frozenset({"text", "structured_output", "streaming"})

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.model = cfg.get("model", "")
        self.max_concurrency = int(cfg.get("max_concurrency", 2))
        rps = cfg.get("rate_limit_rps")
        self._limiter = InMemoryRateLimiter(requests_per_second=float(rps), max_bucket_size=3) if rps else None
        self.breaker = Breaker(self.probe, enabled=bool(cfg.get("health_probe", False)),
                               open_s=float(cfg.get("circuit_open_s", 600)))

    # ---- configuration -------------------------------------------------------------------

    def _value(self, key: str) -> str | None:
        if self.cfg.get(key):
            return str(self.cfg[key])
        env = self.cfg.get(f"{key}_env")
        return os.environ.get(env) if env else None

    @property
    def base_url(self) -> str | None:
        return self._value("base_url")

    def problems(self) -> list[str]:
        out = []
        if not self.base_url:
            out.append(f"base URL missing (set {self.cfg.get('base_url_env') or 'base_url'})")
        if not self._value("api_key") and self.cfg.get("api_key_env"):
            out.append(f"API key missing (set {self.cfg['api_key_env']})")
        if not self.model:
            out.append("model missing")
        return out

    def describe(self) -> dict[str, Any]:
        return {"type": self.type, "model": self.model, "base_url": self.base_url or "-"}

    def fingerprint_parts(self, params: dict) -> dict[str, Any]:
        return {"type": self.type, "base_url": self.base_url, "model": self.model,
                "thinking": params.get("thinking"), "extra": self._thinking_kwargs(params),
                "responses_api": bool(self.cfg.get("use_responses_api"))}

    def _thinking_kwargs(self, params: dict) -> dict:
        thinking = params.get("thinking")
        if thinking is False:
            return dict(self.cfg.get("thinking_off") or {})
        if thinking is True:
            return dict(self.cfg.get("thinking_on") or {})
        return {}

    def chat_model(self, params: dict, *, streaming: bool = False) -> ChatOpenAI:
        kwargs: dict[str, Any] = {k: self.cfg[k] for k in _PASSTHROUGH if k in self.cfg}
        kwargs.setdefault("temperature", params.get("temperature", 0.2))
        kwargs.update(self._thinking_kwargs(params))
        if params.get("max_tokens"):
            kwargs["max_tokens"] = int(params["max_tokens"])
        if self.cfg.get("use_responses_api"):
            kwargs["use_responses_api"] = True
        if self._limiter:
            kwargs["rate_limiter"] = self._limiter
        timeout = float(self.cfg.get("timeout_s", 180))
        return ChatOpenAI(
            base_url=self.base_url,
            api_key=self._value("api_key") or "not-needed",
            model=self.model,
            timeout=httpx.Timeout(timeout, connect=min(15.0, timeout)),
            max_retries=int(self.cfg.get("max_retries", 1)),
            streaming=streaming,
            **kwargs,
        )

    # ---- calls ---------------------------------------------------------------------------

    def invoke_text(self, messages: Messages, params: dict) -> str:
        return self.chat_model(params).invoke(as_pairs(messages)).text

    def invoke_structured(self, messages: Messages, schema: type[BaseModel], params: dict) -> BaseModel:
        method = self.cfg.get("structured_method", "function_calling")
        model = self.chat_model(params).with_structured_output(schema, method=method)
        out = model.invoke(as_pairs(messages))
        if out is None:
            raise ValueError("model returned no structured output")
        return out if isinstance(out, schema) else schema.model_validate(out)

    def stream_text(self, messages: Messages, params: dict) -> Iterator[str]:
        for chunk in self.chat_model(params, streaming=True).stream(as_pairs(messages)):
            yield chunk.text or ""

    def probe(self) -> bool:
        """Cheap liveness check: list models, then a 1-token completion, each with a short timeout."""
        base = (self.base_url or "").rstrip("/")
        headers = {"Authorization": f"Bearer {self._value('api_key') or 'not-needed'}"}
        try:
            r = httpx.get(f"{base}/models", headers=headers, timeout=10)
            if r.status_code >= 500:
                return False
            r = httpx.post(f"{base}/chat/completions", headers=headers, timeout=20, json={
                "model": self.model, "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
                **self._thinking_kwargs({"thinking": False})})
            return r.status_code < 500
        except httpx.HTTPError:
            return False
