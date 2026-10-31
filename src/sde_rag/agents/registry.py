"""Roles -> backends, from config/agents.toml + agents.local.toml + SDE_ROLE__/SDE_BACKEND__ env overrides.

Code asks for a role ("router", "generator", ...); which backend and model serve it is configuration.
Every call goes through the role's attempt chain (primary backend, then fallbacks) with stall detection.
"""

import hashlib
import json
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from pydantic import BaseModel

from sde_rag import notify
from sde_rag.agents.backends import Backend, Messages
from sde_rag.agents.backends.cli import CLIBackend
from sde_rag.agents.backends.openai_compat import OpenAICompatBackend
from sde_rag.agents.errors import (AgentError, AllAttemptsFailed, BackendUnavailable, ConfigError,
                                   RunawayError, StallError)
from sde_rag.agents.guard import call_with_deadline, guarded_stream
from sde_rag.settings import apply_env_overrides, load_env, load_layered

BACKEND_TYPES: dict[str, Callable[[str, dict], Backend]] = {
    "openai_compat": OpenAICompatBackend,
    "cli": CLIBackend,
}

# What each role needs from its backend. Streaming roles only need "streaming" when stream = true.
ROLE_REQUIREMENTS: dict[str, set[str]] = {
    "router": {"structured_output"},
    "tagger": {"structured_output"},
    "extractor": {"structured_output"},
    "grader": {"structured_output"},
    "dedup_judge": {"structured_output"},
    "reviewer": {"structured_output"},
    "judge": {"structured_output"},
    "quality_sampler": {"structured_output"},
    "eval_query_writer": {"structured_output"},
    "composer": {"text"},
    "generator": {"text"},
    "reference_writer": {"text"},
}
INDEPENDENT_OF_GENERATOR = ("reviewer", "judge")
_PARAM_KEYS = ("thinking", "stream", "timeout_s", "first_token_timeout_s", "idle_timeout_s", "max_total_s",
               "max_tokens", "temperature", "max_chars", "queue_timeout_s", "prompt_variant")


@dataclass
class Attempt:
    backend: str
    params: dict


@dataclass
class Role:
    name: str
    attempts: list[Attempt]
    sources: dict[str, str] = field(default_factory=dict)   # field -> config layer it came from


def load_agents_config() -> dict:
    load_env()
    return apply_env_overrides(load_layered("agents"),
                               {"SDE_ROLE__": "roles", "SDE_BACKEND__": "backends", "SDE_MODEL__": "models"})


def _role_attempts(name: str, raw: dict) -> list[Attempt]:
    if "backend" not in raw:
        raise ConfigError(f"role {name!r} has no backend")
    base = {k: raw[k] for k in _PARAM_KEYS if k in raw}
    attempts = [Attempt(raw["backend"], base)]
    for fb in raw.get("fallbacks") or []:
        if isinstance(fb, str):
            fb = {"backend": fb}
        params = {**base, **{k: fb[k] for k in _PARAM_KEYS if k in fb}}
        attempts.append(Attempt(fb["backend"], params))
    return attempts


class AgentRegistry:
    def __init__(self, cfg: dict | None = None, backends: dict[str, Backend] | None = None):
        self.cfg = load_agents_config() if cfg is None else cfg
        self.backends: dict[str, Backend] = dict(backends or {})
        for name, bcfg in (self.cfg.get("backends") or {}).items():
            if name in self.backends:
                continue
            btype = bcfg.get("type")
            if btype not in BACKEND_TYPES:
                raise ConfigError(f"backend {name!r}: unknown type {btype!r} (known: {', '.join(BACKEND_TYPES)})")
            self.backends[name] = BACKEND_TYPES[btype](name, bcfg)
        self.roles: dict[str, Role] = {n: Role(n, _role_attempts(n, r)) for n, r in (self.cfg.get("roles") or {}).items()}
        self._slots = {n: threading.Semaphore(max(1, b.max_concurrency)) for n, b in self.backends.items()}
        self._stalls: Counter = Counter()
        self._stall_lock = threading.Lock()

    # ---- inspection ----------------------------------------------------------------------

    def role(self, name: str) -> Role:
        if name not in self.roles:
            raise ConfigError(f"role {name!r} is not configured (add [roles.{name}] to config/agents.toml)")
        return self.roles[name]

    def problems(self) -> dict[str, list[tuple[str, str]]]:
        """role -> [(level, message)], level is "error" (role unusable) or "warning"."""
        out: dict[str, list[tuple[str, str]]] = {}
        for name, role in self.roles.items():
            issues: list[tuple[str, str]] = []
            if name not in ROLE_REQUIREMENTS:
                issues.append(("warning", "unknown role name (not used by the code)"))
            for i, a in enumerate(role.attempts):
                where = "primary" if i == 0 else f"fallback {i}"
                b = self.backends.get(a.backend)
                if b is None:
                    issues.append(("error", f"{where}: backend {a.backend!r} is not defined"))
                    continue
                for p in b.problems():
                    issues.append(("error" if i == 0 else "warning", f"{where} {a.backend}: {p}"))
                need = set(ROLE_REQUIREMENTS.get(name, set()))
                if a.params.get("stream"):
                    need.add("streaming")
                missing = need - set(b.capabilities)
                if missing:
                    hint = " (set stream = false for this role)" if missing == {"streaming"} else ""
                    issues.append(("error" if i == 0 else "warning",
                                   f"{where} {a.backend}: lacks {', '.join(sorted(missing))}{hint}"))
            if name in INDEPENDENT_OF_GENERATOR and "generator" in self.roles:
                if role.attempts[0].backend == self.roles["generator"].attempts[0].backend:
                    issues.append(("warning", "same backend as the generator: it would be judging its own answers"))
            if issues:
                out[name] = issues
        return out

    def usable(self, name: str) -> bool:
        return name in self.roles and not any(l == "error" for l, _ in self.problems().get(name, []))

    def fingerprint(self, name: str) -> str:
        parts = []
        for a in self.role(name).attempts:
            b = self.backends.get(a.backend)
            parts.append({"backend": a.backend, **(b.fingerprint_parts(a.params) if b else {}),
                          "params": {k: a.params.get(k) for k in ("thinking", "max_tokens", "prompt_variant")}})
        return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for name, role in sorted(self.roles.items()):
            for i, a in enumerate(role.attempts):
                b = self.backends.get(a.backend)
                d = b.describe() if b else {"type": "?", "model": "?", "base_url": "?"}
                rows.append({"role": name if i == 0 else "", "order": "primary" if i == 0 else f"fallback {i}",
                             "backend": a.backend, **d,
                             "thinking": a.params.get("thinking"), "stream": a.params.get("stream", False),
                             "fingerprint": self.fingerprint(name) if i == 0 else ""})
        return rows

    # ---- calls ---------------------------------------------------------------------------

    def _acquire(self, a: Attempt) -> Callable[[], None]:
        slot = self._slots[a.backend]
        wait = float(a.params.get("queue_timeout_s", 900))
        if not slot.acquire(timeout=wait):
            raise BackendUnavailable(f"{a.backend}: still busy after waiting {wait:.0f}s for a free slot")
        released = threading.Event()

        def release():
            if not released.is_set():
                released.set()
                slot.release()
        return release

    def _usable_backend(self, a: Attempt) -> Backend:
        b = self.backends.get(a.backend)
        if b is None:
            raise BackendUnavailable(f"backend {a.backend!r} is not defined")
        probs = b.problems()
        if probs:
            raise BackendUnavailable(f"{a.backend}: {'; '.join(probs)}")
        if b.breaker.is_open():
            raise BackendUnavailable(f"{a.backend}: circuit open after failed health probes")
        return b

    def _note_failure(self, role: str, b: Backend | None, e: Exception) -> None:
        if isinstance(e, (StallError, RunawayError)):
            with self._stall_lock:
                self._stalls[f"{role}@{b.name if b else '?'}"] += 1
        if b is not None and isinstance(e, StallError):
            was_open = b.breaker.is_open()
            b.breaker.on_stall()
            if b.breaker.is_open() and not was_open:
                notify.alert(f"llm:{b.name}", f"Model backend {b.name} not responding",
                             f"stalled during {role} and failed two health probes; calls to it pause for "
                             f"{b.breaker.open_s:.0f}s and queued work waits (fallbacks are used where configured)")

    def _note_success(self, b: Backend) -> None:
        if b.breaker.on_success():
            notify.resolve(f"llm:{b.name}", f"Model backend {b.name} responding again")

    def take_stalls(self) -> dict[str, int]:
        """Stalled or runaway calls by role@backend since the last call (an ingest run's report takes them)."""
        with self._stall_lock:
            out, self._stalls = dict(self._stalls), Counter()
        return out

    def _run_chain(self, name: str, fn: Callable[[Backend, Attempt], Any], trace: list | None) -> Any:
        errors = []
        for a in self.role(name).attempts:
            b = None
            t0 = time.monotonic()
            try:
                b = self._usable_backend(a)
                result = fn(b, a)
                self._note_success(b)
                if trace is not None:
                    trace.append({"role": name, "backend": a.backend, "ms": int((time.monotonic() - t0) * 1000),
                                  "ok": True})
                return result
            except Exception as e:  # every failure moves on to the next attempt
                self._note_failure(name, b, e)
                msg = f"{a.backend}: {type(e).__name__}: {str(e)[:300]}"
                errors.append(msg)
                if trace is not None:
                    trace.append({"role": name, "backend": a.backend, "ms": int((time.monotonic() - t0) * 1000),
                                  "ok": False, "error": msg, "stalled": isinstance(e, StallError)})
        raise AllAttemptsFailed(name, errors)

    def structured(self, name: str, schema: type[BaseModel], messages: Messages, *,
                   trace: list | None = None) -> BaseModel:
        def call(b: Backend, a: Attempt):
            release = self._acquire(a)
            try:
                return call_with_deadline(lambda: b.invoke_structured(messages, schema, a.params),
                                          float(a.params.get("timeout_s", 180)), what=f"{name}@{a.backend}",
                                          on_exit=release)
            except StallError:
                raise
            except BaseException:
                release()
                raise
        return self._run_chain(name, call, trace)

    def text(self, name: str, messages: Messages, *, trace: list | None = None) -> str:
        return "".join(e["text"] for e in self.stream(name, messages, trace=trace) if e["type"] == "token")

    def stream(self, name: str, messages: Messages, *, trace: list | None = None) -> Iterator[dict]:
        """Events: attempt {backend, model}, token {text}, reset {reason} (discard earlier tokens), done {backend}.

        Raises AllAttemptsFailed when no backend produced a complete answer.
        """
        errors = []
        for i, a in enumerate(self.role(name).attempts):
            b = None
            t0 = time.monotonic()
            yielded = False
            try:
                b = self._usable_backend(a)
                yield {"type": "attempt", "backend": a.backend, "model": getattr(b, "model", ""), "index": i}
                release = self._acquire(a)
                timeout = float(a.params.get("timeout_s", 180))
                if a.params.get("stream") and "streaming" in b.capabilities:
                    opener = lambda b=b, a=a: b.stream_text(messages, a.params)
                    first, idle = (float(a.params.get("first_token_timeout_s", timeout)),
                                   float(a.params.get("idle_timeout_s", timeout)))
                else:
                    opener = lambda b=b, a=a: iter([b.invoke_text(messages, a.params)])
                    first = idle = timeout
                for piece in guarded_stream(opener, what=f"{name}@{a.backend}", first_token_timeout_s=first,
                                            idle_timeout_s=idle, max_total_s=a.params.get("max_total_s"),
                                            max_chars=a.params.get("max_chars"), on_exit=release):
                    yielded = True
                    yield {"type": "token", "text": piece}
                self._note_success(b)
                if trace is not None:
                    trace.append({"role": name, "backend": a.backend, "ms": int((time.monotonic() - t0) * 1000),
                                  "ok": True})
                yield {"type": "done", "backend": a.backend}
                return
            except Exception as e:  # every failure moves on to the next attempt
                self._note_failure(name, b, e)
                msg = f"{a.backend}: {type(e).__name__}: {str(e)[:300]}"
                errors.append(msg)
                if trace is not None:
                    trace.append({"role": name, "backend": a.backend, "ms": int((time.monotonic() - t0) * 1000),
                                  "ok": False, "error": msg,
                                  "stalled": isinstance(e, (StallError, RunawayError))})
                if yielded:
                    yield {"type": "reset", "reason": msg}
                elif i + 1 < len(self.role(name).attempts):
                    yield {"type": "fallback", "reason": msg}
        raise AllAttemptsFailed(name, errors)


_registry: AgentRegistry | None = None
_lock = threading.Lock()


def get_registry() -> AgentRegistry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = AgentRegistry()
        return _registry


def set_registry(reg: AgentRegistry | None) -> None:
    """Tests inject a registry with fake backends; None resets to the configured one."""
    global _registry
    with _lock:
        _registry = reg
