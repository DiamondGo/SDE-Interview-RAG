"""Backend types. A backend turns (messages, params) into text, a structured object, or a text stream.

Messages are a str (one user turn) or a list of (role, content) pairs with role in system/user/assistant.
"""

import threading
import time
from typing import Any, Callable, Iterator, Protocol

from pydantic import BaseModel

Messages = str | list[tuple[str, str]]


def as_pairs(messages: Messages) -> list[tuple[str, str]]:
    return [("user", messages)] if isinstance(messages, str) else list(messages)


def flatten(messages: Messages) -> str:
    """One prompt string, for backends that take a single prompt (CLI agents)."""
    pairs = as_pairs(messages)
    if len(pairs) == 1 and pairs[0][0] == "user":
        return pairs[0][1]
    labels = {"system": "Instructions", "user": "User", "assistant": "Assistant"}
    return "\n\n".join(f"## {labels.get(r, r)}\n{c}" for r, c in pairs)


class Backend(Protocol):
    name: str
    type: str
    capabilities: frozenset[str]          # subset of {"text", "structured_output", "streaming"}
    max_concurrency: int

    def problems(self) -> list[str]: ...
    def describe(self) -> dict[str, Any]: ...
    def fingerprint_parts(self, params: dict) -> dict[str, Any]: ...
    def invoke_text(self, messages: Messages, params: dict) -> str: ...
    def invoke_structured(self, messages: Messages, schema: type[BaseModel], params: dict) -> BaseModel: ...
    def stream_text(self, messages: Messages, params: dict) -> Iterator[str]: ...
    def probe(self) -> bool: ...


class Breaker:
    """After a stall, probe the backend; two failed probes in a row open the circuit for `open_s` seconds.

    A stuck single-slot server keeps working on the abandoned request, so sending more work only queues it.
    """

    def __init__(self, probe: Callable[[], bool], *, enabled: bool, open_s: float = 600):
        self._probe = probe
        self.enabled = enabled
        self.open_s = open_s
        self.failed_probes = 0
        self.open_until = 0.0
        self.tripped = False                  # opened since the last success
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        return time.monotonic() < self.open_until

    def on_stall(self) -> bool:
        """Returns True when the backend still answers."""
        if not self.enabled:
            return True
        with self._lock:
            ok = False
            try:
                ok = self._probe()
            except Exception:
                ok = False
            if ok:
                self.failed_probes = 0
                return True
            self.failed_probes += 1
            if self.failed_probes >= 2:
                self.open_until = time.monotonic() + self.open_s
                self.tripped = True
            return False

    def on_success(self) -> bool:
        """Returns True when this success ends an outage (the circuit had opened)."""
        self.failed_probes = 0
        was, self.tripped = self.tripped, False
        return was
