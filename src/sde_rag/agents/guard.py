"""Stall and runaway detection for model calls.

Slow is fine; stuck is not. A call is stuck when it makes no progress: no first chunk within
`first_token_timeout_s`, no further chunk within `idle_timeout_s`, or (for non-streaming calls) no result
within its hard deadline. Abandoned calls keep running in a daemon thread until their own HTTP timeout,
because a blocked socket read cannot be interrupted from another thread.
"""

import queue
import threading
import time
from typing import Any, Callable, Iterator

from sde_rag.agents.errors import RunawayError, StallError

_END = object()


def call_with_deadline(fn: Callable[[], Any], timeout_s: float, *, what: str,
                       on_exit: Callable[[], None] | None = None) -> Any:
    box: dict[str, Any] = {}
    done = threading.Event()

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # re-raised in the caller
            box["error"] = e
        finally:
            done.set()
            if on_exit:
                on_exit()

    threading.Thread(target=run, daemon=True, name=f"deadline:{what}").start()
    if not done.wait(timeout_s):
        raise StallError(f"{what}: no result after {timeout_s:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def looks_looping(text: str, *, min_period: int = 12, max_period: int = 400, repeats: int = 4) -> bool:
    """True when the tail of `text` is one block repeated `repeats` times (a model stuck in a loop)."""
    n = len(text)
    for p in range(min_period, min(max_period, n // repeats) + 1):
        tail = text[n - p:]
        if len(set(tail)) <= 3:  # rules like "-----" or "=====" are not loops
            continue
        if all(text[n - (i + 1) * p:n - i * p] == tail for i in range(1, repeats)):
            return True
    return False


def guarded_stream(open_stream: Callable[[], Iterator[str]], *, what: str, first_token_timeout_s: float,
                   idle_timeout_s: float, max_total_s: float | None = None, max_chars: int | None = None,
                   on_exit: Callable[[], None] | None = None, loop_check_every: int = 400) -> Iterator[str]:
    """Yield text chunks from `open_stream()` (run in a worker thread) and raise on stall or runaway.

    Chunks with empty text (e.g. reasoning deltas) count as progress but are not yielded.
    """
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def worker():
        try:
            for chunk in open_stream():
                if cancel.is_set():
                    break
                q.put(chunk)
            q.put(_END)
        except BaseException as e:
            q.put(e)
        finally:
            if on_exit:
                on_exit()

    threading.Thread(target=worker, daemon=True, name=f"stream:{what}").start()
    start = time.monotonic()
    got_first = False
    text_len = 0
    since_check = 0
    acc: list[str] = []
    try:
        while True:
            wait = idle_timeout_s if got_first else first_token_timeout_s
            if max_total_s is not None:
                left = max_total_s - (time.monotonic() - start)
                if left <= 0:
                    raise StallError(f"{what}: exceeded the total limit of {max_total_s:.0f}s")
                wait = min(wait, left)
            try:
                item = q.get(timeout=wait)
            except queue.Empty:
                if max_total_s is not None and time.monotonic() - start >= max_total_s:
                    raise StallError(f"{what}: exceeded the total limit of {max_total_s:.0f}s") from None
                phase = "a further chunk" if got_first else "the first chunk"
                raise StallError(f"{what}: no {phase} within {wait:.0f}s") from None
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            got_first = True
            if not item:
                continue
            acc.append(item)
            text_len += len(item)
            since_check += len(item)
            if max_chars is not None and text_len > max_chars:
                raise RunawayError(f"{what}: output exceeded {max_chars} characters")
            if since_check >= loop_check_every:
                since_check = 0
                if looks_looping("".join(acc)[-4000:]):
                    raise RunawayError(f"{what}: output is repeating itself")
            yield item
    finally:
        cancel.set()
