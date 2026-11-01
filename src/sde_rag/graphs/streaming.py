"""Custom stream events of the query graph, shared by its nodes and the planner's.

  {"type": "token", "text": ...}   append to the answer being shown
  {"type": "reset"}               the model failed mid-answer: discard what was shown so far
  {"type": "status", "text": ...} progress ("searching", "model not responding, switching to ...")
"""

from langgraph.config import get_stream_writer

from sde_rag.agents import AgentError, get_registry


def emit(event: dict) -> None:
    try:
        get_stream_writer()(event)
    except Exception:  # not streaming (plain invoke): nothing to emit to
        pass


def stream_role(role: str, messages) -> tuple[str | None, list, str | None]:
    """Stream a role's answer as custom events. Returns (text or None when every backend failed, calls, error)."""
    calls: list = []
    parts: list[str] = []
    try:
        for ev in get_registry().stream(role, messages, trace=calls):
            if ev["type"] == "token":
                parts.append(ev["text"])
                emit({"type": "token", "text": ev["text"]})
            elif ev["type"] == "reset":
                parts = []
                emit({"type": "reset"})
                emit({"type": "status", "text": "model stopped responding, switching to a fallback"})
            elif ev["type"] == "fallback":
                emit({"type": "status", "text": "model unavailable, switching to a fallback"})
            elif ev["type"] == "attempt":
                emit({"type": "status", "text": f"writing the answer ({ev['backend']})"})
        return "".join(parts), calls, None
    except AgentError as e:
        if parts:
            emit({"type": "reset"})
        return None, calls, str(e)
