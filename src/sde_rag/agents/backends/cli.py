"""Subscription CLI agents run headlessly and read-only: `claude -p`, `agy -p`, Cursor `agent -p`."""

from __future__ import annotations  # the class attribute `type` would shadow the builtin in annotations

import json
import shutil
import subprocess
from typing import Any, Iterator

from pydantic import BaseModel

from sde_rag.agents.backends import Breaker, Messages, flatten
from sde_rag.agents.errors import AgentError, StallError

_CAPS = {
    "claude": frozenset({"text", "structured_output"}),
    "agy": frozenset({"text", "structured_output"}),
    "agent": frozenset({"text"}),               # Cursor's CLI has no JSON-schema output
}


class CLIBackend:
    type = "cli"

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.cli = cfg.get("cli", "")
        self.model = cfg.get("model", "")
        self.max_concurrency = int(cfg.get("max_concurrency", 1))
        self.capabilities = _CAPS.get(self.cli, frozenset({"text"}))
        self.breaker = Breaker(self.probe, enabled=False)

    def problems(self) -> list[str]:
        out = []
        if not self.cli:
            return [f"no `cli` set (one of: {', '.join(_CAPS)})"]
        if self.cli not in _CAPS:
            out.append(f"unknown CLI {self.cli!r} (supported: {', '.join(_CAPS)})")
        if not shutil.which(self.cli):
            out.append(f"command {self.cli!r} not found in PATH")
        return out

    def describe(self) -> dict[str, Any]:
        return {"type": self.type, "model": self.model or "(CLI default)", "base_url": f"cli:{self.cli}"}

    def fingerprint_parts(self, params: dict) -> dict[str, Any]:
        return {"type": self.type, "cli": self.cli, "model": self.model}

    def _cmd(self, prompt: str, schema: dict | None) -> list[str]:
        model = ["--model", self.model] if self.model else []
        if self.cli == "claude":
            cmd = ["claude", "-p", prompt, *model, "--output-format", "json", "--permission-mode", "plan"]
            return cmd + (["--json-schema", json.dumps(schema)] if schema else [])
        if self.cli == "agy":
            cmd = ["agy", "-p", prompt, *model, "--output-format", "json", "--mode", "plan"]
            return cmd + (["--json-schema", json.dumps(schema)] if schema else [])
        if self.cli == "agent":
            if schema:
                raise AgentError("Cursor agent does not support structured output")
            return ["agent", "-p", prompt, *model, "--output-format", "json", "--mode", "ask", "--trust"]
        raise AgentError(f"unknown CLI {self.cli!r}")

    def _run(self, prompt: str, schema: dict | None, params: dict) -> tuple[Any, dict]:
        timeout = float(params.get("timeout_s") or self.cfg.get("timeout_s", 300))
        try:
            proc = subprocess.run(self._cmd(prompt, schema), capture_output=True, text=True, timeout=timeout,
                                  cwd=self.cfg.get("cwd") or None)
        except subprocess.TimeoutExpired as e:  # subprocess.run kills the child
            raise StallError(f"{self.cli}: no result after {timeout:.0f}s") from e
        lines = proc.stdout.strip().splitlines()
        try:
            data = json.loads(proc.stdout) if self.cli == "claude" else json.loads(lines[-1])
        except (json.JSONDecodeError, IndexError) as e:
            tail = (proc.stderr or proc.stdout)[-800:]
            raise AgentError(f"{self.cli}: exit {proc.returncode}, unparseable output: {tail}") from e
        if self.cli == "agy":
            if data.get("status") != "SUCCESS":
                raise AgentError(f"agy: {data.get('status')}: {str(data.get('response'))[:800]}")
            result = data.get("structured_output") or data.get("response")
        else:
            if data.get("is_error"):
                raise AgentError(f"{self.cli}: {str(data.get('result'))[:800]}")
            result = data.get("structured_output") or data.get("result")
        usage = {"cost_usd": data.get("total_cost_usd"), "usage": data.get("usage")}
        return result, usage

    def invoke_text(self, messages: Messages, params: dict) -> str:
        result, _ = self._run(flatten(messages), None, params)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def invoke_structured(self, messages: Messages, schema: type[BaseModel], params: dict) -> BaseModel:
        result, _ = self._run(flatten(messages), schema.model_json_schema(), params)
        if isinstance(result, str):
            result = json.loads(result)
        return schema.model_validate(result)

    def stream_text(self, messages: Messages, params: dict) -> Iterator[str]:
        yield self.invoke_text(messages, params)

    def probe(self) -> bool:
        return not self.problems()
