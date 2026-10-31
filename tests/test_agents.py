import time

import pytest
from pydantic import BaseModel

from sde_rag.agents import AgentRegistry, AllAttemptsFailed, StallError
from sde_rag.agents.guard import call_with_deadline, guarded_stream, looks_looping
from sde_rag.agents.registry import load_agents_config
from sde_rag.settings import apply_env_overrides, deep_merge, load_layered
from tests.conftest import FakeBackend, make_registry


class Ans(BaseModel):
    answer: str


def test_layered_config_and_env_overrides(tmp_path, monkeypatch):
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "agents.toml").write_text('[backends.a]\ntype = "openai_compat"\nmodel = "m1"\n'
                                   '[roles.router]\nbackend = "a"\nthinking = false\n')
    (d / "agents.local.toml").write_text('[backends.a]\nmodel = "m2"\n')
    monkeypatch.setenv("SDE_CONFIG_DIR", str(d))
    monkeypatch.delenv("SDE_AGENTS_CONFIG")
    assert load_layered("agents")["backends"]["a"]["model"] == "m2"
    monkeypatch.setenv("SDE_ROLE__ROUTER__BACKEND", "b")
    monkeypatch.setenv("SDE_ROLE__ROUTER__MAX_TOKENS", "123")
    cfg = load_agents_config()
    assert cfg["roles"]["router"]["backend"] == "b"
    assert cfg["roles"]["router"]["max_tokens"] == 123           # JSON-typed
    assert cfg["roles"]["router"]["thinking"] is False          # untouched fields survive


def test_deep_merge_and_override_shape():
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}
    out = apply_env_overrides({}, {"P__": "roles"}, {"P__GEN__STREAM": "true", "P__BAD": "1"})
    assert out == {"roles": {"gen": {"stream": True}}}


def test_default_config_validates_role_requirements():
    reg = AgentRegistry(cfg={
        "backends": {"cli": {"type": "cli", "cli": "agent"}, "api": {"type": "openai_compat", "model": "m",
                                                                    "base_url": "http://x"}},
        "roles": {"router": {"backend": "cli"}, "generator": {"backend": "cli", "stream": True},
                  "reviewer": {"backend": "api"}, "judge": {"backend": "api"}}})
    probs = reg.problems()
    assert any("lacks structured_output" in m for _, m in probs["router"])         # Cursor CLI has no schemas
    assert any("lacks streaming" in m for _, m in probs["generator"])
    assert not reg.usable("router")


def test_reviewer_on_generator_backend_is_a_warning():
    reg = make_registry({"generator": {"backend": "a"}, "reviewer": {"backend": "a"}}, {"a": FakeBackend("a")})
    levels = [lvl for lvl, _ in reg.problems()["reviewer"]]
    assert levels == ["warning"]
    assert reg.usable("reviewer")


def test_missing_backend_and_fingerprint_changes():
    reg = make_registry({"router": {"backend": "nope"}}, {})
    assert not reg.usable("router")
    reg2 = make_registry({"router": {"backend": "a", "thinking": False}}, {"a": FakeBackend("a")})
    reg3 = make_registry({"router": {"backend": "a", "thinking": True}}, {"a": FakeBackend("a")})
    assert reg2.fingerprint("router") != reg3.fingerprint("router")


def test_structured_falls_back_on_error():
    bad = FakeBackend("bad", error=RuntimeError("boom"))
    good = FakeBackend("good", structured={"answer": "fine"})
    reg = make_registry({"router": {"backend": "bad", "fallbacks": ["good"]}}, {"bad": bad, "good": good})
    trace = []
    assert reg.structured("router", Ans, "q", trace=trace).answer == "fine"
    assert [t["ok"] for t in trace] == [False, True]


def test_structured_hard_deadline_is_a_stall():
    slow = FakeBackend("slow", structured={"answer": "late"}, delay=5)
    reg = make_registry({"router": {"backend": "slow", "timeout_s": 0.3}}, {"slow": slow})
    t0 = time.monotonic()
    with pytest.raises(AllAttemptsFailed) as e:
        reg.structured("router", Ans, "q")
    assert time.monotonic() - t0 < 2
    assert "StallError" in str(e.value)


def test_stream_stall_before_first_token_falls_back():
    hang = FakeBackend("hang", stream_script=["HANG"])
    ok = FakeBackend("ok", stream_script=["Hel", "lo"])
    reg = make_registry({"generator": {"backend": "hang", "stream": True, "first_token_timeout_s": 0.3,
                                       "fallbacks": [{"backend": "ok"}]}}, {"hang": hang, "ok": ok})
    events = list(reg.stream("generator", "q"))
    types = [e["type"] for e in events]
    assert "fallback" in types and "reset" not in types
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello"


def test_stream_stall_mid_answer_resets():
    stalls = FakeBackend("s", stream_script=["partial ", "HANG"])
    ok = FakeBackend("ok", stream_script=["full answer"])
    reg = make_registry({"generator": {"backend": "s", "stream": True, "first_token_timeout_s": 2,
                                       "idle_timeout_s": 0.3, "fallbacks": [{"backend": "ok"}]}},
                        {"s": stalls, "ok": ok})
    events = list(reg.stream("generator", "q"))
    types = [e["type"] for e in events]
    assert types.index("reset") < types.index("done")
    after_reset = events[types.index("reset"):]
    assert "".join(e["text"] for e in after_reset if e["type"] == "token") == "full answer"


def test_stream_runaway_repetition_is_stopped():
    loop = FakeBackend("loop", stream_script=["tag: python, "] * 400)
    reg = make_registry({"generator": {"backend": "loop", "stream": True}}, {"loop": loop})
    with pytest.raises(AllAttemptsFailed) as e:
        list(reg.stream("generator", "q"))
    assert "repeating" in str(e.value)


def test_slow_but_progressing_stream_is_not_a_stall():
    slow = FakeBackend("slow", stream_script=["a", 0.2, "b", 0.2, "c", 0.2, "d"])
    reg = make_registry({"generator": {"backend": "slow", "stream": True, "first_token_timeout_s": 1,
                                       "idle_timeout_s": 0.5}}, {"slow": slow})
    assert reg.text("generator", "q") == "abcd"


def test_looks_looping():
    assert looks_looping("intro " + "the same sentence again. " * 6)
    assert not looks_looping("A normal answer that explains a concept once and moves on to the next point.")
    assert not looks_looping("Table:\n" + "-" * 200)


def test_guarded_stream_total_limit():
    def forever():
        while True:
            time.sleep(0.05)
            yield "x"
    with pytest.raises(StallError):
        list(guarded_stream(forever, what="t", first_token_timeout_s=1, idle_timeout_s=1, max_total_s=0.4))


def test_call_with_deadline_returns_value():
    assert call_with_deadline(lambda: 42, 1, what="t") == 42
