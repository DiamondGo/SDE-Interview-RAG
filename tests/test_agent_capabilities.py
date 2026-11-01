"""Capability checks of the LLM roles against their golden sets, with the real configured backends (DESIGN 8).

    uv run pytest -m agent                       # quick: the first cases of each role, a few minutes
    uv run pytest -m agent --agent-level full    # every case
Results are stored like `sde-rag agent-check` stores them, so the gates see them.
"""

import json

import pytest

from sde_rag import agent_checks

pytestmark = pytest.mark.agent


@pytest.mark.parametrize("role", agent_checks.ROLES)
def test_role_meets_its_thresholds(role, agent_level):
    out = agent_checks.run(role, agent_level)
    if out.get("skipped"):
        pytest.skip(out["skipped"])
    bad = [c for c in out["checks"] if not c["ok"]]
    assert not bad, "\n".join([f"{c['metric']} = {c['value']} (needs {c['op']} {c['threshold']})" for c in bad]
                              + [json.dumps(f, ensure_ascii=False)[:300] for f in out["failures"][:5]])
