"""
Shared fixtures for kernel compliance tests.
Provides _EngineStub and common helpers.
"""
import pytest
from unittest.mock import MagicMock

from core.types import (
    SkillMetadata, SkillExecutionContext, Capability,
    RetryPolicy, StepStatus,
)
from core.trace import RequestTrace, StepTrace
from core.result import StepResult
from core.security import SecurityGuardian
from core.policy import PolicyEngine


# ============================================================================
# Lightweight engine stub — avoids real CoreEngine.__init__
# ============================================================================
class _EngineStub:
    """Minimal object that borrows CoreEngine methods under test."""

    def __init__(self):
        self.skills = {}
        self.cap_index = {}
        self.middlewares = []
        self.state = MagicMock()
        self.rag = None
        self.security = SecurityGuardian()
        self.policy = PolicyEngine()

    from core.engine import CoreEngine as _CE
    register_skill           = _CE.register_skill
    _execute_step            = _CE._execute_step
    _resolve_params          = _CE._resolve_params
    _resolve_string_template = _CE._resolve_string_template


@pytest.fixture
def engine():
    """Fresh engine stub per test."""
    return _EngineStub()


@pytest.fixture
def trace():
    """Mock trace with request_id and empty steps list."""
    t = MagicMock(spec=RequestTrace)
    t.request_id = "test-request-001"
    t.steps = []
    return t


def make_skill(name="test_skill", retry=RetryPolicy.NONE,
               capabilities=None, timeout=60.0, estimated_cost=0.0):
    """Factory for lightweight mock skills."""
    from unittest.mock import AsyncMock
    caps = capabilities or [Capability.REASONING]
    skill = MagicMock()
    skill.name = name
    skill.metadata = SkillMetadata(
        name=name, description=f"Mock {name}",
        capabilities=caps, retry_policy=retry,
        timeout_sec=timeout, estimated_cost=estimated_cost,
    )
    skill.input_schema = MagicMock(return_value=MagicMock())
    skill.execute = AsyncMock(return_value="ok")
    return skill


def make_step(step_id="s1", skill_name="test_skill",
              params=None, depends_on=None):
    """Factory for mock step definitions."""
    step = MagicMock()
    step.step_id = step_id
    step.skill_name = skill_name
    step.params = params or {}
    step.depends_on = depends_on or []
    return step
