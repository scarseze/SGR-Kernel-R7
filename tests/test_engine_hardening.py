"""
Verification tests for Engine Hardening fixes (Rounds 1-3).
Tests: AttemptTrace, param resolution, metadata normalization, string templates, middleware retry.
"""
import pytest
import re
import time
from unittest.mock import MagicMock, AsyncMock, patch
from core.types import SkillMetadata, SkillExecutionContext, Capability, RetryPolicy, StepStatus
from core.trace import RequestTrace, StepTrace, AttemptTrace
from core.result import StepResult


# ---------------------------------------------------------------------------
# Lightweight Engine stub — avoids real CoreEngine.__init__ (Qdrant, DB, etc.)
# ---------------------------------------------------------------------------
class _EngineStub:
    """Minimal object that has only the methods under test."""

    def __init__(self):
        self.skills = {}
        self.cap_index = {}
        self.middlewares = []          # no middlewares → clean unit test
        self.state = MagicMock()
        self.rag = None

    # Import the real methods we want to test
    from core.engine import CoreEngine as _CE
    register_skill            = _CE.register_skill
    _execute_step             = _CE._execute_step
    _resolve_params           = _CE._resolve_params
    _resolve_string_template  = _CE._resolve_string_template


# ---------------------------------------------------------------------------
# Mock Skill
# ---------------------------------------------------------------------------
class MockSkill:
    def __init__(self, name, retry_policy=RetryPolicy.STANDARD):
        self.name = name
        self.metadata = SkillMetadata(
            name=name,
            description="Mock Skill",
            retry_policy=retry_policy,
            capabilities=["reasoning"],
            risk_level="low",
        )
        self.input_schema = MagicMock()
        self.input_schema.return_value = MagicMock()

    async def execute(self, input_data, state):
        return "success"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def engine():
    return _EngineStub()


# ===========================================================================
# Round 1 Tests
# ===========================================================================
@pytest.mark.asyncio
async def test_attempt_tracing_success(engine):
    """Verify a single successful attempt is recorded."""
    skill = MockSkill("test_skill")
    engine.skills["test_skill"] = skill

    step_def = MagicMock()
    step_def.step_id = "step_1"
    step_def.skill_name = "test_skill"
    step_def.params = {}

    trace = RequestTrace(user_request="test")

    await engine._execute_step(step_def, {}, trace)

    step_trace = trace.steps[0]
    assert len(step_trace.attempts) == 1
    assert step_trace.attempts[0].attempt_number == 1
    assert step_trace.attempts[0].result_snippet is not None
    assert step_trace.attempts[0].error is None


@pytest.mark.asyncio
async def test_attempt_tracing_retry(engine):
    """Verify multiple attempts are recorded on retry then success."""
    skill = MockSkill("flaky_skill", RetryPolicy.STANDARD)  # 3 max attempts

    call_count = 0

    async def side_effect(*args):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Fail")
        return "Success"

    skill.execute = side_effect
    engine.skills["flaky_skill"] = skill

    step_def = MagicMock()
    step_def.step_id = "step_retry"
    step_def.skill_name = "flaky_skill"
    step_def.params = {}

    trace = RequestTrace(user_request="test")

    await engine._execute_step(step_def, {}, trace)

    step_trace = trace.steps[0]
    assert len(step_trace.attempts) == 3
    assert step_trace.attempts[0].error == "Fail"
    assert step_trace.attempts[1].error == "Fail"
    assert step_trace.attempts[2].result_snippet is not None
    assert step_trace.attempts[2].error is None


def test_recursive_param_resolution(engine):
    """Verify list items with {{ref}} are resolved."""
    outputs = {"step_1": "value1"}
    params = {
        "list_mixed": ["static", "{{step_1.output}}", {"key": "{{step_1.output}}"}]
    }

    resolved = engine._resolve_params(params, outputs)
    assert resolved["list_mixed"][0] == "static"
    assert resolved["list_mixed"][1] == "value1"
    assert resolved["list_mixed"][2]["key"] == "value1"


def test_metadata_normalization(engine):
    """Verify dict metadata is converted to SkillMetadata model."""

    class LegacySkill:
        name = "legacy"
        metadata = {
            "name": "legacy",
            "description": "legacy desc",
            "risk_level": "medium",
            "capabilities": ["code"],
        }
        manifest = None

    skill = LegacySkill()
    engine.register_skill(skill)

    assert isinstance(engine.skills["legacy"].metadata, SkillMetadata)
    assert engine.skills["legacy"].metadata.risk_level == "medium"


# ===========================================================================
# Round 3 Tests
# ===========================================================================
def test_string_template_single_ref(engine):
    """Full reference returns raw value (type preserved)."""
    outputs = {"step_1": {"key": "val"}}
    result = engine._resolve_string_template("{{step_1.output}}", outputs)
    assert result == {"key": "val"}  # dict, not string


def test_string_template_mixed_text(engine):
    """Mixed text + reference is interpolated as string."""
    outputs = {"step_1": "world"}
    result = engine._resolve_string_template("hello {{step_1.output}}!", outputs)
    assert result == "hello world!"


def test_string_template_multiple_refs(engine):
    """Multiple references in one string."""
    outputs = {"a": "X", "b": "Y"}
    result = engine._resolve_string_template("{{a.output}} and {{b.output}}", outputs)
    assert result == "X and Y"


def test_string_template_unresolved(engine):
    """Unresolved reference stays as-is."""
    outputs = {}
    result = engine._resolve_string_template("{{unknown.output}}", outputs)
    assert result == "{{unknown.output}}"


def test_string_template_nested_field(engine):
    """Nested field access: {{step.output.key}}."""
    outputs = {"step_1": {"name": "Alice"}}
    result = engine._resolve_string_template("{{step_1.output.name}}", outputs)
    assert result == "Alice"


def test_metadata_cloning_no_mutation(engine):
    """Verify register_skill doesn't mutate original dict metadata."""

    original_dict = {
        "name": "test",
        "description": "test desc",
        "risk_level": "low",
        "capabilities": ["api"],
    }
    original_copy = dict(original_dict)

    class Skill:
        name = "test"
        metadata = original_dict
        manifest = None

    skill = Skill()
    engine.register_skill(skill)

    # Original dict should not be altered
    assert original_dict == original_copy
    # But engine's version is a SkillMetadata instance
    assert isinstance(engine.skills["test"].metadata, SkillMetadata)


def test_ctx_is_retry_property():
    """Verify is_retry is False on attempt 1, True on attempt > 1."""
    ctx = SkillExecutionContext(
        request_id="r1",
        step_id="s1",
        skill_name="test",
        params={},
        state=MagicMock(),
        skill=MagicMock(),
        metadata=SkillMetadata(name="test", capabilities=["reasoning"]),
        trace=MagicMock(),
        attempt=1,
    )
    assert ctx.is_retry is False

    ctx.attempt = 2
    assert ctx.is_retry is True


def test_ctx_timeout_public():
    """Verify timeout is a public field with default 0."""
    ctx = SkillExecutionContext(
        request_id="r1",
        step_id="s1",
        skill_name="test",
        params={},
        state=MagicMock(),
        skill=MagicMock(),
        metadata=SkillMetadata(name="test", capabilities=["reasoning"]),
        trace=MagicMock(),
    )
    assert ctx.timeout == 0.0
    ctx.timeout = 30.0
    assert ctx.timeout == 30.0
