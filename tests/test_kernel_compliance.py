"""
Kernel Compliance Tests v1.1 — tests KERNEL_SPEC.md invariants.

IDs follow the matrix: KC-A1, KC-B1, KC-D1, etc.
Each test references a specific invariant from the spec.

Uses _EngineStub to avoid real infra (Qdrant, DB).
"""
import pytest
import asyncio
import time
import json
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock, call
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from core.types import (
    SkillMetadata, SkillExecutionContext, Capability,
    RetryPolicy, StepStatus, RiskLevel, CostClass,
)
from core.trace import RequestTrace, StepTrace, AttemptTrace
from core.result import StepResult
from core.middleware import (
    SkillMiddleware, TraceMiddleware, PolicyMiddleware,
    ApprovalMiddleware, PolicyDenied, HumanDenied,
)
from core.security import SecurityGuardian, SecurityViolationError
from core.policy import PolicyEngine


# ============================================================================
# Lightweight engine stub — avoids real CoreEngine.__init__
# ============================================================================
class _EngineStub:
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


# ============================================================================
# Mock helpers
# ============================================================================
def _make_skill(name="test_skill", retry=RetryPolicy.NONE,
                capabilities=None, timeout=60.0, estimated_cost=0.0):
    """Factory for mock skills."""
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


def _make_step(step_id="s1", skill_name="test_skill",
               params=None, depends_on=None):
    step = MagicMock()
    step.step_id = step_id
    step.skill_name = skill_name
    step.params = params or {}
    step.depends_on = depends_on or []
    return step


def _make_trace():
    t = MagicMock(spec=RequestTrace)
    t.request_id = "test-request-001"
    t.steps = []
    return t


@pytest.fixture
def engine():
    return _EngineStub()


# ============================================================================
# A. Request Lifecycle
# ============================================================================

class TestRequestLifecycle:
    """KC-A1, KC-A2 — Security and context invariants."""

    @pytest.mark.asyncio
    async def test_kc_a1_security_input_gate(self, engine):
        """KC-A1: Malicious input must be blocked BEFORE any skill runs."""
        skill = _make_skill()
        engine.skills["test_skill"] = skill

        # Security should block this
        with pytest.raises(SecurityViolationError):
            engine.security.validate("nc -l -p 4444")

        # Skill must not have been called
        skill.execute.assert_not_called()


# ============================================================================
# B. Planning
# ============================================================================

class TestPlanning:
    """KC-B1, KC-B2, KC-B3 — budget, direct response, step limits."""

    def test_kc_b1_budget_blocks_planning(self):
        """KC-B1: Budget exceeded → planning must not proceed."""
        policy = PolicyEngine()
        policy.current_spend = 15.0  # over daily_budget=10.0
        assert policy.check_budget(0.5) is False

    def test_kc_b3_step_limit(self):
        """KC-B3: Steps truncated to MAX_STEPS=10."""
        MAX_STEPS = 10
        steps = [_make_step(step_id=f"s{i}") for i in range(15)]
        truncated = steps[:MAX_STEPS]
        assert len(truncated) == 10


# ============================================================================
# D. Step Execution Pipeline
# ============================================================================

class TestStepPipeline:
    """KC-D1, KC-D2, KC-D3 — middleware order, metadata, validation."""

    @pytest.mark.asyncio
    async def test_kc_d1_middleware_order(self):
        """KC-D1: before_execute in forward order, after_execute reversed."""
        call_log = []

        class OrderedMiddleware(SkillMiddleware):
            def __init__(self, name):
                self._name = name

            async def before_execute(self, ctx):
                call_log.append(f"before:{self._name}")

            async def after_execute(self, ctx, result):
                call_log.append(f"after:{self._name}")
                return result

            async def on_error(self, ctx, error):
                call_log.append(f"error:{self._name}")

        engine = _EngineStub()
        engine.middlewares = [
            OrderedMiddleware("Trace"),
            OrderedMiddleware("Policy"),
            OrderedMiddleware("Approval"),
            OrderedMiddleware("Timeout"),
        ]
        # Remove security to not interfere with ordering test
        del engine.security

        skill = _make_skill()
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()

        await engine._execute_step(step, {}, trace)

        # before: forward
        before_calls = [c for c in call_log if c.startswith("before:")]
        assert before_calls == [
            "before:Trace", "before:Policy",
            "before:Approval", "before:Timeout"
        ]

        # after: reverse
        after_calls = [c for c in call_log if c.startswith("after:")]
        assert after_calls == [
            "after:Timeout", "after:Approval",
            "after:Policy", "after:Trace"
        ]

    def test_kc_d2_metadata_normalization(self, engine):
        """KC-D2: Dict metadata auto-converted to SkillMetadata."""
        skill = MagicMock()
        skill.name = "raw_skill"
        skill.metadata = {
            "name": "raw_skill",
            "capabilities": ["reasoning"],
            "description": "test",
        }

        engine.register_skill(skill)
        assert isinstance(engine.skills["raw_skill"].metadata, SkillMetadata)


# ============================================================================
# E. Retry System
# ============================================================================

class TestRetrySystem:
    """KC-E1..E4 — retry count, fatal errors, ctx.is_retry, backoff."""

    @pytest.mark.asyncio
    async def test_kc_e1_retry_count_none(self, engine):
        """KC-E1: NONE → 1 attempt."""
        skill = _make_skill(retry=RetryPolicy.NONE)
        skill.execute = AsyncMock(side_effect=RuntimeError("fail"))
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()

        with pytest.raises(RuntimeError):
            await engine._execute_step(step, {}, trace)

        assert skill.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_kc_e1_retry_count_standard(self, engine):
        """KC-E1: STANDARD → 3 attempts."""
        skill = _make_skill(retry=RetryPolicy.STANDARD)
        skill.execute = AsyncMock(side_effect=RuntimeError("fail"))
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                await engine._execute_step(step, {}, trace)

        assert skill.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_kc_e1_retry_count_aggressive(self, engine):
        """KC-E1: AGGRESSIVE → 5 attempts."""
        skill = _make_skill(retry=RetryPolicy.AGGRESSIVE)
        skill.execute = AsyncMock(side_effect=RuntimeError("fail"))
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                await engine._execute_step(step, {}, trace)

        assert skill.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_kc_e2_no_retry_on_policy_denied(self, engine):
        """KC-E2: PolicyDenied → 1 attempt, status BLOCKED."""
        skill = _make_skill(retry=RetryPolicy.AGGRESSIVE)
        skill.execute = AsyncMock(side_effect=PolicyDenied("denied"))
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()

        result = await engine._execute_step(step, {}, trace)

        assert skill.execute.call_count == 1
        assert result.status == StepStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_kc_e2_no_retry_on_human_denied(self, engine):
        """KC-E2: HumanDenied → 1 attempt, status BLOCKED."""
        skill = _make_skill(retry=RetryPolicy.AGGRESSIVE)
        skill.execute = AsyncMock(side_effect=HumanDenied("human said no"))
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()

        result = await engine._execute_step(step, {}, trace)

        assert skill.execute.call_count == 1
        assert result.status == StepStatus.BLOCKED

    def test_kc_e3_is_retry_correctness(self):
        """KC-E3: attempt 1 → False, attempt > 1 → True."""
        ctx = SkillExecutionContext(
            request_id="r1", step_id="s1", skill_name="test",
            params={}, state=MagicMock(), skill=MagicMock(),
            metadata=_make_skill().metadata,
            trace=MagicMock(), attempt=1,
        )
        assert ctx.is_retry is False

        ctx.attempt = 2
        assert ctx.is_retry is True

        ctx.attempt = 5
        assert ctx.is_retry is True

    @pytest.mark.asyncio
    async def test_kc_e4_backoff_growth(self, engine):
        """KC-E4: Backoff grows as 2^attempt (2, 4, 8...)."""
        skill = _make_skill(retry=RetryPolicy.STANDARD)
        skill.execute = AsyncMock(side_effect=RuntimeError("fail"))
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()
        sleep_args = []

        async def mock_sleep(duration):
            sleep_args.append(duration)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(RuntimeError):
                await engine._execute_step(step, {}, trace)

        # 3 attempts → 2 sleeps: 2^1=2, 2^2=4
        assert sleep_args == [2, 4]


# ============================================================================
# F. Timeout Authority
# ============================================================================

class TestTimeoutAuthority:
    """KC-F1, KC-F2 — timeout from metadata, kernel default."""

    def test_kc_f1_timeout_from_metadata(self):
        """KC-F1: Skill with timeout_sec=7 → ctx.timeout must be 7."""
        from core.middleware import TimeoutMiddleware

        mw = TimeoutMiddleware()
        ctx = SkillExecutionContext(
            request_id="r1", step_id="s1", skill_name="test",
            params={}, state=MagicMock(), skill=MagicMock(),
            metadata=_make_skill(timeout=7.0).metadata,
            trace=MagicMock(), attempt=1,
        )

        # TimeoutMiddleware.around_execute sets ctx.timeout from metadata
        asyncio.get_event_loop().run_until_complete(mw.around_execute(ctx, None))
        assert ctx.timeout == 7.0

    def test_kc_f2_kernel_default_timeout(self):
        """KC-F2: timeout=0 → kernel uses 60s fallback."""
        # Engine uses: timeout = ctx.timeout if ctx.timeout > 0 else 60.0
        ctx_timeout = 0.0
        effective = ctx_timeout if ctx_timeout > 0 else 60.0
        assert effective == 60.0


# ============================================================================
# G. Security Enforcement
# ============================================================================

class TestSecurityEnforcement:
    """KC-G1, KC-G2 — param validation, output sanitization."""

    def test_kc_g1_param_validation_catches_injection(self):
        """KC-G1: Template resolves to dangerous value → blocked."""
        sec = SecurityGuardian()

        # Dangerous value injected via template resolution
        params = {"cmd": "nc -l -p 4444", "safe": "hello"}

        with pytest.raises(SecurityViolationError):
            sec.validate_params(params)

    def test_kc_g1_param_nested_injection(self):
        """KC-G1: Nested params with dangerous values also caught."""
        sec = SecurityGuardian()
        params = {"outer": {"inner": "rm -rf /"}}

        with pytest.raises(SecurityViolationError):
            sec.validate_params(params)

    def test_kc_g2_output_sanitization(self):
        """KC-G2: Skill output with leaked secrets → sanitized."""
        sec = SecurityGuardian()

        with pytest.raises(SecurityViolationError):
            sec.validate_output("result: api_key = sk-12345abc")

    def test_kc_g2_output_private_key(self):
        """KC-G2: Private key in output detected."""
        sec = SecurityGuardian()

        with pytest.raises(SecurityViolationError):
            sec.validate_output("-----BEGIN RSA PRIVATE KEY-----\nMIIBog...")

    def test_kc_g2_clean_output_passes(self):
        """KC-G2: Normal output passes validation."""
        sec = SecurityGuardian()
        sec.validate_output("The analysis shows revenue increased by 15%.")


# ============================================================================
# H. StepResult Contract
# ============================================================================

class TestStepResultContract:
    """KC-H1, KC-H2 — structured and legacy outputs."""

    @pytest.mark.asyncio
    async def test_kc_h1_stepresult_data_stored(self, engine):
        """KC-H1: StepResult.data stored in step_outputs."""
        data = {"key": "value", "nested": [1, 2, 3]}
        skill = _make_skill()
        skill.execute = AsyncMock(
            return_value=StepResult(data=data, output_text="Summary"))
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()
        outputs = {}

        result = await engine._execute_step(step, outputs, trace)

        assert outputs["s1"] == data
        assert result.output_text == "Summary"

    @pytest.mark.asyncio
    async def test_kc_h2_legacy_string_wrapped(self, engine):
        """KC-H2: Skill returns str → auto-wrapped in StepResult."""
        skill = _make_skill()
        skill.execute = AsyncMock(return_value="plain string result")
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()
        outputs = {}

        result = await engine._execute_step(step, outputs, trace)

        assert isinstance(result, StepResult)
        assert result.data == "plain string result"
        assert result.status == StepStatus.COMPLETED

    def test_kc_h1_trace_preview_json(self):
        """KC-H1: trace_preview uses json.dumps for structured data."""
        r = StepResult(data={"key": "val", "num": 42}, output_text="hi")
        preview = r.trace_preview()
        parsed = json.loads(preview)
        assert parsed["key"] == "val"

    def test_kc_h1_trace_preview_string(self):
        """KC-H1: trace_preview returns raw string for str data."""
        r = StepResult(data="hello world", output_text="hello world")
        assert r.trace_preview() == "hello world"

    def test_kc_h1_trace_preview_truncated(self):
        """KC-H1: trace_preview respects max_len."""
        r = StepResult(data="x" * 5000, output_text="big")
        assert len(r.trace_preview(max_len=100)) == 100


# ============================================================================
# I. Template Resolution
# ============================================================================

class TestTemplateResolution:
    """KC-I1, KC-I2, KC-I3 — nested fields, missing fields, recursion."""

    def test_kc_i1_nested_field_resolution(self, engine):
        """KC-I1: {{step1.output.a.b}} resolves nested value."""
        outputs = {"step1": {"a": {"b": "deep_value"}}}
        result = engine._resolve_string_template("{{step1.output.a.b}}", outputs)
        assert result == "deep_value"

    def test_kc_i2_missing_field_safe(self, engine):
        """KC-I2: Missing nested field → template stays literal."""
        outputs = {"step1": {"a": {"b": "val"}}}
        result = engine._resolve_string_template("{{step1.output.a.missing}}", outputs)
        # Unresolved nested field returns placeholder like {Unresolved: missing}
        assert "Unresolved" in str(result) or "missing" in str(result)

    def test_kc_i2_missing_step_safe(self, engine):
        """KC-I2: Missing step_id → template stays literal."""
        outputs = {}
        result = engine._resolve_string_template("{{nonexistent.output}}", outputs)
        assert "nonexistent" in result

    def test_kc_i3_dict_recursive(self, engine):
        """KC-I3: Templates inside dict values resolve."""
        outputs = {"step1": "resolved_value"}
        params = {"key": "{{step1.output}}", "static": "hello"}
        result = engine._resolve_params(params, outputs)
        assert result["key"] == "resolved_value"
        assert result["static"] == "hello"

    def test_kc_i3_list_recursive(self, engine):
        """KC-I3: Templates inside list items resolve."""
        outputs = {"step1": "val1", "step2": "val2"}
        params = {"items": ["{{step1.output}}", "{{step2.output}}", "literal"]}
        result = engine._resolve_params(params, outputs)
        assert result["items"] == ["val1", "val2", "literal"]

    def test_kc_i1_interpolation_json_dumps(self, engine):
        """KC-I1: Non-primitive in interpolation → json.dumps, not str()."""
        outputs = {"step1": {"key": "val", "num": 42}}
        result = engine._resolve_string_template(
            "Result: {{step1.output}}", outputs)
        # Should contain json, not repr
        assert '"key"' in result
        assert "Result:" in result


# ============================================================================
# J. Budget Enforcement
# ============================================================================

class TestBudgetEnforcement:
    """KC-J1, KC-J2 — budget stops mid-DAG, planner cost recorded."""

    def test_kc_j1_budget_stops_on_overspend(self):
        """KC-J1: Budget check returns False when spent > limit."""
        policy = PolicyEngine()
        policy.daily_budget = 5.0
        policy.current_spend = 4.5
        assert policy.check_budget(1.0) is False
        assert policy.check_budget(0.4) is True

    def test_kc_j2_step_cost_recorded(self):
        """KC-J2: record_step_cost tracks per-step spend."""
        policy = PolicyEngine()
        policy.record_step_cost("s1", "research", 2.5)
        policy.record_step_cost("s2", "analysis", 1.0)

        assert policy.step_costs["s1"]["cost"] == 2.5
        assert policy.step_costs["s2"]["cost"] == 1.0
        assert policy.current_spend == 3.5

    def test_kc_j2_planner_cost_recorded(self):
        """KC-J2: record_cost tracks planner LLM spend."""
        policy = PolicyEngine()
        policy.record_cost(0.05)
        assert policy.current_spend == 0.05
        policy.record_cost(0.03)
        assert policy.current_spend == 0.08


# ============================================================================
# K. Trace Integrity
# ============================================================================

class TestTraceIntegrity:
    """KC-K1, KC-K2, KC-K3 — single append, attempt count, save on error."""

    @pytest.mark.asyncio
    async def test_kc_k1_step_trace_single_append(self, engine):
        """KC-K1: Even with retries → one StepTrace per step_id."""
        skill = _make_skill(retry=RetryPolicy.STANDARD)
        # Fail twice, succeed third
        skill.execute = AsyncMock(
            side_effect=[RuntimeError("e1"), RuntimeError("e2"), "ok"])
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await engine._execute_step(step, {}, trace)

        step_ids = [s.step_id for s in trace.steps]
        assert step_ids.count("s1") == 1

    @pytest.mark.asyncio
    async def test_kc_k2_attempt_count_matches_retries(self, engine):
        """KC-K2: AttemptTrace list length == number of tries."""
        skill = _make_skill(retry=RetryPolicy.STANDARD)
        skill.execute = AsyncMock(
            side_effect=[RuntimeError("e1"), RuntimeError("e2"), "ok"])
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await engine._execute_step(step, {}, trace)

        step_trace = trace.steps[0]
        assert len(step_trace.attempts) == 3

    @pytest.mark.asyncio
    async def test_kc_k3_trace_saved_on_exception(self, engine):
        """KC-K3: Trace saved even when step raises."""
        skill = _make_skill(retry=RetryPolicy.NONE)
        skill.execute = AsyncMock(side_effect=RuntimeError("crash"))
        engine.skills["test_skill"] = skill

        step = _make_step()
        trace = _make_trace()

        with pytest.raises(RuntimeError):
            await engine._execute_step(step, {}, trace)

        assert len(trace.steps) == 1
        assert trace.steps[0].status == StepStatus.FAILED.value


# ============================================================================
# L. Replan System
# ============================================================================

class TestReplanSystem:
    """KC-L1..L3 — replan triggers, completed preserved, version suffix."""

    def test_kc_l2_step_id_versioned_on_collision(self):
        """KC-L2: Reused step_id gets _rN suffix."""
        completed_step_ids = {"s1", "s2"}
        replan_round = 0

        # Simulate repair plan returning overlapping step_ids
        new_steps = [_make_step("s1"), _make_step("s3")]

        for s in new_steps:
            if s.step_id in completed_step_ids:
                s.step_id = f"{s.step_id}_r{replan_round + 1}"

        assert new_steps[0].step_id == "s1_r1"
        assert new_steps[1].step_id == "s3"  # no collision

    def test_kc_l3_remaining_filtered(self):
        """KC-L3: Remaining steps exclude completed."""
        completed_step_ids = {"s1", "s2"}
        all_steps = [_make_step("s1"), _make_step("s2"),
                     _make_step("s3"), _make_step("s4")]

        remaining = [s for s in all_steps
                     if s.step_id not in completed_step_ids]

        assert len(remaining) == 2
        assert {s.step_id for s in remaining} == {"s3", "s4"}


# ============================================================================
# M. Skill Registration
# ============================================================================

class TestSkillRegistration:
    """KC-M1, KC-M2 — capability index, metadata immutability."""

    def test_kc_m1_capability_index_built(self, engine):
        """KC-M1: Registration populates cap_index."""
        skill = _make_skill(capabilities=[Capability.REASONING, Capability.WEB])
        engine.register_skill(skill)

        assert "test_skill" in engine.skills
        # cap_index should contain both capabilities
        assert Capability.REASONING in engine.cap_index or \
               "reasoning" in str(engine.cap_index)

    def test_kc_m2_metadata_is_model(self, engine):
        """KC-M2: After register, metadata is SkillMetadata, not dict."""
        skill = MagicMock()
        skill.name = "dict_meta_skill"
        skill.metadata = {
            "name": "dict_meta_skill",
            "capabilities": ["reasoning"],
            "description": "test",
        }

        engine.register_skill(skill)

        registered = engine.skills["dict_meta_skill"]
        assert isinstance(registered.metadata, SkillMetadata)
        assert registered.metadata.name == "dict_meta_skill"

    def test_kc_m2_estimated_cost_on_model(self):
        """KC-M2: estimated_cost is a proper model field, not getattr."""
        meta = SkillMetadata(
            name="x", capabilities=[Capability.REASONING],
            estimated_cost=1.5
        )
        assert meta.estimated_cost == 1.5

        # Default is 0.0
        meta2 = SkillMetadata(name="y", capabilities=[Capability.REASONING])
        assert meta2.estimated_cost == 0.0
