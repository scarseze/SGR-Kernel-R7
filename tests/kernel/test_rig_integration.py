"""
Rig-based kernel compliance tests — demonstrate KernelTestRig usage.

These tests exercise the FULL run() pipeline through FakeSkill + FakePlanner,
proving kernel invariants at integration level (not stub level).
"""
import pytest
from core.planner import PlanStep, ExecutionPlan
from core.types import RetryPolicy, StepStatus
from core.middleware import SkillMiddleware

from tests.harness.kernel_rig import KernelTestRig
from tests.fakes.fake_skill import FakeSkill


# ============================================================================
# E. Retry (via full pipeline)
# ============================================================================

@pytest.mark.asyncio
async def test_rig_retry_standard():
    """STANDARD → 3 attempts via full run()."""
    skill = FakeSkill(fail_times=2)
    rig = KernelTestRig().with_skill(skill)

    await rig.run("retry please")
    rig.assert_skill_calls(skill, 3)


@pytest.mark.asyncio
async def test_rig_retry_none_no_retry():
    """NONE → 1 attempt, fails."""
    skill = FakeSkill(fail_times=1, retry=RetryPolicy.NONE)
    rig = KernelTestRig().with_skill(skill)

    # Should complete (error handled) or raise
    await rig.run("fail once")
    rig.assert_skill_calls(skill, 1)


# ============================================================================
# F. Timeout (via full pipeline)
# ============================================================================

@pytest.mark.asyncio
async def test_rig_timeout_enforced():
    """Skill exceeding timeout_sec → step fails with timeout."""
    skill = FakeSkill(sleep_sec=5, timeout=0.5)

    rig = KernelTestRig().with_skill(skill)
    result = await rig.run("timeout test")

    # Engine wraps errors; the result should mention timeout or error
    result_lower = result.lower()
    assert ("timeout" in result_lower or "timed out" in result_lower
            or "error" in result_lower or "failed" in result_lower)


# ============================================================================
# B. Planning
# ============================================================================

@pytest.mark.asyncio
async def test_rig_direct_response_no_dag():
    """Direct response → no DAG execution."""
    skill = FakeSkill()
    rig = KernelTestRig().with_skill(skill).with_direct_response("Hello!")

    result = await rig.run("hi")
    assert result == "Hello!"
    rig.assert_skill_calls(skill, 0)


@pytest.mark.asyncio
async def test_rig_budget_blocks_planner():
    """Budget exceeded → planner not called."""
    rig = KernelTestRig().with_budget(False)

    result = await rig.run("anything")
    assert "budget" in result.lower() or "denied" in result.lower() \
           or rig.planner_calls == 0


# ============================================================================
# D. Middleware (via full pipeline)
# ============================================================================

@pytest.mark.asyncio
async def test_rig_middleware_order_probe():
    """Probe middleware sees before → after in correct order."""
    log = []

    class ProbeMW(SkillMiddleware):
        async def before_execute(self, ctx):
            log.append("before")
        async def after_execute(self, ctx, r):
            log.append("after")
            return r
        async def on_error(self, ctx, e):
            log.append("error")

    skill = FakeSkill()
    rig = (KernelTestRig()
           .without_security()
           .with_middlewares([ProbeMW()])
           .with_skill(skill))

    await rig.run("go")
    assert log == ["before", "after"]


# ============================================================================
# I. Template Resolution (direct)
# ============================================================================

def test_rig_template_resolution():
    """Nested template resolution via rig.engine."""
    rig = KernelTestRig()
    outputs = {"s1": {"x": {"y": 7}}}

    val = rig.engine._resolve_string_template("{{s1.output.x.y}}", outputs)
    assert val == 7


# ============================================================================
# H. StepResult (via run_step)
# ============================================================================

@pytest.mark.asyncio
async def test_rig_direct_step_execution():
    """run_step bypasses planner, runs single step."""
    skill = FakeSkill()
    step = PlanStep(
        step_id="s1", skill_name="fake",
        description="test", params={"x": 5}, depends_on=[],
    )

    rig = KernelTestRig().with_skill(skill)
    result = await rig.run_step(step)

    assert "ok" in str(result)
    rig.assert_skill_calls(skill, 1)


# ============================================================================
# Multi-step DAG (via full pipeline)
# ============================================================================

@pytest.mark.asyncio
async def test_rig_multi_step_dag():
    """Two sequential steps with dependency."""
    skill = FakeSkill()
    plan = ExecutionPlan(
        steps=[
            PlanStep(step_id="s1", skill_name="fake",
                     description="first", params={"x": 1}, depends_on=[]),
            PlanStep(step_id="s2", skill_name="fake",
                     description="second", params={"x": 2}, depends_on=["s1"]),
        ],
        reasoning="two steps"
    )

    rig = KernelTestRig().with_skill(skill).with_plan(plan)
    await rig.run("multi step")

    rig.assert_skill_calls(skill, 2)
