"""
Kernel Hardening R7 — Architectural Risks & Features
Covers: StepTrace timing, Budget-per-attempt, Skill Concurrency, DAG Topo Trace, Plan Hash.
"""
import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

from core.types import SkillMetadata, Capability, RetryPolicy, StepStatus
from core.result import StepResult
from tests.kernel.conftest import make_skill
from tests.fakes.fake_skill import FakeSkill
from tests.harness.kernel_rig import KernelTestRig
from core.dag_executor import DAGExecutor, StepNode, DAGResult

# ───────────────────────── R1: StepTrace Timing ──────────────────────────────

@pytest.mark.asyncio
async def test_steptrace_timing_populated():
    """R1: start_time and duration are set on execution."""
    skill = FakeSkill(name="fake", sleep_sec=0.1)
    rig = KernelTestRig().with_skill(skill)
    
    result = await rig.run("test timing")
    
    assert rig.engine.tracer.last_trace() is not None
    trace = rig.engine.tracer.last_trace()
    assert len(trace.steps) > 0
    step_trace = trace.steps[0]
    
    assert step_trace.start_time > 0
    assert step_trace.duration >= 0.1  # at least sleep time
    assert len(step_trace.attempts) == 1
    assert step_trace.attempts[0].start_time > 0
    assert step_trace.attempts[0].end_time > step_trace.attempts[0].start_time


# ────────────────────── R3: Budget Per Attempt ──────────────────────────────

@pytest.mark.asyncio
async def test_budget_recorded_per_attempt():
    """R3: Cost recorded for failed attempts too."""
    # First attempt fails, second succeeds. Cost should be recorded twice.
    # We mock policy.record_step_cost
    
    mock_policy = MagicMock()
    mock_policy.check_budget.return_value = True
    
    # We need to inject this policy into engine
    # Rig doesn't easily expose replacing policy, so we monkeypatch engine.policy
    
    skill = FakeSkill(name="fake", fail_times=1) # fails once, then succeeds
    skill.metadata.estimated_cost = 1.0
    skill.metadata.retry_policy = RetryPolicy.STANDARD # allows retry
    
    rig = KernelTestRig().with_skill(skill)
    rig.engine.policy = mock_policy
    
    await rig.run("test budget")
    
    # record_step_cost(step_id, skill_name, cost)
    # fail_times=1 means: 
    # Attempt 1: Fails. catch block -> record cost (if enabled in catch)
    # Attempt 2: Succeeds. success block -> record cost.
    # Total calls: 2.
    
    assert mock_policy.record_step_cost.call_count == 2


# ────────────────────── F2: Skill Concurrency Guard ─────────────────────────

@pytest.mark.asyncio
async def test_skill_concurrency_serialization():
    """F2: max_concurrency=1 serialized execution."""
    # We run 2 parallel calls to the same skill via DAGExecutor directly (or just execute_step concurrently)
    # Using DAGExecutor is easier to simulate parallel launch
    
    slow_skill = FakeSkill(name="slow_serial", sleep_sec=0.2)
    slow_skill.metadata.max_concurrency = 1
    
    rig = KernelTestRig().with_skill(slow_skill)
    
    # Create 2 steps using this skill
    from types import SimpleNamespace
    step1 = SimpleNamespace(
        step_id="s1", 
        skill_name="slow_serial", 
        params={}, 
        depends_on=set(),
        skill_name_for_log="slow_serial"
    )
    step2 = SimpleNamespace(
        step_id="s2", 
        skill_name="slow_serial", 
        params={}, 
        depends_on=set(),
        skill_name_for_log="slow_serial"
    )
    
    executor = DAGExecutor(
        steps=[step1, step2],
        execute_fn=rig.engine._execute_step,
        budget_check_fn=lambda: True,
        max_concurrent=5 # DAG allows 5, but skill allows 1
    )
    
    from core.trace import RequestTrace
    trace = RequestTrace(user_request="test_concurrency")
    
    start_time = time.time()
    result = await executor.run({}, trace)
    duration = time.time() - start_time
    
    # If parallel: ~0.2s. If serial: ~0.4s.
    assert duration >= 0.4
    assert result.success


# ────────────────────── F3: DAG Topo Trace ──────────────────────────────────

@pytest.mark.asyncio
async def test_dag_execution_order_trace():
    """F3: DAGResult records execution start order."""
    # s1 -> s2. Order must be [s1, s2]
    # s3 (independent). Order could be [s1, s3, s2] or [s3, s1, s2].
    
    skill = FakeSkill(name="noop")
    rig = KernelTestRig().with_skill(skill)
    
    from types import SimpleNamespace
    s1 = SimpleNamespace(step_id="s1", skill_name="noop", params={}, depends_on=set())
    s2 = SimpleNamespace(step_id="s2", skill_name="noop", params={}, depends_on={"s1"})
    
    executor = DAGExecutor(
        steps=[s1, s2],
        execute_fn=rig.engine._execute_step,
        budget_check_fn=lambda: True
    )
    
    from core.trace import RequestTrace
    trace = RequestTrace(user_request="test_topo")
    
    result = await executor.run({}, trace)
    
    assert result.execution_order == ["s1", "s2"]


# ────────────────────── F4: Plan Hash ───────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_hash_recorded():
    """F4: Trace has plan_hash populated."""
    rig = KernelTestRig().with_skill(FakeSkill(name="test"))
    
    # We need strict=False because we can't easily mock Planner response perfectly 
    # without full setup, but Rig handles basic planning.
    # Rig.run uses the real engine.run which calls _create_plan.
    # We just need to check trace after run.
    
    await rig.run("do something")
    
    trace = rig.engine.tracer.last_trace()
    assert trace.plan_hash is not None
    assert len(trace.plan_hash) > 0


# ────────────────────── F1: Runtime Invariants ──────────────────────────────

@pytest.mark.asyncio
async def test_runtime_invariant_fails_negative_timeout():
    """F1: Negative timeout triggers assertion error."""
    # We can force this by mocking ctx.timeout via a malicious middleware 
    # or just calling _execute_step with bad ctx manually (harder).
    # Easier: Mock middleware to set ctx.timeout = -1
    
    from core.middleware import SkillMiddleware
    class SaboteurMiddleware(SkillMiddleware):
        async def before_execute(self, ctx):
            ctx.timeout = -1.0
            
    skill = FakeSkill(name="fake")
    rig = KernelTestRig().with_skill(skill)
    # Must append to run AFTER TimeoutMiddleware (hardened) to test invariant
    rig.engine.middlewares.append(SaboteurMiddleware())
    
    # Expect assertion error caught by engine
    await rig.run("test invariant")
    
    last_step = rig.last_step
    assert last_step is not None
    assert last_step.status == StepStatus.FAILED.value
    assert "KC-F3" in str(last_step.error)
