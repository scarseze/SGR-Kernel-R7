"""
Tests for DAGExecutor — parallel async step execution with dependency resolution.
"""
import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock
from core.dag_executor import DAGExecutor, DAGResult, DAGError, StepNode
from core.types import StepStatus
from core.result import StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class FakeStep:
    """Minimal plan step with required fields."""
    def __init__(self, step_id, skill_name="mock_skill", depends_on=None):
        self.step_id = step_id
        self.skill_name = skill_name
        self.depends_on = depends_on or []


async def _mock_execute(step_def, step_outputs, trace, delay=0):
    """Simulates step execution with optional delay."""
    if delay:
        await asyncio.sleep(delay)
    result = StepResult(data=f"result_{step_def.step_id}", output_text=f"Done {step_def.step_id}")
    step_outputs[step_def.step_id] = result.data
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_single_step():
    """Trivial DAG with one step."""
    steps = [FakeStep("s1")]

    async def exec_fn(step_def, outputs, trace):
        return await _mock_execute(step_def, outputs, trace)

    executor = DAGExecutor(steps, exec_fn, budget_check_fn=lambda: True)
    result = await executor.run({}, MagicMock())

    assert result.success
    assert "s1" in result.completed
    assert len(result.failed) == 0


@pytest.mark.asyncio
async def test_parallel_independent_steps():
    """Two independent steps should run in parallel (total < 2x single)."""
    steps = [FakeStep("a"), FakeStep("b")]

    async def exec_fn(step_def, outputs, trace):
        return await _mock_execute(step_def, outputs, trace, delay=0.5)

    executor = DAGExecutor(steps, exec_fn, budget_check_fn=lambda: True)

    start = time.monotonic()
    result = await executor.run({}, MagicMock())
    elapsed = time.monotonic() - start

    assert result.success
    assert set(result.completed) == {"a", "b"}
    assert elapsed < 1.0, f"Expected parallel execution < 1s, got {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_dependency_ordering():
    """Step C depends on A and B. Should execute after both."""
    steps = [
        FakeStep("a"),
        FakeStep("b"),
        FakeStep("c", depends_on=["a", "b"]),
    ]

    execution_order = []

    async def exec_fn(step_def, outputs, trace):
        execution_order.append(step_def.step_id)
        return await _mock_execute(step_def, outputs, trace, delay=0.1)

    executor = DAGExecutor(steps, exec_fn, budget_check_fn=lambda: True)
    result = await executor.run({}, MagicMock())

    assert result.success
    # C must be after both A and B
    assert execution_order.index("c") > execution_order.index("a")
    assert execution_order.index("c") > execution_order.index("b")


@pytest.mark.asyncio
async def test_cascade_failure():
    """If A fails, B (depends on A) should be cancelled."""
    steps = [
        FakeStep("a"),
        FakeStep("b", depends_on=["a"]),
    ]

    async def exec_fn(step_def, outputs, trace):
        if step_def.step_id == "a":
            raise ValueError("Step A crashed")
        return await _mock_execute(step_def, outputs, trace)

    executor = DAGExecutor(steps, exec_fn, budget_check_fn=lambda: True)
    result = await executor.run({}, MagicMock())

    assert not result.success
    assert "a" in result.failed
    assert "b" in result.cancelled


@pytest.mark.asyncio
async def test_budget_stop():
    """Budget exhaustion should prevent step execution."""
    steps = [FakeStep("s1")]
    budget_calls = [False]  # budget exhausted

    async def exec_fn(step_def, outputs, trace):
        return await _mock_execute(step_def, outputs, trace)

    executor = DAGExecutor(steps, exec_fn, budget_check_fn=lambda: budget_calls[0])
    result = await executor.run({}, MagicMock())

    assert not result.success
    assert "s1" in result.failed
    assert len(result.completed) == 0


@pytest.mark.asyncio
async def test_concurrency_limit():
    """Semaphore should limit parallel execution."""
    steps = [FakeStep(f"s{i}") for i in range(6)]
    max_concurrent_observed = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def exec_fn(step_def, outputs, trace):
        nonlocal max_concurrent_observed, current_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > max_concurrent_observed:
                max_concurrent_observed = current_concurrent
        await asyncio.sleep(0.2)
        async with lock:
            current_concurrent -= 1
        return await _mock_execute(step_def, outputs, trace)

    executor = DAGExecutor(steps, exec_fn, budget_check_fn=lambda: True, max_concurrent=2)
    result = await executor.run({}, MagicMock())

    assert result.success
    assert max_concurrent_observed <= 2, f"Expected max 2 concurrent, got {max_concurrent_observed}"


def test_cycle_detection():
    """Cyclic dependencies should raise DAGError."""
    steps = [
        FakeStep("a", depends_on=["b"]),
        FakeStep("b", depends_on=["a"]),
    ]
    with pytest.raises(DAGError, match="cycle"):
        DAGExecutor(steps, AsyncMock(), budget_check_fn=lambda: True)


def test_missing_dependency():
    """Reference to unknown step should raise DAGError."""
    steps = [FakeStep("a", depends_on=["nonexistent"])]
    with pytest.raises(DAGError, match="unknown"):
        DAGExecutor(steps, AsyncMock(), budget_check_fn=lambda: True)


@pytest.mark.asyncio
async def test_diamond_dag():
    """
    Diamond: A → B, A → C, B → D, C → D.
    All should complete, D runs last.
    """
    steps = [
        FakeStep("a"),
        FakeStep("b", depends_on=["a"]),
        FakeStep("c", depends_on=["a"]),
        FakeStep("d", depends_on=["b", "c"]),
    ]

    execution_order = []

    async def exec_fn(step_def, outputs, trace):
        execution_order.append(step_def.step_id)
        return await _mock_execute(step_def, outputs, trace, delay=0.05)

    executor = DAGExecutor(steps, exec_fn, budget_check_fn=lambda: True)
    result = await executor.run({}, MagicMock())

    assert result.success
    assert set(result.completed) == {"a", "b", "c", "d"}
    # D must be last
    assert execution_order.index("d") > execution_order.index("b")
    assert execution_order.index("d") > execution_order.index("c")
    assert execution_order.index("a") == 0
