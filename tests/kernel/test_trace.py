"""KC-K: Trace integrity tests — single append, attempt count, save on error."""
import pytest
from unittest.mock import AsyncMock, patch
from core.types import RetryPolicy, StepStatus
from tests.kernel.conftest import make_skill, make_step


class TestTraceIntegrity:

    @pytest.mark.asyncio
    async def test_kc_k1_single_append(self, engine, trace):
        """Even with retries → one StepTrace per step_id."""
        skill = make_skill(retry=RetryPolicy.STANDARD)
        skill.execute = AsyncMock(
            side_effect=[RuntimeError("e1"), RuntimeError("e2"), "ok"])
        engine.skills["test_skill"] = skill

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await engine._execute_step(make_step(), {}, trace)

        ids = [s.step_id for s in trace.steps]
        assert ids.count("s1") == 1

    @pytest.mark.asyncio
    async def test_kc_k2_attempt_count_matches(self, engine, trace):
        """AttemptTrace list length == number of tries."""
        skill = make_skill(retry=RetryPolicy.STANDARD)
        skill.execute = AsyncMock(
            side_effect=[RuntimeError("e1"), RuntimeError("e2"), "ok"])
        engine.skills["test_skill"] = skill

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await engine._execute_step(make_step(), {}, trace)

        assert len(trace.steps[0].attempts) == 3

    @pytest.mark.asyncio
    async def test_kc_k3_trace_saved_on_exception(self, engine, trace):
        """Trace saved even when step raises."""
        skill = make_skill(retry=RetryPolicy.NONE)
        skill.execute = AsyncMock(side_effect=RuntimeError("crash"))
        engine.skills["test_skill"] = skill

        with pytest.raises(RuntimeError):
            await engine._execute_step(make_step(), {}, trace)

        assert len(trace.steps) == 1
        assert trace.steps[0].status == StepStatus.FAILED.value
