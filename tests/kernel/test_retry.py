"""KC-E: Retry system tests — count mapping, fatal errors, is_retry, backoff."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.types import RetryPolicy, StepStatus, SkillExecutionContext, Capability
from core.middleware import PolicyDenied, HumanDenied
from tests.kernel.conftest import make_skill, make_step


class TestRetrySystem:

    @pytest.mark.asyncio
    async def test_kc_e1_retry_count_none(self, engine, trace):
        """NONE → 1 attempt."""
        skill = make_skill(retry=RetryPolicy.NONE)
        skill.execute = AsyncMock(side_effect=RuntimeError("fail"))
        engine.skills["test_skill"] = skill

        with pytest.raises(RuntimeError):
            await engine._execute_step(make_step(), {}, trace)
        assert skill.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_kc_e1_retry_count_standard(self, engine, trace):
        """STANDARD → 3 attempts."""
        skill = make_skill(retry=RetryPolicy.STANDARD)
        skill.execute = AsyncMock(side_effect=RuntimeError("fail"))
        engine.skills["test_skill"] = skill

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                await engine._execute_step(make_step(), {}, trace)
        assert skill.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_kc_e1_retry_count_aggressive(self, engine, trace):
        """AGGRESSIVE → 5 attempts."""
        skill = make_skill(retry=RetryPolicy.AGGRESSIVE)
        skill.execute = AsyncMock(side_effect=RuntimeError("fail"))
        engine.skills["test_skill"] = skill

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                await engine._execute_step(make_step(), {}, trace)
        assert skill.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_kc_e2_no_retry_on_policy_denied(self, engine, trace):
        """PolicyDenied → 1 attempt, status BLOCKED."""
        skill = make_skill(retry=RetryPolicy.AGGRESSIVE)
        skill.execute = AsyncMock(side_effect=PolicyDenied("denied"))
        engine.skills["test_skill"] = skill

        result = await engine._execute_step(make_step(), {}, trace)
        assert skill.execute.call_count == 1
        assert result.status == StepStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_kc_e2_no_retry_on_human_denied(self, engine, trace):
        """HumanDenied → 1 attempt, status BLOCKED."""
        skill = make_skill(retry=RetryPolicy.AGGRESSIVE)
        skill.execute = AsyncMock(side_effect=HumanDenied("no"))
        engine.skills["test_skill"] = skill

        result = await engine._execute_step(make_step(), {}, trace)
        assert skill.execute.call_count == 1
        assert result.status == StepStatus.BLOCKED

    def test_kc_e3_is_retry_correctness(self):
        """attempt 1 → False, attempt > 1 → True."""
        ctx = SkillExecutionContext(
            request_id="r1", step_id="s1", skill_name="test",
            params={}, state=MagicMock(), skill=MagicMock(),
            metadata=make_skill().metadata, trace=MagicMock(), attempt=1,
        )
        assert ctx.is_retry is False
        ctx.attempt = 2
        assert ctx.is_retry is True
        ctx.attempt = 5
        assert ctx.is_retry is True

    @pytest.mark.asyncio
    async def test_kc_e4_backoff_growth(self, engine, trace):
        """Backoff: 2^1=2, 2^2=4 for STANDARD (3 attempts, 2 sleeps)."""
        skill = make_skill(retry=RetryPolicy.STANDARD)
        skill.execute = AsyncMock(side_effect=RuntimeError("fail"))
        engine.skills["test_skill"] = skill
        sleep_args = []

        async def mock_sleep(d):
            sleep_args.append(d)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(RuntimeError):
                await engine._execute_step(make_step(), {}, trace)
        assert sleep_args == [2, 4]
