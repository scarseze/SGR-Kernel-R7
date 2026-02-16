"""KC-D: Step execution pipeline tests — middleware ordering, metadata, validation."""
import pytest
from unittest.mock import AsyncMock
from core.middleware import SkillMiddleware
from tests.kernel.conftest import _EngineStub, make_skill, make_step
from core.types import SkillMetadata
from unittest.mock import MagicMock


class TestStepPipeline:

    @pytest.mark.asyncio
    async def test_kc_d1_middleware_order(self):
        """KC-D1: before=forward, after=reversed."""
        call_log = []

        class OrderedMW(SkillMiddleware):
            def __init__(self, name):
                self._name = name

            async def before_execute(self, ctx):
                call_log.append(f"before:{self._name}")

            async def after_execute(self, ctx, result):
                call_log.append(f"after:{self._name}")
                return result

        engine = _EngineStub()
        engine.middlewares = [
            OrderedMW("Trace"), OrderedMW("Policy"),
            OrderedMW("Approval"), OrderedMW("Timeout"),
        ]
        del engine.security  # avoid interference

        skill = make_skill()
        engine.skills["test_skill"] = skill

        trace = MagicMock()
        trace.request_id = "r1"
        trace.steps = []

        await engine._execute_step(make_step(), {}, trace)

        before = [c for c in call_log if c.startswith("before:")]
        after = [c for c in call_log if c.startswith("after:")]

        assert before == [
            "before:Trace", "before:Policy",
            "before:Approval", "before:Timeout",
        ]
        assert after == [
            "after:Timeout", "after:Approval",
            "after:Policy", "after:Trace",
        ]

    def test_kc_d2_metadata_normalization(self, engine):
        """KC-D2: Dict metadata auto-converted to SkillMetadata."""
        skill = MagicMock()
        skill.name = "raw_skill"
        skill.metadata = {
            "name": "raw_skill", "capabilities": ["reasoning"],
            "description": "test",
        }
        engine.register_skill(skill)
        assert isinstance(engine.skills["raw_skill"].metadata, SkillMetadata)
