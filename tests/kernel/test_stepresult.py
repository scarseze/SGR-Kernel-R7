"""KC-H: StepResult contract tests — structured output, legacy wrapping, trace preview."""
import pytest
import json
from unittest.mock import AsyncMock
from core.result import StepResult
from core.types import StepStatus
from tests.kernel.conftest import make_skill, make_step


class TestStepResultContract:

    @pytest.mark.asyncio
    async def test_kc_h1_data_stored(self, engine, trace):
        """KC-H1: StepResult.data stored in step_outputs."""
        data = {"key": "value", "nested": [1, 2, 3]}
        skill = make_skill()
        skill.execute = AsyncMock(
            return_value=StepResult(data=data, output_text="Summary"))
        engine.skills["test_skill"] = skill
        outputs = {}

        result = await engine._execute_step(make_step(), outputs, trace)
        assert outputs["s1"] == data
        assert result.output_text == "Summary"

    @pytest.mark.asyncio
    async def test_kc_h2_legacy_string_wrapped(self, engine, trace):
        """KC-H2: str → auto-wrapped in StepResult."""
        skill = make_skill()
        skill.execute = AsyncMock(return_value="plain string")
        engine.skills["test_skill"] = skill
        outputs = {}

        result = await engine._execute_step(make_step(), outputs, trace)
        assert isinstance(result, StepResult)
        assert result.data == "plain string"
        assert result.status == StepStatus.COMPLETED

    def test_trace_preview_json(self):
        """trace_preview uses json.dumps for dict data."""
        r = StepResult(data={"key": "val"}, output_text="hi")
        parsed = json.loads(r.trace_preview())
        assert parsed["key"] == "val"

    def test_trace_preview_string(self):
        """trace_preview returns raw string for str data."""
        r = StepResult(data="hello", output_text="hello")
        assert r.trace_preview() == "hello"

    def test_trace_preview_truncated(self):
        """trace_preview respects max_len."""
        r = StepResult(data="x" * 5000, output_text="big")
        assert len(r.trace_preview(max_len=100)) == 100
