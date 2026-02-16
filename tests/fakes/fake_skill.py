"""
FakeSkill — controllable skill for kernel tests.
Supports configurable failures, sleep, cost, and retry policy.
"""
import asyncio
from pydantic import BaseModel
from skills.base import BaseSkill
from core.types import SkillMetadata, RetryPolicy, RiskLevel, CostClass, Capability


class FakeInput(BaseModel):
    x: int = 0


class FakeSkill(BaseSkill):
    """Deterministic skill for testing. Fails `fail_times`, then succeeds."""

    def __init__(self, *, name="fake", fail_times=0, sleep_sec=0,
                 retry=RetryPolicy.STANDARD, timeout=5.0,
                 estimated_cost=0.0, risk=RiskLevel.LOW,
                 capabilities=None):
        self._name = name
        self.fail_times = fail_times
        self.sleep_sec = sleep_sec
        self.calls = 0
        self._metadata = SkillMetadata(
            name=name,
            description=f"FakeSkill({name})",
            capabilities=capabilities or [Capability.REASONING],
            retry_policy=retry,
            risk_level=risk,
            timeout_sec=timeout,
            estimated_cost=estimated_cost,
        )
        self._input_schema = FakeInput

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"FakeSkill({self._name})"

    @property
    def metadata(self) -> SkillMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def input_schema(self):
        return self._input_schema

    async def execute(self, inp, state):
        self.calls += 1

        if self.sleep_sec:
            await asyncio.sleep(self.sleep_sec)

        if self.calls <= self.fail_times:
            raise RuntimeError(f"planned failure #{self.calls}")

        return f"ok:{inp.x}"
