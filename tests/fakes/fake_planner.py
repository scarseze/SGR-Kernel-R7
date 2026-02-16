"""FakePlanner — returns pre-configured plans without LLM calls."""
from core.planner import ExecutionPlan, PlanStep


class FakePlanner:
    def __init__(self, plan=None, direct_response=None):
        self.plan = plan
        self.direct_response = direct_response
        self.create_plan_calls = 0
        self.repair_plan_calls = 0

    async def create_plan(self, text, skills, history):
        self.create_plan_calls += 1

        if self.direct_response:
            return ExecutionPlan(
                steps=[], reasoning="direct",
                direct_response=self.direct_response
            ), {"model": "fake", "total_cost": 0.0}

        if self.plan:
            return self.plan, {"model": "fake", "total_cost": 0.0}

        step = PlanStep(
            step_id="s1", skill_name="fake",
            description="default step",
            params={"x": 1}, depends_on=[]
        )
        return ExecutionPlan(
            steps=[step], reasoning="default"
        ), {"model": "fake", "total_cost": 0.0}

    async def repair_plan(self, *a, **k):
        self.repair_plan_calls += 1
        return None
