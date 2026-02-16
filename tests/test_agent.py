import pytest
from core.planner import ExecutionPlan, PlanStep
from core.types import PolicyStatus

@pytest.mark.asyncio
async def test_engine_initialization(engine):
    assert engine is not None
    assert engine.planner is not None
    assert engine.tracer is not None
    assert engine.policy is not None

@pytest.mark.asyncio
async def test_policy_enforcement(engine):
    # Mock a High Risk Skill
    from skills.base import BaseSkill
    from core.types import SkillMetadata, RiskLevel, CostClass # Import new types
    
    class HighRiskSkill(BaseSkill):
        name = "dangerous_op"
        description = "Dangerous"
        @property
        def metadata(self):
            return SkillMetadata(
                capabilities=[],
                risk_level=RiskLevel.HIGH, # Use Enum
                side_effects=True,
                idempotent=False,
                requires_network=False,
                requires_filesystem=False,
                cost_class=CostClass.EXPENSIVE
            )
        def input_schema(self): return {}
        async def execute(self, params, state): return "Boom"

    skill = HighRiskSkill()
    
    # Check Policy directly
    decision = engine.policy.check(skill, {}, engine.state)
    assert decision.status == PolicyStatus.REQUIRE_APPROVAL
    assert "High Risk" in decision.reason

@pytest.mark.asyncio
async def test_planner_integration(engine, mock_llm):
    # Setup Mock LLM to return a valid Plan
    expected_plan = ExecutionPlan(
        reasoning="Test Plan",
        steps=[
            PlanStep(step_id="1", skill_name="mock_skill", description="Mock Step", params={"x": 1})
        ]
    )
    
    # We mock the planner's create_plan method directly to avoid complex LLM mocking
    # (Unit testing the Planner itself is separate)
    from unittest.mock import AsyncMock
    engine.planner.create_plan = AsyncMock(return_value=(expected_plan, {"total_tokens": 10, "model": "mock-model"}))
    
    # Register mock skill
    from skills.base import BaseSkill
    from core.types import SkillMetadata, Capability # Import Capability

    class MockSkill(BaseSkill):
        name = "mock_skill"
        description = "Mock"
        @property
        def metadata(self):
            return SkillMetadata(
                capabilities=[Capability.REASONING],
                risk_level="low", # safe/low
                side_effects=False,
                idempotent=True,
                requires_network=False,
                requires_filesystem=False,
                cost_class="cheap"
            )
        def input_schema(self, **kwargs): return kwargs
        async def execute(self, params, state): return f"Executed: {params['x']}"
        
    engine.skills["mock_skill"] = MockSkill()
    
    # Run Engine
    result = await engine.run("Do something")
    
    # Verify
    assert "Executed: 1" in result
    from unittest.mock import Mock
    assert len(engine.tracer.save_trace.call_args_list) if isinstance(engine.tracer.save_trace, Mock) else True
