"""
Verification Script for SGR Kernel RFC v2 Refactor.
Strict Tests for FSMs, Lifecycle Phases, and Reliability Tables.
"""
import sys
import os
import asyncio
import traceback
from typing import Dict, Any, Set

# Path hack
sys.path.append(os.getcwd())

# Mock Env Vars
os.environ["OPENAI_API_KEY"] = "sk-mock-key" 

from core.engine import CoreEngine
from core.execution import ExecutionState, StepStatus, SemanticFailureType, RetryPolicy
from core.reliability import ReliabilityEngine, RecoveryAction
from core.skill_interface import Skill, SkillContext, SkillResult

# Mock Skill
class MockFailSkill(Skill):
    @property
    def name(self) -> str: return "fail_skill"
    @property
    def capabilities(self) -> Set[str]: return set()
    
    async def execute(self, ctx: SkillContext) -> SkillResult:
        raise ValueError("Simulated Tool Error")

async def verify_rfc_v2():
    print("🚀 Starting RFC v2 Verification...")
    try:
        # 1. Test Reliability Decision Table (Unit Test)
        rel = ReliabilityEngine()
        policy = RetryPolicy(repair_allowed=True)
        
        # Schema Fail + Repair Allowed -> Repair
        action = rel.decide(policy, SemanticFailureType.SCHEMA_FAIL, 1)
        print(f"Test 1 (Schema+Repair): {action}")
        assert action == RecoveryAction.REPAIR
        
        # Schema Fail + No Repair -> Retry
        policy_no_repair = RetryPolicy(repair_allowed=False)
        action = rel.decide(policy_no_repair, SemanticFailureType.SCHEMA_FAIL, 1)
        print(f"Test 2 (Schema+NoRepair): {action}")
        assert action == RecoveryAction.RETRY
        
        # Low Confidence -> Escalate
        action = rel.decide(policy, SemanticFailureType.LOW_CONFIDENCE, 1)
        print(f"Test 3 (LowConf): {action}")
        assert action == RecoveryAction.ESCALATE
        
        # Escalation Table
        tier1 = rel.get_escalation_tier(1)
        tier2 = rel.get_escalation_tier(2)
        tier3 = rel.get_escalation_tier(3)
        print(f"Test 4 (Escalation): {tier1}->{tier2}->{tier3}")
        assert tier1 == "fast" and tier2 == "mid" and tier3 == "heavy"

        print("✅ Reliability Tables Verified.")
        
        # 2. Test Partial Lifecycle (Integration)
        kernel = CoreEngine()
        kernel.register_skill(MockFailSkill())
        kernel.skill_adapter.registry["fail_skill"] = MockFailSkill()

        # Inject Dummy Plan
        from core.execution import PlanIR, StepNode
        async def mock_generate_plan(user_input):
            return PlanIR(
                steps=[StepNode(id="step_1", skill_name="fail_skill")],
                edges=[]
            )
        kernel._generate_plan = mock_generate_plan
        
        # Execute (Should Fail and Retry/Escalate)
        # Mock hooks to track events
        events = []
        async def on_fail(*args): events.append("on_failure")
        async def on_retry(*args): events.append("on_retry")
        
        kernel.hooks.register("on_failure", on_fail)
        kernel.hooks.register("on_retry", on_retry)
        
        # This will loop until max retries (3) then fail
        await kernel.run("fail me")
        
        print(f"Events captured: {len(events)}")
        assert "on_failure" in events
        
        print("✅ Lifecycle Integration Verified.")
        print("🎉 RFC v2 Verification Successful!")
        
    except AssertionError as e:
        print(f"❌ Assertion Failed: {e}")
        traceback.print_exc()
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_rfc_v2())
