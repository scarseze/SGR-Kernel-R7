import asyncio
import unittest
import asyncio
import unittest
import sys
import os

# Fix Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import CoreEngine
from core.skill_interface import Skill, SkillContext, SkillResult
from core.execution import StepNode, StepStatus, ExecutionState, PlanIR

class HighRiskSkill(Skill):
    @property
    def name(self) -> str: return "dangerous_tool"
    
    @property
    def capabilities(self): return {"READ", "WRITE_EXTERNAL"}
    
    async def execute(self, ctx: SkillContext) -> SkillResult:
        return SkillResult(output="Danger executed!")

class TestCapabilities(unittest.IsolatedAsyncioTestCase):
    
    async def test_capability_enforcement(self):
        print("\n🧪 Testing Capability Enforcement...")
        engine = CoreEngine(user_id="test_user", llm_config={"api_key": "dummy_test_key"})
        engine.register_skill(HighRiskSkill())
        
        # Case 1: Violation
        # Step only grants READ. Skill needs WRITE_EXTERNAL.
        print("▶️ Case 1: Expecting Violation (Grants: READ vs Needs: WRITE_EXTERNAL)...")
        step_bad = StepNode(
            id="step_bad",
            skill_name="dangerous_tool",
            required_capabilities=["READ"] 
        )
        
        state = ExecutionState(request_id="cap_test", input_payload="test")
        state.initialize_step(step_bad.id)
        
        # We run directly via lifecycle to isolate
        await engine.lifecycle.run_step(step_bad, state)
        
        s_state = state.step_states[step_bad.id]
        print(f"    Status: {s_state.status}")
        print(f"    Failure: {s_state.failure}")
        
        self.assertEqual(s_state.status, StepStatus.FAILED, "Step should have FAILED due to violation")
        self.assertIsNotNone(s_state.failure)
        self.assertEqual(s_state.failure.failure_type, "CAPABILITY_VIOLATION")
        self.assertIn("Security Violation", s_state.failure.error_message)
        
        # Case 2: Success
        # Step grants READ and WRITE_EXTERNAL.
        print("\n▶️ Case 2: Expecting Success (Grants: READ, WRITE_EXTERNAL)...")
        step_good = StepNode(
            id="step_good",
            skill_name="dangerous_tool",
            required_capabilities=["READ", "WRITE_EXTERNAL"]
        )
        
        state.initialize_step(step_good.id)
        await engine.lifecycle.run_step(step_good, state)
        
        s_state_good = state.step_states[step_good.id]
        print(f"    Status: {s_state_good.status}")
        
        self.assertEqual(s_state_good.status, StepStatus.COMMITTED, "Step should have COMMITTED")
        
        print("✅ Capability Verification Successful")

if __name__ == "__main__":
    unittest.main()
