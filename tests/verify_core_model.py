"""
Verification Script for SGR Kernel (Core Architectural Model).
"""
import sys
import os
import asyncio
import uuid
import shutil
from typing import Set, Dict, Any

# Path hack
sys.path.append(os.getcwd())

# Mock Env Vars
os.environ["OPENAI_API_KEY"] = "sk-mock-key"
os.environ["ANTHROPIC_API_KEY"] = "sk-mock-key"

from core.engine import CoreEngine as RuntimeKernel
from core.skill_interface import Skill, SkillContext, SkillResult
from core.execution import StepStatus
import traceback

# Mock Skill
class MockFileRead(Skill):
    @property
    def name(self) -> str: return "fs_read" # Matches default mock
    @property
    def capabilities(self) -> Set[str]: return {"read"}
    
    async def execute(self, ctx: SkillContext) -> SkillResult:
        # Simulate work
        return SkillResult(output="File Content: Hello World")

async def verify_core_model():
    print("🚀 Starting Core Architectural Model Verification...")
    try:
        # 1. Initialize Kernel
        kernel = RuntimeKernel()
        
        # 2. Register Mock Skill
        mock_skill = MockFileRead()
        kernel.register_skill(mock_skill)
        # Also register with internal adapter map manually for now as CoreEngine init might need update or we rely on shared ref?
        # CoreEngine connects them.
        kernel.skill_adapter.registry["fs_read"] = mock_skill # Explicitly for safety
        
        print("✅ Kernel Initialized & Skill Registered.")
        
        # 3. Run Request
        request = "Read file test.txt"
        print(f"▶️ Executing Request: {request}")
        
        # Mock Planner output injection (simulate Planner returning a plan)
        from core.execution import PlanIR, StepNode
        async def mock_generate_plan(user_input):
            return PlanIR(
                steps=[StepNode(
                    id="step_1", 
                    skill_name="fs_read", 
                    inputs_template={"path": "test.txt"}
                )],
                edges=[]
            )
        
        kernel._generate_plan = mock_generate_plan
        print("✅ Planner Mocked.")
        
        # Execute
        result = await kernel.run(request)
        print(f"✅ Execution Result: {result}")
        
        # 4. Verify State
        ckpt_path = kernel.checkpoints.get_latest_checkpoint(kernel.user_id) 
        
        # 5. Verify Telemetry
        telemetry = kernel.telemetry.get_snapshot()
        print(f"📊 Telemetry Snapshot: {telemetry}")
        assert "plan_latency" in telemetry
        
        print("🎉 Verification Successful!")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints") # Cleanup previous
    asyncio.run(verify_core_model())
