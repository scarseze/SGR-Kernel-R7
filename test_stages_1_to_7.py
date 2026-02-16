import asyncio
import os
import sys
import shutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import CoreEngine
from skills.base import SkillMetadata
from core.policy import PolicyStatus

async def main():
    print("=== SGR Core: End-to-End Verification (Stages 1-7) ===\n")

    # 1. Initialize Engine (Tests Stage 2: Dynamic Loading & Stage 5: LLM Service)
    print("[Stage 1, 2, 5] Initializing Engine...")
    try:
        engine = CoreEngine()
        print(f"✅ Engine initialized. Skills loaded: {len(engine.skills)}")
        for name, skill in engine.skills.items():
            print(f"   - {name} (Risk: {skill.metadata.risk_level})")
    except Exception as e:
        print(f"❌ Engine init failed: {e}")
        return

    # 2. Verify Metadata (Stage 1 & 3)
    print("\n[Stage 1, 3] Verifying Metadata & Routing...")
    if "web_search" in engine.skills:
        meta = engine.skills["web_search"].metadata
        if meta.risk_level == "low" and "internet" in meta.capabilities:
            print("✅ WebSearch metadata looks correct.")
        else:
            print(f"❌ WebSearch metadata issue: {meta}")
    else:
        print("❌ WebSearch skill not found!")

    # 3. Simulate Policy Check (Stage 7)
    print("\n[Stage 7] Testing Policy Engine...")
    print("   Test 1: Low risk skill (should ALLOW)")
    # Mocking a low risk input
    try:
        from skills.base import BaseSkill
        # Create a dummy high risk skill
        class NukeSkill(BaseSkill):
            name = "nuke_world"
            description = "Nukes the world"
            
            @property
            def metadata(self):
                return SkillMetadata(
                    capabilities=["destruction"],
                    risk_level="high",
                    side_effects=True,
                    idempotent=False,
                    requires_network=False,
                    requires_filesystem=False,
                    cost_class="expensive"
                )
            def input_schema(self): return {}
            async def execute(self): pass

        nuke = NukeSkill()
        decision = engine.policy.check(nuke, {}, engine.state)
        if decision.status == PolicyStatus.REQUIRE_APPROVAL:
             print(f"✅ Policy correctly flagged HIGH RISK skill: {decision.reason}")
        else:
             print(f"❌ Policy failed to flag HIGH RISK skill: {decision}")

    except Exception as e:
         print(f"❌ Policy test failed: {e}")


    # 4. Run Execution Loop (Stage 4 & 6 & DB)
    print("\n[Stage 4, 6] Testing Planner & Trace System...")
    user_query = "What is 2 + 2? Calculate it using python."
    
    # We will mock the LLM response to avoid network/cost during this quick check
    # Or we can run it "live" if user wants code execution.
    # Let's try a real run but catch errors if LLM is not configured
    if not os.getenv("LLM_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
         print("⚠️ No LLM_API_KEY found, skipping live execution test.")
    else:
        print(f"   Query: '{user_query}'")
        try:
            result = await engine.run(user_query)
            print(f"   Result: {result[:100]}...")
            
            # Check Trace
            trace_dir = engine.tracer.trace_dir
            # Find latest trace
            today = datetime.now().strftime("%Y-%m-%d")
            today_dir = os.path.join(trace_dir, today)
            if os.path.exists(today_dir):
                files = os.listdir(today_dir)
                if files:
                    print(f"✅ Trace file created: {files[-1]}")
                else:
                    print("❌ Trace directory empty!")
            else:
                print("❌ Trace directory not found!")
                
        except Exception as e:
             print(f"❌ Execution failed (check connection/api key): {e}")

    # 5. Verify Database (Persistent Memory)
    print("\n[DB Refactor] Verifying SQLAlchemy Storage...")
    try:
        # Check if user was created
        from core.memory import User
        session = engine.memory.Session()
        user = session.query(User).first()
        if user:
             print(f"✅ User found in DB: {user.user_id}")
        else:
             print("⚠️ No user in DB (maybe clean run?)")
        session.close()
    except Exception as e:
         print(f"❌ DB Check failed: {e}")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
