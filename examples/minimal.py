import asyncio
import os
from typing import Dict, Any

# Ensure we can import from core
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import CoreEngine
from examples.skill_minimal import MinimalGreetingSkill

async def main():
    print(f"--- SGR Kernel Minimal Example (v{CoreEngine.VERSION}) ---")
    
    # 1. Initialize Engine (No LLM config needed for pure skills)
    engine = CoreEngine()
    
    # 2. Register Skill
    skill = MinimalGreetingSkill()
    engine.register_skill(skill)
    print(f"Registered skill: {skill.name}")
    
    # 3. Create a Request (In v1, we bypass the Planner for direct skill testing if needed, 
    # but here we show the full flow if we had an LLM. Since we don't, we'll mimic a planner 
    # decision or just execute the skill directly via the DAG if we could.
    # For this minimal example, let's just show that the engine starts and has the skill.)
    
    # To run a full plan without an LLM is tricky, but we can inspect the registry.
    assert skill.name in engine.skills
    print("Engine is ready.")
    
    # In a real scenario with LLM:
    # result = await engine.run("Say hello to Alice loudly")
    # print(result)

if __name__ == "__main__":
    asyncio.run(main())
