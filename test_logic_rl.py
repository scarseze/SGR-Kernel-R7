import asyncio
import os
from dotenv import load_dotenv
from skills.logic_rl.handler import LogicRLSkill
from skills.logic_rl.schema import LogicRLInput
from core.state import AgentState

async def test_logic_rl():
    load_dotenv()
    print("Initializing Logic-RL Skill...")
    skill = LogicRLSkill()
    
    problem = "Find the roots of the equation x^2 - 5x + 6 = 0 using Python code."
    print(f"\n[Test] Problem: {problem}")

    # Ensure API Key is set (for testing purposes, check env)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY not set. Test might fail.")
    
    result = await skill.execute(
        LogicRLInput(problem=problem, max_retries=3), 
        AgentState(user_request="test")
    )
    
    print("\n[Result]")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_logic_rl())
