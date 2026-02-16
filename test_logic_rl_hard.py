import asyncio
import os
from dotenv import load_dotenv
from skills.logic_rl.handler import LogicRLSkill
from skills.logic_rl.schema import LogicRLInput
from core.state import AgentState

async def test_hard_logic():
    load_dotenv()
    print("Initializing Logic-RL Skill for Hard Test...")
    skill = LogicRLSkill()
    
    # Cryptarithmetic Puzzle (Simplified for speed)
    problem = (
        "Find a positive number x such that x^3 + x = 30. "
        "Write a Python script to find the solution and verify it."
    )
    
    print(f"\n[Problem]\n{problem}\n")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY not set.")
    
    # Increase retries for hard problems
    result = await skill.execute(
        LogicRLInput(problem=problem, max_retries=5), 
        AgentState(user_request="test_hard")
    )
    
    print("\n[Result]")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_hard_logic())
