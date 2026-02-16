import asyncio
import os
from dotenv import load_dotenv
from skills.rlm.handler import RLMSkill
from skills.rlm.schema import RLMInput
from core.state import AgentState

async def test_rlm_docker():
    load_dotenv()
    print("Initializing RLM (Dockerized)...")
    skill = RLMSkill()
    print("Skill initialized. Starting execution...")
    
    # Simple context
    context = (
        "Project Report: The project 'SGR Core' is an AI agent framework. "
        "It uses Docker for sandboxing and DeepSeek for reasoning. "
        "The framework is written in Python."
    )
    
    query = "Summarize the project and count the words in the description using Python."
    
    print(f"\n[Problem]\nContext: {context}\nQuery: {query}\n")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY not set.")
    
    try:
        # Execute RLM
        result = await skill.execute(
            RLMInput(query=query, context_text=context, max_iterations=5), 
            AgentState(user_request="test_rlm")
        )
        print("\n[Result]")
        print(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] Test Failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_rlm_docker())
