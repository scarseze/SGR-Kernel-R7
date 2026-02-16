import asyncio
from skills.code_interpreter.handler import CodeInterpreterSkill
from skills.code_interpreter.schema import CodeExecutionRequest
from core.state import AgentState

async def test_code_interpreter():
    print("Initializing Code Interpreter Skill...")
    skill = CodeInterpreterSkill()
    
    print("\n[Test 1] Simple Math")
    code_math = "print(10 + 20)"
    result = await skill.execute(CodeExecutionRequest(code=code_math), AgentState())
    print(f"Result: {result.stdout.strip()}")
    assert result.success
    assert result.stdout.strip() == "30"
    
    print("\n[Test 2] Data Analysis (Pandas)")
    code_pandas = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df['a'].sum())
"""
    result = await skill.execute(CodeExecutionRequest(code=code_pandas), AgentState())
    print(f"Result: {result.stdout.strip()}")
    assert result.success
    assert result.stdout.strip() == "6"

    print("\n[Test 3] Security (File Access)")
    code_security = """
try:
    with open('/etc/shadow', 'r') as f:
        print(f.read())
except Exception as e:
    print(f"Access Denied: {e}")
"""
    result = await skill.execute(CodeExecutionRequest(code=code_security), AgentState())
    print(f"Result: {result.stdout.strip()}")
    # We expect success=True (code ran), but output should indicate failure to read
    assert "Access Denied" in result.stdout or "Permission denied" in result.stdout

    print("\nAll tests passed!")

if __name__ == "__main__":
    asyncio.run(test_code_interpreter())
