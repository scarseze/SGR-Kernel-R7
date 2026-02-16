import asyncio
import os
import re
import json
import httpx
from typing import Optional, List, Dict, Any, Type
from pydantic import BaseModel

from skills.base import BaseSkill
from skills.logic_rl.schema import LogicRLInput
from skills.code_interpreter.handler import CodeInterpreterSkill
from skills.code_interpreter.schema import CodeExecutionRequest
from core.state import AgentState

class LogicRLSkill(BaseSkill):
    name: str = "logic_rl"
    description: str = (
        "Solves complex logic puzzles and reasoning tasks using a 'Reason-Act-Refine' loop. "
        "It proposes solutions and verifies them by running Python code in a sandbox."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize internal skills
        self.sandbox = CodeInterpreterSkill()
        self.http_client = httpx.AsyncClient(timeout=120.0)

    @property
    def input_schema(self) -> Type[LogicRLInput]:
        return LogicRLInput

    async def execute(self, params: LogicRLInput, state: AgentState) -> str:
        problem = params.problem
        max_retries = params.max_retries
        history = []
        
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return "Error: DEEPSEEK_API_KEY is not set. Cannot perform Logic-RL."

        print(f"[{self.name}] Solving: {problem} (Max Retries: {max_retries})")

        for attempt in range(max_retries):
            # 1. Generate Solution & Verification Code
            response = await self._query_llm(problem, history, api_key)
            
            # Extract explanation and code
            code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
            code = code_match.group(1) if code_match else None
            
            if not code:
                history.append(f"Attempt {attempt+1}: Failed to generate code. LLM Response: {response[:200]}...")
                continue

            print(f"[{self.name}] Attempt {attempt+1}: Verifying candidate solution...")

            # 2. Verify in Sandbox
            # We wrap the code to print "SUCCESS" if verification passes
            wrapped_code = f"""
try:
{self._indent_code(code)}
except Exception as e:
    print(f"RUNTIME_ERROR: {{e}}")
"""
            exec_result = await self.sandbox.execute(
                CodeExecutionRequest(code=wrapped_code), 
                state
            )

            output = exec_result.stdout + exec_result.stderr
            
            # 3. Analyze Result
            if "SUCCESS" in output:
                return f"✅ **Solved in {attempt+1} iterations!**\n\n**Solution:**\n{self._extract_non_code(response)}\n\n**Verification:**\n```python\n{code}\n```\n\n**Output:**\n{output}"
            else:
                # RL Step: Feedback
                error_msg = f"Verification Failed. Output:\n{output}"
                history.append(f"Attempt {attempt+1} Code:\n{code}\n\nResult:\n{error_msg}")
                print(f"[{self.name}] Retrying... ({error_msg.strip()[:100]}...)")

        return f"❌ Failed to solve after {max_retries} attempts.\n\nHistory:\n" + "\n".join(history[-3:])

    async def _query_llm(self, problem: str, history: List[str], api_key: str) -> str:
        system_prompt = (
            "You are a Logic-RL agent. Your goal is to solve the user's puzzle and VERIFY it with Python code.\n"
            "Process:\n"
            "1. Think step-by-step.\n"
            "2. Propose a solution.\n"
            "3. Write a Python script to VERIFY the solution.\n"
            "4. The script MUST print 'SUCCESS' if the solution is correct, otherwise print failure details.\n"
            "5. NO USER INPUT allowed.\n"
            "6. NO EXTERNAL LIBRARIES allowed (e.g. numpy, scipy). Use only standard Python libraries (math, random, itertools).\n"
            "7. Wrap code in ```python blocks.\n"
        )
        
        user_prompt = f"Problem: {problem}\n"
        if history:
            user_prompt += "\nPrevious Attempts/Errors:\n" + "\n".join(history)
            user_prompt += "\n\nAnalyze the errors and try a different approach."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = await self.http_client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat", # or "deepseek-reasoner" if available
                    "messages": messages,
                    "max_tokens": 4096
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LLM Error: {e}"

    def _indent_code(self, code: str) -> str:
        return "\n".join("    " + line for line in code.splitlines())

    def _extract_non_code(self, text: str) -> str:
        # Remove code blocks to get the explanation
        return re.sub(r"```python.*?```", "", text, flags=re.DOTALL).strip()
