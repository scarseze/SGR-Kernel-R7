import asyncio
import re
import os
import json
import httpx
import threading
from typing import Optional, List, Dict, Any, Type
from dataclasses import dataclass

from skills.base import BaseSkill, SkillMetadata
from skills.rlm.schema import RLMInput
from skills.code_interpreter.handler import CodeInterpreterSkill
from skills.code_interpreter.schema import CodeExecutionRequest
from core.state import AgentState
from skills.rlm.proxy_server import start_proxy_server, app
import uvicorn

# --- RLM Core Classes (Dockerized) ---

class LLMProvider:
    # Now just a holder, actual calls happen via Proxy Server
    pass

@dataclass
class RLMConfig:
    max_iterations: int = 20
    max_output_chars: int = 50000
    sub_call_limit: int = 50

@dataclass
class IterationRecord:
    step: int
    code: Optional[str]
    output: str
    error: Optional[str] = None

class RLMOrchestrator:
    CODE_PATTERN = re.compile(r"```(?:repl|python)\n([\s\S]*?)```")
    FINAL_PATTERN = re.compile(r"FINAL\(([\s\S]*?)\)")
    FINAL_VAR_PATTERN = re.compile(r"FINAL_VAR\((\w+)\)")

    def __init__(self, config: RLMConfig, sandbox: CodeInterpreterSkill):
        self.config = config
        self.sandbox = sandbox

    def _build_system_prompt(self, context_len: int) -> str:
        return f"""You are a Python RLM (Recursive Language Model) engine.
Your GOAL is to answer the user query by writing and executing Python code.
You have a variable `context` loaded in memory (len={context_len}).

CRITICAL INSTRUCTIONS:
1. The text is ALREADY in `context`. DO NOT try to read files.
2. DO NOT reply with natural language. You MUST write code in ```repl``` blocks.
3. If you need to analyze the text, split `context` and use `llm_query(chunk)`.
4. Output final answer using `FINAL(text)`.
5. `llm_query(prompt)` is available in your environment.

Example Step 1:
```repl
print(context[:100])
print(len(context))
```
"""

    def _build_iteration_prompt(self, query: str, history: list) -> str:
        prompt = [f"TASK: {query}\n"]
        if history:
            prompt.append("PREVIOUS STEPS:")
            for h in history[-3:]: 
                prompt.append(f"\n[Step {h.step}]")
                if h.code: prompt.append(f"Code:\n```repl\n{h.code}\n```")
                prompt.append(f"Output:\n{h.output}" if not h.error else f"Error: {h.error}")
        prompt.append("\nWrite next step code in ```repl block or FINAL().")
        return "\n".join(prompt)

    async def _query_llm_host(self, prompt: str, system: Optional[str] = None) -> str:
        # Host-side LLM query for the Orchestrator itself
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system or "You are a helpful AI."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 4096
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def process(self, query: str, context: str, state: AgentState) -> str:
        history = []
        
        # 0. Upload Context to Sandbox
        # We save context to a file in Docker so script can read it
        # Actually simplest is to just inject it as a string in the first script
        # But for huge contexts, file is better. Let's use file.
        # However, CodeInterpreter doesn't support file upload explicitly yet in this version.
        # We will wrap it in the python script if it's not too huge, 
        # or we rely on the fact that CodeInterpreter might persist state? 
        # No, CodeInterpreter resets per 'execute' call unless we implemented persistence.
        # Wait, existing CodeInterpreter starts a container once. State persists if we use same container?
        # The current implementation of CodeInterpreterSkill re-runs `exec_run`. Variables are NOT persisted between calls 
        # unless we use a persistent kernel (like Jupyter). 
        # THE CURRENT CodeInterpreterSkill executes separate `python script.py` calls. 
        # So variables are LOST.
        
        # FIX: We must send the full context (or load from file) every time? 
        # That's inefficient.
        # BUT: implementing fully persistent Jupyter kernel is a big Task.
        # WORKAROUND: We will use a file `/home/sanduser/context.txt` inside Docker.
        # We write it ONCE.
        
        setup_code = f"""
with open('context.txt', 'w', encoding='utf-8') as f:
    f.write({json.dumps(context)})
"""
        await self.sandbox.execute(CodeExecutionRequest(code=setup_code), state)
        
        system_prompt = self._build_system_prompt(len(context))
        
        for i in range(self.config.max_iterations):
            prompt = self._build_iteration_prompt(query, history)
            response = await self._query_llm_host(prompt, system_prompt)
            
            code_match = self.CODE_PATTERN.search(response)
            output = ""
            error = None
            code = None

            if code_match:
                code = code_match.group(1)
                
                # WRAP CODE to use transparent proxy for llm_query
                # And reload context from file
                wrapped_code = f"""
import requests
import json

# Proxy function to call host
def llm_query(prompt):
    try:
        # host.docker.internal works on Docker Desktop for Windows/Mac
        res = requests.post("http://host.docker.internal:8090/llm", json={{"prompt": prompt}}, timeout=120)
        res.raise_for_status()
        return res.json()["text"]
    except Exception as e:
        return f"LLM Error: {{e}}"

# Load Context
try:
    with open('context.txt', 'r', encoding='utf-8') as f:
        context = f.read()
except FileNotFoundError:
    context = ""

# --- USER CODE ---
{code}
"""
                exec_result = await self.sandbox.execute(CodeExecutionRequest(code=wrapped_code), state)
                output = exec_result.stdout + exec_result.stderr
                if not exec_result.success:
                    error = output

            # Check Final
            final_match = self.FINAL_PATTERN.search(response)
            if final_match: return final_match.group(1)
            
            # Log
            history.append(IterationRecord(i+1, code, output, error))
            print(f"[RLM Step {i+1}] Code: {bool(code)} | Error: {bool(error)}")
            
        return "Max iterations reached."

# --- Skill Wrapper ---

class RLMSkill(BaseSkill):
    name: str = "rlm_skill"
    description: str = "Recursive Language Model (Dockerized) for deep analysis."

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            capabilities=["reasoning", "coding", "analysis"],
            risk_level="high", # Executes code
            side_effects=True, # Can modify state/vars
            idempotent=False,
            requires_network=True,
            requires_filesystem=True, 
            cost_class="expensive"
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sandbox = CodeInterpreterSkill()
        self.proxy_process = None

    @property
    def input_schema(self) -> Type[RLMInput]:
        return RLMInput

    async def execute(self, params: RLMInput, state: AgentState) -> str:
        # 1. Start Proxy Server (if not already running)
        # In a real app, this should be a separate service. 
        # Here we spawn it in a thread or assume it's running. 
        # For simplicity, let's start it in a thread if check fails?
        # Actually, `uvicorn.run` blocks. We need to run it in a separate process or thread.
        # Let's rely on the user running `python server.py` which triggers this skill? 
        # No, let's lazy-start it in a thread.
        
        if not self._is_proxy_running():
            print("[RLM] Starting internal Proxy Server on port 8090...")
            # Use uvicorn.run directly which creates its own loop
            t = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 8090, "log_level": "error"}, daemon=True)
            t.start()
            await asyncio.sleep(2) # Wait for startup

        # 2. Context Logic (Same as before)
        # 3. Run Async Process
        context = ""
        if params.context_text:
            context = params.context_text
        elif params.context_file_path:
             # ... (File reading logic simplified for brevity - assumes text or handled elsewhere) ...
             if os.path.exists(params.context_file_path):
                 with open(params.context_file_path, "r", encoding="utf-8", errors='ignore') as f:
                     context = f.read()
        
        if not context: return "Error: No context."

        # 3. Config & Run
        config = RLMConfig(max_iterations=params.max_iterations)
        orchestrator = RLMOrchestrator(config, self.sandbox)
        
        return await orchestrator.process(params.query, context, state)

    def _is_proxy_running(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', 8090)) == 0
