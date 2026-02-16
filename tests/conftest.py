import pytest
import asyncio
import os
import sys
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import CoreEngine
from core.llm import LLMService

# Mock LLM Response for "Match" (2+2)
MOCK_PLAN_RESPONSE_MATH = """
{
  "reasoning": "User wants to calculate 2+2. I can use the code_interpreter or python skill.",
  "steps": [
    {
      "step_id": "step_1",
      "skill_name": "code_interpreter",
      "params": {
        "code": "print(2 + 2)",
        "language": "python"
      }
    }
  ]
}
"""

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMService)
    
    # Setup Async Mock for generate_structured
    async def side_effect(system_prompt, user_prompt, response_model, **kwargs):
        # Return a real instance of the model, not just dict
        # We need to manually construct the expected object based on the prompt content
        # This is a bit tricky with mocks. 
        # Strategy: We assume the test sets up the return value on the mock based on expected call.
        return response_model(), {"total_tokens": 0} 

    llm.generate_structured = AsyncMock()
    return llm

@pytest.fixture
def engine(mock_llm):
    # Use in-memory DB for tests
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    
    # Disable actual skill loading to speed up tests (or mock them)
    # For integration tests, we might want real skills but mock LLM.
    # Let's load real skills but mock the LLM part of them if any.
    
    engine = CoreEngine(llm_config={"api_key": "mock"})
    engine.llm = mock_llm # Replace real LLM with mock
    
    # Inject a dummy skill for testing if needed
    # engine.skills["test_skill"] = ...
    
    return engine
