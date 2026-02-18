import asyncio
import os
import shutil
import unittest
import sys
from typing import Any, Dict
from pydantic import BaseModel

# Fix Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import CoreEngine
from core.skill_interface import Skill, SkillContext, SkillResult
from core.llm import LLMService

# --- Mock LLM for Testing Replay Integration ---
# We patch LLMService.generate_structured to simulate network calls
# But since we updated LLMService to use ReplayEngine, we can use the real class with a mocked client?
# Actually, LLMService uses AsyncOpenAI. 
# We need to verify that in REPLAY mode, AsyncOpenAI is NOT called.
# We can mock AsyncOpenAI or just observe the 'cached' flag returned by generate_structured.

class EchoModel(BaseModel):
    message: str

class TestReplayMode(unittest.IsolatedAsyncioTestCase):
    
    async def test_record_and_replay(self):
        print("\n🧪 Testing Replay Mode...")
        request_id = "replay_test_req"
        tape_path = f"tapes/{request_id}.json"
        
        if os.path.exists(tape_path):
            os.remove(tape_path)
            
        # 1. RECORD PHASE
        print("\n📼 Phase 1: Recording...")
        engine = CoreEngine(user_id="test_user", llm_config={"api_key": "dummy_test_key"})
        engine.replay.mode = "record"
        engine.replay.start_session(request_id)
        
        # Manually invoke LLM via ModelPool to test integration
        # We need to Mock the actual network call effectively or allow it to fail?
        # If we don't have API key, it will fail.
        # We must Mock the client.
        
        class MockClient:
            class chat:
                class completions:
                    @staticmethod
                    async def create(*args, **kwargs):
                        # Simulate Network Call
                        print("    🌐 [Network] Call Out...")
                        class Choice:
                            class Message:
                                content = '{"message": "Hello from Network"}'
                            message = Message()
                        class Response:
                            choices = [Choice()]
                            usage = None
                        return Response()
                        
        engine.model_pool.heavy.client = MockClient() # Patch client
        
        # Make a call
        resp, meta = await engine.model_pool.heavy.generate_structured(
            "Sort list", "A, B", EchoModel
        )
        print(f"    Record Resp: {resp.message}")
        self.assertEqual(resp.message, "Hello from Network")
        
        # Save Tape
        engine.replay.save_tape()
        self.assertTrue(os.path.exists(tape_path), "Tape file should exist")
        
        # 2. REPLAY PHASE
        print("\n▶️ Phase 2: Replaying...")
        engine2 = CoreEngine(user_id="test_user", llm_config={"api_key": "dummy_test_key"})
        engine2.replay.mode = "replay"
        engine2.replay.start_session(request_id)
        
        # Patch client again to ensure it DOES NOT get called
        class FailClient:
            class chat:
                class completions:
                    @staticmethod
                    async def create(*args, **kwargs):
                        raise RuntimeError("🚨 Network called in Replay Mode!")
        
        engine2.model_pool.heavy.client = FailClient()
        
        # Make SAME call
        resp2, meta2 = await engine2.model_pool.heavy.generate_structured(
            "Sort list", "A, B", EchoModel
        )
        print(f"    Replay Resp: {resp2.message} (Meta: {meta2})")
        
        self.assertEqual(resp2.message, "Hello from Network")
        self.assertTrue(meta2.get("replay"), "Response should be marked as from replay")
        
        print("✅ Replay Verification Successful")

if __name__ == "__main__":
    unittest.main()
