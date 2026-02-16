import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from core.engine import CoreEngine
from core.types import SkillMetadata, RetryPolicy
from core.state import AgentState

class MockSkill:
    def __init__(self, name, policy=RetryPolicy.NONE):
        self.name = name
        self.metadata = SkillMetadata(
            name=name,
            description="Mock Skill",
            capabilities=[],
            retry_policy=policy
        )
        self.input_schema = MagicMock()
        self.input_schema.return_value = {}
        self.execute = AsyncMock()

class TestRetryLogic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Patch dependencies to lightweight mocks
        self.patchers = [
            patch('core.engine.Database'),
            patch('core.rag.vector_store.QdrantAdapter'),
            patch('core.rag.embeddings.OllamaEmbeddingProvider'),
            patch('core.engine.LLMService'),
            patch('core.engine.PersistentMemory'),
            patch('core.summarizer.ConversationSummarizer'),
            patch('core.memory_manager.MemoryManager'),
            patch('core.rag.pipeline.RAGPipeline'), # Mock RAG
            patch('core.rag.repair.AnswerCritic'),
            patch('core.rag.repair.RepairStrategy'),
            patch('core.rag.retriever.RAGRetriever'),
            patch('core.rag.context.RAGContextBuilder'),
        ]
        for p in self.patchers:
            p.start()
            
        self.engine = CoreEngine(llm_config={"api_key": "dummy"})
        # Disable heavy ensuring
        self.engine._ensure_initialized = AsyncMock()
        self.engine.state = AgentState(user_request="test")

    async def asyncTearDown(self):
        for p in self.patchers:
            p.stop()

    async def test_retry_standard_success_after_fail(self):
        """Test that STANDARD policy retries and succeeds."""
        skill = MockSkill("flaky_skill", RetryPolicy.STANDARD)
        # Fail twice, then succeed
        skill.execute.side_effect = [Exception("Fail 1"), Exception("Fail 2"), "Success"]
        
        self.engine.skills["flaky_skill"] = skill
        
        step_def = MagicMock()
        step_def.step_id = "step_1"
        step_def.skill_name = "flaky_skill"
        step_def.params = {}
        
        
        trace = MagicMock()
        trace.request_id = "req_1"
        trace.steps = []
        
        result = await self.engine._execute_step(step_def, {}, trace)
        
        self.assertEqual(result, "Success")
        self.assertEqual(skill.execute.call_count, 3)
        # Verify trace status logic (mock trace object structure simplified here)
        # Ideally check trace.steps[-1].status == "completed"

    async def test_retry_exhausted(self):
        """Test that retries eventually fail if errors persist."""
        skill = MockSkill("broken_skill", RetryPolicy.STANDARD) # 3 attempts
        skill.execute.side_effect = Exception("Permanent Fail")
        
        self.engine.skills["broken_skill"] = skill
        
        step_def = MagicMock()
        step_def.step_id = "step_2"
        step_def.skill_name = "broken_skill"
        step_def.params = {}
        
        trace = MagicMock()
        trace.request_id = "req_2"
        trace.steps = []
        
        with self.assertRaises(Exception):
            await self.engine._execute_step(step_def, {}, trace)
            
        self.assertEqual(skill.execute.call_count, 3)

if __name__ == '__main__':
    unittest.main()
