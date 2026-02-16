import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from core.engine import CoreEngine
from core.planner import ExecutionPlan, PlanStep
from core.types import StepStatus, SkillMetadata, RetryPolicy
from core.state import AgentState

class TestReplanning(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patchers = [
            patch('core.engine.Database'),
            patch('core.rag.vector_store.QdrantAdapter'),
            patch('core.rag.embeddings.OllamaEmbeddingProvider'),
            patch('core.engine.LLMService'),
            patch('core.engine.PersistentMemory'),
            patch('core.summarizer.ConversationSummarizer'),
            patch('core.memory_manager.MemoryManager'),
            patch('core.rag.pipeline.RAGPipeline'),
            patch('core.rag.repair.AnswerCritic'),
            patch('core.rag.repair.RepairStrategy'),
            patch('core.rag.retriever.RAGRetriever'),
            patch('core.rag.context.RAGContextBuilder'),
        ]
        for p in self.patchers:
            p.start()
            
        self.engine = CoreEngine(llm_config={"api_key": "dummy"})
        self.engine._ensure_initialized = AsyncMock()
        self.engine._save_message = AsyncMock()
        self.engine.memory_manager.augment_with_semantic_search = AsyncMock()
        self.engine.state = AgentState(user_request="test")
        
        # Mock Planner
        self.engine.planner = MagicMock()
        self.engine.planner.create_plan = AsyncMock()
        self.engine.planner.repair_plan = AsyncMock()
        
        # Mock Security
        self.engine.security.validate = MagicMock()

    async def asyncTearDown(self):
        for p in self.patchers:
            p.stop()

    async def test_replan_on_failure(self):
        # 1. Setup Skills
        skill_fail = MagicMock()
        skill_fail.name = "fail_skill"
        skill_fail.metadata = SkillMetadata(name="fail_skill", description="Fail", retry_policy=RetryPolicy.NONE, capabilities=[])
        skill_fail.input_schema.return_value = {}
        skill_fail.execute = AsyncMock(side_effect=Exception("Boom"))
        skill_fail.is_sensitive = MagicMock(return_value=False)
        
        skill_fix = MagicMock()
        skill_fix.name = "fix_skill"
        skill_fix.metadata = SkillMetadata(name="fail_skill", description="Fix", retry_policy=RetryPolicy.NONE, capabilities=[])
        skill_fix.input_schema.return_value = {}
        skill_fix.execute = AsyncMock(return_value="Fixed!")
        skill_fix.is_sensitive = MagicMock(return_value=False)
        
        self.engine.skills = {"fail_skill": skill_fail, "fix_skill": skill_fix}
        
        # 2. Initial Plan
        initial_plan = ExecutionPlan(
            steps=[
                PlanStep(step_id="step_1", skill_name="fail_skill", description="Fail", params={})
            ],
            reasoning="Try fail"
        )
        self.engine.planner.create_plan.return_value = (initial_plan, {})
        
        # 3. Repaired Plan
        repaired_plan = ExecutionPlan(
            steps=[
                # step_1 is removed or replaced
                PlanStep(step_id="step_1_fix", skill_name="fix_skill", description="Fix", params={})
            ],
            reasoning="Fixed plan"
        )
        self.engine.planner.repair_plan.return_value = repaired_plan
        
        # 4. Run
        result = await self.engine.run("do something")
        
        # 5. Assertions
        # Should have called repair_plan
        self.engine.planner.repair_plan.assert_called_once()
        
        # Should have executed fix_skill
        skill_fix.execute.assert_called_once()
        
        # Result should contain success from fix
        self.assertIn("Fixed!", result)

if __name__ == '__main__':
    unittest.main()
