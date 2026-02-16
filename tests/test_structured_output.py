import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from core.engine import CoreEngine
from core.result import StepResult
from core.types import SkillMetadata
from core.planner import PlanStep
from pydantic import BaseModel

class TestStructured(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = CoreEngine(llm_config={"api_key": "dummy"})
        # Mock init/save to avoid DB/File calls
        self.engine._ensure_initialized = AsyncMock()
        self.engine._save_message = AsyncMock()
        self.engine.memory_manager = MagicMock()
        self.engine.memory_manager.augment_with_semantic_search = AsyncMock()
        
        # Mock Security
        self.engine.security.validate = MagicMock()

    async def test_object_passing(self):
        # 1. Producer Skill (Returns StepResult)
        producer = MagicMock()
        producer.name = "producer"
        producer.metadata = SkillMetadata(
            name="producer", 
            description="Prod", 
            retry_policy="none",
            capabilities=["reasoning"],
            risk_level="low"
        )
        producer.input_schema = MagicMock()
        producer.is_sensitive = MagicMock(return_value=False)
        
        # Return a Dict
        result_data = {"key": "value", "nested": {"deep": 42}}
        result_obj = StepResult(data=result_data, metadata={"confidence": 0.99})
        producer.execute = AsyncMock(return_value=result_obj)
        
        # 2. Consumer Skill (Expects value)
        consumer = MagicMock()
        consumer.name = "consumer"
        consumer.metadata = SkillMetadata(
            name="consumer", 
            description="Cons", 
            retry_policy="none",
            capabilities=["reasoning"],
            risk_level="low"
        )
        consumer.input_schema = MagicMock()
        consumer.execute = AsyncMock(return_value="Done")
        consumer.is_sensitive = MagicMock(return_value=False)
        
        self.engine.skills = {"producer": producer, "consumer": consumer}
        
        # 3. Define Plan
        # Step 1: Produce
        # Step 2: Consume {{step_1.output.key}} and {{step_1.output.nested.deep}}
        step_1 = PlanStep(step_id="step_1", skill_name="producer", description="Prod", params={})
        step_2 = PlanStep(step_id="step_2", skill_name="consumer", description="Cons", params={
            "input_val": "{{step_1.output.key}}",
            "deep_val": "{{step_1.output.nested.deep}}"
        })
        
        # Bypass planner logic and manual execute loop components for unit testing _execute_step is hard because of state dependencies
        # So we test _resolve_params directly after running step 1 manually or simulate outputs.
        
        # Simulate Step 1 execution
        step_outputs = {}
        # Execute Step 1
        trace = MagicMock()
        trace.request_id = "req_1"
        trace.steps = []
        
        res1 = await self.engine._execute_step(step_1, step_outputs, trace)
        self.assertIn("value", str(res1)) # String representation
        
        # Check step_outputs has the dict
        self.assertEqual(step_outputs["step_1"], result_data)
        
        # Simulate Step 2 parameter resolution
        resolved = self.engine._resolve_params(step_2.params, step_outputs)
        
        self.assertEqual(resolved["input_val"], "value")
        self.assertEqual(resolved["deep_val"], 42)
        
if __name__ == '__main__':
    unittest.main()
