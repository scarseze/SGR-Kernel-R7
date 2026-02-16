import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from skills.research_agent.handler import ResearchSubAgent, ResearchInput
from skills.xbrl_analyst.handler import XBRLAnalystSkill
from skills.xbrl_analyst.schema import XBRLInput
from core.result import StepResult

class TestSkillsV2(unittest.IsolatedAsyncioTestCase):
    async def test_research_agent_structured_output(self):
        # Setup
        llm_config = {"api_key": "dummy", "model": "test"}
        agent = ResearchSubAgent(llm_config)
        agent.llm = MagicMock()
        agent.llm.complete = AsyncMock(side_effect=["Step 1\nStep 2", "Summary Report"])
        
        params = ResearchInput(topic="Test Topic")
        
        # Execute
        result = await agent.execute(params, MagicMock())
        
        # Verify
        self.assertIsInstance(result, StepResult)
        self.assertEqual(result.data["summary"], "Summary Report")
        self.assertEqual(len(result.data["plan"]), 2)
        self.assertEqual(result.data["simulated_sources"][1], "web_simulation")
        self.assertIn("Deep Research Report", str(result))

    @patch("skills.xbrl_analyst.handler.etree")
    @patch("skills.xbrl_analyst.handler.os.path.exists")
    async def test_xbrl_analyst_structured_output(self, mock_exists, mock_etree):
        # Setup
        skill = XBRLAnalystSkill()
        mock_exists.return_value = True
        
        # Mock parsing logic to avoid needing real XML file
        # We'll mock the internal methods _parse_xbrl_logic and _calculate_metrics_logic
        # because testing lxml intricacies is not the goal here, just the StepResult wrapper.
        
        skill._parse_xbrl_logic = MagicMock(return_value=[{"concept": "Assets", "value": "100"}])
        skill._calculate_metrics_logic = MagicMock(return_value={
            "Assets": 100, 
            "Health Check": "OK"
        })
        
        params = XBRLInput(file_path="test.xml")
        
        # Execute
        result = await skill.execute(params, MagicMock())
        
        # Verify
        self.assertIsInstance(result, StepResult)
        self.assertEqual(result.data["metrics"]["Assets"], 100)
        self.assertEqual(result.data["health_check"], "OK")
        self.assertIn("**Health Status**: ✅ OK", str(result))

if __name__ == '__main__':
    unittest.main()
