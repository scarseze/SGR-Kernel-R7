import unittest
import os
import shutil
import json
from datetime import datetime
from core.trace import TraceManager, RequestTrace, RAGQueryTrace, StepTrace
from unittest.mock import MagicMock

class TestTraceMetrics(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_traces"
        self.tracer = TraceManager(trace_dir=self.test_dir)
        
    def tearDown(self):
        if os.path.exists(self.tracer.trace_dir):
            shutil.rmtree(self.tracer.trace_dir)

    def test_get_last_trace(self):
        """Test retrieving the last trace."""
        # Create a dummy trace
        trace = RequestTrace(user_request="test", status="completed")
        self.tracer.save_trace(trace)
        
        # Retrieve
        loaded_trace = self.tracer.get_last_trace()
        self.assertIsNotNone(loaded_trace)
        self.assertEqual(loaded_trace.request_id, trace.request_id)

    def test_metrics_formatting_logic(self):
        """Simulate the logic used in telegram_bot.py to ensure it doesn't crash."""
        # logical test of the snippet from telegram_bot
        
        trace = RequestTrace(user_request="rag test", total_duration=1.5, status="completed")
        step = StepTrace(step_id="step1", skill_name="rag_skill", input_params={})
        qa = RAGQueryTrace(
            query="test query",
            latency_ms=100.0,
            found_docs=5,
            used_docs=2,
            sources=["doc1", "doc2"]
        )
        step.rag_queries.append(qa)
        trace.steps.append(step)
        
        # Simulation of bot logic
        txt = f"🆔 `{trace.request_id[:8]}`\n"
        rag_queries = []
        for s in trace.steps:
            rag_queries.extend(s.rag_queries)
            
        self.assertEqual(len(rag_queries), 1)
        self.assertEqual(rag_queries[0].query, "test query")
        
        # Ensure formatting string works
        msg = f"   ⏱ {qa.latency_ms:.0f}ms | 📄 Found: {qa.found_docs}"
        self.assertIn("100ms", msg)

if __name__ == '__main__':
    unittest.main()
