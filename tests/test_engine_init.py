import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import CoreEngine

class TestEngineInit(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock external dependencies to avoid real connections
        self.mock_db_patch = patch('core.engine.Database')
        self.mock_qdrant_patch = patch('core.rag.vector_store.QdrantAdapter')
        self.mock_ollama_patch = patch('core.rag.embeddings.OllamaEmbeddingProvider')
        self.mock_llm_patch = patch('core.engine.LLMService')
        self.mock_redis_patch = patch('core.task_queue.redis_queue.redis.from_url') # Mock Redis if loaded
        
        self.mock_db = self.mock_db_patch.start()
        self.mock_qdrant = self.mock_qdrant_patch.start()
        self.mock_ollama = self.mock_ollama_patch.start()
        self.mock_llm_cls = self.mock_llm_patch.start()
        
        # Setup LLM mock instance
        self.mock_llm_instance = MagicMock()
        self.mock_llm_cls.return_value = self.mock_llm_instance
        
        # Since CoreEngine instantiates LLMService internally, we need to ensure it uses the mock
        # which patch does automatically for the class.
        pass

    async def asyncTearDown(self):
        self.mock_db_patch.stop()
        self.mock_qdrant_patch.stop()
        self.mock_ollama_patch.stop()
        self.mock_llm_patch.stop()

    def test_rag_components_initialized(self):
        """Verify RAG components are initialized in correct order."""
        engine = CoreEngine()
        
        # Check attributes existence
        self.assertTrue(hasattr(engine, 'rag'))
        self.assertTrue(hasattr(engine, 'critic'))
        self.assertTrue(hasattr(engine, 'repair_strategy'))
        
        # Check Injection
        self.assertEqual(engine.rag.critic, engine.critic)
        self.assertEqual(engine.rag.repair_strategy, engine.repair_strategy)
        
        print("Engine RAG initialization verify: SUCCESS")

if __name__ == '__main__':
    unittest.main()
