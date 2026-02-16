import unittest
from unittest.mock import MagicMock, AsyncMock
from core.rag.components import ScoreReranker, DocFilter, QueryRewriter
from core.rag.context import RAGDocument

class TestRAGOptimization(unittest.IsolatedAsyncioTestCase):
    async def test_score_reranker_threshold(self):
        """Test that ScoreReranker respects the threshold."""
        reranker = ScoreReranker()
        docs = [
            RAGDocument(content="A", score=0.9, metadata={}, source="doc1"),
            RAGDocument(content="B", score=0.4, metadata={}, source="doc2"),
            RAGDocument(content="C", score=0.7, metadata={}, source="doc3")
        ]
        
        # Test with threshold 0.5 (Should drop B)
        reranked = reranker.rerank(docs, threshold=0.5)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].content, "A")
        self.assertEqual(reranked[1].content, "C")
        
        # Test with threshold 0.8 (Should drop B and C)
        reranked_strict = reranker.rerank(docs, threshold=0.8)
        self.assertEqual(len(reranked_strict), 1)
        self.assertEqual(reranked_strict[0].content, "A")

    def test_doc_filter_min_score(self):
        """Test that DocFilter respects min_score."""
        filterer = DocFilter()
        docs = [
            RAGDocument(content="A", score=0.9, metadata={}, source="doc1"),
            RAGDocument(content="B", score=0.5, metadata={}, source="doc2")
        ]
        
        filtered = filterer.filter(docs, min_score=0.6)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].content, "A")

    async def test_query_rewriter_prompt(self):
        """Test that QueryRewriter uses the new prompt structure (mocked LLM)."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value="rewritten query")
        
        rewriter = QueryRewriter(mock_llm)
        await rewriter.rewrite("test query")
        
        # Check that the prompt contains the new instructions
        call_args = mock_llm.complete.call_args[0][0]
        self.assertIn("Act as an expert search engineer", call_args)
        self.assertIn("Remove conversational filler", call_args)

if __name__ == '__main__':
    unittest.main()
