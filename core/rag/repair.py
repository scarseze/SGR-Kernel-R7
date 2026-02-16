from typing import List, Optional
from core.llm import LLMService
from core.rag.context import RAGDocument

class AnswerCritic:
    def __init__(self, llm: LLMService):
        self.llm = llm

    async def critique(self, query: str, context: str) -> bool:
        """
        Evaluates if the retrieved context is sufficient to answer the query.
        Returns True if sufficient, False otherwise.
        """
        prompt = f"""
        You are an expert judge of information retrieval quality.
        Target Query: "{query}"

        Retrieved Context:
        {context[:8000]} # Limit check

        Task: Determine if the context contains sufficient information to answer the query comprehensively.
        Output "YES" if sufficient, "NO" if missing key information.
        Only output YES or NO.
        """
        response = await self.llm.complete(prompt)
        return "YES" in response.upper()

class RepairStrategy:
    def suggest_fix(self, query: str, attempt: int) -> str:
        """
        Suggests a new query strategy based on failure count.
        """
        if attempt == 1:
            return f"broaden: {query}"
        elif attempt == 2:
            return f"keywords: {query}"
        return query # Give up?

class RepairableRAG:
    """
    Mixin or Wrapper to add repair loop to RAGPipeline.
    For now, implemented as a separate logic helper.
    """
    pass
