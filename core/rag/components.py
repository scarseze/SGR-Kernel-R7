from typing import List, Optional
from core.rag.context import RAGDocument
from core.llm import LLMService

class QueryRewriter:
    def __init__(self, llm: LLMService):
        self.llm = llm

    async def rewrite(self, query: str) -> str:
        # Improved Prompt with Zero-Shot CoT
        prompt = f"""
        Act as an expert search engineer. Your goal is to rewrite the user's query to maximize retrieval quality for a semantic vector search engine.
        
        Rules:
        1. Remove conversational filler ("hello", "please", "can you").
        2. Expand acronyms if ambiguous.
        3. Convert questions into declarative statements or keyword-rich phrases.
        4. Focus on technical entities and domain-specific terms.
        5. Return ONLY the rewritten query text. No explanations.

        User Query: "{query}"
        Rewritten Query:
        """
        response = await self.llm.complete(prompt)
        return response.strip().strip('"')

class QueryExpander:
    async def expand(self, query: str) -> List[str]:
        # Simple expansion strategy
        return [
            query,
            f"detailed explanation of {query}",
            f"technical description of {query}"
        ]

class DomainRouter:
    def route(self, query: str) -> List[str]:
        q = query.lower()
        if "python" in q or "code" in q or "function" in q or "class" in q:
            return ["code"]
        if "policy" in q or "rule" in q or "security" in q:
            return ["docs"]
        return ["default"]

class ScoreReranker:
    def rerank(self, docs: List[RAGDocument], threshold: float = 0.0) -> List[RAGDocument]:
        # Sort by score descending
        sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)
        # Apply optional secondary threshold if provided
        if threshold > 0:
            sorted_docs = [d for d in sorted_docs if d.score >= threshold]
        return sorted_docs

class DocFilter:
    def filter(self, docs: List[RAGDocument], min_score: float = 0.6) -> List[RAGDocument]:
        return [d for d in docs if d.score >= min_score]
