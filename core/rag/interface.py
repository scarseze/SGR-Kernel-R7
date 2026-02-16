from abc import ABC, abstractmethod
from typing import List, Tuple
from core.rag.context import RAGDocument

class RAGInterface(ABC):
    @abstractmethod
    async def run(self, query: str, domain: str = "default") -> Tuple[str, List[RAGDocument]]:
        """
        Execute RAG Pipeline.
        Returns: (formatted_context, list_of_docs)
        """
        pass
