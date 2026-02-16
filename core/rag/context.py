from typing import List, Dict, Any
from pydantic import BaseModel

class RAGDocument(BaseModel):
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]

class RAGContextBuilder:
    def compress(self, docs: List[RAGDocument], max_tokens: int = 4000) -> List[RAGDocument]:
        """
        Naive compression based on char length approximation (4 chars ~= 1 token).
        """
        out = []
        total_chars = 0
        limit_chars = max_tokens * 4
        
        for d in docs:
            length = len(d.content)
            if total_chars + length > limit_chars:
                break
            out.append(d)
            total_chars += length
            
        return out

    def format(self, docs: List[RAGDocument]) -> str:
        """
        Format docs into a single string for LLM context.
        """
        blocks = []
        for i, d in enumerate(docs):
            header = f"[DOC {i+1} | score={d.score:.2f} | source={d.source}]"
            blocks.append(f"{header}\n{d.content}")
            
        return "\n\n".join(blocks)
