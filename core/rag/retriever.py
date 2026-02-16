from typing import List, Optional, Dict, Any
from core.rag.embeddings import EmbeddingProvider
from core.rag.vector_store import VectorStoreAdapter
from core.rag.context import RAGDocument

class RAGRetriever:
    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStoreAdapter):
        self.embedder = embedder
        self.store = vector_store

    async def retrieve(self, query: str, collection: str = "default", limit: int = 5) -> List[RAGDocument]:
        # 1. Embed
        vector = await self.embedder.embed(query)
        
        # 2. Search
        results = await self.store.search(collection=collection, vector=vector, limit=limit)
        
        # 3. Map to RAGDocument
        docs = []
        for hit in results:
            content = hit.payload.get("content", "")
            if not content:
                continue
                
            docs.append(RAGDocument(
                content=str(content),
                score=hit.score,
                source=hit.payload.get("source", "unknown"),
                metadata=hit.payload
            ))
            
        return docs
