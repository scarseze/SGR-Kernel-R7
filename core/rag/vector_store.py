from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class VectorSearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None

class VectorStoreAdapter(ABC):
    @abstractmethod
    async def search(self, collection: str, vector: List[float], limit: int = 5, score_threshold: float = 0.0) -> List[VectorSearchResult]:
        pass
    
    @abstractmethod
    async def upsert(self, collection: str, points: List[Any]):
        pass

class QdrantAdapter(VectorStoreAdapter):
    def __init__(self, host: str, port: int, api_key: Optional[str] = None):
        from qdrant_client import QdrantClient
        # Async client is better for production, but using sync for compatibility with current env mostly
        # Switching to AsyncQdrantClient if possible
        try:
             from qdrant_client import AsyncQdrantClient
             self.client = AsyncQdrantClient(host=host, port=port, api_key=api_key)
        except ImportError:
             self.client = QdrantClient(host=host, port=port, api_key=api_key)
             
    async def search(self, collection: str, vector: List[float], limit: int = 5, score_threshold: float = 0.0) -> List[VectorSearchResult]:
        # Check if client is async
        import inspect
        
        args = {
            "collection_name": collection,
            "query_vector": vector,
            "limit": limit,
            "score_threshold": score_threshold,
            "with_payload": True
        }
        
        if inspect.iscoroutinefunction(self.client.search):
            results = await self.client.search(**args)
        else:
            results = self.client.search(**args)
            
        return [
            VectorSearchResult(
                id=str(hit.id),
                score=hit.score,
                payload=hit.payload or {}
            )
            for hit in results
        ]

    async def upsert(self, collection: str, points: List[Any]):
        # Implementation depends on PointStruct format
        if hasattr(self.client, "upsert_async"): # older async clients
             await self.client.upsert(collection_name=collection, points=points)
        elif hasattr(self.client, "upsert"):
             if inspect.iscoroutinefunction(self.client.upsert):
                  await self.client.upsert(collection_name=collection, points=points)
             else:
                  self.client.upsert(collection_name=collection, points=points)
