from abc import ABC, abstractmethod
from typing import List
import httpx

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url
        self.model = model

    async def embed(self, text: str) -> List[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Ollama doesn't always support batch well, loop for now or usage specific endpoint if available
        # n.b. efficient implementation would use gather
        import asyncio
        return await asyncio.gather(*[self.embed(t) for t in texts])

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        
    async def embed(self, text: str) -> List[float]:
        # Placeholder for OpenAI implementation
        raise NotImplementedError("OpenAI embeddings not yet configured")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("OpenAI embeddings not yet configured")
