import os
import requests
import logging
from typing import List, Dict, Any, Optional

# Optional Qdrant import
try:
    from qdrant_client import QdrantClient
except ImportError:
    QdrantClient = None

class CoreRAGService:
    def __init__(self, ollama_base_url: str = "http://localhost:11434", qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.logger = logging.getLogger(__name__)
        self.ollama_url = ollama_base_url
        self.enabled = False
        
        # Initialize Qdrant
        self.qdrant = None
        if QdrantClient:
            try:
                self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
                # Check connection (lightweight)
                self.qdrant.get_collections()
                self.enabled = True
                self.logger.info("✅ CoreRAGService connected to Qdrant.")
            except Exception as e:
                self.logger.warning(f"⚠️ CoreRAGService disabled: Qdrant connection failed ({e})")
        else:
            self.logger.warning("⚠️ CoreRAGService disabled: qdrant-client not installed.")

    def get_embedding(self, text: str, model: str = "nomic-embed-text") -> Optional[List[float]]:
        """Generates embedding via Ollama."""
        if not self.enabled: return None
        
        try:
            res = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=10
            )
            if res.status_code == 200:
                return res.json()["embedding"]
            else:
                self.logger.error(f"Ollama Error: {res.text}")
                return None
        except Exception as e:
            self.logger.error(f"Ollama Connection Error: {e}")
            return None

    def search(self, collection_name: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Semantic search in Qdrant.
        Returns list of dicts: {'content': str, 'score': float, 'metadata': dict}
        """
        if not self.enabled:
            return [{"content": "❌ RAG Service is unavailable (Qdrant/Ollama offline).", "score": 0.0}]

        vector = self.get_embedding(query)
        if not vector:
            return [{"content": "❌ Failed to generate embedding for query.", "score": 0.0}]

        try:
            # Check for available search method (compatibility with different qdrant-client versions)
            if hasattr(self.qdrant, 'search'):
                results = self.qdrant.search(
                    collection_name=collection_name, 
                    query_vector=vector,
                    limit=limit
                )
            elif hasattr(self.qdrant, 'query_points'):
                 # New API syntax might be needed here, but let's try basic args
                 # query_points(collection_name, query, limit, with_payload)
                 results = self.qdrant.query_points(
                    collection_name=collection_name,
                    query=vector,
                    limit=limit,
                    with_payload=True
                 ).points
            elif hasattr(self.qdrant, 'search_points'):
                # Another potential alias
                results = self.qdrant.search_points(
                    collection_name=collection_name, 
                    vector=vector, 
                    limit=limit,
                    with_payload=True
                ).result
            else:
                 return [{"content": f"❌ Qdrant client has no compatible search method. Methods: {dir(self.qdrant)}", "score": 0.0}]
            
            output = []
            # ...
            for hit in results:
                doc = {
                    "content": hit.payload.get('content', ''),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != 'content'}
                }
                output.append(doc)
            
            return output

        except Exception as e:
            self.logger.error(f"Qdrant Search Error: {e}")
            return [{"content": f"❌ Search failed: {e}", "score": 0.0}]
