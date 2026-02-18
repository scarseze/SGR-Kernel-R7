"""
Artifact Store Abstraction for SGR Kernel.
Implements immutable, content-addressed storage contract (Release Gate v1).
"""
import os
import json
import time
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field

class ArtifactRef(BaseModel):
    """
    Reference to an immutable artifact.
    """
    id: str
    key: str # User-friendly key (e.g., "plan_v1")
    uri: str
    size_bytes: int
    created_at: float = Field(default_factory=time.time)
    hash_sha256: str
    content_type: str = "application/json"

class ArtifactStore(ABC):
    """
    Abstract contract for artifact persistence.
    
    API STABILITY: STABLE (v1.x)
    """
    
    @abstractmethod
    def put(self, key: str, data: Any) -> ArtifactRef:
        """
        Store an artifact.
        """
        pass
    
    @abstractmethod
    def get(self, ref: ArtifactRef) -> Any:
        """
        Retrieve an artifact.
        """
        pass
        
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if artifact exists.
        """
        pass

class LocalArtifactStore(ArtifactStore):
    """
    File-system based artifact store.
    """
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        
    def put(self, key: str, data: Any) -> ArtifactRef:
        # 1. Serialize
        # Simple JSON serialization for MVP
        if isinstance(data, (dict, list)):
            content_bytes = json.dumps(data, default=str).encode("utf-8")
            content_type = "application/json"
        elif isinstance(data, str):
            content_bytes = data.encode("utf-8")
            content_type = "text/plain"
        elif isinstance(data, bytes):
            content_bytes = data
            content_type = "application/octet-stream"
        else:
            # Fallback to string repr
            content_bytes = str(data).encode("utf-8")
            content_type = "text/plain"
            
        # 2. Hash
        sha256 = hashlib.sha256(content_bytes).hexdigest()
        
        # 3. Write (Content-Addressed)
        # Store by hash to ensure immutability/deduplication
        filename = f"{sha256}.dat"
        file_path = os.path.join(self.base_path, filename)
        
        # Atomic Write
        temp_path = file_path + ".tmp"
        with open(temp_path, "wb") as f:
            f.write(content_bytes)
        os.replace(temp_path, file_path)
        
        # 4. Return Ref
        return ArtifactRef(
            id=sha256,
            key=key,
            uri=f"file://{file_path}",
            size_bytes=len(content_bytes),
            hash_sha256=sha256,
            content_type=content_type
        )
        
    def get(self, ref: ArtifactRef) -> Any:
        path = ref.uri.replace("file://", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact {ref.id} not found at {path}")
            
        with open(path, "rb") as f:
            content_bytes = f.read()
            
        # Verify Hash
        current_hash = hashlib.sha256(content_bytes).hexdigest()
        if current_hash != ref.hash_sha256:
             raise ValueError(f"Artifact Integrity Check Failed! Expected {ref.hash_sha256}, got {current_hash}")
             
        if ref.content_type == "application/json":
            return json.loads(content_bytes.decode("utf-8"))
        elif ref.content_type == "text/plain":
            return content_bytes.decode("utf-8")
        else:
            return content_bytes

    def exists(self, key: str) -> bool:
        # Local store is content-addressed, so checking by 'key' implies checking external registry mapped to this store.
        # But here 'key' is just metadata. 
        # For this simple impl, we always return False as we don't index keys -> refs mapping locally in the store itself.
        # The ExecutionState holds the mapping.
        return False 
