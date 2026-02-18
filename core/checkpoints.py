"""
Checkpoint Subsystem for SGR Kernel (Target Model).
"""
import os
import time
import json
import logging
from typing import Optional, Any, Tuple, Dict
from pydantic import BaseModel, Field
from core.execution import ExecutionState

logger = logging.getLogger(__name__)

class Checkpoint(BaseModel):
    """
    Formal Checkpoint Object.
    """
    state_snapshot: Dict[str, Any] # Serialized ExecutionState
    step_id: Optional[str]
    timestamp: float = Field(default_factory=time.time)
    reason: str # e.g. "step_complete", "planner_output", "approval_required"

class CheckpointManager:
    """
    Manages persistence of Checkpoints.
    """
    
    def __init__(self, storage_path: str = "checkpoints"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _get_path(self, request_id: str, filename: str) -> str:
        request_dir = os.path.join(self.storage_path, request_id)
        os.makedirs(request_dir, exist_ok=True)
        return os.path.join(request_dir, filename)
    
    def save_checkpoint(
        self, 
        state: ExecutionState, 
        reason: str
    ) -> str:
        """
        Save the current execution state to a checkpoint file.
        Returns the absolute path of the checkpoint.
        """
        # Create Snapshot
        snapshot = state.model_dump()
        
        # Create Checkpoint Object
        ckpt = Checkpoint(
            state_snapshot=snapshot,
            step_id=state.current_step_id if hasattr(state, 'current_step_id') else None, # ExecutionState removed current_step_id logic largely? 
            # Wait, ExecutionState has `step_status` but no single pointer. 
            # We can use the last active step or pass it in.
            # For now, let's assume reason context handles it or pass None.
            # Actually, let's add `step_id` arg to save_checkpoint if needed.
            reason=reason
        )
        
        # Filename strategy: {timestamp}_{reason}.json
        timestamp = int(ckpt.timestamp * 1000)
        filename = f"{timestamp}_{reason}.json"
        path = self._get_path(state.request_id, filename)
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ckpt.model_dump(), f, indent=2, default=str)
            logger.info(f"Checkpoint saved: {path}")
            
            # Update state (in memory only, avoid recursion loop)
            state.checkpoints.append(path)
            
            return path
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, path: str) -> Tuple[ExecutionState, Checkpoint]:
        """
        Load ExecutionState from a checkpoint file.
        Returns (ExecutionState, CheckpointObject).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            ckpt = Checkpoint.model_validate(data)
            execution_state = ExecutionState.model_validate(ckpt.state_snapshot)
            
            return execution_state, ckpt
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {e}")
            raise

    def get_latest_checkpoint(self, request_id: str) -> Optional[str]:
        """Find the most recent checkpoint for a request."""
        request_dir = os.path.join(self.storage_path, request_id)
        if not os.path.exists(request_dir):
            return None
            
        files = [
            os.path.join(request_dir, f) 
            for f in os.listdir(request_dir) 
            if f.endswith(".json")
        ]
        
        if not files:
            return None
            
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
