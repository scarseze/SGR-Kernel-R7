"""
ReplayEngine for SGR Kernel.
Manages deterministic replay and recording of LLM calls.
"""
import json
import os
import hashlib
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

class LLMCallRecord(BaseModel):
    """
    Record of a single LLM call for deterministic replay.
    """
    prompt_hash: str
    model: str
    temperature: float
    seed: Optional[int] = None
    response: str
    usage: Dict[str, Any] = Field(default_factory=dict)
    full_prompt: Optional[str] = None # Debug only

class ReplayTape(BaseModel):
    request_id: str
    records: List[LLMCallRecord] = Field(default_factory=list)

class ReplayEngine:
    """
    Manages recording and replaying LLM interactions.
    """
    def __init__(self, mode: str = "record", storage_path: str = "tapes"):
        self.mode = mode # record, replay, off, mock
        self.storage_path = storage_path
        self._tape: Optional[ReplayTape] = None
        self._replay_index = 0
        os.makedirs(self.storage_path, exist_ok=True)
        
    def start_session(self, request_id: str):
        if self.mode == "record":
            self._tape = ReplayTape(request_id=request_id)
        elif self.mode == "replay":
            self._load_tape(request_id)
            self._replay_index = 0
            
    def _load_tape(self, request_id: str):
        path = os.path.join(self.storage_path, f"{request_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tape {path} not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._tape = ReplayTape.model_validate(data)
            
    def record_call(self, prompt: str, model: str, temperature: float, response: str, usage: Dict[str, int]):
        if self.mode != "record" or not self._tape:
            return
            
        phash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        rec = LLMCallRecord(
            prompt_hash=phash,
            model=model,
            temperature=temperature,
            response=response,
            usage=usage,
            full_prompt=prompt[:500]
        )
        self._tape.records.append(rec)
        
    def get_replay(self, prompt: str) -> Optional[str]:
        if self.mode != "replay" or not self._tape:
            return None
            
        if self._replay_index < len(self._tape.records):
            rec = self._tape.records[self._replay_index]
            # Verify hash?
            # For strict mode, yes.
            phash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if phash != rec.prompt_hash:
                 # Warning: Divergence
                 pass
            
            self._replay_index += 1
            return rec.response
        return None
        
    def save_tape(self):
        if self.mode == "record" and self._tape:
            path = os.path.join(self.storage_path, f"{self._tape.request_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._tape.model_dump_json(indent=2))
