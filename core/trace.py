import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
import logging

from core.planner import ExecutionPlan

import logging
import contextvars

logger = logging.getLogger("core.trace")

# Global Context for implicit tracing
current_step_trace = contextvars.ContextVar("current_step_trace", default=None)

class LLMCallTrace(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    context: str = "unknown" # e.g. "planning", "skill:research"

class PolicyEvent(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    step_id: str
    skill_name: str
    decision: str # "ALLOW", "DENY", "REQUIRE_APPROVAL"
    reason: str

class ApprovalEvent(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    step_id: str
    skill_name: str
    approved: bool
    approver: str = "user"
    reason: Optional[str] = None

class RAGQueryTrace(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    domains: List[str] = []
    latency_ms: float = 0.0
    found_docs: int = 0
    used_docs: int = 0
    sources: List[str] = []
    attempt: int = 1
    critique_passed: Optional[bool] = None
    repair_strategy: Optional[str] = None

class AttemptTrace(BaseModel):
    attempt_number: int
    start_time: float
    end_time: float = 0.0
    error: Optional[str] = None
    result_snippet: Optional[str] = None
    tier: Optional[str] = None # Log which tier was used (fast, mid, heavy)

class StepTrace(BaseModel):
    step_id: str
    skill_name: str
    input_params: Dict[str, Any]
    output_data: Optional[str] = None
    attempts: List[AttemptTrace] = Field(default_factory=list)
    start_time: float = 0.0
    duration: float = 0.0
    status: str = "pending" # mapped to StepStatus values
    error: Optional[str] = None
    policy_events: List[PolicyEvent] = Field(default_factory=list)
    approval_events: List[ApprovalEvent] = Field(default_factory=list)
    llm_calls: List[LLMCallTrace] = Field(default_factory=list)
    rag_queries: List[RAGQueryTrace] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    policy_required: bool = False

class RequestTrace(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_request: str
    plan: Optional[ExecutionPlan] = None
    steps: List[StepTrace] = Field(default_factory=list)
    llm_calls: List[LLMCallTrace] = Field(default_factory=list) # Planner LLM calls
    total_duration: float = 0.0
    execution_order: List[str] = Field(default_factory=list) # Trace of actual execution sequence
    plan_hash: str = "" # SHA256 of the initial plan for drift detection
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "running" # running, completed, error, security_error
    error: Optional[str] = None

class TraceManager:
    def __init__(self, trace_dir: str = "traces"):
        # Ensure absolute path relative to project root (approx)
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.trace_dir = os.path.join(base_path, trace_dir)
        
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir, exist_ok=True)

    def save_trace(self, trace: RequestTrace):
        try:
            # Organise by date YYYY-MM-DD
            date_str = datetime.now().strftime("%Y-%m-%d")
            day_dir = os.path.join(self.trace_dir, date_str)
            os.makedirs(day_dir, exist_ok=True)
            
            filename = f"{trace.request_id}.json"
            filepath = os.path.join(day_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(trace.model_dump_json(indent=2))
                
            logger.info(f"Trace saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")

    def get_last_trace(self) -> Optional[RequestTrace]:
        """Retrieves the most recent trace from disk."""
        try:
            # 1. Find latest day directory
            if not os.path.exists(self.trace_dir):
                return None
                
            days = sorted([d for d in os.listdir(self.trace_dir) if os.path.isdir(os.path.join(self.trace_dir, d))])
            if not days:
                return None
                
            latest_day = days[-1]
            day_path = os.path.join(self.trace_dir, latest_day)
            
            # 2. Find latest file in that directory
            files = sorted([f for f in os.listdir(day_path) if f.endswith(".json")])
            if not files:
                return None
                
            latest_file = files[-1] # Assuming standard sorting works for UUIDs/Time or we rely on OS creation time?
            # Actually UUIDs are random. We need to check file modification time or content.
            # Better: Sort by modification time.
            
            full_paths = [os.path.join(day_path, f) for f in files]
            latest_path = max(full_paths, key=os.path.getmtime)
            
            with open(latest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return RequestTrace(**data)
                
        except Exception as e:
            logger.error(f"Failed to load last trace: {e}")
            return None
