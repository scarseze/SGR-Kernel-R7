"""
Core Execution Data Contracts for SGR Kernel (RFC v2).
Strict FSMs and Failure Records.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
try:
    from core.artifacts import ArtifactRef
except ImportError:
    # Forward reference fallback if circular import issues (unlikely here)
    ArtifactRef = Any

# --- Enums (RFC v2 Spec) ---

class ExecutionStatus(str, Enum):
    """Global execution status FSM."""
    CREATED = "CREATED"
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    PAUSED_APPROVAL = "PAUSED_APPROVAL"
    REPAIRING = "REPAIRING"
    ESCALATING = "ESCALATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"

class StepStatus(str, Enum):
    """
    State of a specific step phase FSM.
    """
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    CRITIC = "CRITIC"
    REPAIR = "REPAIR"
    APPROVAL = "APPROVAL"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"
    RETRY_WAIT = "RETRY_WAIT"
    
    # Legacy/Aliases for compat during refactor if needed (or just strict switch)
    # We stick to strict spec.

class SemanticFailureType(str, Enum):
    """
    Classification of failure for Reliability Engine.
    """
    SCHEMA_FAIL = "SCHEMA_FAIL"
    CRITIC_FAIL = "CRITIC_FAIL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    TIMEOUT = "TIMEOUT"
    TOOL_ERROR = "TOOL_ERROR"
    CAPABILITY_VIOLATION = "CAPABILITY_VIOLATION"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    UNKNOWN = "UNKNOWN"

# --- Graph Models ---

class DependencyEdge(BaseModel):
    """Explicit dependency between steps."""
    source_id: str
    target_id: str
    type: str = "hard"

class RetryPolicy(BaseModel):
    """Policy for retrying steps."""
    max_attempts: int = 3
    escalation_tiers: List[str] = Field(default_factory=lambda: ["fast", "mid", "heavy"])
    repair_allowed: bool = True
    fallback_skill: Optional[str] = None
    abort_on_fail_types: List[SemanticFailureType] = Field(default_factory=list)

class StepNode(BaseModel):
    """
    A logical unit of work in the Execution Graph.
    
    API STABILITY: STABLE (v1.x)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    skill_name: str
    
    inputs_template: Dict[str, Any] = Field(default_factory=dict)
    
    required_capabilities: List[str] = Field(default_factory=list)
    
    critic_required: bool = False
    approval_required: bool = False
    
    retry_policy: Optional[RetryPolicy] = None 
    
    description: Optional[str] = None
    
    # Resource Governance (Release Gate v1)
    timeout_seconds: float = Field(default=300.0, description="Max execution time in seconds")
    idempotent: bool = Field(default=False, description="Is this step safe to re-run?")
    idempotency_key: Optional[str] = None

class PlanIR(BaseModel):
    """
    Intermediate Representation of the Execution Plan.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    steps: List[StepNode] = Field(default_factory=list)
    edges: List[DependencyEdge] = Field(default_factory=list)
    
    global_risk_level: str = "low"
    deterministic_required: bool = False
    created_at: float = Field(default_factory=time.time)

# --- Runtime State Models ---

class FailureRecord(BaseModel):
    """
    RFC v2 Spec 5.1: Failure Must Include
    """
    step_id: str
    failure_type: SemanticFailureType
    phase: str # e.g. "EXECUTE", "VALIDATE"
    error_class: str
    retryable: bool
    repairable: bool
    timestamp: float = Field(default_factory=time.time)
    error_message: Optional[str] = None

class StepState(BaseModel):
    """
    Runtime state of a single step.
    """
    step_id: str
    status: StepStatus = StepStatus.PENDING
    
    attempts: int = 0
    tier_used: Optional[str] = None
    
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    
    confidence: float = 1.0
    failure: Optional[FailureRecord] = None # Last failure
    
    output: Any = None
    
    # Track Phase Transitions?
    history: List[str] = Field(default_factory=list)

class ExecutionState(BaseModel):
    """
    Single Source of Truth for Kernel Runtime.
    
    API STABILITY: STABLE (v1.x)
    """
    request_id: str
    plan_id: Optional[str] = None
    
    input_payload: Any
    
    plan_ir: Optional[PlanIR] = None
    
    step_states: Dict[str, StepState] = Field(default_factory=dict)
    
    # Data artifacts (Release Gate v1: Store Refs)
    artifacts: Dict[str, ArtifactRef] = Field(default_factory=dict)
    skill_outputs: Dict[str, Any] = Field(default_factory=dict)
    llm_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Logs
    critic_reports: List[Dict[str, Any]] = Field(default_factory=list)
    repair_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    retry_log: List[Dict[str, Any]] = Field(default_factory=list)
    escalation_log: List[Dict[str, Any]] = Field(default_factory=list)
    
    approvals: List[Dict[str, Any]] = Field(default_factory=list)
    
    checkpoints: List[str] = Field(default_factory=list)
    
    status: ExecutionStatus = ExecutionStatus.CREATED
    
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # Resource Governance (Release Gate v1)
    token_budget: Optional[int] = None
    tokens_used: int = 0
    llm_call_budget: Optional[int] = None
    llm_calls_used: int = 0

    def initialize_step(self, step_id: str):
        if step_id not in self.step_states:
            self.step_states[step_id] = StepState(step_id=step_id)
        self.updated_at = time.time()
        
    def get_step_output(self, step_id: str) -> Any:
        return self.skill_outputs.get(step_id)
