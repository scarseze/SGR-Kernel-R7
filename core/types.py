import time
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

# --- Enums ---
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CostClass(str, Enum):
    CHEAP = "cheap"
    NORMAL = "normal"
    EXPENSIVE = "expensive"

class RetryPolicy(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"

class PolicyStatus(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    FAILED = "failed"
    BLOCKED = "blocked"
    COMPLETED = "completed" # Success

class Capability(str, Enum):
    REASONING = "reasoning"
    WEB = "web"
    FILESYSTEM = "filesystem"
    DB = "db"
    CODE = "code"
    API = "api"
    LLM = "llm"
    DEEP_RESEARCH = "deep_research" # Legacy support / specific
    PLANNING = "planning"
    REPORT_WRITING = "report_writing"

# --- Metadata Models ---
class SkillMetadata(BaseModel):
    name: str = "unknown"
    version: str = "1.0.0"
    description: str = ""
    
    capabilities: List[Capability]
    risk_level: RiskLevel = RiskLevel.LOW
    cost_class: CostClass = CostClass.CHEAP
    retry_policy: RetryPolicy = RetryPolicy.NONE

    # Tier Routing Constraints
    min_tier: Optional[str] = None
    max_tier: Optional[str] = None

    @field_validator('retry_policy', mode='before')
    @classmethod
    def normalize_retry_policy(cls, v: Any) -> RetryPolicy:
        if isinstance(v, str):
            try:
                return RetryPolicy(v.lower())
            except ValueError:
                return RetryPolicy.NONE
        return v
    
    side_effects: bool = False
    idempotent: bool = False
    deterministic: bool = False
    
    requires_network: bool = False
    requires_filesystem: bool = False
    requires_gpu: bool = False
    requires_sandbox: bool = False
    
    requires_approval_hint: bool = False
    
    # Budget: Estimated cost per execution (used by policy.record_step_cost)
    estimated_cost: float = 0.0
    
    # INVARIANT: timeout_sec is the single source of truth for execution timeout.
    # TimeoutMiddleware is the only authority that writes ctx.timeout from this value.
    # Do NOT set ctx.timeout from any other source.
    timeout_sec: float = 60.0
    max_concurrency: int = 1
    cooldown_sec: float = 0.0
    
    output_schema: Optional[Dict[str, Any]] = None

class SkillManifest(BaseModel):
    """YAML representation of metadata."""
    name: str
    version: str = "1.0.0"
    description: str
    entrypoint: str = "handler.py"
    class_name: Optional[str] = None
    
    requires: List[str] = []
    capabilities: List[Capability] = []
    
    risk_level: RiskLevel = RiskLevel.LOW
    cost_class: CostClass = CostClass.CHEAP
    
    side_effects: bool = False
    idempotent: bool = False
    deterministic: bool = False
    
    requires_network: bool = False
    requires_filesystem: bool = False
    requires_sandbox: bool = False
    requires_approval_hint: bool = False
    
    # Tier Routing Constraints
    min_tier: Optional[str] = None # e.g. "mid", "heavy"
    max_tier: Optional[str] = None # e.g. "fast"
    # End Tier Constraints
    
    timeout_sec: float = 60.0
    retry_policy: RetryPolicy = RetryPolicy.NONE
    output_schema: Optional[Dict[str, Any]] = None

# --- Execution Context ---
class SkillExecutionContext(BaseModel):
    request_id: str
    step_id: str
    
    skill_name: str
    params: Dict[str, Any]
    
    # We avoid circular dependency by not typing 'skill' or 'state' strictly here yet, 
    # or we handle it via ForwardRefs if needed. 
    # For Pydantic models to pass through, we might need arbitrary types allowed.
    state: Any = Field(..., description="AgentState object")
    skill: Any = Field(..., description="BaseSkill instance")
    
    metadata: SkillMetadata
    trace: Any = Field(..., description="StepTrace object")
    
    start_time: float = Field(default_factory=time.time)
    
    # Middleware-specific scratchpad
    timeout: float = 0.0
    attempt: int = 1
    
    # Tier Routing Injection
    # The engine injects the specific LLM service selected by the router for this execution
    llm: Optional[Any] = Field(None, description="Router-selected LLMService")

    @property
    def is_retry(self) -> bool:
        """True if this is not the first attempt. Middleware should check this to avoid side-effect duplication."""
        return self.attempt > 1
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
