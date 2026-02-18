"""
Reliability Policy Engine for SGR Kernel.
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class FailureReason(str, Enum):
    """Reasons for failure."""
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    CRITIC_REJECTED = "critic_rejected"
    LOGIC_ERROR = "logic_error"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN = "unknown"

class RecoveryAction(str, Enum):
    """Actions to take upon failure."""
    RETRY_SAME = "retry_same"
    RETRY_WITH_REPAIR = "retry_with_repair"  # Run a repair skill first
    ESCALATE_MODEL = "escalate_model"  # Switch to a more capable model
    FALLBACK_SKILL = "fallback_skill"  # Try a different skill
    DEGRADE_PLAN = "degrade_plan"  # Simplify the objective
    FAIL_FAST = "fail_fast"  # Give up immediately

class RetryPolicy(BaseModel):
    """
    Policy deciding how to handle failures.
    """
    max_retries: int = 3
    backoff_factor: float = 1.5
    
    # Map specific failure reasons to actions
    action_map: Dict[FailureReason, RecoveryAction] = Field(
        default_factory=lambda: {
            FailureReason.SCHEMA_VALIDATION_FAILED: RecoveryAction.RETRY_WITH_REPAIR,
            FailureReason.CRITIC_REJECTED: RecoveryAction.RETRY_WITH_REPAIR,
            FailureReason.LOGIC_ERROR: RecoveryAction.ESCALATE_MODEL,
            FailureReason.TOOL_ERROR: RecoveryAction.FALLBACK_SKILL,
            FailureReason.TIMEOUT: RecoveryAction.DEGRADE_PLAN,
            FailureReason.RESOURCE_EXHAUSTED: RecoveryAction.FAIL_FAST,
            FailureReason.UNKNOWN: RecoveryAction.RETRY_SAME
        }
    )

class RetryGraph:
    """
    Implements the non-linear retry logic (Policy Graph).
    """
    def __init__(self, policy: Optional[RetryPolicy] = None):
        self.policy = policy or RetryPolicy()
    
    def decide_next_action(
        self, 
        failure_reason: FailureReason, 
        current_retry_count: int
    ) -> RecoveryAction:
        """
        Determine the next action based on the failure reason and retry count.
        """
        if current_retry_count >= self.policy.max_retries:
            return RecoveryAction.FAIL_FAST
        
        # Get action from map, default to retry_same
        action = self.policy.action_map.get(failure_reason, RecoveryAction.RETRY_SAME)
        
        return action

class PolicyRegistry:
    """
    Registry for different retry policies.
    """
    _policies: Dict[str, RetryPolicy] = {}

    @classmethod
    def register(cls, name: str, policy: RetryPolicy):
        cls._policies[name] = policy

    @classmethod
    def get(cls, name: str) -> RetryPolicy:
        return cls._policies.get(name, RetryPolicy())

# Register default policies
PolicyRegistry.register("default", RetryPolicy())
PolicyRegistry.register("strict", RetryPolicy(max_retries=1, action_map={
    FailureReason.SCHEMA_VALIDATION_FAILED: RecoveryAction.FAIL_FAST
}))
