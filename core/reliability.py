"""
ReliabilityEngine for SGR Kernel (RFC v2).
Implements Strict Decision Tables and Failure Semantics via Pluggable Strategies.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from core.execution import SemanticFailureType, RetryPolicy

class RecoveryAction(str, Enum):
    """Actions to take upon failure."""
    RETRY = "RETRY" # Simple retry
    REPAIR = "REPAIR" # Delta repair
    ESCALATE = "ESCALATE" # Tier escalation
    FALLBACK_SKILL = "FALLBACK_SKILL"
    ABORT = "ABORT" # Stop step or plan

class ReliabilityStrategy(ABC):
    """
    Pluggable strategy for Reliability Logic (Policy Injection).
    
    API STABILITY: STABLE (v1.x)
    """
    @abstractmethod
    def decide(
        self,
        policy: RetryPolicy,
        failure_type: SemanticFailureType,
        attempts: int,
        tier_used: str
    ) -> RecoveryAction:
        pass

    @abstractmethod
    def get_escalation_tier(self, attempt: int) -> str:
        pass

class RFCv2ReliabilityStrategy(ReliabilityStrategy):
    """
    Default Strategy implementing RFC v2 Decision Tables.
    """
    def decide(
        self,
        policy: RetryPolicy,
        failure_type: SemanticFailureType,
        attempts: int,
        tier_used: str = "fast"
    ) -> RecoveryAction: 
        
        # 1. Check Max Attempts (Table condition: attempts < max)
        if attempts >= policy.max_attempts:
            return RecoveryAction.ABORT 
            
        # 2. Check Abort Policy
        if failure_type in policy.abort_on_fail_types:
            return RecoveryAction.ABORT
            
        # 3. Decision Matrix (RFC v2 Section 4.1)
        if failure_type == SemanticFailureType.SCHEMA_FAIL:
            return RecoveryAction.REPAIR if policy.repair_allowed else RecoveryAction.RETRY
            
        if failure_type == SemanticFailureType.CRITIC_FAIL:
             return RecoveryAction.REPAIR if policy.repair_allowed else RecoveryAction.ESCALATE
             
        if failure_type == SemanticFailureType.LOW_CONFIDENCE:
            return RecoveryAction.ESCALATE
            
        if failure_type == SemanticFailureType.TIMEOUT:
             return RecoveryAction.RETRY
             
        if failure_type == SemanticFailureType.TOOL_ERROR:
             return RecoveryAction.FALLBACK_SKILL if policy.fallback_skill else RecoveryAction.RETRY
             
        if failure_type == SemanticFailureType.CAPABILITY_VIOLATION:
             return RecoveryAction.ABORT
             
        if failure_type == SemanticFailureType.POLICY_VIOLATION:
             return RecoveryAction.ABORT

        return RecoveryAction.RETRY # Default

    def get_escalation_tier(self, attempt: int) -> str:
        """
        RFC v2 Section 4.2: Escalation Table
        """
        if attempt <= 1:
            return "fast"
        if attempt == 2:
            return "mid"
        return "heavy"

class ReliabilityEngine:
    """
    Context for Reliability Strategies.
    """
    
    def __init__(self, strategy: ReliabilityStrategy = None):
        self.strategy = strategy or RFCv2ReliabilityStrategy()
    
    def decide(
        self,
        policy: RetryPolicy,
        failure_type: SemanticFailureType,
        attempts: int,
        tier_used: str = "fast"
    ) -> RecoveryAction: 
        return self.strategy.decide(policy, failure_type, attempts, tier_used)

    def get_escalation_tier(self, attempt: int) -> str:
        return self.strategy.get_escalation_tier(attempt)
