"""
Confidence-Driven Routing Logic for SGR Kernel.
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from core.execution import PlanIR
from core.validator import ValidationResult

class RoutingAction(str, Enum):
    """Actions taken by the router."""
    PROCEED = "proceed"
    ESCALATE = "escalate"
    REPLAN = "replan"
    CRITIC_REVIEW = "critic_review"
    REQUEST_APPROVAL = "request_approval"

class RouterConfig(BaseModel):
    """Configuration for routing thresholds."""
    min_confidence_proceed: float = 0.8
    min_confidence_critic: float = 0.6
    # Below 0.6 -> Replan or Escalate

class RoutingDecision(BaseModel):
    """Result of a routing decision."""
    action: RoutingAction
    target_tier: Optional[str] = None
    reason: str
    confidence: float

class Router:
    """
    Decides the next step based on confidence and validation results.
    """
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        
    def route_step(
        self, 
        plan_confidence: float, 
        validation_result: ValidationResult
    ) -> RoutingDecision:
        """
        Determine execution path based on confidence.
        """
        # 1. Hard validation fail -> Replan needed
        if not validation_result.valid:
            return RoutingDecision(
                action=RoutingAction.REPLAN,
                reason=f"Validation failed: {validation_result.errors}",
                confidence=0.0
            )
            
        # 2. Check confidence thresholds
        if plan_confidence >= self.config.min_confidence_proceed:
            return RoutingDecision(
                action=RoutingAction.PROCEED,
                reason="High confidence plan.",
                confidence=plan_confidence
            )
            
        if plan_confidence >= self.config.min_confidence_critic:
            return RoutingDecision(
                action=RoutingAction.CRITIC_REVIEW,
                reason="Moderate confidence. Requires critic review.",
                confidence=plan_confidence
            )
            
        # 3. Low confidence -> Escalate or Replan
        return RoutingDecision(
            action=RoutingAction.ESCALATE,
            reason="Low confidence. Escalating to higher tier model.",
            confidence=plan_confidence
        )
