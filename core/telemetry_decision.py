"""
Decision Telemetry and Structured Logging for SGR Kernel.
"""
import time
import json
import logging
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger("decision_telemetry")

class DecisionEvent(BaseModel):
    """
    Structured record of a key runtime decision.
    """
    timestamp: float = Field(default_factory=time.time)
    request_id: str
    step_id: Optional[str] = None
    
    event_type: str # e.g. "routing", "retry", "model_selection", "critic"
    
    # Context
    decision: str 
    reason: str
    confidence: float
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DecisionTracer:
    """
    Instruments the runtime to capture why decisions were made.
    """
    
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        
    def log_decision(self, event: DecisionEvent):
        """
        Log a decision event.
        """
        # 1. Log to standard logger (structured)
        logger.info(f"[DECISION] {event.event_type}: {event.decision} (conf={event.confidence:.2f}) - {event.reason}")
        
        # 2. Log to file if configured (for future ML training)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(event.model_dump_json() + "\n")
                
    def log_routing(self, request_id: str, step_id: str, decision: str, reason: str, confidence: float):
        self.log_decision(DecisionEvent(
            request_id=request_id,
            step_id=step_id,
            event_type="routing",
            decision=decision,
            reason=reason,
            confidence=confidence
        ))

    def log_retry(self, request_id: str, step_id: str, reason: str, action: str):
        self.log_decision(DecisionEvent(
            request_id=request_id,
            step_id=step_id,
            event_type="retry",
            decision=action,
            reason=reason,
            confidence=1.0 
        ))

# Global Tracer instance
_tracer = DecisionTracer()

def get_tracer() -> DecisionTracer:
    return _tracer
