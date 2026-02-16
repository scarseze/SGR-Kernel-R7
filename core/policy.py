import time
from enum import Enum
from typing import Dict, Any, List, Optional
from skills.base import BaseSkill
from core.state import AgentState
from core.types import PolicyStatus, SkillMetadata, Capability, RiskLevel, CostClass # Use Platform Types

class PolicyDecision:
    def __init__(self, status: PolicyStatus, reason: str = ""):
        self.status = status
        self.reason = reason

class PolicyEngine:
    """
    Centralized Policy Enforcement Point (PEP).
    Decides if an action should be allowed, blocked, or require human approval.
    """
    def __init__(self):
        # Configuration (Could be loaded from yaml/env)
        self.enforce_whitelist = True
        self.allowed_roots = ["/data", "./workspace"]
        
        # Context Rules
        self.role_permissions = {
            "guest": {"allow_high_risk": False, "allow_filesystem": False},
            "user": {"allow_high_risk": True, "allow_filesystem": True}, # User needs approval for high risk
            "admin": {"allow_high_risk": True, "allow_filesystem": True}
        }
        
        # Rate Limiting (Simple Token Bucket)
        self.rate_limits = {} # user_id -> {tokens: int, last_refill: float}
        self.refill_rate = 1.0 # tokens per second
        self.max_tokens = 60.0 # Burst size
        
        # Budget Check (Mock)
        self.daily_budget = 10.0 # $
        self.current_spend = 0.0
        
        # RAG Limits & Tuning
        self.rag_max_docs = 10
        self.rag_max_tokens = 6000
        self.rag_min_score = 0.6
        self.rag_rerank_threshold = 0.5 

    def check_budget(self, estimated_cost: float = 0.0) -> bool:
        """Check if adding estimated_cost would exceed budget."""
        if self.daily_budget > 0:
            if self.current_spend + estimated_cost > self.daily_budget:
                return False
        return True

    def record_cost(self, cost: float):
        """Record incurred cost."""
        self.current_spend += cost
        # In a real system, you'd persist this to DB/Redis here.

    def record_step_cost(self, step_id: str, skill_name: str, cost: float):
        """Record cost for a specific step. Enables per-step budget audit."""
        self.current_spend += cost
        if not hasattr(self, 'step_costs'):
            self.step_costs = {}
        self.step_costs[step_id] = {
            "skill": skill_name,
            "cost": cost,
            "cumulative_spend": self.current_spend,
        }

    def check(self, skill: BaseSkill, input_data: Any, state: AgentState) -> PolicyDecision:
        """
        Evaluate policy rules against the proposed action.
        """
        # 0. User Context & Role
        user_role = getattr(state, "user_role", "user") # Default to user
        perms = self.role_permissions.get(user_role, self.role_permissions["guest"])
        
        # 1. Rate Limiting Check
        if not self._check_rate_limit("global_rate", cost=1.0):
             return PolicyDecision(status=PolicyStatus.DENY, reason="Rate limit exceeded.")

        # 2. Metadata-based Rules
        try:
            meta = skill.metadata
            
            # Rule: High Risk checks
            if meta.risk_level == "high":
                if not perms["allow_high_risk"]:
                     return PolicyDecision(status=PolicyStatus.DENY, reason=f"Role '{user_role}' cannot execute High Risk skills.")
                
                return PolicyDecision(
                    status=PolicyStatus.REQUIRE_APPROVAL, 
                    reason=f"Skill '{skill.name}' is classified as High Risk."
                )
            
            # Rule: Filesystem checks
            if meta.requires_filesystem and not perms["allow_filesystem"]:
                return PolicyDecision(status=PolicyStatus.DENY, reason=f"Role '{user_role}' cannot access filesystem.")

            # Rule: Expensive operations
            if meta.cost_class == "expensive":
                if self.current_spend >= self.daily_budget:
                     return PolicyDecision(status=PolicyStatus.DENY, reason="Daily budget exceeded.")
                
                return PolicyDecision(
                    status=PolicyStatus.REQUIRE_APPROVAL, 
                    reason=f"Skill '{skill.name}' is classified as Expensive."
                )

        except Exception as e:
            return PolicyDecision(
                status=PolicyStatus.REQUIRE_APPROVAL, 
                reason=f"Metadata verification failed: {e}"
            )

        # 3. Legacy/Skill-Specific Logic (is_sensitive)
        if hasattr(skill, 'is_sensitive'):
            try:
                if skill.is_sensitive(input_data):
                    return PolicyDecision(
                        status=PolicyStatus.REQUIRE_APPROVAL, 
                        reason=f"Skill '{skill.name}' self-flagged action as sensitive."
                    )
            except Exception as e:
                return PolicyDecision(
                    status=PolicyStatus.DENY,
                    reason=f"Error during sensitivity check: {e}"
                )

        # Default: Allow
        return PolicyDecision(status=PolicyStatus.ALLOW, reason="Policy check passed.")

    def _check_rate_limit(self, key: str, cost: float = 1.0) -> bool:
        """
        Token bucket algorithm.
        """
        now = time.time()
        bucket = self.rate_limits.get(key, {"tokens": self.max_tokens, "last_refill": now})
        
        # Refill
        elapsed = now - bucket["last_refill"]
        tokens_to_add = elapsed * self.refill_rate
        bucket["tokens"] = min(self.max_tokens, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        # Consume
        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            self.rate_limits[key] = bucket
            return True
        else:
            self.rate_limits[key] = bucket
            return False
