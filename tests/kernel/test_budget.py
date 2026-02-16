"""KC-J: Budget enforcement tests."""
import pytest
from core.policy import PolicyEngine


class TestBudgetEnforcement:

    def test_kc_j1_budget_stops_on_overspend(self):
        """Budget check returns False when spent > limit."""
        policy = PolicyEngine()
        policy.daily_budget = 5.0
        policy.current_spend = 4.5
        assert policy.check_budget(1.0) is False
        assert policy.check_budget(0.4) is True

    def test_kc_j2_step_cost_recorded(self):
        """record_step_cost tracks per-step spend."""
        policy = PolicyEngine()
        policy.record_step_cost("s1", "research", 2.5)
        policy.record_step_cost("s2", "analysis", 1.0)
        assert policy.step_costs["s1"]["cost"] == 2.5
        assert policy.step_costs["s2"]["cost"] == 1.0
        assert policy.current_spend == 3.5

    def test_kc_j2_planner_cost_recorded(self):
        """record_cost tracks planner LLM spend."""
        policy = PolicyEngine()
        policy.record_cost(0.05)
        assert policy.current_spend == 0.05
        policy.record_cost(0.03)
        assert policy.current_spend == 0.08
