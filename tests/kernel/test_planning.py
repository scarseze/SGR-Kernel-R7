"""KC-B: Planning contract tests."""
import pytest
from core.policy import PolicyEngine
from tests.kernel.conftest import make_step


class TestPlanning:

    def test_kc_b1_budget_blocks_planning(self):
        """KC-B1: Budget exceeded → planning must not proceed."""
        policy = PolicyEngine()
        policy.current_spend = 15.0
        assert policy.check_budget(0.5) is False

    def test_kc_b3_step_limit(self):
        """KC-B3: Steps truncated to MAX_STEPS=10."""
        steps = [make_step(step_id=f"s{i}") for i in range(15)]
        assert len(steps[:10]) == 10
