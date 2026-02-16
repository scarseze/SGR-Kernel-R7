"""FakePolicy — controllable policy engine for tests."""
from core.types import PolicyStatus
from core.policy import PolicyDecision


class FakePolicy:
    def __init__(self, budget_ok=True, daily_budget=10.0):
        self.budget_ok = budget_ok
        self.daily_budget = daily_budget
        self.current_spend = 0.0
        self.costs = []
        self.step_costs = {}

    def check(self, skill, input_data, state):
        """Policy gate — always ALLOW in tests."""
        return PolicyDecision(status=PolicyStatus.ALLOW, reason="test allow")


    def check_budget(self, estimated_cost=0.0):
        return self.budget_ok

    def record_cost(self, cost):
        self.costs.append(cost)
        self.current_spend += cost

    def record_step_cost(self, step_id, skill_name, cost):
        self.step_costs[step_id] = {
            "skill": skill_name, "cost": cost,
            "cumulative_spend": self.current_spend + cost,
        }
        self.current_spend += cost

    # RAG policy defaults
    rag_max_docs = 5
    rag_max_tokens = 2000
    rag_min_score = 0.1
    rag_rerank_threshold = 0.2
