"""
Tests for Wave 2 B.1: Search Space Mutator

Components:
  - MutationGuardrails (min_diversity, max_shrink)
  - compute_param_sensitivity
  - SearchSpaceMutator (shrink, drop, widen)
  - Integration with HPOSkill
"""
import pytest
import asyncio
from unittest.mock import MagicMock

from skills.lora_trainer.search_mutator import (
    SearchSpaceMutator,
    MutationGuardrails,
    MutationReport,
    MutationTrace,
    SensitivityResult,
    compute_param_sensitivity,
)
from skills.lora_trainer.hpo_skill import HPOSkill
from skills.lora_trainer.schema import HPOInput


# ═══════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════

def make_history(n: int, vary_rank=True, vary_dropout=False):
    """Create trial history with various configs and decreasing loss."""
    history = []
    ranks = [8, 16, 32, 64]
    lrs = [1e-5, 5e-5, 1e-4, 2e-4]
    for i in range(n):
        rank = ranks[i % len(ranks)] if vary_rank else 16
        lr = lrs[i % len(lrs)]
        # Lower rank → better loss (for testing shrink_around_best)
        loss = 1.5 + (rank / 64) * 0.5 + (i * 0.02)
        history.append({
            "trial_id": i + 1,
            "config": {
                "lora_rank": rank,
                "learning_rate": lr,
                "lora_dropout": 0.05 if not vary_dropout else [0.0, 0.05, 0.1][i % 3],
                "target_modules": ["q_proj", "v_proj"],
            },
            "eval_loss": loss,
            "status": "done",
        })
    return history


DEFAULT_SPACE = {
    "lora_rank": [8, 16, 32, 64, 128],
    "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
    "lora_dropout": [0.0, 0.05, 0.1],
    "target_modules_presets": {
        "attention_only": ["q_proj", "v_proj"],
        "hybrid": ["q_proj", "v_proj", "gate_proj", "down_proj"],
    },
}


# ═══════════════════════════════════════════
# Guardrails Tests
# ═══════════════════════════════════════════

class TestMutationGuardrails:
    def test_min_diversity_restores(self):
        guard = MutationGuardrails(min_diversity=3)
        original = [8, 16, 32, 64]
        mutated = [16]  # Too few
        result, triggered = guard.enforce(original, mutated, "lora_rank")
        assert len(result) >= 3
        assert any("min_diversity" in t for t in triggered)

    def test_max_shrink_prevents_over_removal(self):
        guard = MutationGuardrails(max_shrink_pct=0.5, min_diversity=1)
        original = [8, 16, 32, 64, 128]
        mutated = [16]  # Removed 4/5 = 80% > 50%
        result, triggered = guard.enforce(original, mutated, "lora_rank")
        assert len(result) >= 3  # Can remove at most 2 of 5
        assert any("max_shrink" in t for t in triggered)

    def test_empty_set_restored(self):
        guard = MutationGuardrails(min_diversity=1)  # Low diversity so empty guard fires
        original = [8, 16, 32]
        mutated = []
        result, triggered = guard.enforce(original, mutated, "lora_rank")
        assert result == original
        assert any("empty" in t for t in triggered)

    def test_no_guardrail_when_ok(self):
        guard = MutationGuardrails(min_diversity=3)
        original = [8, 16, 32, 64]
        mutated = [8, 16, 32]
        result, triggered = guard.enforce(original, mutated, "lora_rank")
        assert len(triggered) == 0
        assert result == mutated


# ═══════════════════════════════════════════
# Sensitivity Tests
# ═══════════════════════════════════════════

class TestSensitivity:
    def test_high_sensitivity_param(self):
        # rank 8 → loss 1.0, rank 64 → loss 3.0 → high range
        history = [
            {"config": {"lora_rank": 8}, "eval_loss": 1.0},
            {"config": {"lora_rank": 64}, "eval_loss": 3.0},
            {"config": {"lora_rank": 8}, "eval_loss": 1.1},
            {"config": {"lora_rank": 64}, "eval_loss": 2.9},
        ]
        result = compute_param_sensitivity(history, "lora_rank")
        assert result.sensitivity_score > 0.5

    def test_low_sensitivity_param(self):
        # All same loss regardless of dropout → low sensitivity
        history = [
            {"config": {"lora_dropout": 0.0}, "eval_loss": 1.5},
            {"config": {"lora_dropout": 0.05}, "eval_loss": 1.51},
            {"config": {"lora_dropout": 0.1}, "eval_loss": 1.49},
            {"config": {"lora_dropout": 0.0}, "eval_loss": 1.5},
        ]
        result = compute_param_sensitivity(history, "lora_dropout")
        assert result.sensitivity_score < 0.05

    def test_insufficient_data(self):
        history = [{"config": {"lora_rank": 8}, "eval_loss": 1.0}]
        result = compute_param_sensitivity(history, "lora_rank")
        assert result.sensitivity_score == 0.5  # default


# ═══════════════════════════════════════════
# Mutator Tests
# ═══════════════════════════════════════════

class TestSearchSpaceMutator:
    def test_no_mutation_insufficient_data(self):
        mutator = SearchSpaceMutator(mutation_trigger_trials=5)
        history = make_history(3)
        space, report = mutator.mutate(DEFAULT_SPACE.copy(), history)
        assert "no_mutation" in report.mutations_applied[0]

    def test_shrink_around_best(self):
        mutator = SearchSpaceMutator(mutation_trigger_trials=5, flat_threshold=0.001)
        history = make_history(10)
        space = {
            "lora_rank": [8, 16, 32, 64, 128],
            "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            "lora_dropout": [0.0, 0.05, 0.1],
        }
        new_space, report = mutator.mutate(space, history)
        # Should have fewer total values than original
        assert report.new_size <= report.original_size
        assert "shrink_around_best" in report.mutations_applied

    def test_widen_if_flat(self):
        mutator = SearchSpaceMutator(mutation_trigger_trials=3, flat_threshold=0.1)
        # All losses nearly identical → flat landscape
        flat_history = [
            {"trial_id": i, "config": {"lora_rank": 16, "learning_rate": 1e-4, "lora_dropout": 0.05},
             "eval_loss": 1.500 + 0.001 * i, "status": "done"}
            for i in range(6)
        ]
        space = {
            "lora_rank": [8, 16, 32],
            "learning_rate": [1e-4, 2e-4],
            "lora_dropout": [0.0, 0.05],
        }
        new_space, report = mutator.mutate(space, flat_history)
        assert "widen_if_flat" in report.mutations_applied
        # Should have more values than original
        assert report.new_size >= report.original_size

    def test_guardrails_enforced(self):
        """Guardrails must prevent excessive shrinkage."""
        mutator = SearchSpaceMutator(
            mutation_trigger_trials=5,
            flat_threshold=0.001,
            guardrails=MutationGuardrails(min_diversity=3, max_shrink_pct=0.3),
        )
        history = make_history(10)
        space = {
            "lora_rank": [8, 16, 32, 64, 128],
            "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            "lora_dropout": [0.0, 0.05, 0.1],
        }
        new_space, report = mutator.mutate(space, history)
        # Check each param still has min_diversity values
        for param in ["lora_rank", "learning_rate"]:
            if param in new_space and isinstance(new_space[param], list):
                assert len(new_space[param]) >= 3


class TestHPOWithMutator:
    """Integration: HPO skill uses mutator."""

    @pytest.mark.asyncio
    async def test_hpo_mutates_after_enough_trials(self):
        skill = HPOSkill()
        history = make_history(8)
        result = await skill.execute(
            HPOInput(
                trial_history=history,
                search_space=DEFAULT_SPACE.copy(),
                strategy="bayesian",
            ),
            state=MagicMock(),
        )
        assert result.data["search_space_mutated"] is True

    @pytest.mark.asyncio
    async def test_hpo_no_mutation_early(self):
        skill = HPOSkill()
        result = await skill.execute(
            HPOInput(
                trial_history=make_history(2),
                search_space=DEFAULT_SPACE.copy(),
                strategy="random",
            ),
            state=MagicMock(),
        )
        assert result.data["search_space_mutated"] is False


class TestMutationTrace:
    """MutationTrace: audit trail with before/after snapshots."""

    def test_trace_recorded_on_mutation(self):
        mutator = SearchSpaceMutator(mutation_trigger_trials=5, flat_threshold=0.001)
        history = make_history(8)
        space = {
            "lora_rank": [8, 16, 32, 64, 128],
            "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            "lora_dropout": [0.0, 0.05, 0.1],
        }
        _, report = mutator.mutate(space, history)
        assert len(mutator.trace_log) == 1
        trace = mutator.trace_log[0]
        assert isinstance(trace, MutationTrace)
        assert trace.search_space_before != {}  # Not empty
        assert trace.search_space_after != {}   # Not empty
        assert trace.mutation_reason != ""
        assert trace.report is not None

    def test_no_trace_when_insufficient_data(self):
        mutator = SearchSpaceMutator(mutation_trigger_trials=5)
        history = make_history(3)
        mutator.mutate(DEFAULT_SPACE.copy(), history)
        assert len(mutator.trace_log) == 0

    def test_trace_before_after_differ(self):
        mutator = SearchSpaceMutator(mutation_trigger_trials=5, flat_threshold=0.001)
        history = make_history(10)
        space = {
            "lora_rank": [8, 16, 32, 64, 128],
            "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            "lora_dropout": [0.0, 0.05, 0.1],
        }
        mutator.mutate(space, history)
        trace = mutator.trace_log[0]
        # Before should have original sizes, after may differ
        assert trace.search_space_before is not trace.search_space_after
