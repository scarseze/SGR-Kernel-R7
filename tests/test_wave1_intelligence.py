"""
Tests for Wave 1: Research Agent Intelligence

Components tested:
  A.1 — MetaPlannerSkill (ResearchPlan DSL, PlanValidator, plan→DAG compiler)
  A.2 — MethodSelectorSkill (DatasetSignals, rule engine)
  A.3 — LoopController (plateau, budget, variance, patience)
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

# ── A.1: Meta Planner ──
from skills.lora_trainer.meta_planner import (
    ResearchAction,
    ResearchStep,
    ResearchPlan,
    PlanValidator,
    PlanValidationError,
    MetaPlannerSkill,
    MetaPlannerInput,
    plan_to_execution_plan,
)

# ── A.2: Method Selector ──
from skills.lora_trainer.method_selector import (
    DatasetSignals,
    MethodSelection,
    MethodSelectorSkill,
    MethodSelectorInput,
    extract_signals,
    select_by_rules,
)

# ── A.3: Loop Controller ──
from skills.lora_trainer.loop_controller import (
    LoopController,
    ContinueDecision,
    LoopMetrics,
)

from skills.lora_trainer.experiment_spec import ExperimentSpec, PEFTMethod
from skills.lora_trainer.state import ExperimentState, TrialRecord
from skills.lora_trainer.experiment_spec import TrialConfig, TrialStatus


# ═══════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════

@pytest.fixture
def default_spec():
    return ExperimentSpec(
        goal="improve instruction tuning quality",
        dataset_path="data/train.jsonl",
        budget_gpu_hours=40,
        max_trials=20,
    )

@pytest.fixture
def small_spec():
    return ExperimentSpec(
        goal="quick test",
        dataset_path="data/test.jsonl",
        budget_gpu_hours=3,
        max_trials=5,
    )

@pytest.fixture
def large_spec():
    return ExperimentSpec(
        goal="full ablation study",
        dataset_path="data/full.jsonl",
        budget_gpu_hours=50,
        max_trials=30,
    )


def make_trial(trial_id: int, eval_loss: float, gpu_hours: float = 0.5, status=TrialStatus.DONE):
    """Helper: create a TrialRecord with given metrics."""
    return TrialRecord(
        trial_id=trial_id,
        config=TrialConfig(trial_id=trial_id),
        status=status,
        eval_loss=eval_loss,
        gpu_hours=gpu_hours,
    )


def make_state(
    trials: list[TrialRecord] = None,
    budget_total: float = 40.0,
    max_trials: int = 20,
    patience: int = 0,
) -> ExperimentState:
    """Helper: create ExperimentState with provided data."""
    state = ExperimentState(
        budget_total_gpu_hours=budget_total,
        budget_max_trials=max_trials,
    )
    for t in (trials or []):
        state.register_trial(t)
    state.trials_without_improvement = patience
    return state


# ═══════════════════════════════════════════
# A.1: Meta Planner Tests
# ═══════════════════════════════════════════

class TestResearchPlan:
    """Test ResearchPlan DSL validation."""

    def test_valid_plan(self):
        plan = ResearchPlan(
            steps=[
                ResearchStep(step_id="step_1", action=ResearchAction.ABLATION, params={}),
                ResearchStep(
                    step_id="step_2", action=ResearchAction.HPO,
                    params={"strategy": "bayesian"}, depends_on=["step_1"],
                ),
            ],
            reasoning="test",
            estimated_total_gpu_hours=5,
        )
        assert len(plan.steps) == 2
        assert plan.steps[1].depends_on == ["step_1"]

    def test_empty_plan_rejected(self):
        with pytest.raises(Exception):
            ResearchPlan(steps=[], reasoning="empty")


class TestPlanValidator:
    """Test PlanValidator constraint checking."""

    def test_valid_plan_passes(self):
        plan = ResearchPlan(
            steps=[
                ResearchStep(step_id="s1", action=ResearchAction.HPO, estimated_gpu_hours=5),
                ResearchStep(step_id="s2", action=ResearchAction.EVAL, depends_on=["s1"]),
            ],
            reasoning="ok",
            estimated_total_gpu_hours=5,
        )
        validator = PlanValidator(max_gpu_hours=10, max_trials=20)
        errors = validator.validate(plan)
        assert len(errors) == 0

    def test_budget_exceeded(self):
        plan = ResearchPlan(
            steps=[
                ResearchStep(step_id="s1", action=ResearchAction.HPO, estimated_gpu_hours=15),
            ],
            reasoning="over budget",
        )
        validator = PlanValidator(max_gpu_hours=10, max_trials=20)
        errors = validator.validate(plan)
        assert any("GPU hours" in e for e in errors)

    def test_max_trials_exceeded(self):
        steps = [
            ResearchStep(step_id=f"s{i}", action=ResearchAction.TRAIN)
            for i in range(25)
        ]
        plan = ResearchPlan(steps=steps, reasoning="too many")
        validator = PlanValidator(max_gpu_hours=100, max_trials=20)
        errors = validator.validate(plan)
        assert any("Trial count" in e for e in errors)

    def test_invalid_dependency(self):
        plan = ResearchPlan(
            steps=[
                ResearchStep(step_id="s1", action=ResearchAction.HPO, depends_on=["nonexistent"]),
            ],
            reasoning="bad dep",
        )
        validator = PlanValidator(max_gpu_hours=100, max_trials=50)
        errors = validator.validate(plan)
        assert any("not found" in e for e in errors)

    def test_cycle_detection(self):
        plan = ResearchPlan(
            steps=[
                ResearchStep(step_id="s1", action=ResearchAction.HPO, depends_on=["s2"]),
                ResearchStep(step_id="s2", action=ResearchAction.EVAL, depends_on=["s1"]),
            ],
            reasoning="cycle",
        )
        validator = PlanValidator(max_gpu_hours=100, max_trials=50)
        errors = validator.validate(plan)
        assert any("cycle" in e.lower() for e in errors)


class TestPlanToDAG:
    """Test plan→ExecutionPlan compiler."""

    def test_compiles_correctly(self):
        plan = ResearchPlan(
            steps=[
                ResearchStep(step_id="s1", action=ResearchAction.ABLATION, params={"target": "rank"}),
                ResearchStep(step_id="s2", action=ResearchAction.HPO, depends_on=["s1"]),
                ResearchStep(step_id="s3", action=ResearchAction.EVAL, depends_on=["s2"]),
            ],
            reasoning="test",
        )
        exec_plan = plan_to_execution_plan(plan)
        assert len(exec_plan.steps) == 3
        assert exec_plan.steps[0].skill_name == "lora_trainer"  # ablation → lora_trainer
        assert exec_plan.steps[1].skill_name == "hpo_skill"
        assert exec_plan.steps[2].skill_name == "eval_skill"
        assert exec_plan.steps[0].params.get("protocol") == "ablation"


class TestMetaPlannerSkill:
    """Test MetaPlannerSkill (no LLM, rule-based fallback)."""

    @pytest.mark.asyncio
    async def test_default_plan_large_budget(self, large_spec):
        skill = MetaPlannerSkill(llm=None)
        result = await skill.execute(
            MetaPlannerInput(experiment=large_spec),
            state=MagicMock(),
        )
        assert result.data["valid"] is True
        assert len(result.data["plan"]["steps"]) >= 3  # ablation + hpo + ensemble + eval

    @pytest.mark.asyncio
    async def test_default_plan_small_budget(self, small_spec):
        skill = MetaPlannerSkill(llm=None)
        result = await skill.execute(
            MetaPlannerInput(experiment=small_spec),
            state=MagicMock(),
        )
        assert result.data["valid"] is True
        assert len(result.data["plan"]["steps"]) >= 2

    @pytest.mark.asyncio
    async def test_plan_respects_budget(self, default_spec):
        skill = MetaPlannerSkill(llm=None)
        result = await skill.execute(
            MetaPlannerInput(experiment=default_spec),
            state=MagicMock(),
        )
        plan = result.data["plan"]
        total_hours = sum(s["estimated_gpu_hours"] for s in plan["steps"])
        assert total_hours <= default_spec.budget_gpu_hours


# ═══════════════════════════════════════════
# A.2: Method Selector Tests
# ═══════════════════════════════════════════

class TestExtractSignals:
    """Test DatasetSignals extraction from profile."""

    def test_basic_extraction(self):
        profile = {
            "total_samples": 5000,
            "avg_tokens": 128.0,
            "duplicates_found": 50,
            "noise_flags": ["short_outputs"],
            "has_instruction": True,
            "has_input": True,
        }
        signals = extract_signals(profile)
        assert signals.size == 5000
        assert signals.avg_prompt_length == 128.0
        assert signals.duplicates_ratio == pytest.approx(0.01)
        assert signals.noise_level == pytest.approx(0.15)
        assert signals.has_multi_turn is True

    def test_empty_profile(self):
        signals = extract_signals({})
        assert signals.size == 0
        assert signals.noise_level == 0.0


class TestRuleBasedSelection:
    """Test rule-based method selection."""

    def test_small_dataset_random(self, default_spec):
        signals = DatasetSignals(size=500)
        sel = select_by_rules(signals, default_spec)
        assert sel is not None
        assert sel.hpo_strategy == "random"

    def test_high_imbalance_curriculum(self, default_spec):
        signals = DatasetSignals(size=5000, class_imbalance=0.8)
        sel = select_by_rules(signals, default_spec)
        assert sel is not None
        assert sel.protocol == "curriculum"

    def test_tight_budget_halving(self, small_spec):
        signals = DatasetSignals(size=5000)
        sel = select_by_rules(signals, small_spec)
        assert sel is not None
        assert sel.hpo_strategy == "halving"

    def test_large_budget_ablation(self, large_spec):
        signals = DatasetSignals(size=10000)
        sel = select_by_rules(signals, large_spec)
        assert sel is not None
        assert sel.protocol == "ablation"

    def test_noisy_data_bandit(self):
        # Use medium budget so large-budget rule (ablation) doesn't trigger first
        medium_spec = ExperimentSpec(
            goal="noisy test",
            dataset_path="data/noisy.jsonl",
            budget_gpu_hours=15,
            max_trials=9,
        )
        signals = DatasetSignals(size=5000, noise_level=0.5)
        sel = select_by_rules(signals, medium_spec)
        assert sel is not None
        assert sel.hpo_strategy == "bandit"


class TestMethodSelectorSkill:
    """Test MethodSelectorSkill end-to-end."""

    @pytest.mark.asyncio
    async def test_rule_based_selection(self, default_spec):
        skill = MethodSelectorSkill(llm=None)
        result = await skill.execute(
            MethodSelectorInput(
                experiment=default_spec,
                dataset_profile={"total_samples": 800, "avg_tokens": 100},
            ),
            state=MagicMock(),
        )
        data = result.data
        assert data["method"] in ("random", "bayesian", "bandit", "halving")
        assert data["confidence"] > 0

    @pytest.mark.asyncio
    async def test_fallback_when_no_rules_match(self, default_spec):
        """Even with empty signals and no LLM, should return something."""
        skill = MethodSelectorSkill(llm=None)
        result = await skill.execute(
            MethodSelectorInput(experiment=default_spec),
            state=MagicMock(),
        )
        assert result.data["method"] is not None


# ═══════════════════════════════════════════
# A.3: Loop Controller Tests
# ═══════════════════════════════════════════

class TestLoopControllerBudget:
    """Test budget monitoring."""

    def test_budget_exhausted(self):
        state = make_state(
            trials=[make_trial(i, 1.0, gpu_hours=5) for i in range(8)],
            budget_total=40.0,
        )
        controller = LoopController()
        decision = controller.should_continue(state)
        assert not decision.should_continue
        assert decision.signal == "budget"

    def test_trials_exhausted(self):
        state = make_state(
            trials=[make_trial(i, 1.0, gpu_hours=0.1) for i in range(20)],
            budget_total=100.0,
            max_trials=20,
        )
        controller = LoopController()
        decision = controller.should_continue(state)
        assert not decision.should_continue


class TestLoopControllerPlateau:
    """Test plateau detection."""

    def test_flat_losses_trigger_plateau(self):
        # All losses ~= 1.0 → slope ≈ 0
        trials = [make_trial(i, 1.0 + 0.00001 * (i % 2)) for i in range(10)]
        state = make_state(trials=trials, budget_total=100.0, max_trials=50)
        controller = LoopController(plateau_window=5, plateau_epsilon=1e-3)
        decision = controller.should_continue(state)
        assert not decision.should_continue
        assert decision.signal == "plateau"

    def test_improving_losses_no_plateau(self):
        # Steadily decreasing loss
        trials = [make_trial(i, 2.0 - i * 0.1) for i in range(10)]
        state = make_state(trials=trials, budget_total=100.0, max_trials=50)
        controller = LoopController(plateau_window=5, plateau_epsilon=1e-4)
        decision = controller.should_continue(state)
        assert decision.should_continue


class TestLoopControllerPatience:
    """Test patience exhaustion."""

    def test_patience_stop(self):
        state = make_state(
            trials=[make_trial(i, 1.5 + i * 0.01) for i in range(8)],
            budget_total=100.0,
            max_trials=50,
            patience=6,
        )
        controller = LoopController(max_patience=5)
        decision = controller.should_continue(state)
        assert not decision.should_continue
        assert decision.signal == "patience"


class TestLoopControllerConvergence:
    """Test convergence detection via CV."""

    def test_converged_losses(self):
        # Very similar losses → low CV
        trials = [make_trial(i, 1.000 + 0.0001 * i) for i in range(10)]
        state = make_state(trials=trials, budget_total=100.0, max_trials=50)
        controller = LoopController(
            convergence_cv_threshold=0.01,
            convergence_min_trials=5,
            plateau_epsilon=1e-6,  # disable plateau for this test
        )
        decision = controller.should_continue(state)
        # Either plateau or variance should trigger
        assert not decision.should_continue


class TestLoopControllerMinTrials:
    """Test minimum trial protection."""

    def test_always_runs_minimum(self):
        state = make_state(
            trials=[make_trial(0, 5.0)],  # Just 1 terrible trial
            budget_total=100.0,
            max_trials=50,
            patience=10,
        )
        controller = LoopController(min_trials_before_stop=3)
        decision = controller.should_continue(state)
        assert decision.should_continue


class TestLoopMetrics:
    """Test metrics computation."""

    def test_metrics_computation(self):
        trials = [make_trial(i, 2.0 - i * 0.1) for i in range(5)]
        state = make_state(trials=trials, budget_total=40.0, max_trials=20)
        controller = LoopController()
        metrics = controller.get_loop_metrics(state)
        assert metrics.metric_slope < 0  # Improving
        assert metrics.trials_completed == 5
        assert metrics.budget_utilization > 0
