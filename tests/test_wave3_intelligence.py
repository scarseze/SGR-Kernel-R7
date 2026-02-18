"""
Tests for Wave 3: Multi-Agent (C.1), Online Learning (D.1), Paper Writer (D.2)
"""
import pytest
import math

# ── C.1: Multi-Agent ──
from skills.lora_trainer.agents import (
    AgentRole,
    ActionType,
    AgentMessage,
    TrainerAgent,
    AnalystAgent,
    ProtocolAgent,
    StrategyAgent,
    AgentCoordinator,
)

# ── D.1: Online Learning ──
from skills.lora_trainer.online_learning import (
    OnlineLearner,
    OnlineState,
    ParamPrior,
    DriftSignal,
)

# ── D.2: Paper Writer ──
from skills.lora_trainer.paper_writer import (
    PaperWriter,
    ExperimentReport,
)


# ═══════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════

def make_trial_dict(trial_id: int, loss: float, rank: int = 16, lr: float = 2e-4, gpu: float = 0.5, status="done"):
    return {
        "trial_id": trial_id,
        "config": {
            "lora_rank": rank,
            "learning_rate": lr,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "eval_loss": loss,
        "gpu_hours": gpu,
        "status": status,
    }


def make_state_dict(n_trials=10, budget_util=0.5):
    history = [make_trial_dict(i, 2.0 - i * 0.05, rank=[8, 16, 32, 64][i % 4]) for i in range(n_trials)]
    return {
        "trial_history": history,
        "budget_utilization": budget_util,
    }


# ═══════════════════════════════════════════
# C.1: Multi-Agent Tests
# ═══════════════════════════════════════════

class TestAgentMessage:
    def test_message_creation(self):
        msg = AgentMessage(
            sender=AgentRole.TRAINER,
            receiver=AgentRole.ANALYST,
            action=ActionType.REPORT,
            payload={"trial_id": 1},
        )
        assert msg.sender == "trainer"
        assert msg.action == "report"


class TestTrainerAgent:
    def test_reports_latest_trial(self):
        agent = TrainerAgent()
        state = {"trial_history": [make_trial_dict(1, 1.5)]}
        messages = agent.process(state)
        assert len(messages) >= 1
        assert messages[0].action == ActionType.REPORT

    def test_alerts_on_failure(self):
        agent = TrainerAgent()
        state = {"trial_history": [make_trial_dict(1, None, status="failed")]}
        state["trial_history"][0]["eval_loss"] = None
        messages = agent.process(state)
        alerts = [m for m in messages if m.action == ActionType.ALERT]
        assert len(alerts) >= 1

    def test_empty_history(self):
        agent = TrainerAgent()
        assert agent.process({"trial_history": []}) == []


class TestAnalystAgent:
    def test_detects_plateau(self):
        agent = AnalystAgent(plateau_window=5, plateau_eps=0.01)
        # Flat losses
        history = [make_trial_dict(i, 1.5 + 0.001 * (i % 2)) for i in range(10)]
        state = {"trial_history": history}
        messages = agent.process(state)
        alerts = [m for m in messages if m.action == ActionType.ALERT]
        assert any("plateau" in m.payload.get("signal", "") for m in alerts)

    def test_detects_divergence(self):
        agent = AnalystAgent()
        history = [make_trial_dict(i, 1.0 + i * 0.5) for i in range(5)]
        state = {"trial_history": history}
        messages = agent.process(state)
        alerts = [m for m in messages if m.action == ActionType.ALERT]
        assert any("divergence" in m.payload.get("signal", "") for m in alerts)

    def test_reports_improvement_rate(self):
        agent = AnalystAgent()
        history = [make_trial_dict(i, 2.0 - i * 0.1) for i in range(5)]
        state = {"trial_history": history}
        messages = agent.process(state)
        reports = [m for m in messages if m.action == ActionType.REPORT]
        assert len(reports) >= 1
        assert "improvement_rate" in reports[0].payload


class TestProtocolAgent:
    def test_switches_to_bayesian(self):
        agent = ProtocolAgent(explore_threshold=5)
        # Send analyst report with enough trials
        report_msg = AgentMessage(
            sender=AgentRole.ANALYST,
            receiver=AgentRole.PROTOCOL,
            action=ActionType.REPORT,
            payload={"improvement_rate": 0.5, "total_trials": 6},
        )
        agent.receive(report_msg)
        messages = agent.process({"trial_history": [None] * 6})
        strategy_switches = [m for m in messages if m.action == ActionType.SWITCH_STRATEGY]
        assert len(strategy_switches) >= 1
        assert strategy_switches[0].payload["new_strategy"] == "bayesian"


class TestStrategyAgent:
    def test_recommends_mutation_on_plateau(self):
        agent = StrategyAgent()
        alert = AgentMessage(
            sender=AgentRole.ANALYST,
            receiver=AgentRole.STRATEGY,
            action=ActionType.ALERT,
            payload={"signal": "plateau"},
        )
        agent.receive(alert)
        messages = agent.process({"budget_utilization": 0.5})
        mutations = [m for m in messages if m.action == ActionType.MUTATE_SPACE]
        assert len(mutations) >= 1

    def test_budget_warning(self):
        agent = StrategyAgent()
        messages = agent.process({"budget_utilization": 0.9})
        alerts = [m for m in messages if m.action == ActionType.ALERT]
        assert any("budget" in m.payload.get("signal", "") for m in alerts)


class TestAgentCoordinator:
    def test_full_tick(self):
        coord = AgentCoordinator()
        state = make_state_dict(n_trials=10)
        actions = coord.tick(state)
        # Should produce some messages
        assert isinstance(actions, list)

    def test_recommendation(self):
        coord = AgentCoordinator()
        state = make_state_dict(n_trials=5)
        rec = coord.get_recommendation(state)
        assert "should_continue" in rec
        assert "alerts" in rec


# ═══════════════════════════════════════════
# D.1: Online Learning Tests
# ═══════════════════════════════════════════

class TestOnlineLearner:
    def test_welford_mean(self):
        learner = OnlineLearner()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            learner.update({"eval_loss": v, "config": {"lora_rank": 16}})
        assert abs(learner.state.running_mean - 3.0) < 0.01

    def test_prior_updates(self):
        learner = OnlineLearner()
        for i in range(10):
            learner.update({
                "eval_loss": 2.0 - i * 0.1,
                "config": {"lora_rank": 16, "learning_rate": 2e-4},
            })
        assert "lora_rank" in learner.state.priors
        assert learner.state.priors["lora_rank"].n_observations == 10

    def test_sampling_weights(self):
        learner = OnlineLearner()
        # Train with rank=16 being best
        for i in range(10):
            learner.update({
                "eval_loss": 1.0 if i % 2 == 0 else 2.0,
                "config": {"lora_rank": 16 if i % 2 == 0 else 64},
            })
        weights = learner.get_sampling_weights("lora_rank", [8, 16, 32, 64])
        assert len(weights) == 4
        assert abs(sum(weights) - 1.0) < 0.01  # Normalized

    def test_uniform_weights_insufficient_data(self):
        learner = OnlineLearner()
        weights = learner.get_sampling_weights("lora_rank", [8, 16, 32])
        assert all(abs(w - 1 / 3) < 0.01 for w in weights)

    def test_drift_detection(self):
        learner = OnlineLearner(drift_window=5, drift_threshold=0.2)
        # Phase 1: stable around 2.0
        for i in range(10):
            learner.update({"eval_loss": 2.0 + 0.01 * i, "config": {}})
        # Phase 2: sudden shift to 3.0
        for i in range(10):
            learner.update({"eval_loss": 3.0 + 0.01 * i, "config": {}})
        drift = learner.get_drift_status()
        assert drift is not None
        assert drift.detected is True

    def test_state_serialization(self):
        learner = OnlineLearner()
        for i in range(5):
            learner.update({"eval_loss": 1.0 + i, "config": {"rank": i}})
        state = learner.get_state()
        assert isinstance(state, OnlineState)
        # Restore
        new_learner = OnlineLearner()
        new_learner.load_state(state)
        assert new_learner.state.n_updates == state.n_updates

    def test_cooldown_blocks_retrain(self):
        learner = OnlineLearner(cooldown_window=5, max_retrain_frequency=3)
        # Initially can retrain (last_retrain_update = -999)
        assert learner.can_retrain() is True
        # Retrain, then immediately try again
        learner.record_retrain()
        # Now only 0 updates since retrain, need 5
        assert learner.can_retrain() is False
        # Feed updates to pass cooldown
        for i in range(5):
            learner.update({"eval_loss": 1.5, "config": {}})
        assert learner.can_retrain() is True

    def test_max_retrain_frequency_blocks(self):
        learner = OnlineLearner(cooldown_window=1, max_retrain_frequency=2)
        # First retrain OK
        assert learner.can_retrain() is True
        learner.record_retrain()
        learner.update({"eval_loss": 1.5, "config": {}})  # pass cooldown
        # Second retrain OK
        assert learner.can_retrain() is True
        learner.record_retrain()
        learner.update({"eval_loss": 1.5, "config": {}})  # pass cooldown
        # Third retrain BLOCKED — max frequency reached
        assert learner.can_retrain() is False
        assert learner.state.retrain_count == 2


# ═══════════════════════════════════════════
# D.2: Paper Writer Tests
# ═══════════════════════════════════════════

class TestPaperWriter:
    def test_generates_report(self):
        writer = PaperWriter()
        spec = {
            "goal": "test LoRA",
            "base_model": "llama-7b",
            "dataset_path": "data/train.jsonl",
            "budget_gpu_hours": 10,
            "max_trials": 20,
            "stop_metric": "eval_loss",
        }
        history = [make_trial_dict(i, 2.0 - i * 0.1) for i in range(8)]
        report = writer.generate(spec, history)
        assert isinstance(report, ExperimentReport)
        assert len(report.sections) >= 5
        assert "# " in report.markdown

    def test_empty_history(self):
        writer = PaperWriter()
        report = writer.generate({"goal": "test"}, [])
        assert isinstance(report, ExperimentReport)

    def test_report_has_tables(self):
        writer = PaperWriter()
        spec = {"goal": "test", "base_model": "model", "budget_gpu_hours": 10}
        history = [make_trial_dict(i, 2.0 - i * 0.1) for i in range(5)]
        report = writer.generate(spec, history)
        results_section = next((s for s in report.sections if s.title == "Results"), None)
        assert results_section is not None
        assert len(results_section.tables) >= 1

    def test_report_with_appendix(self):
        writer = PaperWriter()
        spec = {"goal": "test"}
        history = [make_trial_dict(i, 1.5) for i in range(3)]
        report = writer.generate(spec, history, extra_context={
            "mutations": ["shrunk lora_rank"],
            "drift_signals": ["drift at trial 5"],
        })
        appendix = next((s for s in report.sections if s.title == "Appendix"), None)
        assert appendix is not None
        assert "shrunk" in appendix.content
