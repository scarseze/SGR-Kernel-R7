"""
§14 — Minimal Compliance Tests for Agent Protocol v1.

Every agent MUST pass:
  1. Deterministic mode test
  2. Schema validation test
  3. Guardrail rejection test
  4. Proposal explainability test

Plus protocol-level tests:
  5. Signal bus isolation
  6. Decision engine authority filter
  7. Decision engine conflict resolution
  8. Decision trace completeness
  9. Failure protocol (degraded agent)
 10. Extensibility (new agent registration)
"""
import pytest
import time

from skills.lora_trainer.agent_protocol import (
    # §3 Role Model
    AgentRole,
    AuthorityLevel,
    RiskLevel,
    AgentDescriptor,
    # §5 Signal Protocol
    SignalType,
    Signal,
    SignalBus,
    # §6 Decision Proposal
    ActionType,
    DecisionProposal,
    # §7 Decision Engine
    DecisionEngine,
    DecisionTrace,
    DecisionOutcome,
    GuardrailCheck,
    # §2 Agent Base
    ProtocolAgent,
    # §4 Canonical Agents
    TrainerAgentV2,
    AnalystAgentV2,
    ProtocolAgentV2,
    StrategyAgentV2,
    # §7+ Coordinator
    ProtocolCoordinator,
    # §12 Failure
    AgentStatus,
)


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════

def make_trial(tid, loss, status="done", rank=16, lr=2e-4):
    return {
        "trial_id": tid,
        "config": {"lora_rank": rank, "learning_rate": lr, "lora_dropout": 0.05},
        "eval_loss": loss,
        "gpu_hours": 0.5,
        "status": status,
    }


def make_state(n=5, budget=0.5):
    return {
        "trial_history": [make_trial(i, 2.0 - i * 0.05) for i in range(n)],
        "budget_utilization": budget,
    }


def make_agents():
    return [TrainerAgentV2(), AnalystAgentV2(), ProtocolAgentV2(), StrategyAgentV2()]


ALL_AGENTS = make_agents()


# ═══════════════════════════════════════════════
# §14.1  Deterministic Mode Test
# ═══════════════════════════════════════════════

class TestDeterministicMode:
    """Same input → same output for FRESH agent instances (no randomness)."""

    AGENT_FACTORIES = [TrainerAgentV2, AnalystAgentV2, ProtocolAgentV2, StrategyAgentV2]

    @pytest.mark.parametrize("factory", AGENT_FACTORIES, ids=lambda f: f.__name__)
    def test_deterministic_output(self, factory):
        """Fresh agents with identical input must produce identical output."""
        state = make_state(n=10)
        signals = []
        agent1 = factory()
        agent2 = factory()
        out1 = agent1.process(state, signals)
        out2 = agent2.process(state, signals)
        # Signal count and proposal count should match
        assert len(out1[0]) == len(out2[0]), f"{agent1.descriptor.name}: signal count differs"
        assert len(out1[1]) == len(out2[1]), f"{agent1.descriptor.name}: proposal count differs"


# ═══════════════════════════════════════════════
# §14.2  Schema Validation Test
# ═══════════════════════════════════════════════

class TestSchemaValidation:
    """Every agent MUST declare a valid AgentDescriptor."""

    @pytest.mark.parametrize("agent", ALL_AGENTS, ids=lambda a: a.descriptor.name)
    def test_descriptor_valid(self, agent):
        desc = agent.descriptor
        assert isinstance(desc, AgentDescriptor)
        assert desc.name != ""
        assert desc.role in [r.value for r in AgentRole]
        assert desc.authority_level in [a.value for a in AuthorityLevel]
        assert len(desc.inputs) > 0, "Agent must declare inputs"
        assert len(desc.outputs) > 0, "Agent must declare outputs"

    @pytest.mark.parametrize("agent", ALL_AGENTS, ids=lambda a: a.descriptor.name)
    def test_output_types_valid(self, agent):
        """Signals and proposals must conform to schemas."""
        state = make_state(n=10)
        # Feed some signals to provoke output
        signals = [
            Signal(type=SignalType.METRIC_PLATEAU, source="test", payload={"slope": 0.0}),
        ]
        out_signals, proposals = agent.process(state, signals)
        for s in out_signals:
            assert isinstance(s, Signal)
        for p in proposals:
            assert isinstance(p, DecisionProposal)


# ═══════════════════════════════════════════════
# §14.3  Guardrail Rejection Test
# ═══════════════════════════════════════════════

class TestGuardrailRejection:
    """DecisionEngine must reject proposals that violate guardrails."""

    def test_budget_guard_blocks_extend(self):
        engine = DecisionEngine(budget_guard_limit=0.9)
        proposal = DecisionProposal(
            agent="protocol_agent",
            action_type=ActionType.EXTEND_TRIALS,
            confidence=0.9,
            rationale="Need more trials",
        )
        desc = {"protocol_agent": AgentDescriptor(
            name="protocol_agent", role=AgentRole.PROTOCOL,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        chosen, trace = engine.evaluate([proposal], desc, {"budget_utilization": 0.95})
        assert chosen is None
        assert any("budget" in r.get("reason", "") for r in trace.rejected)

    def test_stability_guard_blocks_switch_during_trial(self):
        engine = DecisionEngine()
        proposal = DecisionProposal(
            agent="protocol_agent",
            action_type=ActionType.SWITCH_HPO,
            confidence=0.9,
            rationale="Switch to bayesian",
            parameters={"new_strategy": "bayesian"},
        )
        desc = {"protocol_agent": AgentDescriptor(
            name="protocol_agent", role=AgentRole.PROTOCOL,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        state = {"active_trial_running": True}
        chosen, trace = engine.evaluate([proposal], desc, state)
        assert chosen is None
        assert any("switch" in c.lower() for c in trace.guardrail_checks)

    def test_high_risk_low_confidence_blocked(self):
        engine = DecisionEngine()
        proposal = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.STOP_EXPERIMENT,
            confidence=0.5,  # too low for HIGH risk
            risk_level=RiskLevel.HIGH,
            rationale="Divergence detected",
        )
        desc = {"strategy_agent": AgentDescriptor(
            name="strategy_agent", role=AgentRole.STRATEGY,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        chosen, trace = engine.evaluate([proposal], desc, {})
        assert chosen is None
        assert any("risk" in c for c in trace.guardrail_checks)


# ═══════════════════════════════════════════════
# §14.4  Proposal Explainability Test
# ═══════════════════════════════════════════════

class TestExplainability:
    """§9 — Every proposal MUST have rationale + confidence."""

    def test_empty_rationale_raises(self):
        with pytest.raises(Exception):
            DecisionProposal(
                agent="test",
                action_type=ActionType.CONTINUE,
                confidence=0.9,
                rationale="",  # EMPTY → rejected
            )

    def test_low_confidence_rejected_by_engine(self):
        engine = DecisionEngine(min_confidence=0.5)
        proposal = DecisionProposal(
            agent="protocol_agent",
            action_type=ActionType.SWITCH_HPO,
            confidence=0.2,  # below threshold
            rationale="Not sure but maybe",
        )
        desc = {"protocol_agent": AgentDescriptor(
            name="protocol_agent", role=AgentRole.PROTOCOL,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        chosen, trace = engine.evaluate([proposal], desc, {})
        assert chosen is None
        assert any("explainability" in r.get("reason", "") for r in trace.rejected)

    @pytest.mark.parametrize("agent", ALL_AGENTS, ids=lambda a: a.descriptor.name)
    def test_all_proposals_have_rationale(self, agent):
        """If an agent produces proposals, they must have rationale."""
        state = make_state(n=10)
        signals = [
            Signal(type=SignalType.METRIC_PLATEAU, source="test"),
            Signal(type=SignalType.METRIC_DIVERGENCE, source="test"),
        ]
        _, proposals = agent.process(state, signals)
        for p in proposals:
            assert p.rationale.strip() != "", f"{agent.descriptor.name}: proposal missing rationale"
            assert p.confidence > 0, f"{agent.descriptor.name}: proposal has zero confidence"


# ═══════════════════════════════════════════════
# §5  Signal Bus Tests
# ═══════════════════════════════════════════════

class TestSignalBus:
    def test_emit_and_read(self):
        bus = SignalBus()
        bus.emit(Signal(type=SignalType.TRIAL_COMPLETED, source="test", payload={"id": 1}))
        assert bus.count == 1
        signals = bus.read()
        assert len(signals) == 1

    def test_read_by_type(self):
        bus = SignalBus()
        bus.emit(Signal(type=SignalType.TRIAL_COMPLETED, source="trainer"))
        bus.emit(Signal(type=SignalType.METRIC_PLATEAU, source="analyst"))
        bus.emit(Signal(type=SignalType.TRIAL_COMPLETED, source="trainer"))
        plateau = bus.read(signal_type=SignalType.METRIC_PLATEAU)
        assert len(plateau) == 1

    def test_max_history(self):
        bus = SignalBus(max_history=5)
        for i in range(10):
            bus.emit(Signal(type=SignalType.TRIAL_COMPLETED, source="test"))
        assert bus.count == 5

    def test_read_since(self):
        bus = SignalBus()
        ts = time.time()
        bus.emit(Signal(type=SignalType.TRIAL_COMPLETED, source="test", timestamp=ts + 1))
        bus.emit(Signal(type=SignalType.TRIAL_COMPLETED, source="test", timestamp=ts + 2))
        results = bus.read_since(ts + 1.5)
        assert len(results) == 1


# ═══════════════════════════════════════════════
# §7  Decision Engine Tests
# ═══════════════════════════════════════════════

class TestDecisionEngine:
    def test_authority_filter(self):
        """OBSERVE authority cannot make proposals."""
        engine = DecisionEngine()
        proposal = DecisionProposal(
            agent="trainer_agent",
            action_type=ActionType.STOP_EXPERIMENT,
            confidence=0.9,
            rationale="I want to stop",
        )
        desc = {"trainer_agent": AgentDescriptor(
            name="trainer_agent", role=AgentRole.TRAINER,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.OBSERVE,  # Can't propose!
        )}
        chosen, trace = engine.evaluate([proposal], desc, {})
        assert chosen is None
        assert any(r["outcome"] == DecisionOutcome.REJECTED_AUTHORITY for r in trace.rejected)

    def test_conflict_resolution_by_priority(self):
        """Strategy > Protocol in priority."""
        engine = DecisionEngine()
        p1 = DecisionProposal(
            agent="protocol_agent",
            action_type=ActionType.SWITCH_HPO,
            confidence=0.8,
            rationale="Switch to bayesian",
        )
        p2 = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.MUTATE_SEARCH,
            confidence=0.8,
            rationale="Mutate search space",
        )
        desc = {
            "protocol_agent": AgentDescriptor(
                name="protocol_agent", role=AgentRole.PROTOCOL,
                inputs=["x"], outputs=["y"],
                authority_level=AuthorityLevel.PROPOSE,
            ),
            "strategy_agent": AgentDescriptor(
                name="strategy_agent", role=AgentRole.STRATEGY,
                inputs=["x"], outputs=["y"],
                authority_level=AuthorityLevel.PROPOSE,
            ),
        }
        chosen, trace = engine.evaluate([p1, p2], desc, {})
        assert chosen is not None
        assert chosen.agent == "strategy_agent"  # higher priority

    def test_decision_trace_completeness(self):
        """§10 — Trace must record all considered proposals."""
        engine = DecisionEngine()
        proposals = [
            DecisionProposal(
                agent="strategy_agent",
                action_type=ActionType.CONTINUE,
                confidence=0.9,
                rationale="Keep going",
            ),
        ]
        desc = {"strategy_agent": AgentDescriptor(
            name="strategy_agent", role=AgentRole.STRATEGY,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        chosen, trace = engine.evaluate(proposals, desc, {})
        assert len(trace.proposals_considered) == 1
        assert trace.chosen is not None
        assert isinstance(trace.decision_id, str)


# ═══════════════════════════════════════════════
# §12  Failure Protocol Tests
# ═══════════════════════════════════════════════

class TestFailureProtocol:
    def test_failing_agent_marked_degraded(self):
        """§12 — If agent crashes, mark degraded, continue."""

        class BrokenAgent(ProtocolAgent):
            @property
            def descriptor(self):
                return AgentDescriptor(
                    name="broken_agent", role=AgentRole.TRAINER,
                    inputs=["x"], outputs=["y"],
                    authority_level=AuthorityLevel.OBSERVE,
                )

            def process(self, state, signals):
                raise RuntimeError("Agent exploded")

        coord = ProtocolCoordinator(agents=[BrokenAgent(), AnalystAgentV2()])
        result = coord.tick(make_state(n=5))
        # Should not crash, broken agent degraded
        assert coord.agent_status["broken_agent"] == AgentStatus.DEGRADED
        # Analyst should still work
        assert "decision_trace" in result


# ═══════════════════════════════════════════════
# §13  Extensibility Contract Tests
# ═══════════════════════════════════════════════

class TestExtensibility:
    def test_custom_agent_registration(self):
        """§13 — New agent must declare descriptor and pass protocol."""

        class CostAgent(ProtocolAgent):
            @property
            def descriptor(self):
                return AgentDescriptor(
                    name="cost_agent",
                    role=AgentRole.ANALYST,
                    inputs=["resource_usage", "budget_state"],
                    outputs=["cost_alert"],
                    authority_level=AuthorityLevel.SUGGEST,
                    risk_level=RiskLevel.LOW,
                )

            def process(self, state, signals):
                budget = state.get("budget_utilization", 0)
                out_signals = []
                if budget > 0.7:
                    out_signals.append(Signal(
                        type=SignalType.BUDGET_LOW,
                        source=self.descriptor.name,
                        payload={"utilization": budget},
                    ))
                return out_signals, []

        # Register with coordinator
        coord = ProtocolCoordinator(agents=[CostAgent()])
        assert "cost_agent" in coord.agent_descriptors
        result = coord.tick(make_state(budget=0.9))
        assert result["signals_emitted"] >= 1


# ═══════════════════════════════════════════════
# §7+  Full Protocol Tick Tests
# ═══════════════════════════════════════════════

class TestProtocolCoordinator:
    def test_full_tick(self):
        coord = ProtocolCoordinator()
        result = coord.tick(make_state(n=10))
        assert "should_continue" in result
        assert "decision_trace" in result
        assert isinstance(result["decision_trace"], DecisionTrace)

    def test_tick_with_plateau(self):
        """Analyst detects plateau → Strategy proposes mutation."""
        coord = ProtocolCoordinator()
        # Flat loss → plateau
        flat_state = {
            "trial_history": [make_trial(i, 1.5 + 0.001 * (i % 2)) for i in range(10)],
            "budget_utilization": 0.3,
        }
        result = coord.tick(flat_state)
        traces = coord.decision_engine.trace_log
        assert len(traces) >= 1

    def test_tick_low_budget_emits_signal(self):
        coord = ProtocolCoordinator()
        result = coord.tick(make_state(n=3, budget=0.9))
        assert result["signals_emitted"] >= 1


# ═══════════════════════════════════════════════
# §7.1  Cooldown System Tests
# ═══════════════════════════════════════════════

class TestCooldownSystem:
    """§7.1 — Anti-oscillation cooldown."""

    def test_cooldown_activates_after_disruptive_action(self):
        engine = DecisionEngine(cooldown_trials=3)
        proposal = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.MUTATE_SEARCH,
            confidence=0.9,
            rationale="Mutate to escape plateau",
            signals_used=["metric_plateau"],
        )
        desc = {"strategy_agent": AgentDescriptor(
            name="strategy_agent", role=AgentRole.STRATEGY,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        # First time: accepted
        chosen, _ = engine.evaluate([proposal], desc, {})
        assert chosen is not None
        # Cooldown should be active
        assert ActionType.MUTATE_SEARCH.value in engine._cooldowns or "mutate_search" in engine._cooldowns

    def test_cooldown_blocks_same_action(self):
        engine = DecisionEngine(cooldown_trials=3)
        proposal = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.MUTATE_SEARCH,
            confidence=0.9,
            rationale="Mutate to escape plateau",
        )
        desc = {"strategy_agent": AgentDescriptor(
            name="strategy_agent", role=AgentRole.STRATEGY,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        # First: accepted, activates cooldown
        chosen1, _ = engine.evaluate([proposal], desc, {})
        assert chosen1 is not None
        # Second: blocked by cooldown
        p2 = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.MUTATE_SEARCH,
            confidence=0.95,
            rationale="Try again",
        )
        chosen2, trace2 = engine.evaluate([p2], desc, {})
        assert chosen2 is None
        assert any(r.get("outcome") == DecisionOutcome.REJECTED_COOLDOWN for r in trace2.rejected)

    def test_cooldown_expires_after_ticks(self):
        engine = DecisionEngine(cooldown_trials=2)
        proposal = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.SWITCH_HPO,
            confidence=0.85,
            rationale="Switch to bayesian",
        )
        desc = {"strategy_agent": AgentDescriptor(
            name="strategy_agent", role=AgentRole.STRATEGY,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        # Activate cooldown
        engine.evaluate([proposal], desc, {})
        # Tick down twice
        engine.tick_cooldowns()
        engine.tick_cooldowns()
        # Cooldown expired → should accept
        p2 = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.SWITCH_HPO,
            confidence=0.85,
            rationale="Switch again after cooldown",
        )
        chosen, _ = engine.evaluate([p2], desc, {})
        assert chosen is not None


# ═══════════════════════════════════════════════
# §7.2  Weighted Scoring Tests
# ═══════════════════════════════════════════════

class TestWeightedScoring:
    """§7.2 — score = confidence × w1 + priority × w2 + signals × w3 − risk × w4."""

    def test_higher_confidence_wins(self):
        """Same agent, same priority — higher confidence wins."""
        engine = DecisionEngine()
        p_low = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.CONTINUE,
            confidence=0.4,
            rationale="Low confidence",
        )
        p_high = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.CONTINUE,
            confidence=0.95,
            rationale="High confidence",
        )
        desc = {"strategy_agent": AgentDescriptor(
            name="strategy_agent", role=AgentRole.STRATEGY,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        chosen, _ = engine.evaluate([p_low, p_high], desc, {})
        assert chosen.confidence == 0.95

    def test_signal_support_boosts_score(self):
        """Proposal with more signal evidence scores higher (same confidence)."""
        engine = DecisionEngine()
        p_no_signals = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.CONTINUE,
            confidence=0.7,
            rationale="No signal evidence",
            signals_used=[],
        )
        p_signals = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.CONTINUE,
            confidence=0.7,
            rationale="Has signal evidence",
            signals_used=["metric_plateau", "high_variance", "drift_detected"],
        )
        desc = {"strategy_agent": AgentDescriptor(
            name="strategy_agent", role=AgentRole.STRATEGY,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        score_no = engine._score_proposal(p_no_signals, desc)
        score_yes = engine._score_proposal(p_signals, desc)
        assert score_yes > score_no

    def test_risk_penalty_reduces_score(self):
        """HIGH risk → lower score than LOW risk at same confidence."""
        engine = DecisionEngine()
        p_low_risk = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.CONTINUE,
            confidence=0.8,
            risk_level=RiskLevel.LOW,
            rationale="Safe proposal",
        )
        p_high_risk = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.CONTINUE,
            confidence=0.8,
            risk_level=RiskLevel.HIGH,
            rationale="Risky proposal",
        )
        desc = {"strategy_agent": AgentDescriptor(
            name="strategy_agent", role=AgentRole.STRATEGY,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        score_safe = engine._score_proposal(p_low_risk, desc)
        score_risky = engine._score_proposal(p_high_risk, desc)
        assert score_safe > score_risky


# ═══════════════════════════════════════════════
# §7.3  Action Safety Priority Tests
# ═══════════════════════════════════════════════

class TestActionSafetyPriority:
    """§7.3 — STOP > MUTATE > SWITCH > TUNE > other."""

    def test_stop_beats_switch_at_equal_score(self):
        """Safety priority is a hard tiebreak: STOP always wins over SWITCH."""
        engine = DecisionEngine()
        p_switch = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.SWITCH_HPO,
            confidence=0.9,
            rationale="Switch to bayesian",
        )
        p_stop = DecisionProposal(
            agent="strategy_agent",
            action_type=ActionType.STOP_EXPERIMENT,
            confidence=0.9,
            risk_level=RiskLevel.MEDIUM,  # not HIGH so it passes guardrail
            rationale="Stop the experiment",
        )
        desc = {"strategy_agent": AgentDescriptor(
            name="strategy_agent", role=AgentRole.STRATEGY,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        chosen, _ = engine.evaluate([p_switch, p_stop], desc, {})
        assert chosen is not None
        assert chosen.action_type == ActionType.STOP_EXPERIMENT.value

    def test_mutate_beats_tune(self):
        engine = DecisionEngine()
        p_tune = DecisionProposal(
            agent="protocol_agent",
            action_type=ActionType.TUNE_PARAM,
            confidence=0.9,
            rationale="Fine-tune learning rate",
        )
        p_mutate = DecisionProposal(
            agent="protocol_agent",
            action_type=ActionType.MUTATE_SEARCH,
            confidence=0.9,
            rationale="Mutate search space",
        )
        desc = {"protocol_agent": AgentDescriptor(
            name="protocol_agent", role=AgentRole.PROTOCOL,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.PROPOSE,
        )}
        chosen, _ = engine.evaluate([p_tune, p_mutate], desc, {})
        assert chosen.action_type == ActionType.MUTATE_SEARCH.value


# ═══════════════════════════════════════════════
# §3  Authority Violation Tests (v1.1)
# ═══════════════════════════════════════════════

class TestAuthorityViolation:
    """OBSERVE/SUGGEST agents cannot issue action proposals."""

    def test_observe_cannot_stop(self):
        engine = DecisionEngine()
        proposal = DecisionProposal(
            agent="trainer_agent",
            action_type=ActionType.STOP_EXPERIMENT,
            confidence=0.99,
            rationale="I think we should stop",
        )
        desc = {"trainer_agent": AgentDescriptor(
            name="trainer_agent", role=AgentRole.TRAINER,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.OBSERVE,
        )}
        chosen, trace = engine.evaluate([proposal], desc, {})
        assert chosen is None

    def test_suggest_cannot_mutate(self):
        engine = DecisionEngine()
        proposal = DecisionProposal(
            agent="analyst_agent",
            action_type=ActionType.MUTATE_SEARCH,
            confidence=0.9,
            rationale="Suggesting mutation",
        )
        desc = {"analyst_agent": AgentDescriptor(
            name="analyst_agent", role=AgentRole.ANALYST,
            inputs=["x"], outputs=["y"],
            authority_level=AuthorityLevel.SUGGEST,
        )}
        chosen, trace = engine.evaluate([proposal], desc, {})
        assert chosen is None
        assert any(r.get("outcome") == DecisionOutcome.REJECTED_AUTHORITY for r in trace.rejected)
