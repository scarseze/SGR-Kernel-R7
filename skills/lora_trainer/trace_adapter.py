"""
LoRA Experiment Trace Adapter.

Bridges LoRA trainer events to SGR Kernel's TraceManager.
Each experiment event maps to a StepTrace with LoRA-specific metadata.
"""
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field

from core.trace import StepTrace, RequestTrace, TraceManager, AttemptTrace
from skills.lora_trainer.experiment_spec import TrialConfig, TrialStatus

logger = logging.getLogger("lora_trainer.trace")


class LoRATraceEvent(BaseModel):
    """Base event model for LoRA traces."""
    event_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trial_id: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


class TrialStartedEvent(LoRATraceEvent):
    event_type: str = "trial_started"
    config: dict = Field(default_factory=dict)


class TrialCompletedEvent(LoRATraceEvent):
    event_type: str = "trial_completed"
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None
    gpu_hours: float = 0.0
    adapter_path: Optional[str] = None


class TrialFailedEvent(LoRATraceEvent):
    event_type: str = "trial_failed"
    error: str = ""
    partial_metrics: dict = Field(default_factory=dict)


class HPODecisionEvent(LoRATraceEvent):
    event_type: str = "hpo_decision"
    strategy: str = ""
    suggested_config: dict = Field(default_factory=dict)


class EarlyStopEvent(LoRATraceEvent):
    event_type: str = "early_stop"
    reason: str = ""
    best_trial_id: Optional[int] = None
    best_loss: Optional[float] = None


class BudgetAlertEvent(LoRATraceEvent):
    event_type: str = "budget_alert"
    remaining_hours: float = 0.0
    remaining_trials: int = 0
    alert_level: str = "warning"  # warning | critical


class ExperimentTracer:
    """
    Tracks LoRA experiment events and integrates with TraceManager.
    
    Usage:
        tracer = ExperimentTracer(experiment_id="exp_001")
        tracer.trial_started(trial_id=1, config=config.model_dump())
        tracer.trial_completed(trial_id=1, eval_loss=0.42, gpu_hours=0.5)
        tracer.save()  # Saves to TraceManager
    """

    def __init__(self, experiment_id: str = "default", trace_manager: Optional[TraceManager] = None):
        self.experiment_id = experiment_id
        self.trace_manager = trace_manager or TraceManager(trace_dir="traces/lora")
        self.events: list[LoRATraceEvent] = []
        self.start_time = time.time()

        # Build a RequestTrace for the entire experiment
        self._trace = RequestTrace(
            user_request=f"LoRA Experiment: {experiment_id}",
        )

    def trial_started(self, trial_id: int, config: dict) -> None:
        """Record trial start."""
        event = TrialStartedEvent(trial_id=trial_id, config=config)
        self.events.append(event)
        logger.info(f"[{self.experiment_id}] Trial {trial_id} started")

        step = StepTrace(
            step_id=f"trial_{trial_id}_start",
            skill_name="training_skill",
            input_params=config,
            status="running",
        )
        step.start_time = time.time()
        self._trace.steps.append(step)

    def trial_completed(self, trial_id: int, eval_loss: float = 0.0,
                        eval_perplexity: float = 0.0, gpu_hours: float = 0.0,
                        adapter_path: str = "") -> None:
        """Record trial completion."""
        event = TrialCompletedEvent(
            trial_id=trial_id,
            eval_loss=eval_loss,
            eval_perplexity=eval_perplexity,
            gpu_hours=gpu_hours,
            adapter_path=adapter_path,
        )
        self.events.append(event)
        logger.info(
            f"[{self.experiment_id}] Trial {trial_id} completed: "
            f"loss={eval_loss:.4f} gpu_hours={gpu_hours:.2f}"
        )

        # Update corresponding step trace
        for step in self._trace.steps:
            if step.step_id == f"trial_{trial_id}_start":
                step.status = "completed"
                step.duration = time.time() - step.start_time
                step.output_data = f"loss={eval_loss:.4f}, ppl={eval_perplexity:.2f}"
                break

    def trial_failed(self, trial_id: int, error: str, partial_metrics: dict = None) -> None:
        """Record trial failure."""
        event = TrialFailedEvent(
            trial_id=trial_id,
            error=error,
            partial_metrics=partial_metrics or {},
        )
        self.events.append(event)
        logger.warning(f"[{self.experiment_id}] Trial {trial_id} failed: {error}")

        for step in self._trace.steps:
            if step.step_id == f"trial_{trial_id}_start":
                step.status = "error"
                step.error = error
                step.duration = time.time() - step.start_time
                break

    def hpo_decision(self, trial_id: int, strategy: str, suggested_config: dict) -> None:
        """Record HPO decision."""
        event = HPODecisionEvent(
            trial_id=trial_id,
            strategy=strategy,
            suggested_config=suggested_config,
        )
        self.events.append(event)

        step = StepTrace(
            step_id=f"hpo_{trial_id}",
            skill_name="hpo_skill",
            input_params={"strategy": strategy},
            output_data=str(suggested_config),
            status="completed",
        )
        self._trace.steps.append(step)

    def early_stop(self, reason: str, best_trial_id: int = None, best_loss: float = None) -> None:
        """Record early stopping."""
        event = EarlyStopEvent(
            reason=reason,
            best_trial_id=best_trial_id,
            best_loss=best_loss,
        )
        self.events.append(event)
        logger.info(f"[{self.experiment_id}] Early stop: {reason}")

    def budget_alert(self, remaining_hours: float, remaining_trials: int) -> None:
        """Record budget alert."""
        level = "critical" if remaining_hours < 0.5 or remaining_trials <= 1 else "warning"
        event = BudgetAlertEvent(
            remaining_hours=remaining_hours,
            remaining_trials=remaining_trials,
            alert_level=level,
        )
        self.events.append(event)
        if level == "critical":
            logger.warning(f"[{self.experiment_id}] ⚠️ Budget critical: {remaining_hours:.1f}h, {remaining_trials} trials left")

    def save(self) -> None:
        """Save experiment trace to disk."""
        self._trace.total_duration = time.time() - self.start_time
        self._trace.status = "completed"
        self.trace_manager.save_trace(self._trace)
        logger.info(f"[{self.experiment_id}] Trace saved ({len(self.events)} events)")

    @property
    def summary(self) -> dict:
        """Summary of all events for reporting."""
        return {
            "experiment_id": self.experiment_id,
            "total_events": len(self.events),
            "trials_started": sum(1 for e in self.events if e.event_type == "trial_started"),
            "trials_completed": sum(1 for e in self.events if e.event_type == "trial_completed"),
            "trials_failed": sum(1 for e in self.events if e.event_type == "trial_failed"),
            "hpo_decisions": sum(1 for e in self.events if e.event_type == "hpo_decision"),
            "early_stops": sum(1 for e in self.events if e.event_type == "early_stop"),
            "budget_alerts": sum(1 for e in self.events if e.event_type == "budget_alert"),
            "duration_sec": round(time.time() - self.start_time, 2),
        }
