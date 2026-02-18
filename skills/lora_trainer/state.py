"""
Experiment State — mutable state tracked by the Kernel across the training loop.

Follows the Kernel State Schema from the mentor's design:
- experiment_goal, dataset_profile, search_space
- trial_history, budget_used, best_trial, stop_flags
"""
from typing import Any, Optional
from pydantic import BaseModel, Field

from skills.lora_trainer.experiment_spec import TrialConfig, TrialStatus


class TrialRecord(BaseModel):
    """Immutable record of a completed (or failed) trial."""
    trial_id: int
    config: TrialConfig
    status: TrialStatus = TrialStatus.CREATED

    # Metrics (populated after eval)
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None
    prompt_score: Optional[float] = None
    delta_vs_base: Optional[float] = None

    # Cost tracking
    gpu_hours: float = 0.0
    total_tokens: int = 0
    training_steps: int = 0

    # Artifacts
    adapter_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    log_path: Optional[str] = None

    # Metadata
    error_message: Optional[str] = None
    seed: int = 42


class DatasetProfile(BaseModel):
    """Output of dataset_skill.analyze — static profile of the dataset."""
    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0

    avg_tokens: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0
    p95_tokens: int = 0

    format_detected: str = "unknown"
    columns: list[str] = Field(default_factory=list)
    has_instruction: bool = False
    has_input: bool = False
    has_output: bool = False

    # Quality flags
    duplicates_found: int = 0
    outliers_found: int = 0
    noise_flags: list[str] = Field(default_factory=list)


class ExperimentState(BaseModel):
    """
    Mutable state for the entire experiment.
    Updated by the Kernel after each skill execution.
    """
    experiment_goal: str = ""
    dataset_profile: Optional[DatasetProfile] = None
    search_space: dict[str, Any] = Field(default_factory=dict)

    # Trial tracking
    trial_history: list[TrialRecord] = Field(default_factory=list)
    current_trial_id: int = 0

    # Budget
    budget_total_gpu_hours: float = 10.0
    budget_used_gpu_hours: float = 0.0
    budget_max_trials: int = 20

    # Best result
    best_trial: Optional[TrialRecord] = None
    best_metric_value: Optional[float] = None
    best_metric_name: str = "eval_loss"

    # Stop control
    stop_flags: list[str] = Field(default_factory=list)
    trials_without_improvement: int = 0

    @property
    def budget_remaining_gpu_hours(self) -> float:
        return max(0.0, self.budget_total_gpu_hours - self.budget_used_gpu_hours)

    @property
    def trials_remaining(self) -> int:
        return max(0, self.budget_max_trials - len(self.trial_history))

    @property
    def should_stop(self) -> bool:
        if self.stop_flags:
            return True
        if self.trials_remaining <= 0:
            return True
        if self.budget_remaining_gpu_hours <= 0:
            return True
        return False

    def register_trial(self, record: TrialRecord) -> None:
        """Register a completed trial and update best tracking."""
        self.trial_history.append(record)

        if record.status == TrialStatus.DONE and record.eval_loss is not None:
            metric_val = getattr(record, self.best_metric_name, record.eval_loss)
            if metric_val is not None:
                if self.best_metric_value is None or metric_val < self.best_metric_value:
                    self.best_trial = record
                    self.best_metric_value = metric_val
                    self.trials_without_improvement = 0
                else:
                    self.trials_without_improvement += 1

        self.budget_used_gpu_hours += record.gpu_hours
