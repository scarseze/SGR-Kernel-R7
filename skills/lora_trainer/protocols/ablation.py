"""
Ablation Study Protocol.

Systematically varies one parameter while keeping all others fixed.
Answers: "What is the effect of parameter X on performance?"
"""
import copy
from typing import Any
from pydantic import BaseModel, Field

from skills.lora_trainer.experiment_spec import TrialConfig


class AblationResult(BaseModel):
    param_name: str
    baseline_loss: float | None = None
    results: list[dict] = Field(default_factory=list)
    # Each entry: {value: X, trial_id: N, eval_loss: Y, delta: Z}


class AblationProtocol:
    """
    Run N trials with one parameter varied, rest fixed.
    
    Usage:
        protocol = AblationProtocol()
        configs = protocol.generate_configs(
            base_config=TrialConfig(trial_id=0),
            vary_param="lora_rank",
            values=[4, 8, 16, 32, 64, 128],
        )
        # Execute each config through TrainingSkill + EvalSkill
    """

    def generate_configs(self, base_config: TrialConfig, vary_param: str,
                         values: list[Any], start_trial_id: int = 1) -> list[TrialConfig]:
        """
        Generate configs varying one parameter.
        
        Args:
            base_config: The baseline config (all defaults)
            vary_param: Parameter name to vary (e.g., "lora_rank", "learning_rate")
            values: List of values to test
            start_trial_id: Starting trial ID for generated configs
            
        Returns:
            List of TrialConfig objects, one per value
        """
        configs = []
        for i, val in enumerate(values):
            config_dict = base_config.model_dump()
            config_dict["trial_id"] = start_trial_id + i

            if vary_param in config_dict:
                config_dict[vary_param] = val
            else:
                raise ValueError(f"Parameter '{vary_param}' not found in TrialConfig")

            configs.append(TrialConfig(**config_dict))

        return configs

    def analyze_results(self, param_name: str, trials: list[dict]) -> AblationResult:
        """
        Analyze ablation results.
        
        Args:
            param_name: The parameter that was varied
            trials: List of dicts with keys: value, trial_id, eval_loss
        """
        if not trials:
            return AblationResult(param_name=param_name)

        baseline = trials[0].get("eval_loss")
        results = []
        for t in trials:
            delta = (t["eval_loss"] - baseline) if baseline and t.get("eval_loss") else None
            results.append({
                "value": t.get("value"),
                "trial_id": t.get("trial_id"),
                "eval_loss": t.get("eval_loss"),
                "delta": delta,
            })

        return AblationResult(
            param_name=param_name,
            baseline_loss=baseline,
            results=results,
        )

    def suggested_ablations(self) -> list[dict]:
        """Return commonly useful ablation studies."""
        return [
            {
                "param": "lora_rank",
                "values": [4, 8, 16, 32, 64, 128],
                "rationale": "Find minimal rank that preserves quality",
            },
            {
                "param": "learning_rate",
                "values": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
                "rationale": "Learning rate sensitivity check",
            },
            {
                "param": "lora_dropout",
                "values": [0.0, 0.01, 0.05, 0.1, 0.2],
                "rationale": "Regularization effect on generalization",
            },
            {
                "param": "num_epochs",
                "values": [1, 2, 3, 5, 10],
                "rationale": "Find optimal training duration vs overfitting",
            },
        ]
