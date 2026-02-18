"""
Multi-Adapter Ensemble Protocol.

Trains N adapters with different configs and combines them:
- Weighted merge: linear combination of adapter weights
- Router: select adapter per input based on routing logic
"""
from typing import Optional
from pydantic import BaseModel, Field

from skills.lora_trainer.experiment_spec import TrialConfig


class EnsembleMember(BaseModel):
    """Single adapter in the ensemble."""
    trial_id: int
    config: TrialConfig
    adapter_path: str = ""
    eval_loss: Optional[float] = None
    weight: float = 1.0  # Merge weight


class EnsembleResult(BaseModel):
    members: list[EnsembleMember] = Field(default_factory=list)
    merged_path: Optional[str] = None
    ensemble_loss: Optional[float] = None
    strategy: str = "weighted_merge"


class EnsembleProtocol:
    """
    Multi-adapter ensemble strategies.
    
    Strategy 1 — Weighted Merge:
        Train N adapters → merge weights proportional to performance
    
    Strategy 2 — Diversity Ensemble:
        Train N adapters with diverse configs → combine
    
    Usage:
        protocol = EnsembleProtocol()
        configs = protocol.generate_diverse_configs(
            base_config=TrialConfig(trial_id=0),
            n_members=5,
        )
        # Train each, then merge
    """

    def generate_diverse_configs(self, base_config: TrialConfig,
                                 n_members: int = 5,
                                 start_trial_id: int = 1) -> list[TrialConfig]:
        """
        Generate N diverse adapter configs for ensemble.
        Varies target_modules, rank, and learning rate.
        """
        import random

        module_presets = [
            ["q_proj", "v_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["q_proj", "v_proj", "gate_proj", "down_proj"],
            ["gate_proj", "up_proj", "down_proj"],
            ["q_proj", "v_proj", "up_proj"],
        ]

        ranks = [8, 16, 32, 64]
        lrs = [5e-5, 1e-4, 2e-4, 3e-4]

        configs = []
        for i in range(n_members):
            config_dict = base_config.model_dump()
            config_dict["trial_id"] = start_trial_id + i
            config_dict["target_modules"] = module_presets[i % len(module_presets)]
            config_dict["lora_rank"] = ranks[i % len(ranks)]
            config_dict["lora_alpha"] = config_dict["lora_rank"] * 2
            config_dict["learning_rate"] = lrs[i % len(lrs)]
            config_dict["seed"] = 42 + i * 7
            configs.append(TrialConfig(**config_dict))

        return configs

    def compute_merge_weights(self, members: list[EnsembleMember],
                              strategy: str = "inverse_loss") -> list[EnsembleMember]:
        """
        Compute merge weights for ensemble members.
        
        Strategies:
        - "equal": uniform weights
        - "inverse_loss": weight ∝ 1/loss (better adapters get higher weight)
        - "softmax": softmax over negative losses
        """
        if not members:
            return members

        if strategy == "equal":
            for m in members:
                m.weight = 1.0 / len(members)
        elif strategy == "inverse_loss":
            losses = [m.eval_loss for m in members if m.eval_loss and m.eval_loss > 0]
            if losses:
                inv_losses = [1.0 / l for l in losses]
                total = sum(inv_losses)
                for i, m in enumerate(members):
                    if m.eval_loss and m.eval_loss > 0:
                        m.weight = (1.0 / m.eval_loss) / total
                    else:
                        m.weight = 0.0
        elif strategy == "softmax":
            import math
            losses = [-(m.eval_loss or 0) for m in members]
            max_l = max(losses) if losses else 0
            exp_losses = [math.exp(l - max_l) for l in losses]
            total = sum(exp_losses)
            for i, m in enumerate(members):
                m.weight = exp_losses[i] / total if total > 0 else 1.0 / len(members)

        return members
