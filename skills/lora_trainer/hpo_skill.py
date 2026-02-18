"""
HPOSkill — Hyperparameter Optimization для LoRA training.

Контракт:
- Вход: HPOInput (trial_history, search_space, strategy, budget)
- Выход: StepResult(data=TrialConfig)
- deterministic=False (bayesian), requires_gpu=False, cost_class=CHEAP
"""
import random
import math
from typing import Type
from pydantic import BaseModel

from skills.base import BaseSkill
from skills.lora_trainer.schema import HPOInput
from skills.lora_trainer.experiment_spec import TrialConfig, SearchSpace, PEFTMethod
from core.types import SkillMetadata, Capability, RiskLevel, CostClass
from core.result import StepResult


class HPOSkill(BaseSkill):

    @property
    def name(self) -> str:
        return "hpo_skill"

    @property
    def description(self) -> str:
        return "Suggests next trial config using HPO strategies: random, bayesian, bandit."

    @property
    def input_schema(self) -> Type[HPOInput]:
        return HPOInput

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=self.name,
            version="1.0.0",
            description=self.description,
            capabilities=[Capability.REASONING],
            risk_level=RiskLevel.LOW,
            cost_class=CostClass.CHEAP,
            deterministic=False,
            timeout_sec=30.0,
        )

    async def execute(self, params: HPOInput, state) -> StepResult:
        """Suggest next trial configuration."""
        strategy = params.strategy
        history = params.trial_history
        space = params.search_space
        trial_id = len(history) + 1

        if strategy == "random":
            config = self._random_search(trial_id, space)
        elif strategy == "bayesian":
            config = self._bayesian_search(trial_id, space, history)
        elif strategy == "bandit":
            config = self._bandit_search(trial_id, space, history)
        elif strategy == "halving":
            config = self._successive_halving(trial_id, space, history)
        else:
            config = self._random_search(trial_id, space)

        # Check for pruning recommendations
        prune_list = self._suggest_prune(history)

        return StepResult(
            data={
                "suggested_config": config.model_dump(),
                "strategy": strategy,
                "prune_trials": prune_list,
            },
            output_text=(
                f"🔬 HPO ({strategy}): Trial {trial_id} | "
                f"rank={config.lora_rank} lr={config.learning_rate} "
                f"modules={config.target_modules}"
            ),
            metadata={"trial_id": trial_id, "strategy": strategy},
        )

    def _random_search(self, trial_id: int, space: dict) -> TrialConfig:
        """Pure random sampling from search space."""
        ranks = space.get("lora_rank", [8, 16, 32, 64])
        lr_range = space.get("learning_rate", [1e-5, 5e-5, 1e-4, 2e-4, 5e-4])
        dropouts = space.get("lora_dropout", [0.0, 0.05, 0.1])

        # Target modules presets
        module_presets = space.get("target_modules_presets", {
            "attention_only": ["q_proj", "v_proj"],
            "hybrid": ["q_proj", "v_proj", "gate_proj", "down_proj"],
        })
        preset_name = random.choice(list(module_presets.keys()))

        return TrialConfig(
            trial_id=trial_id,
            lora_rank=random.choice(ranks),
            lora_alpha=random.choice(ranks) * 2,  # alpha = 2 * rank
            lora_dropout=random.choice(dropouts),
            learning_rate=random.choice(lr_range),
            target_modules=module_presets[preset_name],
            seed=random.randint(1, 99999),
        )

    def _bayesian_search(self, trial_id: int, space: dict, history: list) -> TrialConfig:
        """
        Simplified Bayesian-like search:
        - If insufficient history (<3 trials): use random
        - Otherwise: perturb best config within narrowed ranges
        """
        if len(history) < 3:
            return self._random_search(trial_id, space)

        # Find best trial
        completed = [t for t in history if t.get("status") == "done" and t.get("eval_loss") is not None]
        if not completed:
            return self._random_search(trial_id, space)

        best = min(completed, key=lambda t: t["eval_loss"])
        best_config = best.get("config", {})

        # Perturb best config
        ranks = space.get("lora_rank", [8, 16, 32, 64])
        lr_range = space.get("learning_rate", [1e-5, 5e-5, 1e-4, 2e-4, 5e-4])

        best_rank = best_config.get("lora_rank", 16)
        best_lr = best_config.get("learning_rate", 2e-4)

        # Choose nearby values
        rank_idx = min(range(len(ranks)), key=lambda i: abs(ranks[i] - best_rank))
        new_rank_idx = max(0, min(len(ranks) - 1, rank_idx + random.choice([-1, 0, 1])))

        lr_idx = min(range(len(lr_range)), key=lambda i: abs(lr_range[i] - best_lr))
        new_lr_idx = max(0, min(len(lr_range) - 1, lr_idx + random.choice([-1, 0, 1])))

        return TrialConfig(
            trial_id=trial_id,
            lora_rank=ranks[new_rank_idx],
            lora_alpha=ranks[new_rank_idx] * 2,
            learning_rate=lr_range[new_lr_idx],
            lora_dropout=random.choice(space.get("lora_dropout", [0.0, 0.05, 0.1])),
            target_modules=best_config.get("target_modules", ["q_proj", "v_proj"]),
            seed=random.randint(1, 99999),
        )

    def _bandit_search(self, trial_id: int, space: dict, history: list) -> TrialConfig:
        """
        Multi-armed bandit: explore vs exploit.
        Epsilon-greedy with epsilon = max(0.1, 1 - len(history)/20)
        """
        epsilon = max(0.1, 1.0 - len(history) / 20)
        if random.random() < epsilon:
            return self._random_search(trial_id, space)
        else:
            return self._bayesian_search(trial_id, space, history)

    def _successive_halving(self, trial_id: int, space: dict, history: list) -> TrialConfig:
        """
        Successive halving: start broad, narrow based on performance.
        For MVP, equivalent to bayesian with tighter perturbation.
        """
        return self._bayesian_search(trial_id, space, history)

    def _suggest_prune(self, history: list) -> list[int]:
        """Suggest trials to prune (stop early) based on performance."""
        if len(history) < 5:
            return []

        completed = [t for t in history if t.get("eval_loss") is not None]
        if not completed:
            return []

        # Find median loss
        losses = sorted(t["eval_loss"] for t in completed)
        median_loss = losses[len(losses) // 2]

        # Prune trials more than 2x worse than median (if running)
        prune_candidates = []
        for t in history:
            if t.get("status") == "running":
                partial_loss = t.get("eval_loss", t.get("final_loss"))
                if partial_loss and partial_loss > median_loss * 2:
                    prune_candidates.append(t.get("trial_id", 0))

        return prune_candidates
