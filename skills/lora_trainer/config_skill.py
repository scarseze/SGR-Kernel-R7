"""
LoRAConfigSkill — генерация конфигураций LoRA/PEFT и search space.

Контракт:
- Вход: LoRAConfigInput (base_model, task, dataset_profile, vram_limit)
- Выход: StepResult(data=TrialConfig)
- deterministic=True, requires_gpu=False, cost_class=CHEAP
"""
import math
from typing import Type
from pydantic import BaseModel

from skills.base import BaseSkill
from skills.lora_trainer.schema import LoRAConfigInput
from skills.lora_trainer.experiment_spec import TrialConfig, SearchSpace, PEFTMethod
from core.types import SkillMetadata, Capability, RiskLevel, CostClass
from core.result import StepResult


# Approximate model sizes in billions of parameters
MODEL_SIZES = {
    "unsloth/Llama-3.2-1B": 1.0,
    "unsloth/Llama-3.2-3B": 3.0,
    "meta-llama/Llama-3.1-8B": 8.0,
    "mistralai/Mistral-7B-v0.3": 7.0,
    "Qwen/Qwen2.5-7B": 7.0,
    "google/gemma-2-2b": 2.0,
}

# VRAM estimation: ~2 bytes per param for fp16, ~0.5 bytes for 4-bit
BYTES_PER_PARAM_FP16 = 2.0
BYTES_PER_PARAM_4BIT = 0.5
GB = 1024 ** 3


class LoRAConfigSkill(BaseSkill):

    @property
    def name(self) -> str:
        return "lora_config_skill"

    @property
    def description(self) -> str:
        return "Generates LoRA/PEFT configurations and search spaces. Validates memory fit."

    @property
    def input_schema(self) -> Type[LoRAConfigInput]:
        return LoRAConfigInput

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=self.name,
            version="1.0.0",
            description=self.description,
            capabilities=[Capability.REASONING],
            risk_level=RiskLevel.LOW,
            cost_class=CostClass.CHEAP,
            deterministic=True,
            timeout_sec=30.0,
        )

    async def execute(self, params: LoRAConfigInput, state) -> StepResult:
        """Generate a LoRA config based on model, task, and dataset profile."""
        method = PEFTMethod(params.method)
        profile = params.dataset_profile

        # 1. Estimate model size
        model_size_b = self._estimate_model_size(params.base_model)

        # 2. Validate memory fit
        fit_result = self._validate_memory_fit(
            model_size_b=model_size_b,
            method=method,
            vram_limit_gb=params.vram_limit_gb,
        )

        if not fit_result["fits"]:
            return StepResult(
                data=fit_result,
                output_text=f"❌ Model {params.base_model} ({model_size_b}B params) does not fit in {params.vram_limit_gb}GB VRAM. "
                            f"Estimated: {fit_result['estimated_vram_gb']:.1f}GB. Consider QLoRA.",
                metadata={"fits": False},
            )

        # 3. Select optimal parameters based on task/dataset
        target_modules = self._select_target_modules(params.task)
        seq_length = self._estimate_seq_length(profile)
        batch_size = self._estimate_batch_size(model_size_b, method, params.vram_limit_gb, seq_length)

        config = TrialConfig(
            trial_id=0,  # Will be set by HPO
            method=method,
            base_model=params.base_model,
            lora_rank=self._suggest_rank(model_size_b, params.task),
            lora_alpha=self._suggest_alpha(model_size_b),
            target_modules=target_modules,
            learning_rate=self._suggest_lr(method),
            num_epochs=3,
            batch_size=batch_size,
            gradient_accumulation=max(1, 16 // batch_size),
            max_seq_length=seq_length,
            quantization_bits=4 if method == PEFTMethod.QLORA else None,
        )

        # 4. Build search space
        search_space = SearchSpace(
            lora_rank=[8, 16, 32, 64] if model_size_b <= 3 else [4, 8, 16, 32],
            learning_rate=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
        )

        return StepResult(
            data={
                "trial_config": config.model_dump(),
                "search_space": search_space.model_dump(),
                "memory_estimate": fit_result,
            },
            output_text=(
                f"⚙️ Config: {method.value} | rank={config.lora_rank} | "
                f"lr={config.learning_rate} | batch={config.batch_size} | "
                f"VRAM≈{fit_result['estimated_vram_gb']:.1f}GB"
            ),
        )

    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in billions of parameters."""
        if model_name in MODEL_SIZES:
            return MODEL_SIZES[model_name]
        # Try to extract from name (e.g., "Llama-3.1-8B" -> 8.0)
        import re
        match = re.search(r"(\d+\.?\d*)[Bb]", model_name)
        if match:
            return float(match.group(1))
        return 7.0  # Default assumption

    def _validate_memory_fit(self, model_size_b: float, method: PEFTMethod, vram_limit_gb: float) -> dict:
        """Estimate VRAM usage and check if model fits."""
        params = model_size_b * 1e9

        if method == PEFTMethod.QLORA:
            model_vram = params * BYTES_PER_PARAM_4BIT / GB
        else:
            model_vram = params * BYTES_PER_PARAM_FP16 / GB

        # LoRA adapter + optimizer + gradients overhead (~30% of model for LoRA)
        overhead_ratio = 0.3 if method in (PEFTMethod.LORA, PEFTMethod.QLORA) else 0.2
        total_vram = model_vram * (1 + overhead_ratio)

        return {
            "fits": total_vram <= vram_limit_gb,
            "estimated_vram_gb": round(total_vram, 2),
            "model_vram_gb": round(model_vram, 2),
            "overhead_gb": round(total_vram - model_vram, 2),
            "method": method.value,
        }

    def _select_target_modules(self, task: str) -> list[str]:
        """Select LoRA target modules based on task type."""
        presets = {
            "instruction": ["q_proj", "v_proj", "o_proj"],
            "domain": ["q_proj", "v_proj", "gate_proj", "down_proj"],
            "chat": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "style": ["q_proj", "v_proj"],
        }
        return presets.get(task, ["q_proj", "v_proj"])

    def _suggest_rank(self, model_size_b: float, task: str) -> int:
        """Suggest LoRA rank based on model size and task complexity."""
        if model_size_b <= 1:
            base = 32
        elif model_size_b <= 3:
            base = 16
        else:
            base = 8
        # Domain adaptation needs higher rank
        if task == "domain":
            base = min(base * 2, 128)
        return base

    def _suggest_alpha(self, model_size_b: float) -> int:
        """Suggest LoRA alpha (typically 2x rank)."""
        rank = self._suggest_rank(model_size_b, "instruction")
        return rank * 2

    def _suggest_lr(self, method: PEFTMethod) -> float:
        """Suggest learning rate based on method."""
        return {
            PEFTMethod.LORA: 2e-4,
            PEFTMethod.QLORA: 1e-4,
            PEFTMethod.PREFIX: 3e-4,
            PEFTMethod.IA3: 5e-4,
        }.get(method, 2e-4)

    def _estimate_seq_length(self, profile: dict) -> int:
        """Estimate optimal sequence length from dataset profile."""
        p95 = profile.get("p95_tokens", 0)
        if p95 > 0:
            # Round up to nearest power of 2, cap at 2048
            return min(2048, 2 ** math.ceil(math.log2(max(64, p95))))
        return 512

    def _estimate_batch_size(self, model_size_b: float, method: PEFTMethod,
                             vram_gb: float, seq_length: int) -> int:
        """Estimate max batch size that fits in VRAM."""
        # Very rough heuristic: remaining VRAM after model / (seq_length * 2KB per sample)
        model_vram = model_size_b * (0.5 if method == PEFTMethod.QLORA else 2.0)
        free_vram = max(1, vram_gb - model_vram)
        kb_per_sample = seq_length * 2 / 1024  # ~2 bytes per token
        estimated = int(free_vram * 1024 * 1024 / (kb_per_sample * 1024))
        return max(1, min(estimated, 32))  # Clamp 1..32
