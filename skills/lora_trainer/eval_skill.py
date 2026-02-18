"""
EvalSkill — evaluation of LoRA adapters against base model.

Контракт:
- Вход: EvalInput (adapter_path, base_model, eval_dataset_path)
- Выход: StepResult(data=eval_metrics)
- deterministic=True, requires_gpu=True, cost_class=NORMAL
"""
import os
import math
import time
from typing import Type
from pydantic import BaseModel

from skills.base import BaseSkill
from skills.lora_trainer.schema import EvalInput
from core.types import SkillMetadata, Capability, RiskLevel, CostClass
from core.result import StepResult


class EvalMetrics(BaseModel):
    """Structured evaluation output."""
    eval_loss: float = 0.0
    eval_perplexity: float = 0.0
    prompt_score: float = 0.0
    delta_vs_base_loss: float = 0.0
    delta_vs_base_perplexity: float = 0.0
    win_rate_vs_base: float = 0.0
    samples_evaluated: int = 0
    gpu_hours: float = 0.0


class EvalSkill(BaseSkill):

    @property
    def name(self) -> str:
        return "eval_skill"

    @property
    def description(self) -> str:
        return "Evaluates LoRA adapters: loss, perplexity, prompt suite, base model comparison."

    @property
    def input_schema(self) -> Type[EvalInput]:
        return EvalInput

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=self.name,
            version="1.0.0",
            description=self.description,
            capabilities=[Capability.REASONING],
            risk_level=RiskLevel.LOW,
            cost_class=CostClass.NORMAL,
            deterministic=True,
            requires_gpu=True,
            timeout_sec=600.0,
        )

    async def execute(self, params: EvalInput, state) -> StepResult:
        """Run evaluation suite on a trained adapter."""
        start_time = time.time()

        try:
            metrics = await self._run_eval(params)
        except ImportError as e:
            # Fallback: mock eval for environments without GPU libs
            metrics = self._mock_eval(params)
            metrics_dict = metrics.model_dump()
            metrics_dict["_mock"] = True
            return StepResult(
                data=metrics_dict,
                output_text=f"📊 Mock Eval (no GPU libs): loss={metrics.eval_loss:.4f} ppl={metrics.eval_perplexity:.2f}",
                metadata={"mock": True, "error": str(e)},
            )

        elapsed = (time.time() - start_time) / 3600
        metrics.gpu_hours = round(elapsed, 4)

        return StepResult(
            data=metrics.model_dump(),
            output_text=(
                f"📊 Eval: loss={metrics.eval_loss:.4f} | ppl={metrics.eval_perplexity:.2f} | "
                f"Δloss={metrics.delta_vs_base_loss:+.4f} | "
                f"score={metrics.prompt_score:.2f}"
            ),
            metadata={"adapter_path": params.adapter_path, "gpu_hours": metrics.gpu_hours},
        )

    async def _run_eval(self, params: EvalInput) -> EvalMetrics:
        """Run actual evaluation using transformers."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(params.adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load eval dataset
        if os.path.exists(params.eval_dataset_path):
            eval_dataset = load_dataset("json", data_files=params.eval_dataset_path, split="train")
        else:
            eval_dataset = load_dataset(params.eval_dataset_path, split="validation")

        # 1. Eval adapted model
        base_model = AutoModelForCausalLM.from_pretrained(params.base_model, device_map="auto")
        adapted_model = PeftModel.from_pretrained(base_model, params.adapter_path)
        adapted_model.eval()

        adapted_loss = self._compute_loss(adapted_model, tokenizer, eval_dataset)

        # 2. Eval base model (for delta)
        base_loss = 0.0
        if params.compute_delta:
            # Unmerge adapter to get base performance
            adapted_model.unload()
            base_loss = self._compute_loss(base_model, tokenizer, eval_dataset)

        return EvalMetrics(
            eval_loss=adapted_loss,
            eval_perplexity=math.exp(min(adapted_loss, 20)),  # Cap to avoid overflow
            delta_vs_base_loss=adapted_loss - base_loss if params.compute_delta else 0.0,
            delta_vs_base_perplexity=(
                math.exp(min(adapted_loss, 20)) - math.exp(min(base_loss, 20))
                if params.compute_delta else 0.0
            ),
            samples_evaluated=len(eval_dataset),
        )

    def _compute_loss(self, model, tokenizer, dataset, max_samples: int = 500) -> float:
        """Compute average loss on dataset."""
        import torch

        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for i, sample in enumerate(dataset):
                if i >= max_samples:
                    break
                text = sample.get("text", sample.get("output", str(sample)))
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()

                outputs = model(**inputs)
                total_loss += outputs.loss.item()
                count += 1

        return total_loss / max(count, 1)

    def _mock_eval(self, params: EvalInput) -> EvalMetrics:
        """Generate mock metrics for testing without GPU."""
        import random
        random.seed(42)
        loss = round(random.uniform(0.5, 2.0), 4)
        return EvalMetrics(
            eval_loss=loss,
            eval_perplexity=round(math.exp(loss), 2),
            delta_vs_base_loss=round(random.uniform(-0.5, 0.1), 4),
            samples_evaluated=100,
        )
