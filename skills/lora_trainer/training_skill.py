"""
TrainingSkill — GPU Job Wrapper для LoRA/PEFT обучения.

Контракт:
- Вход: TrainingInput (trial_config, dataset_path, output_dir)
- Выход: StepResult(data=training_result, artifacts=[adapter_path])
- idempotent=True, job-based, retry-safe
- requires_gpu=True, cost_class=EXPENSIVE, timeout_sec=3600
"""
import os
import json
import time
from typing import Type
from pydantic import BaseModel

from skills.base import BaseSkill
from skills.lora_trainer.schema import TrainingInput
from skills.lora_trainer.experiment_spec import TrialConfig, PEFTMethod
from core.types import SkillMetadata, Capability, RiskLevel, CostClass, RetryPolicy
from core.result import StepResult


class TrainingJobSpec(BaseModel):
    """Job specification sent to GPU worker."""
    trial_id: int
    config: TrialConfig
    dataset_path: str
    output_dir: str
    resume_from: str | None = None
    resources: str = "1xGPU"


class TrainingResult(BaseModel):
    """Structured output of a training job."""
    trial_id: int
    adapter_path: str = ""
    checkpoint_path: str = ""
    final_loss: float = 0.0
    training_steps: int = 0
    gpu_hours: float = 0.0
    total_tokens: int = 0
    status: str = "completed"  # completed | failed | dry_run
    error: str | None = None


class TrainingSkill(BaseSkill):

    @property
    def name(self) -> str:
        return "training_skill"

    @property
    def description(self) -> str:
        return "Runs LoRA/PEFT training jobs. GPU execution wrapper with checkpoint support."

    @property
    def input_schema(self) -> Type[TrainingInput]:
        return TrainingInput

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=self.name,
            version="1.0.0",
            description=self.description,
            capabilities=[Capability.CODE],
            risk_level=RiskLevel.HIGH,
            cost_class=CostClass.EXPENSIVE,
            retry_policy=RetryPolicy.STANDARD,
            idempotent=True,
            requires_gpu=True,
            requires_filesystem=True,
            side_effects=True,
            timeout_sec=3600.0,
            max_concurrency=1,
        )

    async def execute(self, params: TrainingInput, state) -> StepResult:
        """Execute a LoRA training job."""
        config = params.trial_config
        output_dir = os.path.join(params.output_dir, f"trial_{config.trial_id}")

        # 1. Build job spec
        job_spec = TrainingJobSpec(
            trial_id=config.trial_id,
            config=config,
            dataset_path=params.dataset_path,
            output_dir=output_dir,
            resume_from=params.resume_from,
        )

        # 2. Dry run mode — validate config without training
        if params.dry_run:
            return self._dry_run(job_spec)

        # 3. Execute training
        try:
            result = await self._run_training(job_spec)
        except Exception as e:
            result = TrainingResult(
                trial_id=config.trial_id,
                status="failed",
                error=str(e),
            )

        artifacts = []
        if result.adapter_path:
            artifacts.append(result.adapter_path)
        if result.checkpoint_path:
            artifacts.append(result.checkpoint_path)

        return StepResult(
            data=result.model_dump(),
            artifacts=artifacts,
            output_text=(
                f"🏋️ Trial {config.trial_id}: {result.status} | "
                f"loss={result.final_loss:.4f} | steps={result.training_steps} | "
                f"GPU={result.gpu_hours:.2f}h"
                if result.status == "completed"
                else f"❌ Trial {config.trial_id}: {result.status} — {result.error}"
            ),
            metadata={"trial_id": config.trial_id, "gpu_hours": result.gpu_hours},
        )

    def _dry_run(self, job_spec: TrainingJobSpec) -> StepResult:
        """Validate config and estimate training cost without running."""
        config = job_spec.config

        # Estimate steps
        # Assume 10k samples, effective batch = batch * grad_accum
        assumed_samples = 10000
        effective_batch = config.batch_size * config.gradient_accumulation
        steps_per_epoch = assumed_samples // effective_batch
        total_steps = steps_per_epoch * config.num_epochs

        # Estimate GPU hours (~0.5s per step on A100 for 1-3B models)
        estimated_gpu_hours = total_steps * 0.5 / 3600

        result = TrainingResult(
            trial_id=config.trial_id,
            training_steps=total_steps,
            gpu_hours=estimated_gpu_hours,
            status="dry_run",
        )

        return StepResult(
            data=result.model_dump(),
            output_text=(
                f"🔍 Dry Run: Trial {config.trial_id} | "
                f"~{total_steps} steps | ~{estimated_gpu_hours:.2f} GPU hours | "
                f"method={config.method.value} rank={config.lora_rank}"
            ),
        )

    async def _run_training(self, job_spec: TrainingJobSpec) -> TrainingResult:
        """
        Execute actual LoRA training.
        
        MVP: Uses transformers + peft + trl directly.
        Production: Would dispatch to GPU worker queue.
        """
        config = job_spec.config
        start_time = time.time()

        try:
            # Late imports — only needed on GPU worker
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
            )
            from peft import LoraConfig, get_peft_model, TaskType
            from trl import SFTTrainer
            from datasets import load_dataset

            # 1. Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 2. Load model
            load_kwargs = {"device_map": "auto"}
            if config.method == PEFTMethod.QLORA and config.quantization_bits:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True if config.quantization_bits == 4 else False,
                    load_in_8bit=True if config.quantization_bits == 8 else False,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            model = AutoModelForCausalLM.from_pretrained(config.base_model, **load_kwargs)

            # 3. Apply LoRA
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)

            # 4. Load dataset
            if os.path.exists(job_spec.dataset_path):
                dataset = load_dataset("json", data_files=job_spec.dataset_path, split="train")
            else:
                dataset = load_dataset(job_spec.dataset_path, split="train")

            # 5. Training arguments
            os.makedirs(job_spec.output_dir, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=job_spec.output_dir,
                num_train_epochs=config.num_epochs,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation,
                learning_rate=config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                weight_decay=config.weight_decay,
                logging_steps=10,
                save_strategy="epoch",
                seed=config.seed,
                fp16=True,
                report_to="none",
                max_grad_norm=1.0,
            )

            # 6. Train
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
            )

            if job_spec.resume_from:
                trainer.train(resume_from_checkpoint=job_spec.resume_from)
            else:
                trainer.train()

            # 7. Save adapter
            adapter_path = os.path.join(job_spec.output_dir, "adapter")
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)

            # 8. Collect metrics
            elapsed = (time.time() - start_time) / 3600
            logs = trainer.state.log_history
            final_loss = logs[-1].get("loss", 0.0) if logs else 0.0

            return TrainingResult(
                trial_id=config.trial_id,
                adapter_path=adapter_path,
                checkpoint_path=job_spec.output_dir,
                final_loss=final_loss,
                training_steps=trainer.state.global_step,
                gpu_hours=round(elapsed, 4),
                total_tokens=trainer.state.global_step * config.batch_size * config.max_seq_length,
                status="completed",
            )

        except ImportError as e:
            return TrainingResult(
                trial_id=config.trial_id,
                status="failed",
                error=f"Missing dependency: {e}. Install: pip install transformers peft trl datasets",
            )
        except Exception as e:
            elapsed = (time.time() - start_time) / 3600
            return TrainingResult(
                trial_id=config.trial_id,
                gpu_hours=round(elapsed, 4),
                status="failed",
                error=str(e),
            )
