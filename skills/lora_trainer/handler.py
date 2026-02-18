"""
LoRATrainerSkill — оркестратор LoRA/PEFT обучения.

Как LogicRLSkill вызывает CodeInterpreterSkill в цикле,
так LoRATrainerSkill вызывает sub-skills:
  dataset_skill → config_skill → hpo_skill → training_skill → eval_skill → artifact_skill

Kernel Loop:
1. Analyze dataset
2. Build search space
3. Suggest trial (HPO)
4. Run training
5. Evaluate
6. Store artifact
7. Update state → decide next trial or stop
"""
import os
from typing import Type

from skills.base import BaseSkill
from skills.lora_trainer.schema import (
    LoRATrainerInput,
    DatasetAnalyzeInput,
    LoRAConfigInput,
    TrainingInput,
    EvalInput,
    HPOInput,
    ArtifactStoreInput,
)
from skills.lora_trainer.experiment_spec import TrialConfig, ExperimentSpec
from skills.lora_trainer.state import ExperimentState, TrialRecord, DatasetProfile
from skills.lora_trainer.dataset_skill import DatasetSkill
from skills.lora_trainer.config_skill import LoRAConfigSkill
from skills.lora_trainer.training_skill import TrainingSkill
from skills.lora_trainer.eval_skill import EvalSkill
from skills.lora_trainer.hpo_skill import HPOSkill
from skills.lora_trainer.artifact_skill import ArtifactSkill
from skills.lora_trainer.merge_skill import MergeSkill, MergeInput, ExportFormat

from core.types import SkillMetadata, Capability, RiskLevel, CostClass, RetryPolicy
from core.result import StepResult
from core.state import AgentState


class LoRATrainerSkill(BaseSkill):
    """
    Top-level orchestrator skill for LoRA/PEFT research training.
    
    Architecture:
    - Agent Kernel calls this skill with ExperimentSpec
    - This skill internally orchestrates 6 sub-skills in a loop
    - Returns best TrialConfig + metrics as StepResult
    """

    def __init__(self, output_dir: str = "./lora_outputs", **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir

        # Initialize sub-skills
        self.dataset_skill = DatasetSkill()
        self.config_skill = LoRAConfigSkill()
        self.training_skill = TrainingSkill()
        self.eval_skill = EvalSkill()
        self.hpo_skill = HPOSkill()
        self.artifact_skill = ArtifactSkill(artifact_root=os.path.join(output_dir, "artifacts"))
        self.merge_skill = MergeSkill()

    @property
    def name(self) -> str:
        return "lora_trainer"

    @property
    def description(self) -> str:
        return (
            "LoRA/PEFT research trainer. Автоматический HPO-поиск оптимального адаптера. "
            "Orchestrates dataset analysis, config generation, training, evaluation, "
            "and artifact management in an automated loop."
        )

    @property
    def input_schema(self) -> Type[LoRATrainerInput]:
        return LoRATrainerInput

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=self.name,
            version="1.0.0",
            description=self.description,
            capabilities=[Capability.REASONING, Capability.CODE],
            risk_level=RiskLevel.HIGH,
            cost_class=CostClass.EXPENSIVE,
            retry_policy=RetryPolicy.STANDARD,
            requires_gpu=True,
            requires_filesystem=True,
            side_effects=True,
            idempotent=True,
            timeout_sec=7200.0,
            max_concurrency=1,
        )

    async def execute(self, params: LoRATrainerInput, state: AgentState) -> StepResult:
        """
        Main orchestration loop.
        
        Flow:
        1. dataset_skill.analyze → DatasetProfile
        2. config_skill.generate → initial TrialConfig + SearchSpace
        3. Loop:
            a. hpo_skill.suggest → TrialConfig
            b. training_skill.run → TrainingResult
            c. eval_skill.run → EvalMetrics
            d. artifact_skill.store → ArtifactRecord
            e. Update ExperimentState → decide stop/continue
        4. Return best trial
        """
        experiment = params.experiment
        dry_run = params.dry_run
        max_iterations = params.max_loop_iterations

        # Initialize experiment state
        exp_state = ExperimentState(
            experiment_goal=experiment.goal,
            budget_total_gpu_hours=experiment.budget_gpu_hours,
            budget_max_trials=experiment.max_trials,
            best_metric_name=experiment.stop_metric,
        )

        log = []  # Execution log for output

        # ── Step 1: Analyze Dataset ──
        log.append("📦 Step 1: Analyzing dataset...")
        ds_result = await self.dataset_skill.execute(
            DatasetAnalyzeInput(
                dataset_path=experiment.dataset_path,
                format=experiment.dataset_format,
            ),
            state,
        )
        profile_data = ds_result.data if isinstance(ds_result.data, dict) else {}
        exp_state.dataset_profile = DatasetProfile(**profile_data) if profile_data else DatasetProfile()
        log.append(f"  {ds_result.output_text}")

        # ── Step 2: Generate Initial Config + Search Space ──
        log.append("⚙️ Step 2: Building config and search space...")
        cfg_result = await self.config_skill.execute(
            LoRAConfigInput(
                base_model=experiment.base_model,
                task="instruction",
                dataset_profile=profile_data,
                method=experiment.methods[0].value if experiment.methods else "lora",
            ),
            state,
        )
        cfg_data = cfg_result.data if isinstance(cfg_result.data, dict) else {}
        exp_state.search_space = cfg_data.get("search_space", {})
        log.append(f"  {cfg_result.output_text}")

        # Check memory fit
        memory_est = cfg_data.get("memory_estimate", {})
        if not memory_est.get("fits", True):
            log.append(f"  ❌ Model does not fit in VRAM")
            return StepResult(
                data=exp_state.model_dump(),
                output_text="\n".join(log),
                status="failed",
            )

        # ── Step 3: Orchestration Loop ──
        log.append(f"\n🔁 Starting HPO loop (max {max_iterations} trials)...")

        for iteration in range(max_iterations):
            if exp_state.should_stop:
                stop_reason = ", ".join(exp_state.stop_flags) if exp_state.stop_flags else "budget/trials exhausted"
                log.append(f"\n🛑 Stopping: {stop_reason}")
                break

            # 3a. HPO: Suggest next trial
            hpo_result = await self.hpo_skill.execute(
                HPOInput(
                    trial_history=[t.model_dump() for t in exp_state.trial_history],
                    search_space=exp_state.search_space,
                    strategy="bayesian" if iteration >= 3 else "random",
                    budget_remaining=exp_state.budget_remaining_gpu_hours,
                    trials_remaining=exp_state.trials_remaining,
                ),
                state,
            )
            hpo_data = hpo_result.data if isinstance(hpo_result.data, dict) else {}
            suggested = hpo_data.get("suggested_config", {})

            # Build TrialConfig from suggestion, inheriting base model
            trial_config = TrialConfig(
                **{**suggested, "base_model": experiment.base_model}
            )
            exp_state.current_trial_id = trial_config.trial_id
            log.append(f"\n--- Trial {trial_config.trial_id} ---")
            log.append(f"  {hpo_result.output_text}")

            # 3b. Training
            train_result = await self.training_skill.execute(
                TrainingInput(
                    trial_config=trial_config,
                    dataset_path=experiment.dataset_path,
                    output_dir=self.output_dir,
                    dry_run=dry_run,
                ),
                state,
            )
            train_data = train_result.data if isinstance(train_result.data, dict) else {}
            log.append(f"  {train_result.output_text}")

            # Handle training failure
            if train_data.get("status") in ("failed",):
                record = TrialRecord(
                    trial_id=trial_config.trial_id,
                    config=trial_config,
                    status="failed",
                    error_message=train_data.get("error", "Unknown"),
                    gpu_hours=train_data.get("gpu_hours", 0),
                )
                exp_state.register_trial(record)
                continue

            # 3c. Evaluation (skip for dry_run)
            eval_metrics = {}
            if not dry_run and train_data.get("adapter_path"):
                eval_result = await self.eval_skill.execute(
                    EvalInput(
                        adapter_path=train_data["adapter_path"],
                        base_model=experiment.base_model,
                        eval_dataset_path=experiment.dataset_path,
                    ),
                    state,
                )
                eval_metrics = eval_result.data if isinstance(eval_result.data, dict) else {}
                log.append(f"  {eval_result.output_text}")

            # 3d. Store artifact
            if train_data.get("adapter_path"):
                await self.artifact_skill.execute(
                    ArtifactStoreInput(
                        adapter_path=train_data.get("adapter_path", ""),
                        trial_id=trial_config.trial_id,
                        metrics=eval_metrics,
                        tags=[experiment.methods[0].value if experiment.methods else "lora"],
                    ),
                    state,
                )

            # 3e. Register trial
            from skills.lora_trainer.experiment_spec import TrialStatus
            record = TrialRecord(
                trial_id=trial_config.trial_id,
                config=trial_config,
                status=TrialStatus.DONE if train_data.get("status") != "dry_run" else TrialStatus.DONE,
                eval_loss=eval_metrics.get("eval_loss"),
                eval_perplexity=eval_metrics.get("eval_perplexity"),
                delta_vs_base=eval_metrics.get("delta_vs_base_loss"),
                gpu_hours=train_data.get("gpu_hours", 0),
                training_steps=train_data.get("training_steps", 0),
                adapter_path=train_data.get("adapter_path"),
                seed=trial_config.seed,
            )
            exp_state.register_trial(record)

            # 3f. Check stop criteria
            if experiment.stop_threshold > 0 and exp_state.best_metric_value is not None:
                if exp_state.best_metric_value <= experiment.stop_threshold:
                    exp_state.stop_flags.append(f"target_reached ({experiment.stop_metric} <= {experiment.stop_threshold})")

            if exp_state.trials_without_improvement >= experiment.stop_patience:
                exp_state.stop_flags.append(f"patience_exhausted ({experiment.stop_patience} trials without improvement)")

        # ── Step 4: Auto-Merge (optional) ──
        merged_path = None
        if params.auto_merge and not dry_run and exp_state.best_trial and exp_state.best_trial.adapter_path:
            log.append(f"\n🔗 Step 4: Merging best adapter (Trial {exp_state.best_trial.trial_id})...")
            try:
                fmt = ExportFormat(params.merge_format)
            except ValueError:
                fmt = ExportFormat.SAFETENSORS

            merge_result = await self.merge_skill.execute(
                MergeInput(
                    adapter_path=exp_state.best_trial.adapter_path,
                    base_model=experiment.base_model,
                    output_dir=os.path.join(self.output_dir, "merged"),
                    output_format=fmt,
                ),
                state,
            )
            merge_data = merge_result.data if isinstance(merge_result.data, dict) else {}
            merged_path = merge_data.get("merged_path")
            log.append(f"  {merge_result.output_text}")

        # ── Final Summary ──
        log.append(f"\n{'='*50}")
        log.append(f"🏁 Experiment Complete: {len(exp_state.trial_history)} trials")
        if exp_state.best_trial:
            bt = exp_state.best_trial
            log.append(f"🏆 Best: Trial {bt.trial_id} | loss={bt.eval_loss} | rank={bt.config.lora_rank}")
        if merged_path:
            log.append(f"📦 Merged model: {merged_path}")
        log.append(f"💰 GPU Hours: {exp_state.budget_used_gpu_hours:.2f}/{exp_state.budget_total_gpu_hours}")

        final_artifacts = []
        if merged_path:
            final_artifacts.append(merged_path)
        elif exp_state.best_trial and exp_state.best_trial.adapter_path:
            final_artifacts.append(exp_state.best_trial.adapter_path)

        return StepResult(
            data=exp_state.model_dump(),
            output_text="\n".join(log),
            artifacts=final_artifacts,
            metadata={
                "total_trials": len(exp_state.trial_history),
                "best_trial_id": exp_state.best_trial.trial_id if exp_state.best_trial else None,
                "gpu_hours_used": exp_state.budget_used_gpu_hours,
                "merged_path": merged_path,
            },
        )
