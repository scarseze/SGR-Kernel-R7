"""
Input schemas for each LoRA Trainer sub-skill.
Each schema is a Pydantic model consumed by the corresponding skill's execute() method.
"""
from typing import Optional
from pydantic import BaseModel, Field

from skills.lora_trainer.experiment_spec import TrialConfig, ExperimentSpec


# --- Dataset Skill ---
class DatasetAnalyzeInput(BaseModel):
    dataset_path: str = Field(description="Path to dataset (local or HuggingFace ID)")
    format: str = Field(default="auto", description="auto | jsonl | csv | parquet | hf")
    tokenizer_name: Optional[str] = Field(default=None, description="Tokenizer for token stats")
    sample_size: int = Field(default=1000, description="Number of samples for quick analysis")


# --- Config Skill ---
class LoRAConfigInput(BaseModel):
    base_model: str = Field(description="HuggingFace model ID or local path")
    task: str = Field(default="instruction", description="instruction | domain | chat | style")
    dataset_profile: dict = Field(default_factory=dict, description="Output from DatasetSkill.analyze")
    method: str = Field(default="lora", description="lora | qlora | prefix | ia3")
    vram_limit_gb: float = Field(default=24.0, description="Available VRAM in GB")


# --- Training Skill ---
class TrainingInput(BaseModel):
    trial_config: TrialConfig
    dataset_path: str = Field(description="Path to prepared dataset")
    output_dir: str = Field(default="./outputs", description="Directory for checkpoints/adapters")
    resume_from: Optional[str] = Field(default=None, description="Checkpoint path to resume from")
    dry_run: bool = Field(default=False, description="If True, validate config without training")


# --- Eval Skill ---
class EvalInput(BaseModel):
    adapter_path: str = Field(description="Path to the trained adapter")
    base_model: str = Field(description="Base model for comparison")
    eval_dataset_path: str = Field(description="Path to evaluation dataset")
    prompt_suite_path: Optional[str] = Field(default=None, description="Path to prompt test suite")
    compute_perplexity: bool = True
    compute_delta: bool = True


# --- HPO Skill ---
class HPOInput(BaseModel):
    trial_history: list[dict] = Field(default_factory=list, description="Previous trial records")
    search_space: dict = Field(default_factory=dict, description="Current search space ranges")
    strategy: str = Field(default="bayesian", description="random | bayesian | bandit | halving")
    budget_remaining: float = Field(default=10.0, description="Remaining GPU hours")
    trials_remaining: int = Field(default=20, description="Remaining trial count")


# --- Artifact Skill ---
class ArtifactStoreInput(BaseModel):
    adapter_path: str = Field(description="Path to adapter weights")
    trial_id: int
    metrics: dict = Field(default_factory=dict, description="Eval metrics for this trial")
    experiment_id: Optional[str] = Field(default=None)
    tags: list[str] = Field(default_factory=list, description="Tags for artifact search")


# --- Orchestrator (main handler) ---
class LoRATrainerInput(BaseModel):
    """Top-level input for the LoRA Trainer orchestrator."""
    experiment: ExperimentSpec
    dry_run: bool = Field(default=False, description="Run full loop but skip actual GPU training")
    max_loop_iterations: int = Field(default=20, description="Safety limit on orchestration loop")
    auto_merge: bool = Field(default=False, description="Merge best adapter after HPO completes")
    merge_format: str = Field(default="safetensors", description="safetensors | gguf | pytorch")
