"""
Experiment Specification & Trial Configuration models.

Defines the contract for LoRA/PEFT experiments:
- ExperimentSpec: top-level goal + constraints
- TrialConfig: single trial hyperparameters
- SearchSpace: HPO search ranges
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PEFTMethod(str, Enum):
    LORA = "lora"
    QLORA = "qlora"
    PREFIX = "prefix"
    IA3 = "ia3"


class TrialStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    EVAL = "eval"
    DONE = "done"
    FAILED = "failed"
    PRUNED = "pruned"


class TrialConfig(BaseModel):
    """Single trial hyperparameters — fully deterministic, reproducible."""
    trial_id: int
    method: PEFTMethod = PEFTMethod.LORA
    base_model: str = "unsloth/Llama-3.2-1B"

    # LoRA-specific
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    learning_rate: float = Field(default=2e-4, gt=0)
    num_epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation: int = Field(default=4, ge=1)
    max_seq_length: int = Field(default=512, ge=64)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    seed: int = 42

    # QLoRA-specific
    quantization_bits: Optional[int] = Field(default=None, description="4 or 8 for QLoRA")


class SearchSpace(BaseModel):
    """HPO search ranges for hyperparameters."""
    lora_rank: list[int] = Field(default_factory=lambda: [8, 16, 32, 64])
    lora_alpha: list[int] = Field(default_factory=lambda: [16, 32, 64])
    learning_rate: list[float] = Field(default_factory=lambda: [1e-5, 5e-5, 1e-4, 2e-4, 5e-4])
    lora_dropout: list[float] = Field(default_factory=lambda: [0.0, 0.05, 0.1])
    target_modules_presets: dict[str, list[str]] = Field(default_factory=lambda: {
        "attention_only": ["q_proj", "v_proj"],
        "attention_full": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp_only": ["gate_proj", "up_proj", "down_proj"],
        "hybrid": ["q_proj", "v_proj", "gate_proj", "down_proj"],
    })


class ExperimentSpec(BaseModel):
    """Top-level experiment definition — the 'GOAL' for the Kernel."""
    goal: str = Field(description="Natural language description of the experiment objective")
    dataset_path: str = Field(description="Path to the dataset (local or HuggingFace)")
    dataset_format: str = Field(default="auto", description="auto | jsonl | csv | parquet | hf")
    base_model: str = "unsloth/Llama-3.2-1B"

    methods: list[PEFTMethod] = Field(default_factory=lambda: [PEFTMethod.LORA])
    search_space: Optional[SearchSpace] = None

    # Budget constraints
    budget_gpu_hours: float = Field(default=10.0, gt=0)
    max_trials: int = Field(default=20, ge=1)

    # Stop criteria
    stop_metric: str = "eval_loss"
    stop_threshold: float = Field(default=0.0, description="Stop if metric reaches this value (0=disabled)")
    stop_patience: int = Field(default=5, description="Stop after N trials without improvement")

    # Eval protocol
    eval_split: str = "validation"
    prompt_suite_path: Optional[str] = None
