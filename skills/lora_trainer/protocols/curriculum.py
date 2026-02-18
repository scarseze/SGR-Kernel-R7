"""
Curriculum Learning Protocol.

Progressive training: easy → hard data ordering.
Train on simple examples first, then gradually increase complexity.
"""
from typing import Optional
from pydantic import BaseModel, Field

from skills.lora_trainer.experiment_spec import TrialConfig


class CurriculumStage(BaseModel):
    """Single stage in curriculum learning."""
    name: str
    dataset_path: str  # Dataset filtered to this difficulty level
    num_epochs: int = 1
    learning_rate_multiplier: float = 1.0  # Scale LR relative to base config
    description: str = ""


class CurriculumResult(BaseModel):
    stages_completed: int = 0
    total_stages: int = 0
    stage_results: list[dict] = Field(default_factory=list)
    final_loss: Optional[float] = None
    improvement_vs_direct: Optional[float] = None


class CurriculumProtocol:
    """
    Progressive training from easy to hard data.
    
    Key idea: start with cleaner/simpler data, progressively add noise/complexity.
    This often produces better final models than training on all data at once.
    
    Usage:
        protocol = CurriculumProtocol()
        stages = protocol.build_stages(
            dataset_splits={"easy": "data/easy.jsonl", "medium": "data/med.jsonl", "hard": "data/hard.jsonl"},
        )
        # Execute sequentially: train stage1, then continue from checkpoint to stage2, etc.
    """

    def build_stages(self, dataset_splits: dict[str, str],
                     epochs_per_stage: int = 1,
                     lr_decay: float = 0.8) -> list[CurriculumStage]:
        """
        Build curriculum stages from dataset splits.
        
        Args:
            dataset_splits: {"easy": path, "medium": path, "hard": path}
            epochs_per_stage: Epochs per stage
            lr_decay: LR multiplier decrease per stage (0.8 = 20% decay each stage)
        """
        stages = []
        for i, (name, path) in enumerate(dataset_splits.items()):
            stages.append(CurriculumStage(
                name=name,
                dataset_path=path,
                num_epochs=epochs_per_stage,
                learning_rate_multiplier=lr_decay ** i,
                description=f"Stage {i+1}: {name} data",
            ))
        return stages

    def build_length_curriculum(self, dataset_path: str,
                                length_buckets: list[int] = None) -> list[CurriculumStage]:
        """
        Build curriculum based on sequence length.
        Start with short sequences, progressively increase.
        
        Args:
            dataset_path: Base dataset path
            length_buckets: Max token lengths per stage [128, 256, 512, 1024]
        """
        if length_buckets is None:
            length_buckets = [128, 256, 512, 1024]

        stages = []
        for i, max_len in enumerate(length_buckets):
            stages.append(CurriculumStage(
                name=f"len_{max_len}",
                dataset_path=dataset_path,
                num_epochs=1,
                learning_rate_multiplier=0.9 ** i,
                description=f"Stage {i+1}: sequences up to {max_len} tokens",
            ))
        return stages

    def modify_config_for_stage(self, base_config: TrialConfig,
                                stage: CurriculumStage) -> TrialConfig:
        """Create a modified config for a specific curriculum stage."""
        config_dict = base_config.model_dump()
        config_dict["num_epochs"] = stage.num_epochs
        config_dict["learning_rate"] = base_config.learning_rate * stage.learning_rate_multiplier
        return TrialConfig(**config_dict)
