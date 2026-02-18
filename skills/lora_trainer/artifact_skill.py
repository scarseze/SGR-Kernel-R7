"""
ArtifactSkill — управление артефактами LoRA экспериментов.

Контракт:
- Вход: ArtifactStoreInput (adapter_path, trial_id, metrics)
- Выход: StepResult(data=artifact_record)
- requires_filesystem=True, side_effects=True, cost_class=CHEAP
"""
import os
import json
import shutil
import hashlib
from datetime import datetime, timezone
from typing import Type
from pydantic import BaseModel

from skills.base import BaseSkill
from skills.lora_trainer.schema import ArtifactStoreInput
from core.types import SkillMetadata, Capability, RiskLevel, CostClass
from core.result import StepResult


class ArtifactRecord(BaseModel):
    """Immutable record of a stored artifact."""
    trial_id: int
    version: str  # e.g. "v1", "v2"
    adapter_path: str
    config_snapshot_path: str = ""
    metrics_path: str = ""
    created_at: str = ""
    adapter_hash: str = ""
    tags: list[str] = []
    experiment_id: str = ""


class ArtifactSkill(BaseSkill):

    def __init__(self, artifact_root: str = "./artifacts/lora", **kwargs):
        super().__init__(**kwargs)
        self.artifact_root = artifact_root

    @property
    def name(self) -> str:
        return "artifact_skill"

    @property
    def description(self) -> str:
        return "Stores, versions, and manages LoRA adapter artifacts and experiment records."

    @property
    def input_schema(self) -> Type[ArtifactStoreInput]:
        return ArtifactStoreInput

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=self.name,
            version="1.0.0",
            description=self.description,
            capabilities=[Capability.FILESYSTEM],
            risk_level=RiskLevel.LOW,
            cost_class=CostClass.CHEAP,
            requires_filesystem=True,
            side_effects=True,
            idempotent=True,
            timeout_sec=60.0,
        )

    async def execute(self, params: ArtifactStoreInput, state) -> StepResult:
        """Store adapter artifact with versioning and metadata."""
        trial_id = params.trial_id
        experiment_id = params.experiment_id or "default"

        # 1. Determine version
        version = self._next_version(experiment_id, trial_id)

        # 2. Create artifact directory
        artifact_dir = os.path.join(
            self.artifact_root, experiment_id, f"trial_{trial_id}", version
        )
        os.makedirs(artifact_dir, exist_ok=True)

        # 3. Copy adapter files
        stored_adapter_path = os.path.join(artifact_dir, "adapter")
        if os.path.exists(params.adapter_path):
            if os.path.isdir(params.adapter_path):
                if os.path.exists(stored_adapter_path):
                    shutil.rmtree(stored_adapter_path)
                shutil.copytree(params.adapter_path, stored_adapter_path)
            else:
                os.makedirs(stored_adapter_path, exist_ok=True)
                shutil.copy2(params.adapter_path, stored_adapter_path)

        # 4. Save metrics
        metrics_path = os.path.join(artifact_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(params.metrics, f, indent=2, ensure_ascii=False)

        # 5. Compute adapter hash for reproducibility
        adapter_hash = self._compute_hash(stored_adapter_path)

        # 6. Create record
        record = ArtifactRecord(
            trial_id=trial_id,
            version=version,
            adapter_path=stored_adapter_path,
            metrics_path=metrics_path,
            created_at=datetime.now(timezone.utc).isoformat(),
            adapter_hash=adapter_hash,
            tags=params.tags,
            experiment_id=experiment_id,
        )

        # 7. Save record
        record_path = os.path.join(artifact_dir, "record.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record.model_dump(), f, indent=2)

        return StepResult(
            data=record.model_dump(),
            artifacts=[stored_adapter_path, metrics_path],
            output_text=(
                f"🗂️ Artifact stored: trial_{trial_id}/{version} | "
                f"hash={adapter_hash[:12]}... | tags={params.tags}"
            ),
            metadata={"version": version, "hash": adapter_hash},
        )

    def _next_version(self, experiment_id: str, trial_id: int) -> str:
        """Determine next version number for a trial."""
        trial_dir = os.path.join(self.artifact_root, experiment_id, f"trial_{trial_id}")
        if not os.path.exists(trial_dir):
            return "v1"
        existing = [d for d in os.listdir(trial_dir) if d.startswith("v") and os.path.isdir(os.path.join(trial_dir, d))]
        if not existing:
            return "v1"
        max_version = max(int(v[1:]) for v in existing if v[1:].isdigit())
        return f"v{max_version + 1}"

    def _compute_hash(self, path: str) -> str:
        """Compute SHA256 hash of adapter files for reproducibility."""
        sha256 = hashlib.sha256()
        if os.path.isdir(path):
            for root, _, files in sorted(os.walk(path)):
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "rb") as f:
                            for chunk in iter(lambda: f.read(8192), b""):
                                sha256.update(chunk)
                    except (OSError, IOError):
                        continue
        elif os.path.isfile(path):
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
        else:
            return "no_file"
        return sha256.hexdigest()
