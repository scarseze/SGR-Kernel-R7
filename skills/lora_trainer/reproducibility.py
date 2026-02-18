"""
Reproducibility Layer — ExperimentManifest + Replay.

Ensures every experiment can be exactly reproduced:
- Full config capture (model, data, hyperparams, seeds, versions)
- SHA256 hashes for data and adapter integrity
- Git-like lineage tracking
"""
import os
import json
import hashlib
import platform
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field

from skills.lora_trainer.experiment_spec import ExperimentSpec, TrialConfig
from skills.lora_trainer.state import ExperimentState, TrialRecord


class EnvironmentSnapshot(BaseModel):
    """Capture of the compute environment."""
    python_version: str = platform.python_version()
    platform: str = platform.platform()
    torch_version: str = ""
    transformers_version: str = ""
    peft_version: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_count: int = 0


class ExperimentManifest(BaseModel):
    """
    Complete reproducible experiment record.
    Everything needed to reproduce the experiment exactly.
    """
    manifest_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Identity
    experiment_id: str = ""
    goal: str = ""
    author: str = ""

    # Spec
    experiment_spec: Optional[ExperimentSpec] = None

    # Environment
    environment: EnvironmentSnapshot = Field(default_factory=EnvironmentSnapshot)

    # Data
    dataset_path: str = ""
    dataset_hash: str = ""  # SHA256 of dataset files
    dataset_size: int = 0

    # Results
    best_trial: Optional[TrialRecord] = None
    total_trials: int = 0
    total_gpu_hours: float = 0.0

    # Artifact hashes
    adapter_hash: str = ""
    config_hash: str = ""

    # Lineage
    parent_experiment_id: Optional[str] = None
    base_model_hash: str = ""

    # Full state snapshot
    full_state: Optional[dict] = None


def capture_environment() -> EnvironmentSnapshot:
    """Capture current environment versions."""
    env = EnvironmentSnapshot()

    try:
        import torch
        env.torch_version = torch.__version__
        env.cuda_version = torch.version.cuda or ""
        if torch.cuda.is_available():
            env.gpu_name = torch.cuda.get_device_name(0)
            env.gpu_count = torch.cuda.device_count()
    except ImportError:
        pass

    try:
        import transformers
        env.transformers_version = transformers.__version__
    except ImportError:
        pass

    try:
        import peft
        env.peft_version = peft.__version__
    except ImportError:
        pass

    return env


def compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file or directory."""
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
    return sha256.hexdigest()


def create_manifest(experiment_spec: ExperimentSpec,
                    experiment_state: ExperimentState,
                    experiment_id: str = "",
                    author: str = "") -> ExperimentManifest:
    """Create a complete experiment manifest from spec and state."""
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        goal=experiment_spec.goal,
        author=author,
        experiment_spec=experiment_spec,
        environment=capture_environment(),
        dataset_path=experiment_spec.dataset_path,
        best_trial=experiment_state.best_trial,
        total_trials=len(experiment_state.trial_history),
        total_gpu_hours=experiment_state.budget_used_gpu_hours,
        full_state=experiment_state.model_dump(),
    )

    # Hash dataset if accessible
    if os.path.exists(experiment_spec.dataset_path):
        manifest.dataset_hash = compute_file_hash(experiment_spec.dataset_path)
        if os.path.isfile(experiment_spec.dataset_path):
            manifest.dataset_size = os.path.getsize(experiment_spec.dataset_path)

    # Hash adapter if available
    if experiment_state.best_trial and experiment_state.best_trial.adapter_path:
        if os.path.exists(experiment_state.best_trial.adapter_path):
            manifest.adapter_hash = compute_file_hash(experiment_state.best_trial.adapter_path)

    return manifest


def save_manifest(manifest: ExperimentManifest, output_path: str) -> str:
    """Save manifest to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    return output_path


def load_manifest(path: str) -> ExperimentManifest:
    """Load manifest from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return ExperimentManifest(**json.load(f))


def verify_reproducibility(manifest: ExperimentManifest) -> dict:
    """
    Verify that current environment can reproduce the experiment.
    Returns dict of checks and their pass/fail status.
    """
    checks = {}
    current_env = capture_environment()

    checks["python_version"] = {
        "match": current_env.python_version == manifest.environment.python_version,
        "expected": manifest.environment.python_version,
        "actual": current_env.python_version,
    }
    checks["torch_version"] = {
        "match": current_env.torch_version == manifest.environment.torch_version,
        "expected": manifest.environment.torch_version,
        "actual": current_env.torch_version,
    }
    checks["gpu_available"] = {
        "match": current_env.gpu_count >= manifest.environment.gpu_count,
        "expected": f">={manifest.environment.gpu_count}",
        "actual": str(current_env.gpu_count),
    }

    # Check dataset integrity
    if manifest.dataset_path and os.path.exists(manifest.dataset_path):
        actual_hash = compute_file_hash(manifest.dataset_path)
        checks["dataset_hash"] = {
            "match": actual_hash == manifest.dataset_hash,
            "expected": manifest.dataset_hash[:16] + "...",
            "actual": actual_hash[:16] + "...",
        }
    else:
        checks["dataset_hash"] = {
            "match": False,
            "expected": "exists",
            "actual": "not found",
        }

    all_pass = all(c["match"] for c in checks.values())
    checks["_all_pass"] = all_pass

    return checks
