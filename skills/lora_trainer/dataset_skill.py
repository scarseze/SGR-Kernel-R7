"""
DatasetSkill — анализ, нормализация и профилирование датасетов для LoRA-обучения.

Контракт:
- Вход: DatasetAnalyzeInput (path, format, tokenizer)
- Выход: StepResult(data=DatasetProfile)
- deterministic=True, requires_gpu=False, cost_class=CHEAP
"""
import json
import os
from typing import Type
from pydantic import BaseModel

from skills.base import BaseSkill
from skills.lora_trainer.schema import DatasetAnalyzeInput
from skills.lora_trainer.state import DatasetProfile
from core.types import SkillMetadata, Capability, RiskLevel, CostClass, RetryPolicy
from core.result import StepResult


class DatasetSkill(BaseSkill):

    @property
    def name(self) -> str:
        return "dataset_skill"

    @property
    def description(self) -> str:
        return "Analyzes and profiles datasets for LoRA/PEFT training: format detection, token stats, splits."

    @property
    def input_schema(self) -> Type[DatasetAnalyzeInput]:
        return DatasetAnalyzeInput

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
            requires_filesystem=True,
            timeout_sec=120.0,
        )

    async def execute(self, params: DatasetAnalyzeInput, state) -> StepResult:
        """Analyze a dataset and return a DatasetProfile."""
        dataset_path = params.dataset_path
        fmt = params.format
        sample_size = params.sample_size

        # 1. Detect format
        if fmt == "auto":
            fmt = self._detect_format(dataset_path)

        # 2. Load samples
        samples = self._load_samples(dataset_path, fmt, sample_size)
        if not samples:
            return StepResult(
                data=DatasetProfile().model_dump(),
                output_text=f"❌ No data found at {dataset_path}",
                status="failed",
            )

        # 3. Analyze structure
        columns = list(samples[0].keys()) if isinstance(samples[0], dict) else []
        has_instruction = "instruction" in columns
        has_input = "input" in columns
        has_output = "output" in columns or "response" in columns

        # 4. Token stats (approximate by char count / 4 if no tokenizer)
        token_counts = []
        for s in samples:
            text = self._sample_to_text(s)
            # Approximate token count: ~4 chars per token
            token_counts.append(len(text) // 4)

        token_counts.sort()
        total = len(samples)

        profile = DatasetProfile(
            total_samples=total,
            train_samples=int(total * 0.8),
            val_samples=int(total * 0.1),
            test_samples=int(total * 0.1),
            avg_tokens=sum(token_counts) / len(token_counts) if token_counts else 0,
            max_tokens=max(token_counts) if token_counts else 0,
            min_tokens=min(token_counts) if token_counts else 0,
            p95_tokens=token_counts[int(len(token_counts) * 0.95)] if token_counts else 0,
            format_detected=fmt,
            columns=columns,
            has_instruction=has_instruction,
            has_input=has_input,
            has_output=has_output,
            duplicates_found=self._count_duplicates(samples),
        )

        return StepResult(
            data=profile.model_dump(),
            output_text=f"📊 Dataset Profile: {total} samples, avg {profile.avg_tokens:.0f} tokens, format={fmt}",
            metadata={"dataset_path": dataset_path, "format": fmt},
        )

    def _detect_format(self, path: str) -> str:
        """Detect dataset format from file extension or directory structure."""
        if not os.path.exists(path):
            # Assume HuggingFace dataset ID
            return "hf"
        if os.path.isdir(path):
            files = os.listdir(path)
            if any(f.endswith(".jsonl") for f in files):
                return "jsonl"
            if any(f.endswith(".parquet") for f in files):
                return "parquet"
            if any(f.endswith(".csv") for f in files):
                return "csv"
            return "unknown"
        ext = os.path.splitext(path)[1].lower()
        return {".jsonl": "jsonl", ".json": "jsonl", ".csv": "csv", ".parquet": "parquet"}.get(ext, "unknown")

    def _load_samples(self, path: str, fmt: str, sample_size: int) -> list:
        """Load up to sample_size samples from the dataset."""
        samples = []
        try:
            if fmt == "jsonl" and os.path.exists(path):
                file_path = path if os.path.isfile(path) else self._find_first(path, ".jsonl")
                if file_path:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i >= sample_size:
                                break
                            line = line.strip()
                            if line:
                                samples.append(json.loads(line))
            elif fmt == "csv" and os.path.exists(path):
                import csv as csv_mod
                file_path = path if os.path.isfile(path) else self._find_first(path, ".csv")
                if file_path:
                    with open(file_path, "r", encoding="utf-8") as f:
                        reader = csv_mod.DictReader(f)
                        for i, row in enumerate(reader):
                            if i >= sample_size:
                                break
                            samples.append(dict(row))
            elif fmt == "hf":
                # Placeholder: would use datasets.load_dataset()
                # For MVP, return empty and let user provide local files
                pass
        except Exception as e:
            print(f"[dataset_skill] Error loading: {e}")
        return samples

    def _find_first(self, directory: str, extension: str) -> str | None:
        """Find first file with given extension in directory."""
        if not os.path.isdir(directory):
            return None
        for f in sorted(os.listdir(directory)):
            if f.endswith(extension):
                return os.path.join(directory, f)
        return None

    def _sample_to_text(self, sample) -> str:
        """Convert a sample to a single text string for token counting."""
        if isinstance(sample, str):
            return sample
        if isinstance(sample, dict):
            parts = []
            for key in ["instruction", "input", "output", "response", "text", "content"]:
                if key in sample and sample[key]:
                    parts.append(str(sample[key]))
            return " ".join(parts) if parts else json.dumps(sample, ensure_ascii=False)
        return str(sample)

    def _count_duplicates(self, samples: list) -> int:
        """Count exact duplicate samples."""
        seen = set()
        dupes = 0
        for s in samples:
            key = json.dumps(s, sort_keys=True, ensure_ascii=False) if isinstance(s, dict) else str(s)
            if key in seen:
                dupes += 1
            seen.add(key)
        return dupes
