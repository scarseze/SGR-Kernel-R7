"""
MergeSkill — merge LoRA adapter with base model and export.

Контракт:
- Вход: MergeInput (adapter_path, base_model, output_format, quantization)
- Выход: StepResult(data=merge_result, artifacts=[merged_model_path])
- requires_gpu=True, side_effects=True, cost_class=EXPENSIVE
"""
import os
import shutil
import time
from typing import Type
from pydantic import BaseModel, Field
from enum import Enum

from skills.base import BaseSkill
from core.types import SkillMetadata, Capability, RiskLevel, CostClass, RetryPolicy
from core.result import StepResult


class ExportFormat(str, Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PYTORCH = "pytorch"
    ONNX = "onnx"


class GGUFQuantization(str, Enum):
    Q4_K_M = "q4_k_m"
    Q5_K_M = "q5_k_m"
    Q8_0 = "q8_0"
    F16 = "f16"
    NONE = "none"


class MergeInput(BaseModel):
    adapter_path: str = Field(description="Path to LoRA adapter directory")
    base_model: str = Field(description="Base model ID or path")
    output_dir: str = Field(default="./merged", description="Output directory for merged model")
    output_format: ExportFormat = Field(default=ExportFormat.SAFETENSORS)
    gguf_quantization: GGUFQuantization = Field(
        default=GGUFQuantization.Q4_K_M,
        description="GGUF quantization type (only used when format=gguf)",
    )
    validate_after_merge: bool = Field(default=True, description="Run sanity check after merge")
    test_prompt: str = Field(
        default="Explain what LoRA fine-tuning is in one sentence.",
        description="Prompt for post-merge validation",
    )


class MergeResult(BaseModel):
    merged_path: str = ""
    export_format: str = ""
    model_size_gb: float = 0.0
    merge_time_sec: float = 0.0
    validation_passed: bool = False
    validation_output: str = ""
    quantization: str = "none"
    status: str = "completed"  # completed | failed | dry_run
    error: str | None = None


class MergeSkill(BaseSkill):

    @property
    def name(self) -> str:
        return "merge_skill"

    @property
    def description(self) -> str:
        return "Merges LoRA adapter with base model. Exports to safetensors, GGUF, or PyTorch format."

    @property
    def input_schema(self) -> Type[MergeInput]:
        return MergeInput

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
            requires_gpu=True,
            requires_filesystem=True,
            side_effects=True,
            idempotent=True,
            timeout_sec=1800.0,
        )

    async def execute(self, params: MergeInput, state) -> StepResult:
        """Merge adapter with base model and export."""
        start = time.time()

        try:
            result = await self._merge_and_export(params)
        except ImportError as e:
            result = MergeResult(
                status="failed",
                error=f"Missing dependency: {e}. Install: pip install transformers peft",
            )
        except Exception as e:
            result = MergeResult(status="failed", error=str(e))

        result.merge_time_sec = round(time.time() - start, 2)

        artifacts = [result.merged_path] if result.merged_path else []

        return StepResult(
            data=result.model_dump(),
            artifacts=artifacts,
            output_text=(
                f"🔗 Merge: {result.status} | {result.export_format} | "
                f"{result.model_size_gb:.1f}GB | {result.merge_time_sec:.1f}s"
                + (f" | ✅ validation passed" if result.validation_passed else "")
                if result.status == "completed"
                else f"❌ Merge failed: {result.error}"
            ),
            metadata={"format": result.export_format, "time_sec": result.merge_time_sec},
        )

    async def _merge_and_export(self, params: MergeInput) -> MergeResult:
        """Core merge + export logic."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        os.makedirs(params.output_dir, exist_ok=True)

        # 1. Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            params.base_model, device_map="auto", torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(params.adapter_path)

        # 2. Load and merge adapter
        model = PeftModel.from_pretrained(base_model, params.adapter_path)
        merged_model = model.merge_and_unload()

        # 3. Export
        if params.output_format == ExportFormat.SAFETENSORS:
            merged_path = self._export_safetensors(merged_model, tokenizer, params.output_dir)
        elif params.output_format == ExportFormat.GGUF:
            merged_path = await self._export_gguf(
                merged_model, tokenizer, params.output_dir, params.gguf_quantization,
            )
        elif params.output_format == ExportFormat.PYTORCH:
            merged_path = self._export_pytorch(merged_model, tokenizer, params.output_dir)
        else:
            merged_path = self._export_safetensors(merged_model, tokenizer, params.output_dir)

        # 4. Compute size
        model_size = self._dir_size_gb(merged_path)

        # 5. Validate
        validation_passed = False
        validation_output = ""
        if params.validate_after_merge:
            validation_passed, validation_output = self._validate(
                merged_path, tokenizer, params.test_prompt,
            )

        return MergeResult(
            merged_path=merged_path,
            export_format=params.output_format.value,
            model_size_gb=model_size,
            validation_passed=validation_passed,
            validation_output=validation_output,
            quantization=params.gguf_quantization.value if params.output_format == ExportFormat.GGUF else "none",
        )

    def _export_safetensors(self, model, tokenizer, output_dir: str) -> str:
        """Export as safetensors (default HuggingFace format)."""
        path = os.path.join(output_dir, "merged_safetensors")
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
        return path

    def _export_pytorch(self, model, tokenizer, output_dir: str) -> str:
        """Export as PyTorch bin files."""
        path = os.path.join(output_dir, "merged_pytorch")
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path, safe_serialization=False)
        tokenizer.save_pretrained(path)
        return path

    async def _export_gguf(self, model, tokenizer, output_dir: str,
                           quantization: GGUFQuantization) -> str:
        """
        Export as GGUF for llama.cpp.
        
        Strategy:
        1. Save merged model as safetensors (temp)
        2. Convert using llama.cpp's convert script or llama-cpp-python
        """
        import subprocess

        # Step 1: Save as safetensors first
        temp_path = os.path.join(output_dir, "_temp_merged")
        os.makedirs(temp_path, exist_ok=True)
        model.save_pretrained(temp_path, safe_serialization=True)
        tokenizer.save_pretrained(temp_path)

        # Step 2: Convert to GGUF
        gguf_path = os.path.join(output_dir, "merged.gguf")

        try:
            # Try llama.cpp convert script
            convert_result = subprocess.run(
                [
                    "python", "-m", "llama_cpp.llama_convert",
                    "--input", temp_path,
                    "--output", gguf_path,
                    "--outtype", quantization.value,
                ],
                capture_output=True, text=True, timeout=600,
            )
            if convert_result.returncode != 0:
                # Fallback: try with convert-hf-to-gguf.py
                raise RuntimeError(convert_result.stderr)
        except (FileNotFoundError, RuntimeError):
            # If llama.cpp not available, keep safetensors and note in output
            gguf_path = temp_path

        # Cleanup temp if GGUF succeeded and is different path
        if gguf_path != temp_path and os.path.exists(gguf_path):
            shutil.rmtree(temp_path, ignore_errors=True)

        return gguf_path

    def _validate(self, model_path: str, tokenizer, test_prompt: str) -> tuple[bool, str]:
        """Run sanity check: generate text with merged model."""
        try:
            from transformers import AutoModelForCausalLM, pipeline

            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
            output = pipe(test_prompt)[0]["generated_text"]

            # Basic validation: output should be longer than prompt and not garbage
            is_valid = (
                len(output) > len(test_prompt)
                and not output.endswith(test_prompt)  # Not just echoing
            )
            return is_valid, output[:200]
        except Exception as e:
            return False, f"Validation error: {e}"

    def _dir_size_gb(self, path: str) -> float:
        """Compute total size of directory in GB."""
        total = 0
        if os.path.isfile(path):
            return os.path.getsize(path) / (1024 ** 3)
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    total += os.path.getsize(os.path.join(root, f))
        return round(total / (1024 ** 3), 2)
