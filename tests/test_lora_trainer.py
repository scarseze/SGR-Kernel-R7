"""
Integration tests for LoRA Trainer Skills.
Runs the full orchestration loop in dry-run mode (no GPU required).
"""
import pytest
import asyncio
import os
import sys
import tempfile
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.lora_trainer.experiment_spec import (
    TrialConfig, ExperimentSpec, SearchSpace, PEFTMethod, TrialStatus
)
from skills.lora_trainer.state import (
    ExperimentState, TrialRecord, DatasetProfile
)
from core.dispatcher import JobStatus
from skills.lora_trainer.schema import (
    DatasetAnalyzeInput, LoRAConfigInput, TrainingInput,
    EvalInput, HPOInput, ArtifactStoreInput, LoRATrainerInput
)
from skills.lora_trainer.dataset_skill import DatasetSkill
from skills.lora_trainer.config_skill import LoRAConfigSkill
from skills.lora_trainer.training_skill import TrainingSkill
from skills.lora_trainer.hpo_skill import HPOSkill
from skills.lora_trainer.artifact_skill import ArtifactSkill


# ─── Fixtures ───

@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a temporary JSONL dataset."""
    data_file = tmp_path / "train.jsonl"
    samples = [
        {"instruction": f"Explain concept {i}", "input": "", "output": f"Concept {i} is about testing LoRA training pipelines."}
        for i in range(50)
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return str(data_file)


class MockState:
    """Minimal mock for AgentState."""
    conversation_history = []
    user_id = "test_user"


# ─── Unit Tests: Experiment Spec ───

class TestExperimentSpec:
    def test_trial_config_defaults(self):
        config = TrialConfig(trial_id=1)
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.method == PEFTMethod.LORA
        assert config.seed == 42

    def test_trial_config_validation(self):
        config = TrialConfig(trial_id=1, lora_rank=64, learning_rate=1e-5)
        assert config.lora_rank == 64
        assert config.learning_rate == 1e-5

    def test_experiment_spec_defaults(self):
        spec = ExperimentSpec(
            goal="Test fine-tuning",
            dataset_path="/data/test.jsonl",
        )
        assert spec.max_trials == 20
        assert spec.budget_gpu_hours == 10.0
        assert spec.stop_metric == "eval_loss"

    def test_search_space_defaults(self):
        space = SearchSpace()
        assert 16 in space.lora_rank
        assert "attention_only" in space.target_modules_presets


# ─── Unit Tests: Experiment State ───

class TestExperimentState:
    def test_initial_state(self):
        state = ExperimentState(experiment_goal="Test")
        assert state.budget_remaining_gpu_hours == 10.0
        assert state.trials_remaining == 20
        assert not state.should_stop

    def test_register_trial(self):
        state = ExperimentState(experiment_goal="Test")
        record = TrialRecord(
            trial_id=1,
            config=TrialConfig(trial_id=1),
            status=TrialStatus.DONE,
            eval_loss=0.5,
            gpu_hours=0.1,
        )
        state.register_trial(record)

        assert len(state.trial_history) == 1
        assert state.best_trial is not None
        assert state.best_trial.trial_id == 1
        assert state.best_metric_value == 0.5

    def test_best_trial_updates(self):
        state = ExperimentState(experiment_goal="Test")

        # Register trial with loss=0.5
        state.register_trial(TrialRecord(
            trial_id=1, config=TrialConfig(trial_id=1),
            status=TrialStatus.DONE, eval_loss=0.5, gpu_hours=0.1,
        ))
        # Register better trial with loss=0.3
        state.register_trial(TrialRecord(
            trial_id=2, config=TrialConfig(trial_id=2),
            status=TrialStatus.DONE, eval_loss=0.3, gpu_hours=0.1,
        ))

        assert state.best_trial.trial_id == 2
        assert state.best_metric_value == 0.3
        assert state.trials_without_improvement == 0

    def test_should_stop_budget(self):
        state = ExperimentState(
            experiment_goal="Test",
            budget_total_gpu_hours=0.1,
            budget_used_gpu_hours=0.2,
        )
        assert state.should_stop

    def test_should_stop_trials(self):
        state = ExperimentState(
            experiment_goal="Test",
            budget_max_trials=2,
        )
        state.register_trial(TrialRecord(trial_id=1, config=TrialConfig(trial_id=1), status=TrialStatus.DONE, eval_loss=0.5))
        state.register_trial(TrialRecord(trial_id=2, config=TrialConfig(trial_id=2), status=TrialStatus.DONE, eval_loss=0.4))
        assert state.should_stop


# ─── Skill Tests ───

class TestDatasetSkill:
    @pytest.mark.asyncio
    async def test_analyze_jsonl(self, tmp_dataset):
        skill = DatasetSkill()
        result = await skill.execute(
            DatasetAnalyzeInput(dataset_path=tmp_dataset),
            MockState(),
        )
        assert result.data["total_samples"] == 50
        assert result.data["format_detected"] == "jsonl"
        assert result.data["has_instruction"] is True
        assert result.data["has_output"] is True

    @pytest.mark.asyncio
    async def test_analyze_missing(self):
        skill = DatasetSkill()
        result = await skill.execute(
            DatasetAnalyzeInput(dataset_path="/nonexistent/path.jsonl"),
            MockState(),
        )
        assert result.data["total_samples"] == 0


class TestConfigSkill:
    @pytest.mark.asyncio
    async def test_generate_config(self):
        skill = LoRAConfigSkill()
        result = await skill.execute(
            LoRAConfigInput(base_model="unsloth/Llama-3.2-1B", task="instruction"),
            MockState(),
        )
        data = result.data
        assert "trial_config" in data
        assert "search_space" in data
        assert "memory_estimate" in data
        assert data["memory_estimate"]["fits"] is True

    @pytest.mark.asyncio
    async def test_qlora_config(self):
        skill = LoRAConfigSkill()
        result = await skill.execute(
            LoRAConfigInput(base_model="meta-llama/Llama-3.1-8B", method="qlora", vram_limit_gb=8.0),
            MockState(),
        )
        data = result.data
        assert data["trial_config"]["quantization_bits"] == 4


class TestTrainingSkill:
    @pytest.mark.asyncio
    async def test_dry_run(self):
        skill = TrainingSkill()
        config = TrialConfig(trial_id=1, base_model="test-model")
        result = await skill.execute(
            TrainingInput(trial_config=config, dataset_path="/data/test", dry_run=True),
            MockState(),
        )
        assert result.data["status"] == "dry_run"
        assert result.data["training_steps"] > 0
        assert result.data["gpu_hours"] > 0


class TestHPOSkill:
    @pytest.mark.asyncio
    async def test_random_search(self):
        skill = HPOSkill()
        result = await skill.execute(
            HPOInput(strategy="random"),
            MockState(),
        )
        assert "suggested_config" in result.data
        assert result.data["suggested_config"]["trial_id"] == 1

    @pytest.mark.asyncio
    async def test_bayesian_with_history(self):
        skill = HPOSkill()
        history = [
            {"trial_id": i, "config": {"lora_rank": 16, "learning_rate": 2e-4, "target_modules": ["q_proj", "v_proj"]}, "status": "done", "eval_loss": 0.5 - i * 0.1}
            for i in range(5)
        ]
        result = await skill.execute(
            HPOInput(trial_history=history, strategy="bayesian"),
            MockState(),
        )
        assert result.data["strategy"] == "bayesian"


class TestArtifactSkill:
    @pytest.mark.asyncio
    async def test_store_artifact(self, tmp_path):
        # Create a fake adapter dir
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_text("fake weights")

        skill = ArtifactSkill(artifact_root=str(tmp_path / "artifacts"))
        result = await skill.execute(
            ArtifactStoreInput(
                adapter_path=str(adapter_dir),
                trial_id=1,
                metrics={"eval_loss": 0.42},
                tags=["lora", "test"],
            ),
            MockState(),
        )
        assert result.data["version"] == "v1"
        assert result.data["trial_id"] == 1
        assert len(result.data["adapter_hash"]) == 64  # SHA256


# ─── Integration Test ───

class TestLoRATrainerIntegration:
    """Full dry-run loop test — no GPU required."""

    @pytest.mark.asyncio
    async def test_dry_run_loop(self, tmp_dataset, tmp_path):
        from skills.lora_trainer.handler import LoRATrainerSkill

        skill = LoRATrainerSkill(output_dir=str(tmp_path / "outputs"))
        spec = ExperimentSpec(
            goal="Test dry-run loop",
            dataset_path=tmp_dataset,
            base_model="unsloth/Llama-3.2-1B",
            max_trials=3,
            budget_gpu_hours=1.0,
        )

        result = await skill.execute(
            LoRATrainerInput(experiment=spec, dry_run=True, max_loop_iterations=3),
            MockState(),
        )

        assert result.output_text is not None
        assert "Step 1" in result.output_text
        assert "Step 2" in result.output_text
        data = result.data
        assert len(data["trial_history"]) == 3
        assert data["budget_used_gpu_hours"] > 0


# ─── Phase 9: Merge Skill Tests ───

class TestMergeSkill:
    def test_merge_input_schema(self):
        from skills.lora_trainer.merge_skill import MergeInput, ExportFormat, GGUFQuantization
        inp = MergeInput(
            adapter_path="/path/to/adapter",
            base_model="meta-llama/Llama-3.2-1B",
        )
        assert inp.output_format == ExportFormat.SAFETENSORS
        assert inp.gguf_quantization == GGUFQuantization.Q4_K_M
        assert inp.validate_after_merge is True

    def test_export_format_enum(self):
        from skills.lora_trainer.merge_skill import ExportFormat
        assert ExportFormat.GGUF.value == "gguf"
        assert ExportFormat.SAFETENSORS.value == "safetensors"
        assert ExportFormat.PYTORCH.value == "pytorch"

    def test_merge_result_model(self):
        from skills.lora_trainer.merge_skill import MergeResult
        result = MergeResult(
            merged_path="/merged/model",
            export_format="safetensors",
            model_size_gb=4.2,
        )
        assert result.status == "completed"
        assert result.model_size_gb == 4.2

    def test_orchestrator_auto_merge_field(self):
        """LoRATrainerInput should have auto_merge and merge_format."""
        spec = ExperimentSpec(goal="test", dataset_path="/data.jsonl")
        inp = LoRATrainerInput(experiment=spec, auto_merge=True, merge_format="gguf")
        assert inp.auto_merge is True
        assert inp.merge_format == "gguf"


# ─── Phase 10: Tracing & Dashboard Tests ───

class TestExperimentTracer:
    def test_event_lifecycle(self):
        from skills.lora_trainer.trace_adapter import ExperimentTracer
        tracer = ExperimentTracer(experiment_id="test_exp")

        tracer.trial_started(trial_id=1, config={"lora_rank": 16})
        tracer.hpo_decision(trial_id=1, strategy="random", suggested_config={"lora_rank": 16})
        tracer.trial_completed(trial_id=1, eval_loss=0.42, gpu_hours=0.5)
        tracer.budget_alert(remaining_hours=2.0, remaining_trials=5)

        summary = tracer.summary
        assert summary["trials_started"] == 1
        assert summary["trials_completed"] == 1
        assert summary["hpo_decisions"] == 1
        assert summary["budget_alerts"] == 1

    def test_trial_failure(self):
        from skills.lora_trainer.trace_adapter import ExperimentTracer
        tracer = ExperimentTracer(experiment_id="fail_exp")
        tracer.trial_started(trial_id=1, config={})
        tracer.trial_failed(trial_id=1, error="OOM")

        assert tracer.summary["trials_failed"] == 1

    def test_early_stop(self):
        from skills.lora_trainer.trace_adapter import ExperimentTracer
        tracer = ExperimentTracer(experiment_id="stop_exp")
        tracer.early_stop(reason="target_reached", best_trial_id=3, best_loss=0.15)

        assert tracer.summary["early_stops"] == 1


class TestDashboard:
    def _make_state(self):
        state = ExperimentState(experiment_goal="Test dashboard")
        for i in range(3):
            state.register_trial(TrialRecord(
                trial_id=i + 1,
                config=TrialConfig(trial_id=i + 1, lora_rank=16 * (i + 1)),
                status=TrialStatus.DONE,
                eval_loss=0.5 - i * 0.1,
                gpu_hours=0.3 + i * 0.1,
            ))
        return state

    def test_text_render(self):
        from skills.lora_trainer.dashboard import ExperimentDashboard
        dashboard = ExperimentDashboard(self._make_state())
        text = dashboard.render_text()
        assert "LoRA Experiment Dashboard" in text
        assert "Trial" in text
        assert "Budget" in text

    def test_markdown_render(self):
        from skills.lora_trainer.dashboard import ExperimentDashboard
        dashboard = ExperimentDashboard(self._make_state())
        md = dashboard.render_markdown()
        assert "# Experiment Report" in md
        assert "| #" in md  # Table header
        assert "GPU Hours" in md

    def test_generate_report_shortcut(self):
        from skills.lora_trainer.dashboard import generate_report
        state = self._make_state()
        text = generate_report(state, format="text")
        md = generate_report(state, format="markdown")
        assert len(text) > 50
        assert len(md) > 50


# ─── Phase 11: Dispatcher Tests ───

class TestDispatcher:
    def test_create_local_dispatcher(self):
        from core.dispatcher import get_dispatcher, LocalDispatcher
        d = get_dispatcher("local")
        assert isinstance(d, LocalDispatcher)

    def test_create_unknown_dispatcher(self):
        from core.dispatcher import get_dispatcher
        with pytest.raises(ValueError, match="Unknown backend"):
            get_dispatcher("nonexistent")

    def test_job_status_enum(self):
        from core.dispatcher import JobStatus
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.COMPLETED.value == "completed"


# ─── Phase 12: Protocol Tests ───

class TestAblationProtocol:
    def test_generate_configs(self):
        from skills.lora_trainer.protocols.ablation import AblationProtocol
        protocol = AblationProtocol()
        base = TrialConfig(trial_id=0, lora_rank=16)
        configs = protocol.generate_configs(base, "lora_rank", [4, 8, 16, 32, 64])

        assert len(configs) == 5
        assert configs[0].lora_rank == 4
        assert configs[4].lora_rank == 64
        assert configs[2].trial_id == 3  # start_trial_id=1, so 1+2=3

    def test_invalid_param(self):
        from skills.lora_trainer.protocols.ablation import AblationProtocol
        protocol = AblationProtocol()
        with pytest.raises(ValueError, match="not found"):
            protocol.generate_configs(TrialConfig(trial_id=0), "nonexistent_param", [1, 2])

    def test_analyze_results(self):
        from skills.lora_trainer.protocols.ablation import AblationProtocol
        protocol = AblationProtocol()
        trials = [
            {"value": 4, "trial_id": 1, "eval_loss": 0.6},
            {"value": 16, "trial_id": 2, "eval_loss": 0.45},
            {"value": 64, "trial_id": 3, "eval_loss": 0.42},
        ]
        result = protocol.analyze_results("lora_rank", trials)
        assert result.baseline_loss == 0.6
        assert len(result.results) == 3
        assert result.results[2]["delta"] < 0  # Improvement

    def test_suggested_ablations(self):
        from skills.lora_trainer.protocols.ablation import AblationProtocol
        suggestions = AblationProtocol().suggested_ablations()
        assert len(suggestions) >= 3
        params = [s["param"] for s in suggestions]
        assert "lora_rank" in params


class TestCurriculumProtocol:
    def test_build_stages(self):
        from skills.lora_trainer.protocols.curriculum import CurriculumProtocol
        protocol = CurriculumProtocol()
        stages = protocol.build_stages({
            "easy": "/data/easy.jsonl",
            "medium": "/data/med.jsonl",
            "hard": "/data/hard.jsonl",
        })
        assert len(stages) == 3
        assert stages[0].name == "easy"
        assert stages[0].learning_rate_multiplier == 1.0
        assert stages[2].learning_rate_multiplier < 1.0  # Decays

    def test_length_curriculum(self):
        from skills.lora_trainer.protocols.curriculum import CurriculumProtocol
        protocol = CurriculumProtocol()
        stages = protocol.build_length_curriculum("/data/train.jsonl")
        assert len(stages) == 4
        assert stages[0].name == "len_128"

    def test_modify_config(self):
        from skills.lora_trainer.protocols.curriculum import CurriculumProtocol, CurriculumStage
        protocol = CurriculumProtocol()
        base = TrialConfig(trial_id=1, learning_rate=2e-4, num_epochs=5)
        stage = CurriculumStage(name="easy", dataset_path="/d.jsonl", num_epochs=2, learning_rate_multiplier=0.5)
        config = protocol.modify_config_for_stage(base, stage)
        assert config.num_epochs == 2
        assert config.learning_rate == 1e-4  # 2e-4 * 0.5


class TestEnsembleProtocol:
    def test_generate_diverse(self):
        from skills.lora_trainer.protocols.ensemble import EnsembleProtocol
        protocol = EnsembleProtocol()
        configs = protocol.generate_diverse_configs(TrialConfig(trial_id=0), n_members=5)
        assert len(configs) == 5
        # Configs should have different ranks
        ranks = {c.lora_rank for c in configs}
        assert len(ranks) > 1  # Not all the same

    def test_merge_weights_inverse(self):
        from skills.lora_trainer.protocols.ensemble import EnsembleProtocol, EnsembleMember
        protocol = EnsembleProtocol()
        members = [
            EnsembleMember(trial_id=1, config=TrialConfig(trial_id=1), eval_loss=0.5),
            EnsembleMember(trial_id=2, config=TrialConfig(trial_id=2), eval_loss=0.25),
        ]
        weighted = protocol.compute_merge_weights(members, "inverse_loss")
        # Lower loss should get higher weight
        assert weighted[1].weight > weighted[0].weight

    def test_merge_weights_equal(self):
        from skills.lora_trainer.protocols.ensemble import EnsembleProtocol, EnsembleMember
        protocol = EnsembleProtocol()
        members = [
            EnsembleMember(trial_id=1, config=TrialConfig(trial_id=1), eval_loss=0.5),
            EnsembleMember(trial_id=2, config=TrialConfig(trial_id=2), eval_loss=0.3),
        ]
        weighted = protocol.compute_merge_weights(members, "equal")
        assert abs(weighted[0].weight - 0.5) < 0.01
        assert abs(weighted[1].weight - 0.5) < 0.01


class TestReproducibility:
    def test_environment_snapshot(self):
        from skills.lora_trainer.reproducibility import capture_environment
        env = capture_environment()
        assert len(env.python_version) > 0
        assert len(env.platform) > 0

    def test_file_hash(self, tmp_path):
        from skills.lora_trainer.reproducibility import compute_file_hash
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = compute_file_hash(str(f))
        assert len(h) == 64  # SHA256

    def test_create_manifest(self):
        from skills.lora_trainer.reproducibility import create_manifest
        spec = ExperimentSpec(goal="Manifest test", dataset_path="/nonexistent.jsonl")
        state = ExperimentState(experiment_goal="Manifest test")
        state.register_trial(TrialRecord(
            trial_id=1, config=TrialConfig(trial_id=1),
            status=TrialStatus.DONE, eval_loss=0.4, gpu_hours=0.5,
        ))
        manifest = create_manifest(spec, state, experiment_id="exp_001")
        assert manifest.experiment_id == "exp_001"
        assert manifest.total_trials == 1
        assert manifest.best_trial.trial_id == 1

    def test_save_load_manifest(self, tmp_path):
        from skills.lora_trainer.reproducibility import (
            create_manifest, save_manifest, load_manifest,
        )
        spec = ExperimentSpec(goal="Save test", dataset_path="/data.jsonl")
        state = ExperimentState(experiment_goal="Save test")
        manifest = create_manifest(spec, state, experiment_id="exp_save")

        path = str(tmp_path / "manifest.json")
        save_manifest(manifest, path)
        loaded = load_manifest(path)
        assert loaded.experiment_id == "exp_save"
        assert loaded.goal == "Save test"

    def test_verify_reproducibility(self):
        from skills.lora_trainer.reproducibility import (
            ExperimentManifest, EnvironmentSnapshot, verify_reproducibility,
        )
        import platform as plat
        manifest = ExperimentManifest(
            experiment_id="test",
            environment=EnvironmentSnapshot(python_version=plat.python_version()),
        )
        checks = verify_reproducibility(manifest)
        assert checks["python_version"]["match"] is True
