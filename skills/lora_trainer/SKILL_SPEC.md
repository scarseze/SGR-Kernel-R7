# LoRA Trainer — Skill Specification

> Контракты, параметры и примеры использования всех skills.
> Рабочая директория: `c:\Users\macht\Scar\sgr_kernel\skills\lora_trainer\`

---

## 1. Orchestrator — `LoRATrainerSkill`

**Файл**: [handler.py](file:///c:/Users/macht/Scar/sgr_kernel/skills/lora_trainer/handler.py)

### Input: `LoRATrainerInput`
```python
LoRATrainerInput(
    experiment=ExperimentSpec(
        goal="Fine-tune for QA",
        dataset_path="data/train.jsonl",
        base_model="unsloth/Llama-3.2-1B",
        max_trials=10,
        budget_gpu_hours=5.0,
        methods=[PEFTMethod.LORA],
        stop_metric="eval_loss",
        stop_threshold=0.3,
        stop_patience=3,
    ),
    dry_run=False,
    max_loop_iterations=20,
    auto_merge=True,          # Phase 9
    merge_format="safetensors",
)
```

### Output: `StepResult`
| Field | Type | Описание |
|-------|------|----------|
| `data` | `dict` | `ExperimentState.model_dump()` — trial_history, best_trial, budget |
| `artifacts` | `list[str]` | Пути к merged model или best adapter |
| `output_text` | `str` | Лог выполнения (Step 1..4) |
| `metadata` | `dict` | `total_trials`, `best_trial_id`, `gpu_hours_used`, `merged_path` |

---

## 2. DatasetSkill

**Файл**: [dataset_skill.py](file:///c:/Users/macht/Scar/sgr_kernel/skills/lora_trainer/dataset_skill.py) | **GPU**: ❌

### Input
```python
DatasetAnalyzeInput(
    dataset_path="data/train.jsonl",
    format="auto",          # auto | jsonl | csv | parquet | hf
    tokenizer_name=None,     # для точных token stats
    sample_size=1000,
)
```

### Output `data` dict
| Key | Type | Описание |
|-----|------|----------|
| `total_samples` | `int` | Кол-во записей |
| `format_detected` | `str` | jsonl / csv / parquet / hf |
| `has_instruction` | `bool` | Наличие поля instruction |
| `has_output` | `bool` | Наличие поля output |
| `avg_tokens` | `float` | Средняя длина в ~токенах |
| `max_tokens` | `int` | Макс. длина |
| `columns` | `list` | Список колонок |
| `duplicates_found` | `int` | Кол-во дубликатов |

---

## 3. LoRAConfigSkill

**Файл**: [config_skill.py](file:///c:/Users/macht/Scar/sgr_kernel/skills/lora_trainer/config_skill.py) | **GPU**: ❌

### Input
```python
LoRAConfigInput(
    base_model="meta-llama/Llama-3.1-8B",
    task="instruction",      # instruction | domain | chat | style
    method="qlora",           # lora | qlora | prefix | ia3
    vram_limit_gb=24.0,
    dataset_profile={},       # output from DatasetSkill
)
```

### Output `data` dict
| Key | Type | Описание |
|-----|------|----------|
| `trial_config` | `dict` | TrialConfig с rank, alpha, lr, modules |
| `search_space` | `dict` | HPO ranges для всех гиперпараметров |
| `memory_estimate` | `dict` | `{vram_gb, fits, model_params}` |

---

## 4. TrainingSkill

**Файл**: [training_skill.py](file:///c:/Users/macht/Scar/sgr_kernel/skills/lora_trainer/training_skill.py) | **GPU**: ✅

### Input
```python
TrainingInput(
    trial_config=TrialConfig(trial_id=1, lora_rank=16, learning_rate=2e-4),
    dataset_path="data/train.jsonl",
    output_dir="./outputs",
    resume_from=None,         # checkpoint path
    dry_run=False,
)
```

### Output `data` dict
| Key | Type | Описание |
|-----|------|----------|
| `status` | `str` | `completed` / `dry_run` / `failed` |
| `adapter_path` | `str` | Путь к сохранённому адаптеру |
| `training_steps` | `int` | Кол-во шагов |
| `gpu_hours` | `float` | Затраченные GPU-часы |
| `final_loss` | `float` | Финальный training loss |

---

## 5. EvalSkill

**Файл**: [eval_skill.py](file:///c:/Users/macht/Scar/sgr_kernel/skills/lora_trainer/eval_skill.py) | **GPU**: ✅

### Input
```python
EvalInput(
    adapter_path="outputs/trial_1/adapter",
    base_model="unsloth/Llama-3.2-1B",
    eval_dataset_path="data/test.jsonl",
    compute_perplexity=True,
    compute_delta=True,
)
```

### Output `data` dict
| Key | Type |
|-----|------|
| `eval_loss` | `float` |
| `eval_perplexity` | `float` |
| `base_loss` | `float` |
| `delta_vs_base_loss` | `float` |

---

## 6. HPOSkill

**Файл**: [hpo_skill.py](file:///c:/Users/macht/Scar/sgr_kernel/skills/lora_trainer/hpo_skill.py) | **GPU**: ❌

### Input
```python
HPOInput(
    trial_history=[...],      # list[dict] — предыдущие trials
    search_space={},          # ranges из ConfigSkill
    strategy="bayesian",      # random | bayesian | bandit | halving
    budget_remaining=8.0,
    trials_remaining=15,
)
```

### Стратегии
| Strategy | Когда | Логика |
|----------|-------|--------|
| `random` | Первые 3 trial | Случайный sampling из search_space |
| `bayesian` | После 3 trials | Perturbation лучшего конфига |
| `bandit` | Epsilon-greedy | 80% exploit + 20% explore |
| `halving` | Большой бюджет | Successive halving |

---

## 7. ArtifactSkill

**Файл**: [artifact_skill.py](file:///c:/Users/macht/Scar/sgr_kernel/skills/lora_trainer/artifact_skill.py) | **GPU**: ❌

### Input
```python
ArtifactStoreInput(
    adapter_path="outputs/trial_1/adapter",
    trial_id=1,
    metrics={"eval_loss": 0.42},
    tags=["lora", "instruction"],
)
```

### Output `data` dict
| Key | Type | Описание |
|-----|------|----------|
| `version` | `str` | `v1`, `v2`, ... |
| `adapter_hash` | `str` | SHA256 (64 chars) |
| `stored_path` | `str` | Путь в artifact storage |

---

## 8. MergeSkill (Phase 9)

**Файл**: [merge_skill.py](file:///c:/Users/macht/Scar/sgr_kernel/skills/lora_trainer/merge_skill.py) | **GPU**: ✅

### Input
```python
MergeInput(
    adapter_path="outputs/trial_1/adapter",
    base_model="unsloth/Llama-3.2-1B",
    output_format=ExportFormat.GGUF,
    gguf_quantization=GGUFQuantization.Q4_K_M,
    validate_after_merge=True,
)
```

### Форматы экспорта
| Format | Расширение | Использование |
|--------|-----------|---------------|
| `safetensors` | `.safetensors` | HuggingFace, vLLM |
| `gguf` | `.gguf` | llama.cpp, Ollama |
| `pytorch` | `.bin` | Legacy PyTorch |

---

## 9. Research Protocols (Phase 12)

### Ablation
```python
from skills.lora_trainer.protocols.ablation import AblationProtocol
protocol = AblationProtocol()
configs = protocol.generate_configs(base, "lora_rank", [4, 8, 16, 32, 64])
```

### Curriculum
```python
from skills.lora_trainer.protocols.curriculum import CurriculumProtocol
stages = CurriculumProtocol().build_stages({"easy": "e.jsonl", "hard": "h.jsonl"})
```

### Ensemble
```python
from skills.lora_trainer.protocols.ensemble import EnsembleProtocol
configs = EnsembleProtocol().generate_diverse_configs(base, n_members=5)
members = protocol.compute_merge_weights(members, "inverse_loss")
```

### Reproducibility
```python
from skills.lora_trainer.reproducibility import create_manifest, save_manifest
manifest = create_manifest(spec, state, experiment_id="exp_001")
save_manifest(manifest, "experiment_manifest.json")
```

---

## Зависимости

| Модуль | Для dry-run | Для GPU |
|--------|-------------|---------|
| `pydantic` | ✅ | ✅ |
| `torch` | ❌ | ✅ |
| `transformers` | ❌ | ✅ |
| `peft` | ❌ | ✅ |
| `trl` | ❌ | ✅ |
| `datasets` | ❌ | ✅ |
| `bitsandbytes` | ❌ | ✅ (QLoRA) |
