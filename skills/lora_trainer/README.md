# LoRA Trainer Skill

Автоматизированный HPO-поиск оптимального LoRA/PEFT адаптера для LLM.

## Архитектура

```
ExperimentSpec → Orchestrator → [Dataset → Config → HPO → Training → Eval → Artifact] × N → Best Trial
```

**6 sub-skills**, вызываемых оркестратором в цикле:

| Skill | GPU | Задача |
|-------|-----|--------|
| `DatasetSkill` | ❌ | Анализ датасета, token stats, format detection |
| `ConfigSkill` | ❌ | Генерация LoRA-конфигов, VRAM-оценка |
| `HPOSkill` | ❌ | Bayesian/random/bandit search, pruning |
| `TrainingSkill` | ✅ | LoRA/QLoRA обучение (transformers + peft + trl) |
| `EvalSkill` | ✅ | Loss, perplexity, delta vs base model |
| `ArtifactSkill` | ❌ | Версионирование адаптеров, SHA256 |

## Быстрый старт

```python
from skills.lora_trainer.handler import LoRATrainerSkill
from skills.lora_trainer.schema import LoRATrainerInput
from skills.lora_trainer.experiment_spec import ExperimentSpec

skill = LoRATrainerSkill(output_dir="./outputs")

result = await skill.execute(
    LoRATrainerInput(
        experiment=ExperimentSpec(
            goal="Fine-tune for instruction following",
            dataset_path="data/train.jsonl",
            base_model="unsloth/Llama-3.2-1B",
            max_trials=10,
            budget_gpu_hours=5.0,
        ),
        dry_run=True,  # без GPU
    ),
    state,
)
```

## PEFT методы

- **LoRA** — Low-Rank Adaptation (q_proj, v_proj)
- **QLoRA** — 4-bit quantized LoRA (NF4 + double quant)
- **Prefix Tuning** — virtual token prefixes
- **IA³** — Infused Adapter by Inhibiting and Amplifying

## HPO стратегии

- `random` — случайный поиск (первые 3 trial)
- `bayesian` — perturbation лучшего конфига
- `bandit` — epsilon-greedy (explore → exploit)
- `halving` — successive halving

## Зависимости

Для dry-run: только `pydantic`. Для GPU-обучения:
```
pip install torch transformers peft trl datasets bitsandbytes
```

## Тесты

```bash
python -m pytest tests/test_lora_trainer.py -v
```
