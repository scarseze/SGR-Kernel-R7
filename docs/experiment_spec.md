# Experiment DSL Specification

**File**: `docs/experiment_spec.md`  
**Purpose**: Definition of the YAML/JSON language used to define LoRA experiments.

---

## 1. Experiment YAML Structure

Experiments are defined in a declarative configuration file.

```yaml
version: "1.0"
experiment_name: "llama3-sql-finetune"
dataset:
  path: "data/text_to_sql.json"
  format: "alpaca"

# Search Space (for HPO) or Fixed Config
config:
  base_model: "meta-llama/Meta-Llama-3-8B"
  method: "qlora"
  
  # Hyperparameters
  learning_rate: 2e-4
  batch_size: 4
  epochs: 3
  lora_rank: 64
  lora_alpha: 16
  
  # Advanced
  target_modules: ["q_proj", "v_proj"]
  quantization_bits: 4

protocol:
  type: "standard" # or "ablation", "curriculum"
  budget_gpu_hours: 10.0
  trials: 5

output:
  export_format: "gguf"
  merge_after: true
```

## 2. Core Objects

### TrialConfig
The concrete configuration for a single training run.
*   `trial_id`: Integer ID.
*   `seed`: Random seed for reproducibility.
*   `gradient_accumulation`: Calculated from effective batch size.

### SearchSpace (HPO)
Defines ranges for optimization.
*   `lr_min` / `lr_max`: Log-uniform range.
*   `rank_choices`: Categorical `[16, 32, 64]`.

## 3. Protocol Flags

*   **Standard**: Run N independent trials (Random/Bayesian Search).
*   **Ablation**: Systematically vary one parameter (e.g., `rank`) while fixing others.
*   **Curriculum**: Train on easy data first, then hard.
*   **Ensemble**: Train multiple diverse adapters to merge later.

## 4. Budget Model
Controls resource usage to prevent runaway costs.
*   `budget_gpu_hours`: Max aggregate GPU time (e.g., 10h).
*   `budget_dollars`: Est. cost limit (requires cost metadata).
*   `early_stop_loss`: Stop if loss > threshold (Divergence protection).

---
---

# Спецификация Экспериментов (DSL)

**Файл**: `docs/experiment_spec.md`

## 1. Структура YAML
Эксперименты описываются декларативно.

## 2. Основные Объекты
*   **TrialConfig**: Конфиг конкретного прогона.
    *   `seed`: Зерно для воспроизводимости.
*   **SearchSpace (HPO)**: Пространство поиска гиперпараметров.

## 3. Протоколы
*   **Standard**: N независимых попыток.
*   **Ablation**: Систематический перебор одного параметра.
*   **Curriculum**: Обучение от простого к сложному.
*   **Ensemble**: Обучение ансамбля моделей.

## 4. Модель Бюджета
Контроль ресурсов.
*   `budget_gpu_hours`: Лимит часов GPU.
*   `early_stop_loss`: Остановка при расхождении лосса (NaN/Inf).

