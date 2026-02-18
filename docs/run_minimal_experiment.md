# Minimal Reproduction Guide

**File**: `docs/run_minimal_experiment.md`  
**Goal**: Validate the system works in < 5 minutes.

---

## Prerequisites
*   Python 3.10+
*   Docker Desktop (running)
*   1x GPU (NVIDIA, 8GB+) *Optional for Dry Run*

## 1. Install
```bash
git clone https://github.com/macht/sgr_kernel
cd sgr_kernel
pip install -r requirements.txt
```

## 2. Prepare Data
Create a dummy dataset:
```python
# data/dummy.json
[{"instruction": "Hi", "output": "Hello!"}] * 100
```

## 3. Write Spec
Create `experiment.yaml`:
```yaml
experiment_name: "test_run"
dataset: 
  path: "data/dummy.json"
config:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  method: "qlora"
  epochs: 1
protocol:
  trials: 1
```

## 4. Run Experiment (Dry Run First)
Verify without GPU usage:
```bash
python main.py --experiment experiment.yaml --dry-run
```
*Expected*: "Dry Run Completed. ~0.01 GPU hours estimated."

## 5. Run for Real
```bash
python main.py --experiment experiment.yaml
```
*Expected*:
*   Status: `COMPLETED`
*   Artifacts: `artifacts/test_run/trial_1/adapter/`

## 6. Inspect & Verify
Check the manifest:
```bash
cat artifacts/test_run/manifest.json
```
Verify reproducibility:
```bash
python -m sgr verify artifacts/test_run/manifest.json
```
*Result*: `[SUCCESS] Experiment is reproducible.`

---
---

# Гайд по Минимальному Запуску

**Файл**: `docs/run_minimal_experiment.md`

## Пререквизиты
*   Python 3.10+
*   Docker Desktop
*   GPU (Optional для Dry Run)

## 1. Установка
```bash
pip install -r requirements.txt
```

## 2. Подготовка Данных
Создайте `data/dummy.json`: `[{"instruction": "Hi", "output": "Hello!"}] * 100`

## 3. Спецификация
Создайте `experiment.yaml`. См. пример выше.

## 4. Dry Run (Проверка)
```bash
python main.py --experiment experiment.yaml --dry-run
```

## 5. Запуск
```bash
python main.py --experiment experiment.yaml
```

## 6. Верификация
```bash
python -m sgr verify artifacts/test_run/manifest.json
```

