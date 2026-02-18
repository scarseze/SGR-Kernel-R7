# Reproducibility Specification

**File**: `docs/reproducibility.md`  
**Purpose**: Ensuring scientific rigor and replayability of experiments.

---

## 1. The Reproducibility Crisis
In ML, "it works on my machine" is unacceptable. The Kernel guarantees that any experiment can be verified and re-run.

## 2. The Manifest (`manifest.json`)
Every experiment produces a signed manifest containing all necessary context.

```json
{
  "experiment_id": "exp_12345",
  "timestamp": "2026-02-16T12:00:00",
  "environment": {
    "python_version": "3.10.12",
    "torch_version": "2.1.0+cu121",
    "cuda_device": "NVIDIA A100-SXM4-40GB",
    "requirements_hash": "sha256:..."
  },
  "inputs": {
    "dataset_hash": "sha256:a1b2...",
    "config_hash": "sha256:c3d4...",
    "base_model": "meta-llama/Llama-3-8B"
  },
  "outputs": {
    "adapter_hash": "sha256:e5f6...",
    "metrics_final": {"loss": 0.45},
    "seed_used": 42
  }
}
```

## 3. Hashing Strategy
*   **Datasets**: SHA256 of the raw data file.
*   **Models**: Hash of the adapter weights (`adapter_model.bin` / `safetensors`).
*   **Config**: deterministic serialization of the JSON config.

## 4. Environment Capture
At start, `reproducibility.py` captures:
*   `pip freeze`: Full library list.
*   `nvidia-smi`: GPU hardware details.
*   `git commit`: Current kernel version.

## 5. Verification Flow
command: `sgr verify --manifest experiments/exp_123/manifest.json`

1.  Load manifest.
2.  Check if current env matches `environment`.
3.  Check if input data matches `dataset_hash`.
4.  (Optional) Re-run training with `seed_used`.
5.  Compare new output hash with `adapter_hash`.
    *   *Match* = **Reproduced**.
    *   *Mismatch* = **Failed**.

---
---

# Спецификация Воспроизводимости

**Файл**: `docs/reproducibility.md`

## 1. Кризис Воспроизводимости
"Работает на моем ноутбуке" — неприемлемо. Мы гарантируем, что любой эксперимент можно проверить и повторить.

## 2. Манифест (`manifest.json`)
Каждый запуск создает подписанный манифест:
*   **Environment**: Версии Python, Torch, драйверы CUDA.
*   **Inputs**: Хеши датасетов, конфигов, базовых моделей.
*   **Outputs**: Хеши весов, итоговые метрики, использованный seed.

## 3. Стратегия Хеширования
*   **Dataset**: SHA256 сырого файла.
*   **Model**: Хеш весов адаптера.
*   **Config**: Детерминированная сериализация JSON.

## 4. Захват Окружения
При старте `reproducibility.py` фиксирует `pip freeze`, `nvidia-smi`, `git commit`.

## 5. Процесс Верификации
Команда: `sgr verify --manifest ...`
1.  Загрузить манифест.
2.  Сравнить текущее окружение.
3.  Сравнить хеши входных данных.
4.  (Опционально) Перезапустить обучение с тем же `seed`.
5.  Сравнить хеш выходного адаптера.

