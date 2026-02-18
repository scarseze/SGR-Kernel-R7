# Research Protocol Specification

**File**: `docs/research_protocols.md`  
**Purpose**: Definitions of advanced experimental methodologies supported by the kernel.

---

## 1. Ablation Protocol
**Goal**: Isolate the impact of specific hyperparameters.

*   **Logic**:
    1.  Define a `baseline` config.
    2.  Select a `target` param (e.g., `lora_rank`).
    3.  Define `values` to sweep (e.g., `[8, 16, 32, 64]`).
    4.  Generate N trials where only `target` changes.
*   **Output**: Comparative plot of `target` vs `loss`.

## 2. Curriculum Protocol
**Goal**: Improve convergence by feeding data in order of difficulty.

*   **Logic**:
    1.  **Stage 1**: Train on short/simple samples (length < 512).
    2.  **Stage 2**: Train on medium samples.
    3.  **Stage 3**: Train on full dataset (length < 4096).
*   **Mechanism**: Uses `TrainingSkill` checkpointing to pass `adapter` from Stage 1 $\to$ Stage 2.

## 3. Ensemble Protocol
**Goal**: leverage "Wisdom of Crowds" (MoE-lite).

*   **Logic**:
    1.  Train N adapters with different `seeds` or `architectures` (rank/alpha).
    2.  **Merge Strategy**:
        *   `Uniform`: Average weights.
        *   `Linear`: Weighted by validation accuracy.
        *   `TIES`: Trimming + Electing signs (advanced merging).
*   **Result**: A single robust merged model.

## 4. Metric Comparability
To ensure valid science:
*   All protocols enforce the **Same Validation Set** across all trials.
*   Loss is normalized by token count.

---
---

# Спецификация Научных Протоколов

**Файл**: `docs/research_protocols.md`

## 1. Протокол Абаляции (Ablation)
**Цель**: Изолировать влияние конкретного гиперпараметра.
*   Логика: Фиксируем все параметры, меняем один (Target).
*   Выход: График `Target` vs `Loss`.

## 2. Протокол Curriculum (Обучение по сложности)
**Цель**: Улучшить сходимость.
*   Этап 1: Короткие примеры.
*   Этап 2: Полный датасет.
*   Механизм: Checkpointing.

## 3. Протокол Ансамблирования (Ensemble)
**Цель**: "Мудрость толпы".
*   Тренируем N адаптеров с разными сидами.
*   Сливаем (Merge) их веса.

## 4. Сравнимость Метрик
Для научной валидности:
*   Обязательно использование **Одного Валидационного Сета** во всех трайлах.
*   Лосс нормализуется по количеству токенов.

