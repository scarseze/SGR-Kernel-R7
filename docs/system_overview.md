# System Overview

**Project**: SGR Kernel  
**Version**: 1.0.0-rc1 (Stable v1.x)  
**Status**: Production-Ready  
**License**: MIT  

```text
┌──────────────────────────────────────────────┐
│                  USER / CLI                  │
│        experiment.yaml / run command         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                 AGENT KERNEL                 │
│                                              │
│  • Step Executor                             │
│  • Skill Router                              │
│  • State Store                               │
│  • WAL Trace                                 │
│  • Retry / Timeout                           │
└───────────────┬──────────────────────────────┘
                │ skill call
                ▼
┌──────────────────────────────────────────────┐
│                 SKILL LAYER                  │
│                                              │
│ dataset_skill                                │
│ config_skill                                 │
│ training_skill                               │
│ eval_skill                                   │
│ hpo_skill                                    │
│ merge_skill                                  │
│ artifact_skill                               │
│ protocols/*                                  │
└───────────────┬──────────────────────────────┘
                │ job spec
                ▼
┌──────────────────────────────────────────────┐
│                DISPATCHER                    │
│                                              │
│  LocalDispatcher                             │
│  SSHDispatcher                               │
│  (extensible)                                │
└───────────────┬──────────────────────────────┘
                │
      ┌─────────┴─────────┐
      ▼                   ▼
┌───────────────┐   ┌────────────────┐
│ GPU WORKER    │   │ EVAL WORKER    │
│ training run  │   │ eval suite     │
└───────┬───────┘   └───────┬────────┘
        │                   │
        └──────────┬────────┘
                   ▼
┌──────────────────────────────────────────────┐
│            ARTIFACT & METRIC STORE           │
│                                              │
│ adapters/                                    │
│ metrics/                                     │
│ manifests/                                   │
│ traces/ (WAL)                                │
│ reports/                                     │
└──────────────────────────────────────────────┘
```

## 🎯 Purpose (Зачем это нужно)

The **SGR Kernel** is a specialized Agentic Operating System designed for **automated AI research and engineering**. It bridges the gap between high-level reasoning (LLMs) and low-level execution (training, coding, deployment).

### Primary Goals
1.  **Autonomous ML Engineering**: End-to-end automation of model training, fine-tuning (LoRA/PEFT), and evaluation.
2.  **Reproducible Science**: Ensuring every experiment is tracked, deterministic, and reproducible via strict manifests.
3.  **Enterprise Security**: Sandboxing code execution and enforcing strict policy/budget constraints.
4.  **Scalability**: Distributed execution of heavy workloads (training, rendering, deep research) across remote workers.

### Target Audience
*   **AI Researchers**: To run ablation studies, curriculum learning, and ensemble experiments without boilerplate.
*   **LLM Engineers**: To fine-tune models on custom data with production-grade monitoring and artifacts.
*   **Agent Developers**: To build complex autonomous workflows using the Kernel's DAG and Skill primitives.

---

## 🧱 System Scope

### ✅ What is in Scope (Что входит)
*   **Kernel Core**: DAG orchestration, state management, tracing, policy enforcement.
*   **Skill Ecosystem**: Modular tools for Coding (Docker), Training (PEFT), Research, and Analysis.
*   **PEFT Lab**: Specialized stack for LoRA training, merging, and evaluation.
*   **Distributed Layer**: Unified dispatcher for remote job execution (SSH, Cloud).
*   **Observability**: Full trace history (WAL), metric dashboards, and cost tracking.

### ❌ What is NOT in Scope (Что НЕ входит)
*   **General Purpose OS**: It is not a replacement for Linux/Windows; it runs *on top* of them.
*   **Web UI**: The kernel provides APIs and artifacts; it is UI-agnostic (though includes a CLI dashboard).
*   **Raw Hardware Management**: It relies on existing drivers (CUDA) and platforms (Docker, SSH); it does not manage bare metal directly.

---

## 🧩 Subsystems

### 1. Agent Kernel (`core/`)
The brain of the system. Responsibilities:
*   **Dispatcher**: Unified transport for remote execution.
*   **Lifecycle**: Formal 7-phase step execution.
*   **Reliability**: Semantic failure engine and recovery.
*   **Replay**: Deterministic record/replay inter-layer.
*   **Artifact Store**: Content-addressed immutability.

### 2. Skill Layer (`skills/`)
The hands of the system. Key modules:
*   `code_interpreter`: Sandboxed Python execution.
*   `lora_trainer`: Fine-tuning orchestration.
*   `research_agent`: Deep web research.
*   `file_system`: Workspace management.

### 3. Dispatcher (`core/dispatcher.py`)
The nervous system for remote actions.
*   Abstracts **where** code runs (Local vs SSH vs Cloud).
*   Handles job submission, polling, and result collection.
*   Ensures asynchronous execution of long-running tasks.

### 4. PEFT Lab (`skills/lora_trainer/`)
A specialized research environment embedded in the kernel.
*   **TrainingSkill**: QLoRA/LoRA fine-tuning.
*   **HPOSkill**: Hyperparameter optimization (Optuna-like).
*   **MergeSkill**: Adapter fusion and export (GGUF, Safetensors).

### 5. Research Protocols (`protocols/`)
Higher-order logic for scientific rigor.
*   `ablation`: Systematic parameter sweeping.
*   `curriculum`: Difficulty-based training stages.
*   `ensemble`: Diversity-driven model training.

### 6. Reproducibility (`reproducibility.py`)
The "Black Box" recorder.
*   Captures environment snapshots (pip, cuda).
*   Hashes all inputs (data, config) and outputs (models).
*   Generates `manifest.json` for every experiment.

---

## 🔁 Execution Flow (Как это работает)

1.  **Request**: User sends a goal (e.g., "Fine-tune Llama 3 on dataset X").
2.  **Planning**: Kernel generates a `ExecutionPlan` (DAG of steps).
3.  **Orchestration**:
    *   Kernel executes steps sequentially or in parallel.
    *   **Trace**: Every step is logged to `RequestTrace`.
    *   **Checkpoint**: State is saved to disk (WAL) after each step.
4.  **Skill Execution**:
    *   If `remote=True`, Dispatcher sends job to Worker.
    *   If `job`, Worker runs training/analysis.
5.  **Artifacts**: Results (adapters, logs, metrics) are stored in `artifacts/`.
6.  **Loop**: Kernel replans if errors occur (Self-Correction).

---


---
---

# Обзор Системы (System Overview)

**Проект**: SGR Kernel  
**Версия**: 1.2 (Enterprise/Research)  
**Статус**: Production-Ready  

## 🎯 Назначение (Purpose)

**SGR Kernel** — это специализированная Агентная Операционная Система, разработанная для **автоматизированных научных исследований и ML-инжиниринга**. Она устраняет разрыв между высокоуровневыми рассуждениями (LLM) и низкоуровневым исполнением (тренировка, кодинг, деплой).

### Основные Цели
1.  **Автономный ML-инжиниринг**: End-to-end автоматизация тренировки моделей, файн-тюнинга (LoRA/PEFT) и оценки.
2.  **Воспроизводимая Наука**: Гарантия того, что каждый эксперимент отслеживается, детерминирован и воспроизводим через строгие манифесты.
3.  **Enterprise Безопасность**: Изоляция исполнения кода (Sandbox) и соблюдение строгих политик/бюджетов.
4.  **Масштабируемость**: Распределенное выполнение тяжелых задач (тренировка, рендеринг) на удаленных воркерах.

---

## 🧱 Границы Системы (System Scope)

### ✅ Что Входит (In Scope)
*   **Ядро (Kernel Core)**: Оркестрация DAG, управление состоянием, трейсинг, политики.
*   **Экосистема Скиллов**: Модульные инструменты для Кодинга, Тренировки, Ресерча.
*   **PEFT Lab**: Специализированный стек для LoRA тренировок и мержинга.
*   **Распределенный Слой**: Унифицированный диспетчер для удаленных задач.
*   **Наблюдаемость**: Полная история трейсов (WAL), дашборды, метрики.

### ❌ Что НЕ Входит (Out of Scope)
*   **OS Общего Назначения**: Это не замена Linux/Windows; работает поверх них.
*   **Web UI**: Ядро предоставляет API и артефакты; не зависит от UI.
*   **Управление Железом**: Полагается на существующие драйверы (CUDA) и платформы.

---

## 🧩 Подсистемы (Subsystems)

### 1. Agent Kernel (`core/`)
Мозг системы.
*   **Planner**: Превращает запросы в план (DAG).
*   **Executor**: Запускает скиллы, обрабатывает ретраи.
*   **Memory**: Управляет контекстом (Vector DB/SQL).
*   **Middleware**: Безопасность, Политики, Апрувы.

### 2. Skill Layer (`skills/`)
Руки системы.
*   `code_interpreter`: Python в песочнице.
*   `lora_trainer`: Оркестрация файн-тюнинга.
*   `file_system`: Работа с файлами.

### 3. Dispatcher (`core/dispatcher.py`)
Нервная система для удаленных действий.
*   Абстрагирует **где** выполняется код (Local vs SSH vs Cloud).
*   Управляет отправкой задач и сбором результатов.

### 4. PEFT Lab (`skills/lora_trainer/`)
Встроенная исследовательская среда.
*   **TrainingSkill**: QLoRA/LoRA.
*   **HPOSkill**: Оптимизация гиперпараметров.
*   **MergeSkill**: Слияние адаптеров.

### 5. Research Protocols (`protocols/`)
Высокоуровневая логика для научной строгости.
*   `ablation`: Абаляции (sweep).
*   `curriculum`: Обучение по сложности.
*   `ensemble`: Ансамблирование.

### 6. Reproducibility (`reproducibility.py`)
"Черный ящик" самописца.
*   Снимки окружения (pip, cuda).
*   Хеширование всех входов и выходов.
*   Генерация `manifest.json`.

---

## 🔁 Поток Выполнения (Execution Flow)

1.  **Request**: Пользователь ставит цель (напр. "Файн-тюн Llama 3").
2.  **Planning**: Кернел генерирует `ExecutionPlan` (DAG шагов).
3.  **Orchestration**:
    *   Кернел выполняет шаги.
    *   **Trace**: Каждый шаг логируется.
    *   **Checkpoint**: Состояние сохраняется на диск (WAL).
4.  **Skill Execution**:
    *   Если `remote=True`, Диспетчер шлет задачу Воркеру.
5.  **Artifacts**: Результаты сохраняются в `artifacts/`.
6.  **Loop**: Кернел перепланирует при ошибках (Self-Correction).

