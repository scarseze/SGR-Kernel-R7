# Architecture

## High-Level Agent Architecture

```mermaid
graph TD
    User[User / Client] -->|Request| Kernel[Agent Kernel]
    
    subgraph CoreEngine [Core Engine]
        Planner[Planner Agent]
        Lifecycle[Lifecycle Engine]
        Reliability[Reliability Engine]
        Storage_Ctrl[Artifact & Checkpoint Ctrl]
    end
    
    Kernel --> CoreEngine
    
    CoreEngine -->|Select| Router{Skill Router}
    
    subgraph Skills [Skill Ecosystem]
        Code[Code Interpreter]
        Research[Deep Research]
        FS[File System]
        LoRA[PEFT Lab]
    end
    
    Router --> Skills
    
    subgraph Execution [Execution Layer]
        Dispatcher[Unified Dispatcher]
        Local[Local Worker]
        Remote[SSH / Cloud Worker]
    end
    
    Skills -->|Job Spec| Dispatcher
    Dispatcher -->|In-Process| Local
    Dispatcher -->|Async| Remote
    
    subgraph Storage [Persistence Layer]
        DB[(SQL / Vector DB)]
        Traces[Trace WAL]
        Artifacts[Artifact Store]
    end
    
    CoreEngine --> DB
    CoreEngine --> Traces
    Remote --> Artifacts
    Local --> Artifacts
```

## PEFT Lab Skill Stack (LoRA Trainer)

Detailed view of the `lora_trainer` subsystem.

```mermaid
graph LR
    Input[User Goal] --> Orchestrator[LoRA Orchestrator]
    
    subgraph Preparation
        Dataset[Dataset Skill]
        Config[Config Skill]
    end
    
    subgraph Optimization Loop
        HPO[HPO Skill]
        Training[Training Skill]
        Eval[Eval Skill]
    end
    
    subgraph Finalization
        Merge[Merge Skill]
        Artifact[Artifact Skill]
    end
    
    Orchestrator --> Preparation
    Preparation --> OptimizationLoop
    OptimizationLoop --> Finalization
    
    Training -->|Dispatch| Dispatcher[Core Dispatcher]
    Dispatcher -->|Train| GPU[GPU Worker]
    GPU -->|Metrics| Trace[Trace Manager]
```


## ASCII: LoRA / PEFT Research Flow

```text
experiment.yaml
     │
     ▼
dataset_skill.analyze
     │
     ▼
config_skill.build_search_space
     │
     ▼
FOR trial in search_space:
    │
    ├── training_skill.run
    │       │
    │       ▼
    │   adapter checkpoint
    │
    ├── eval_skill.run
    │       │
    │       ▼
    │   metrics.json
    │
    └── artifact_skill.store

     │
     ▼
hpo_skill.select_best
     │
     ▼
merge_skill (optional)
     │
     ▼
report + manifest + reproducibility verify
```

---

# Архитектура (Architecture)

## Высокоуровневая Архитектура Агента

*   **Core Engine**: Мозг. Планирует, контролирует жизненный цикл и надежность.
*   **Skill Router**: Маршрутизатор. Выбирает нужный инструмент (Capability-aware).
*   **Skills**: Инструменты (Код, Ресерч, ML).
*   **Dispatcher**: Транспорт. Отправляет задачи на Local/SSH/Cloud.
*   **Execution Layer**: Воркеры, где происходит реальная работа.
*   **Persistence**: База знаний и логов. БД, Трейсы, Артефакты.

## Стек PEFT Lab (LoRA Trainer)

Детальный вид подсистемы `lora_trainer`.
*   **Orchestrator**: Управляет циклом обучения.
*   **Preparation**: Подготовка данных и конфигов.
*   **Optimization Loop**: Цикл HPO -> Train -> Eval.
*   **Finalization**: Слияние весов (Merge) и экспорт.

