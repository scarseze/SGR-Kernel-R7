# Design Decisions Log

**File**: `docs/design_decisions.md`  
**Purpose**: Record of architectural choices, trade-offs, and rationale.

---

## 001. Dispatcher Moved to Kernel Core

*   **Date**: 2026-02-16
*   **Status**: Accepted

### Decision
Promote the `Dispatcher` logic (originally in `skills/lora_trainer`) to a first-class citizen in `core/dispatcher.py`.

### Why
We realized that distributed execution is a generic requirement, not specific to LoRA training. Other skills like `DeepResearch` (scraping), `VideoGen`, or `HeavyBacktest` also need offloading to remote workers.

### Trade-offs
*   **Pros**:
    *   **Reuse**: Write dispatch logic once, use in any skill.
    *   **Standardization**: All remote jobs follow `RemoteJobSpec`.
    *   **Central Control**: Easier to monitor global resource usage.
*   **Cons**:
    *   **Complexity**: The Kernel core becomes slightly larger.
    *   **Coupling**: Skills depend on core dispatcher interfaces.

### Alternatives Rejected
*   **Skill-Level Dispatcher**: Leaving it inside `lora_trainer`.
    *   *Reason*: Leads to code duplication if we add another heavy skill later.

---

## 002. Trace as WAL (Write-Ahead Log)

*   **Date**: 2026-02-16
*   **Status**: Accepted

### Decision
Use the `RequestTrace` JSON files as the definitive source of truth for persistence and recovery. Save after every step.

### Why
We needed a way to recover from crashes without implementing a complex database schema for DAG state. Traces already captureinputs, outputs, and status.

### Trade-offs
*   **Pros**:
    *   **Simplicity**: No extra DB tables for "Job State".
    *   **Observability**: The log is human-readable.
*   **Cons**:
    *   **IO Overhead**: Writing full JSON after every step scales poorly with very large histories. *Mitigation: Append-only logs planned for v2.*

---

## 003. Docker Sandboxing for Code Interpreter

*   **Date**: 2026-02-16
*   **Status**: Accepted

### Decision
Use Docker containers (`sgr-sandbox`) for executing Python code, rather than local `exec()`.

### Why
Safety is paramount. LLM-generated code is untrusted and could destroy the host filesystem (`rm -rf /`) or exfiltrate credentials.

### Trade-offs
*   **Pros**:
    *   **Isolation**: File system and network are fenced.
    *   **Resource Control**: Can limit CPU/RAM.
*   **Cons**:
    *   **Latency**: Container startup takes time (mitigated by keeping it warm).
    *   **Dependency**: Requires Docker daemon on host.

---

## 004. Mermaid as Primary Diagramming Tool

*   **Date**: 2026-02-16
*   **Status**: Accepted

### Decision
Use Mermaid.js inside Markdown for all architectural diagrams.

### Why
 Diagrams as Code. They are version-controllable, diff-able, and easier to update than binary PNGs.

### Trade-offs
*   **Visuals**: Less pretty than Figma/Visio.
*   **Rendering**: Depends on GitHub/IDE support (which is good now).

---
---

# Журнал Архитектурных Решений (ADR)

**Файл**: `docs/design_decisions.md`

## 001. Dispatcher перенесен в Ядро (Kernel Core)
*   **Решение**: Перенести логику диспетчера из `lora_trainer` в `core/dispatcher.py`.
*   **Почему**: Распределенное выполнение нужно многим скиллам (Research, Render), не только LoRA.
*   **Trade-off**: Увеличивает сложность ядра, но дает переиспользование кода.

## 002. Trace как WAL (Write-Ahead Log)
*   **Решение**: Использовать JSON файлы трейсов как источник истины для персистентности.
*   **Почему**: Проще, чем поддерживать отдельную БД для состояния DAG.
*   **Trade-off**: IO нагрузка (пишем полный JSON каждый шаг), но зато надежно и читаемо.

## 003. Docker Sandbox для Кодинга
*   **Решение**: Использовать Docker контейнеры вместо локального `exec()`.
*   **Почему**: Безопасность критична. LLM код ненадежен.
*   **Trade-off**: Латенси старта контейнера, зависимость от демона Docker.

---

## 005. Modular Engine Refactor (RFC v2)

*   **Date**: 2026-02-18
*   **Status**: Accepted

### Decision
Surgically split the monolithic `engine.py` (40KB) into a multi-layered modular architecture: `LifecycleEngine`, `ReliabilityEngine`, `ReplayEngine`, and `CheckpointManager`.

### Why
The monolithic engine was becoming impossible to test and maintain. Implementing RFC v2 requirements (strict phases, deterministic replay) required clearly defined interfaces between the state machine and the execution logic.

### Trade-offs
*   **Pros**:
    *   **Testability**: Engines can be unit-tested in isolation (fakes/mocks).
    *   **Extensibility**: Strategic plug-ins (e.g., custom reliability policies) are now possible.
*   **Cons**:
    *   **Indirection**: Trace-through of a single step now crosses multiple files.
