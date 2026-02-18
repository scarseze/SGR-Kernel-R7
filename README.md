# SGR Kernel (Agentic OS) 🧠

> **Enterprise-Grade Agentic Kernel for Automated Research & Engineering**

![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Coverage](https://img.shields.io/badge/Tests-46%20Passing-brightgreen)

> [!WARNING]
> **Release Candidate (v1.0.0-rc1)**
> This is a stable release candidate for the v1.x series.
> *   **Production Policy**: Use allowed with supervision.
> *   **API Stability**: Public API (`CoreEngine`, `Skill`) is stable. Internal implementations (`_fsm_impl`) may evolve.
> *   **Feedback**: Please report issues to the core team.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/badge/Release-v1.0.0--rc1-green)](https://github.com/sgr/kernel/releases)

**Agentic Operating System for Automated Machine Learning Research**
> Kernel runtime is stable.
> Plugin ABI may evolve in v1.x minors.

## 📖 Documentation Index (Artifact Pack)

| Document | Purpose | Audience |
| :--- | :--- | :--- |
| **[Standard Overview](docs/system_overview.md)** | 🎯 Purpose, Scope, Subsystems | All |
| **[Architecture](docs/architecture.md)** | 🧩 Diagrams & Component Flow | Engineers |
| **[Execution Model](docs/execution_model.md)** | ⚙️ Deterministic FSM & Core Loop | Core Devs |
| **[Skill Development](docs/skill_development.md)** | 🤝 Interface, Context, & Registration | Skill Devs |
| **[Security Model](docs/security_model.md)** | 🛡️ ACLs, Capabilities, & Sandboxing | Security |
| **[Reliability Engine](docs/reliability.md)** | 💥 Fault Classification & Recovery | SREs |
| **[Replay Model](docs/replay_model.md)** | 📼 Deterministic Replay & Tapes | Engineers |
| **[Artifact Store](docs/artifact_store.md)** | 📦 Content-Addressed Storage (CAS) | SREs |
| **[Lifecycle State](docs/lifecycle.md)** | 🔄 7-Phase Execution Workflow | Core Devs |
| **[Experiment Spec](docs/experiment_spec.md)** | 🧪 DSL & Config Reference | Researchers |
| **[Deployment Guide](docs/deployment.md)** | 🚀 Setup & Production Staging | DevOps |
| **[Reproducibility](docs/reproducibility.md)** | 🔬 Manifests & Hashing | Scientists |
| **[Charts](docs/diagrams/)** | 📊 Editable Draw.io files | Architects |


---

## 🏗️ Architecture High-Level

The SGR Kernel acts as an operating system for AI Agents. It provides:
1.  **Orchestration**: DAG-based planning and execution.
2.  **Safety**: Sandboxed code execution (Docker) and policy enforcement.
3.  **Observability**: Full trace history (WAL) and metrics.
4.  **Distribution**: Unified dispatching of heavy jobs (Training, Rendering).

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run Tests
pytest tests/

# 3. Start Kernel
python main.py
```

## 🧯 Safety & Responsibility
> **Warning**: SGR Kernel is an agent execution runtime. Skills may cause real-world side effects.
> *   Always use **Capability Enforcement** (ACLs).
> *   Enable **Approval Gates** for sensitive actions (WRITE, DELETE).
> *   Run in **Dockerized Sandboxes** whenever possible.

## ⚖️ Guarantees
*   **Idempotency**: Safe to retry.
*   **Reproducibility**: `manifest.json` guarantees exact replay.
*   **Security**: No code runs outside the sandbox.

## 🛡️ Compatibility / Совместимость
*   **v1.x**: Backward compatible API (Strict Semantic Versioning).
*   **v2.0**: Reserved only for breaking changes in the **Execution Model**.

---
*Built by SGR Team | 2026*

---

# Russian Section / Русская Секция 🇷🇺

## 📖 Индекс Документации

| Документ | Назначение | Аудитория |
| :--- | :--- | :--- |
| **[Обзор Системы](docs/system_overview.md)** | 🎯 Цели, Границы, Подсистемы | Все |
| **[Архитектура](docs/architecture.md)** | 🧩 Диаграммы и Потоки | Инженеры |
| **[Модель Исполнения](docs/execution_model.md)** | ⚙️ FSM и Основной Цикл | Core Devs |
| **[Разработка Скиллов](docs/skill_development.md)** | 🤝 Интерфейс, Контекст, Регистрация | Skill Devs |
| **[Модель Безопасности](docs/security_model.md)** | 🛡️ ACL, Права и Песочницы | Security |
| **[Reliability Engine](docs/reliability.md)** | 💥 Классификация сбоев и Ретраи | SRE |
| **[Replay Model](docs/replay_model.md)** | 📼 Детерминированный Реплей | Engineers |
| **[Хранилище Артефактов](docs/artifact_store.md)** | 📦 Content-Addressed Storage (CAS) | SRE |
| **[Жизненный Цикл](docs/lifecycle.md)** | 🔄 7 Фаз Выполнения Шага | Core Devs |
| **[Спецификация Эксперимента](docs/experiment_spec.md)** | 🧪 DSL и Конфиги | Researchers |
| **[Руководство по Развертыванию](docs/deployment.md)** | 🚀 Установка и Production | DevOps |
| **[Воспроизводимость](docs/reproducibility.md)** | 🔬 Манифесты и Хеширование | Ученые |
| **[Диаграммы](docs/diagrams/)** | 📊 Исходники Draw.io (.xml) | Архитекторы |


## 🏗️ Архитектура (Кратко)
SGR Kernel — это операционная система для AI Агентов.
1.  **Оркестрация**: Планирование DAG.
2.  **Безопасность**: Изоляция кода в Docker.
3.  **Наблюдаемость**: Полная история (WAL).
4.  **Распределение**: Унифицированный диспетчер.

