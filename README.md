# SGR Kernel (Schema-Guided Reasoning Kernel)

**An Enterprise-grade Operating System for Autonomous AI Agents.**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Architecture](https://img.shields.io/badge/Architecture-Kernel--Based-orange) ![Security](https://img.shields.io/badge/Security-Ring--0-red) ![Status](https://img.shields.io/badge/Status-Hardened_%28R7%29-green)

---

## 🚀 Overview

**SGR Kernel** is not just another agent framework. It is a **runtime environment** designed to orchestrate Large Language Models (LLMs) with the same rigor as an operating system orchestrates processes.

It abstracts away the interaction with LLMs, Vector DBs, and Docker containers, providing a stable, secure, and observable foundation for building complex reasoning applications ("Skills").

**Key Philosophy:**
*   **Kernel vs. User Space**: The Core handles scheduling, security, and memory. Skills handle business logic.
*   **Defense in Depth**: Every skill execution passes through a rigorous Middleware Ring (Policy, Timeout, Approval).
*   **Crash Safety**: A failed skill does not crash the agent. The Kernel catches exceptions, enforces budgets, and triggers replan loops.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "External World"
        User[User / Client]
        Telegram[Telegram Bot]
        API[REST API]
    end

    subgraph "SGR Kernel (The Runtime)"
        direction TB
        
        Scheduler[DAG Scheduler & Replan Loop]
        
        subgraph "Security Ring (Middleware)"
            Policy[Policy Engine]
            Firewall[Input/Output Validator]
            Timeout[Timeout Authority]
        end
        
        Memory[Trace System & Context]
        Resources[Budget & Concurrency Guard]
    end

    subgraph "User Space (Skills)"
        Portfolio[Finance Analyst]
        RAG[Deep Research / RAG]
        LogicRL[Logic-RL Solver]
        Sandbox[Code Interpreter (Docker)]
    end

    User --> Telegram
    Telegram --> Firewall
    Firewall --> Scheduler
    
    Scheduler -->|Orchestrate| Policy
    Policy -->|Execute| Skills
    
    Skills -->|Syscall| Memory
    Skills -->|Lock/Cost| Resources
```

---

## 🛡️ Kernel Services

### 1. Process Management (The Scheduler)
*   **DAG Execution**: Supports complex dependency graphs, not just linear chains.
*   **Replan Loop**: Automatically detects execution failures and triggers localized replanning (Self-Healing).
*   **Concurrency Guard (R7)**: Semaphores prevent resource exhaustion by limiting parallel skill execution.

### 2. Memory & Observability
*   **Atomic Tracing**: Every state change is recorded in a structured `RequestTrace`.
*   **Step Replay**: Full execution history allows for deterministic replay and debugging.
*   **Plan Hashing (F4)**: Detects drift between initial plans and execution reality.

### 3. Security Ring (Middleware)
*   **Policy Engine**: Role-Based Access Control (RBAC) to allow/deny skills based on user context.
*   **Timeout Authority (F1)**: Strict enforcement of execution time limits per skill metadata.
*   **Invariant Validators**: Runtime checks to ensure system stability guarantees are never violated.

### 4. Resource Management
*   **Budget Accounting (R3)**: Tracks token usage and execution costs per attempt.
*   **Sanitization**: Automatically strips sensitive data from logs and outputs.

---

## 📦 Installed Capabilities (User Space)

The Kernel comes pre-loaded with powerful skills:

1.  **Logic-RL**: A reasoning engine that solves complex logic puzzles by iteratively writing and verifying Python code in a sandbox.
2.  **RLM (Recursive Reader)**: Analyzes massive documents by recursively breaking them down and summarizing sections.
3.  **Deep Research**: Autonomous web surfing and information synthesis.
4.  **Office Automation**: Generation of `.docx` and `.pptx` reports.

---

## 🚀 Getting Started

### Prerequisites
*   Docker & Docker Compose
*   Python 3.11+
*   Make (optional)

### Quick Start

1.  **Configure Environment**:
    ```bash
    cp .env.example .env
    # Set OPENAI_API_KEY, GROQ_API_KEY, etc.
    ```

2.  **Boot the Kernel**:
    ```bash
    docker-compose up -d --build
    ```
    *   **UI**: http://localhost:8501
    *   **Vector DB**: localhost:6343

3.  **Run Tests (Verification)**:
    ```bash
    python -m pytest tests/kernel/test_hardening_r7.py
    ```

---

## 📚 Documentation
*   [KERNEL_SPEC.md](KERNEL_SPEC.md): Strict architectural specification.
*   [SKILL_DEVELOPMENT.md](SKILL_DEVELOPMENT.md): Guide to creating new kernel modules.

---

**Status**: Hardened Release (R7). Tested for production stability.
