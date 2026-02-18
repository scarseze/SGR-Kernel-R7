# SGR Kernel Execution Model (v1.x)

## Overview
The SGR Kernel acts as a deterministic state machine that orchestrates the execution of AI agents. It provides a formal runtime environment for Directed Acyclic Graphs (DAG) of steps, ensuring resource governance, safety, and reliability.

## 1. The Core Loop
The execution follows a strict **Plan -> Graph -> Lifecycle** flow:

1.  **Request**: User input or task is received.
2.  **Plan Generation**: A `PlanIR` is created, representing the decomposition of the task into steps.
3.  **Graph Sorting**: The `ExecutionGraphEngine` topologically sorts the DAG to identify runnable steps (those with satisfied dependencies).
4.  **Step Lifecycle**: Each step is executed via the `StepLifecycleEngine` through formal phases:
    - **Validation**: Input schema and security check.
    - **Execution**: Invocation of the skill.
    - **Verification**: Output validation and critic review.
5.  **Checkpointing**: After every successful step, the `ExecutionState` is persisted via `CheckpointManager` to allow for crash recovery.

## 2. State Machine (FSM)
The global `ExecutionState` moves through specific lifecycle states:

| State | Description |
| :--- | :--- |
| `CREATED` | Initial state before planning. |
| `PLANNED` | DAG generated and steps initialized. |
| `RUNNING` | Active execution of the step graph. |
| `COMPLETED` | All steps in the DAG successfully committed. |
| `FAILED` | Terminal failure state (max retries/recovery exhausted). |
| `ABORTED` | Manually terminated by user or security policy. |

## 3. Data Flow & Artifacts
- **Inputs**: Parameters are passed to skills via `SkillContext`.
- **Outputs**: Results are stored in `state.skill_outputs` and used for variable hydration in downstream steps.
- **Artifacts**: Large binary or structured data objects are stored in the **Content-Addressed Artifact Store (CAS)** and referenced by SHA256 hashes.

## 4. System Guarantees
| Guarantee | Mechanism |
| :--- | :--- |
| **Idempotency** | Steps marked `idempotent=True` are skipped if already committed, enabling safe retries. |
| **Determinism** | **Replay Mode** allows bit-for-bit reconstruction of agent execution using recorded LLM calls and artifacts. |
| **Safety** | **Capability Enforcement** ensures skills only access resources (IO, NET) explicitly granted by the step. |
| **Crash Recovery** | `resume(request_id)` loads the last valid checkpoint to continue interrupted runs. |
