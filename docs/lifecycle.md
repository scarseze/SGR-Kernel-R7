# Step Lifecycle (v1.x)

## Overview
Every step in the SGR Kernel undergoes a rigorous lifecycle managed by `StepLifecycleEngine`.

## Phase Flow
1.  **Pending**: Waiting for dependencies.
2.  **Validation**:
    *   **Schema Check**: Are inputs valid?
    *   **Capability Check**: Does the skill have permissions?
    *   **Policy Check**: Is this allowed by governance?
3.  **Execution**:
    *   The `Skill.execute()` method is called.
    *   Outputs and Artifacts are captured.
4.  **Verification (Critic)**:
    *   If `critic_required=True`, the `CriticEngine` evaluates the output.
    *   Score < Threshold triggers `REPAIR` or `FAIL`.
5.  **Commit**: Result is written to `ExecutionState` and a Checkpoint is saved.

## Reliability
- **Retries**: Managed by `ReliabilityEngine` based on `RetryPolicy`.
- **Repair**: If a Critic fails, the `RepairEngine` attempts to fix the output without re-running the tool (if possible) or suggests different inputs.
