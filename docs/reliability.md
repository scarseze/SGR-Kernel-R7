# Reliability & Fault Tolerance (v1.x)

## Overview
The SGR Kernel targets industrial-grade reliability. It uses a **Semantic Failure Engine** to decide whether to retry, repair, or escalate when a step fails.

## 1. Semantic Failure Classification
All errors are mapped to internal failure types to allow for intelligent recovery logic:

| Type | Description |
| :--- | :--- |
| `SCHEMA_FAIL` | Skill output does not match expected JSON/Pydantic schema. |
| `TIMEOUT` | Skill exceeded its `timeout_seconds`. |
| `TOOL_ERROR` | Runtime exception within the skill logic (e.g., FileNotFoundError). |
| `CRITIC_FAIL` | The Output Critic rejected the skill's result. |
| `CAPABILITY_VIOLATION` | Security block (Step lacks granted permissions for skill requirements). |
| `RATE_LIMIT` | Provider returned 429 (LLM or External API). |

## 2. Recovery Strategies (RFC v2)
The Kernel follows a prioritized decision matrix for recovery:

1.  **Retry**: For transient errors (Network, Timeout). Uses exponential backoff.
2.  **Repair**: For semantic errors (Schema, Critic). The error is fed back to the model with context to "fix" the output.
3.  **Escalate**: If a "Low Confidence" or repeated failure occurs, the Kernel switches to a higher-tier model (e.g., from *Flash* to *Pro/Ultra*).
4.  **Abort**: For fatal errors (Security violation). Execution stops immediately to prevent damage.

## 3. Escalation Policy
The `ReliabilityStrategy` defines standard tiers for model escalation:
- **Tier 1 (Fast)**: Optimized for speed and cost.
- **Tier 2 (Standard)**: Balance of performance and reasoning.
- **Tier 3 (Heavy)**: Reserved for complex repairs and critical decision-making.

## 4. Crash Recovery
Thanks to the WAL (Write-Ahead Log) checkpointing system, the Kernel can survive process termination:
- Execution state is saved to `checkpoints/{request_id}/` after every step.
- `CoreEngine.resume(request_id)` restores the graph and continues from the first uncommitted step.
