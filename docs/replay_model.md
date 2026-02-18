# Replay Model (v1.x)

## Overview
The SGR Kernel supports a **Deterministic Replay** mode. This allows you to "record" an execution trace and then "replay" it later without making any external API calls (LLM, Network).

## Use Cases
- **Regression Testing**: Ensure new code doesn't break old logic.
- **Debugging**: Step-through a past failure exactly as it happened.
- **Auditing**: Verify what happened during a run.

## Mechanism
1.  **Recording**:
    *   All LLM calls and Skill outputs are intercepted.
    *   Inputs (Prompt/Args) and Outputs are saved to `trace.json` (or WAL).
    *   Artifacts are stored in `ArtifactStore`.

2.  **Replaying**:
    *   Kernel starts in `REPLAY` mode.
    *   When an LLM call or Skill execution is requested, the Kernel checks the Tape.
    *   If a matching entry is found, the **Recorded Output** is returned immediately.
    *   **No external side-effects** occur during replay.

## Artifacts
Replay relies on the **Artifact Store** to provide file contents exactly as they were during recording, ensuring bit-for-bit reproducibility.
