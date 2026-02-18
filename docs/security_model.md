# Security Model (v1.x)

## Overview
The SGR Kernel treats security as a first-class concern through a **Capability-Based ACL** system and **Proactive Sandboxing**.

## 1. Capability Enforcement
Skills are not trusted by default. They must declare their requirements, and the Plan must explicitly grant those permissions at the step level.

- **Skill Side**: `Skill.capabilities` (e.g., `{"NET", "WRITE_EXTERNAL"}`)
- **Plan Side**: `StepNode.required_capabilities` (e.g., `["NET"]`)

**Result**: If `capabilities` $\not\subseteq$ `required_capabilities`, the Kernel raises a `CAPABILITY_VIOLATION` (PermissionError) and aborts the step before execution.

## 2. Sandbox Isolation
The SGR Kernel supports external sandbox providers (e.g., Docker, firecracker):
- **FileSystem**: Skills should only operate within the `ArtifactStore` paths or specific mount points.
- **Network**: Capability-based blocking at the adapter level.

## 3. Human-in-the-Loop (Approval Gates)
For high-risk actions, the Kernel can be configured with **Approval Gates**:
- Steps marked with high risk levels trigger the `HOOK_BEFORE_STEP`.
- Execution halts until an external signal (Human Approval) is received.
- State is set to `PAUSED_APPROVAL`.

## 4. Output Sanitization
The `CriticEngine` and `SecurityGuardian` scan skill outputs for:
- API Keys and secrets leakage.
- Malicious code generation (in code interpreter skills).
- Over-broad permissions requests.
