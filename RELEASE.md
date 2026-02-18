# SGR Kernel Release Information

## 📦 Supported Versions
*   **Python**: 3.10, 3.11, 3.12
*   **OS**: Linux (Production), Windows/macOS (Dev)
*   **Docker**: 20.10+ (Required for Sandboxing)

## ⚖️ Guarantees
1.  **Idempotency**: Re-running a completed step within the same `request_id` and unchanged step inputs is a no-op properly handled by the state machine.
2.  **Sandboxing**: If a Skill declares `sandboxed=True` (default), it WILL be executed via the configured sandbox backend (Docker by default). Sandboxing backend support currently targets Linux Docker environments. If no backend is available, execution FAILS CLOSED (blocked).
3.  **Determinism**: Replay Mode guarantees bit-for-bit identical execution of kernel decisions and recorded outputs when LLM/tool traces are captured.
4.  **Capability Enforcement**: A Skill CANNOT execute actions outside its declared capability set. Violations are blocked by the Kernel and recorded as `CAPABILITY_VIOLATION`. Capabilities are enforced at step execution time by the Kernel.
5.  **Resource Governance**: Token, cost, and time budgets are enforced at runtime according to Kernel accounting and configured budget policies. Budget overflow results in controlled abort — never silent degradation. Budget and capability violations are always recorded in execution telemetry.

## 🚫 Non-Goals
*   **Web UI**: The Kernel is a backend library. Use `sgr-core` or Chainlit for UI.
*   **Cloud Hosting**: We provide the runtime; you provide the infrastructure (Kubernetes/Borg).
*   **Model Serving**: We orchestrate LLMs, we don't host them. Use Ollama/vLLM.

## ⚠️ Known Limitations (v1.0.0-rc1)
*   **Parallelism**: Parallel execution model is defined but not yet enabled by default (limited to `max_workers=1` for safety).
*   **Streaming**: Token streaming is supported but not yet fully integrated into the `StepResult` history.
*   **Replay**: Replaying non-deterministic tools (e.g., `requests.get` to a changing URL) without a captured trace will diverge.
*   **External Side Effects**: The Kernel cannot automatically roll back external side effects (emails, API writes, shell commands). Use approval gates and capabilities.

## 🐛 Reporting Issues
Please use the [GitHub Issue Tracker](https://github.com/sgr/kernel/issues) with the tag `v1.0.0-rc1`.

Please include:
- kernel version
- Python version
- OS
- replay manifest (if available)
- step trace id

If reporting replay issues — attach replay manifest and checkpoint metadata if available.
