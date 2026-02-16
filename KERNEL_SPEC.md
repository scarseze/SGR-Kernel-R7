# Agent Kernel Specification v1.1 (Implementation-Aligned)

> Formal execution protocol for `sgr_core` CoreEngine runtime.
> Aligned with actual codebase as of 2026-02-16.

---

## 1. Kernel Responsibility Boundary

**Kernel (CoreEngine) owns:**
- Plan lifecycle, step orchestration (DAG), skill invocation
- Middleware enforcement, retry + timeout, policy + approval gates
- Security validation (input / params / output)
- Trace collection, budget accounting, replan loop, memory persistence

**Kernel does NOT own:** UI, transport, skill logic, RAG internals, planner prompting details.

---

## 2. Request Lifecycle Contract

```
run()
 ├─ security.validate(user_input)
 ├─ db.session scope
 ├─ ensure_initialized()
 ├─ load memory context
 ├─ semantic memory augment
 ├─ create_plan()
 ├─ DAGExecutor.run()
 │    └─ _execute_step() × N
 ├─ optional replan loop (max 2)
 ├─ trace finalize
 └─ response
```

---

## 3. Planning Contract

```python
ExecutionPlan:
    steps: List[PlanStep]
    reasoning: str
    direct_response: Optional[str]
```

| Rule | Behavior |
|------|----------|
| `direct_response` present | No DAG execution, return immediately |
| `steps` empty | `empty_plan` status |
| `len(steps) > MAX_STEPS` | Truncated to 10 |
| Budget exhausted | Planning blocked |

---

## 4. Step Execution Contract

### 4.1 Step Pipeline

```
params → template resolution → security.validate_params()
→ pydantic schema validation → skill.execute()
→ security.validate_output() → policy.record_step_cost()
→ trace append
```

### 4.2 Param Template Resolution

| Syntax | Example |
|--------|---------|
| Full ref | `{{step_1.output}}` |
| Nested | `{{step_1.output.field.sub}}` |
| Mixed | `"hello {{step_1.output}}!"` |

| Case | Behavior |
|------|----------|
| Full template → single ref | Returns **raw typed value** |
| Interpolated string | String substitution |
| `dict/list` in interpolation | `json.dumps` (not `str()`) |
| Pydantic model | `model_dump()` → `json.dumps` |
| Unresolved ref | Stays literal |
| Recursive structures | Resolved for `dict`, `list` |

---

## 5. SkillExecutionContext Contract

```python
SkillExecutionContext:
    request_id, step_id, skill_name
    params, state, skill, metadata, trace
    timeout: float = 0.0    # set ONLY by TimeoutMiddleware
    attempt: int = 1
    start_time: float

    @property
    is_retry → bool = (attempt > 1)
```

Middleware **MUST** check `ctx.is_retry` for: side-effect suppression, approval dedup, external call dedup.

---

## 6. Middleware Pipeline Contract

### Fixed Ordering (Hard Invariant)

> [!CAUTION]
> Reordering = spec violation. Requires spec bump.

| Phase | Order |
|-------|-------|
| `before_execute` | Trace → Policy → Approval → Timeout |
| `after_execute` / `on_error` | Timeout → Approval → Policy → Trace |

### Authority Domains

| Middleware | May Mutate | Retry Behavior |
|-----------|------------|----------------|
| Trace | `StepTrace` | Always runs |
| Policy | allow/deny decision | **Skips on retry** |
| Approval | allow/block (HITL) | **Skips on retry** |
| Timeout | `ctx.timeout` **ONLY** | Always runs |

---

## 7. Timeout Authority Rule

```
SkillMetadata.timeout_sec = single source of truth
TimeoutMiddleware        = only writer of ctx.timeout
Kernel fallback          = if ctx.timeout == 0 → 60s
```

> [!IMPORTANT]
> No other component may set `ctx.timeout`.

---

## 8. Retry Model

| Policy | Max Attempts |
|--------|-------------|
| `NONE` | 1 |
| `STANDARD` | 3 |
| `AGGRESSIVE` | 5 |

| Error Type | Retryable? |
|-----------|-----------|
| `Exception` | ✅ |
| `TimeoutError` | ✅ |
| `PolicyDenied` | ❌ Fatal |
| `HumanDenied` | ❌ Fatal |

Backoff: `2^attempt` seconds. Each attempt recorded in `AttemptTrace`, `ctx.attempt`, `ctx.is_retry`.

---

## 9. StepResult Contract

```python
StepResult:
    data: Any            # raw → step_outputs[step_id]
    output_text: str      # human → chat/history
    status: StepStatus    # COMPLETED | BLOCKED | FAILED
```

- Trace uses `result.trace_preview()` (json-safe, truncated)
- Legacy `str` → auto-wrapped in `StepResult`
- Security sanitizer may rewrite output but **must preserve status**

---

## 10. Security Enforcement Contract

| Phase | Method | When |
|-------|--------|------|
| Input | `security.validate(user_input)` | Before planning |
| Params | `security.validate_params(resolved)` | After template resolution |
| Output | `security.validate_output(str[:5000])` | After skill execution |

Output violation → sanitize (`[Output sanitized]`), not crash step.

---

## 11. DAG Executor Contract

**Receives:** `steps`, `execute_fn`, `budget_check_fn`, `max_concurrent`

**Guarantees:** dependency ordering, bounded concurrency (5), result capture, failure capture, cascade cancellation, partial completion allowed.

```python
DAGResult:
    success, completed[], failed[], cancelled[]
    results: Dict[step_id, StepResult]
    summary: str
```

---

## 12. Replan Contract

**Trigger:** `dag_result.failed AND replan_round < max_replan_attempts`

| Invariant | Enforcement |
|-----------|-------------|
| Completed steps preserved | `completed_step_ids` set |
| Step ID collision | Versioned with `_rN` suffix |
| Outputs bindable | `step_outputs` shared across rounds |
| Remaining steps filtered | Rebuilt from new plan |

---

## 13. Budget Contract

| Phase | Mechanism |
|-------|-----------|
| Before planning | `policy.check_budget()` |
| DAG scheduling | `budget_check_fn()` per node |
| After step | `policy.record_step_cost(step_id, skill, estimated_cost)` |
| Planner usage | `policy.record_cost(usage.total_cost)` |

Budget is **monotonic**. Breach may: stop DAG, skip replan, return blocked/error.

---

## 14. Trace Contract

```python
RequestTrace: request_id, plan, steps[], llm_calls[], status, duration, error?
StepTrace:    step_id, skill_name, input_params, status, output_data,
              attempts[], policy_events[], approval_events[]
```

**Invariant:** Append once per `step_id` (set-based dedup). Attempts appended per try.

---

## 15. Skill Registration Contract

`register_skill()` guarantees:
- Metadata normalized to `SkillMetadata` (via `model_validate`)
- Capabilities indexed in `cap_index`
- RAG injected if capability requires (`deep_research`/`reasoning`)
- **Metadata immutable after registration**

---

## 16. Kernel Safety Invariants

> [!CAUTION]
> Violating any = **kernel bug**. Must hold at all times.

1. No skill runs without middleware pass
2. No skill runs without timeout
3. No param bypasses `security.validate_params`
4. No output bypasses `security.validate_output`
5. Retry attempts always traced
6. `PolicyDenied` never retried
7. `HumanDenied` never retried
8. Timeout comes only from `metadata.timeout_sec`
9. `step_id` never overwritten in trace
10. Template resolution never executes code

---

## 17. Extension Stability Zones

| ✅ Safe to extend | ❌ Spec-breaking |
|-------------------|------------------|
| New middleware | Middleware ordering |
| New retry policies | `StepResult` schema |
| New policy rules | `ctx.is_retry` semantics |
| New metadata fields | Timeout authority |
| New trace fields | Template syntax |
| New DAG strategies | Retry semantics |
