import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any
from core.types import SkillExecutionContext, PolicyStatus, RetryPolicy, StepStatus
from core.policy import PolicyEngine 
# Unified PolicyStatus is now in core.types

class PolicyDenied(Exception):
    pass

class HumanDenied(Exception):
    pass

class ReturnCached(Exception):
    def __init__(self, result):
        self.result = result

class SkillMiddleware(ABC):
    async def before_execute(self, ctx: SkillExecutionContext):
        pass

    async def after_execute(self, ctx: SkillExecutionContext, result: Any) -> Any:
        return result

    async def on_error(self, ctx: SkillExecutionContext, error: Exception):
        pass

# --- Middlewares ---

class TraceMiddleware(SkillMiddleware):
    async def before_execute(self, ctx: SkillExecutionContext):
        ctx.trace.start_time = time.time()
        ctx.trace.status = StepStatus.RUNNING.value
        
        # Set context var for deep tracing (RAG, wrappers)
        from core.trace import current_step_trace
        self.token = current_step_trace.set(ctx.trace)

    async def after_execute(self, ctx: SkillExecutionContext, result: Any) -> Any:
        # Reset context var
        from core.trace import current_step_trace
        if hasattr(self, 'token'):
            current_step_trace.reset(self.token)

        duration = time.time() - ctx.trace.start_time
        ctx.trace.duration = duration
        ctx.trace.status = StepStatus.COMPLETED.value
        ctx.trace.output_data = str(result)[:1000] # Truncate for trace
        return result

    async def on_error(self, ctx: SkillExecutionContext, error: Exception):
        ctx.trace.status = StepStatus.FAILED.value
        ctx.trace.error = str(error)
        ctx.trace.duration = time.time() - ctx.trace.start_time

class PolicyMiddleware(SkillMiddleware):
    def __init__(self, policy_engine: Any):
        self.policy_engine = policy_engine

    async def before_execute(self, ctx: SkillExecutionContext):
        # Skip policy re-check on retry — decision was already made on attempt 1
        if ctx.is_retry:
            return

        # We need to map types if policy engine uses different models, 
        # but locally it receives skill and state.
        decision = self.policy_engine.check(
            ctx.skill,
            ctx.params, # validated model or dict? Check policy expectations.
            ctx.state
        )
        
        # Log event to trace
        # Assuming ctx.trace has policy_events list (from StepTrace)
        from core.trace import PolicyEvent
        ctx.trace.policy_events.append(PolicyEvent(
            step_id=ctx.step_id,
            skill_name=ctx.skill_name,
            decision=decision.status.value, # Enum conversion
            reason=decision.reason
        ))

        if decision.status == PolicyStatus.DENY:
             raise PolicyDenied(decision.reason)
        
        if decision.status == PolicyStatus.REQUIRE_APPROVAL:
             ctx.trace.policy_required = True # Flag implementation needed on StepTrace or Context
             ctx.metadata.requires_approval_hint = True # Force approval hint

class ApprovalMiddleware(SkillMiddleware):
    def __init__(self, callback: Any):
        self.callback = callback

    async def before_execute(self, ctx: SkillExecutionContext):
        # Skip approval re-request on retry — user already approved on attempt 1
        if ctx.is_retry:
            return

        # Check if approval required (either by Policy or Metadata hint)
        required = getattr(ctx.trace, 'policy_required', False) or ctx.metadata.requires_approval_hint
        
        if not required:
            return

        # Prepare message
        msg = f"Skill '{ctx.skill_name}' request APPROVAL.\n\nReason: Policy Flagged.\nParams: {ctx.params}"
        
        is_approved = False
        if self.callback:
            try:
                is_approved = await self.callback(msg)
            except Exception as e:
                # Log error?
                pass
        
        # Log to Trace
        from core.trace import ApprovalEvent
        ctx.trace.approval_events.append(ApprovalEvent(
             step_id=ctx.step_id,
             skill_name=ctx.skill_name,
             approved=is_approved,
             reason="Policy/High Risk"
        ))

        if not is_approved:
            raise HumanDenied("Operation cancelled by user.")

class TimeoutMiddleware(SkillMiddleware):
    async def before_execute(self, ctx: SkillExecutionContext):
        # Timeout authority: metadata.timeout_sec is the single source of truth.
        # This middleware is the ONLY writer of ctx.timeout (KC-F1 / KC-F3).
        timeout = ctx.metadata.timeout_sec
        if timeout > 0:
            ctx.timeout = timeout

class RetryMiddleware(SkillMiddleware):
    # This also suggests an 'around' or 'runner' logic.
    # Logic: "if fail, retry".
    # In 'on_error', we could potentially trigger a retry, 
    # but clean retry loops are best handled by a wrapper around the execution.
    # Let's adapt the design: The Engine loop handles the Retry logic by checking Metadata,
    # OR we implement a robust 'around' pattern. 
    pass 
    # We will implement logic in Engine for Retry based on Metadata for this iteration,
    # or keep it simple.
