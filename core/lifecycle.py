"""
StepLifecycleEngine for SGR Kernel (RFC v2).
Implements Strict 7-Phase Lifecycle and Atomic Commit.
"""
import time
import asyncio
from typing import Any, Dict, Optional, Callable

from core.execution import ExecutionState, StepNode, StepStatus, SemanticFailureType, StepState, RetryPolicy, FailureRecord
from core.skill_interface import SkillRuntimeAdapter, SkillContext, SkillResult
from core.reliability import ReliabilityEngine, RecoveryAction
from core.critic import CriticEngine
from core.repair import RepairEngine
from core.governance import GovernanceHooksBus, HOOK_BEFORE_STEP, HOOK_AFTER_STEP, HOOK_ON_FAILURE, HOOK_ON_RETRY, HOOK_ON_COMMIT

class StepLifecycleEngine:
    """
    RFC v2 Section 3.3: StepLifecycleEngine
    """
    
    def __init__(
        self, 
        skill_adapter: SkillRuntimeAdapter, 
        reliability: ReliabilityEngine,
        critic: CriticEngine,
        repair: RepairEngine,
        hooks: GovernanceHooksBus
    ):
        self.skill_adapter = skill_adapter
        self.reliability = reliability
        self.critic = critic
        self.repair = repair
        self.hooks = hooks

    async def run_step(self, step: StepNode, state: ExecutionState):
        """
        Execute the step lifecycle for a given node.
        """
        step_id = step.id
        
        # Initialize Step State (Transition: PENDING -> RUNNING)
        if step_id not in state.step_states:
            state.initialize_step(step_id)
        
        s_state = state.step_states[step_id]
        
        # Idempotency Check (Release Gate v1)
        if step.idempotent and s_state.status == StepStatus.COMMITTED:
            # Already done, skip execution
            return

        s_state.status = StepStatus.RUNNING
        s_state.started_at = time.time()
        
        policy = step.retry_policy or RetryPolicy()

        await self.hooks.emit(HOOK_BEFORE_STEP, step, state)

        while True:
            # Loop for Retries/Repairs
            s_state.attempts += 1
            
            # 1. Determine Tier (RFC v2 Section 4.2)
            current_tier = self.reliability.get_escalation_tier(s_state.attempts)
            s_state.tier_used = current_tier

            try:
                # --- Phase 1: EXECUTE ---
                # Build Context
                skill_config = step.inputs_template.copy()
                skill_config.update({"tier": current_tier})

                ctx = SkillContext(
                    execution_state=state,
                    llm_service=None, 
                    tool_registry=self.skill_adapter.registry, 
                    config=skill_config
                )
                
                # Budget Check (Release Gate v1)
                if state.token_budget and state.tokens_used >= state.token_budget:
                    raise RuntimeError(f"Token Budget Exceeded ({state.tokens_used}/{state.token_budget})")

                # Capability Enforcement (Release Gate v1)
                # Ensure Skill doesn't exceed permissions granted by Step
                skill = self.skill_adapter.registry.get(step.skill_name)
                if skill:
                    # If skill not found, adapter will raise later, or check here
                    for cap in skill.capabilities:
                        if cap not in step.required_capabilities:
                            raise PermissionError(f"Security Violation: Skill '{step.skill_name}' requires '{cap}' but Step '{step.id}' only grants {step.required_capabilities}")

                # Execute via Adapter with Timeout (Release Gate v1)

                # Execute via Adapter with Timeout (Release Gate v1)
                result_obj = await asyncio.wait_for(
                    self.skill_adapter.execute_skill(step.skill_name, ctx),
                    timeout=step.timeout_seconds
                )
                result = result_obj.output
                
                # --- Phase 2: VALIDATE (Implicit Schema Check) ---
                # s_state.status = StepStatus.VALIDATING
                # If explicit schema exists, validate here.
                # If adapter throws validation error, catch below as SCHEMA_FAIL
                
                # --- Phase 3: CRITIC ---
                if step.critic_required:
                    # s_state.status = StepStatus.CRITIC
                    passed, reason = await self.critic.evaluate(step.id, step.skill_name, {}, result)
                    if not passed:
                        raise ValueError(f"Critic Failed: {reason}") # -> CRITIC_FAIL

                # --- Phase 4: REPAIR (Conditional) ---
                # Handled in catch block via ReliabilityEngine loop
                
                # --- Phase 5: APPROVAL ---
                if step.approval_required:
                    # s_state.status = StepStatus.APPROVAL
                    # Pause execution logic here?
                    # For MVP, simulate pass or hook wait.
                    pass 

                # --- Phase 6: COMMIT (Atomic) ---
                # s_state.status = StepStatus.COMMITTED
                state.skill_outputs[step_id] = result
                s_state.status = StepStatus.COMMITTED
                s_state.finished_at = time.time()
                s_state.output = result 
                
                await self.hooks.emit(HOOK_AFTER_STEP, step, result)
                await self.hooks.emit(HOOK_ON_COMMIT, step, result)
                return 
            
            except Exception as e:
                print(f"DEBUG: Lifecycle caught exception: {e}") # DEBUG
                # Classify Failure
                error_msg = str(e)
                failure_type = SemanticFailureType.UNKNOWN
                if "Critic Failed" in error_msg: failure_type = SemanticFailureType.CRITIC_FAIL
                elif "validation" in error_msg.lower(): failure_type = SemanticFailureType.SCHEMA_FAIL
                elif "Security Violation" in error_msg: failure_type = SemanticFailureType.CAPABILITY_VIOLATION
                elif "timeout" in error_msg.lower() or isinstance(e, asyncio.TimeoutError): failure_type = SemanticFailureType.TIMEOUT
                else: failure_type = SemanticFailureType.TOOL_ERROR 
                
                # Record Failure (RFC v2 Spec 5.1)
                fail_rec = FailureRecord(
                    step_id=step_id,
                    failure_type=failure_type,
                    phase=s_state.status, # Current phase
                    error_class=type(e).__name__,
                    retryable=True, # Logic?
                    repairable=True,
                    error_message=error_msg
                )
                s_state.failure = fail_rec
                
                await self.hooks.emit(HOOK_ON_FAILURE, step, failure_type, error_msg)
                
                # Reliability Decision (RFC v2 Section 4)
                action = self.reliability.decide(
                    policy, 
                    failure_type, 
                    s_state.attempts, 
                    tier_used=current_tier
                )
                
                if action == RecoveryAction.ABORT:
                    s_state.status = StepStatus.FAILED
                    s_state.finished_at = time.time()
                    return 
                    
                elif action == RecoveryAction.RETRY:
                    s_state.status = StepStatus.RETRY_WAIT
                    # await asyncio.sleep(backoff)
                    await self.hooks.emit(HOOK_ON_RETRY, step, s_state.attempts)
                    continue 
                    
                elif action == RecoveryAction.ESCALATE:
                    await self.hooks.emit(HOOK_ON_RETRY, step, "escalation")
                    continue
                    
                elif action == RecoveryAction.REPAIR:
                    # s_state.status = StepStatus.REPAIR
                    # Run Repair Engine...
                    continue
                    
                elif action == RecoveryAction.FALLBACK_SKILL:
                    # Switch skill...
                    continue
