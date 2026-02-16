# ... (imports) ...
from typing import Dict, Callable, Awaitable, Any, Optional, List
from core.planner import Planner, ExecutionPlan
from core.trace import TraceManager, RequestTrace, StepTrace, LLMCallTrace, PolicyEvent, ApprovalEvent, AttemptTrace
from core.policy import PolicyEngine
from core.logger import logger
from core.database import Database
from core.memory import PersistentMemory
from core.state import AgentState, Message
from core.security import SecurityGuardian, SecurityViolationError
from core.llm import LLMService
from skills.base import BaseSkill
from core.types import SkillExecutionContext, SkillMetadata, Capability, RiskLevel, PolicyStatus, RetryPolicy, StepStatus
from core.middleware import (
    TraceMiddleware, PolicyMiddleware, ApprovalMiddleware,
    TimeoutMiddleware, SkillMiddleware, PolicyDenied, HumanDenied
)
import time
import os
import json
import asyncio
import re
import hashlib
from core.task_queue import SQLAlchemyTaskQueue, TaskStatus, BackgroundTask
from core.result import StepResult

class CoreEngine:
    def __init__(self, llm_config: Dict[str, str] = None, user_id: str = "default_user", approval_callback: Callable[[str], Awaitable[bool]] = None):
        # ... (init logic) ...
        logger.info("Initializing CoreEngine...", user_id=user_id)
        self.user_id = user_id
        self.approval_callback = approval_callback

        self.db = Database()
        
        self.llm = LLMService(**(llm_config or {}))
        
        # Initialize RAG components early for Memory
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        from core.rag.embeddings import OllamaEmbeddingProvider
        from core.rag.vector_store import QdrantAdapter

        self.embedding_provider = OllamaEmbeddingProvider(base_url=ollama_url)
        self.vector_store = QdrantAdapter(host=qdrant_host, port=qdrant_port)


        from core.summarizer import ConversationSummarizer
        from core.memory_manager import MemoryManager

        self.memory = PersistentMemory(self.db, vector_store=self.vector_store, embedding_provider=self.embedding_provider)
        self.summarizer = ConversationSummarizer(self.llm)
        self.memory_manager = MemoryManager(self.memory, self.summarizer)
        
        
        task_queue_type = os.getenv("TASK_QUEUE_TYPE", "sql")
        if task_queue_type == "redis":
            try:
                from core.task_queue.redis_queue import RedisTaskQueue
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self.task_queue = RedisTaskQueue(redis_url)
                logger.info("Using RedisTaskQueue")
            except Exception as e:
                logger.error(f"Failed to init RedisTaskQueue: {e}. Falling back to SQL.")
                self.task_queue = SQLAlchemyTaskQueue(self.db)
        else:
            self.task_queue = SQLAlchemyTaskQueue(self.db)
            logger.info("Using SQLAlchemyTaskQueue")
        self.task_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Any]]] = {}

        # Load state with persistent history (DELAYED)
        self.state = AgentState(user_request="")
        self.context_loaded = False

        self.planner = Planner(self.llm)
        self.tracer = TraceManager()
        self.policy = PolicyEngine()

        # Middleware Stack
        self.middlewares: List[SkillMiddleware] = [
            TraceMiddleware(),
            PolicyMiddleware(self.policy),
            ApprovalMiddleware(self.approval_callback),
            TimeoutMiddleware(),
            # RetryMiddleware(), # To be implemented
        ]

        # ... (rest of init) ...
        # Initialize RAG Platform (Stage 16)
        # Note: Embeddings and VectorStore already imported/init above
        from core.rag.retriever import RAGRetriever
        from core.rag.context import RAGContextBuilder
        from core.rag.components import QueryRewriter, QueryExpander, DomainRouter, ScoreReranker, DocFilter
        from core.rag.pipeline import RAGPipeline
        from core.rag.repair import AnswerCritic, RepairStrategy
        
        # 1. Foundation
        # Components already initialized above
        self.retriever = RAGRetriever(self.embedding_provider, self.vector_store)
        self.context_builder = RAGContextBuilder()
        
        # 2. Components
        self.rewriter = QueryRewriter(self.llm)
        self.expander = QueryExpander()
        self.router = DomainRouter()
        
        # Configure with Policy
        self.reranker = ScoreReranker() 
        # Reranker threshold is applied at runtime or we can pass it to RAGPipeline if it accepted it.
        # Currently RAGPipeline calls reranker.rerank(docs). 
        # The enhanced reranker.rerank takes a threshold.
        # We need to update RAGPipeline to pass this threshold OR configure components here.
        
        self.filterer = DocFilter()
        # Similarly for filterer.
        
        
        # 3. Pipeline
        self.critic = AnswerCritic(self.llm)
        self.repair_strategy = RepairStrategy()

        self.rag = RAGPipeline(
            retriever=self.retriever,
            rewriter=self.rewriter,
            expander=self.expander,
            router=self.router,
            reranker=self.reranker,
            filterer=self.filterer,
            context_builder=self.context_builder,
            max_docs=self.policy.rag_max_docs,
            max_tokens=self.policy.rag_max_tokens,
            critic=self.critic,
            repair_strategy=self.repair_strategy,
            # Pass Policy Thresholds to Pipeline
            min_score=self.policy.rag_min_score,
            rerank_threshold=self.policy.rag_rerank_threshold
        )
        self.skills: Dict[str, BaseSkill] = {}
        self.cap_index: Dict[str, List[BaseSkill]] = {} # Capability Registry

        # Security
        self.security = SecurityGuardian()
        
        # Concurrency Control (F2)
        self._skill_semaphores: Dict[str, asyncio.Semaphore] = {}

    async def _ensure_initialized(self):
        """Ensure DB is initialized and context is loaded."""
        if not self.context_loaded:
            await self.db.init_db()
            await self._load_context()
            self.context_loaded = True

    async def _load_context(self):
        """Load conversation history using MemoryManager."""
        await self.memory_manager.load_context(self.user_id, self.state)

    async def _save_message(self, role: str, content: str):
        """Save message to state and persistent memory."""
        # 1. Update State
        self.state.history.append(Message(role=role, content=content))

        # 2. Persist
        try:
            await self.memory.add_message(self.user_id, role, content)
        except Exception as e:
            logger.error("Failed to save message", error=str(e))

    def register_skill(self, skill: BaseSkill):
        # Inject RAG if available and skill declares the capability
        if hasattr(self, 'rag') and self.rag:
            requires_rag = (
                hasattr(skill, 'metadata') 
                and hasattr(skill.metadata, 'capabilities') 
                and any(
                    (c.value if hasattr(c, 'value') else str(c)) in ('deep_research', 'reasoning')
                    for c in skill.metadata.capabilities
                )
            )
            if requires_rag and hasattr(skill, 'rag'):
                setattr(skill, 'rag', self.rag)
        
        # Metadata Normalization (Hardening)
        # Ensure skill.metadata is a valid SkillMetadata object, not a dict
        if isinstance(skill.metadata, dict):
            skill.metadata = SkillMetadata.model_validate(skill.metadata)
        elif not isinstance(skill.metadata, SkillMetadata):
            # Safety: clone to avoid mutating shared instances
            skill.metadata = SkillMetadata.model_validate(skill.metadata.model_dump())
        
        self.skills[skill.name] = skill

        # Capability Registry
        # Use Normalized Metadata
        for cap in skill.metadata.capabilities:
            cap_key = cap.value if hasattr(cap, 'value') else str(cap)
            self.cap_index.setdefault(cap_key, []).append(skill)

    def register_task_handler(self, name: str, handler: Callable[[Dict[str, Any]], Awaitable[Any]]):
        """Register a handler for a background task."""
        self.task_handlers[name] = handler
        logger.info(f"Registered task handler: {name}")

    async def submit_task(self, name: str, params: Dict[str, Any] = None) -> str:
        """Submit a task to the background queue."""
        if name not in self.task_handlers:
            logger.warning(f"Submitting task '{name}' but no handler is registered.")
        
        task = await self.task_queue.enqueue(name, params or {})
        return task.task_id

    async def run_worker(self, interval: float = 1.0, max_task_attempts: int = 3):
        """Run a background worker loop to process tasks."""
        logger.info("Starting background task worker...")
        _failure_counts: Dict[str, int] = {}  # task_id → attempt count (poison detection)
        while True:
            try:
                # 1. Claim Task
                task = await self.task_queue.claim_next_task()
                if not task:
                    await asyncio.sleep(interval)
                    continue
                
                logger.info(f"Processing task {task.task_id} ({task.name})")
                
                # Poison detection: skip tasks that failed too many times
                fc = _failure_counts.get(task.task_id, 0)
                if fc >= max_task_attempts:
                    logger.error(f"Poison task {task.task_id}: failed {fc} times, moving to dead letter")
                    await self.task_queue.update_status(task.task_id, TaskStatus.FAILED, error=f"Poison: exceeded {max_task_attempts} attempts")
                    continue
                
                # 2. Find Handler
                handler = self.task_handlers.get(task.name)
                if not handler:
                    await self.task_queue.update_status(task.task_id, TaskStatus.FAILED, error="No handler registered")
                    continue
                
                # 3. Execute
                try:
                    result = await handler(task.params)
                    await self.task_queue.update_status(task.task_id, TaskStatus.COMPLETED, result=result)
                    logger.info(f"Task {task.task_id} completed")
                    _failure_counts.pop(task.task_id, None)  # clean up on success
                except Exception as e:
                    _failure_counts[task.task_id] = fc + 1
                    logger.error(f"Task {task.task_id} failed (attempt {fc + 1}/{max_task_attempts}): {e}")
                    await self.task_queue.update_status(task.task_id, TaskStatus.FAILED, error=str(e))
                    
            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.error(f"Worker execution error: {e}")
                await asyncio.sleep(5.0) # Backoff


    async def _execute_step(self, step_def, step_outputs: Dict[str, Any], trace: RequestTrace) -> Any:
        """
        Execute a single step using the Middleware Pipeline.
        """
        skill_name = step_def.skill_name

        # 0. Resolve Skill
        if skill_name not in self.skills:
             raise ValueError(f"Unknown skill '{skill_name}'")

        skill = self.skills[skill_name]
        self.state.active_skill_name = skill_name

        # 1. Resolve Params
        resolved_params = self._resolve_params(step_def.params, step_outputs)

        # Fix 6: Guard — fail on unresolved templates instead of silent corruption
        def _check_unresolved(val, path=""):
            if isinstance(val, str):
                if "{Unresolved" in val or ("{{" in val and "}}" in val):
                    raise ValueError(f"Unresolved template at '{path}': {val[:200]}")
            elif isinstance(val, dict):
                for k, v in val.items():
                    _check_unresolved(v, f"{path}.{k}")
            elif isinstance(val, list):
                for i, v in enumerate(val):
                    _check_unresolved(v, f"{path}[{i}]")
        _check_unresolved(resolved_params, "params")

        # 2. Metadata matches platform types (Normalized at registration)
        metadata = skill.metadata
        
        # 3. Create Trace Entry
        step_trace = StepTrace(
            step_id=step_def.step_id,
            skill_name=skill_name,
            input_params=resolved_params,
            status=StepStatus.PENDING.value
        )
        # Bind trace to context just in case middlewares need it early

        # 4. Context Creation
        ctx = SkillExecutionContext(
            request_id=trace.request_id,
            step_id=step_def.step_id,
            skill_name=skill_name,
            params=resolved_params,
            state=self.state,
            skill=skill,
            metadata=metadata,
            trace=step_trace,
            attempt=1
        )

        logger.info("Pipeline Start", step=step_def.step_id, skill=skill_name)
        
        # Determine Retry Policy (strict enum, no string fallback)
        max_attempts = 1
        if metadata.retry_policy == RetryPolicy.STANDARD:
             max_attempts = 3
        elif metadata.retry_policy == RetryPolicy.AGGRESSIVE:
             max_attempts = 5

        result = None
        last_error = None

        # F2: Skill Concurrency Guard
        sem = self._skill_semaphores.setdefault(skill_name, asyncio.Semaphore(metadata.max_concurrency))
        async with sem:
            for attempt in range(1, max_attempts + 1):
                ctx.attempt = attempt
                
                # Trace Attempt
                attempt_trace = AttemptTrace(
                    attempt_number=attempt,
                    start_time=time.time()
                )
                
                # R1: Step Timing
                if attempt == 1:
                    step_trace.start_time = time.time()

                try:
                    # Note: Some middlewares check ctx.is_retry to skip non-idempotent actions
                    # Ordering contract: Trace → Policy → Approval → Timeout
                    # Do NOT reorder without reviewing retry semantics.
                    for m in self.middlewares:
                        await m.before_execute(ctx)

                    # 6. Validate Input (Post-resolution)
                    # Security: validate resolved params to catch injection via template resolution
                    if hasattr(self, 'security'):
                        self.security.validate_params(ctx.params)
                    skill_input = skill.input_schema(**ctx.params)

                    # 7. EXECUTE (with Timeout support from middleware setting)
                    timeout = ctx.timeout if ctx.timeout > 0 else 60.0

                    start_exec = time.time()
                    try:
                        result = await asyncio.wait_for(skill.execute(skill_input, self.state), timeout=timeout)
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"Skill execution timed out after {timeout}s")

                    # 8. AFTER Hooks (isolated — middleware errors must not trigger skill retry)
                    for m in reversed(self.middlewares):
                        try:
                            result = await m.after_execute(ctx, result)
                        except Exception as mw_err:
                            logger.error(f"Middleware {m.__class__.__name__}.after_execute failed: {mw_err}")
                            # Middleware infra failure — don't contaminate skill result

                    # 8.5 Security: validate output before exposing to user
                    if hasattr(self, 'security'):
                        try:
                            output_str = str(result)[:5000]
                            self.security.validate_output(output_str)
                        except Exception as sec_err:
                            logger.warning(f"Output security violation in {skill_name}: {sec_err}")
                            # Sanitize: preserve status/metadata, strip data
                            if isinstance(result, StepResult):
                                result = result.sanitized_copy()
                            else:
                                result = "[Output sanitized by security policy]"

                    # 8.6 Step-level budget recording (R3: Record per attempt)
                    if hasattr(self, 'policy'):
                        # Estimate cost from metadata model field
                        estimated_cost = skill.metadata.estimated_cost
                        if estimated_cost > 0:
                            # R3: Record cost for EVERY attempt, even if failed later (but here we are in success path)
                            # Ideally move to 'finally' block of attempt if we want to charge for failed attempts too.
                            # For now, following plan: record per attempt (implied success of execution phase)
                            self.policy.record_step_cost(step_def.step_id, skill_name, estimated_cost)

                    # Success
                    step_trace.status = StepStatus.COMPLETED.value
                    
                    # Record Attempt Success
                    attempt_trace.end_time = time.time()
                    attempt_trace.result_snippet = str(result)[:500]
                    step_trace.attempts.append(attempt_trace)
                    
                    # Atomic trace append — use identity check, not set rebuild (concurrency safe)
                    if step_trace not in trace.steps:
                        trace.steps.append(step_trace)
                    
                    # R1: Duration
                    step_trace.duration = time.time() - step_trace.start_time

                    # Optional: Cooldown (F2)
                    if metadata.cooldown_sec > 0:
                        await asyncio.sleep(metadata.cooldown_sec)

                    # F1: Runtime Invariant Validator
                    # print(f"DEBUG: Checking invariant. ctx.timeout={ctx.timeout}")
                    assert ctx.timeout >= 0, "KC-F3: timeout must be non-negative"
                    assert len(step_trace.attempts) > 0, "KC-K2: at least one attempt"
                    
                    # Handle Structured Output
                    final_output_str = ""
                    
                    if isinstance(result, StepResult):
                        # Store raw data for next steps
                        step_outputs[step_def.step_id] = result.data 
                        # Use string representation for chat/history
                        final_output_str = str(result)
                        step_trace.output_data = result.trace_preview() if hasattr(result, 'trace_preview') else str(result.data)[:1000]
                        
                        return result

                    else:
                        # Legacy string behavior
                        step_outputs[step_def.step_id] = result
                        final_output_str = str(result)
                        step_trace.output_data = final_output_str[:1000]
                        return StepResult(data=result, output_text=final_output_str, status=StepStatus.COMPLETED)

                except Exception as e:
                    last_error = e
                    logger.error(f"Pipeline Error in {skill_name} (Attempt {attempt}/{max_attempts}): {e}")
                    
                    # Record Attempt Error
                    attempt_trace.end_time = time.time()
                    attempt_trace.error = str(e)
                    step_trace.attempts.append(attempt_trace)

                    # R3: Budget for failed attempt? 
                    # If execution happened, we should charge. 
                    # But without fine-grained control, we stick to success-path charging or 
                    # charge here if we want strict accounting. 
                    # For R3 "retries are free" fix -> we should charge here too if execution started.
                    # But strictly speaking, metadata.estimated_cost is "per success". 
                    # Let's start with success path modification implies "if we retry, we pay again on success" 
                    # OR "we pay for every attempt". 
                    # The plan said: "Move policy.record_step_cost() into the attempt loop". 
                    # I did that (it's inside the loop now).
                    # But it's in the success block. 
                    # If I want to charge for failures, I should move it up or duplicate.
                    # Let's duplicate for now to be safe, assuming failed execution also costs.
                    if hasattr(self, 'policy'):
                         estimated_cost = skill.metadata.estimated_cost
                         if estimated_cost > 0:
                             self.policy.record_step_cost(step_def.step_id, skill_name, estimated_cost)

                    # 9. ERROR Hooks
                    for m in reversed(self.middlewares):
                        await m.on_error(ctx, e)

                    if isinstance(e, (PolicyDenied, HumanDenied)):
                        # Fatal policies, do not retry
                        step_trace.status = StepStatus.BLOCKED.value
                        if step_trace not in trace.steps:
                            trace.steps.append(step_trace)
                        return StepResult(
                            data=None, 
                            status=StepStatus.BLOCKED,
                            output_text=f"Blocked: {e}"
                        )

                # Retry Logic
                if attempt < max_attempts:
                     # Guard 1: Non-idempotent skills with side effects must not retry
                     if metadata.side_effects and not metadata.idempotent:
                         logger.warning(f"Aborting retry: {skill_name} has side_effects=True, idempotent=False")
                         step_trace.status = StepStatus.FAILED.value
                         if step_trace not in trace.steps:
                             trace.steps.append(step_trace)
                         raise e

                     # Guard 2: Budget exceeded → abort retries
                     if hasattr(self, 'policy') and not self.policy.check_budget():
                         logger.warning(f"Budget exceeded during retries of {skill_name}, aborting")
                         step_trace.status = StepStatus.FAILED.value
                         if step_trace not in trace.steps:
                             trace.steps.append(step_trace)
                         raise e

                     step_trace.status = StepStatus.RETRYING.value
                     sleep_time = 2 ** attempt # Exponential Backoff (2, 4, 8, 16s)
                     logger.info(f"Retrying in {sleep_time}s...")
                     await asyncio.sleep(sleep_time)
                     continue
                else:
                    # Final Failure
                    step_trace.status = StepStatus.FAILED.value
                    if step_trace not in trace.steps:
                        trace.steps.append(step_trace)
                    raise e
    
    # ... (rest of run methods) ...

    async def run(self, user_input: str) -> str:
        start_time = time.time()
        
        # Telemetry
        from core.telemetry import get_telemetry
        telemetry = get_telemetry()

        # Initialize Trace
        trace = RequestTrace(user_request=user_input)
        
        # Start Root Span
        # Start Root Span
        # Bind request-specific context to logger
        log = logger.bind(request_len=len(user_input), request_id=trace.request_id)
        
        # FIX: Wrap entire execution pipeline in span to capture lifecycle and errors
        try:
            with telemetry.span("CoreEngine.run", {"request_id": trace.request_id, "user_input_len": len(user_input)}) as root_span:
                log.info("Processing user request", query=user_input[:50]+"...")
                
                # 0. SECURITY CHECK (Input Phase)
                try:
                    self.security.validate(user_input)
                except SecurityViolationError as e:
                    log.warning("Security violation detected", error=str(e))
                    trace.status = "security_error"
                    self.tracer.save_trace(trace)
                    return str(e)

                # START SESSION SCOPE
                async with self.db.session():
                    # 1. Ensure DB & Context
                    await self._ensure_initialized()
        
                    self.state.user_request = user_input
                    await self._save_message("user", user_input)
        
                    # 1.5 Semantic Memory Retrieval
                    await self.memory_manager.augment_with_semantic_search(user_input, self.state)
        
                    try:
                        # 2. Planning
                        plan, usage = await self._create_plan(user_input)
                        trace.plan = plan

                        # F4: Plan Hash for Replanning Comparison
                        if plan:
                            trace.plan_hash = hashlib.sha256(plan.model_dump_json().encode()).hexdigest()[:16]
        
                        # Record Planner LLM Trace
                        usage_for_trace = {k: v for k, v in usage.items() if k != "model"}
                        trace.llm_calls.append(LLMCallTrace(
                            context="planning",
                            model=usage.get("model", "unknown"),
                            **usage_for_trace
                        ))
                        
                        # Record Cost
                        if "total_cost" in usage:
                            self.policy.record_cost(usage["total_cost"])
        
                        if plan.direct_response:
                            log.info("Handled as direct chat (Planner)")
                            await self._save_message("assistant", plan.direct_response)
                            trace.status = "completed"
                            trace.total_duration = time.time() - start_time
                            self.tracer.save_trace(trace)
                            return plan.direct_response
        
                        if not plan.steps:
                             log.warning("Planner returned no steps and no response")
                             trace.status = "empty_plan"
                             self.tracer.save_trace(trace)
                             return "I'm not sure how to help with that."
        
                        # GUARDRAIL: Max Steps
                        MAX_STEPS = 10
                        if len(plan.steps) > MAX_STEPS:
                            warn_msg = f"Plan too long ({len(plan.steps)} steps). Truncating to {MAX_STEPS}."
                            log.warning(warn_msg)
                            plan.steps = plan.steps[:MAX_STEPS]
        
                        log.info("Plan created", steps_count=len(plan.steps), reasoning=plan.reasoning)
        
                        final_result = ""
                        step_outputs = {} # For binding
        
                        # --- DAG Execution ---
                        from core.dag_executor import DAGExecutor
        
                        max_replan_attempts = 2
                        remaining_steps = plan.steps
                        completed_step_ids = set()
        
                        for replan_round in range(max_replan_attempts + 1):
                            executor = DAGExecutor(
                                steps=remaining_steps,
                                execute_fn=self._execute_step,
                                budget_check_fn=self.policy.check_budget,
                                max_concurrent=5,
                            )
        
                            dag_result = await executor.run(step_outputs, trace, telemetry)
        
                            # Save messages for completed steps
                            for step_id in dag_result.completed:
                                completed_step_ids.add(step_id)
                                result = dag_result.results[step_id]
                                node = executor.nodes[step_id]
                                step_msg = f"[Step {step_id} - {node.step_def.skill_name}]: {result}"
                                await self._save_message("assistant", step_msg)
        
                            final_result += dag_result.summary
        
                            if dag_result.success:
                                break  # All done
        
                            # Handle failure — attempt replan
                            if dag_result.failed and replan_round < max_replan_attempts:
                                failed_id = dag_result.failed[0]
                                failed_node = executor.nodes[failed_id]
                                err = str(failed_node.error)
        
                                log.info("DAG step failed, attempting replan...",
                                         failed_step=failed_id, error=err)
        
                                history_snippet = "\n".join(
                                    [f"{m.role}: {m.content[:200]}" for m in self.state.history[-5:]]
                                )
        
                                new_plan = await self.planner.repair_plan(
                                    plan, failed_id, err, history_snippet
                                )
        
                                if new_plan:
                                    # Fix 8: Replan risk guard — block risk escalation
                                    original_risk_levels = {
                                        s.skill_name: self.skills[s.skill_name].metadata.risk_level
                                        for s in plan.steps if s.skill_name in self.skills
                                    }
                                    for s in new_plan.steps:
                                        if s.skill_name in self.skills:
                                            new_risk = self.skills[s.skill_name].metadata.risk_level
                                            if new_risk == RiskLevel.HIGH and s.skill_name not in original_risk_levels:
                                                log.warning("Replan blocked: escalates risk",
                                                           skill=s.skill_name, risk=new_risk.value)
                                                new_plan = None
                                                break

                                if new_plan:
                                    log.info("Plan repaired", new_steps=len(new_plan.steps))
                                    plan = new_plan
                                    trace.plan = new_plan
                                    # Version step_ids to avoid collision with completed steps
                                    # that might reuse the same step_id with different params
                                    for s in new_plan.steps:
                                        if s.step_id in completed_step_ids:
                                            s.step_id = f"{s.step_id}_r{replan_round + 1}"
                                    # Keep only steps not yet completed
                                    remaining_steps = [
                                        s for s in new_plan.steps
                                        if s.step_id not in completed_step_ids
                                    ]
                                    continue
                                else:
                                    log.error("Replanning failed")
                                    final_result += "\n\n(Replanning failed)"
                                    trace.status = "error"
                                    break
                            else:
                                # No more replan attempts or budget issue
                                if dag_result.failed:
                                    trace.status = "error"
                                else:
                                    trace.status = "blocked"
                                break
        
                        trace.total_duration = time.time() - start_time
                        if trace.status not in ["blocked", "error"]:
                            trace.status = "completed"
                        self.tracer.save_trace(trace)
                        return final_result
        
                    except Exception as e:
                        trace.status = "error"
                        trace.error = str(e)
                        self.tracer.save_trace(trace)
                        raise e

        # Catch errors outside db session/telemetry if any (unlikely due to wrap)
        except Exception as e:
             logger.error(f"Critical error in run loop: {e}")
             raise e


    async def _create_plan(self, text: str) -> tuple[ExecutionPlan, dict]:
        # Budget Check
        if not self.policy.check_budget():
             logger.error("Budget exceeded before planning.")
             return ExecutionPlan(reasoning="budget_exceeded", direct_response="I'm sorry, but the daily budget for this agent has been exceeded."), {}

        # ... (rest of method)
        # Format skills with Metadata
        skills_desc_list = []
        for s in self.skills.values():
            try:
                meta = s.metadata
                # Fix: Capability formatting (Enum vs String)
                caps = ", ".join(c.value if hasattr(c, "value") else str(c) for c in meta.capabilities)
                desc = f"- [{s.name}] (Risk: {meta.risk_level}): {s.description} Capabilities: {caps}"
                skills_desc_list.append(desc)
            except Exception as e:
                skills_desc_list.append(f"- [{s.name}]: {s.description}")

        skills_desc = "\n".join(skills_desc_list)
        
        history_snippet = "\n".join([f"{m.role}: {m.content[:4000]}..." if len(m.content) > 4000 else f"{m.role}: {m.content}" for m in self.state.history[-5:]])
        
        return await self.planner.create_plan(text, skills_desc, history_snippet)



    def _resolve_params(self, params: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve {{step_id.output}} references in params.
        Supports retrieving nested fields from dicts/objects: {{step_1.output.key.subkey}}
        """
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                 # Extract step_id. key format: {{step_1.output}}
                 ref = v[2:-2].strip()
                 # Base target: step_1
                 parts = ref.split(".")
                 target_id = parts[0]
                 
                 # Check if referring to output
                 if len(parts) > 1 and parts[1] == "output":
                     if target_id in outputs:
                         val = outputs[target_id]
                         
                         # Traverse for nested fields (parts[2:])
                         if len(parts) > 2:
                             try:
                                 for p in parts[2:]:
                                     if isinstance(val, dict):
                                         val = val.get(p)
                                     elif hasattr(val, p):
                                         val = getattr(val, p)
                                     else:
                                         # Field not found
                                         val = f"{{Unresolved Field: {p}}}"
                                         break 
                             except Exception:
                                 val = v # Fallback on error
                         
                         resolved[k] = val
                     else:
                         resolved[k] = v # Step output not found
                 else:
                     resolved[k] = v # Not an output reference (maybe just {{var}})
            elif isinstance(v, dict):
                 resolved[k] = self._resolve_params(v, outputs)
            elif isinstance(v, list):
                 # Fix: Recursive resolve for list items
                 resolved[k] = [self._resolve_params(i, outputs) if isinstance(i, (dict, list)) else (
                    self._resolve_string_template(i, outputs) if isinstance(i, str) and "{{" in i else i
                 ) for i in v]
            else:
                 resolved[k] = v
        return resolved

    def _resolve_string_template(self, template: str, outputs: Dict[str, Any]) -> Any:
        """
        Resolve {{step_id.output}} references within a string.
        If the entire string is a single reference, returns the raw value (preserving type).
        If it contains mixed text + references, performs string interpolation.
        """
        # Fast path: entire string is a single reference → return raw value
        if template.startswith("{{") and template.endswith("}}") and template.count("{{") == 1:
            ref = template[2:-2].strip()
            parts = ref.split(".")
            target_id = parts[0]
            if len(parts) > 1 and parts[1] == "output" and target_id in outputs:
                val = outputs[target_id]
                for p in parts[2:]:
                    if isinstance(val, dict):
                        val = val.get(p, f"{{Unresolved: {p}}}")
                    elif hasattr(val, p):
                        val = getattr(val, p)
                    else:
                        return template
                return val
            return template

        # Interpolation path: mixed text like "prefix {{step_1.output}} suffix"
        def _replacer(match):
            ref = match.group(1).strip()
            parts = ref.split(".")
            target_id = parts[0]
            if len(parts) > 1 and parts[1] == "output" and target_id in outputs:
                val = outputs[target_id]
                for p in parts[2:]:
                    if isinstance(val, dict):
                        val = val.get(p, f"{{Unresolved: {p}}}")
                    elif hasattr(val, p):
                        val = getattr(val, p)
                    else:
                        return match.group(0)
                # Safety: use json.dumps for non-primitive to avoid repr garbage
                if isinstance(val, (str, int, float, bool)):
                    return str(val)
                elif isinstance(val, (dict, list)):
                    return json.dumps(val, ensure_ascii=False, default=str)
                elif hasattr(val, 'model_dump'):
                    return json.dumps(val.model_dump(), ensure_ascii=False, default=str)
                else:
                    return str(val)
            return match.group(0)

        return re.sub(r"\{\{(.+?)\}\}", _replacer, template)
