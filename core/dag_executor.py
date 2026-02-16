"""
DAG Executor — parallel async step execution with dependency resolution.

Replaces sequential `while pending_steps: pop(0)` with a DAG-based executor
that runs independent steps in parallel via asyncio, respecting dependency
ordering through asyncio.Event signals.

Architecture:
    - Each step is wrapped in a StepNode with its own asyncio.Event
    - All nodes are launched as concurrent asyncio.Tasks
    - Each task awaits the done_events of its dependencies before executing
    - Independent steps (no shared deps) run in parallel
    - A Semaphore controls max concurrency
    - Cascade cancellation: if a step fails, all dependents are cancelled
"""

import asyncio
import time
import structlog
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum

from core.types import StepStatus
from core.result import StepResult

logger = structlog.get_logger()


class DAGError(Exception):
    """Raised when the DAG has structural problems (cycles, missing deps)."""
    pass


class BudgetExhaustedError(Exception):
    """Raised when the budget check fails before step execution."""
    pass


@dataclass
class StepNode:
    """Wraps a plan step with execution state and synchronization primitives."""
    step_id: str
    step_def: Any                                      # PlanStep from planner
    depends_on: Set[str] = field(default_factory=set)
    status: StepStatus = StepStatus.PENDING
    result: Optional[StepResult] = None
    error: Optional[Exception] = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)

    def mark_done(self, result: StepResult):
        self.status = StepStatus.COMPLETED
        self.result = result
        self.done_event.set()

    def mark_failed(self, error: Exception):
        self.status = StepStatus.FAILED
        self.error = error
        self.done_event.set()  # unblock dependents so they can see the failure

    def mark_cancelled(self):
        self.status = StepStatus.BLOCKED
        self.error = Exception("Cancelled: upstream dependency failed")
        self.done_event.set()


@dataclass
class DAGResult:
    """Aggregated result from a DAG execution run."""
    outputs: Dict[str, Any] = field(default_factory=dict)      # step_id -> raw output
    results: Dict[str, StepResult] = field(default_factory=dict)  # step_id -> StepResult
    completed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    cancelled: List[str] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list) # Actual start order
    summary: str = ""

    @property
    def success(self) -> bool:
        return len(self.failed) == 0 and len(self.cancelled) == 0

    @property
    def failed_step_id(self) -> Optional[str]:
        return self.failed[0] if self.failed else None

    @property
    def failed_error(self) -> Optional[str]:
        return None  # populated by executor


class DAGExecutor:
    """
    Executes plan steps as a DAG with async parallelism.

    Usage:
        executor = DAGExecutor(steps, execute_fn, budget_check_fn)
        result = await executor.run(step_outputs, trace, telemetry)

    Args:
        steps: List of plan step objects (must have step_id, depends_on, skill_name)
        execute_fn: async callable(step_def, step_outputs, trace) -> StepResult
        budget_check_fn: callable() -> bool
        max_concurrent: maximum parallel steps (via Semaphore)
    """

    def __init__(
        self,
        steps: List[Any],
        execute_fn: Callable,
        budget_check_fn: Callable,
        max_concurrent: int = 5,
    ):
        self.execute_fn = execute_fn
        self.budget_check_fn = budget_check_fn
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Build node graph
        self.nodes: Dict[str, StepNode] = {}
        for step in steps:
            deps = set(step.depends_on or [])
            self.nodes[step.step_id] = StepNode(
                step_id=step.step_id,
                step_def=step,
                depends_on=deps,
            )

        # Validate graph
        self._validate()

    def _validate(self):
        """Check for missing dependencies and cycles."""
        all_ids = set(self.nodes.keys())

        for node in self.nodes.values():
            missing = node.depends_on - all_ids
            if missing:
                raise DAGError(
                    f"Step '{node.step_id}' depends on unknown steps: {missing}"
                )

        # Cycle detection via topological sort attempt
        visited = set()
        temp = set()

        def _visit(node_id: str):
            if node_id in temp:
                raise DAGError(f"Dependency cycle detected involving '{node_id}'")
            if node_id in visited:
                return
            temp.add(node_id)
            for dep_id in self.nodes[node_id].depends_on:
                _visit(dep_id)
            temp.remove(node_id)
            visited.add(node_id)

        for nid in self.nodes:
            _visit(nid)

    def _get_dependents(self, step_id: str) -> Set[str]:
        """Get all steps that transitively depend on step_id."""
        dependents = set()
        queue = [step_id]
        while queue:
            current = queue.pop()
            for nid, node in self.nodes.items():
                if current in node.depends_on and nid not in dependents:
                    dependents.add(nid)
                    queue.append(nid)
        return dependents

    async def run(
        self,
        step_outputs: Dict[str, Any],
        trace: Any,
        telemetry: Any = None,
    ) -> DAGResult:
        """
        Execute all steps in the DAG.

        Returns a DAGResult with completed/failed/cancelled info.
        """
        dag_result = DAGResult()
        dag_result.outputs = step_outputs  # shared mutable dict for cross-step refs

        # Create a task for each node
        tasks = []
        for node in self.nodes.values():
            task = asyncio.create_task(
                self._run_node(node, dag_result, trace, telemetry),
                name=f"dag-{node.step_id}",
            )
            tasks.append(task)

        # Await all — exceptions are captured inside _run_node
        await asyncio.gather(*tasks, return_exceptions=True)

        # Build summary
        parts = []
        for step_id in dag_result.completed:
            result = dag_result.results.get(step_id)
            if result:
                parts.append(f"\n\nStep {step_id}: {result}")
        for step_id in dag_result.failed:
            node = self.nodes[step_id]
            parts.append(f"\n\nStep {step_id} ({node.step_def.skill_name}): FAILED — {node.error}")
        for step_id in dag_result.cancelled:
            parts.append(f"\n\nStep {step_id}: CANCELLED (upstream failure)")

        dag_result.summary = "".join(parts)
        return dag_result

    async def _run_node(
        self,
        node: StepNode,
        dag_result: DAGResult,
        trace: Any,
        telemetry: Any,
    ):
        """Execute a single node, awaiting its dependencies first."""
        log = logger.bind(step_id=node.step_id, skill=node.step_def.skill_name)

        # 1. Wait for all dependencies
        for dep_id in node.depends_on:
            dep_node = self.nodes[dep_id]
            await dep_node.done_event.wait()

            # Check if dependency failed or was cancelled
            if dep_node.status in (StepStatus.FAILED, StepStatus.BLOCKED):
                log.warning("Skipping — upstream dependency failed", dep=dep_id)
                node.mark_cancelled()
                dag_result.cancelled.append(node.step_id)
                return

        # 2. Check if we were already cancelled (cascade)
        if node.status == StepStatus.BLOCKED:
            dag_result.cancelled.append(node.step_id)
            return

        # 3. Budget check
        if not self.budget_check_fn():
            log.error("Budget exhausted before step execution")
            node.mark_failed(BudgetExhaustedError("Budget exceeded"))
            dag_result.failed.append(node.step_id)
            return

        # 4. Acquire concurrency slot
        async with self.semaphore:
            node.status = StepStatus.RUNNING
            log.info("DAG step starting")

            try:
                # F3: Record execution order
                dag_result.execution_order.append(node.step_id)

                # 5. Execute with optional telemetry span
                if telemetry:
                    with telemetry.span(
                        f"Step.{node.step_def.skill_name}",
                        {"step_id": node.step_id, "skill": node.step_def.skill_name},
                    ):
                        step_result = await self.execute_fn(
                            node.step_def, dag_result.outputs, trace
                        )
                else:
                    step_result = await self.execute_fn(
                        node.step_def, dag_result.outputs, trace
                    )

                # 6. Check for blocked result
                if step_result.status == StepStatus.BLOCKED:
                    log.warning("Step blocked by policy")
                    node.mark_failed(Exception(f"Blocked: {step_result.output_text}"))
                    dag_result.failed.append(node.step_id)
                    return

                # 7. Success
                node.mark_done(step_result)
                dag_result.results[node.step_id] = step_result
                dag_result.completed.append(node.step_id)
                log.info("DAG step completed")

            except Exception as e:
                log.error("DAG step failed", error=str(e))
                node.mark_failed(e)
                dag_result.failed.append(node.step_id)
