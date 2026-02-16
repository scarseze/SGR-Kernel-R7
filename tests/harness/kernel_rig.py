"""
KernelTestRig — fluent harness for kernel compliance and integration tests.

Bypasses heavy CoreEngine.__init__ (DB, Qdrant, Ollama, RAG) via __new__,
wires only attributes needed by run() and _execute_step().

Usage:
    rig = KernelTestRig()
    rig.with_skill(FakeSkill(fail_times=2))
    result = await rig.run("test query")
    rig.assert_skill_calls("fake", 3)
"""
import asyncio
from typing import List, Optional, Any
from unittest.mock import MagicMock, AsyncMock

from core.engine import CoreEngine
from core.planner import ExecutionPlan, PlanStep
from core.trace import RequestTrace, TraceManager
from core.state import AgentState
from core.security import SecurityGuardian
from core.middleware import (
    TraceMiddleware, PolicyMiddleware,
    ApprovalMiddleware, TimeoutMiddleware,
)

from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_planner import FakePlanner
from tests.fakes.fake_policy import FakePolicy


class _NoOpTraceManager:
    """TraceManager that doesn't write to filesystem."""
    def save_trace(self, trace: RequestTrace):
        self._last = trace

    def last_trace(self) -> Optional[RequestTrace]:
        return getattr(self, "_last", None)


class KernelTestRig:
    """Fluent builder for kernel test scenarios."""

    def __init__(self):
        # Bypass __init__ — no DB, no Qdrant, no Ollama
        eng = object.__new__(CoreEngine)

        # --- Wire minimal attributes needed by run() ---
        eng.user_id = "test_user"
        eng.approval_callback = None
        eng.state = AgentState(user_request="")
        eng.context_loaded = True  # skip _ensure_initialized

        # Subsystems — fakes
        eng.llm = FakeLLM()
        eng.planner = FakePlanner()
        eng.policy = FakePolicy()
        eng.security = SecurityGuardian()
        eng.tracer = _NoOpTraceManager()

        # No-op DB session
        eng.db = MagicMock()
        eng.db.session = MagicMock(return_value=_AsyncCtx())

        # No-op memory
        eng.memory_manager = MagicMock()
        eng.memory_manager.augment_with_semantic_search = AsyncMock()

        # Skills
        eng.skills = {}
        eng.cap_index = {}
        eng.rag = None
        eng._skill_semaphores = {}

        # Task queue
        eng.task_queue = MagicMock()
        eng.task_handlers = {}

        # Middleware — production ordering
        eng.middlewares = [
            TraceMiddleware(),
            PolicyMiddleware(eng.policy),
            ApprovalMiddleware(eng.approval_callback),
            TimeoutMiddleware(),
        ]

        # No-op message saving
        eng._save_message = AsyncMock()
        eng._ensure_initialized = AsyncMock()

        self.engine = eng
        self._skills = []

    # ======================== Configuration ========================

    def with_skill(self, skill) -> "KernelTestRig":
        """Register a skill (real or fake)."""
        self.engine.register_skill(skill)
        self._skills.append(skill)
        return self

    def with_skills(self, skills: List) -> "KernelTestRig":
        for s in skills:
            self.with_skill(s)
        return self

    def with_plan(self, plan: ExecutionPlan) -> "KernelTestRig":
        """Override planner output."""
        self.engine.planner.plan = plan
        return self

    def with_steps(self, steps: List[PlanStep]) -> "KernelTestRig":
        """Shorthand: build plan from steps."""
        plan = ExecutionPlan(steps=steps, reasoning="test")
        return self.with_plan(plan)

    def with_direct_response(self, text: str) -> "KernelTestRig":
        """Planner returns direct response (no DAG)."""
        self.engine.planner.direct_response = text
        return self

    def with_budget(self, ok: bool) -> "KernelTestRig":
        """Control budget gate."""
        self.engine.policy.budget_ok = ok
        return self

    def with_middlewares(self, mws: List) -> "KernelTestRig":
        """Replace middleware stack."""
        self.engine.middlewares = mws
        return self

    def without_security(self) -> "KernelTestRig":
        """Replace security with no-op for isolated tests."""
        noop_sec = MagicMock()
        noop_sec.validate = MagicMock()
        noop_sec.validate_params = MagicMock()
        noop_sec.validate_output = MagicMock()
        self.engine.security = noop_sec
        return self

    # ======================== Execution ========================

    async def run(self, text: str = "test query") -> str:
        """Execute full run() pipeline."""
        return await self.engine.run(text)

    async def run_step(self, step_def: PlanStep,
                       outputs: dict = None) -> Any:
        """Execute a single step directly (bypass planner/DAG)."""
        trace = RequestTrace(user_request="direct_step")
        outputs = outputs or {}
        return await self.engine._execute_step(step_def, outputs, trace)

    # ======================== Inspection ========================

    @property
    def last_trace(self) -> Optional[RequestTrace]:
        return self.engine.tracer.last_trace()

    @property
    def last_step(self):
        tr = self.last_trace
        if not tr or not tr.steps:
            return None
        return tr.steps[-1]

    @property
    def planner_calls(self) -> int:
        return self.engine.planner.create_plan_calls

    @property
    def policy(self) -> FakePolicy:
        return self.engine.policy

    # ======================== Assertions ========================

    def assert_skill_calls(self, skill, expected: int):
        """Assert a FakeSkill was called N times."""
        assert skill.calls == expected, \
            f"{skill.name}: expected {expected} calls, got {skill.calls}"

    def assert_step_status(self, expected_status):
        """Assert last step trace has expected status."""
        step = self.last_step
        assert step is not None, "No step trace found"
        assert step.status == expected_status, \
            f"Expected {expected_status}, got {step.status}"

    def assert_planner_not_called(self):
        assert self.planner_calls == 0, \
            f"Planner was called {self.planner_calls} time(s)"

    def assert_planner_called(self, times: int = 1):
        assert self.planner_calls == times, \
            f"Expected {times} planner call(s), got {self.planner_calls}"


class _AsyncCtx:
    """Fake async context manager for db.session()."""
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        pass
