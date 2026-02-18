"""
RuntimeKernel for SGR Kernel (Core Architectural Model).
Main orchestrator.

API STABILITY: STABLE (v1.x)
"""
import os
from typing import Dict, Callable, Awaitable, Any, Optional, List
import uuid
import time
import asyncio
import logging
from core.execution import ExecutionState, PlanIR, StepNode, DependencyEdge, StepStatus
from core.dag_executor import ExecutionGraphEngine
from core.lifecycle import StepLifecycleEngine
from core.skill_interface import SkillRuntimeAdapter, Skill
from core.reliability import ReliabilityEngine
from core.checkpoints import CheckpointManager
from core.replay import ReplayEngine
from core.critic import CriticEngine
from core.repair import RepairEngine
from core.governance import GovernanceHooksBus
from core.telemetry_kernel import KernelTelemetry
from core.planner import Planner # Legacy
from core.llm import LLMService, ModelPool
from core.database import Database
from core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

# @stable v1.x
class CoreEngine:
    """
    The main entry point for the SGR Kernel.
    Orchestrates the entire lifecycle of an agent execution:
    Planning -> Graph -> Step(Lifecycle) -> State Update.
    """
    
    VERSION = "1.0.0-rc1"

    def __init__(self, 
                 llm_config: Optional[Dict[str, Any]] = None,
                 approval_callback: Optional[Callable[[str], Awaitable[bool]]] = None):
        logger.info("Initializing RuntimeKernel (Core Model)...")
        self._llm_config = llm_config or {}
        
        # Core Components (Modular v2 Architecture)
        self.planner = Planner(self._llm_config) # Changed from PlannerAgent to Planner to match imports
        self.dag = ExecutionGraphEngine(ExecutionState()) # Changed from DAGExecutor to ExecutionGraphEngine to match imports, requires state
        self.lifecycle = StepLifecycleEngine(
            skill_adapter=SkillRuntimeAdapter({}), # Placeholder, will be updated
            reliability=ReliabilityEngine(),
            critic=CriticEngine(ModelPool(self._llm_config).heavy), # Requires ModelPool
            repair=RepairEngine(ModelPool(self._llm_config).heavy), # Requires ModelPool
            hooks=GovernanceHooksBus()
        )
        self.reliability = ReliabilityEngine() # Redundant if passed to lifecycle, but keeping for now
        self.replay = ReplayEngine()
        
        # State
        self.execution_state: Optional[ExecutionState] = None
        self._approval_callback = approval_callback

        # Original components that are still needed or need to be re-integrated
        self.db = Database()
        self.model_pool = ModelPool(self._llm_config, replay_engine=self.replay) # Re-initialize model_pool
        self.checkpoints = CheckpointManager()
        self.telemetry = KernelTelemetry()
        # 4. Skill Registry & Adapter
        self.skills: Dict[str, Skill] = {}
        # Adapter initialized later when skills are registered? 
        # Or we keep a dynamic registry inside adapter?
        # Adapter takes a dict.
        self.skill_adapter = SkillRuntimeAdapter(self.skills)
        
        # 5. Lifecycle Manager
        self.lifecycle = StepLifecycleEngine(
            skill_adapter=self.skill_adapter,
            reliability=self.reliability,
            critic=self.critic,
            repair=self.repair,
            hooks=self.hooks
        )

    def abort(self, reason: str = "Manual Abort"):
        """
        Abort the current execution (Release Gate v1).
        """
        # We need access to current state. 
        # In a real async engine, we'd have a registry of active states.
        # For this MVP, we assume single-threaded run() blocking, 
        # so this method would be called from a signal handler or another thread/task.
        # But we don't have 'self.current_state' stored on 'self'.
        # We should store it.
        if hasattr(self, 'current_state') and self.current_state:
            self.current_state.status = ExecutionStatus.ABORTED
            logger.warning(f"Aborting execution: {reason}")
            # Persist abort state
            self.checkpoints.save_checkpoint(self.current_state, "aborted")

    def register_skill(self, skill: Skill):
        self.skills[skill.name] = skill

    async def run(self, user_input: str) -> str:
        """
        Main Execution Entry Point.
        Request -> Planner -> Graph -> Lifecycle
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # 1. Initialize State
        state = ExecutionState(
            request_id=request_id,
            input_payload=user_input
        )
        self.current_state = state
        
        # 2. Plan (Simulated for now, usually via Planner)
        # TODO: Integrate real Planner -> PlanIR conversion
        # For now, we assume simple plan or direct response
        # Using legacy planner adapter from previous step? 
        # Let's use the one I wrote in previous turn but adapted
        
        plan_ir = await self._generate_plan(user_input)
        state.plan_ir = plan_ir
        if plan_ir:
            state.plan_id = plan_ir.id
            # Initialize Step States from Plan
            for step in plan_ir.steps:
                state.initialize_step(step.id)
            
        # Checkpoint: After Planner
        self.checkpoints.save_checkpoint(state, "planner_output")
        
        # 3. Execution Graph
        if not plan_ir or not plan_ir.steps:
            return "No plan generated."
            
        return await self._execute_loop(state)

    async def resume(self, request_id: str) -> str:
        """
        Resume execution from the latest checkpoint (Crash Recovery).
        """
        path = self.checkpoints.get_latest_checkpoint(request_id)
        if not path:
            return f"No checkpoint found for request {request_id}"
            
        state, _ = self.checkpoints.load_checkpoint(path)
        self.current_state = state
        logger.info(f"♻️ Resuming request {request_id} from checkpoint {os.path.basename(path)}")
        
        return await self._execute_loop(state)

    async def _execute_loop(self, state: ExecutionState) -> str:
        """
        Core DAG Execution Loop.
        """
        graph_engine = ExecutionGraphEngine(state)
        
        while not graph_engine.is_complete():
            # Check Abort (Release Gate v1)
            if state.status == "ABORTED": # String check as per enum
                logger.info("Execution loop halted due to ABORT.")
                break

            # Get Runnable Steps
            runnable = graph_engine.get_runnable_steps()
            
            if not runnable:
                # Deadlock or partial failure
                state.status = "FAILED"
                break
            
            # Execute Steps (Sequential MVP)
            for step in runnable:
                graph_engine.mark_step_running(step.id)
                await self.lifecycle.run_step(step, state)
                
                # Checkpoint: After Step Success
                self.checkpoints.save_checkpoint(state, "step_complete")
                
                # Check for runtime abort during step (if lifecycle yields)
                if state.status == "ABORTED":
                    break
        
        # 4. Final Commit
        if state.status not in ["ABORTED", "FAILED"]:
            state.status = "COMPLETED"
            
        # Checkpoint: Final
        self.checkpoints.save_checkpoint(state, "completed")
            
        return self._summarize_result(state)

    async def _generate_plan(self, user_input: str) -> PlanIR:
        # Wrapper around legacy planner to produce PlanIR
        # Placeholder for demonstration
        skills_desc = "\n".join([s.name for s in self.skills.values()])
        legacy_plan, _ = await self.planner.create_plan(user_input, skills_desc, "")
        
        steps = []
        edges = []
        for p_step in legacy_plan.steps:
             node = StepNode(
                 id=p_step.step_id,
                 skill_name=p_step.skill_name,
                 inputs_template=p_step.params, # Treating params as template
                 description=p_step.description
             )
             steps.append(node)
             for dep in p_step.depends_on:
                 edges.append(DependencyEdge(source_id=dep, target_id=node.id))
                 
        return PlanIR(steps=steps, edges=edges)

    def _summarize_result(self, state: ExecutionState) -> str:
        # Collect outputs
        results = []
        for step_id, output in state.skill_outputs.items():
            results.append(f"Step {step_id}: {output}")
        return "\n".join(results)
