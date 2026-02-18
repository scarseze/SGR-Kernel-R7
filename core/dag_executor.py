"""
ExecutionGraphEngine for SGR Kernel.
Responsible for DAG scheduling and dependency resolution.
"""
from typing import List, Set, Dict, Optional
from core.execution import ExecutionState, StepNode, StepStatus, PlanIR

class ExecutionGraphEngine:
    """
    Manages the execution flow of the PlanIR DAG.
    """
    
    def __init__(self, state: ExecutionState):
        self.state = state
        self.plan = state.plan_ir
        if not self.plan:
             raise ValueError("ExecutionState has no PlanIR")
        
        # Build LUTs
        self.steps_lut = {s.id: s for s in self.plan.steps}
        self.incoming_edges: Dict[str, List[str]] = {} # target -> [sources]
        
        for edge in self.plan.edges:
            self.incoming_edges.setdefault(edge.target_id, []).append(edge.source_id)

    def is_complete(self) -> bool:
        """Check if all steps are in a terminal state."""
        for step in self.plan.steps:
            s_state = self.state.step_states.get(step.id)
            if not s_state:
                return False # Not even started
            if s_state.status not in [StepStatus.COMMITTED, StepStatus.FAILED]:
                return False
        return True

    def get_runnable_steps(self) -> List[StepNode]:
        """
        Identify steps that are:
        1. PENDING
        2. Have all dependencies satisfied (COMMITTED)
        """
        runnable = []
        for step in self.plan.steps:
            # Check current status
            s_state = self.state.step_states.get(step.id)
            if not s_state or s_state.status != StepStatus.PENDING:
                continue
                
            # Check dependencies
            deps_met = True
            params_ready = True # In future, check params availability
            
            dependencies = self.incoming_edges.get(step.id, [])
            for source_id in dependencies:
                source_state = self.state.step_states.get(source_id)
                # Strict dependency: Source must be COMMITTED
                if not source_state or source_state.status != StepStatus.COMMITTED:
                    deps_met = False
                    break
            
            if deps_met:
                runnable.append(step)
                
        return runnable

    def mark_step_running(self, step_id: str):
        if step_id in self.state.step_states:
            self.state.step_states[step_id].status = StepStatus.RUNNING
