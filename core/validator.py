"""
Reasoning Contract Enforcement Engine for SGR Kernel.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.execution import PlanIR, PlanStep
from core.policy_graph import FailureReason

class ValidationResult(BaseModel):
    """Result of reasoning validation."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence: float = 1.0

class ReasoningValidator:
    """
    Enforces reasoning contracts, ensuring plans are logically sound
    and safe before execution.
    """
    
    def validate_plan(self, plan: PlanIR) -> ValidationResult:
        """
        Validate a complete execution plan.
        """
        errors = []
        warnings = []
        
        # 1. Check for empty plan
        if not plan.steps:
            errors.append("Plan contains no steps.")
            return ValidationResult(valid=False, errors=errors)
            
        # 2. Dependency Loop Detection
        try:
            self._check_dependency_loops(plan.steps)
        except ValueError as e:
            errors.append(str(e))
            
        # 3. Step validation
        for step in plan.steps:
            step_errors = self._validate_step(step)
            errors.extend(step_errors)
            
        return ValidationResult(
            valid=len(errors) == 0, 
            errors=errors, 
            warnings=warnings
        )
    
    def _validate_step(self, step: PlanStep) -> List[str]:
        """Validate an individual step."""
        errors = []
        
        # Check required fields
        if not step.type:
            errors.append(f"Step {step.id}: Missing type.")
            
        if not step.name:
            errors.append(f"Step {step.id}: Missing name.")
            
        # Check capability requirements (mock check for now)
        # In real impl, would checking against available resources
        pass
        
        return errors

    def _check_dependency_loops(self, steps: List[PlanStep]) -> None:
        """
        Verify that the dependency graph is a DAG (no cycles).
        """
        # Build graph
        adj = {s.id: s.dependencies for s in steps}
        
        # Simple DFS for cycle detection
        visited = set()
        recursion_stack = set()
        
        def visit(node_id):
            visited.add(node_id)
            recursion_stack.add(node_id)
            
            for neighbor in adj.get(node_id, []):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(node_id)
            return False
            
        for step_id in adj:
            if step_id not in visited:
                if visit(step_id):
                    raise ValueError("Dependency cycle detected in plan.")
