"""
Skill Runtime Adapter for SGR Kernel (Core Architectural Model).
Isolates Kernel from Skill Implementation details.
"""
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Optional, Callable
from pydantic import BaseModel, Field

from core.execution import ExecutionState

class SkillContext:
    """
    Context passed to a skill during execution.
    Provides safe access to kernel services.
    """
    def __init__(
        self,
        execution_state: ExecutionState,
        llm_service: Any, # Placeholder
        tool_registry: Any, # Placeholder
        config: Dict[str, Any],
        approval_api: Optional[Callable[[str], bool]] = None,
        checkpoint_api: Optional[Callable[[ExecutionState], str]] = None
    ):
        self.execution_state = execution_state
        self.llm = llm_service
        self.tool_registry = tool_registry
        self.config = config
        self.approval_api = approval_api
        self.checkpoint_api = checkpoint_api

class SkillResult(BaseModel):
    """
    Standardized result from a skill execution.
    """
    output: Any
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    side_effects: List[str] = Field(default_factory=list)
    telemetry: Dict[str, Any] = Field(default_factory=dict)

class Skill(ABC):
    """
    Abstract Base Class for SGR Skills.
    
    API STABILITY: STABLE (v1.x)
    """
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> Set[str]:
        pass

    @abstractmethod
    async def execute(self, ctx: SkillContext) -> SkillResult:
        pass

class SkillRuntimeAdapter:
    """
    Executes skills securely and uniformly.
    """
    def __init__(self, skill_registry: Dict[str, Skill]):
        self.registry = skill_registry

    async def execute_skill(
        self, 
        skill_name: str, 
        ctx: SkillContext
    ) -> SkillResult:
        """
        Execute a skill by name with the given context.
        Handles missing skills, exceptions, and result normalization.
        """
        skill = self.registry.get(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not found in registry.")
            
        # P2: Validate Capabilities? 
        # Done in PREPARE phase typically, but can double check here.
        
        start_time = time.time()
        try:
            # EXECUTE
            result = await skill.execute(ctx)
            
            # Post-processing / Normalization
            if not isinstance(result, SkillResult):
                # Auto-wrap raw output (legacy support)
                result = SkillResult(output=result)
                
            return result
            
        except Exception as e:
            # Log error?
            # Re-raise for Kernel to handle as Failure
            raise e
        finally:
             duration = time.time() - start_time
             # Telemetry emission here?
