"""
Skill Execution Sandbox and Safety Layer for SGR Kernel.
"""
import time
import logging
import asyncio
from typing import Any, Dict, Optional, Callable, Awaitable
from core.registry import CapabilityRegistry

logger = logging.getLogger(__name__)

class SandboxViolation(Exception):
    """Raised when a sandbox constraint is violated."""
    pass

class SkillSandbox:
    """
    Wraps skill execution to enforce timeouts, resource limits, and capability checks.
    """
    
    def __init__(
        self, 
        dry_run: bool = False,
        timeout_seconds: float = 300.0,
        allowed_capabilities: Optional[set] = None
    ):
        self.dry_run = dry_run
        self.timeout_seconds = timeout_seconds
        self.allowed_capabilities = allowed_capabilities or set()
        
    async def execute(
        self, 
        skill_name: str, 
        params: Dict[str, Any], 
        func: Callable[..., Awaitable[Any]],
        state: Any = None
    ) -> Any:
        # 1. Check Capabilities
        if not CapabilityRegistry.check_capabilities(skill_name, self.allowed_capabilities):
            raise SandboxViolation(f"Skill '{skill_name}' requires unavailable capabilities.")
            
        skill_meta = CapabilityRegistry.get_skill(skill_name)
        
        # 2. Dry Run Check
        if self.dry_run and skill_meta and skill_meta.side_effects:
            logger.info(f"[DRY-RUN] Skipping execution of {skill_name}")
            return {"status": "skipped", "reason": "dry_run mode"}
            
        # 3. Execution with Timeout
        start_time = time.time()
        
        try:
            # SGR Kernel uses async execution
            if state:
                 task = func(params, state)
            else:
                 task = func(params)
                 
            result = await asyncio.wait_for(task, timeout=self.timeout_seconds)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Skill execution timed out for {skill_name} after {self.timeout_seconds}s")
            raise SandboxViolation(f"Timeout: Skill {skill_name} exceeded {self.timeout_seconds}s limit")
            
        except SandboxViolation:
            raise
            
        except Exception as e:
            logger.error(f"Sandbox execution failed for {skill_name}: {e}")
            raise
