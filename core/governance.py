"""
GovernanceHooksBus for SGR Kernel.
Side-effect isolated event bus for policy hooks.
"""
import asyncio
import logging
from typing import Callable, List, Dict, Any, Awaitable
from core.execution import ExecutionState

logger = logging.getLogger(__name__)

# Hook Types
HOOK_BEFORE_STEP = "before_step"
HOOK_AFTER_STEP = "after_step"
HOOK_BEFORE_LLM = "before_llm"
HOOK_AFTER_LLM = "after_llm"
HOOK_ON_FAILURE = "on_failure"
HOOK_ON_RETRY = "on_retry"
HOOK_ON_ESCALATION = "on_escalation"
HOOK_ON_REPAIR = "on_repair"
HOOK_ON_COMMIT = "on_commit"

class GovernanceHooksBus:
    """
    Manages registration and dispatch of governance hooks.
    """
    def __init__(self):
        self.hooks: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {
            HOOK_BEFORE_STEP: [],
            HOOK_AFTER_STEP: [],
            HOOK_BEFORE_LLM: [],
            HOOK_AFTER_LLM: [],
            HOOK_ON_FAILURE: [],
            HOOK_ON_RETRY: [],
            HOOK_ON_ESCALATION: [],
            HOOK_ON_REPAIR: [],
            HOOK_ON_COMMIT: []
        }

    def register(self, hook_type: str, handler: Callable[[Any], Awaitable[None]]):
        if hook_type in self.hooks:
            self.hooks[hook_type].append(handler)
        else:
            logger.warning(f"Unknown hook type: {hook_type}")

    async def emit(self, hook_type: str, *args, **kwargs):
        """
        Dispatch event to all registered hooks.
        Ensure isolation: exceptions in hooks do not crash the kernel.
        """
        if hook_type not in self.hooks:
            return

        for handler in self.hooks[hook_type]:
            try:
                # Timeout bounded execution (e.g. 1s)
                await asyncio.wait_for(handler(*args, **kwargs), timeout=1.0)
            except asyncio.TimeoutError:
                logger.error(f"Hook {handler} timed out")
            except Exception as e:
                logger.error(f"Hook {handler} failed: {e}")
