from typing import Set, Dict, Any
from pydantic import BaseModel, Field
from core.skill_interface import Skill, SkillContext, SkillResult, StepStatus

# 1. Input Schema
class GreetingParams(BaseModel):
    name: str = Field(..., description="Name of the person to greet")
    loud: bool = Field(False, description="If true, returns uppercase greeting")

# 2. Skill Definition
class MinimalGreetingSkill(Skill):
    """
    A minimal example skill that generates a greeting.
    Shows the standard v1.x contract: pure logic, typed inputs, structured output.
    """

    @property
    def name(self) -> str:
        return "minimal_greeting"

    @property
    def capabilities(self) -> Set[str]:
        # This skill is pure logic, so it needs no special capabilities.
        # If it needed network, we'd return {"NET"}
        return set()

    async def execute(self, ctx: SkillContext) -> SkillResult:
        # 3. Validation is automatic via Pydantic using ctx.config
        params = GreetingParams(**ctx.config)

        # 4. Core Logic
        message = f"Hello, {params.name}!"
        if params.loud:
            message = message.upper()

        # 5. Return Structured Result
        return SkillResult(
            output={"message": message, "timestamp": "2026-02-18"},
            output_text=message,
            status=StepStatus.COMPLETED,
            artifacts=[]
        )

# 6. Usage Example (Pseudo-code for docs)
# engine.register_skill(MinimalGreetingSkill())
