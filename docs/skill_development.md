# Skill Development Guide (v1.x)

## Overview
A **Skill** is an atomic functional block that the SGR Kernel can execute. Every skill must implement the standard `Skill` interface and handle a `SkillContext`.

## 1. Skill Interface
Your skill should inherit from `core.skill_interface.Skill` and define basic metadata:

```python
from core.skill_interface import Skill, SkillContext, SkillResult

class MyNewSkill(Skill):
    @property
    def name(self) -> str:
        return "my_feature_skill"

    @property
    def capabilities(self) -> Set[str]:
        # Declare what this skill needs to work
        return {"NET", "IO"}

    async def execute(self, ctx: SkillContext) -> SkillResult:
        # Access config from ctx.config
        target = ctx.config.get("target")
        
        # Perform logic
        # ...
        
        return SkillResult(output=f"Success: {target}")
```

## 2. Skill Context
The `SkillContext` provides the skill with everything it needs without granting direct access to the Kernel:
- `ctx.config`: Dictionary of parameters resolved from the Step's `inputs_template`.
- `ctx.execution_state`: Read-only access to global state.
- `ctx.llm_service`: Pre-configured LLM client (intercepted by ReplayEngine).
- `ctx.tool_registry`: Ability to look up other skills (if needed for composition).

## 3. Registration
Register your skill in the `CoreEngine` before starting a run:

```python
engine = CoreEngine()
engine.register_skill(MyNewSkill())
```

## 4. Best Practices
- **Idempotency**: If your skill has side-effects, attempt to make it idempotent so the Kernel can safely retry it.
- **Output Schema**: Return structured data (Dict/Pydantic) whenever possible to facilitate automated verification by the Critic.
- **Error Handling**: Raise clear exceptions. The Reliability Engine will classify them automatically.
