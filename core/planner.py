from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.llm import LLMService

class PlanStep(BaseModel):
    step_id: str = Field(..., description="Unique identifier for this step (e.g., 'step_1')")
    skill_name: str = Field(..., description="Name of the skill to execute")
    description: str = Field(..., description="What this step does")
    params: Dict[str, Any] = Field(..., description="Parameters for the skill")
    depends_on: List[str] = Field(default_factory=list, description="List of step_ids that must complete before this one")

class ExecutionPlan(BaseModel):
    steps: List[PlanStep] = Field(default_factory=list, description="Sequence of steps to execute")
    reasoning: str = Field(..., description="Reasoning behind this plan")
    direct_response: Optional[str] = Field(None, description="If no skills are needed (chitchat), providing the response here.")

class Planner:
    def __init__(self, llm: LLMService):
        self.llm = llm

    async def create_plan(self, user_request: str, skills_desc: str, history: str) -> tuple[ExecutionPlan, dict]:
        system_prompt = (
            "You are an expert Planner for an AI Agent.\n"
            "Your goal is to break down a User Request into a sequence of steps (Execution Plan).\n"
            f"Available Skills:\n{skills_desc}\n\n"
            "Rules:\n"
            "1. If the request requires multiple steps (e.g. 'Search X then Write Y'), create a dependency chain.\n"
            "2. Ensure parameters for later steps can be derived from earlier steps or context.\n"
            "3. If the request is simple and matches a single skill, create a 1-step plan.\n"
            "4. If the request is just conversation (hello, how are you), return empty steps and a direct_response.\n"
            "5. Use 'step_1', 'step_2' etc. for step_ids.\n"
        )
        
        # We want the Planner to be smart about parameters. 
        # If a param depends on previous step's output, it might be tricky to guess NOW.
        # But our Skills usually look at History/Context too.
        
        plan, usage = await self.llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=f"Context:\n{history}\n\nUser Request: {user_request}",
            response_model=ExecutionPlan
        )
        return plan, usage

    async def repair_plan(self, original_plan: ExecutionPlan, failed_step_id: str, error: str, history: str) -> ExecutionPlan:
        """
        Generates a new plan to recover from a failed step.
        """
        system_prompt = (
            "You are an expert Planner. Evaluation Mode.\n"
            "A step in the previous plan failed. You must propose a fix.\n"
            "1. Analyze the failure.\n"
            "2. Remove the failed step or replace it with an alternative approach.\n"
            "3. Keep successful steps if they are still relevant.\n"
            "4. Return the complete updated plan (including untried steps)."
        )
        
        user_prompt = f"""
        Original Plan:
        {original_plan.model_dump_json(indent=2)}
        
        Failed Step ID: {failed_step_id}
        Error Message: {error}
        
        Execution History:
        {history}
        
        Generate a repaired Execution Plan.
        """
        
        try:
            plan, _ = await self.llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=ExecutionPlan
            )
            return plan
        except Exception as e:
            # Fallback
            return None
