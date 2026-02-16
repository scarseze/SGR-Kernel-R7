from pydantic import BaseModel, Field

class LogicRLInput(BaseModel):
    problem: str = Field(description="The logic problem or puzzle to solve.")
    max_retries: int = Field(default=5, description="Maximum number of refinement iterations.")
