from typing import Optional
from pydantic import BaseModel, Field

class RLMInput(BaseModel):
    """
    Input schema for the Recursive Language Model (RLM) Skill.
    Use this skill when you need to answer complex queries over specific long documents/contexts.
    """
    query: str = Field(description="The question or task to perform on the context.")
    context_text: Optional[str] = Field(default=None, description="Use this ONLY for short raw text typed by the user. NEVER populate this from message history or if a file is mentioned.")
    context_file_path: Optional[str] = Field(default=None, description="Absolute path to the file to analyze. REQUIRED if user mentions a file. This is the preferred method for long content.")
    max_iterations: int = Field(default=10, description="Maximum number of RLM recursive steps.")
