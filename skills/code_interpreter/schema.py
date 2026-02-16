from typing import Optional, List
from pydantic import BaseModel, Field

class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="The Python code to execute.")
    language: str = Field("python", description="The programming language (default: python).")
    timeout: int = Field(30, description="Execution timeout in seconds.")
    packages: Optional[List[str]] = Field(None, description="List of pip packages required (e.g. ['pandas', 'numpy'])")

class CodeExecutionResult(BaseModel):
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    artifact_paths: Optional[List[str]] = None
