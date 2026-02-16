from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import json
from core.types import StepStatus

class StepResult(BaseModel):
    """
    Standard result object for a skill execution step.
    Allows passing structured data between steps.
    """
    data: Any = Field(..., description="The main output (str, dict, list, model)")
    artifacts: List[str] = Field(default_factory=list, description="Paths to generated files")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution stats/meta")
    output_text: Optional[str] = Field(None, description="Human-readable summary or full text output")
    status: StepStatus = Field(default=StepStatus.COMPLETED, description="Execution status")
    
    def __str__(self):
        """
        String representation for logging or simple string-based consumers.
        Prioritizes output_text if present.
        """
        if self.output_text:
            return self.output_text
            
        if isinstance(self.data, str):
            return self.data
        return str(self.data)

    def trace_preview(self, max_len: int = 1000) -> str:
        """
        Produce a trace-safe, truncated preview of the result data.
        Uses json.dumps for structured types to avoid repr garbage.
        """
        try:
            if isinstance(self.data, str):
                return self.data[:max_len]
            elif isinstance(self.data, (dict, list)):
                return json.dumps(self.data, ensure_ascii=False, default=str)[:max_len]
            elif hasattr(self.data, 'model_dump'):
                return json.dumps(self.data.model_dump(), ensure_ascii=False, default=str)[:max_len]
            else:
                return str(self.data)[:max_len]
        except Exception:
            return str(self.data)[:max_len]

    def sanitized_copy(self) -> "StepResult":
        """Return a copy with data/output sanitized but status and metadata preserved."""
        return StepResult(
            data="[Output sanitized by security policy]",
            output_text="[Output sanitized by security policy]",
            status=self.status,
            artifacts=self.artifacts,
            metadata={**self.metadata, "_sanitized": True},
        )
