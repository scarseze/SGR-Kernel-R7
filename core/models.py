from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Callable, Awaitable

class Tool(BaseModel):
    """
    Represents a tool (function) that can be called by an LLM.
    Used primarily by the MCP Adapter to wrap external tools.
    """
    name: str
    description: str
    parameters: Dict[str, Any] # JSON Schema
    handler: Callable[[Dict[str, Any]], Awaitable[str]]

    class Config:
        arbitrary_types_allowed = True
