from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentState(BaseModel):
    """
    Global state for the SGR Agent.
    Maintains context across the reasoning loop.
    """
    user_request: str
    history: List[Message] = Field(default_factory=list)
    
    # Context variables derived from reasoning (e.g., {"ticker": "AAPL", "simulation_id": 123})
    current_context: Dict[str, Any] = Field(default_factory=dict)
    
    # List of steps completed in the current session
    completed_steps: List[str] = Field(default_factory=list)
    
    # The currently active skill (if any)
    active_skill_name: Optional[str] = None
    
    # Error state tracking
    last_error: Optional[str] = None

    def add_message(self, role: str, content: str):
        self.history.append(Message(role=role, content=content))
