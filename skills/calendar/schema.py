from typing import Literal, Optional
from pydantic import BaseModel, Field

class CalendarInput(BaseModel):
    action: Literal['create_event'] = Field(
        ..., 
        description="The action to perform. Currently only 'create_event' is supported."
    )
    summary: str = Field(
        ..., 
        description="Short title or summary of the event (e.g., 'Meeting with John')."
    )
    start_time: str = Field(
        ..., 
        description="Start time in format 'YYYY-MM-DD HH:MM:SS'. Infer from user request relative to current time."
    )
    end_time: Optional[str] = Field(
        None, 
        description="End time in format 'YYYY-MM-DD HH:MM:SS'. If not provided, defaults to 1 hour after start."
    )
    description: Optional[str] = Field(
        "", 
        description="Detailed description or notes for the event."
    )
