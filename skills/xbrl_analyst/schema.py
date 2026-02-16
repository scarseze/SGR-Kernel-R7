from typing import List, Optional
from pydantic import BaseModel, Field

class XBRLInput(BaseModel):
    """
    Input for the XBRL Analyst Skill.
    Parses an XBRL file and calculates financial metrics.
    """
    file_path: str = Field(
        ..., 
        description="The path to the .xbrl file to analyze. Can be absolute or relative to project root."
    )
    
    extract_all: bool = Field(
        default=False, 
        description="If True, returns all parsed facts. If False, returns specific key metrics."
    )
