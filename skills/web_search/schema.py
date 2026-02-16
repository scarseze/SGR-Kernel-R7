from typing import Optional
from pydantic import BaseModel, Field

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to find information about (e.g. 'Bitcoin price', 'Python 3.14 release date').")
    max_results: int = Field(default=5, description="Number of results to return.")
    region: str = Field(default="wt-wt", description="Region code (e.g. 'us-en', 'ru-ru', 'wt-wt' for worldwide).")
