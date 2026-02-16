from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class PortfolioInput(BaseModel):
    """
    Input for the Portfolio Management Skill.
    Can analyze financial assets or search the internal knowledge base (RAG).
    """
    action: Literal["analyze", "search"] = Field(
        ..., 
        description="The action to perform: 'analyze' for stock metrics, 'search' for market news/knowledge."
    )
    
    tickers: List[str] = Field(
        default=[], 
        description="List of tickers (e.g. 'SBER.MOEX', 'AAPL.US') for analysis."
    )
    
    search_query: str = Field(
        default="", 
        description="The query string for RAG search (only if action='search')."
    )
    
    period: str = Field(
        default="1y", 
        description="Historical period for analysis (e.g., '1y', '5y')."
    )
