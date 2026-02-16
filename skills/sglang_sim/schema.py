from typing import List
from pydantic import BaseModel, Field

class ClientData(BaseModel):
    name: str = Field(description="Full name of the client")
    salary: str = Field(description="Monthly income string (e.g. '150 000 руб')")
    credit_history: str = Field(description="Summary of credit history")

class RiskAnalysisInput(BaseModel):
    """
    Input for the Credit Risk Analysis simulation.
    Takes a list of clients to process.
    """
    clients: List[ClientData] = Field(description="List of clients to analyze")
