# SGR Skill Development Guide

This document explains how to add new capabilities ("Skills") to the Core Agent using the **Schema-Guided Reasoning (SGR)** methodology.

## Philosophy: Why SGR?
Unlike standard agents that follow text instructions (SOPs), an SGR agent follows **Data Structures**.
- **SOP**: "Sök for user, then calculate risk." (Ambiguous)
- **SGR**: "Fill this JSON Schema: `{ user_id: str, risk_score: int }`." (Structured)

## Anatomy of a Skill
Every skill consists of 3 parts in `sgr_core/skills/<skill_name>/`:

1.  **`schema.py`**: The "contract". What inputs does this skill need?
2.  **`handler.py`**: The "brain". The code that executes the logic (Python).
3.  **`__init__.py`**: Exposes the skill class.

## Step-by-Step Tutorial

### 1. Define the Schema (`schema.py`)
Think: *What information must the LLM extract from the user to do this job?*

```python
from pydantic import BaseModel, Field

class StockAnalysisInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL)")
    period: str = Field(description="Time period (1mo, 1y)", default="1mo")
```

### 2. Implement the Logic (`handler.py`)
Inherit from `BaseSkill`. Implement `execute`.

```python
from typing import Type
from ...core.state import AgentState
from ..base import BaseSkill
from .schema import StockAnalysisInput

class StockAnalystSkill(BaseSkill):
    name: str = "stock_analyst"
    description: str = "Analyzes technical indicators for a given stock ticker."

    @property
    def input_schema(self) -> Type[BaseModel]:
        return StockAnalysisInput

    def execute(self, params: StockAnalysisInput, state: AgentState) -> str:
        # Use params.ticker and params.period here
        # Return a markdown string result
        return f"Analysis for {params.ticker}: Buy (RSI < 30)"
```

### 3. Register the Skill (`main.py`)
Add it to the Core Engine at startup.

```python
from skills.stock_analyst.handler import StockAnalystSkill
engine.register_skill(StockAnalystSkill())
```

## Best Practices for "Experts"
When writing requirements for a new skill:
1.  **Be Strict with Field Descriptions**: The LLM uses the `Field(description="...")` to understand what to put there. Clear descriptions = Better performance.
2.  **Return Markdown**: The `execute` method should return clean, readable Markdown that the user will see.
3.  **Handle Errors**: Wrap API calls in try/except blocks and return a friendly error message string.
