from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field

class PerformAnalysis(BaseModel):
    file_name: str = Field(..., description="Name of the file to analyze (must exist in generated_files directory), e.g., 'sales.xlsx'.")
    action: Literal["summarize", "chart"] = Field(..., description="The action to perform: 'summarize' for data overview, 'chart' for visualization.")
    
    # Chart specific fields
    chart_type: Optional[Literal["bar", "line", "pie", "scatter", "hist"]] = Field(None, description="Type of chart to generate (required for action='chart').")
    x_column: Optional[str] = Field(None, description="Column name for the X-axis.")
    y_column: Optional[str] = Field(None, description="Column name for the Y-axis (for bar/line/scatter).")
    title: Optional[str] = Field(None, description="Title of the chart.")
    
    # Analysis specific
    query: Optional[str] = Field(None, description="Optional specific question about the data, e.g. 'sum of sales by region'.")
