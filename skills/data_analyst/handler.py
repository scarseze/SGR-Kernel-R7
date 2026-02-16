import os
import io
import structlog
from typing import Dict, Any, List, Union

# Safe imports
try:
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg') # Non-GUI backend for Docker
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pd = None
    plt = None
    sns = None

from skills.base import BaseSkill, SkillMetadata
from skills.data_analyst.schema import PerformAnalysis

logger = structlog.get_logger(__name__)

class DataAnalystSkill(BaseSkill):
    name = "data_analyst"
    description = "Analyzes data files (Excel, CSV) and generates charts."

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            capabilities=["data_analysis", "charts", "csv", "excel"],
            risk_level="low",
            side_effects=True,
            idempotent=True,
            requires_network=False,
            requires_filesystem=True,
            cost_class="medium"
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return PerformAnalysis

    def is_sensitive(self, params: Any) -> bool:
        """Reading data is generally safe, but we can flag it if needed."""
        return False

    async def execute(self, params: Any, state: Any = None) -> str:
        if pd is None:
            return "Error: Data Analysis libraries (pandas, matplotlib) are not installed. Please rebuild the container."

        try:
            # Handle input types
            if isinstance(params, dict):
                req = PerformAnalysis(**params)
            else:
                req = params
            
            # Locate file
            output_dir = "generated_files"
            filepath = os.path.join(output_dir, req.file_name)
            
            if not os.path.exists(filepath):
                 return f"Error: File '{req.file_name}' not found in {output_dir}. Please generate or upload it first."

            # Load Data
            df = self._load_data(filepath)
            
            if req.action == "summarize":
                return self._summarize_data(df, req.file_name)
            elif req.action == "chart":
                return self._generate_chart(df, req, output_dir)
            else:
                return f"Error: Unknown action '{req.action}'"

        except Exception as e:
            logger.error("data_analyst_error", error=str(e))
            return f"Error performing data analysis: {str(e)}"

    def _load_data(self, filepath: str):
        if filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")

    def _summarize_data(self, df, filename: str) -> str:
        summary = f"### Data Analysis: {filename}\n\n"
        
        # Shape
        rows, cols = df.shape
        summary += f"**Dimensions**: {rows} rows, {cols} columns\n\n"
        
        # Columns
        summary += "**Columns**:\n"
        for col in df.columns:
            dtype = df[col].dtype
            summary += f"- `{col}` ({dtype})\n"
        
        # Basic Stats (numeric)
        desc = df.describe().to_markdown()
        summary += f"\n**Statistics**:\n{desc}\n\n"
        
        # HEAD
        head = df.head(5).to_markdown()
        summary += f"**First 5 rows**:\n{head}"
        
        return summary

    def _generate_chart(self, df, req: PerformAnalysis, output_dir: str) -> str:
        plt.figure(figsize=(10, 6))
        
        # Plotting logic
        try:
            if req.chart_type == "bar":
                if not req.x_column or not req.y_column:
                    return "Error: Bar chart requires x_column and y_column."
                sns.barplot(data=df, x=req.x_column, y=req.y_column)
            
            elif req.chart_type == "line":
                if not req.x_column or not req.y_column:
                    return "Error: Line chart requires x_column and y_column."
                sns.lineplot(data=df, x=req.x_column, y=req.y_column)
            
            elif req.chart_type == "scatter":
                 if not req.x_column or not req.y_column:
                    return "Error: Scatter chart requires x_column and y_column."
                 sns.scatterplot(data=df, x=req.x_column, y=req.y_column)
            
            elif req.chart_type == "hist":
                if not req.x_column:
                     return "Error: Histogram requires x_column."
                sns.histplot(data=df, x=req.x_column)
                
            elif req.chart_type == "pie":
                 # Basic pie chart using pandas/matplotlib directly
                 if not req.x_column or not req.y_column:
                      return "Error: Pie chart requires labels (x) and values (y)."
                 plt.pie(df[req.y_column], labels=df[req.x_column], autopct='%1.1f%%')
            
            else:
                return f"Error: Unsupported chart type '{req.chart_type}'"

            # Customization
            if req.title:
                plt.title(req.title)
            elif req.x_column:
                 plt.title(f"{req.chart_type.capitalize()} of {req.y_column} by {req.x_column}")
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save
            chart_filename = f"{req.file_name.split('.')[0]}_{req.chart_type}.png"
            chart_path = os.path.join(output_dir, chart_filename)
            plt.savefig(chart_path)
            plt.close()
            
            return f"✅ Chart generated: [{chart_filename}](file:///{chart_path.replace(os.sep, '/')})"
            
        except Exception as e:
            plt.close() # Ensure cleanup
            raise e
