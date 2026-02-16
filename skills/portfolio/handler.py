import sys
import os
import requests
from typing import Type
from pydantic import BaseModel

from core.state import AgentState
from skills.base import BaseSkill, SkillMetadata
from skills.portfolio.schema import PortfolioInput

# Optional imports with graceful fallback
try:
    import okama as ok
except ImportError:
    ok = None

class PortfolioSkill(BaseSkill):
    name: str = "portfolio_manager"
    description: str = (
        "Financial analysis tool. Can calculate risk/return metrics for assets (tickers) "
        "and search for market information in the internal knowledge base (RAG)."
    )
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            capabilities=["finance", "portfolio_analysis", "rag"],
            risk_level="low",
            side_effects=False,
            idempotent=True,
            requires_network=True,
            requires_filesystem=False,
            cost_class="cheap"
        )

    def __init__(self, **data):
        super().__init__(**data)
        # RAG is injected by Engine
        self.rag = None

    @property
    def input_schema(self) -> Type[BaseModel]:
        return PortfolioInput

    async def execute(self, params: PortfolioInput, state: AgentState) -> str:
        if params.action == "analyze":
            return self._analyze_assets(params.tickers, params.period)
        elif params.action == "search":
            return self._search_rag(params.search_query)
        else:
            return f"Unknown action: {params.action}"

    def _analyze_assets(self, tickers: list[str], period: str) -> str:
        if not ok:
            return "Error: 'okama' library is not installed."
        if not tickers:
            return "Please provide a list of tickers to analyze."

        try:
            # Okama analysis
            al = ok.AssetList(tickers, ccy="RUB") # Defaulting to RUB for now
            
            # Basic textual summary
            desc = al.names
            risk = getattr(al, "risk_annual", None)
            
            # Try different return metrics
            cagr = getattr(al, "mean_return", None) # Some versions use simple mean_return
            if cagr is None:
                cagr = getattr(al, "cagr", {}) # Fallback to CAGR
            
            summary = [f"**Analysis for {', '.join(tickers)} ({period})**"]
            
            for t in tickers:
                name = desc.get(t, t)
                
                # Safe access and casting
                try:
                    r_val = float(risk[t]) if risk is not None and t in risk else 0.0
                except: r_val = 0.0
                
                try:
                    c_val = float(cagr[t]) if cagr is not None and t in cagr else 0.0
                except: c_val = 0.0
                
                summary.append(f"- **{t}** ({name}): Risk {r_val*100:.2f}%, Return {c_val*100:.2f}%")
                
            return "\n".join(summary)
            
        except Exception as e:
            return f"Okama Analysis Failed: {str(e)}"

    def _search_rag(self, query: str) -> str:
        if not hasattr(self, 'rag') or not self.rag:
            return "Error: RAG Service is not available in this skill."
            
        results = self.rag.search(collection_name="finance_docs", query=query, limit=3)
        
        if not results:
             return "No relevant information found in the knowledge base."
             
        # Format output
        docs = []
        for hit in results:
            content = hit['content'][:300] + "..."
            docs.append(f"> {content}\n*(Score: {hit['score']:.2f})*")
            
        return f"**Found {len(results)} relevant documents:**\n\n" + "\n\n".join(docs)
