from typing import Type
from pydantic import BaseModel
from skills.base import BaseSkill, SkillMetadata
from core.state import AgentState
from skills.web_search.schema import WebSearchInput

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

class WebSearchSkill(BaseSkill):
    name: str = "web_search"
    description: str = (
        "Searches the internet for real-time information, news, market data, "
        "and other facts not present in the internal knowledge base."
    )
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            capabilities=["internet_search", "news"],
            risk_level="low",
            side_effects=False,
            idempotent=True,
            requires_network=True,
            requires_filesystem=False,
            cost_class="medium"
        )

    @property
    def input_schema(self) -> Type[BaseModel]:
        return WebSearchInput

    async def execute(self, params: WebSearchInput, state: AgentState) -> str:
        if DDGS is None:
            return "Error: 'duckduckgo-search' library is not installed."
        
        try:
            print(f"  🔍 Searching web for: '{params.query}'...")
            # Use 'html' backend for potentially better reliability on some IPs
            # or 'lite' if 'html' fails. API backend seems to be returning trending topics?
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    keywords=params.query,
                    region=params.region,
                    max_results=params.max_results,
                    backend="html" # Explicitly use scraping backend which is often more robust for queries
                ))
            
            if not results:
                return f"No results found for query: {params.query}"
            
            output = [f"### Web Search Results: {params.query}\n"]
            for i, res in enumerate(results, 1):
                title = res.get('title', 'No Title')
                href = res.get('href', '#')
                body = res.get('body', '')
                output.append(f"**{i}. [{title}]({href})**")
                output.append(f"> {body}\n")
                
            return "\n".join(output)
            
        except Exception as e:
            return f"Web Search Failed: {str(e)}"
