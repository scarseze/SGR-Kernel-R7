import os
import sys
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import CoreEngine
from core.logger import configure_logger, get_logger

# Import Skills
from skills.sglang_sim.handler import SGLangSkill
from skills.portfolio.handler import PortfolioSkill
from skills.gost_writer.handler import GostWriterSkill
from skills.calendar.handler import CalendarSkill
from skills.rlm.handler import RLMSkill
from skills.web_search.handler import WebSearchSkill
from skills.office_suite.handler import OfficeSkill
from skills.data_analyst.handler import DataAnalystSkill
from skills.research_agent.handler import ResearchSubAgent
from skills.filesystem.handler import ReadFileSkill, ListDirSkill
from skills.logic_rl.handler import LogicRLSkill

load_dotenv()
configure_logger()
logger = get_logger("api")

app = FastAPI(title="SGR Core Agent API", description="Universal Personal Agent Interface")

class AgentRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    source_app: Optional[str] = "unknown"

class AgentResponse(BaseModel):
    result: str

engine: Optional[CoreEngine] = None

@app.on_event("startup")
async def startup_event():
    global engine
    logger.info("Starting SGR Core Server...")
    
    # Config Logic (Mirrors main.py)
    llm_config = {}
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        llm_config = {
            "base_url": "https://api.deepseek.com",
            "api_key": deepseek_key,
            "model": "deepseek-chat"
        }
    
    # Auto-approve for API mode (or implement webhook callback later)
    async def api_approval(msg: str) -> bool:
        logger.info(f"⚡ Auto-approving action via API: {msg}")
        return True

    engine = CoreEngine(llm_config=llm_config, approval_callback=api_approval)
    
    # Register Skills
    engine.register_skill(SGLangSkill())
    engine.register_skill(PortfolioSkill())
    engine.register_skill(GostWriterSkill())
    engine.register_skill(CalendarSkill())
    engine.register_skill(RLMSkill())
    engine.register_skill(WebSearchSkill())
    engine.register_skill(OfficeSkill())
    engine.register_skill(DataAnalystSkill())
    engine.register_skill(ResearchSubAgent(llm_config))
    engine.register_skill(ReadFileSkill())
    engine.register_skill(ListDirSkill())
    engine.register_skill(LogicRLSkill())
    
    logger.info("Agent Engine Ready & Listening.")

@app.post("/v1/agent/process", response_model=AgentResponse)
async def process_request(req: AgentRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Inject Context into Prompt
    full_prompt = req.query
    full_prompt = req.query
    if req.context:
        # Smart Context Formatting
        ctx = req.context
        context_str = ""
        
        if "file_path" in ctx:
            context_str += f"CURRENT FILE: {ctx['file_path']}\n"
        
        if "selection" in ctx and ctx['selection']:
            sel = ctx['selection']
            context_str += f"\nSELECTED CODE ({sel.get('start_line')}-{sel.get('end_line')}):\n```\n{sel.get('text', '')}\n```\n"
        elif "content" in ctx:
            # If no selection, show partial content around cursor? 
            # For now, let's truncate if too huge, or rely on LLM window
            content = ctx.get('content', '')
            if len(content) > 10000:
                 content = content[:10000] + "...(truncated)"
            context_str += f"\nFILE CONTENT:\n```\n{content}\n```\n"

        # Fallback for generic context
        other_keys = {k:v for k,v in ctx.items() if k not in ['file_path', 'content', 'selection', 'cursor_line']}
        if other_keys:
             context_str += f"\nMETADATA: {other_keys}\n"

        full_prompt = f"CONTEXT FROM {req.source_app.upper()}:\n{context_str}\n\nUSER REQUEST: {req.query}"
    
    try:
        logger.info(f"Processing request from {req.source_app}")
        # The engine.run() method handles skill selection and execution
        result = await engine.run(full_prompt)
        return AgentResponse(result=result)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
