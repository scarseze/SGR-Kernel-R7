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
    
    logger.info("Agent Engine Ready & Listening.")

@app.post("/v1/agent/process", response_model=AgentResponse)
async def process_request(req: AgentRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Inject Context into Prompt
    full_prompt = req.query
    if req.context:
        context_block = "\n".join([f"- {k}: {v}" for k, v in req.context.items()])
        full_prompt = f"CONTEXT FROM {req.source_app.upper()}:\n{context_block}\n\nUSER REQUEST: {req.query}"
    
    try:
        logger.info(f"Processing request from {req.source_app}")
        result = await engine.run(full_prompt)
        return AgentResponse(result=result)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))