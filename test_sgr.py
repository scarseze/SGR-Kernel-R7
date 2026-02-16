import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add path
sys.path.append(os.path.dirname(__file__))

from core.engine import CoreEngine
from core.llm import LLMService
from skills.sglang_sim.handler import SGLangSkill
from skills.portfolio.handler import PortfolioSkill
from skills.xbrl_analyst.handler import XBRLAnalystSkill
from skills.gost_writer.handler import GostWriterSkill
from skills.calendar.handler import CalendarSkill
from skills.rlm.handler import RLMSkill
from skills.web_search.handler import WebSearchSkill

import asyncio

async def test():
    print("=== Testing SGR Core Agent (Async) ===")
    
    # Check for keys (simplified)
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    # Manually configure LLM for testing
    
    # Configure using direct args
    config = {
        "base_url": "https://api.deepseek.com",
        "api_key": api_key,
        "model": "deepseek-chat"
    }
    
    skill_llm = LLMService(**config) 
    
    try:
        # Engine init
        engine = CoreEngine(llm_config=config)
        
        # Register skills
        engine.register_skill(SGLangSkill(llm_service=skill_llm))
        engine.register_skill(PortfolioSkill())
        engine.register_skill(XBRLAnalystSkill(llm_service=skill_llm))
        engine.register_skill(GostWriterSkill(llm_service=skill_llm))
        engine.register_skill(CalendarSkill())
        engine.register_skill(RLMSkill())
        engine.register_skill(WebSearchSkill())
        
        # Test Query for Web Search
        query = "What is the current price of Bitcoin in USD?"
        print(f"\nUser Query: {query}\n")
        
        response = await engine.run(query)
        print("--- AGENT RESPONSE ---")
        print(response)
        print("----------------------")
        
    except Exception as e:
        print(f"❌ Runtime Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
