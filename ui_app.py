import os
import sys
import chainlit as cl

# Add core to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import CoreEngine
from core.logger import get_logger

logger = get_logger("ui")

def get_engine():
    """Retrieve or create the engine logic."""
    # In Chainlit, we usually store session-specific data in user_session
    # but here we can re-init per session for simplicity.
    
    # Check for Proxy URL just like main.py
    llm_config = {}
    proxy_url = os.getenv("PROXY_URL")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    if proxy_url:
        llm_config = {
            "base_url": proxy_url + "/v1",
            "api_key": "dummy-key",
            "model": "deepseek-chat"
        }
    elif deepseek_key:
        llm_config = {
            "base_url": "https://api.deepseek.com",
            "api_key": deepseek_key,
            "model": "deepseek-chat"
        }
    
    # Define Async Approval Callback
    async def ui_approval(msg: str) -> bool:
        """
        Shows a Chainlit Action (Button) to Approve/Reject.
        """
        res = await cl.AskActionMessage(
            content=f"⚠️ **APPROVAL REQUIRED** ⚠️\n\n{msg}",
            actions=[
                cl.Action(name="approve", payload={"value": "yes"}, label="✅ Approve", description="Allow this action"),
                cl.Action(name="reject", payload={"value": "no"}, label="❌ Reject", description="Block this action")
            ],
            timeout=600 # 10 mins decision time
        ).send()
        
        if not res:
            return False

        # Handle Dict (likely) vs Object
        action_name = ""
        if isinstance(res, dict):
            action_name = res.get("name", "")
        else:
            action_name = getattr(res, "name", "")
            
        if action_name == "approve":
            return True
            
        return False

    engine = CoreEngine(llm_config=llm_config, approval_callback=ui_approval)
    
    # Register Skills
    from skills.sglang_sim.handler import SGLangSkill
    from skills.portfolio.handler import PortfolioSkill
    from skills.gost_writer.handler import GostWriterSkill
    from skills.calendar.handler import CalendarSkill
    from skills.rlm.handler import RLMSkill
    from skills.web_search.handler import WebSearchSkill
    from skills.office_suite.handler import OfficeSkill
    from skills.data_analyst.handler import DataAnalystSkill
    
    engine.register_skill(SGLangSkill())
    engine.register_skill(PortfolioSkill())
    # engine.register_skill(XBRLAnalystSkill()) 
    engine.register_skill(GostWriterSkill())
    engine.register_skill(CalendarSkill())
    engine.register_skill(RLMSkill())
    engine.register_skill(WebSearchSkill())
    engine.register_skill(OfficeSkill())
    engine.register_skill(DataAnalystSkill())
    
    return engine

@cl.on_chat_start
async def start():
    engine = get_engine()
    cl.user_session.set("engine", engine)
    
    await cl.Message(
        content="**SGR Core Agent** готов к работе 🤖\n\nЯ помогу с финансами, анализом документов и поиском. Все действия защищены.",
        author="System"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    engine = cl.user_session.get("engine")
    
    # Show "Thinking..." loader
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Run Engine
        result = await engine.run(message.content)
        msg.content = result
        
        # Auto-detect generated files to attach them for download/viewing
        import re
        # Look for [Title](file:///path) pattern
        matches = re.findall(r"\(file:///(.+?)\)", result)
        elements = []
        
        for path in matches:
            # Clean path (remove possibly leading / if relative)
            # In handler we construct: file:///generated_files/file.ext
            # So path will be "generated_files/file.ext"
            
            if os.path.exists(path):
                name = os.path.basename(path)
                ext = name.split('.')[-1].lower()
                
                if ext in ['png', 'jpg', 'jpeg']:
                    elements.append(cl.Image(path=path, name=name, display="inline"))
                else:
                    elements.append(cl.File(path=path, name=name))
        
        if elements:
            msg.elements = elements
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg.content = f"**Error:** {str(e)}"
    
    await msg.update()
