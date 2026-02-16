import sys
import os
import json
import asyncio
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mocking missing modules for verification if needed
# But we should try to import real ones first

try:
    from skills.base import BaseSkill, SkillMetadata
    from skills.calendar.handler import CalendarSkill
    from skills.code_interpreter.handler import CodeInterpreterSkill
    from skills.filesystem.handler import ReadFileSkill, ListDirSkill
    from skills.web_search.handler import WebSearchSkill
    from skills.data_analyst.handler import DataAnalystSkill
    from skills.office_suite.handler import OfficeSkill
    from skills.portfolio.handler import PortfolioSkill
    from skills.research_agent.handler import ResearchSubAgent
    from skills.gost_writer.handler import GostWriterSkill
    from skills.mcp_adapter.handler import McpSkill
    from skills.rlm.handler import RLMSkill
    from skills.sglang_sim.handler import SGLangSkill
    from skills.xbrl_analyst.handler import XBRLAnalystSkill
except ImportError as e:
    print(f"Import Error: {e}")
    # Print sys.path to help debug
    print(f"Sys Path: {sys.path}")
    sys.exit(1)

def verify_skill(skill: BaseSkill):
    print(f"Testing {skill.name}...")
    try:
        meta = skill.metadata
        if not isinstance(meta, SkillMetadata):
            print(f"❌ {skill.name}: Metadata is not of type SkillMetadata")
            return
        
        # Print valid metadata
        print(f"✅ {skill.name}: {meta.model_dump_json(indent=2)}")
        
    except Exception as e:
        print(f"❌ {skill.name}: Failed to get metadata. Error: {e}")

def main():
    print("Verifying Skill Metadata Implementation (SA/sgr_core)...\n")
    
    skills = []
    
    # Instantiate skills (handling args)
    skills.append(CalendarSkill())
    skills.append(ReadFileSkill())
    skills.append(ListDirSkill())
    skills.append(WebSearchSkill())
    try:
        skills.append(DataAnalystSkill())
    except Exception as e:
        print(f"⚠️ DataAnalystSkill Init Failed: {e}")

    skills.append(OfficeSkill())
    skills.append(PortfolioSkill()) # qdrant optional
    skills.append(GostWriterSkill())
    skills.append(RLMSkill())
    skills.append(SGLangSkill())
    skills.append(XBRLAnalystSkill())
    
    # ResearchSubAgent requires llm_config
    skills.append(ResearchSubAgent(llm_config={"base_url": "http://dummy", "api_key": "dummy", "model": "dummy"}))
    
    # CodeInterpreter might try to connect to Docker
    try:
        skills.append(CodeInterpreterSkill())
    except Exception as e:
        print(f"⚠️ CodeInterpreterSkill Init Failed (Docker?): {e}")

    # McpSkill requires args
    skills.append(McpSkill(server_name="test", command="echo", args=[]))

    for s in skills:
        verify_skill(s)

if __name__ == "__main__":
    main()
