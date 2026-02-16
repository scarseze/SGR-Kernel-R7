from skills.base import BaseSkill, SkillMetadata
from core.models import Tool
from core.interfaces.mcp_client import McpClientWrapper
import logging
import asyncio
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class McpSkill(BaseSkill):
    """
    Universally adapts any MCP Server into an SGR Core Skill.
    """
    @property
    def input_schema(self) -> Any:
        # MCP skills are dynamic, they don't have a single input schema
        # This is a special case.
        return None

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            capabilities=["mcp_tool"],
            risk_level="high", # Unknown external tool
            side_effects=True,
            idempotent=False,
            requires_network=True,
            requires_filesystem=False,
            cost_class="medium"
        )
    def __init__(self, server_name: str, command: str, args: List[str] = None):
        super().__init__()
        self.name = f"mcp_{server_name}"
        self.description = f"Dynamic tools from MCP Server: {server_name}"
        self.client = McpClientWrapper(server_name, command, args)
        self.tools_map: Dict[str, str] = {} # Map valid_name -> original_name

    async def initialize(self):
        """Connects to MCP and registers tools."""
        try:
            await self.client.connect()
            mcp_tools = await self.client.list_tools()
            
            logger.info(f"🛠️ MCP Server '{self.client.server_name}' exposed {len(mcp_tools)} tools.")

            for t in mcp_tools:
                # Sanitize name for LLM (DeepSeek prefers underscores)
                safe_name = t.name.replace("-", "_")
                self.tools_map[safe_name] = t.name

                tool_def = Tool(
                    name=safe_name,
                    description=t.description,
                    parameters=t.input_schema,
                    handler=self._make_handler(safe_name)
                )
                self.register_tool(tool_def)
                
        except Exception as e:
            logger.error(f"❌ Failed to init MCP Skill '{self.name}': {e}")

    def _make_handler(self, tool_name: str):
        """Creates a closure for tool execution."""
        async def handler(args: Dict[str, Any]) -> str:
            original_name = self.tools_map.get(tool_name, tool_name)
            logger.info(f"🚀 Executing MCP Tool: {original_name}")
            return await self.client.call_tool(original_name, args)
        return handler

    async def shutdown(self):
        """Cleanup connection."""
        await self.client.disconnect()
