import asyncio
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class McpToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]

class McpClientWrapper:
    """
    Wrapper around the official MCP Python SDK to simplify integration 
    with SGR Core's Skill system.
    """
    def __init__(self, server_name: str, command: str, args: List[str] = None, env: Dict[str, str] = None):
        self.server_name = server_name
        self.command = command
        self.args = args or []
        self.env = env or os.environ.copy()
        self.session: Optional[ClientSession] = None
        self._exit_stack = None

    async def connect(self):
        """Establishes connection to the MCP Server via Stdio."""
        if not MCP_AVAILABLE:
            logger.error("❌ 'mcp' library not installed. Cannot connect to MCP Server.")
            return

        logger.info(f"🔌 Connecting to MCP Server '{self.server_name}'...")
        
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env
        )

        try:
            # We use the stdio_client context manager
            # In a real async app, we need to keep this context alive.
            # This is a simplified implementation; robust handling requires AsyncExitStack
            from contextlib import AsyncExitStack
            self._exit_stack = AsyncExitStack()
            
            read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
            self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            
            await self.session.initialize()
            logger.info(f"✅ Connected to MCP Server '{self.server_name}'")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP Server: {e}")
            if self._exit_stack:
                await self._exit_stack.aclose()
            raise

    async def list_tools(self) -> List[McpToolDefinition]:
        """Fetches available tools from the MCP Server."""
        if not self.session:
            logger.warning("MCP Session not active.")
            return []

        try:
            result = await self.session.list_tools()
            tools = []
            for t in result.tools:
                tools.append(McpToolDefinition(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema
                ))
            return tools
        except Exception as e:
            logger.error(f"Error listing MCP tools: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Executes a tool on the MCP Server."""
        if not self.session:
            raise RuntimeError("MCP Session not active")

        try:
            result = await self.session.call_tool(tool_name, arguments)
            # Combine text content from result
            output = []
            if hasattr(result, 'content'):
                for content in result.content:
                    if content.type == 'text':
                        output.append(content.text)
                    elif content.type == 'image':
                        output.append(f"[Image: {content.mimeType}]") # Placeholder
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            return f"Error: {str(e)}"

    async def disconnect(self):
        """Closes the connection."""
        if self._exit_stack:
            await self._exit_stack.aclose()
        logger.info(f"🔌 Disconnected from MCP Server '{self.server_name}'")
