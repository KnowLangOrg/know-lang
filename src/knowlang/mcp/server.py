import asyncio
from mcp.server.fastmcp import FastMCP

from knowlang.mcp.tools.keyword_search import KeywordSearchTool
from knowlang.mcp.tools.vector_search import VectorSearchTool
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class KnowlangMCPServer:
    """MCP Server for Knowlang.
    
    This server exposes Knowlang's search capabilities through the Model
    Context Protocol (MCP).
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 7773,
                 server_name: str = "knowlang-search"):
        """Initialize the MCP server.
        
        Args:
            host: The host to listen on
            port: The port to listen on
            server_name: The name of the server
        """
        self.host = host
        self.port = port
        self.server_name = server_name
        self.mcp_server = FastMCP(server_name)
        
        # Register tools
        self._register_tools()
        
        LOG.info(f"Initialized Knowlang MCP server on {host}:{port}")
    
    def _register_tools(self):
        """Register all MCP tools with the server."""
        # Keyword search tool
        self.mcp_server.add_tool(KeywordSearchTool())
        
        # Vector search tool
        self.mcp_server.add_tool(VectorSearchTool())
        
        LOG.info("Registered all MCP tools")
    
    async def start(self):
        """Start the MCP server."""
        try:
            await self.mcp_server.run()
            LOG.info(f"MCP server started on {self.host}:{self.port}")
            
            # Keep server running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            LOG.error(f"Error starting MCP server: {e}")
            raise
        finally:
            LOG.info("Stopping MCP server")