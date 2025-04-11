"""
MCP Server implementation for Knowlang.

This module provides the main server implementation for exposing
Knowlang's search capabilities through the Model Context Protocol.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

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
            await self.mcp_server.start(self.host, self.port)
            LOG.info(f"MCP server started on {self.host}:{self.port}")
            
            # Keep server running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            LOG.error(f"Error starting MCP server: {e}")
            raise
        finally:
            await self.mcp_server.stop()
    
    async def stop(self):
        """Stop the MCP server."""
        try:
            await self.mcp_server.stop()
            LOG.info("MCP server stopped")
        except Exception as e:
            LOG.error(f"Error stopping MCP server: {e}")
            raise

def run_server():
    """Run the MCP server."""
    server = KnowlangMCPServer()
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        LOG.info("Server shutdown requested")
    except Exception as e:
        LOG.error(f"Server error: {e}")
    
if __name__ == "__main__":
    # When run as a script, start the server
    run_server()
