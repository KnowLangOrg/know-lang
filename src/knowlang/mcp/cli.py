"""
Command-line interface for Knowlang MCP server.

This module provides a command-line interface for launching the
Knowlang MCP server.
"""

import argparse
import asyncio
import logging
import sys

from knowlang.mcp.server import KnowlangMCPServer
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Launch Knowlang MCP server")
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="Host to listen on (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7773,
        help="Port to listen on (default: 7773)"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="knowlang-search",
        help="Name of the MCP server (default: knowlang-search)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level)
    
    LOG.info(f"Starting Knowlang MCP server on {args.host}:{args.port}")
    
    # Create and start the server
    server = KnowlangMCPServer(
        host=args.host,
        port=args.port,
        server_name=args.name
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        LOG.info("Server shutdown requested")
    except Exception as e:
        LOG.error(f"Server error: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()
