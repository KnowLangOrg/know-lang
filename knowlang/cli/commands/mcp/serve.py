"""MCP serve command for Knowlang CLI."""
from knowlang.cli.types import MCPServeCommandArgs
from knowlang.cli.utils import create_config
from knowlang.mcp.server import KnowlangMCPServer
from knowlang.utils import FancyLogger
from knowlang.vector_stores.factory import VectorStoreFactory
from knowlang.vector_stores import VectorStoreError

LOG = FancyLogger(__name__)

async def mcp_serve_command(args: MCPServeCommandArgs) -> None:
    """Execute the MCP serve command.
    
    Args:
        args: Typed command line arguments
    """
    # Create configuration
    config = create_config(args.config)
    
    # Initialize vector store if needed
    try:
        VectorStoreFactory.get(config)
    except VectorStoreError as e:
        LOG.error(
            "Vector store initialization failed. Please run 'knowlang parse' first to index your codebase."
            f"\nError: {str(e)}"
        )
        return
        
    # Set up logging level
    log_level = "DEBUG" if args.verbose else "INFO"
    LOG.info(f"Starting Knowlang MCP server on {args.host}:{args.port}")
    
    # Create and start the server
    server = KnowlangMCPServer(
        config=config,
        host=args.host,
        port=args.port,
        server_name=args.name
    )
    
    try:
        # Start the server
        await server.start()
    except Exception as e:
        LOG.error(f"Server error: {e}")
        raise