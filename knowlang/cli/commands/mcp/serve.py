"""MCP serve command for Knowlang CLI."""

from knowlang.assets.registry import DomainRegistry, RegistryConfig
from knowlang.cli.types import MCPServeCommandArgs
from knowlang.mcp.server import KnowlangMCPServer
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


async def mcp_serve_command(args: MCPServeCommandArgs) -> None:
    """Execute the MCP serve command.

    Args:
        args: Typed command line arguments
    """
    # Create configuration
    config = RegistryConfig()
    registry = DomainRegistry(config)
    await registry.discover_and_register()

    LOG.info(f"Starting Knowlang MCP server on {args.host}:{args.port}")

    # Create and start the server
    server = KnowlangMCPServer(
        registry=registry, host=args.host, port=args.port, server_name=args.name
    )

    try:
        # Start the server
        await server.start()
    except Exception as e:
        LOG.error(f"Server error: {e}")
        raise
