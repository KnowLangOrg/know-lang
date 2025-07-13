from knowlang.configs.config import AppConfig
from knowlang.mcp.tools.vector_search import VectorSearchTool
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    raise ImportError(
        "MCP server is not installed. Please install it using `pip install 'knowlang[mcp]'`."
    ) from e


class KnowlangMCPServer:
    """MCP Server for Knowlang.

    This server exposes Knowlang's search capabilities through the Model
    Context Protocol (MCP).
    """

    def __init__(
        self,
        config: AppConfig,
        host: str = "localhost",
        port: int = 7773,
        server_name: str = "knowlang-search",
    ):
        """Initialize the MCP server.

        Args:
            host: The host to listen on
            port: The port to listen on
            server_name: The name of the server
        """
        self.host = host
        self.port = port
        self.server_name = server_name
        self.mcp_server = FastMCP(server_name, host=host, port=port)
        self.config = config

        # Register tools
        self._register_tools()

        LOG.info(f"Initialized Knowlang MCP server on {host}:{port}")

    def _register_tools(self):
        """Register all MCP tools with the server."""

        # Vector search tool
        VectorSearchTool.initialize(self.config)
        self.mcp_server.add_tool(
            VectorSearchTool.run,
            name=VectorSearchTool.name,
            description=VectorSearchTool.description,
        )

        LOG.info("Registered all MCP tools")

    async def start(self):
        """Start the MCP server."""
        try:
            await self.mcp_server.run_stdio_async()

        except Exception as e:
            LOG.error(f"Error starting MCP server: {e}")
            raise
        finally:
            LOG.info("Stopping MCP server")
