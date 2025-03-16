import uvicorn
from knowlang.cli.types import ServeCommandArgs
from knowlang.cli.utils import create_config
from knowlang.utils import FancyLogger
from knowlang.vector_stores.factory import VectorStoreFactory
from knowlang.vector_stores import VectorStoreError

LOG = FancyLogger(__name__)


async def serve_command(args: ServeCommandArgs) -> None:
    """Execute the serve command.
    
    Args:
        args: Typed command line arguments
    """
    config = create_config(args.config)
    
    # Initialize vector store
    try:
        VectorStoreFactory.get(config.db, config.embedding)
    except VectorStoreError as e:
        LOG.error(
            "Vector store initialization failed. Please run 'knowlang parse' first to index your codebase."
            f"\nError: {str(e)}"
        )
        return

    # Configure uvicorn server using Server class directly
    config = uvicorn.Config(
        "knowlang.chat_bot.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )
    
    server = uvicorn.Server(config)
    # Use await instead of run() to respect the existing event loop
    await server.serve()