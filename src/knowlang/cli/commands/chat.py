from knowlang.chat_bot import create_chatbot
from knowlang.cli.types import ChatCommandArgs
from knowlang.cli.utils import create_config
from knowlang.utils import FancyLogger
from knowlang.vector_stores import VectorStoreError
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = FancyLogger(__name__)


async def chat_command(args: ChatCommandArgs) -> None:
    """Execute the chat command.
    
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
    
    # Create and launch chatbot
    demo = create_chatbot(config)
    
    launch_kwargs = {
        "server_port": args.server_port,
        "server_name": args.server_name or "127.0.0.1",
        "share": args.share,
    }
    if args.port:
        launch_kwargs["port"] = args.port
        
    demo.launch(**launch_kwargs)