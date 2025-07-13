from knowlang.cli.types import ChatCommandArgs
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


async def chat_command(args: ChatCommandArgs) -> None:
    """Execute the chat command.

    Args:
        args: Typed command line arguments
    """

    # Create and launch chatbot
    from knowlang.configs.chat_config import ChatConfig
    from knowlang.chat_bot.chat_interface import create_chatbot

    demo = create_chatbot(ChatConfig())

    launch_kwargs = {
        "server_port": args.server_port,
        "server_name": args.server_name or "127.0.0.1",
        "share": args.share,
    }
    if args.port:
        launch_kwargs["port"] = args.port

    demo.launch(**launch_kwargs)
