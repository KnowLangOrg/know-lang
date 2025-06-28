"""Command implementation for parsing codebases."""

from knowlang.cli.types import ParseCommandArgs
from knowlang.assets.registry import DomainRegistry, RegistryConfig
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


async def parse_command(args: ParseCommandArgs) -> None:
    """Execute the parse command.
    
    Args:
        args: Typed command line arguments
    """
    # TODO: which args should we support for parse command?
    config = RegistryConfig()
    registry = DomainRegistry(config)
    await registry.discover_and_register()
    await registry.process_all_domains()