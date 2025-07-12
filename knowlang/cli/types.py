"""Type definitions for CLI arguments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass(kw_only=True)
class BaseCommandArgs:
    """Base arguments for all commands."""

    verbose: bool = False
    config: Optional[Path] = None


@dataclass
class ParseCommandArgs(BaseCommandArgs):
    """Arguments for the parse command."""

    output: Literal["table", "json"] = "table"
    command: Literal["parse"] = "parse"


@dataclass
class ChatCommandArgs(BaseCommandArgs):
    """Arguments for the chat command."""

    command: Literal["chat"]
    port: Optional[int] = None
    share: bool = False
    server_port: Optional[int] = None
    server_name: Optional[str] = None


@dataclass
class MCPServeCommandArgs(BaseCommandArgs):
    """Arguments for the MCP serve command."""

    command: Literal["mcp"]
    subcommand: Literal["serve"] = "serve"
    host: str = "localhost"
    port: int = 7773
    name: str = "knowlang-search"
