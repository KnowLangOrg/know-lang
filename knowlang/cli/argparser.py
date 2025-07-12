"""Argument parsing for KnowLang CLI."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Union

from knowlang.cli.commands.chat import chat_command
from knowlang.cli.commands.mcp.serve import mcp_serve_command
from knowlang.cli.commands.parse import parse_command
from knowlang.cli.types import (
    BaseCommandArgs,
    ChatCommandArgs,
    MCPServeCommandArgs,
    ParseCommandArgs,
)


def _convert_to_args(
    parsed_namespace: argparse.Namespace,
) -> Union[
    ParseCommandArgs,
    ChatCommandArgs,
    MCPServeCommandArgs,
]:
    """Convert parsed namespace to typed arguments."""
    base_args = {
        "verbose": parsed_namespace.verbose,
        "config": (
            parsed_namespace.config if hasattr(parsed_namespace, "config") else None
        ),
        "command": parsed_namespace.command,
    }

    args = None
    command_func = None

    if parsed_namespace.command == "parse":
        command_func = parse_command
        args = ParseCommandArgs(
            **base_args,
            output=parsed_namespace.output,
        )
    elif parsed_namespace.command == "chat":
        command_func = chat_command
        args = ChatCommandArgs(
            **base_args,
            port=parsed_namespace.port,
            share=parsed_namespace.share,
            server_port=parsed_namespace.server_port,
            server_name=parsed_namespace.server_name,
        )
    elif parsed_namespace.command == "mcp":
        if parsed_namespace.subcommand == "serve":
            command_func = mcp_serve_command
            args = MCPServeCommandArgs(
                **base_args,
                subcommand=parsed_namespace.subcommand,
                host=parsed_namespace.host,
                port=parsed_namespace.port,
                name=parsed_namespace.name,
            )
        else:
            raise ValueError(
                f"Unknown subcommand for mcp: {parsed_namespace.subcommand}"
            )
    else:
        raise ValueError(f"Unknown command: {parsed_namespace.command}")

    args.command_func = command_func
    return args


def parse_json_dict(arg: str) -> dict:
    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON provided: {e}")


def _create_parse_parser(subparsers):
    """Create the parser for the 'parse' command."""
    parse_parser = subparsers.add_parser("parse", help="Parse and index a codebase")
    parse_parser.add_argument(
        "--output",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parse_parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=".",
        help="Path to codebase directory or repository URL",
    )
    parse_parser.add_argument(
        "--extra-fields",
        type=parse_json_dict,
        default={},
        help='Additional dynamic fields as a JSON string. For example: \'{"key1": "value1", "key2": "value2"}\'',
    )
    return parse_parser


def _create_chat_parser(subparsers):
    """Create the parser for the 'chat' command."""
    chat_parser = subparsers.add_parser("chat", help="Launch the chat interface")
    chat_parser.add_argument("--port", type=int, help="Port to run the interface on")
    chat_parser.add_argument(
        "--share", action="store_true", help="Create a shareable link"
    )
    chat_parser.add_argument(
        "--server-port",
        type=int,
        help="Port to run the server on (if different from --port)",
    )
    chat_parser.add_argument(
        "--server-name", type=str, help="Server name to listen on (default: 0.0.0.0)"
    )
    return chat_parser


def _create_mcp_parser(subparsers):
    """Create the parser for the 'mcp' command and its subcommands."""
    mcp_parser = subparsers.add_parser("mcp", help="Model Context Protocol (MCP) tools")
    mcp_subparsers = mcp_parser.add_subparsers(
        title="subcommands",
        description="MCP subcommands",
        dest="subcommand",
        required=True,
    )

    # Create MCP serve subcommand
    serve_parser = mcp_subparsers.add_parser(
        "serve", help="Start the MCP server for Knowlang search capabilities"
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server to (default: localhost)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=7773,
        help="Port to run the server on (default: 7773)",
    )
    serve_parser.add_argument(
        "--name",
        type=str,
        default="knowlang-search",
        help="Name of the MCP server (default: knowlang-search)",
    )

    return mcp_parser


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="KnowLang - Code Understanding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--config", type=Path, help="Path to custom configuration file", default=None
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        title="commands", description="Available commands", dest="command"
    )
    subparsers.required = True

    # Create command parsers
    _create_parse_parser(subparsers)
    _create_chat_parser(subparsers)
    _create_mcp_parser(subparsers)

    return parser


def parse_args(
    args: Optional[Sequence[str]] = None,
) -> Union[
    ParseCommandArgs,
    BaseCommandArgs,
    MCPServeCommandArgs,
]:
    """Parse command line arguments into typed objects."""
    parser = create_parser()
    parsed_namespace = parser.parse_args(args)
    return _convert_to_args(parsed_namespace)
