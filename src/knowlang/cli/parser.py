"""Argument parsing for KnowLang CLI."""
import argparse
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Type, Union

from knowlang.cli.commands.chat import chat_command
from knowlang.cli.commands.evaluations.prepare_dataset import prepare_dataset_command
from knowlang.cli.commands.parse import parse_command
from knowlang.cli.commands.serve import serve_command
from knowlang.cli.types import (
    BaseCommandArgs, ChatCommandArgs,
    ParseCommandArgs, ServeCommandArgs,
    PrepareDatasetCommandArgs
)

# Define command configurations
COMMAND_CONFIGS: Dict[str, tuple[Type[BaseCommandArgs], Callable]] = {
    "parse": (ParseCommandArgs, parse_command),
    "chat": (ChatCommandArgs, chat_command),
    "serve": (ServeCommandArgs, serve_command),
    "evaluate": (PrepareDatasetCommandArgs, prepare_dataset_command),
}

def _convert_to_args(parsed_namespace: argparse.Namespace) -> Union[ParseCommandArgs, ChatCommandArgs, ServeCommandArgs]:
    """Convert parsed namespace to typed arguments."""
    base_args = {
        "verbose": parsed_namespace.verbose,
        "config": parsed_namespace.config if hasattr(parsed_namespace, "config") else None,
        "command": parsed_namespace.command
    }
    
    # Get the appropriate argument class and command function
    args_class, command_func = COMMAND_CONFIGS[parsed_namespace.command]
    
    if parsed_namespace.command == "parse":
        args = args_class(
            **base_args,
            path=parsed_namespace.path,
            output=parsed_namespace.output
        )
    elif parsed_namespace.command == "chat":
        args = args_class(
            **base_args,
            port=parsed_namespace.port,
            share=parsed_namespace.share,
            server_port=parsed_namespace.server_port,
            server_name=parsed_namespace.server_name
        )
    elif parsed_namespace.command == "serve":
        args = args_class(
            **base_args,
            host=parsed_namespace.host,
            port=parsed_namespace.port,
            reload=parsed_namespace.reload,
            workers=parsed_namespace.workers
        )
    elif parsed_namespace.command == "evaluate":
        if parsed_namespace.subcommand == "prepare":
            args = args_class(
                **base_args,
                subcommand=parsed_namespace.subcommand,
                data_dir=parsed_namespace.data_dir,
                output_dir=parsed_namespace.output_dir,
                dataset=parsed_namespace.dataset,
                languages=parsed_namespace.languages,
                split=parsed_namespace.split,
                skip_indexing=parsed_namespace.skip_indexing
            )
        else:
            raise ValueError(f"Unknown subcommand for evaluate: {parsed_namespace.subcommand}")
    else:
        raise ValueError(f"Unknown command: {parsed_namespace.command}")
        
    args.command_func = command_func
    return args

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="KnowLang - Code Understanding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to custom configuration file",
        default=None
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command"
    )
    subparsers.required = True
    
    # Parse command
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse and index a codebase"
    )
    parse_parser.add_argument(
        "--output",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)"
    )
    parse_parser.add_argument(
        "path",
        type=str,
        nargs="?", # Make path optional
        default=".", # Default to current directory
        help="Path to codebase directory or repository URL"
    )

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Launch the chat interface"
    )
    chat_parser.add_argument(
        "--port",
        type=int,
        help="Port to run the interface on"
    )
    chat_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shareable link"
    )
    chat_parser.add_argument(
        "--server-port",
        type=int,
        help="Port to run the server on (if different from --port)"
    )
    chat_parser.add_argument(
        "--server-name",
        type=str,
        help="Server name to listen on (default: 0.0.0.0)"
    )
    
    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the API server"
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes"
    )
    serve_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluation tools for code search"
    )
    evaluate_subparsers = evaluate_parser.add_subparsers(
        title="subcommands",
        description="Evaluation subcommands",
        dest="subcommand",
        required=True
    )
    
    # Prepare dataset subcommand
    prepare_parser = evaluate_subparsers.add_parser(
        "prepare",
        help="Prepare benchmark datasets for evaluation"
    )
    prepare_parser.add_argument(
        "--data-dir",
        type=Path,
        default=PrepareDatasetCommandArgs.data_dir,
        help="Directory containing benchmark datasets"
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PrepareDatasetCommandArgs.output_dir,
        help="Output directory for query mappings"
    )
    prepare_parser.add_argument(
        "--dataset",
        type=str,
        choices=["codesearchnet", "cosqa", "all"],
        default=PrepareDatasetCommandArgs.dataset,
        help="Dataset to prepare"
    )
    prepare_parser.add_argument(
        "--languages",
        default='python',
        type=str,
        nargs="+",
        help="Languages to include (e.g., python java)"
    )
    prepare_parser.add_argument(
        "--split",
        type=str,
        default=PrepareDatasetCommandArgs.split,
        help="Dataset split to use (train, valid, test)"
    )
    prepare_parser.add_argument(
        "--skip-indexing",
        type=bool,
        default=PrepareDatasetCommandArgs.skip_indexing,
        help="Skip indexing, only generate query mappings"
    )
    
    return parser

def parse_args(args: Optional[Sequence[str]] = None) -> Union[
    ParseCommandArgs, BaseCommandArgs, ServeCommandArgs, PrepareDatasetCommandArgs
]:
    """Parse command line arguments into typed objects."""
    parser = create_parser()
    parsed_namespace = parser.parse_args(args)
    return _convert_to_args(parsed_namespace)