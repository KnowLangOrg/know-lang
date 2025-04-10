"""Argument parsing for KnowLang CLI."""
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Type, Union

from knowlang.cli.commands.chat import chat_command
from knowlang.cli.commands.evaluations.prepare_dataset import \
    prepare_dataset_command
from knowlang.cli.commands.evaluations.run_evaluation import \
    run_evaluation_command
from knowlang.cli.commands.parse import parse_command
from knowlang.cli.commands.serve import serve_command
from knowlang.cli.types import (BaseCommandArgs, ChatCommandArgs,
                                ParseCommandArgs, PrepareDatasetCommandArgs,
                                RunEvaluationCommandArgs, ServeCommandArgs)


def _convert_to_args(parsed_namespace: argparse.Namespace) -> Union[ParseCommandArgs, ChatCommandArgs, ServeCommandArgs]:
    """Convert parsed namespace to typed arguments."""
    base_args = {
        "verbose": parsed_namespace.verbose,
        "config": parsed_namespace.config if hasattr(parsed_namespace, "config") else None,
        "command": parsed_namespace.command
    }
    
    args = None
    command_func = None
    
    if parsed_namespace.command == "parse":
        command_func = parse_command
        args = ParseCommandArgs(
            **base_args,
            path=parsed_namespace.path,
            output=parsed_namespace.output,
            extra_fields=parsed_namespace.extra_fields
        )
    elif parsed_namespace.command == "chat":
        command_func = chat_command
        args = ChatCommandArgs(
            **base_args,
            port=parsed_namespace.port,
            share=parsed_namespace.share,
            server_port=parsed_namespace.server_port,
            server_name=parsed_namespace.server_name
        )
    elif parsed_namespace.command == "serve":
        command_func = serve_command
        args = ServeCommandArgs(
            **base_args,
            host=parsed_namespace.host,
            port=parsed_namespace.port,
            reload=parsed_namespace.reload,
            workers=parsed_namespace.workers
        )
    elif parsed_namespace.command == "evaluate":
        if parsed_namespace.subcommand == "prepare":
            command_func = prepare_dataset_command
            args = PrepareDatasetCommandArgs(
                **base_args,
                subcommand=parsed_namespace.subcommand,
                data_dir=parsed_namespace.data_dir,
                output_dir=parsed_namespace.output_dir,
                dataset=parsed_namespace.dataset,
                languages=parsed_namespace.languages,
                splits=parsed_namespace.splits,
                skip_indexing=parsed_namespace.skip_indexing
            )
        elif parsed_namespace.subcommand == "run":
            command_func = run_evaluation_command
            args = RunEvaluationCommandArgs(
                **base_args,
                subcommand=parsed_namespace.subcommand,
                data_dir=parsed_namespace.data_dir,
                output_dir=parsed_namespace.output_dir,
                config_dir=parsed_namespace.config_dir,
                dataset=parsed_namespace.dataset,
                language=parsed_namespace.language,
                configuration=parsed_namespace.configuration,
                limit=parsed_namespace.limit,
                grid_search=parsed_namespace.grid_search,
                list_configurations=parsed_namespace.list_configurations,
                generate_reranking_data=parsed_namespace.generate_reranking_data
            )
        else:
            raise ValueError(f"Unknown subcommand for evaluate: {parsed_namespace.subcommand}")
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
        nargs="?",
        default=".",
        help="Path to codebase directory or repository URL"
    )
    parse_parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User ID to associate with the codebase",
    )
    parse_parser.add_argument(
        "--extra-fields",
        type=parse_json_dict,
        default={},
        help="Additional dynamic fields as a JSON string. For example: '{\"key1\": \"value1\", \"key2\": \"value2\"}'"
    )
    return parse_parser

def _create_chat_parser(subparsers):
    """Create the parser for the 'chat' command."""
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
    return chat_parser

def _create_serve_parser(subparsers):
    """Create the parser for the 'serve' command."""
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
    return serve_parser

def _create_prepare_dataset_parser(evaluate_subparsers):
    """Create the parser for the 'evaluate prepare' command."""
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
        default=['python'],
        type=str,
        nargs="+",
        help="Languages to include (e.g., python java)"
    )
    prepare_parser.add_argument(
        "--splits",
        default=['test', 'train', 'valid'],
        type=str,
        nargs="+",
        help="Dataset split to use (train, valid, test)"
    )
    prepare_parser.add_argument(
        "--skip-indexing",
        type=bool,
        default=PrepareDatasetCommandArgs.skip_indexing,
        help="Skip indexing, only generate query mappings"
    )
    return prepare_parser

def _create_run_evaluation_parser(evaluate_subparsers):
    """Create the parser for the 'evaluate run' command."""
    run_parser = evaluate_subparsers.add_parser(
        "run",
        help="Run code search evaluations"
    )
    run_parser.add_argument(
        "--data-dir",
        type=Path,
        default=RunEvaluationCommandArgs.data_dir,
        help="Directory containing dataset query mappings"
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RunEvaluationCommandArgs.output_dir,
        help="Output directory for results"
    )
    run_parser.add_argument(
        "--config-dir",
        type=Path,
        default=RunEvaluationCommandArgs.config_dir,
        help="Directory for search configurations"
    )
    run_parser.add_argument(
        "--dataset",
        type=str,
        default=RunEvaluationCommandArgs.dataset,
        help="Dataset to evaluate"
    )
    run_parser.add_argument(
        "--language",
        type=str,
        default=RunEvaluationCommandArgs.language,
        help="Language to evaluate"
    )
    run_parser.add_argument(
        "--configuration",
        type=str,
        default=RunEvaluationCommandArgs.configuration,
        help="Search configuration to use"
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=RunEvaluationCommandArgs.limit,
        help="Limit number of queries to evaluate"
    )
    run_parser.add_argument(
        "--grid-search",
        action="store_true",
        default=RunEvaluationCommandArgs.grid_search,
        help="Run grid search over configurations"
    )
    run_parser.add_argument(
        "--list-configurations",
        action="store_true",
        default=RunEvaluationCommandArgs.list_configurations,
        help="List available search configurations"
    )
    run_parser.add_argument(
        "--generate-reranking-data",
        action="store_true",
        help="Generate data for reranking evaluation"
    )
    return run_parser

def _create_evaluate_parser(subparsers):
    """Create the parser for the 'evaluate' command and its subcommands."""
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
    
    # Create subcommand parsers
    _create_prepare_dataset_parser(evaluate_subparsers)
    _create_run_evaluation_parser(evaluate_subparsers)
    
    return evaluate_parser

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
    
    # Create command parsers
    _create_parse_parser(subparsers)
    _create_chat_parser(subparsers)
    _create_serve_parser(subparsers)
    _create_evaluate_parser(subparsers)
    
    return parser

def parse_args(args: Optional[Sequence[str]] = None) -> Union[
    ParseCommandArgs, BaseCommandArgs, ServeCommandArgs, PrepareDatasetCommandArgs
]:
    """Parse command line arguments into typed objects."""
    parser = create_parser()
    parsed_namespace = parser.parse_args(args)
    return _convert_to_args(parsed_namespace)