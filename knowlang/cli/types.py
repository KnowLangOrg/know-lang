"""Type definitions for CLI arguments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

from knowlang.evaluations.types import DatasetType


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
class PrepareDatasetCommandArgs(BaseCommandArgs):
    """Arguments for the prepare-dataset command."""

    command: Literal["evaluate"]
    subcommand: Literal["prepare"] = "prepare"
    data_dir: Path = Path("datasets/code_search_net/data")
    output_dir: Path = Path("datasets/output")
    dataset: Literal["codesearchnet", "cosqa", "all"] = "all"
    languages: Optional[List[str]] = field(default_factory=lambda: ["python"])
    splits: Optional[str] = field(default_factory=lambda: ["test", "train", "valid"])
    skip_indexing: bool = False


@dataclass
class RunEvaluationCommandArgs(BaseCommandArgs):
    """Arguments for the run-evaluation command."""

    command: Literal["evaluate"]
    subcommand: Literal["run"] = "run"
    data_dir: Path = Path("datasets/output")
    output_dir: Path = Path("evaluation/results")
    config_dir: Path = Path("evaluation/settings")
    dataset: str = DatasetType.CODESEARCHNET
    language: str = "python"
    configuration: str = "baseline"
    limit: Optional[int] = None
    grid_search: bool = False
    generate_reranking_data: bool = False
    list_configurations: bool = False


@dataclass
class MCPServeCommandArgs(BaseCommandArgs):
    """Arguments for the MCP serve command."""

    command: Literal["mcp"]
    subcommand: Literal["serve"] = "serve"
    host: str = "localhost"
    port: int = 7773
    name: str = "knowlang-search"
