from pydantic import BaseModel, Field
from typing import Dict, Optional, TypeAlias

from knowlang.configs.config import LanguageConfig, PathPatterns
from knowlang.assets.models import (
    DomainManagerData,
    GenericAssetData,
    GenericAssetChunkData,
)
from knowlang.assets.config import ProcessorConfigBase
from knowlang.core.types import CodeChunk, LanguageEnum


class CodebaseMetaData(BaseModel):
    """Data model for a codebase asset manager (represents a single directory/repo)."""

    directory_path: str = Field(
        ..., description="Absolute path to the codebase directory"
    )
    git_url: Optional[str] = Field(
        default=None, description="Git repository URL if applicable"
    )
    git_branch: Optional[str] = Field(
        default=None, description="Git branch being indexed"
    )
    git_commit_hash: Optional[str] = Field(
        default=None, description="Current git commit hash"
    )

    def get_display_name(self) -> str:
        """Get a human-readable display name for this asset manager."""
        # Implementation will be filled later
        pass

    pass


class CodeAssetMetaData(BaseModel):
    """Metadata for a single code file in the codebase."""

    file_path: str = Field(..., description="Relative path from asset manager root")


class CodeAssetChunkMetaData(BaseModel):
    """Metadata for a code chunk within a code file."""

    content: str = Field(..., description="The actual code content of the chunk")

    # Location information
    file_path: str = Field(..., description="Relative path from asset manager root")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")

    # Code analysis
    chunk_name: str = Field(
        ..., description="Name of the code element (function name, class name, etc.)"
    )
    docstring: Optional[str] = Field(
        default=None, description="Associated documentation"
    )

    @classmethod
    def from_code_chunk(cls, chunk: "CodeChunk") -> "CodeAssetChunkMetaData":
        """Create metadata from a CodeChunk instance."""
        return cls(
            file_path=chunk.location.file_path,
            start_line=chunk.location.start_line,
            end_line=chunk.location.end_line,
            chunk_name=chunk.name,
            docstring=chunk.docstring,
            content=chunk.content,
        )


CodebaseManagerData: TypeAlias = DomainManagerData[CodebaseMetaData]
CodeAssetData: TypeAlias = GenericAssetData[CodeAssetMetaData]
CodeAssetChunkData: TypeAlias = GenericAssetChunkData[CodeAssetChunkMetaData]


class CodeProcessorConfig(ProcessorConfigBase):
    """Configuration for the codebase processor."""

    directory_path: str = Field(
        default="./", description="Path to the codebase directory to index"
    )
    languages: Dict[str, LanguageConfig] = Field(
        default={
            LanguageEnum.PYTHON.value: LanguageConfig(
                file_extensions=[".py"],
                tree_sitter_language="python",
                chunk_types=["class_definition", "function_definition"],
                max_file_size=1_000_000,
            ),
            LanguageEnum.TYPESCRIPT.value: LanguageConfig(
                file_extensions=[".ts", ".tsx"],
                tree_sitter_language="typescript",
                chunk_types=["class_definition", "function_definition"],
                max_file_size=1_000_000,
            ),
            LanguageEnum.CPP.value: LanguageConfig(
                file_extensions=[".cpp", ".h", ".hpp", ".cc"],
                tree_sitter_language="cpp",
                chunk_types=["class_definition", "function_definition"],
                max_file_size=1_000_000,
            ),
            LanguageEnum.CSHARP.value: LanguageConfig(
                file_extensions=[".cs"],
                tree_sitter_language="csharp",
                chunk_types=[
                    "class_declaration",
                    "method_declaration",
                ],  # Using common tree-sitter type names
                max_file_size=1_000_000,
            ),
            LanguageEnum.UNITYASSET.value: LanguageConfig(
                file_extensions=[".meta"],
                tree_sitter_language="unity_asset",
                chunk_types=["group", "connection", "node"],
                max_file_size=10_000_000,
            ),
        }
    )
    path_patterns: PathPatterns = Field(default_factory=PathPatterns)
