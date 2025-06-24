from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

from knowlang.configs.config import LanguageConfig
from knowlang.core.types import LanguageEnum
from knowlang.assets.models import (
    DomainManagerData,
    GenericAssetData,
    GenericAssetChunkData,
)


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


class CodebaseManagerData(DomainManagerData[CodebaseMetaData]):
    pass

class CodeAssetData(GenericAssetData[CodeAssetMetaData]):
    pass

class CodeAssetChunkData(GenericAssetChunkData[CodeAssetChunkMetaData]):
    pass

class CodeProcessorConfig(BaseModel):
    """Configuration for the codebase processor."""
    directory_path: str = Field(
        default="./",
        description="Path to the codebase directory to index"
    )
    languages: List[LanguageConfig] = Field(
        default=[
            LanguageConfig(
                file_extensions=[".py"],
                tree_sitter_language="python",
                chunk_types=["class_definition", "function_definition"],
                max_file_size=1_000_000,
            ),
            LanguageConfig(
                file_extensions=[".ts", ".tsx"],
                tree_sitter_language="typescript",
                chunk_types=["class_definition", "function_definition"],
                max_file_size=1_000_000,
            ),
            LanguageConfig(
                file_extensions=[".cpp", ".h", ".hpp", ".cc"],
                tree_sitter_language="cpp",
                chunk_types=["class_definition", "function_definition"],
                max_file_size=1_000_000,
            ),
            LanguageConfig(
                file_extensions=[".cs"],
                tree_sitter_language="csharp",
                chunk_types=[
                    "class_declaration",
                    "method_declaration",
                ],  # Using common tree-sitter type names
                max_file_size=1_000_000,
            ),
        ]
    )