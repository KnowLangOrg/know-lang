from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

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
    is_active: bool = Field(
        default=True, description="Whether this asset manager is active"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    last_indexed_at: Optional[datetime] = Field(
        default=None, description="Last time this was indexed"
    )

    def get_display_name(self) -> str:
        """Get a human-readable display name for this asset manager."""
        # Implementation will be filled later
        pass
    pass


class CodeAssetMetaData(BaseModel):
    """Metadata for a single code file in the codebase."""

    file_path: str = Field(..., description="Relative path from asset manager root")
    absolute_path: str = Field(..., description="Absolute file path")
    language: LanguageEnum = Field(..., description="Programming language")
    file_size_bytes: int = Field(..., description="File size in bytes")
    last_modified: datetime = Field(..., description="File last modification time")
    file_hash: str = Field(..., description="File content hash for change detection")


class CodeAssetChunkMetaData(BaseModel):
    """Metadata for a code chunk within a code file."""

    chunk_type: str = Field(
        ..., description="Type of code chunk (function, class, etc.)"
    )
    language: LanguageEnum = Field(..., description="Programming language")

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