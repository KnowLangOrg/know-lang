import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings

from knowlang.configs.defaults import DEFAULT_VECTOR_COLLECTION_NAME
from knowlang.configs.llm_config import LLMConfig
from knowlang.core.types import ModelProvider, VectorStoreProvider

from .base import _validate_api_key
from .state_store_config import StateStoreConfig


class PathPatterns(BaseSettings):
    include: List[str] = Field(
        default=["**/*"], description="Glob patterns for paths to include"
    )
    exclude: List[str] = Field(
        default=[
            "**/venv/**",
            "**/.git/**",
            "**/__pycache__/**",
            "**/tests/**",
        ],
        description="Glob patterns for paths to exclude",
    )

    def should_process_path(self, path: str) -> bool:
        """Check if a path should be processed based on include/exclude patterns"""
        path_str = str(path)

        # First check exclusions
        for pattern in self.exclude:
            if fnmatch.fnmatch(path_str, pattern):
                return False

        # Then check inclusions
        for pattern in self.include:
            if fnmatch.fnmatch(path_str, pattern):
                return True

        return False


class LanguageConfig(BaseSettings):
    enabled: bool = True
    file_extensions: List[str]
    tree_sitter_language: str
    chunk_types: List[str] = Field(
        default=["class_definition", "function_definition"],
        description="Types of code chunks to extract",
    )
    max_file_size: int = Field(
        default=1_000_000, description="Maximum file size to process in bytes"  # 1MB
    )


class ParserConfig(BaseSettings):
    languages: Dict[str, LanguageConfig] = Field(
        default={
            "python": LanguageConfig(
                file_extensions=[".py"],
                tree_sitter_language="python",
                chunk_types=["class_definition", "function_definition"],
                max_file_size=1_000_000,
            ),
            "typescript": LanguageConfig(
                file_extensions=[".ts", ".tsx"],
                tree_sitter_language="typescript",
                chunk_types=["class_definition", "function_definition"],
                max_file_size=1_000_000,
            ),
            "cpp": LanguageConfig(
                file_extensions=[".cpp", ".h", ".hpp", ".cc"],
                tree_sitter_language="cpp",
                chunk_types=["class_definition", "function_definition"],
                max_file_size=1_000_000,
            ),
            "csharp": LanguageConfig(
                file_extensions=[".cs"],
                tree_sitter_language="csharp",
                chunk_types=[
                    "class_declaration",
                    "method_declaration",
                ],  # Using common tree-sitter type names
                max_file_size=1_000_000,
            ),
            "unity_asset": LanguageConfig(
                file_extensions=[".asset"],
                tree_sitter_language="unity_asset",
                # TODO: revisit these chunk types
                chunk_types=["function_definition"],
                max_file_size=1_000_000,
            ),
        }
    )
    path_patterns: PathPatterns = Field(default_factory=PathPatterns)
    enable_code_summarization: bool = Field(
        default=False,
        description="Enable code summarization to be stored in the vector store",
    )


class EmbeddingConfig(BaseSettings):
    """Shared embedding configuration"""

    model_name: str = Field(
        default="nomic-ai/CodeRankEmbed", description="Name of the embedding model"
    )
    model_provider: ModelProvider = Field(
        default=ModelProvider.NOMIC_AI, description="Provider for embeddings"
    )
    dimension: int = Field(default=768, description="Embedding dimension")
    settings: Dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific settings"
    )
    api_key: Optional[str] = Field(
        default=None, description="API key for the model provider"
    )

    @field_validator("api_key", mode="after")
    @classmethod
    def validate_api_key(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_api_key(v, info)


class EvaluatorConfig(LLMConfig):
    evaluation_rounds: int = Field(
        default=1, description="Number of evaluation rounds per test case"
    )
