import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings

from knowlang.configs.llm_config import LLMConfig
from knowlang.core.types import ModelProvider, VectorStoreProvider

from .base import _validate_api_key, generate_model_config
from .chat_config import ChatbotAnalyticsConfig, ChatConfig
from .retrieval_config import MultiStageRetrievalConfig
from .state_store_config import StateStoreConfig
from .server_config import ServerConfig


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
    chunk_types: List[str]
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
            "unity-asset": LanguageConfig(
                file_extensions=[".asset"],
                tree_sitter_language=None,
                # TODO: revisit these chunk types
                chunk_typese=[],
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


class DBConfig(BaseSettings):
    db_provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.SQLITE, description="Vector Database provider"
    )
    connection_url: Optional[str] = Field(
        default=None,
        description="Database connection URL (for network-based stores like PostgreSQL)",
    )
    persist_directory: Path = Field(
        default=Path("./vectordb"), description="Directory to vector store"
    )
    collection_name: str = Field(
        default="code", description="Name of the vector store collection"
    )
    codebase_directory: Path = Field(
        default=Path("./"), description="Root directory of the codebase to analyze"
    )
    codebase_url: Optional[str] = Field(
        default=None, description="URL of the codebase repository"
    )
    similarity_metric: Literal["cosine"] = Field(
        default="cosine", description="Similarity metric for vector search"
    )
    content_field: Optional[str] = Field(
        default="content",
        description="Field to store the actual content in the vector store",
    )
    state_store: StateStoreConfig = Field(default_factory=StateStoreConfig)


class RerankerConfig(BaseSettings):
    enabled: bool = Field(
        # The reranker is disabled by default, since the reranker should be fine-tuned against each codebase
        default=False,
        description="Enable reranking",
    )
    model_name: str = Field(
        default="llama3.2", description="Name of the reranker model to use"
    )
    model_provider: str = Field(
        default=ModelProvider.OLLAMA,
        description="Reranker provider (anthropic, openai, ollama, etc)",
    )
    api_key: Optional[str] = Field(
        default=None, description="API key for the model provider"
    )
    top_k: int = Field(
        default=10,
        description="Number of most relevant documents to return from reranking",
    )
    relevance_threshold: float = Field(
        default=0.5,
        description="Minimum relevance score to include a document in reranking",
    )
    max_sequence_length: int = Field(
        default=256, description="Maximum sequence length for the reranker model"
    )

    @field_validator("api_key", mode="after")
    @classmethod
    def validate_api_key(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_api_key(v, info)


class EvaluatorConfig(LLMConfig):
    evaluation_rounds: int = Field(
        default=1, description="Number of evaluation rounds per test case"
    )


class AppConfig(BaseSettings):
    model_config = generate_model_config(
        env_file=".env.app",
    )
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    db: DBConfig = Field(default_factory=DBConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chat_analytics: ChatbotAnalyticsConfig = Field(
        default_factory=ChatbotAnalyticsConfig
    )
    retrieval: MultiStageRetrievalConfig = Field(
        default_factory=MultiStageRetrievalConfig
    )
    server: ServerConfig = Field(default_factory=ServerConfig)

