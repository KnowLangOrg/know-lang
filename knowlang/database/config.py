from pydantic import BaseModel, Field

from knowlang.configs.defaults import DEFAULT_VECTOR_COLLECTION_NAME, DEFAULT_SQLITE_DB_CONNECTION_URL_ASYNC
from knowlang.core.types import VectorStoreProvider
from knowlang.configs import EmbeddingConfig


class VectorStoreConfig(BaseModel):
    provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.SQLITE,
        description="Vector Database provider"
    )
    connection_string: str = Field(
        default=DEFAULT_SQLITE_DB_CONNECTION_URL_ASYNC,
        description="Connection string for the vector store"
    )
    table_name: str = Field(
        default=DEFAULT_VECTOR_COLLECTION_NAME,
        description="Name of the table in the vector store"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Configuration for the embeddings"
    )
