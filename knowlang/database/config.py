from pydantic import BaseModel, Field

from knowlang.core.types import VectorStoreProvider
from knowlang.configs import EmbeddingConfig


class VectorStoreConfig(BaseModel):
    provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.SQLITE,
        description="Vector Database provider"
    )
    connection_string: str = Field(
        default="sqlite:///vector_store.db",
        description="Connection string for the vector store"
    )
    table_name: str = Field(
        default="vector_store",
        description="Name of the table in the vector store"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Configuration for the embeddings"
    )
