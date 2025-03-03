from __future__ import annotations
from typing import Dict, Dict, Type
from knowlang.vector_stores import (
    VectorStore, 
    VectorStoreError,
    VectorStoreInitError
)
from knowlang.core.types import VectorStoreProvider
from knowlang.vector_stores.chroma import ChromaVectorStore
from knowlang.vector_stores.postgres import PostgresVectorStore
from knowlang.configs import DBConfig, EmbeddingConfig

VECTOR_STORE_CLASS_DICT : Dict[VectorStoreProvider, Type[VectorStore]] = {
    VectorStoreProvider.POSTGRES: PostgresVectorStore,
    VectorStoreProvider.CHROMA: ChromaVectorStore
}

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    @staticmethod
    def get(
        config: DBConfig,
        embedding_config: EmbeddingConfig
    ) -> VectorStore:
        """
        Create and initialize a vector store instance
        
        Args:
            config: Database configuration
            
        Returns:
            Initialized vector store instance
            
        Raises:
            VectorStoreInitError: If initialization fails
        """
        try:
            store_cls = VECTOR_STORE_CLASS_DICT.get(config.db_provider)
            vector_store: VectorStore = store_cls.create_from_config(config, embedding_config)
            
            # Initialize the store
            vector_store.initialize()
            
            return vector_store
            
        except VectorStoreError:
            # Re-raise VectorStoreError subclasses as-is
            raise
        except Exception as e:
            # Wrap any other exceptions
            raise VectorStoreInitError(f"Failed to create vector store: {str(e)}") from e