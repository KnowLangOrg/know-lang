from __future__ import annotations

from knowlang.configs import DBConfig, EmbeddingConfig
from knowlang.vector_stores import (VectorStore, VectorStoreError,
                                    VectorStoreInitError)
from knowlang.vector_stores.base import get_vector_store


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
            store_cls = get_vector_store(config.db_provider)
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