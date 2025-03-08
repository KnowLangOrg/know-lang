from __future__ import annotations
from typing import Type, Dict, TypeVar, Union, cast, Optional, TYPE_CHECKING
from knowlang.configs import DBConfig, EmbeddingConfig
from knowlang.core.types import VectorStoreProvider
from knowlang.vector_stores.base import VectorStore, VectorStoreError, VectorStoreInitError

# for type hinting during development
if TYPE_CHECKING:
    from knowlang.vector_stores.postgres import PostgresVectorStore
    from knowlang.vector_stores.postgres_hybrid import PostgresHybridStore
    from knowlang.vector_stores.chroma import ChromaVectorStore
    T = TypeVar('T', bound=Union["PostgresVectorStore", "PostgresHybridStore", "ChromaVectorStore"])
else:
    T = TypeVar('T')

VECTOR_STORE_CLASS_DICT: Dict[VectorStoreProvider, T] = {}

def register_vector_store(provider: VectorStoreProvider):
    """Decorator to register a state store implementation for a given provider key."""
    def decorator(cls: T) -> T:
        VECTOR_STORE_CLASS_DICT[provider] = cls
        return cast(T, cls)
    return decorator

def get_vector_store(provider: VectorStoreProvider) -> T:
    """Factory method to retrieve a vector store class."""
    if provider not in VECTOR_STORE_CLASS_DICT:
        raise ValueError(f"Vector store provider {provider} is not registered.")
    return VECTOR_STORE_CLASS_DICT.get(provider)

class VectorStoreFactory:
    """Factory for creating vector store instances with singleton pattern"""
    
    # Dictionary to store singleton instances
    _instances: Dict[str, VectorStore] = {}
    
    @classmethod
    def get(
        cls,
        config: DBConfig,
        embedding_config: EmbeddingConfig
    ) -> VectorStore:
        """
        Create and initialize a vector store instance or return existing instance
        
        Args:
            config: Database configuration
            embedding_config: Embedding configuration
            
        Returns:
            Initialized vector store instance
            
        Raises:
            VectorStoreInitError: If initialization fails
        """
        # Create a unique key based on the configuration
        instance_key = f"{config.db_provider}_{config.connection_url}"
        
        # Check if an instance with this configuration already exists
        if instance_key not in cls._instances:
            try:
                # Create new instance
                store_cls = get_vector_store(config.db_provider)
                vector_store: VectorStore = store_cls.create_from_config(config, embedding_config)
                
                # Initialize the store
                vector_store.initialize()
                
                # Save the instance
                cls._instances[instance_key] = vector_store
                
            except VectorStoreError:
                # Re-raise VectorStoreError subclasses as-is
                raise
            except Exception as e:
                # Wrap any other exceptions
                raise VectorStoreInitError(f"Failed to create vector store: {str(e)}") from e
        
        # Return the instance (either existing or newly created)
        return cls._instances[instance_key]
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset all singleton instances - primarily used for testing
        """
        cls._instances.clear()