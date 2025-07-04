from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type, TypeVar, Union, cast

from knowlang.configs.config import AppConfig
from knowlang.core.types import VectorStoreProvider
from knowlang.vector_stores.base import (
    VectorStore, 
    VectorStoreError,
    VectorStoreInitError
)
from knowlang.database.config import VectorStoreConfig

# for type hinting during development
if TYPE_CHECKING:
    from knowlang.vector_stores.chroma import ChromaVectorStore
    from knowlang.vector_stores.postgres import PostgresVectorStore
    from knowlang.vector_stores.postgres_hybrid import PostgresHybridStore
    from knowlang.vector_stores.sqlite import SqliteVectorStore # Add this line
    T = TypeVar('T', bound=Union["PostgresVectorStore", "PostgresHybridStore", "ChromaVectorStore", "SqliteVectorStore"]) # Add here
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
    def _get(
        cls,
        cfg: VectorStoreConfig
    ):
        cfg: VectorStoreConfig = cfg
        instance_key = f"{cfg.provider}_{cfg.connection_string}_{cfg.table_name}"

        if instance_key in cls._instances:
            return cls._instances[instance_key]
        
        try:
            # Get the vector store class based on the provider
            store_cls: Type[VectorStore] = get_vector_store(cfg.provider)
            # Create an instance from the configuration
            vector_store: VectorStore = store_cls.from_cfg(cfg)

            # Save the instance
            cls._instances[instance_key] = vector_store
            
            return vector_store
        except Exception as e:
            raise VectorStoreInitError(f"Failed to create vector store: {str(e)}") from e
    
    @classmethod
    def get(
        cls,
        app_config: AppConfig
    ) -> VectorStore:
        """
        Create and initialize a vector store instance or return existing instance
        
        Args:
            app_config: App configuration
            
        Returns:
            Initialized vector store instance
            
        Raises:
            VectorStoreInitError: If initialization fails
        """
        # FIXME: we should deprcated this method and use _get instead after refactoring
        if isinstance(app_config, VectorStoreConfig):
            return cls._get(app_config)

        db_config = app_config.db
        # Create a unique key based on the configuration
        instance_key = f"{db_config.db_provider}_{db_config.connection_url}"
        
        # Check if an instance with this configuration already exists
        if instance_key not in cls._instances:
            try:
                # Create new instance
                store_cls: Type[VectorStore] = get_vector_store(db_config.db_provider)
                vector_store: VectorStore = store_cls.create_from_config(app_config)
                
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