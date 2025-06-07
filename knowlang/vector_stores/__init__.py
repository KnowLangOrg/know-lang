# to register vector stores to factory
from . import (
    chroma,
    postgres, 
    postgres_hybrid,
    sqlite # Add this line
)
from .base import (VectorStore, VectorStoreError, VectorStoreInitError,
                   VectorStoreNotFoundError)
from .sqlite import SqliteVectorStore # Added this line

__all__ = [
    "VectorStoreError",
    "VectorStoreInitError",
    "VectorStoreNotFoundError",
    "VectorStore",
    "SqliteVectorStore", # Added this line
]
