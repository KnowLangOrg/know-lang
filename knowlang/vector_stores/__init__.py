# TODO: ideally, we should refactor the register_vectorestore to be called in the factory instead of script intialization stage
# ruff: noqa: F401
from . import sqlite  # Add this line
from . import chroma, postgres
from .base import (
    VectorStore,
    VectorStoreError,
    VectorStoreInitError,
    VectorStoreNotFoundError,
)
from .sqlite import SqliteVectorStore  # Added this line

__all__ = [
    "VectorStoreError",
    "VectorStoreInitError",
    "VectorStoreNotFoundError",
    "VectorStore",
    "SqliteVectorStore",  # Added this line
]
