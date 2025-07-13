from __future__ import annotations

import asyncio
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from knowlang.configs.config import AppConfig
from knowlang.database.config import VectorStoreConfig
from knowlang.search import SearchResult
from knowlang.search.base import SearchMethodology
from knowlang.search.searchable_store import SearchableStore
from knowlang.search.vector_search import VectorSearchStrategy


class InitializationState(str, Enum):
    """Enum for tracking vector store initialization state"""

    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"


class VectorStoreError(Exception):
    """Base exception for vector store errors"""

    pass


class VectorStoreInitError(VectorStoreError):
    """Error during vector store initialization"""

    pass


class VectorStoreNotFoundError(VectorStoreError):
    """Error when requested vector store provider is not found"""

    pass


class VectorStore(SearchableStore):
    """Abstract base class for vector store implementations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_capability(SearchMethodology.VECTOR)
        self.register_strategy(VectorSearchStrategy())

        # Initialization state management
        self._init_state: InitializationState = InitializationState.NOT_INITIALIZED
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._init_error: Optional[Exception] = None

    async def ensure_initialized(self) -> None:
        """
        Ensure the vector store is initialized, performing lazy initialization if needed.
        This method is thread-safe and will only initialize once.
        """
        # Fast path - already initialized
        if self._init_state == InitializationState.INITIALIZED:
            return

        # Handle failed initialization
        if self._init_state == InitializationState.FAILED:
            if self._init_error:
                raise VectorStoreInitError(
                    f"Vector store initialization failed previously: {self._init_error}"
                ) from self._init_error
            else:
                raise VectorStoreInitError(
                    "Vector store initialization failed previously"
                )

        # Use lock to ensure only one initialization attempt happens
        async with self._init_lock:
            # Double-check pattern - another coroutine might have initialized while we waited
            if self._init_state == InitializationState.INITIALIZED:
                return

            if self._init_state == InitializationState.FAILED:
                if self._init_error:
                    raise VectorStoreInitError(
                        f"Vector store initialization failed: {self._init_error}"
                    ) from self._init_error
                else:
                    raise VectorStoreInitError("Vector store initialization failed")

            # Check if currently initializing (shouldn't happen with proper lock usage)
            if self._init_state == InitializationState.INITIALIZING:
                raise VectorStoreInitError("Vector store is already being initialized")

            # Perform initialization
            self._init_state = InitializationState.INITIALIZING
            self._init_error = None

            try:
                await self.initialize()
                self._init_state = InitializationState.INITIALIZED
            except Exception as e:
                self._init_state = InitializationState.FAILED
                self._init_error = e
                raise VectorStoreInitError(
                    f"Failed to initialize vector store: {str(e)}"
                ) from e

    @classmethod
    @abstractmethod
    def assert_initialized(cls) -> None:
        """Assert that the vector store is initialized"""
        pass

    @classmethod
    def from_cfg(cls, cfg: VectorStoreConfig) -> "VectorStore":
        raise NotImplementedError(
            "VectorStore subclasses must implement the from_cfg method to create an instance from configuration."
        )

    @classmethod
    @abstractmethod
    def create_from_config(config: AppConfig) -> "VectorStore":
        """Create a VectorStore instance from configuration"""
        pass

    @classmethod
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store"""
        pass

    @abstractmethod
    def accumulate_result(
        self,
        acc: List[SearchResult],
        record: Any,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Accumulate search result"""
        pass

    @abstractmethod
    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents with their embeddings and metadata"""
        pass

    @abstractmethod
    async def query(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """Query the vector store for similar documents"""
        pass

    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """Search for similar documents"""
        await self.ensure_initialized()

        return await self.query(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            **kwargs,
        )

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete documents by their IDs"""
        pass

    @abstractmethod
    async def get_documents(
        self, ids: List[str], **kwargs
    ) -> Optional[List[SearchResult]]:
        """Retrieve multiple documents by their IDs"""
        pass

    @abstractmethod
    async def update_document(
        self, id: str, document: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> None:
        """Update an existing document"""
        pass

    @abstractmethod
    async def get_all(self) -> List[SearchResult]:
        """Get all documents in the store"""
        pass
