from __future__ import annotations

import json
import struct
import uuid
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from knowlang.database.config import VectorStoreConfig
from knowlang.configs import AppConfig
from knowlang.core.types import VectorStoreProvider
from knowlang.vector_stores.base import (
    SearchResult,
    VectorStore,
    VectorStoreError,
    VectorStoreInitError,
)
from knowlang.vector_stores.factory import register_vector_store
from knowlang.utils import FancyLogger


if TYPE_CHECKING:
    try:
        import sqlite3
        import sqlite_vec
    except ImportError as e:
        raise ImportError(
            'SQLite vector store is not installed. Please install it using `pip install "knowlang[sqlite]"`.'
        ) from e

LOG = FancyLogger(__name__)


@register_vector_store(VectorStoreProvider.SQLITE)
class SqliteVectorStore(VectorStore):
    """SQLite implementation of VectorStore using SQLAlchemy and the sqlite-vec extension.
    
    This implementation uses a single virtual table to store both vector embeddings
    and document content/metadata, eliminating redundancy and improving performance.
    """

    @classmethod
    def from_cfg(cls, cfg: VectorStoreConfig) -> "SqliteVectorStore":
        """Create SqliteVectorStore from VectorStoreConfig."""
        return cls(
            app_config=cfg,
            db_path=cfg.connection_string,
            table_name=cfg.table_name,
            embedding_dim=cfg.embedding.dimension,
        )

    @classmethod
    def create_from_config(cls, config: AppConfig) -> "SqliteVectorStore":
        db_config = config.db
        embedding_config = config.embedding
        if not db_config.connection_url:
            raise VectorStoreInitError("Database path not set for SqliteVectorStore.")
        return cls(
            app_config=config,
            db_path=db_config.connection_url,
            table_name=db_config.collection_name,
            embedding_dim=embedding_config.dimension,
            similarity_metric=db_config.similarity_metric,
            content_field=db_config.content_field,
        )

    def __init__(
        self,
        app_config: AppConfig,
        db_path: str,
        table_name: str,
        embedding_dim: int,
        similarity_metric: Literal["cosine", "l1", "l2", "inner_product"] = "cosine",
        content_field: Optional[str] = "content",
    ):
        super().__init__()
        self.app_config = app_config
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.content_field = content_field or "content"
        self.engine = None
        self.AsyncSession = None

    def assert_initialized(self) -> None:
        """Assert that the vector store is initialized"""
        if self.engine is None or self.AsyncSession is None:
            raise ValueError(f"{self.__class__.__name__} is not initialized. Call initialize() first.")

    async def initialize(self) -> None:
        """Initialize the SQLite vector store with sqlite-vec extension."""
        try:
            # Create async engine
            self.engine = create_async_engine(self.db_path)

            # Create async session factory
            self.AsyncSession = async_sessionmaker(bind=self.engine)

            # Create the virtual table with all required columns
            async with self.get_session(auto_commit=False) as session:
                # Create virtual table that stores everything we need
                await session.execute(
                    text(
                        f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{self.embedding_dim}],
                        content TEXT,
                        doc_metadata TEXT
                    )
                """
                    )
                )
                await session.commit()

        except SQLAlchemyError as e:
            self.engine = None
            self.AsyncSession = None
            raise VectorStoreInitError(f"Failed to initialize SqliteVectorStore: {e}")

    def _get_content_from_document_or_metadata(
        self, document: str, metadata: Dict[str, Any]
    ) -> str:
        """Extract content from metadata or use document string."""
        if self.content_field in metadata:
            return str(metadata[self.content_field])
        return document

    @asynccontextmanager
    async def get_session(self, auto_commit: bool = True):
        """Context manager to get a database session with sqlite-vec extension loaded."""
        async with self.AsyncSession() as session:
            try:
                raw_connection = await session.connection()
                dbapi_connection = await raw_connection.get_raw_connection()
                
                try:
                    import sqlite_vec
                    await dbapi_connection.driver_connection.enable_load_extension(True)
                    await dbapi_connection.driver_connection.load_extension(sqlite_vec.loadable_path())
                    await dbapi_connection.driver_connection.enable_load_extension(False)
                except Exception as e:
                    raise VectorStoreInitError(
                        f"Failed to load sqlite-vec extension. Ensure it's installed and accessible. {e}"
                    ) from e

                yield session
                if auto_commit:
                    await session.commit()
            except SQLAlchemyError as e:
                await session.rollback()
                LOG.error(f"Database operation failed: {e}")
                raise e

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents to the vector store."""
        await self.ensure_initialized()

        if not (len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError(
                "documents, embeddings, and metadatas lists must have the same length."
            )
        if ids and len(ids) != len(documents):
            raise ValueError(
                "If provided, ids list must have the same length as documents."
            )

        try:
            doc_ids = ids if ids else [str(uuid.uuid4()) for _ in documents]

            async with self.get_session() as session:
                for i, (doc_ref_content, embedding_list, metadata) in enumerate(
                    zip(documents, embeddings, metadatas)
                ):
                    doc_id = doc_ids[i]
                    content_to_store = self._get_content_from_document_or_metadata(
                        doc_ref_content, metadata
                    )
                    
                    from sqlite_vec import serialize_float32
                    embedding_bytes = serialize_float32(embedding_list)
                    metadata_str = json.dumps(metadata)

                    # Insert directly into the virtual table
                    await session.execute(
                        text(
                            f"""
                        INSERT INTO {self.table_name} (id, embedding, content, doc_metadata) 
                        VALUES (:id, :embedding, :content, :doc_metadata)
                    """
                        ),
                        {
                            "id": doc_id,
                            "embedding": embedding_bytes,
                            "content": content_to_store,
                            "doc_metadata": metadata_str,
                        },
                    )

        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to add documents: {e}")
        except Exception as e:
            raise VectorStoreError(
                f"An unexpected error occurred while adding documents: {e}"
            )

    def accumulate_result(
        self,
        acc: List[SearchResult],
        record: Any,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Accumulate search results from database records."""
        db_doc_id, db_content, db_metadata_json, distance = record

        score = 1.0 - distance

        # NOTE: we are intentionally skipping the score threshold check here
        # if score_threshold is not None and score < score_threshold:
        #     return acc

        try:
            metadata = json.loads(db_metadata_json) if db_metadata_json else {}
        except json.JSONDecodeError:
            metadata = {}

        acc.append(
            SearchResult(
                id=db_doc_id, document=db_content, metadata=metadata, score=score
            )
        )
        return acc

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Query the vector store for similar documents."""
        await self.ensure_initialized()

        from sqlite_vec import serialize_float32
        query_embedding_bytes = serialize_float32(query_embedding)

        try:
            async with self.get_session() as session:
                # Query directly from the virtual table - no joins needed!
                sql_query = text(
                    f"""
                    SELECT
                        id,
                        content,
                        doc_metadata,
                        distance
                    FROM
                        {self.table_name}
                    WHERE
                        embedding MATCH :query_embedding
                    AND
                        k = :top_k
                    ORDER BY
                        distance ASC
                """
                )

                results_raw = await session.execute(
                    sql_query,
                    {"query_embedding": query_embedding_bytes, "top_k": top_k},
                )
                results_fetchall = results_raw.fetchall()

                search_results: List[SearchResult] = []
                for record in results_fetchall:
                    self.accumulate_result(search_results, record, score_threshold)

                # Apply metadata filtering if provided
                if filter:
                    filtered_results = [
                        sr
                        for sr in search_results
                        if all(
                            sr.metadata.get(key) == value
                            for key, value in filter.items()
                        )
                    ]
                    return filtered_results
                
                return search_results

        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to query: {e}")
        except struct.error as e:
            raise VectorStoreError(f"Failed to pack query embedding: {e}")

    async def delete(self, ids: List[str]) -> None:
        """Delete documents from the vector store."""
        await self.ensure_initialized()
        if not ids:
            return

        try:
            async with self.get_session() as session:
                # Simple deletion from single table
                placeholders = ",".join([":id" + str(i) for i in range(len(ids))])
                params = {f"id{i}": doc_id for i, doc_id in enumerate(ids)}
                
                result = await session.execute(
                    text(f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"),
                    params,
                )

                LOG.debug(f"Successfully deleted {result.rowcount} documents from vector store.")

        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to delete documents: {e}")

    async def get_documents(self, ids: List[str]) -> Optional[List[SearchResult]]:
        """Get specific documents by their IDs."""
        await self.ensure_initialized()

        try:
            async with self.get_session() as session:
                placeholders = ",".join([":id" + str(i) for i in range(len(ids))])
                params = {f"id{i}": doc_id for i, doc_id in enumerate(ids)}
                
                result = await session.execute(
                    text(f"SELECT id, content, doc_metadata FROM {self.table_name} WHERE id IN ({placeholders})"),
                    params,
                )
                rows = result.fetchall()

                results = []
                for row in rows:
                    doc_id, content, metadata_str = row
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    results.append(SearchResult(
                        id=doc_id,
                        document=content,
                        metadata=metadata,
                        score=0.0,
                    ))
                return results

        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to get documents: {e}")

    async def update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Update a document in the vector store.
        
        Note: Since virtual tables may not support UPDATE operations,
        we use DELETE + INSERT pattern.
        """
        await self.ensure_initialized()

        from sqlite_vec import serialize_float32
        embedding_bytes = serialize_float32(embedding)
        metadata_str = json.dumps(metadata)
        content_to_store = self._get_content_from_document_or_metadata(document, metadata)

        try:
            async with self.get_session() as session:
                # Check if document exists
                check_result = await session.execute(
                    text(f"SELECT id FROM {self.table_name} WHERE id = :id"),
                    {"id": id},
                )
                if not check_result.fetchone():
                    raise VectorStoreError(f"Document with id {id} not found for update.")

                # Delete the existing document
                await session.execute(
                    text(f"DELETE FROM {self.table_name} WHERE id = :id"),
                    {"id": id},
                )

                # Insert the updated document
                await session.execute(
                    text(
                        f"""
                    INSERT INTO {self.table_name} (id, embedding, content, doc_metadata) 
                    VALUES (:id, :embedding, :content, :doc_metadata)
                """
                    ),
                    {
                        "id": id,
                        "embedding": embedding_bytes,
                        "content": content_to_store,
                        "doc_metadata": metadata_str,
                    },
                )

        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to update document: {e}")
        except struct.error as e:
            raise VectorStoreError(f"Failed to pack embedding for update: {e}")

    async def get_all(self) -> List[SearchResult]:
        """Get all documents from the vector store."""
        await self.ensure_initialized()

        try:
            async with self.get_session() as session:
                result = await session.execute(
                    text(f"SELECT id, content, doc_metadata FROM {self.table_name}")
                )
                rows = result.fetchall()

                results: List[SearchResult] = []
                for row in rows:
                    doc_id, content, metadata_str = row
                    try:
                        metadata = json.loads(metadata_str) if metadata_str else {}
                    except json.JSONDecodeError:
                        metadata = {}
                    
                    results.append(
                        SearchResult(
                            id=doc_id,
                            document=content,
                            metadata=metadata,
                            score=0.0,
                        )
                    )
                return results

        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to get all documents: {e}")

    async def close(self) -> None:
        """Close the database connection."""
        if self.engine:
            try:
                await self.engine.dispose()
            except SQLAlchemyError as e:
                raise VectorStoreError(f"Failed to close connection: {e}")
            finally:
                self.engine = None
                self.AsyncSession = None
