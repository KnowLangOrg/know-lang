from __future__ import annotations

import json
import struct
import uuid
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from sqlalchemy import (
    BLOB, Column, Integer, String, Text, create_engine, event, select, text
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

from knowlang.configs import AppConfig
from knowlang.core.types import VectorStoreProvider
from knowlang.vector_stores.base import (
    SearchResult,
    VectorStore,
    VectorStoreError,
    VectorStoreInitError,
)
from knowlang.vector_stores.factory import register_vector_store

if TYPE_CHECKING:
    try:
        import sqlite3
        import sqlite_vec
    except ImportError as e:
        raise ImportError(
            'SQLite vector store is not installed. Please install it using `pip install "knowlang[sqlite]"`.'
        ) from e

Base = declarative_base()


class VectorDocumentModel(Base):
    """SQLAlchemy model for vector documents."""
    __tablename__ = 'vector_documents'
    
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    embedding = Column(BLOB)
    doc_metadata = Column(Text)  # Store metadata as JSON string


@register_vector_store(VectorStoreProvider.SQLITE)
class SqliteVectorStore(VectorStore):
    """SQLite implementation of VectorStore using SQLAlchemy and the sqlite-vec extension."""

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
        self.Session = None
        
        # Update the model's table name dynamically
        VectorDocumentModel.__tablename__ = self.table_name
    
    @property
    def virtual_table(self) -> str:
        """Returns the name of the virtual table used for vector indexing."""
        return f"{self.table_name}_vec_idx"

    def _setup_sqlite_vec_extension(self) -> None:
        """Set up sqlite-vec extension loading for the engine."""
        @event.listens_for(self.engine, "connect")
        def load_sqlite_vec(conn : sqlite3.Connection,connection_record):
            try:
                import sqlite_vec
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception as e:
                raise VectorStoreInitError(
                    "Failed to load sqlite-vec extension. Ensure it's installed and accessible. {e}"
                ) from e
        
    def initialize(self) -> None:
        try:
            # Create engine
            self.engine = create_engine(self.db_path)
            
            # Set up sqlite-vec extension loading
            self._setup_sqlite_vec_extension()
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)

            # Create tables
            Base.metadata.create_all(self.engine)

            # Create virtual table for vector indexing (using raw SQL as it's sqlite-vec specific)
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.virtual_table} USING vec0(
                        embedding float[{self.embedding_dim}]
                    )
                """))
                conn.commit()
                
        except SQLAlchemyError as e:
            self.engine = None
            self.Session = None
            raise VectorStoreInitError(f"Failed to initialize SqliteVectorStore: {e}")

    def _get_content_from_document_or_metadata(
        self, document: str, metadata: Dict[str, Any]
    ) -> str:
        """Extract content from metadata or use document string."""
        if self.content_field in metadata:
            return str(metadata[self.content_field])
        return document

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        if not self.engine or not self.Session:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )

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

            with self.Session() as session:
                for i, (doc_ref_content, embedding_list, metadata) in enumerate(
                    zip(documents, embeddings, metadatas)
                ):
                    doc_id = doc_ids[i]
                    content_to_store = self._get_content_from_document_or_metadata(
                        doc_ref_content, metadata
                    )
                    embedding_bytes = struct.pack(f"{self.embedding_dim}f", *embedding_list)
                    metadata_str = json.dumps(metadata)

                    # Create document record
                    doc_model = VectorDocumentModel(
                        id=doc_id,
                        content=content_to_store,
                        embedding=embedding_bytes,
                        doc_metadata=metadata_str
                    )
                    session.add(doc_model)
                    session.flush()  # Ensure the record is inserted to get rowid

                    # Get the rowid for the virtual table
                    rowid_result = session.execute(text(f"""
                        SELECT rowid FROM {self.table_name} WHERE id = :doc_id
                    """), {"doc_id": doc_id}).scalar()

                    # Insert into virtual table for vector indexing
                    session.execute(text(f"""
                        INSERT INTO {self.virtual_table} (rowid, embedding) 
                        VALUES (:rowid, :embedding)
                    """), {
                        "rowid": rowid_result,
                        "embedding": embedding_bytes
                    })

                session.commit()
                
        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to add documents: {e}")
        except struct.error as e:
            raise VectorStoreError(f"Failed to pack embedding into bytes: {e}")
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

        if score_threshold is not None and score < score_threshold:
            return acc

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
    ) -> List[SearchResult]:
        if not self.engine or not self.Session:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )

        query_embedding_bytes = struct.pack(f"{self.embedding_dim}f", *query_embedding)

        try:
            with self.Session() as session:
                # Use raw SQL for vector similarity search as it's sqlite-vec specific
                sql_query = text(f"""
                    SELECT
                        m.id,
                        m.{self.content_field},
                        m.doc_metadata,
                        v.distance
                    FROM
                        {self.virtual_table} v
                    JOIN
                        {self.table_name} m ON v.rowid = m.rowid
                    WHERE
                        v.embedding MATCH :query_embedding
                    LIMIT :top_k
                """)
                
                results_raw = session.execute(sql_query, {
                    "query_embedding": query_embedding_bytes,
                    "top_k": top_k
                }).fetchall()

                search_results: List[SearchResult] = []
                for record in results_raw:
                    self.accumulate_result(search_results, record)

                if filter:
                    # Post-filtering based on metadata
                    filtered_results = [
                        sr
                        for sr in search_results
                        if all(
                            sr.metadata.get(key) == value for key, value in filter.items()
                        )
                    ]
                    return filtered_results
                return search_results
                
        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to query: {e}")
        except struct.error as e:
            raise VectorStoreError(f"Failed to pack query embedding: {e}")

    async def delete(self, ids: List[str]) -> None:
        if not self.engine or not self.Session:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )
        if not ids:
            return

        try:
            with self.Session() as session:
                # Get rowids for the documents to delete
                stmt = select(VectorDocumentModel).where(VectorDocumentModel.id.in_(ids))
                docs_to_delete = session.execute(stmt).scalars().all()
                
                if not docs_to_delete:
                    return

                row_ids_to_delete = []
                for doc in docs_to_delete:
                    # Get the rowid
                    rowid_result = session.execute(text(f"""
                        SELECT rowid FROM {self.table_name} WHERE id = :doc_id
                    """), {"doc_id": doc.id}).scalar()
                    if rowid_result:
                        row_ids_to_delete.append(rowid_result)

                # Delete from main table
                for doc in docs_to_delete:
                    session.delete(doc)

                # Delete from virtual table
                if row_ids_to_delete:
                    for rowid in row_ids_to_delete:
                        session.execute(text(f"""
                            DELETE FROM {self.virtual_table} WHERE rowid = :rowid
                        """), {"rowid": rowid})

                session.commit()
                
        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to delete documents: {e}")

    async def get_document(self, id: str) -> Optional[SearchResult]:
        if not self.engine or not self.Session:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )
            
        try:
            with self.Session() as session:
                stmt = select(VectorDocumentModel).where(VectorDocumentModel.id == id)
                doc = session.execute(stmt).scalar_one_or_none()
                
                if doc:
                    try:
                        metadata = json.loads(doc.doc_metadata) if doc.doc_metadata else {}
                    except json.JSONDecodeError:
                        metadata = {}
                    return SearchResult(
                        id=doc.id, 
                        document=getattr(doc, self.content_field, doc.content), 
                        metadata=metadata, 
                        score=0.0
                    )
                return None
                
        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to get document: {e}")

    async def update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        if not self.engine or not self.Session:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )

        embedding_bytes = struct.pack(f"{self.embedding_dim}f", *embedding)
        metadata_str = json.dumps(metadata)

        try:
            with self.Session() as session:
                # Get the document to update
                stmt = select(VectorDocumentModel).where(VectorDocumentModel.id == id)
                doc = session.execute(stmt).scalar_one_or_none()
                
                if not doc:
                    raise VectorStoreError(f"Document with id {id} not found for update.")

                # Get rowid for virtual table update
                rowid_result = session.execute(text(f"""
                    SELECT rowid FROM {self.table_name} WHERE id = :doc_id
                """), {"doc_id": id}).scalar()

                # Update main table
                doc.content = document
                doc.embedding = embedding_bytes
                doc.doc_metadata = metadata_str

                # Update virtual table
                if rowid_result:
                    # Delete and re-insert in virtual table
                    session.execute(text(f"""
                        DELETE FROM {self.virtual_table} WHERE rowid = :rowid
                    """), {"rowid": rowid_result})
                    
                    session.execute(text(f"""
                        INSERT INTO {self.virtual_table} (rowid, embedding) 
                        VALUES (:rowid, :embedding)
                    """), {"rowid": rowid_result, "embedding": embedding_bytes})

                session.commit()
                
        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to update document: {e}")
        except struct.error as e:
            raise VectorStoreError(f"Failed to pack embedding for update: {e}")

    async def get_all(self) -> List[SearchResult]:
        if not self.engine or not self.Session:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )
            
        try:
            with self.Session() as session:
                stmt = select(VectorDocumentModel)
                docs = session.execute(stmt).scalars().all()
                
                results: List[SearchResult] = []
                for doc in docs:
                    try:
                        metadata = json.loads(doc.doc_metadata) if doc.doc_metadata else {}
                    except json.JSONDecodeError:
                        metadata = {}
                    results.append(
                        SearchResult(
                            id=doc.id, 
                            document=getattr(doc, self.content_field, doc.content), 
                            metadata=metadata, 
                            score=0.0
                        )
                    )
                return results
                
        except SQLAlchemyError as e:
            raise VectorStoreError(f"Failed to get all documents: {e}")

    async def close(self) -> None:
        if self.engine:
            try:
                self.engine.dispose()
            except SQLAlchemyError as e:
                raise VectorStoreError(f"Failed to close connection: {e}")
            finally:
                self.engine = None
                self.Session = None

    def __del__(self):
        # Attempt to close connection if not already closed.
        if self.engine:
            try:
                self.engine.dispose()
            except Exception:  # nosec B110
                pass  # Suppress errors during garbage collection
