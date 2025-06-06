from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from typing import Any, Dict, List, Literal, Optional

from knowlang.configs import AppConfig
from knowlang.core.types import VectorStoreProvider
from knowlang.vector_stores.base import (
    SearchResult,
    VectorStore,
    VectorStoreError,
    VectorStoreInitError,
)
from knowlang.vector_stores.factory import register_vector_store


@register_vector_store(VectorStoreProvider.SQLITE)
class SqliteVectorStore(VectorStore):
    """SQLite implementation of VectorStore using the sqlite-vec extension."""

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
        self.similarity_metric = similarity_metric  # Store for potential use
        self.content_field = content_field
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def initialize(self) -> None:
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.enable_load_extension(True)
            try:
                # Attempt to load 'vec' extension first (common name)
                self.conn.load_extension("vec")
            except sqlite3.OperationalError:
                # Fallback to searching common paths if direct load fails
                common_paths = [
                    "/usr/local/lib/vec.so",  # Linux
                    "vec.dylib",  # macOS
                    "vec.dll",  # Windows
                    "./vec.so",  # Current dir (dev/test)
                    "./vec.dylib",  # Current dir (dev/test)
                    "./vec.dll",  # Current dir (dev/test)
                ]
                loaded = False
                for path in common_paths:
                    try:
                        self.conn.load_extension(path)
                        loaded = True
                        break
                    except sqlite3.OperationalError:
                        continue
                if not loaded:
                    raise VectorStoreInitError(
                        "Failed to load sqlite-vec extension. Ensure it's installed and accessible. "
                        "Searched common paths and 'vec'."
                    )
            # It's generally recommended to disable extension loading after successful load for security.
            # However, some applications might need it enabled if extensions are loaded on-demand per-connection.
            # For sqlite-vec, it's typically loaded once.
            # self.conn.enable_load_extension(False) # Re-evaluate if this is needed or causes issues.

            self.cursor = self.conn.cursor()

            actual_content_field = (
                self.content_field if self.content_field else "content"
            )
            self.cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    {actual_content_field} TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT -- Store metadata as JSON string
                )
            """
            )

            virtual_table_name = f"{self.table_name}_vec_idx"
            self.cursor.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {virtual_table_name} USING vec_f32(
                    embedding({self.embedding_dim}) /* affinity: BLOB */
                )
            """
            )
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn = None  # Ensure connection is None if init fails
            self.cursor = None
            raise VectorStoreInitError(f"Failed to initialize SqliteVectorStore: {e}")

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        if not self.conn or not self.cursor:
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

        virtual_table_name = f"{self.table_name}_vec_idx"
        actual_content_field = self.content_field if self.content_field else "content"

        try:
            doc_ids = ids if ids else [str(uuid.uuid4()) for _ in documents]

            for i, (doc_ref_content, embedding_list, metadata) in enumerate(
                zip(documents, embeddings, metadatas)
            ):
                doc_id = doc_ids[i]

                if actual_content_field in metadata:
                    content_to_store = str(metadata[actual_content_field])
                else:
                    # If content_field not in metadata, use the string from 'documents' list
                    content_to_store = doc_ref_content

                embedding_bytes = struct.pack(f"{self.embedding_dim}f", *embedding_list)
                metadata_str = json.dumps(metadata)

                self.cursor.execute(
                    f"INSERT INTO {self.table_name} (id, {actual_content_field}, embedding, metadata) VALUES (?, ?, ?, ?)",
                    (doc_id, content_to_store, embedding_bytes, metadata_str),
                )
                last_row_id = self.cursor.lastrowid

                self.cursor.execute(
                    f"INSERT INTO {virtual_table_name} (rowid, embedding) VALUES (?, ?)",
                    (last_row_id, embedding_bytes),
                )
            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()
            raise VectorStoreError(f"Failed to add documents: {e}")
        except struct.error as e:
            if self.conn:
                self.conn.rollback()
            raise VectorStoreError(f"Failed to pack embedding into bytes: {e}")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise VectorStoreError(
                f"An unexpected error occurred while adding documents: {e}"
            )

    def accumulate_result(
        self,
        acc: List[SearchResult],
        record: Any,  # Expected: (doc_id_from_db, content_from_db, metadata_json_from_db, distance_from_db)
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        db_doc_id, db_content, db_metadata_json, distance = record

        score = 1.0 - distance

        if (
            score_threshold is not None and score < score_threshold
        ):  # Higher score (similarity) is better
            return acc

        try:
            metadata = json.loads(db_metadata_json)
        except json.JSONDecodeError:
            metadata = {}

        # The content_field is already resolved by the query (m.{actual_content_field})
        # So db_content IS the document content.
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
        if not self.conn or not self.cursor:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )

        query_embedding_bytes = struct.pack(f"{self.embedding_dim}f", *query_embedding)
        virtual_table_name = f"{self.table_name}_vec_idx"
        actual_content_field = self.content_field if self.content_field else "content"

        # sqlite-vec's MATCH operator implicitly handles KNN and ordering by distance.
        sql_query = f"""
            SELECT
                m.id,
                m.{actual_content_field},
                m.metadata,
                v.distance
            FROM
                {virtual_table_name} v
            JOIN
                {self.table_name} m ON v.rowid = m.rowid
            WHERE
                v.embedding MATCH ?1
            LIMIT ?2;
        """
        try:
            self.cursor.execute(sql_query, (query_embedding_bytes, top_k))
            results_raw = self.cursor.fetchall()
            search_results: List[SearchResult] = []
            for record in results_raw:
                self.accumulate_result(
                    search_results, record
                )  # No score_threshold from this level directly

            if filter:
                # Basic post-filtering. For production, consider if this is efficient enough.
                filtered_results = [
                    sr
                    for sr in search_results
                    if all(
                        sr.metadata.get(key) == value for key, value in filter.items()
                    )
                ]
                return filtered_results
            return search_results
        except sqlite3.Error as e:
            raise VectorStoreError(f"Failed to query: {e}")
        except struct.error as e:
            raise VectorStoreError(f"Failed to pack query embedding: {e}")

    async def delete(self, ids: List[str]) -> None:
        if not self.conn or not self.cursor:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )
        if not ids:
            return

        virtual_table_name = f"{self.table_name}_vec_idx"
        try:
            row_ids_to_delete = []
            for doc_id in ids:
                self.cursor.execute(
                    f"SELECT rowid FROM {self.table_name} WHERE id = ?", (doc_id,)
                )
                result = self.cursor.fetchone()
                if result:
                    row_ids_to_delete.append(result[0])

            if not row_ids_to_delete:
                return

            # Delete from main table by id
            placeholders = ",".join(["?"] * len(ids))
            self.cursor.execute(
                f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})", ids
            )

            # Delete from virtual table by rowid
            row_id_placeholders = ",".join(["?"] * len(row_ids_to_delete))
            self.cursor.execute(
                f"DELETE FROM {virtual_table_name} WHERE rowid IN ({row_id_placeholders})",
                row_ids_to_delete,
            )

            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()
            raise VectorStoreError(f"Failed to delete documents: {e}")

    async def get_document(self, id: str) -> Optional[SearchResult]:
        if not self.conn or not self.cursor:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )
        actual_content_field = self.content_field if self.content_field else "content"
        try:
            self.cursor.execute(
                f"SELECT id, {actual_content_field}, metadata FROM {self.table_name} WHERE id = ?",
                (id,),
            )
            record = self.cursor.fetchone()
            if record:
                doc_id, content, metadata_json = record
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}
                return SearchResult(
                    id=doc_id, content=content, metadata=metadata, score=0.0
                )  # Score not applicable
            return None
        except sqlite3.Error as e:
            raise VectorStoreError(f"Failed to get document: {e}")

    async def update_document(
        self,
        id: str,
        document: str,  # New content for the document
        embedding: List[float],  # Corrected type hint
        metadata: Dict[str, Any],
    ) -> None:
        if not self.conn or not self.cursor:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )

        actual_content_field = self.content_field if self.content_field else "content"
        virtual_table_name = f"{self.table_name}_vec_idx"
        embedding_bytes = struct.pack(f"{self.embedding_dim}f", *embedding)
        metadata_str = json.dumps(metadata)

        try:
            self.cursor.execute(
                f"SELECT rowid FROM {self.table_name} WHERE id = ?", (id,)
            )
            result = self.cursor.fetchone()
            if not result:
                raise VectorStoreError(f"Document with id {id} not found for update.")
            row_id = result[0]

            self.cursor.execute(
                f"UPDATE {self.table_name} SET {actual_content_field} = ?, embedding = ?, metadata = ? WHERE id = ?",
                (document, embedding_bytes, metadata_str, id),
            )
            # For sqlite-vec, updating the embedding in the virtual table usually means deleting the old rowid
            # and inserting it again with the new embedding vector.
            self.cursor.execute(
                f"DELETE FROM {virtual_table_name} WHERE rowid = ?", (row_id,)
            )
            self.cursor.execute(
                f"INSERT INTO {virtual_table_name} (rowid, embedding) VALUES (?, ?)",
                (row_id, embedding_bytes),
            )

            self.conn.commit()
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()
            raise VectorStoreError(f"Failed to update document: {e}")
        except struct.error as e:
            if self.conn:
                self.conn.rollback()
            raise VectorStoreError(f"Failed to pack embedding for update: {e}")

    async def get_all(self) -> List[SearchResult]:
        if not self.conn or not self.cursor:
            raise VectorStoreError(
                "Vector store is not initialized. Call initialize() first."
            )
        actual_content_field = self.content_field if self.content_field else "content"
        results: List[SearchResult] = []
        try:
            self.cursor.execute(
                f"SELECT id, {actual_content_field}, metadata FROM {self.table_name}"
            )
            for row in self.cursor.fetchall():
                doc_id, content, metadata_json = row
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}
                results.append(
                    SearchResult(
                        id=doc_id, content=content, metadata=metadata, score=0.0
                    )
                )  # Score not applicable
            return results
        except sqlite3.Error as e:
            raise VectorStoreError(f"Failed to get all documents: {e}")

    async def close(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except sqlite3.Error as e:
                raise VectorStoreError(f"Failed to close connection: {e}")
            finally:
                self.conn = None
                self.cursor = None

    def __del__(self):
        # Attempt to close connection if not already closed.
        # This is a fallback; explicit close() is preferred.
        if self.conn:
            try:
                self.conn.close()
            except Exception:  # nosec B110
                pass  # Suppress errors during garbage collection
