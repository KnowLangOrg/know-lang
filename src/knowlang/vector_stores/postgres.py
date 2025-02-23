from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from knowlang.vector_stores.base import (SearchResult, VectorStore,
                                         VectorStoreError,
                                         VectorStoreInitError)

if TYPE_CHECKING:
    from knowlang.configs import DBConfig, EmbeddingConfig


class PostgresVectorStore(VectorStore):
    """Postgres implementation of VectorStore compatible with the pgvector extension using psycopg."""

    def __init__(
        self,
        connection_string: str,
        table_name: str,
        embedding_dim: int,
        similarity_metric: Literal['cosine'] = 'cosine'
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.pool: Optional[ConnectionPool] = None

    def initialize(self) -> None:
        """Synchronously initialize the Postgres connection pool and ensure the vector store table exists."""
        try:
            self.pool = ConnectionPool(
                self.connection_string,
                min_size=1,
                max_size=10,
            )
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    # Ensure the pgvector extension is available.
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    # Create the table if it doesn't exist.
                    create_table_query = f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        document TEXT,
                        embedding vector({self.embedding_dim}),
                        metadata JSONB
                    );
                    """
                    cur.execute(create_table_query)
                register_vector(conn)
                conn.commit()
        except Exception as e:
            raise VectorStoreInitError(f"Failed to initialize PostgresVectorStore: {str(e)}") from e

    @classmethod
    def create_from_config(cls, config: DBConfig, embedding_config: EmbeddingConfig) -> "PostgresVectorStore":
        if not config.connection_url:
            raise VectorStoreInitError("Connection url not set for PostgresVectorStore.")
        return cls(
            connection_string=config.connection_url,
            table_name=config.collection_name,
            embedding_dim=embedding_config.dimension,
            similarity_metric=config.similarity_metric,
        )

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                insert_query = f"""
                INSERT INTO {self.table_name} (id, document, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
                """
                for doc, emb, meta, id_ in zip(documents, embeddings, metadatas, ids):
                    cur.execute(insert_query, (id_, doc, emb, Json(meta)))
            conn.commit()

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                search_query = f"""
                SELECT id, document, metadata, (embedding <=> (%s)::vector) AS distance
                FROM {self.table_name}
                ORDER BY embedding <=> (%s)::vector
                LIMIT %s;
                """
                cur.execute(search_query, (query_embedding, query_embedding, top_k))
                records = cur.fetchall()
                results = []
                for record in records:
                    score = 1.0 - record["distance"]  # Convert distance to similarity score
                    if score_threshold is None or score >= score_threshold:
                        results.append(SearchResult(
                            document=record["document"],
                            metadata=record["metadata"],
                            score=score
                        ))
                return results

    async def delete(self, ids: List[str]) -> None:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                delete_query = f"DELETE FROM {self.table_name} WHERE id = ANY(%s);"
                cur.execute(delete_query, (ids,))
            conn.commit()

    async def get_document(self, id: str) -> Optional[SearchResult]:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                query = f"SELECT id, document, metadata FROM {self.table_name} WHERE id = %s;"
                cur.execute(query, (id,))
                record = cur.fetchone()
                if record:
                    return SearchResult(
                        document=record["document"],
                        metadata=record["metadata"],
                        score=1.0  # Assuming direct retrieval is a perfect match.
                    )
                return None

    async def update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                update_query = f"""
                UPDATE {self.table_name}
                SET document = %s, embedding = %s, metadata = %s
                WHERE id = %s;
                """
                cur.execute(update_query, (document, embedding, Json(metadata), id))
            conn.commit()

    async def get_all(self) -> List[SearchResult]:
        if self.pool is None:
            raise VectorStoreError("PostgresVectorStore is not initialized.")
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                query = f"SELECT id, document, metadata FROM {self.table_name};"
                cur.execute(query)
                records = cur.fetchall()
                results = [
                    SearchResult(
                        document=record["document"],
                        metadata=record["metadata"],
                        score=1.0
                    )
                    for record in records
                ]
                return results