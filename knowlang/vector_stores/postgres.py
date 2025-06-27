from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from vecs.collection import Record

from knowlang.configs import AppConfig
from knowlang.utils import FancyLogger
from knowlang.vector_stores.base import (SearchResult, VectorStore,
                                         VectorStoreError,
                                         VectorStoreInitError)
                                        
try:
    import vecs
except ImportError as e:
    raise ImportError(
        'Postgres vector store is not installed. Please install it using `pip install "knowlang[vecs]"`.'
    ) from e

LOG = FancyLogger(__name__)

class PostgresVectorStore(VectorStore):
    """Postgres implementation of VectorStore compatible with the pgvector extension using psycopg."""

    @classmethod
    def create_from_config(cls, config: AppConfig) -> "PostgresVectorStore":
        db_config = config.db
        embedding_config = config.embedding
        if not db_config.connection_url:
            raise VectorStoreInitError("Connection url not set for PostgresVectorStore.")
        return cls(
            app_config=config,
            connection_string=db_config.connection_url,
            table_name=db_config.collection_name,
            embedding_dim=embedding_config.dimension,
            similarity_metric=db_config.similarity_metric,
            content_field=db_config.content_field
        )

    def __init__(
        self,
        app_config: AppConfig,
        connection_string: str,
        table_name: str,
        embedding_dim: int,
        similarity_metric: Literal['cosine'] = 'cosine',
        content_field: Optional[str] = 'content'
    ):
        super().__init__()

        self.app_config = app_config
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.content_field = content_field
        self.collection = None

    def assert_initialized(self) -> None:
        """Assert that the vector store is initialized"""
        if self.collection is None:
            raise ValueError(f"{self.__class__.__name__} is not initialized.")

    async def initialize(self) -> None:
        """Initialize the Postgres vector store client and create a collection of vectors."""
        try:
            self.measure() # Validate similarity metric
            vx = vecs.create_client(self.connection_string)
            self.collection = vx.get_or_create_collection(name=self.table_name, dimension=self.embedding_dim)
        except Exception as e:
            raise VectorStoreInitError(f"Failed to initialize PostgresVectorStore: {str(e)}") from e
        
        try:
            self.collection.create_index(measure=self.measure(), replace=False)
        except Exception as e:
            # index already exists, ignore
            LOG.info(f"Index already exists for collection {self.table_name}")
            return

    def measure(self) -> vecs.IndexMeasure:
        if "cosine" in self.similarity_metric:
            return vecs.IndexMeasure.cosine_distance
        if "l1" in self.similarity_metric:
            return vecs.IndexMeasure.l1_distance
        if "l2" in self.similarity_metric:
            return vecs.IndexMeasure.l2_distance
        if "product" in self.similarity_metric:
            return vecs.IndexMeasure.max_inner_product
        raise VectorStoreError(f"Unsupported similarity metric: {self.similarity_metric}")

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        self.assert_initialized()
        if len(documents) != len(embeddings):
            raise VectorStoreError("Number of documents and embeddings must match.")
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        if len(documents) != len(ids):
            raise VectorStoreError("Number of documents and ids must match.")
        
        # Store the document content in metadata's content field
        for i, doc in enumerate(documents):
            if i < len(metadatas):
                metadatas[i][self.content_field] = doc
        
        vectors = [(id, emb, meta) for id, emb, meta in zip(ids, embeddings, metadatas)]
        self.collection.upsert(records=vectors)

    def accumulate_result(
        self,
        acc: List[SearchResult], 
        record: Record, 
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        id, dist, meta = record
        score = 1.0 - dist  # Convert distance to similarity score
        if score_threshold is None or score >= score_threshold:
            acc.append(SearchResult(
                document=meta.get(self.content_field, ""),
                metadata=meta,
                score=score
            ))
        return acc
    

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        records =  self.collection.query(
            data=query_embedding,
            limit=top_k,
            measure=self.measure(),
            include_value=True,
            include_metadata=True,
            filters=filter
        )

        return self.accumulate_result(
            acc=[],
            record=records,
            score_threshold=score_threshold
        )


    async def delete(self, ids: List[str]) -> None:
        self.assert_initialized()
        self.collection.delete(ids)

    async def get_documents(self, ids: List[str]) -> Optional[List[SearchResult]]:
        self.assert_initialized()
        results = self.collection.fetch(ids=ids)
        return results if results else None

    async def update_document(
        self,
        id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        self.assert_initialized()
        
        metadata[self.content_field] = document
            
        self.collection.upsert([(id, embedding, metadata)])

    async def get_all(self) -> List[SearchResult]:
        raise NotImplementedError("Postgres fetching all documents not implemented yet")