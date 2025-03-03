from typing import Literal
from sqlalchemy import Column, Index, MetaData, String, Table, Text, column, func, select, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.types import UserDefinedType
from sqlalchemy import inspect
from sqlalchemy.schema import DDL

from knowlang.vector_stores.base import VectorStoreInitError
from knowlang.vector_stores.postgres import PostgresVectorStore
from knowlang.search.keyword_search import KeywordSearchableStore
from knowlang.configs import DBConfig, EmbeddingConfig
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)
Base = declarative_base()

# Define a custom type for pgvector, but we won't use it directly
# as we're relying on vecs for vector operations
class Vector(UserDefinedType):
    def get_col_spec(self, **kw):
        return "vector"


class PostgresHybridStore(PostgresVectorStore, KeywordSearchableStore):
    """PostgreSQL implementation that supports both vector similarity and keyword search."""

    @classmethod
    def create_from_config(
        cls, 
        config: DBConfig, 
        embedding_config: EmbeddingConfig
    ) -> "PostgresHybridStore":
        """Create a hybrid store instance from configuration."""
        if not config.connection_url:
            raise VectorStoreInitError("Connection url not set for PostgresHybridVectorStore.")
        
        return cls(
            connection_string=config.connection_url,
            table_name=config.collection_name,
            embedding_dim=embedding_config.dimension,
            similarity_metric=config.similarity_metric,
            content_field=getattr(config, "content_field", "content"),
        )

    def __init__(
        self,
        connection_string: str,
        table_name: str,
        embedding_dim: int,
        similarity_metric: Literal['cosine'] = 'cosine',
        text_search_config: str = "english",
        content_field: str = "content"
    ):
        """Initialize the hybrid store with both vector and text search capabilities.
        
        Args:
            connection_string: PostgreSQL connection URL
            table_name: Name of the collection/table
            embedding_dim: Dimension of the vector embeddings
            similarity_metric: Vector similarity metric to use
            text_search_config: PostgreSQL text search configuration
        """
        # Initialize vector store capabilities
        super().__init__(
            connection_string=connection_string,
            table_name=table_name,
            embedding_dim=embedding_dim,
            similarity_metric=similarity_metric
        )
        
        # Initialize text search specific attributes
        self.text_search_config = text_search_config
        self.content_field = content_field
        self.sqlalchemy_url = self.connection_string
        self.engine = None
        self.session = None 
        
        # Define metadata for direct SQL operations where ORM is not suitable
        self.metadata = None
        self.vecs_table = None
    
    def _setup_sqlalchemy(self):
        """Initialize SQLAlchemy engine and session"""
        if self.engine is None:
            try:
                self.engine = create_engine(self.connection_string)
                self.Session = sessionmaker(bind=self.engine)
                
                # Set up metadata for direct table operations
                self.metadata = Base.metadata
                
                # Define the table structure to match what vecs creates
                # This is for metadata access only, not for creating tables
                self.vecs_table = Table(
                    self.table_name,
                    self.metadata,
                    Column('id', String, primary_key=True),
                    Column('metadata', JSONB),
                    Column('embedding', Vector),  # We don't directly use this
                    Column('tsv', TSVECTOR)  # Will be added in initialize()
                )
                
                LOG.info(f"SQLAlchemy engine initialized for {self.table_name}")
            except Exception as e:
                raise VectorStoreInitError(f"Failed to initialize SQLAlchemy: {str(e)}") from e

    def initialize(self):
        super().initialize()
        self._setup_sqlalchemy()

        with self.Session() as session:
            # Check if table exists using SQLAlchemy
            inspector = inspect(self.engine)
            if not inspector.has_table(self.table_name):
                raise VectorStoreInitError(f"Table {self.table_name} does not exist. vecs should create it.")
            
            # Check if tsvector column exists
            if 'tsv' not in inspector.get_columns(self.table_name):
                LOG.info(f"Adding tsv column to {self.table_name}")
                
                # Add tsvector column for text search
                tsv_ddl = DDL(
                    f"ALTER TABLE {self.table_name} "
                    f"ADD COLUMN tsv tsvector GENERATED ALWAYS AS "
                    f"(to_tsvector('{self.text_search_config}', "
                    f"COALESCE((metadata->>'{self.content_field}')::text, ''))) STORED"
                )
                session.execute(tsv_ddl)
                
                # Create GIN index programmatically
                idx_name = f"idx_{self.table_name}_tsv"
                idx = Index(
                    idx_name,
                    text(f"tsv"),
                    postgresql_using='gin'
                )
                idx.create(bind=self.engine)
                
                session.commit()
                LOG.info(f"Added tsvector column and index to {self.table_name}")

    async def keyword_search(self, query: str, fields: list, top_k: int = 10, score_threshold: float = 0, **kwargs):
        """Perform keyword-based search using PostgreSQL full-text search capabilities."""
        raise NotImplementedError("Postgres keyword search not implemented yet")