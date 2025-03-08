from typing import Literal, List, Optional
from sqlalchemy import Column, Index, MetaData, String, Table, Text, column, func, select, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.types import UserDefinedType
from sqlalchemy.schema import DDL
from sqlalchemy import inspect

from knowlang.vector_stores.factory import register_vector_store
from knowlang.vector_stores.base import VectorStoreError, VectorStoreInitError, SearchResult
from knowlang.vector_stores.postgres import PostgresVectorStore
from knowlang.search.keyword_search import KeywordSearchableStore
from knowlang.configs import DBConfig, EmbeddingConfig
from knowlang.utils import FancyLogger
from knowlang.core.types import VectorStoreProvider
from knowlang.search.base import SearchMethodology

LOG = FancyLogger(__name__)
Base = declarative_base()

# Define a custom type for pgvector, but we won't use it directly
# as we're relying on vecs for vector operations
class Vector(UserDefinedType):
    def get_col_spec(self, **kw):
        return "vector"

@register_vector_store(VectorStoreProvider.POSTGRES)
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
            content_field=config.content_field,
        )

    def __init__(
        self,
        connection_string: str,
        table_name: str,
        embedding_dim: int,
        similarity_metric: Literal['cosine'] = 'cosine',
        text_search_config: str = "english",
        content_field: str = "content",
        schema: str = "vecs"
    ):
        """Initialize the hybrid store with both vector and text search capabilities.
        
        Args:
            connection_string: PostgreSQL connection URL
            table_name: Name of the collection/table
            embedding_dim: Dimension of the vector embeddings
            similarity_metric: Vector similarity metric to use
            text_search_config: PostgreSQL text search configuration
            content_field: The metadata field containing text to be searched
            schema: The PostgreSQL schema where the tables are located (default: 'vecs')
        """
        # Initialize vector store capabilities with content_field
        super().__init__(
            connection_string=connection_string,
            table_name=table_name,
            embedding_dim=embedding_dim,
            similarity_metric=similarity_metric,
            content_field=content_field
        )
        
        # Initialize text search specific attributes
        self.text_search_config = text_search_config
        self.sqlalchemy_url = self.connection_string
        self.schema = schema
        self.engine = None
        self.Session = None 
        
        # Define metadata for direct SQL operations where ORM is not suitable
        self.metadata = None 
    
    def _setup_sqlalchemy(self):
        """Initialize SQLAlchemy engine and session"""
        if self.engine is None:
            try:
                self.engine = create_engine(self.connection_string)
                self.Session = sessionmaker(bind=self.engine)
                
                # Set up metadata for direct table operations
                self.metadata = MetaData(schema=self.schema)
                
                LOG.info(f"SQLAlchemy engine initialized for {self.schema}.{self.table_name}")
            except Exception as e:
                raise VectorStoreInitError(f"Failed to initialize SQLAlchemy: {str(e)}") from e

    def initialize(self):
        """Initialize both vector store and text search capabilities."""
        super().initialize()
        self._setup_sqlalchemy()

        try:
            with self.Session() as session:
                # Use proper SQLAlchemy inspection to check if table exists
                inspector = inspect(self.engine)
                
                if not inspector.has_table(self.table_name, schema=self.schema):
                    raise VectorStoreInitError(
                        f"Table {self.schema}.{self.table_name} does not exist. vecs should create it."
                    )
                
                # Get actual columns from the database - specify schema
                columns = inspector.get_columns(self.table_name, schema=self.schema)
                column_names = [col['name'] for col in columns]
                
                if 'tsv' not in column_names:
                    LOG.info(f"Adding tsv column to {self.schema}.{self.table_name}")
                    
                    # Add tsvector column for text search - fully qualify table name with schema
                    tsv_ddl = DDL(
                        f"ALTER TABLE {self.schema}.{self.table_name} "
                        f"ADD COLUMN tsv tsvector GENERATED ALWAYS AS "
                        f"(to_tsvector('{self.text_search_config}', "
                        f"COALESCE((metadata->>'{self.content_field}')::text, ''))) STORED"
                    )
                    session.execute(tsv_ddl)
                    session.commit()
                    
                    # Create GIN index programmatically - specify schema
                    idx_name = f"idx_{self.table_name}_tsv"
                    # Note: Index creation needs schema specified in the Table object
                    table = Table(
                        self.table_name, 
                        self.metadata,
                        Column('tsv', TSVECTOR),
                        schema=self.schema
                    )
                    idx = Index(
                        idx_name,
                        table.c.tsv,
                        postgresql_using='gin'
                    )
                    idx.create(bind=self.engine)
                    
                    session.commit()
                    LOG.info(f"Added tsvector column and index to {self.schema}.{self.table_name}")
                else:
                    LOG.info(f"tsv column already exists in {self.schema}.{self.table_name}")
        except Exception as e:
            if isinstance(e, VectorStoreInitError):
                raise 
            LOG.error(f"Error initializing tsvector column: {e}")
            raise VectorStoreInitError(f"Failed to initialize tsvector column: {str(e)}") from e

    def has_capability(self, methodology: SearchMethodology) -> bool:
        """Check if this store supports a specific search methodology."""
        if methodology == SearchMethodology.VECTOR:
            return True
        if methodology == SearchMethodology.KEYWORD:
            return True
        return False

    async def keyword_search(
        self, 
        query: str, 
        fields: List[str], 
        top_k: int = 10, 
        score_threshold: float = 0, 
        **kwargs
    ) -> List[SearchResult]:
        """Perform keyword-based search using PostgreSQL full-text search capabilities.
        
        Args:
            query: The search query
            fields: List of metadata fields to search (currently only supports content_field)
            top_k: Maximum number of results to return
            score_threshold: Minimum relevance score threshold (0.0 to 1.0)
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        self.assert_initialized()
        
        try:
            with self.Session() as session:
                # Convert the query to a tsquery expression
                tsquery = func.plainto_tsquery(self.text_search_config, query)
                
                # Define the table structure for querying - include schema
                table = Table(
                    self.table_name,
                    MetaData(schema=self.schema),
                    Column('id', String, primary_key=True),
                    Column('metadata', JSONB),
                    Column('tsv', TSVECTOR),
                    schema=self.schema,
                    extend_existing=True
                )
                
                # Build the SQL query
                sql_query = select(
                    table.c.id,
                    table.c.metadata,
                    func.ts_rank(table.c.tsv, tsquery).label('rank')
                ).where(
                    table.c.tsv.op('@@')(tsquery)
                ).order_by(
                    column('rank').desc()
                ).limit(top_k)
                
                # Execute the query
                results = session.execute(sql_query).fetchall()
                
                # Format results into SearchResult objects
                search_results = []
                for id, metadata, rank in results:
                    if rank >= score_threshold:
                        search_results.append(SearchResult(
                            document=id,
                            metadata=metadata,
                            score=float(rank)
                        ))
                
                return search_results
                
        except Exception as e:
            LOG.error(f"Keyword search failed: {str(e)}")
            raise VectorStoreError(f"Keyword search failed: {str(e)}") from e