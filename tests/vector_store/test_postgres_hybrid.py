import pytest
import unittest.mock as mock
from sqlalchemy import Column, MetaData, Table, String
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.configs import DBConfig, EmbeddingConfig
from knowlang.vector_stores.postgres import PostgresVectorStore
from knowlang.vector_stores.postgres_hybrid import PostgresHybridStore, VectorStoreInitError, Vector

class TestPostgresHybridStore:
    """Tests for the PostgresHybridStore implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self._setup_configs()
        self._setup_mocks()
        self._setup_table_metadata()
    
    def _setup_configs(self):
        """Set up configuration mocks"""
        self.db_config = mock.MagicMock(spec=DBConfig)
        self.db_config.connection_url = "postgresql://user:pass@localhost:5432/testdb"
        self.db_config.collection_name = "test_collection"
        self.db_config.similarity_metric = "cosine"
        self.db_config.content_field = "content"
        
        self.embedding_config = mock.MagicMock(spec=EmbeddingConfig)
        self.embedding_config.dimension = 128
    
    def _setup_mocks(self):
        """Set up all necessary mocks"""
        # Patch and setup vecs
        self.vecs_patcher = mock.patch("knowlang.vector_stores.postgres.vecs")
        self.mock_vecs = self.vecs_patcher.start()
        self.mock_client = mock.MagicMock()
        self.mock_collection = mock.MagicMock()
        self.mock_vecs.create_client.return_value = self.mock_client
        self.mock_client.get_or_create_collection.return_value = self.mock_collection
        
        # Patch SQLAlchemy components
        self.engine_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.create_engine")
        self.sessionmaker_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.sessionmaker")
        self.metadata_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.MetaData")
        self.table_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.Table")
        
        self.mock_create_engine = self.engine_patcher.start()
        self.mock_sessionmaker = self.sessionmaker_patcher.start()
        self.mock_metadata = self.metadata_patcher.start()
        self.mock_table = self.table_patcher.start()
        
        # Set up mock engine and session
        self.mock_engine = mock.MagicMock()
        self.mock_session = mock.MagicMock()
        self.mock_session_class = mock.MagicMock()
        self.mock_create_engine.return_value = self.mock_engine
        self.mock_sessionmaker.return_value = self.mock_session_class
        
        # Configure session context manager
        self.mock_session_ctx = mock.MagicMock()
        self.mock_session_class.return_value.__enter__.return_value = self.mock_session_ctx
    
    def _setup_table_metadata(self):
        """Set up table metadata and columns"""
        self.mock_metadata_instance = mock.MagicMock()
        self.mock_metadata.return_value = self.mock_metadata_instance
        
        self.mock_table_instance = mock.MagicMock()
        self.mock_table.return_value = self.mock_table_instance
        
        # Set up default columns
        self.mock_table_instance.columns = [
            mock.MagicMock(name='id'),
            mock.MagicMock(name='vec')
        ]
        for col in self.mock_table_instance.columns:
            col.name = col._mock_name
            
        self.mock_table_instance.metadata = self.mock_metadata_instance
        self.mock_metadata_instance.tables = {"test_collection": self.mock_table_instance}
    
    def teardown_method(self):
        """Tear down test fixtures"""
        self.vecs_patcher.stop()
        self.engine_patcher.stop()
        self.sessionmaker_patcher.stop()
        self.metadata_patcher.stop()
        self.table_patcher.stop()
    
    def _create_store(self) -> PostgresHybridStore:
        """Helper method to create a store instance"""
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        store.collection = self.mock_collection
        store.engine = self.mock_engine
        store.Session = self.mock_session_class

        return store

    
    def _setup_initialized_store(self) -> PostgresHybridStore:
        """Helper method to create and setup an initialized store"""
        store = self._create_store()
        
        # Create a proper mock table
        metadata = MetaData()
        store.vecs_table = Table(
            'test_collection', metadata,
            Column('id', String, primary_key=True),
            Column('metadata', JSONB),
            Column('embedding', Vector),
            Column('tsv', TSVECTOR)
        )
        return store
    
    def _setup_mock_search_results(self, mock_session):
        """Helper method to setup mock search results"""
        mock_result = mock.MagicMock()
        mock_results = [
            ("doc1", {"content": "sample document"}, 0.85),
            ("doc2", {"content": "another document"}, 0.65),
            ("doc3", {"content": "less relevant document"}, 0.45)
        ]
        mock_result.fetchall.return_value = mock_results
        mock_session.execute.return_value = mock_result
        return mock_results
    
    def test_create_from_config(self):
        """Test creating a hybrid store from config"""
        store = self._create_store()
        assert isinstance(store, PostgresHybridStore)
        assert store.connection_string == self.db_config.connection_url
        assert store.table_name == self.db_config.collection_name
        assert store.embedding_dim == self.embedding_config.dimension
        assert store.content_field == self.db_config.content_field
        
        # Test error when connection URL is missing
        self.db_config.connection_url = None
        with pytest.raises(VectorStoreInitError):
            self._create_store()
    
    def test_capabilities(self):
        """Test that hybrid store has both vector and keyword capabilities"""
        store = self._create_store()
        assert store.has_capability(SearchMethodology.VECTOR)
        assert store.has_capability(SearchMethodology.KEYWORD)
        assert not store.has_capability("invalid_methodology")
    
    def test_initialize_with_tsv_column(self):
        """Test initialization with tsv column already present"""
        store = self._create_store()
        
        # Add TSV column to existing columns
        tsv_mock_col = mock.MagicMock(name='tsv')
        tsv_mock_col.name = 'tsv'
        self.mock_table_instance.columns += [tsv_mock_col]
        
        store._setup_sqlalchemy = mock.MagicMock(side_effect=lambda: setattr(store, 'vecs_table', self.mock_table_instance))
        
        with mock.patch.object(PostgresVectorStore, 'initialize') as mock_super_init:
            store.initialize()
            mock_super_init.assert_called_once()
            store._setup_sqlalchemy.assert_called_once()
            
            # Verify no attempt to add TSV column
            add_column_calls = [call for call in self.mock_session_ctx.execute.call_args_list 
                              if call and "ADD COLUMN tsv" in str(call)]
            assert not add_column_calls
    
    def test_initialize_add_tsv_column(self):
        """Test initialization with adding tsv column"""
        store = self._create_store()
        store._setup_sqlalchemy = mock.MagicMock(side_effect=lambda: setattr(store, 'vecs_table', self.mock_table_instance))
        
        store.initialize()
        
        store._setup_sqlalchemy.assert_called_once()
        self.mock_session_ctx.execute.assert_called()
        add_column_calls = [call for call in self.mock_session_ctx.mock_calls 
                           if "ADD COLUMN tsv" in str(call)]
        assert add_column_calls
    
    def test_table_not_found(self):
        """Test error when table doesn't exist"""
        self.mock_metadata_instance.tables = {}
        store = self._create_store()
        store._setup_sqlalchemy = mock.MagicMock(side_effect=lambda: setattr(store, 'vecs_table', self.mock_table_instance))
        
        with pytest.raises(VectorStoreInitError):
            store.initialize()
    
    @pytest.mark.asyncio
    async def test_keyword_search(self):
        """Test keyword search functionality"""
        store = self._setup_initialized_store()
        mock_session = mock.MagicMock()
        self.mock_session_class.return_value.__enter__.return_value = mock_session
        
        mock_results = self._setup_mock_search_results(mock_session)
        
        with mock.patch.object(store, 'assert_initialized'):
            results = await store.keyword_search("sample query", ["content"], top_k=3)
            
            mock_session.execute.assert_called_once()
            assert len(results) == 3
            for i, (doc_id, metadata, score) in enumerate(mock_results):
                assert results[i].document == doc_id
                assert results[i].score == score
                assert results[i].metadata == metadata
    
    @pytest.mark.asyncio
    async def test_keyword_search_with_threshold(self):
        """Test keyword search with score threshold"""
        store = self._setup_initialized_store()
        mock_session = mock.MagicMock()
        self.mock_session_class.return_value.__enter__.return_value = mock_session
        
        self._setup_mock_search_results(mock_session)
        
        with mock.patch.object(store, 'assert_initialized'):
            results = await store.keyword_search("sample query", ["content"], top_k=3, score_threshold=0.5)
            
            mock_session.execute.assert_called_once()
            assert len(results) == 2  # Only documents with score >= 0.5
            assert all(result.score >= 0.5 for result in results)
    
    def test_similarity_metrics(self):
        """Test different similarity metrics"""
        metrics = {
            "cosine": lambda x: "cosine" in x.lower(),
            "l2": lambda x: "l2" in x.lower()
        }
        
        for metric, check_func in metrics.items():
            self.db_config.similarity_metric = metric
            store = self._create_store()
            assert check_func(store.measure().__str__())
        
        # Test invalid metric
        self.db_config.similarity_metric = "invalid"
        store = self._create_store()
        with pytest.raises(Exception):
            store.measure()