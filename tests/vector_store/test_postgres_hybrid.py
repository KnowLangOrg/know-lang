import pytest
import unittest.mock as mock

from knowlang.search.base import SearchMethodology
from knowlang.configs import DBConfig, EmbeddingConfig
from knowlang.vector_stores.postgres_hybrid import PostgresHybridStore, VectorStoreInitError

class TestHybridStore:
    """Tests for the PostgresHybridStore implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock configs
        self.db_config : DBConfig = mock.MagicMock(spec=DBConfig)
        self.db_config.connection_url = "postgresql://user:pass@localhost:5432/testdb"
        self.db_config.collection_name = "test_collection"
        self.db_config.similarity_metric = "cosine"
        self.db_config.content_field = "content"
        
        self.embedding_config : EmbeddingConfig = mock.MagicMock(spec=EmbeddingConfig)
        self.embedding_config.dimension = 128
        
        # Patch vecs module
        self.vecs_patcher = mock.patch("knowlang.vector_stores.postgres.vecs")
        self.mock_vecs = self.vecs_patcher.start()
        
        # Create mock vecs client and collection
        self.mock_client = mock.MagicMock()
        self.mock_collection = mock.MagicMock()
        self.mock_vecs.create_client.return_value = self.mock_client
        self.mock_client.get_or_create_collection.return_value = self.mock_collection
        
        # Patch SQLAlchemy
        self.engine_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.create_engine")
        self.sessionmaker_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.sessionmaker")
        self.inspector_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.inspect")
        
        self.mock_create_engine = self.engine_patcher.start()
        self.mock_sessionmaker = self.sessionmaker_patcher.start()
        self.mock_inspect = self.inspector_patcher.start()
        
        # Set up mock engine and session
        self.mock_engine = mock.MagicMock()
        self.mock_session = mock.MagicMock()
        self.mock_session_class = mock.MagicMock()
        self.mock_create_engine.return_value = self.mock_engine
        self.mock_sessionmaker.return_value = self.mock_session_class
        
        # Set up mock inspector
        self.mock_inspector = mock.MagicMock()
        self.mock_inspect.return_value = self.mock_inspector
        
        # Configure mock session context manager
        self.mock_session_ctx = mock.MagicMock()
        self.mock_session_class.return_value.__enter__.return_value = self.mock_session_ctx
        
    def teardown_method(self):
        """Tear down test fixtures"""
        self.vecs_patcher.stop()
        self.engine_patcher.stop()
        self.sessionmaker_patcher.stop()
        self.inspector_patcher.stop()
    
    def test_create_from_config(self):
        """Test creating a hybrid store from config"""
        # Test successful creation
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        assert isinstance(store, PostgresHybridStore)
        assert store.connection_string == self.db_config.connection_url
        assert store.table_name == self.db_config.collection_name
        assert store.embedding_dim == self.embedding_config.dimension
        assert store.content_field == self.db_config.content_field
        
        # Test error when connection URL is missing
        self.db_config.connection_url = None
        with pytest.raises(VectorStoreInitError):
            PostgresHybridStore.create_from_config(
                config=self.db_config,
                embedding_config=self.embedding_config
            )
    
    def test_initialization(self):
        """Test initializing the hybrid store"""
        # Create store
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        # Configure mock inspector
        self.mock_inspector.has_table.return_value = True
        self.mock_inspector.get_columns.return_value = []  # No tsv column initially
        
        # Test initialization success
        store.initialize()
        
        # Verify vector store initialization
        self.mock_vecs.create_client.assert_called_once_with(self.db_config.connection_url)
        self.mock_client.get_or_create_collection.assert_called_once_with(
            name=self.db_config.collection_name,
            dimension=self.embedding_config.dimension
        )
        
        # Verify SQLAlchemy initialization
        self.mock_create_engine.assert_called_once_with(self.db_config.connection_url)
        self.mock_sessionmaker.assert_called_once_with(bind=self.mock_engine)
        
        # Verify tsv column addition
        self.mock_inspector.has_table.assert_called_once_with(self.db_config.collection_name)
        self.mock_inspector.get_columns.assert_called_once_with(self.db_config.collection_name)
        
        # Verify DDL execution for adding tsvector column
        assert self.mock_session_ctx.execute.called
        
        # Test initialization with existing tsv column
        self.mock_inspector.get_columns.return_value = [{'name': 'tsv'}]
        store.initialize()  # Should not try to add tsv column again
        
        # Verify table existence check
        self.mock_inspector.has_table.return_value = False
        with pytest.raises(VectorStoreInitError):
            store.initialize()
    
    def test_capabilities(self):
        """Test that hybrid store has both vector and keyword capabilities"""
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        # Check capabilities
        assert store.has_capability(SearchMethodology.VECTOR)
        assert store.has_capability(SearchMethodology.KEYWORD)
        
    def test_similarity_metrics(self):
        """Test different similarity metrics"""
        # Test cosine similarity
        self.db_config.similarity_metric = "cosine"
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        assert "cosine" in store.measure().__str__().lower()
        
        # Test L2 distance
        self.db_config.similarity_metric = "l2"
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        assert "l2" in store.measure().__str__().lower()
        
        # Test invalid metric
        self.db_config.similarity_metric = "invalid"
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        with pytest.raises(Exception):
            store.measure()