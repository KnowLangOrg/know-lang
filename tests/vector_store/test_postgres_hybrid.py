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
        # Mock configs
        self.db_config = mock.MagicMock(spec=DBConfig)
        self.db_config.connection_url = "postgresql://user:pass@localhost:5432/testdb"
        self.db_config.collection_name = "test_collection"
        self.db_config.similarity_metric = "cosine"
        self.db_config.content_field = "content"
        
        self.embedding_config = mock.MagicMock(spec=EmbeddingConfig)
        self.embedding_config.dimension = 128
        
        # Patch vecs module
        self.vecs_patcher = mock.patch("knowlang.vector_stores.postgres.vecs")
        self.mock_vecs = self.vecs_patcher.start()
        
        # Create mock vecs client and collection
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
        
        # Set up mock metadata and table
        self.mock_metadata_instance = mock.MagicMock()
        self.mock_metadata.return_value = self.mock_metadata_instance
        
        self.mock_table_instance = mock.MagicMock()
        self.mock_table.return_value = self.mock_table_instance
        
        # Configure mock session context manager
        self.mock_session_ctx = mock.MagicMock()
        self.mock_session_class.return_value.__enter__.return_value = self.mock_session_ctx
        
        # Mock table metadata with default columns (no tsv)
        self.mock_metadata_instance.tables = {"test_collection": self.mock_table_instance}
        self.mock_table_instance.columns = [mock.MagicMock(name='id'), mock.MagicMock(name='vec')]
        self.mock_table_instance.metadata = self.mock_metadata_instance
        
        # Set up column names for checks
        for col in self.mock_table_instance.columns:
            col.name = col._mock_name
        
    def teardown_method(self):
        """Tear down test fixtures"""
        self.vecs_patcher.stop()
        self.engine_patcher.stop()
        self.sessionmaker_patcher.stop()
        self.metadata_patcher.stop()
        self.table_patcher.stop()
    
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
    
    def test_capabilities(self):
        """Test that hybrid store has both vector and keyword capabilities"""
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        # Check capabilities
        assert store.has_capability(SearchMethodology.VECTOR)
        assert store.has_capability(SearchMethodology.KEYWORD)
        assert not store.has_capability("invalid_methodology")
    
    def test_initialize_with_tsv_column(self):
        """Test initialization with tsv column already present"""
        # Create and initialize store
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        
        # Mock the _setup_sqlalchemy method to set up attributes properly
        store._setup_sqlalchemy = mock.MagicMock()
        def setup_mock():
            store.engine = self.mock_engine
            store.Session = self.mock_session_class
            store.metadata = self.mock_metadata_instance
            store.vecs_table = self.mock_table_instance
            # Ensure the table_name exists in vecs_table.metadata.tables
            # self.mock_metadata_instance.tables = {store.table_name: self.mock_table_instance}

            tsv_mock_col = mock.MagicMock(name='tsv')
            tsv_mock_col.name = tsv_mock_col._mock_name
            self.mock_table_instance.columns += [tsv_mock_col]

        store._setup_sqlalchemy.side_effect = setup_mock
        
        # Mock vecs store initialization to avoid real calls
        with mock.patch.object(PostgresVectorStore, 'initialize') as mock_super_init:
            # Initialize the store
            store.initialize()
            
            # Verify super().initialize() was called
            mock_super_init.assert_called_once()
            
            # Verify setup_sqlalchemy was called
            store._setup_sqlalchemy.assert_called_once()
            
            # Verify it didn't try to add the tsv column since it already exists
            add_column_called = False
            for call in self.mock_session_ctx.execute.call_args_list:
                if call and "ADD COLUMN tsv" in str(call):
                    add_column_called = True
                    break
            assert not add_column_called

    def test_initialize_add_tsv_column(self):
        """Test initialization with adding tsv column"""
        
        # Create and initialize store
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        # Mock the _setup_sqlalchemy method to set the attributes directly
        store._setup_sqlalchemy = mock.MagicMock()
        def setup_mock():
            store.engine = self.mock_engine
            store.Session = self.mock_session_class
            store.metadata = self.mock_metadata_instance
            store.vecs_table = self.mock_table_instance
        store._setup_sqlalchemy.side_effect = setup_mock
        
        # Initialize the store
        store.initialize()
        
        # Verify SQLAlchemy was set up
        store._setup_sqlalchemy.assert_called_once()
        
        # Verify it tried to add the tsv column
        self.mock_session_ctx.execute.assert_called()
        add_column_called = False
        for call in self.mock_session_ctx.mock_calls:
            if "ADD COLUMN tsv" in str(call):
                add_column_called = True
                break
        assert add_column_called
    
    def test_table_not_found(self):
        """Test error when table doesn't exist"""
        # Set up metadata with no tables
        self.mock_metadata_instance.tables = {}
        
        # Create and initialize store
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        # Mock the _setup_sqlalchemy method to set the attributes directly
        store._setup_sqlalchemy = mock.MagicMock()
        def setup_mock():
            store.engine = self.mock_engine
            store.Session = self.mock_session_class
            store.metadata = self.mock_metadata_instance
            store.vecs_table = self.mock_table_instance
        store._setup_sqlalchemy.side_effect = setup_mock
        
        # Initialize should fail because table doesn't exist
        with pytest.raises(VectorStoreInitError):
            store.initialize()
    
    @pytest.mark.asyncio
    async def test_keyword_search(self):
        """Test keyword search functionality"""
        # Create and initialize store
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        # Set up initialized state manually (avoiding initialize call)
        store.collection = self.mock_collection
        store.engine = self.mock_engine
        store.Session = self.mock_session_class
        
        # Create a proper mock table with SQLAlchemy Column objects
        metadata = MetaData()
        store.vecs_table = Table(
            'test_collection', metadata,
            Column('id', String, primary_key=True),
            Column('metadata', JSONB),
            Column('embedding', Vector),
            Column('tsv', TSVECTOR)
        )
        
        # Mock the assert_initialized method to avoid initialization checks
        with mock.patch.object(store, 'assert_initialized'):
            # Mock the Session context manager
            mock_session = mock.MagicMock()
            mock_result = mock.MagicMock()
            
            # Set up mock results
            mock_results = [
                ("doc1", {"content": "sample document"}, 0.85),
                ("doc2", {"content": "another document"}, 0.65),
                ("doc3", {"content": "less relevant document"}, 0.45)
            ]
            
            # Configure the mock chain
            mock_result.fetchall.return_value = mock_results
            mock_session.execute.return_value = mock_result
            self.mock_session_class.return_value.__enter__.return_value = mock_session
            
            # Test keyword search
            results = await store.keyword_search("sample query", ["content"], top_k=3)
            
            # Verify session was created and execute was called
            mock_session.execute.assert_called_once()
            
            # Verify results formatting
            assert len(results) == 3
            assert results[0].document == "doc1"
            assert results[0].score == 0.85
            assert results[0].metadata == {"content": "sample document"}
            assert results[1].document == "doc2"
            assert results[1].score == 0.65
            assert results[2].document == "doc3"
            assert results[2].score == 0.45
    
    @pytest.mark.asyncio
    async def test_keyword_search_with_threshold(self):
        """Test keyword search with score threshold"""
        # Create and initialize store
        store = PostgresHybridStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        # Set up initialized state manually (avoiding initialize call)
        store.collection = self.mock_collection
        store.engine = self.mock_engine
        store.Session = self.mock_session_class
        
        # Create a proper mock table with SQLAlchemy Column objects
        metadata = MetaData()
        store.vecs_table = Table(
            'test_collection', metadata,
            Column('id', String, primary_key=True),
            Column('metadata', JSONB),
            Column('embedding', Vector),
            Column('tsv', TSVECTOR)
        )
        
        # Mock the assert_initialized method to avoid initialization checks
        with mock.patch.object(store, 'assert_initialized'):
            # Mock the Session context manager
            mock_session = mock.MagicMock()
            mock_result = mock.MagicMock()
            
            # Mock results for keyword search
            mock_results = [
                ("doc1", {"content": "sample document"}, 0.85),
                ("doc2", {"content": "another document"}, 0.65),
                ("doc3", {"content": "less relevant document"}, 0.45)
            ]
            
            # Configure the mock chain
            mock_result.fetchall.return_value = mock_results
            mock_session.execute.return_value = mock_result
            self.mock_session_class.return_value.__enter__.return_value = mock_session
            
            # Test keyword search with threshold
            results = await store.keyword_search("sample query", ["content"], top_k=3, score_threshold=0.5)
            
            # Verify session was created and execute was called
            mock_session.execute.assert_called_once()
            
            # Verify results are filtered by threshold
            assert len(results) == 2  # Only documents with score >= 0.5
            assert results[0].document == "doc1"
            assert results[0].score == 0.85
            assert results[1].document == "doc2"
            assert results[1].score == 0.65
            assert all(result.score >= 0.5 for result in results)
    
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