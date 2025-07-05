import unittest.mock as mock

import pytest
from sqlalchemy import Column, MetaData, String, Table, inspect
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

from knowlang.configs import AppConfig, DBConfig, EmbeddingConfig
from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.vector_stores.postgres import PostgresVectorStore
from knowlang.vector_stores.postgres_hybrid import (PostgresHybridStore,
                                                    Vector,
                                                    VectorStoreInitError)


class TestPostgresHybridStore:
    """Tests for the PostgresHybridStore implementation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self._setup_configs()
        self._setup_mocks()
    
    def _setup_configs(self):
        """Set up configuration mocks"""
        self.db_config = mock.MagicMock(spec=DBConfig)
        self.db_config.connection_url = "postgresql://user:pass@localhost:5432/testdb"
        self.db_config.collection_name = "test_collection"
        self.db_config.similarity_metric = "cosine"
        self.db_config.content_field = "content"
        self.db_config.schema = "vecs"  # Specify schema here
        
        self.embedding_config = mock.MagicMock(spec=EmbeddingConfig)
        self.embedding_config.dimension = 128
        self.app_config = mock.MagicMock(spec=AppConfig)
        self.app_config.db = self.db_config
        self.app_config.embedding = self.embedding_config
    
    def _setup_mocks(self):
        """Set up all necessary mocks"""
        # Store the original __import__ function before patching
        self.original_import = __import__
        
        # Mock the vecs module import in PostgresVectorStore
        self.vecs_import_patcher = mock.patch("builtins.__import__")
        self.mock_import = self.vecs_import_patcher.start()
        
        # Create mock vecs module
        self.mock_vecs = mock.MagicMock()
        self.mock_vecs.IndexMeasure.cosine_distance = "cosine_distance"
        self.mock_vecs.IndexMeasure.l1_distance = "l1_distance"
        self.mock_vecs.IndexMeasure.l2_distance = "l2_distance"
        self.mock_vecs.IndexMeasure.max_inner_product = "max_inner_product"
        
        # Create mock vecs client and collection
        self.mock_client = mock.MagicMock()
        self.mock_collection = mock.MagicMock()
        self.mock_vecs.create_client.return_value = self.mock_client
        self.mock_client.get_or_create_collection.return_value = self.mock_collection
        
        # Configure the import mock to return our mock vecs module
        def mock_import_side_effect(name, *args, **kwargs):
            if name == "vecs":
                return self.mock_vecs
            # For other imports, use the original import function
            return self.original_import(name, *args, **kwargs)
        
        self.mock_import.side_effect = mock_import_side_effect
        
        # Patch SQLAlchemy components
        self.engine_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.create_engine")
        self.sessionmaker_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.sessionmaker")
        self.metadata_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.MetaData")
        self.inspect_patcher = mock.patch("knowlang.vector_stores.postgres_hybrid.inspect")
        
        self.mock_create_engine = self.engine_patcher.start()
        self.mock_sessionmaker = self.sessionmaker_patcher.start()
        self.mock_metadata = self.metadata_patcher.start()
        self.mock_inspect = self.inspect_patcher.start()
        
        # Set up mock engine and session
        self.mock_engine = mock.MagicMock()
        self.mock_session = mock.MagicMock()
        self.mock_session_class = mock.MagicMock()
        self.mock_create_engine.return_value = self.mock_engine
        self.mock_sessionmaker.return_value = self.mock_session_class
        
        # Configure session context manager
        self.mock_session_ctx = mock.MagicMock()
        self.mock_session_class.return_value.__enter__.return_value = self.mock_session_ctx
        
        # Setup inspector
        self.mock_inspector = mock.MagicMock()
        self.mock_inspect.return_value = self.mock_inspector
        
        # Default: table exists but no tsv column
        self.mock_inspector.has_table.return_value = True
        self.mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': String()}, 
            {'name': 'metadata', 'type': JSONB()},
            {'name': 'embedding', 'type': Vector()}
        ]
    
    def teardown_method(self):
        """Tear down test fixtures"""
        self.vecs_import_patcher.stop()
        self.engine_patcher.stop()
        self.sessionmaker_patcher.stop()
        self.metadata_patcher.stop()
        self.inspect_patcher.stop()
    
    def _create_store(self) -> PostgresHybridStore:
        """Helper method to create a store instance"""
        store = PostgresHybridStore.create_from_config(self.app_config)
        store.collection = self.mock_collection
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
        assert store.schema == self.db_config.schema  # Verify schema is set
        
        # Test error when connection URL is missing
        self.db_config.connection_url = None
        with pytest.raises(VectorStoreInitError):
            self._create_store()
    
    def test_schema_default(self):
        """Test schema defaults to 'vecs' when not specified"""
        self.db_config.schema = None  # Remove schema from config
        store = self._create_store()
        assert store.schema == "vecs"  # Should default to 'vecs'
    
    def test_capabilities(self):
        """Test that hybrid store has both vector and keyword capabilities"""
        store = self._create_store()
        assert store.has_capability(SearchMethodology.VECTOR)
        assert store.has_capability(SearchMethodology.KEYWORD)
        assert not store.has_capability("invalid_methodology")
    
    @pytest.mark.asyncio
    async def test_initialize_with_tsv_column(self):
        """Test initialization when tsv column already exists"""
        # Set up mock to indicate tsv column exists
        self.mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': String()}, 
            {'name': 'metadata', 'type': JSONB()},
            {'name': 'embedding', 'type': Vector()},
            {'name': 'tsv', 'type': TSVECTOR()}
        ]
        
        store = self._create_store()
        
        with mock.patch.object(PostgresVectorStore, 'initialize') as mock_super_init:
            await store.initialize()
            mock_super_init.assert_called_once()
            
            # Verify schema is passed to has_table and get_columns
            self.mock_inspector.has_table.assert_called_with(
                store.table_name, schema=store.schema
            )
            self.mock_inspector.get_columns.assert_called_with(
                store.table_name, schema=store.schema
            )
            
            # Verify no attempt to add TSV column
            self.mock_session_ctx.execute.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_initialize_add_tsv_column(self):
        """Test initialization when tsv column needs to be added"""
        # Default setup: tsv column doesn't exist
        
        store = self._create_store()
        
        with mock.patch.object(PostgresVectorStore, 'initialize') as mock_super_init:
            await store.initialize()
            mock_super_init.assert_called_once()
            
            # Verify schema is passed to has_table and get_columns
            self.mock_inspector.has_table.assert_called_with(
                store.table_name, schema=store.schema
            )
            self.mock_inspector.get_columns.assert_called_with(
                store.table_name, schema=store.schema
            )
            
            # Verify attempt to add TSV column
            self.mock_session_ctx.execute.assert_called()
            # Check that the ALTER TABLE command includes the schema
            execute_args = self.mock_session_ctx.execute.call_args[0][0]
            assert f"ALTER TABLE {store.schema}.{store.table_name}" in str(execute_args)
    
    @pytest.mark.asyncio
    async def test_table_not_found(self):
        """Test error when table doesn't exist"""
        # Set up mock to indicate table doesn't exist
        self.mock_inspector.has_table.return_value = False
        
        store = self._create_store()
        
        with pytest.raises(VectorStoreInitError) as excinfo:
            await store.initialize()
        
        # Verify error message contains schema
        assert store.schema in str(excinfo.value)
        assert store.table_name in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_keyword_search(self):
        """Test keyword search functionality"""
        store = self._create_store()
        # Initialize store to setup necessary attributes
        with mock.patch.object(PostgresVectorStore, 'initialize'):
            await store.initialize()
        
        mock_session = mock.MagicMock()
        self.mock_session_class.return_value.__enter__.return_value = mock_session
        
        mock_results = self._setup_mock_search_results(mock_session)
        
        with mock.patch.object(store, 'assert_initialized'):
            results = await store.keyword_search("sample query", ["content"], top_k=3)
            
            mock_session.execute.assert_called_once()
            assert len(results) == 3
            for i, (doc_id, metadata, score) in enumerate(mock_results):
                assert results[i].document == metadata["content"]
                assert results[i].score == score
                assert results[i].metadata == metadata
    
    @pytest.mark.asyncio
    async def test_keyword_search_with_threshold(self):
        """Test keyword search with score threshold"""
        store = self._create_store()
        # Initialize store to setup necessary attributes
        with mock.patch.object(PostgresVectorStore, 'initialize'):
            await store.initialize()

        mock_session = mock.MagicMock()
        self.mock_session_class.return_value.__enter__.return_value = mock_session
        
        self._setup_mock_search_results(mock_session)
        
        with mock.patch.object(store, 'assert_initialized'):
            results = await store.keyword_search("sample query", ["content"], top_k=3, score_threshold=0.5)
            
            mock_session.execute.assert_called_once()
            assert len(results) == 2  # Only documents with score >= 0.5
            assert all(result.score >= 0.5 for result in results)
    
    @pytest.mark.asyncio
    async def test_metadata_schema_specification(self):
        """Test that MetaData is created with schema specified"""
        store = self._create_store()
        
        with mock.patch.object(PostgresVectorStore, 'initialize'):
            await store.initialize()

            # Verify MetaData created with schema
            self.mock_metadata.assert_called_with(schema=store.schema)