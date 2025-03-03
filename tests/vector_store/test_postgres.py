import pytest
import unittest.mock as mock
from typing import List, Dict, Any

from knowlang.search import SearchResult
from knowlang.configs import DBConfig, EmbeddingConfig
from knowlang.vector_stores.base import VectorStore, VectorStoreError, VectorStoreInitError
from knowlang.vector_stores.postgres import PostgresVectorStore


class TestPostgresVectorStore:
    """Tests for the PostgresVectorStore implementation"""

    def setup_method(self):
        """Set up test fixtures"""
        # Mock configs
        self.db_config = mock.MagicMock(spec=DBConfig)
        self.db_config.connection_url = "postgresql://user:pass@localhost:5432/testdb"
        self.db_config.collection_name = "test_collection"
        self.db_config.similarity_metric = "cosine"
        
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
        
    def teardown_method(self):
        """Tear down test fixtures"""
        self.vecs_patcher.stop()
    
    def test_create_from_config(self):
        """Test creating a vector store from config"""
        # Test successful creation
        store = PostgresVectorStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        assert isinstance(store, PostgresVectorStore)
        assert store.connection_string == self.db_config.connection_url
        assert store.table_name == self.db_config.collection_name
        assert store.embedding_dim == self.embedding_config.dimension
        
        # Test error when connection URL is missing
        self.db_config.connection_url = None
        with pytest.raises(VectorStoreInitError):
            PostgresVectorStore.create_from_config(
                config=self.db_config,
                embedding_config=self.embedding_config
            )
    
    def test_initialization(self):
        """Test initializing the vector store"""
        # Create store
        store = PostgresVectorStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        # Test initialization success
        store.initialize()
        self.mock_vecs.create_client.assert_called_once_with(self.db_config.connection_url)
        self.mock_client.get_or_create_collection.assert_called_once_with(
            name=self.db_config.collection_name,
            dimension=self.embedding_config.dimension
        )
        self.mock_collection.create_index.assert_called_once()
        
        # Test initialization error
        self.mock_vecs.create_client.side_effect = Exception("Connection error")
        with pytest.raises(VectorStoreInitError):
            store.initialize()
    
    def test_assert_initialized(self):
        """Test initialization check"""
        store = PostgresVectorStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        
        # Test error when not initialized
        with pytest.raises(VectorStoreError):
            store.assert_initialized()
        
        # Test success when initialized
        store.collection = self.mock_collection
        store.assert_initialized()  # Should not raise exception
    
    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding documents to the store"""
        # Create and initialize store
        store = PostgresVectorStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        store.collection = self.mock_collection
        
        # Test data
        documents = ["doc1", "doc2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"field": "value1"}, {"field": "value2"}]
        ids = ["id1", "id2"]
        
        # Test adding documents
        await store.add_documents(documents, embeddings, metadatas, ids)
        self.mock_collection.upsert.assert_called_once()
        
        # Expected vectors format
        expected_vectors = [
            ("id1", [0.1, 0.2], {"field": "value1"}),
            ("id2", [0.3, 0.4], {"field": "value2"})
        ]
        # Verify upsert call arguments match expected
        args, kwargs = self.mock_collection.upsert.call_args
        if "records" in kwargs:
            assert kwargs["records"] == expected_vectors
        elif len(args) > 0:
            assert args[0] == expected_vectors
        else:
            assert False, "No arguments found in upsert call"
        
        # Test error when documents and embeddings have different lengths
        with pytest.raises(VectorStoreError):
            await store.add_documents(documents, embeddings[:1], metadatas, ids)
        
        # Test auto-generating IDs when not provided
        await store.add_documents(documents, embeddings, metadatas)
        # Second call to upsert
        calls = self.mock_collection.upsert.call_args_list
        assert len(calls) == 2
    
    @pytest.mark.asyncio
    async def test_query(self):
        """Test querying the vector store"""
        # Create and initialize store
        store = PostgresVectorStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        store.collection = self.mock_collection
        
        # Set up mock return value for query
        mock_results = [
            ("id1", 0.2, {"field": "value1"}),
            ("id2", 0.5, {"field": "value2"})
        ]
        self.mock_collection.query.return_value = mock_results
        
        # Test query
        query_embedding = [0.1, 0.2, 0.3]
        results = await store.query(query_embedding, top_k=2)
        
        # Verify query was called with correct args
        self.mock_collection.query.assert_called_once()
        args, kwargs = self.mock_collection.query.call_args
        assert kwargs["data"] == query_embedding
        assert kwargs["limit"] == 2
        assert kwargs["include_value"] is True
        assert kwargs["include_metadata"] is True
        
        # Verify results
        assert results == mock_results
    
    @pytest.mark.asyncio
    async def test_vector_search(self):
        """Test vector search with score threshold"""
        # Create and initialize store
        store = PostgresVectorStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        store.collection = self.mock_collection
        
        # Set up mock return value for query
        mock_results = [
            ("id1", 0.2, {"field": "value1"}),  # Distance 0.2 -> similarity 0.8
            ("id2", 0.5, {"field": "value2"}),  # Distance 0.5 -> similarity 0.5
            ("id3", 0.7, {"field": "value3"})   # Distance 0.7 -> similarity 0.3
        ]
        self.mock_collection.query.return_value = mock_results
        
        # Test vector search with no threshold
        query_embedding = [0.1, 0.2, 0.3]
        results = await store.vector_search(query_embedding, top_k=3)
        
        # Verify all results are included with correct scores
        assert len(results) == 3
        assert results[0].document == "id1"
        assert results[0].score == pytest.approx(0.8)
        assert results[1].document == "id2"
        assert results[1].score == pytest.approx(0.5)
        assert results[2].document == "id3"
        assert results[2].score == pytest.approx(0.3)
        
        # Test vector search with threshold
        results = await store.vector_search(query_embedding, top_k=3, score_threshold=0.6)
        
        # Verify only results above threshold are included
        assert len(results) == 1
        assert results[0].document == "id1"
        assert results[0].score == 0.8
    
    @pytest.mark.asyncio
    async def test_document_operations(self):
        """Test document operations (update, delete, get)"""
        # Create and initialize store
        store = PostgresVectorStore.create_from_config(
            config=self.db_config,
            embedding_config=self.embedding_config
        )
        store.collection = self.mock_collection
        
        # Test update_document
        await store.update_document("id1", "doc1", [0.1, 0.2], {"field": "new_value"})
        self.mock_collection.upsert.assert_called_once_with([("id1", [0.1, 0.2], {"field": "new_value"})])
        
        # Test delete
        await store.delete(["id1", "id2"])
        self.mock_collection.delete.assert_called_once_with(["id1", "id2"])
        
        # Test get_document
        self.mock_collection.fetch.return_value = [("id1", [0.1, 0.2], {"field": "value"})]
        doc = await store.get_document("id1")
        self.mock_collection.fetch.assert_called_once_with(ids=["id1"])
        assert doc == ("id1", [0.1, 0.2], {"field": "value"})
        
        # Test get_document when document doesn't exist
        self.mock_collection.fetch.return_value = []
        self.mock_collection.fetch.reset_mock()
        doc = await store.get_document("id999")
        self.mock_collection.fetch.assert_called_once_with(ids=["id999"])
        assert doc is None