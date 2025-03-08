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
        self.db_config.content_field = "content"  # Add content_field to config
        
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
        assert store.content_field == self.db_config.content_field  # Check content_field
        
        # Test error when connection URL is missing
        self.db_config.connection_url = None
        with pytest.raises(VectorStoreInitError):
            PostgresVectorStore.create_from_config(
                config=self.db_config,
                embedding_config=self.embedding_config
            )
    
    @pytest.mark.asyncio
    async def test_add_documents_with_content_field(self):
        """Test adding documents with content field"""
        # Create store with content_field set
        store = PostgresVectorStore(
            connection_string=self.db_config.connection_url,
            table_name=self.db_config.collection_name,
            embedding_dim=self.embedding_config.dimension,
            content_field="content"
        )
        store.collection = self.mock_collection
        
        # Test data
        documents = ["content1", "content2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"field": "value1"}, {"field": "value2"}]
        ids = ["id1", "id2"]
        
        # Test adding documents
        await store.add_documents(documents, embeddings, metadatas, ids)
        self.mock_collection.upsert.assert_called_once()
        
        # Check that the content field was added to the metadata
        args, kwargs = self.mock_collection.upsert.call_args
        if "records" in kwargs:
            records = kwargs["records"]
        elif len(args) > 0:
            records = args[0]
        else:
            assert False, "No arguments found in upsert call"
        
        # Verify content field was added to metadata
        for i, (id, emb, meta) in enumerate(records):
            assert meta["content"] == documents[i]
            assert meta["field"] == f"value{i+1}"  # Original metadata preserved
    
    @pytest.mark.asyncio
    async def test_update_document_with_content_field(self):
        """Test updating a document with content field"""
        # Create store with content_field set
        store = PostgresVectorStore(
            connection_string=self.db_config.connection_url,
            table_name=self.db_config.collection_name,
            embedding_dim=self.embedding_config.dimension,
            content_field="content"
        )
        store.collection = self.mock_collection
        
        # Test updating a document
        document = "updated content"
        embedding = [0.5, 0.6]
        metadata = {"field": "updated_value"}
        
        await store.update_document("id1", document, embedding, metadata)
        
        # Verify that content was added to metadata
        self.mock_collection.upsert.assert_called_once()
        args, kwargs = self.mock_collection.upsert.call_args
        if len(args) > 0:
            records = args[0]
        else:
            assert False, "No arguments found in upsert call"
        
        id, emb, meta = records[0]
        assert id == "id1"
        assert emb == embedding
        assert meta["content"] == document
        assert meta["field"] == "updated_value"
    
    @pytest.mark.asyncio
    async def test_add_documents_without_content_field(self):
        """Test adding documents without content field"""
        # Create store without content_field
        store = PostgresVectorStore(
            connection_string=self.db_config.connection_url,
            table_name=self.db_config.collection_name,
            embedding_dim=self.embedding_config.dimension,
            content_field=None  # Explicitly set to None
        )
        store.collection = self.mock_collection
        
        # Test data
        documents = ["content1", "content2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"field": "value1"}, {"field": "value2"}]
        ids = ["id1", "id2"]
        
        # Test adding documents
        await store.add_documents(documents, embeddings, metadatas, ids)
        
        # Check that the content field was NOT added to the metadata
        args, kwargs = self.mock_collection.upsert.call_args
        if "records" in kwargs:
            records = kwargs["records"]
        elif len(args) > 0:
            records = args[0]
        else:
            assert False, "No arguments found in upsert call"
        
        # Verify original metadata is preserved but no content field added
        for i, (id, emb, meta) in enumerate(records):
            assert "content" not in meta
            assert meta["field"] == f"value{i+1}"