import json
import pytest
from pathlib import Path
from unittest import mock

from knowlang.evaluations.indexer import DatasetIndexer, QueryManager
from knowlang.models.types import EmbeddingInputType

class TestDatasetIndexer:
    """Tests for the DatasetIndexer."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        return mock.MagicMock()
    
    @pytest.fixture
    def setup_indexer(self, mock_app_config, mock_vector_store):
        """Set up a DatasetIndexer with a mock vector store."""
        indexer = DatasetIndexer(mock_app_config)
        
        # Mock the VectorStoreFactory
        with mock.patch("knowlang.evaluations.indexer.VectorStoreFactory") as mock_factory:
            mock_factory.get.return_value = mock_vector_store
            yield indexer, mock_vector_store
    
    @pytest.mark.asyncio
    async def test_index_code_snippets(self, setup_indexer, sample_query_code_pairs):
        """Test indexing code snippets."""
        indexer, mock_vector_store = setup_indexer
        
        # Mock the generate_embedding function
        with mock.patch("knowlang.evaluations.indexer.generate_embedding") as mock_generate_embedding:
            mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
            
            indexed_ids = await indexer.index_code_snippets(sample_query_code_pairs, batch_size=1)
        
        assert len(indexed_ids) == 2
        assert "code1" in indexed_ids
        assert "code2" in indexed_ids
        
        # Check that generate_embedding was called for each snippet
        assert mock_generate_embedding.call_count == 2
        
        # Check that the correct parameters were passed to generate_embedding
        mock_generate_embedding.assert_any_call(
            input=sample_query_code_pairs[0].code,
            config=indexer.config.embedding,
            input_type=EmbeddingInputType.DOCUMENT
        )

class TestQueryManager:
    """Tests for the QueryManager."""
    
    def test_save_query_mappings(self, temp_dir, sample_query_code_pairs):
        """Test saving query mappings."""
        output_dir = temp_dir / "query_mappings"
        manager = QueryManager(output_dir)
        
        manager.save_query_mappings(sample_query_code_pairs, "test_dataset")
        
        file_path = output_dir / "test_dataset_query_map.json"
        assert file_path.exists()
        
        with open(file_path, "r", encoding="utf-8") as f:
            query_map = json.load(f)
        
        assert "query1" in query_map
        assert "query2" in query_map
        assert query_map["query1"]["query"] == "How to sort a list in Python"
        assert "code1" in query_map["query1"]["relevant_code"]
    
    def test_load_query_mappings(self, temp_dir, sample_query_code_pairs):
        """Test loading query mappings."""
        output_dir = temp_dir / "query_mappings"
        manager = QueryManager(output_dir)
        
        # First save the mappings
        manager.save_query_mappings(sample_query_code_pairs, "test_dataset")
        
        # Then load them
        query_map = manager.load_query_mappings("test_dataset")
        
        assert "query1" in query_map
        assert query_map["query1"]["query"] == "How to sort a list in Python"
        assert "code1" in query_map["query1"]["relevant_code"]