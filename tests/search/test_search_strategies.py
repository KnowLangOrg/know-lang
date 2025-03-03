import pytest
import unittest.mock as mock
from typing import List, Set

from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.query import VectorQuery, KeywordQuery, SearchQuery
from knowlang.search.vector_search import VectorSearchStrategy
from knowlang.search.keyword_search import KeywordSearchStrategy
from knowlang.vector_stores.base import VectorStore
from knowlang.search.keyword_search import KeywordSearchableStore

class TestSearchStrategies:
    """Tests for search strategies"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Set up vector strategy
        self.vector_strategy = VectorSearchStrategy()
        
        # Set up keyword strategy
        self.keyword_strategy = KeywordSearchStrategy()
        
        # Mock vector store
        self.mock_vector_store = mock.MagicMock(spec=VectorStore)
        self.mock_vector_store.vector_search.return_value = [
            SearchResult(document="doc1", metadata={"field": "value1"}, score=0.8),
            SearchResult(document="doc2", metadata={"field": "value2"}, score=0.6)
        ]
        
        # Mock keyword store
        self.mock_keyword_store = mock.MagicMock(spec=KeywordSearchableStore)
        self.mock_keyword_store.keyword_search.return_value = [
            SearchResult(document="doc3", metadata={"field": "value3"}, score=0.9),
            SearchResult(document="doc4", metadata={"field": "value4"}, score=0.7)
        ]
    
    def test_vector_strategy_name_and_capabilities(self):
        """Test vector strategy name and required capabilities"""
        assert self.vector_strategy.name == SearchMethodology.VECTOR
        assert SearchMethodology.VECTOR in self.vector_strategy.required_capabilities
    
    def test_keyword_strategy_name_and_capabilities(self):
        """Test keyword strategy name and required capabilities"""
        assert self.keyword_strategy.name == SearchMethodology.KEYWORD
        assert SearchMethodology.KEYWORD in self.keyword_strategy.required_capabilities
    
    @pytest.mark.asyncio
    async def test_vector_search_strategy(self):
        """Test vector search strategy execution"""
        # Create valid vector query
        query = VectorQuery(embedding=[0.1, 0.2, 0.3], top_k=2)
        
        # Test successful search
        results = await self.vector_strategy.search(self.mock_vector_store, query)
        self.mock_vector_store.vector_search.assert_called_once_with(
            query.embedding,
            top_k=query.top_k,
            score_threshold=query.score_threshold
        )
        assert len(results) == 2
        
        # Test with invalid query type
        invalid_query = KeywordQuery(text="test", top_k=2)
        with pytest.raises(ValueError):
            await self.vector_strategy.search(self.mock_vector_store, invalid_query)
        
        # Test with store that doesn't have vector_search method
        invalid_store = object()
        with pytest.raises(ValueError):
            await self.vector_strategy.search(invalid_store, query)
    
    @pytest.mark.asyncio
    async def test_keyword_search_strategy(self):
        """Test keyword search strategy execution"""
        # Create valid keyword query
        query = KeywordQuery(text="test", fields=["content"], top_k=2)
        
        # Test successful search
        results = await self.keyword_strategy.search(self.mock_keyword_store, query)
        self.mock_keyword_store.keyword_search.assert_called_once_with(
            query.text,
            fields=query.fields,
            top_k=query.top_k,
            score_threshold=query.score_threshold
        )
        assert len(results) == 2
        
        # Test with invalid query type
        invalid_query = VectorQuery(embedding=[0.1, 0.2, 0.3], top_k=2)
        with pytest.raises(ValueError):
            await self.keyword_strategy.search(self.mock_keyword_store, invalid_query)
        
        # Test with store that doesn't have keyword_search method
        invalid_store = object()
        with pytest.raises(ValueError):
            await self.keyword_strategy.search(invalid_store, query)