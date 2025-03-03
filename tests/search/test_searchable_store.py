import pytest
import unittest.mock as mock
from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.keyword_search import KeywordSearchStrategy
from knowlang.search.query import VectorQuery, KeywordQuery, SearchQuery
from knowlang.search.searchable_store import SearchableStore
from knowlang.search.vector_search import VectorSearchStrategy

# Test implementation of SearchableStore
class TestSearchableStore(SearchableStore):
    """Concrete implementation of SearchableStore for testing"""
    pass

class TestSearchableStoreClass:
    """Tests for the SearchableStore base class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.store = TestSearchableStore()
        
        # Set up mock strategies
        self.vector_strategy = mock.MagicMock(spec=VectorSearchStrategy)
        self.vector_strategy.name = SearchMethodology.VECTOR
        self.vector_strategy.required_capabilities = {SearchMethodology.VECTOR}
        
        self.keyword_strategy = mock.MagicMock(spec=KeywordSearchStrategy)
        self.keyword_strategy.name = SearchMethodology.KEYWORD
        self.keyword_strategy.required_capabilities = {SearchMethodology.KEYWORD}
        
        # Set up mock search results
        self.vector_results = [
            SearchResult(document="doc1", metadata={"field": "value1"}, score=0.8),
            SearchResult(document="doc2", metadata={"field": "value2"}, score=0.6)
        ]
        self.keyword_results = [
            SearchResult(document="doc3", metadata={"field": "value3"}, score=0.9),
            SearchResult(document="doc4", metadata={"field": "value4"}, score=0.7)
        ]
        
        # Configure mock strategies
        self.vector_strategy.search.return_value = self.vector_results
        self.keyword_strategy.search.return_value = self.keyword_results
    
    def test_capabilities(self):
        """Test capability registration and checking"""
        # Test initially no capabilities
        assert len(self.store.capabilities) == 0
        assert not self.store.has_capability(SearchMethodology.VECTOR)
        
        # Test registering capability
        self.store.register_capability(SearchMethodology.VECTOR)
        assert self.store.has_capability(SearchMethodology.VECTOR)
        assert not self.store.has_capability(SearchMethodology.KEYWORD)
        
        # Test error with invalid capability
        with pytest.raises(ValueError):
            self.store.has_capability("invalid")
        
        with pytest.raises(ValueError):
            self.store.register_capability("invalid")
    
    def test_strategy_registration(self):
        """Test strategy registration"""
        # Register capabilities
        self.store.register_capability(SearchMethodology.VECTOR)
        self.store.register_capability(SearchMethodology.KEYWORD)
        
        # Test registering strategies
        self.store.register_strategy(self.vector_strategy)
        self.store.register_strategy(self.keyword_strategy)
        
        # Test registering strategy with missing capability
        self.store = TestSearchableStore()  # Reset
        with pytest.raises(ValueError):
            self.store.register_strategy(self.vector_strategy)
    
    def test_strategy_selection(self):
        """Test automatic strategy selection based on query type"""
        # Register capabilities and strategies
        self.store.register_capability(SearchMethodology.VECTOR)
        self.store.register_capability(SearchMethodology.KEYWORD)
        self.store.register_strategy(self.vector_strategy)
        self.store.register_strategy(self.keyword_strategy)
        
        # Test vector query selection
        vector_query = VectorQuery(embedding=[0.1, 0.2])
        strategy = self.store._select_strategy_for_query(vector_query)
        assert strategy == SearchMethodology.VECTOR
        
        # Test keyword query selection
        keyword_query = KeywordQuery(text="test")
        strategy = self.store._select_strategy_for_query(keyword_query)
        assert strategy == SearchMethodology.KEYWORD
        
        # Test error with unknown query type
        with pytest.raises(ValueError):
            self.store._select_strategy_for_query(mock.MagicMock(spec=SearchQuery))
    
    @pytest.mark.asyncio
    async def test_search(self):
        """Test search with both auto and manual strategy selection"""
        # Register capabilities and strategies
        self.store.register_capability(SearchMethodology.VECTOR)
        self.store.register_capability(SearchMethodology.KEYWORD)
        self.store.register_strategy(self.vector_strategy)
        self.store.register_strategy(self.keyword_strategy)
        
        # Test search with vector query (auto-selection)
        vector_query = VectorQuery(embedding=[0.1, 0.2])
        results = await self.store.search(vector_query)
        self.vector_strategy.search.assert_called_once()
        assert results == self.vector_results
        
        # Test search with explicit strategy
        self.vector_strategy.search.reset_mock()
        results = await self.store.search(vector_query, strategy_name=SearchMethodology.VECTOR)
        self.vector_strategy.search.assert_called_once()
        assert results == self.vector_results
        
        # Test search with keyword query
        keyword_query = KeywordQuery(text="test")
        results = await self.store.search(keyword_query)
        self.keyword_strategy.search.assert_called_once()
        assert results == self.keyword_results
        
        # Test error with unknown strategy
        with pytest.raises(ValueError):
            await self.store.search(vector_query, strategy_name="unknown")