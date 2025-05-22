"""Base reranker abstract class for KnowLang."""

from abc import ABC, abstractmethod
from typing import List

from knowlang.search.base import SearchResult


class BaseReranker(ABC):
    """Abstract base class for all rerankers in KnowLang."""
    
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank search results based on the query.
        
        Args:
            query: The search query string
            results: List of search results to rerank
            
        Returns:
            List of reranked search results, sorted by relevance score
        """
        pass
