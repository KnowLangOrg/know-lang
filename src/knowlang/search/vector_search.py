from __future__ import annotations
from typing import List, Set, TYPE_CHECKING
from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.query import VectorQuery, SearchQuery

if TYPE_CHECKING:
    from knowlang.vector_stores import VectorStore


class VectorSearchStrategy:
    """Strategy for vector-based similarity search"""
    
    @property
    def name(self) -> SearchMethodology:
        return SearchMethodology.VECTOR
    
    @property
    def required_capabilities(self) -> Set[str]:
        return {SearchMethodology.VECTOR}
    
    async def search(
        self, 
        store: 'VectorStore',
        query: SearchQuery, 
        **kwargs
    ) -> List[SearchResult]:
        if not isinstance(query, VectorQuery):
            raise ValueError("VectorSearchStrategy requires a VectorQuery")
        
        if not hasattr(store, 'vector_search'):
            raise ValueError(f"Store {store.__class__.__name__} does not support vector search")
        
        results = await store.vector_search(
            query.embedding, 
            top_k=query.top_k,
            score_threshold=query.score_threshold,
            **kwargs
        )
        
        return results