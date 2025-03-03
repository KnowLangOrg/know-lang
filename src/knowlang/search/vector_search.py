from typing import List, Set
from knowlang.search.base import SearchMethodology, SearchStrategy, SearchResult
from knowlang.search.query import VectorQuery, SearchQuery
from knowlang.vector_stores.base import VectorStore


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
        store: VectorStore,
        query: SearchQuery, 
        **kwargs
    ) -> List[SearchResult]:
        if not isinstance(query, VectorQuery):
            raise ValueError("VectorSearchStrategy requires a VectorQuery")
        
        # Implementation will depend on the specific store's interface
        # This is where we'd call the store-specific vector search implementation
        if hasattr(store, "vector_search"):
            results = await store.vector_search(
                query.embedding, 
                top_k=query.top_k,
                score_threshold=query.score_threshold,
                **kwargs
            )
            
            # Ensure results have the source field set
            for result in results:
                result.source = self.name
                
            return results
        else:
            raise NotImplementedError(
                f"Store {store.__class__.__name__} does not implement vector_search"
            )