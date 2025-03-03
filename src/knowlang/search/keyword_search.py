from abc import abstractmethod
from typing import List, Set
from knowlang.search.query import SearchQuery, KeywordQuery
from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.searchable_store import SearchableStore


class KeywordSearchableStore(SearchableStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_capability(SearchMethodology.KEYWORD)
        self.register_strategy(KeywordSearchStrategy())
    
    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        fields: List[str],
        top_k: int = 10,
        score_threshold: float = 0.0,
        **kwargs
    ) -> List[SearchResult]:
        """Search for documents using keyword search"""
        ...

class KeywordSearchStrategy:
    """Strategy for keyword-based search"""
    
    @property
    def name(self) -> SearchMethodology:
        return SearchMethodology.KEYWORD
    
    @property
    def required_capabilities(self) -> Set[SearchMethodology]:
        return {SearchMethodology.KEYWORD}
    
    async def search(
        self, 
        store: KeywordSearchableStore,
        query: SearchQuery, 
        **kwargs
    ) -> List[SearchResult]:
        if not isinstance(query, KeywordQuery):
            raise ValueError("KeywordSearchStrategy requires a KeywordQuery")
        
        if not hasattr(store, 'keyword_search'):
            raise ValueError(f"Store {store.__class__.__name__} does not support keyword search")

        results = await store.keyword_search(
            query.text,
            fields=query.fields,
            top_k=query.top_k,
            score_threshold=query.score_threshold,
            **kwargs
        )
        
        return results