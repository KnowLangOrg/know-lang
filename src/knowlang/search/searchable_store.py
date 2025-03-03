from abc import ABC
from typing import Dict, List, Optional, Set, Union
from knowlang.search.query import SearchQuery, VectorQuery, KeywordQuery
from knowlang.search.base import SearchMethodology, SearchStrategy, SearchResult


class SearchableStore(ABC):
    """Abstract base class for all searchable stores"""
    
    def __init__(self):
        self._strategies: Dict[str, SearchStrategy] = {}
        self._capabilities: Set[str] = set()
        
    @property
    def capabilities(self) -> Set[str]:
        """Get the set of capabilities supported by this store"""
        return self._capabilities
    
    def has_capability(self, capability: Union[str, SearchMethodology]) -> bool:
        """Check if the store has a specific capability"""
        if isinstance(capability, SearchMethodology):
            capability = capability.value
        return capability in self._capabilities
    
    def register_capability(self, capability: Union[str, SearchMethodology]) -> None:
        """Register a capability that this store supports"""
        if isinstance(capability, SearchMethodology):
            capability = capability.value
        self._capabilities.add(capability)
        
    def register_strategy(self, strategy: SearchStrategy) -> None:
        """Register a search strategy"""
        # Check if the store has all required capabilities
        for capability in strategy.required_capabilities:
            if not self.has_capability(capability):
                raise ValueError(
                    f"Cannot register strategy '{strategy.name}': "
                    f"Missing required capability '{capability}'"
                )
        
        self._strategies[strategy.name] = strategy
        
    async def search(
        self, 
        query: SearchQuery,
        strategy_name: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search using a specific strategy or auto-select based on query type
        
        Args:
            query: The search query
            strategy_name: Optional name of strategy to use
            **kwargs: Additional parameters to pass to the strategy
            
        Returns:
            List of search results
        """
        # Auto-select strategy based on query type if not specified
        if strategy_name is None:
            strategy_name = self._select_strategy_for_query(query)
            
        if strategy_name not in self._strategies:
            raise ValueError(f"Unknown search strategy: {strategy_name}")
            
        strategy = self._strategies[strategy_name]
        return await strategy.search(self, query, **kwargs)
    
    def _select_strategy_for_query(self, query: SearchQuery) -> str:
        """Select appropriate strategy based on query type"""
        if isinstance(query, VectorQuery):
            return SearchMethodology.VECTOR
        elif isinstance(query, KeywordQuery):
            return SearchMethodology.KEYWORD
        else:
            raise ValueError(f"No default strategy for query type: {type(query).__name__}")