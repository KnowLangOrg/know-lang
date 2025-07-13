from abc import ABC
from typing import Dict, List, Optional, Set

from knowlang.search.base import SearchMethodology, SearchResult, SearchStrategy
from knowlang.search.query import SearchQuery, VectorQuery


class SearchableStore(ABC):
    """Abstract base class for all searchable stores"""

    def __init__(self):
        self._strategies: Dict[str, SearchStrategy] = {}
        self._capabilities: Set[SearchMethodology] = set()

    @property
    def capabilities(self) -> Set[SearchMethodology]:
        """Get the set of capabilities supported by this store"""
        return self._capabilities

    def has_capability(self, capability: SearchMethodology) -> bool:
        """
        Check if the store has a specific capability
        Seraches like hybrid search requires multiple capabilities
        """
        if not isinstance(capability, SearchMethodology):
            raise ValueError("Unknown search methodology type")

        return capability in self._capabilities

    def register_capability(self, capability: SearchMethodology) -> None:
        """Register a capability that this store supports"""
        if not isinstance(capability, SearchMethodology):
            raise ValueError("Unknown search methodology type")

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
        strategy_name: Optional[SearchMethodology] = None,
        **kwargs,
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
        else:
            raise ValueError(
                f"No default strategy for query type: {type(query).__name__}"
            )
