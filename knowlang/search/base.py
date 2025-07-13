from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Set

from pydantic import BaseModel

from knowlang.api import ApiModelRegistry

if TYPE_CHECKING:
    from knowlang.search.query import SearchQuery
    from knowlang.search.searchable_store import SearchableStore


@ApiModelRegistry.register
class SearchResult(BaseModel):
    """Standardized search result across vector stores"""

    document: str
    metadata: Dict[str, Any]
    score: float  # Similarity/relevance score


class SearchMethodology(str, Enum):
    """Enumeration of capabilities a store might support"""

    VECTOR = "approximate_vector_search"


class SearchStrategy(Protocol):
    """Protocol defining the interface for search strategies"""

    @property
    def name(self) -> "SearchMethodology":
        """Unique name identifying this search strategy"""
        ...

    async def search(
        self, store: "SearchableStore", query: "SearchQuery", **kwargs
    ) -> List[SearchResult]:
        """Perform search using this strategy"""
        ...

    @property
    def required_capabilities(self) -> Set[str]:
        """Set of capabilities required by this search strategy"""
        ...
