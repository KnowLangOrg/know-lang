from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
import inspect
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Type, TypeVar, Union, cast

from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """Standardized search result across vector stores"""
    document: str
    metadata: Dict[str, Any]
    score: float  # Similarity/relevance score



class SearchStrategy(Protocol):
    """Protocol defining the interface for search strategies"""
    
    @property
    def name(self) -> str:
        """Unique name identifying this search strategy"""
        ...
    
    async def search(
        self, 
        store: 'SearchableStore',
        query: 'SearchQuery', 
        **kwargs
    ) -> List[SearchResult]:
        """Perform search using this strategy"""
        ...
    
    @property
    def required_capabilities(self) -> Set[str]:
        """Set of capabilities required by this search strategy"""
        ...