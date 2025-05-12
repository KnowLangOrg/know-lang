from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List
from pydantic import BaseModel
from knowlang.configs.config import AppConfig
from knowlang.search.base import SearchResult, SearchMethodology
from knowlang.search.searchable_store import SearchableStore


class SearchOutputs(BaseModel):
    """Outputs from the search graph"""
    search_results: List[SearchResult]


@dataclass
class SearchState:
    """State maintained throughout the search graph"""
    query: str
    refined_queries: Dict[SearchMethodology, List[str]] = field(default_factory=lambda: defaultdict(list))
    search_results: List[SearchResult] = field(default_factory=list)


@dataclass
class SearchDeps:
    """Dependencies required by the search graph"""
    store: SearchableStore
    config: AppConfig