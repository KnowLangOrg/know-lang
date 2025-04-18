"""
Common models and utilities for MCP tools.

This module provides shared functionality for Knowlang's MCP tools.
"""

from typing import Dict, List, Optional, Any, Tuple

from pydantic import BaseModel, Field

from knowlang.configs.config import AppConfig
from knowlang.search.base import SearchResult, SearchMethodology
from knowlang.search.search_graph.base import SearchState, SearchDeps
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class SearchResultModel(BaseModel):
    """Model for a single search result."""
    document_id: str = Field(description="Unique ID of the document")
    content: str = Field(description="Content of the document")
    metadata: Dict[str, Any] = Field(description="Metadata about the document")
    score: float = Field(description="Search relevance score (0.0 to 1.0)")

async def setup_search_environment(
    query: str,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[SearchState, SearchDeps]:
    """Set up search state and dependencies.
    
    Args:
        query: The search query
        config_overrides: Optional configuration overrides
        
    Returns:
        Tuple of SearchState and SearchDeps
    """
    # Create app and get config
    app = AppConfig()
    config = app.config
    
    # Apply configuration overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if key.startswith("retrieval."):
                parts = key.split(".")
                if len(parts) == 3 and parts[1] in ["keyword_search", "vector_search"]:
                    section = getattr(config.retrieval, parts[1])
                    setattr(section, parts[2], value)
    
    # Create search state
    search_state = SearchState(
        query=query,
        refined_queries={
            SearchMethodology.KEYWORD: [],
            SearchMethodology.VECTOR: []
        },
        search_results=[]
    )
    
    # Create search dependencies
    search_deps = SearchDeps(
        config=config,
        store=app.get_store()
    )
    
    return search_state, search_deps

def format_search_results(
    search_results: List[SearchResult],
    refined_queries: Dict[SearchMethodology, List[str]],
    methodology: SearchMethodology
) -> Dict[str, Any]:
    """Format search results for response.
    
    Args:
        search_results: List of search results
        refined_queries: Dictionary of refined queries by methodology
        methodology: The search methodology used
        
    Returns:
        Formatted response dictionary
    """
    results = []
    
    for result in search_results:
        results.append({
            "document_id": result.id,
            "content": result.content,
            "metadata": result.metadata,
            "score": result.score
        })
    
    # Get refined query if available
    refined_query = None
    if refined_queries[methodology]:
        refined_query = refined_queries[methodology][-1]
    
    return {
        "results": results,
        "refined_query": refined_query,
        "total_results": len(results),
    }
