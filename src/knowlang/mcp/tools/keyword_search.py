"""
MCP tool for keyword-based search in Knowlang.

This module provides an MCP tool that exposes Knowlang's keyword
search capabilities through the Model Context Protocol.
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field
from pydantic_graph import Graph

from knowlang.search.base import SearchMethodology
from knowlang.search.search_graph.keyword_search_agent_node import KeywordSearchAgentNode
from knowlang.mcp.tools.common import (
    SearchResultModel, 
    setup_search_environment,
    format_search_results
)
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class KeywordSearchRequest(BaseModel):
    """Request model for keyword search."""
    query: str = Field(
        description="The search query in natural language"
    )
    top_k: int = Field(
        default=10, 
        description="Maximum number of search results to return"
    )
    fields: List[str] = Field(
        default=["content"],
        description="Fields to search in"
    )
    score_threshold: float = Field(
        default=0.0,
        description="Minimum score threshold for results (0.0 to 1.0)"
    )

class KeywordSearchResponse(BaseModel):
    """Response model for keyword search."""
    results: List[SearchResultModel] = Field(description="Search results")
    refined_query: Optional[str] = Field(description="The refined query used for search")
    total_results: int = Field(description="Total number of results found")

class KeywordSearchTool:
    """MCP tool for keyword-based search in Knowlang."""
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition for the MCP server."""
        return {
            "name": "keyword-search",
            "description": "Search for information using keyword-based search",
            "input_schema": KeywordSearchRequest.model_json_schema(),
            "output_schema": KeywordSearchResponse.model_json_schema(),
        }
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the keyword search.
        
        Args:
            params: The search parameters
            
        Returns:
            The search results
        """
        try:
            # Parse request
            request = KeywordSearchRequest(**params)
            LOG.info(f"Keyword search request: {request.query}")
            
            # Set up configuration overrides
            config_overrides = {
                "retrieval.keyword_search.top_k": request.top_k,
                "retrieval.keyword_search.score_threshold": request.score_threshold,
                "retrieval.keyword_search.fields": request.fields,
                "retrieval.keyword_search.enabled": True,
                "retrieval.vector_search.enabled": False  # Disable vector search
            }
            
            # Set up search environment
            search_state, search_deps = await setup_search_environment(
                query=request.query,
                config_overrides=config_overrides
            )
            
            # Create and run graph with single node
            keyword_graph = Graph(nodes=[KeywordSearchAgentNode])
            await keyword_graph.run(KeywordSearchAgentNode(), state=search_state, deps=search_deps)
            
            # Format and return results
            result = format_search_results(
                search_results=search_state.search_results,
                refined_queries=search_state.refined_queries,
                methodology=SearchMethodology.KEYWORD
            )
            
            LOG.info(f"Keyword search completed with {len(result['results'])} results")
            return result
            
        except Exception as e:
            LOG.error(f"Error in keyword search: {e}")
            # Return empty results on error
            return {
                "results": [],
                "refined_query": None,
                "total_results": 0,
                "error": str(e)
            }
