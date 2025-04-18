"""
MCP tool for vector-based search in Knowlang.

This module provides an MCP tool that exposes Knowlang's vector
search capabilities through the Model Context Protocol.
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field
from pydantic_graph import Graph

from knowlang.search.base import SearchMethodology
from knowlang.search.search_graph.vector_search_agent_node import VectorSearchAgentNode
from knowlang.mcp.tools.common import (
    SearchResultModel, 
    setup_search_environment,
    format_search_results
)
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class VectorSearchRequest(BaseModel):
    """Request model for vector search."""
    query: str = Field(
        description="The search query in natural language"
    )
    top_k: int = Field(
        default=10, 
        description="Maximum number of search results to return"
    )
    score_threshold: float = Field(
        default=0.0,
        description="Minimum score threshold for results (0.0 to 1.0)"
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filter to apply to search results"
    )

class VectorSearchResponse(BaseModel):
    """Response model for vector search."""
    results: List[SearchResultModel] = Field(description="Search results")
    refined_query: Optional[str] = Field(description="The refined query used for search")
    total_results: int = Field(description="Total number of results found")

class VectorSearchTool:
    """MCP tool for vector-based search in Knowlang."""
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition for the MCP server."""
        return {
            "name": "vector-search",
            "description": "Search for information using vector embedding-based semantic search",
            "input_schema": VectorSearchRequest.model_json_schema(),
            "output_schema": VectorSearchResponse.model_json_schema(),
        }
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the vector search.
        
        Args:
            params: The search parameters
            
        Returns:
            The search results
        """
        try:
            # Parse request
            request = VectorSearchRequest(**params)
            LOG.info(f"Vector search request: {request.query}")
            
            # Set up configuration overrides
            config_overrides = {
                "retrieval.vector_search.top_k": request.top_k,
                "retrieval.vector_search.score_threshold": request.score_threshold,
                "retrieval.vector_search.filter": request.filter,
                "retrieval.vector_search.enabled": True,
                "retrieval.keyword_search.enabled": False  # Disable keyword search
            }
            
            # Set up search environment
            search_state, search_deps = await setup_search_environment(
                query=request.query,
                config_overrides=config_overrides
            )
            
            # Create and run graph with single node
            vector_graph = Graph(nodes=[VectorSearchAgentNode])
            await vector_graph.run(VectorSearchAgentNode(), state=search_state, deps=search_deps)
            
            # Format and return results
            result = format_search_results(
                search_results=search_state.search_results,
                refined_queries=search_state.refined_queries,
                methodology=SearchMethodology.VECTOR
            )
            
            LOG.info(f"Vector search completed with {len(result['results'])} results")
            return result
            
        except Exception as e:
            LOG.error(f"Error in vector search: {e}")
            # Return empty results on error
            return {
                "results": [],
                "refined_query": None,
                "total_results": 0,
                "error": str(e)
            }
