"""
MCP tool for keyword-based search in Knowlang.

This module provides an MCP tool that exposes Knowlang's keyword
search capabilities through the Model Context Protocol.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple

import mcp
from pydantic import BaseModel, Field

from knowlang.search.query import KeywordQuery
from knowlang.search.base import SearchResult, SearchMethodology
from knowlang.search.search_graph.keyword_search_agent_node import KeywordSearchAgentNode
from knowlang.search.search_graph.base import SearchState, SearchDeps
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

class SearchResultModel(BaseModel):
    """Model for a single search result."""
    document_id: str = Field(description="Unique ID of the document")
    content: str = Field(description="Content of the document")
    metadata: Dict[str, Any] = Field(description="Metadata about the document")
    score: float = Field(description="Search relevance score (0.0 to 1.0)")

class KeywordSearchResponse(BaseModel):
    """Response model for keyword search."""
    results: List[SearchResultModel] = Field(description="Search results")
    refined_query: Optional[str] = Field(description="The refined query used for search")
    total_results: int = Field(description="Total number of results found")

@mcp.tool()
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
            
            # Set up search state and dependencies
            search_state, search_deps = await self._setup_search(request)
            
            # Execute search using the keyword search agent node
            search_node = KeywordSearchAgentNode()
            search_results = await self._execute_search(search_node, search_state, search_deps)
            
            # Format response
            result = await self._format_response(search_results, search_state)
            
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
    
    async def _setup_search(self, request: KeywordSearchRequest) -> Tuple[SearchState, SearchDeps]:
        """Set up search state and dependencies."""
        # Import here to avoid circular dependencies
        from knowlang.core import get_current_app
        
        # Get the app and config
        app = get_current_app()
        config = app.config
        
        # Create search state
        search_state = SearchState(
            query=request.query,
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
    
    async def _execute_search(self, 
                             search_node: KeywordSearchAgentNode, 
                             search_state: SearchState, 
                             search_deps: SearchDeps) -> List[SearchResult]:
        """Execute the search using the keyword search agent node."""
        try:
            # Create context
            from pydantic_graph import GraphRunContext
            
            ctx = GraphRunContext(state=search_state, deps=search_deps)
            
            # Run the search node
            result = await search_node.run(ctx)
            
            # Get search results
            search_results = ctx.state.search_results
            
            return search_results
        except Exception as e:
            LOG.error(f"Error executing keyword search: {e}")
            return []
    
    async def _format_response(self, 
                              search_results: List[SearchResult],
                              search_state: SearchState) -> Dict[str, Any]:
        """Format the search results for the response."""
        results = []
        
        for result in search_results:
            results.append({
                "document_id": result.id,
                "content": result.content,
                "metadata": result.metadata,
                "score": result.score
            })
        
        # Get refined queries if available
        refined_query = None
        if search_state.refined_queries[SearchMethodology.KEYWORD]:
            refined_query = search_state.refined_queries[SearchMethodology.KEYWORD][-1]
        
        return {
            "results": results,
            "refined_query": refined_query,
            "total_results": len(results),
        }
