"""
MCP tool for vector-based search in Knowlang.

This module provides an MCP tool that exposes Knowlang's vector
search capabilities through the Model Context Protocol.
"""

from typing import Dict, List, Optional, Any

from pydantic_graph import Graph

from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.query import VectorQuery, SearchQuery
from knowlang.configs.retrieval_config import SearchConfig
from knowlang.search.search_graph.vector_search_agent_node import VectorSearchAgentNode
from knowlang.mcp.tools.common import (
    setup_search_environment,
    format_search_results
)
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class VectorSearchTool:
    """MCP tool for vector-based search in Knowlang."""
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition for the MCP server."""
        return {
            "name": "vector-search",
            "description": "Search for information using vector embedding-based semantic search",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query in natural language"},
                    "top_k": {"type": "integer", "default": 10, "description": "Maximum number of search results to return"},
                    "score_threshold": {"type": "number", "default": 0.0, "description": "Minimum score threshold for results (0.0 to 1.0)"},
                    "filter": {"type": ["object", "null"], "default": None, "description": "Optional filter to apply to search results"}
                },
                "required": ["query"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "document_id": {"type": "string", "description": "Unique ID of the document"},
                                "content": {"type": "string", "description": "Content of the document"},
                                "metadata": {"type": "object", "description": "Metadata about the document"},
                                "score": {"type": "number", "description": "Search relevance score (0.0 to 1.0)"}
                            }
                        },
                        "description": "Search results"
                    },
                    "refined_query": {"type": ["string", "null"], "description": "The refined query used for search"},
                    "total_results": {"type": "integer", "description": "Total number of results found"}
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the vector search.
        
        Args:
            params: The search parameters
            
        Returns:
            The search results
        """
        try:
            # Extract parameters from the request
            query = params.get("query", "")
            top_k = params.get("top_k", 10)
            score_threshold = params.get("score_threshold", 0.0)
            filter_params = params.get("filter", None)
            
            LOG.info(f"Vector search request: {query}")
            
            # Set up configuration overrides
            config_overrides = {
                "retrieval.vector_search.top_k": top_k,
                "retrieval.vector_search.score_threshold": score_threshold,
                "retrieval.vector_search.filter": filter_params,
                "retrieval.vector_search.enabled": True,
                "retrieval.keyword_search.enabled": False  # Disable keyword search
            }
            
            # Set up search environment
            search_state, search_deps = await setup_search_environment(
                query=query,
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
