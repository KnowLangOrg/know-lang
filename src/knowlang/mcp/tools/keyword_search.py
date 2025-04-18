"""
MCP tool for keyword-based search in Knowlang.

This module provides an MCP tool that exposes Knowlang's keyword
search capabilities through the Model Context Protocol.
"""

from typing import Dict, List, Optional, Any

from pydantic_graph import Graph

from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.query import KeywordQuery, SearchQuery
from knowlang.configs.retrieval_config import SearchConfig
from knowlang.search.search_graph.keyword_search_agent_node import KeywordSearchAgentNode
from knowlang.mcp.tools.common import (
    setup_search_environment,
    format_search_results
)
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class KeywordSearchTool:
    """MCP tool for keyword-based search in Knowlang."""
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition for the MCP server."""
        # Use KeywordQuery model for input schema
        return {
            "name": "keyword-search",
            "description": "Search for information using keyword-based search",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query in natural language"},
                    "top_k": {"type": "integer", "default": 10, "description": "Maximum number of search results to return"},
                    "fields": {"type": "array", "items": {"type": "string"}, "default": ["content"], "description": "Fields to search in"},
                    "score_threshold": {"type": "number", "default": 0.0, "description": "Minimum score threshold for results (0.0 to 1.0)"}
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
        """Execute the keyword search.
        
        Args:
            params: The search parameters
            
        Returns:
            The search results
        """
        try:
            # Extract parameters from the request
            query = params.get("query", "")
            top_k = params.get("top_k", 10)
            fields = params.get("fields", ["content"])
            score_threshold = params.get("score_threshold", 0.0)
            
            LOG.info(f"Keyword search request: {query}")
            
            # Set up configuration overrides
            config_overrides = {
                "retrieval.keyword_search.top_k": top_k,
                "retrieval.keyword_search.score_threshold": score_threshold,
                "retrieval.keyword_search.enabled": True,
                "retrieval.vector_search.enabled": False  # Disable vector search
            }
            
            # Set up search environment
            search_state, search_deps = await setup_search_environment(
                query=query,
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
