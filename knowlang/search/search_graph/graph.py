from __future__ import annotations
from dataclasses import dataclass
from typing import Union
from pydantic_graph import BaseNode, GraphRunContext, End, Graph

from knowlang.search.base import SearchResult, SearchMethodology
from knowlang.search.search_graph.base import SearchState, SearchDeps, SearchOutputs
from knowlang.search.search_graph.keyword_search_agent_node import KeywordSearchAgentNode
from knowlang.search.search_graph.vector_search_agent_node import VectorSearchAgentNode
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

@dataclass
class FirstStageNode(BaseNode[SearchState, SearchDeps, SearchOutputs]):
    """First stage retrieval node that performs both keyword and vector search.
    
    This node returns End if no results are found, otherwise proceeds to reranking.
    """
    
    async def run(self, ctx: GraphRunContext[SearchState, SearchDeps]) -> Union[End[SearchOutputs], 'RerankerNode']:
        """Run both keyword and vector search"""
        try:
            # Run keyword search
            if ctx.deps.config.retrieval.keyword_search.enabled:
                # Create keyword search graph
                keyword_graph = Graph(nodes=[KeywordSearchAgentNode])
                await keyword_graph.run(KeywordSearchAgentNode(), state=ctx.state, deps=ctx.deps)
            
            if ctx.deps.config.retrieval.vector_search.enabled:
                # Create vector search graph
                vector_graph = Graph(nodes=[VectorSearchAgentNode])
                await vector_graph.run(VectorSearchAgentNode(), state=ctx.state, deps=ctx.deps)

            
            # If no results found, return early
            if not ctx.state.search_results:
                LOG.warning("No results found in first stage retrieval")
                return End(SearchOutputs(search_results=[]))
            
            # Move to reranking stage
            return RerankerNode()
        
        except Exception as e:
            LOG.error(f"Error in first stage retrieval: {e}")
            return End(SearchOutputs(search_results=[]))

@dataclass
class RerankerNode(BaseNode[SearchState, SearchDeps, SearchOutputs]):
    """Node that reranks search results using GraphCodeBERT cross-encoder.
    
    This node is the second stage of the two-stage retrieval process.
    """
    
    async def run(self, ctx: GraphRunContext[SearchState, SearchDeps]) -> End[SearchOutputs]:
        """Rerank search results"""
        if not ctx.state.search_results:
            LOG.warning("No search results to rerank")
            return End(SearchOutputs(search_results=[]))
        
        # Check if reranking is enabled
        if not ctx.deps.config.reranker.enabled:
            LOG.debug("Reranking disabled, returning first stage results")
            return End(SearchOutputs(search_results=ctx.state.search_results))
        
        try:
            # Import the reranker implementation
            from knowlang.search.reranker.knowlang_reranker import KnowLangReranker
            
            # Set up reranker
            reranker = KnowLangReranker(config=ctx.deps.config.reranker)
            
            # Rerank results
            reranked_results = reranker.rerank(
                query=ctx.state.query,
                results=ctx.state.search_results
            )
            
            LOG.info(f"Reranking complete: {len(reranked_results)} of {len(ctx.state.search_results)} results kept")
            
            # Return reranked results
            return End(SearchOutputs(search_results=reranked_results))
        
        except Exception as e:
            LOG.error(f"Error in reranking: {e}")
            # Fall back to non-reranked results
            return End(SearchOutputs(search_results=ctx.state.search_results))

# Create the search graph
search_graph = Graph(nodes=[FirstStageNode, RerankerNode])