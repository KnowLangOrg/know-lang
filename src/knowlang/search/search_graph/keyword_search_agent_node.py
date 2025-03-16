from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING, Union
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext, End

from knowlang.search.base import SearchMethodology
from knowlang.search.keyword_search import KeywordSearchableStore
from knowlang.search.query import KeywordQuery
from knowlang.search.search_graph.base import SearchState, SearchDeps, SearchOutputs
from knowlang.search import SearchResult
from knowlang.utils import FancyLogger, create_pydantic_model


LOG = FancyLogger(__name__)

class KeywordExtractionResult(BaseModel):
    """Result of keyword extraction by the agent"""
    query: str = Field(description="The PostgreSQL tsquery string to execute")
    logic: str = Field(description="The logic type used: 'AND' or 'OR'")
    
@dataclass
class KeywordSearchAgentNode(BaseNode[SearchState, SearchDeps, SearchOutputs]):
    """Node that uses an agent to extract keywords from user queries for Postgres tsquery search.
    
    This node is recursive - it will call itself with adjusted parameters if too few results are found.
    """
    
    # Track attempts to avoid infinite recursion
    attempts: int = 0
    previous_query: Optional[str] = None
    
    # Class-level agent instance for reuse
    _agent_instance = None
    
    system_prompt = """You are a keyword extraction expert specializing in converting natural language questions into effective PostgreSQL tsquery strings.

Your ONLY task is to extract the most important technical keywords from the user's question and format them for PostgreSQL full-text search.

Rules:
1. Output ONLY the tsquery string - no explanations, no other text
2. Focus on technical terms, code elements, and domain-specific language
3. Remove common words that don't help narrow the search
4. Format the query correctly for PostgreSQL tsquery
5. Based on the question's specificity, choose between:
   - AND logic (& operator) for specific questions with multiple constraints
   - OR logic (| operator) for broader questions needing more results

Examples with AND logic (for specific questions):
Input: "How does the keyword search strategy work in the codebase?"
Output: keyword & search & strategy

Input: "Where is the PostgreSQL hybrid store initialized?"
Output: postgresql & hybrid & store & initialize

Examples with OR logic (for broader questions):
Input: "Show me code related to vector embeddings"
Output: vector | embedding | encode

Input: "How are search results ranked?"
Output: search & (result | ranking | rank | score)

If I tell you that previous keywords returned too few results, use more permissive OR logic and fewer terms.
"""

    def _get_agent(self, ctx: GraphRunContext[SearchState, SearchDeps]) -> Agent:
        """Get or create the agent instance"""
        if self.__class__._agent_instance is None:
            self.__class__._agent_instance = Agent(
                create_pydantic_model(
                    model_provider=ctx.deps.config.llm.model_provider,
                    model_name=ctx.deps.config.llm.model_name
                ),
                system_prompt=self.system_prompt
            )
        return self.__class__._agent_instance
    
    async def _extract_keywords(self, 
                               ctx: GraphRunContext[SearchState, SearchDeps],
                               question: str, 
                               too_few_results: bool = False) -> KeywordExtractionResult:
        """Use an agent to extract keywords from the user's question"""
        keyword_agent = self._get_agent(ctx)
        
        prompt = question
        if too_few_results and self.previous_query:
            prompt = f"""Question: {question}

Your previous query "{self.previous_query}" returned too few results. 
Please generate a more permissive query with fewer terms or using more OR logic."""
            
        result = await keyword_agent.run(prompt)
        query_str = result.data.strip()
        
        # Determine if AND or OR logic is being used based on operators
        logic = "AND" if "&" in query_str and "|" not in query_str else "OR"
        
        return KeywordExtractionResult(query=query_str, logic=logic)
    
    async def _perform_keyword_search(
        self,
        query: str,
        vector_store: KeywordSearchableStore,
        top_k: int
    ) -> List[SearchResult]:
        """Perform keyword search using the extracted query"""
        try:
            # Create a keyword query
            keyword_query = KeywordQuery(
                text=query,
                top_k=top_k,
                score_threshold=0.0
            )
            
            # Use keyword search
            results = await vector_store.search(
                query=keyword_query,
                strategy_name="keyword_search"
            )
            
            logfire.info('keyword search results: {query} -> {count} results', 
                query=query, count=len(results))
            
            return results
        except Exception as e:
            LOG.warning(f"Keyword search failed: {e}")
            return []
    
    async def run(self, ctx: GraphRunContext[SearchState, SearchDeps]) -> Union['End', 'KeywordSearchAgentNode']:
        """Run the keyword search agent node"""
        # Get the query
        question = ctx.state.query
        max_retries = ctx.deps.config.retrieval.keyword_search.max_retries
        
        try:
            # Check if we've reached max attempts
            if self.attempts >= max_retries:
                LOG.warning(f"Reached maximum keyword search attempts ({max_retries})")
                # Even with no results, proceed to next stage
                return End(SearchOutputs(search_results=[]))
            
            # Extract keywords
            keyword_result = await self._extract_keywords(
                ctx,
                question=question,
                too_few_results=(self.attempts > 0)
            )
            
            LOG.debug(f"Extracted keywords: {keyword_result.query} (logic: {keyword_result.logic})")

            # Store into search state
            ctx.state.refined_queries[SearchMethodology.KEYWORD].append(keyword_result.query)
            
            # Store the query for potential future recursion
            self.previous_query = keyword_result.query
            
            # Perform the search
            results = await self._perform_keyword_search(
                query=keyword_result.query,
                vector_store=ctx.deps.store,
                top_k=ctx.deps.config.retrieval.keyword_search.top_k
            )
            
            # Store results in state
            if results:
                ctx.state.search_results += results
            
            # Check if we do have results
            if results or self.attempts >= max_retries - 1:
                # Proceed to next stage
                LOG.info(f"Found {len(results)} results, proceeding to next stage")
                return End(SearchOutputs(search_results=results))
            else:
                # Try again with more permissive keywords
                self.attempts += 1
                LOG.debug(f"Too few results ({len(results)}), trying again with attempt {self.attempts}")
                return KeywordSearchAgentNode(
                    attempts=self.attempts,
                    previous_query=keyword_result.query
                )
                
        except Exception as e:
            LOG.error(f"Error in keyword search agent: {e}")
            # Proceed to next stage despite error
            return End(SearchOutputs(search_results=[]))