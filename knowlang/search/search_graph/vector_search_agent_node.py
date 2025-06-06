from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext, End

from knowlang.models.embeddings import generate_embedding
from knowlang.search.query import VectorQuery
from knowlang.search.base import SearchResult, SearchMethodology
from knowlang.search.search_graph.base import SearchState, SearchDeps, SearchOutputs
from knowlang.utils import FancyLogger, create_pydantic_model
from knowlang.models.types import EmbeddingInputType, EmbeddingVector

if TYPE_CHECKING:
    from knowlang.vector_stores.base import VectorStore

LOG = FancyLogger(__name__)

class QueryRefinementResult(BaseModel):
    """Result of query refinement by the agent"""
    refined_query: str = Field(description="The refined query for vector search")
    explanation: str = Field(description="Explanation of why the query was refined")
    
@dataclass
class VectorSearchAgentNode(BaseNode[SearchState, SearchDeps, SearchOutputs]):
    """Node that uses an agent to refine user queries for vector embedding search.
    
    This node is recursive - it will call itself with adjusted parameters if too few results are found.
    """
    
    # Track attempts to avoid infinite recursion
    attempts: int = 0
    previous_query: Optional[str] = None
    
    # Class-level agent instance for reuse
    _agent_instance = None
    
    system_prompt = """You are a query refinement expert specializing in transforming natural language questions into more effective queries for vector semantic search in codebases.

Your task is to refine users' code-related questions to make them more effective for semantic vector search, where similarity is based on meaning rather than exact keyword matching.

Rules:
1. Output ONLY the refined query - no explanations or other text
2. Preserve technical terms, code elements, and programming language names exactly
3. Expand acronyms and abbreviations relevant to programming
4. Add synonyms or alternative phrasings for important concepts
5. Be descriptive but concise - focus on the core intent
6. Include relevant programming concepts, design patterns, or methodologies
7. For API or function questions, include likely parameter or return value terms

Examples of good refinements:
Original: "How do I implement authentication?"
Refined: "authentication implementation login user session token JWT OAuth password hashing security"

Original: "Show me code that handles errors in async functions"
Refined: "error handling asynchronous functions try catch exceptions promises async await error propagation"

Original: "How does the vector search code work?"
Refined: "vector embedding similarity search implementation nearest neighbors cosine distance ANN indexing"

Original: "Where is the database connection established?"
Refined: "database connection initialization establish configure pool ORM SQL connection string"

Original: "Help me understand the keyword extraction code"
Refined: "keyword extraction tokenization natural language processing text analysis parsing tokens search query"

If I tell you that previous query returned too few results, make the query more general by:
1. Adding more alternative terms
2. Removing specific constraints
3. Using broader programming concepts
4. Including related technologies or approaches
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
    
    async def _refine_query(self, 
            ctx: GraphRunContext[SearchState, SearchDeps],
            question: str, 
            too_few_results: bool = False
        ) -> QueryRefinementResult:
        """Use an agent to refine the user's question for vector search"""
        # Check if query refinement is enabled
        if not ctx.deps.config.retrieval.vector_search.query_refinement:
            # No refinement, use the original query
            LOG.debug(f"Query refinement disabled, using original query: {question}")
            return QueryRefinementResult(
                refined_query=question,
                explanation="Query refinement disabled"
            )

        query_agent = self._get_agent(ctx)
        
        prompt = question
        if too_few_results and self.previous_query:
            prompt = f"""Question: {question}

Your previous refinement "{self.previous_query}" returned too few results. 
Please generate a more general query with broader terms or additional synonyms."""
            
        result = await query_agent.run(prompt)
        refined_query = result.data.strip()
        
        # For the first attempt, we might want to log the explanation
        explanation = "Initial query refinement" if not too_few_results else "Generating broader query due to few results"
        
        return QueryRefinementResult(refined_query=refined_query, explanation=explanation)

    async def _generate_embeddings(self, 
            ctx: GraphRunContext[SearchState, SearchDeps], 
            text: str
        ) -> EmbeddingVector:
        """Generate embeddings for the refined query"""
        try:
            embedding = generate_embedding(
                input=text,
                config=ctx.deps.config.embedding,
                input_type=EmbeddingInputType.QUERY
            )
            
            if not embedding or len(embedding) == 0:
                LOG.warning(f"Failed to generate embeddings for query: {text}")
                return []
                
            return embedding
        except Exception as e:
            LOG.error(f"Error generating embeddings: {e}")
            return []
    
    async def _perform_vector_search(
        self,
        embedding: List[float],
        vector_store: VectorStore,
        top_k: int,
        score_threshold: float,
        filter : Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform vector search using the generated embeddings"""
        try:
            # Create a vector query
            vector_query = VectorQuery(
                embedding=embedding,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            # Use vector search
            results = await vector_store.search(
                query=vector_query,
                strategy_name=SearchMethodology.VECTOR,
                filter=filter,
            )
            
            return results
        except Exception as e:
            LOG.warning(f"Vector search failed: {e}")
            return []
    
    async def run(self, ctx: GraphRunContext[SearchState, SearchDeps]) -> Union['End', 'VectorSearchAgentNode']:
        """Run the vector search agent node"""
        # Get the query
        question = ctx.state.query
        
        # Access vector search configuration
        vector_search_config = ctx.deps.config.retrieval.vector_search
        max_retries = vector_search_config.max_retries
        top_k = vector_search_config.top_k
        score_threshold = vector_search_config.score_threshold
        
        try:
            # Check if we've reached max attempts
            if self.attempts >= max_retries:
                LOG.warning(f"Reached maximum vector search attempts ({max_retries})")
                # Even with no results, proceed to next stage
                return End(SearchOutputs(search_results=[]))
            
            # Refine the query for vector search
            query_refinement = await self._refine_query(
                ctx,
                question=question,
                too_few_results=(self.attempts > 0)
            )
            
            LOG.debug(f"Refined query: {query_refinement.refined_query} (reason: {query_refinement.explanation})")

            ctx.state.refined_queries[SearchMethodology.VECTOR].append(query_refinement.refined_query)
            
            # Store the query for potential future recursion
            self.previous_query = query_refinement.refined_query
            
            # Generate embeddings for the refined query
            embedding = await self._generate_embeddings(ctx, query_refinement.refined_query)
            
            if not embedding:
                LOG.warning("Could not generate embeddings, skipping vector search")
                return End(SearchOutputs(search_results=[]))
            
            # Perform the search
            results = await self._perform_vector_search(
                embedding=embedding,
                vector_store=ctx.deps.store,
                top_k=top_k,
                score_threshold=score_threshold,
                filter=ctx.deps.config.retrieval.vector_search.filter
            )
            
            if results:
                ctx.state.search_results += results

            # Check if we do have results
            if results or self.attempts >= max_retries - 1:
                # Proceed to next stage
                LOG.info(f"Found {len(results)} results, proceeding to next stage")
                return End(SearchOutputs(search_results=results))
            else:
                # Try again with broader query
                self.attempts += 1
                LOG.debug(f"Too few results ({len(results)}), trying again with attempt {self.attempts}")
                return VectorSearchAgentNode(
                    attempts=self.attempts,
                    previous_query=query_refinement.refined_query
                )
                
        except Exception as e:
            LOG.error(f"Error in vector search agent: {e}")
            # Proceed to next stage despite error
            return End(SearchOutputs(search_results=[]))