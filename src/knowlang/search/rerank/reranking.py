from typing import List, Any
from knowlang.configs import RerankerConfig
from knowlang.search.base import SearchResult

class GraphCodeBertReranker:
    """Reranker implementation using GraphCodeBERT cross-encoder"""
    
    def __init__(self, config: RerankerConfig):
        """
        Initialize reranker with configuration.
        
        Args:
            config: Reranker configuration
        """
        self.config = config
        
    def rerank(
        self, 
        query: str, 
        raw_search_results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Rerank search results using GraphCodeBERT cross-encoder.
        
        Args:
            query: User query
            raw_search_results: List of retrieved documents/code snippets with metadata
            
        Returns:
            Reranked list of results with scores
        """
        if not self.config.enabled or not raw_search_results:
            return raw_search_results
        
        # Extract content to rerank
        contents = [result.document for result in raw_search_results]
        
        # Import here to avoid circular imports
        from knowlang.models.graph_code_bert import calculate_relevance_scores
        
        # Score query-code pairs using cross-encoder
        scores = calculate_relevance_scores(
            query=query, 
            code_snippets=contents,
        )
        
        # Create results with scores
        ranked_results : List[SearchResult] = []
        for i, (score, search_result) in enumerate(zip(scores, raw_search_results)):
            # Skip if below threshold
            if score < self.config.relevance_threshold:
                continue
                
            ranked_results.append(search_result.model_copy(update={"score": score}))
        
        # Sort by relevance score (descending)
        ranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        return ranked_results[:self.config.top_k]