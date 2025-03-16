from typing import Dict, List, Any
import time
from pydantic import BaseModel, Field

from knowlang.search.base import SearchResult


class QueryCodePair(BaseModel):
    """Represents a single query-code pair from a benchmark dataset."""
    query_id: str
    query: str
    code_id: str
    code: str
    language: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_relevant: bool = True  # For datasets with relevance judgments


class DatasetStats(BaseModel):
    """Statistics about the dataset being processed."""
    dataset_name: str
    total_queries: int = 0
    total_code_snippets: int = 0
    queries_by_language: Dict[str, int] = Field(default_factory=dict)
    snippets_by_language: Dict[str, int] = Field(default_factory=dict)
    
    def update_for_pair(self, pair: QueryCodePair) -> None:
        """Update stats based on a new query-code pair."""
        self.total_queries += 1
        self.total_code_snippets += 1
        
        lang = pair.language
        self.queries_by_language[lang] = self.queries_by_language.get(lang, 0) + 1
        self.snippets_by_language[lang] = self.snippets_by_language.get(lang, 0) + 1
    
    def summary(self) -> str:
        """Get a human-readable summary of the dataset stats."""
        return (
            f"Dataset: {self.dataset_name}\n"
            f"  Total Queries: {self.total_queries}\n"
            f"  Total Code Snippets: {self.total_code_snippets}\n"
            f"  Queries by Language: {self.queries_by_language}\n"
            f"  Snippets by Language: {self.snippets_by_language}\n"
        )


class SearchConfiguration(BaseModel):
    """Configuration for a search evaluation run."""
    name: str
    description: str
    keyword_search_enabled: bool = True
    vector_search_enabled: bool = True
    reranking_enabled: bool = True
    keyword_search_threshold: float = 0.0
    vector_search_threshold: float = 0.6
    reranker_threshold: float = 0.5
    keyword_search_top_k: int = 50
    vector_search_top_k: int = 50
    reranker_top_k: int = 10


class EvaluationRun(BaseModel):
    """Results from a single evaluation run."""
    configuration: SearchConfiguration
    dataset_name: str
    language: str
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    mrr: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    ndcg_at_10: float = 0.0
    avg_query_time: float = 0.0
    num_queries: int = 0


class QueryEvaluationResult(BaseModel):
    """Results for a single query evaluation."""
    query_id: str
    query: str
    relevant_code_ids: List[str]
    results: List[SearchResult]
    query_time: float
    mrr: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    ndcg_at_10: float = 0.0