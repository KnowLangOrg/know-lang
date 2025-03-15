from typing import Dict, List, Set
import numpy as np

from knowlang.search.base import SearchResult


class MetricsCalculator:
    """Calculate standard retrieval metrics."""
    
    @staticmethod
    def calculate_metrics(results: List[SearchResult], relevant_ids: List[str]) -> Dict[str, float]:
        """
        Calculate metrics for a single query.
        
        Args:
            results: List of search results
            relevant_ids: List of relevant code IDs
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Convert relevant IDs to a set for faster lookup
        relevant_set = set(relevant_ids)
        
        # Calculate MRR
        metrics["mrr"] = MetricsCalculator._calculate_mrr(results, relevant_set)
        
        # Calculate Recall@K for different K values
        for k in [1, 5, 10, 100]:
            metrics[f"recall_at_{k}"] = MetricsCalculator._calculate_recall_at_k(results, relevant_set, k)
        
        # Calculate NDCG@10
        metrics["ndcg_at_10"] = MetricsCalculator._calculate_ndcg_at_k(results, relevant_set, 10)
        
        return metrics
    
    @staticmethod
    def _calculate_mrr(results: List[SearchResult], relevant_set: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            results: List of search results
            relevant_set: Set of relevant code IDs
            
        Returns:
            Mean Reciprocal Rank score
        """
        for i, result in enumerate(results):
            if result.metadata.get("id") in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def _calculate_recall_at_k(results: List[SearchResult], relevant_set: Set[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            results: List of search results
            relevant_set: Set of relevant code IDs
            k: Cutoff rank
            
        Returns:
            Recall@K score (fraction of relevant items found in top K results)
        """
        if not relevant_set:
            return 0.0
            
        result_ids = [r.metadata.get("id") for r in results[:k]]
        relevant_found = sum(1 for rid in result_ids if rid in relevant_set)
        
        return relevant_found / len(relevant_set)
    
    @staticmethod
    def _calculate_ndcg_at_k(results: List[SearchResult], relevant_set: Set[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        Args:
            results: List of search results
            relevant_set: Set of relevant code IDs
            k: Cutoff rank
            
        Returns:
            NDCG@K score
        """
        if not results or not relevant_set:
            return 0.0
            
        # Binary relevance for retrieved documents - 1 if relevant, 0 if not
        relevance = [1 if r.metadata.get("id") in relevant_set else 0 for r in results[:k]]
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, rel in enumerate(relevance):
            if rel > 0:
                dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate IDCG (Ideal Discounted Cumulative Gain)
        # For binary relevance, ideal ranking has all relevant items at the top
        ideal_relevance = sorted([1] * min(len(relevant_set), k) + [0] * max(0, k - len(relevant_set)), reverse=True)
        
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            if rel > 0:
                idcg += rel / np.log2(i + 2)
        
        # Calculate NDCG
        return dcg / idcg if idcg > 0 else 0.0