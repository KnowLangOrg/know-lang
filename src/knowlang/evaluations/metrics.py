from typing import Dict, List
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
        
        # Calculate MRR (Mean Reciprocal Rank)
        for i, result in enumerate(results):
            if result.metadata.get("id") in relevant_set:
                metrics["mrr"] = 1.0 / (i + 1)
                break
        else:
            metrics["mrr"] = 0.0
        
        # Calculate Recall@K
        for k in [1, 5, 10, 100]:
            if k > len(results):
                metrics[f"recall_at_{k}"] = 0.0
                continue
                
            result_ids = [r.metadata.get("id") for r in results[:k]]
            found_relevant = any(rid in relevant_set for rid in result_ids)
            
            metrics[f"recall_at_{k}"] = float(found_relevant)
        
        # Calculate NDCG@10
        if len(results) > 0:
            # Binary relevance - 1 if relevant, 0 if not
            relevance = [1 if r.metadata.get("id") in relevant_set else 0 for r in results[:10]]
            
            # DCG calculation
            dcg = 0.0
            for i, rel in enumerate(relevance):
                if rel > 0:
                    dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
            
            # Ideal DCG
            ideal_relevance = sorted(relevance, reverse=True)
            idcg = 0.0
            for i, rel in enumerate(ideal_relevance):
                if rel > 0:
                    idcg += rel / np.log2(i + 2)
            
            # NDCG
            metrics["ndcg_at_10"] = dcg / idcg if idcg > 0 else 0.0
        else:
            metrics["ndcg_at_10"] = 0.0
        
        return metrics