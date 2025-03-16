from knowlang.evaluations.metrics import MetricsCalculator

class TestMetricsCalculator:
    """Tests for the MetricsCalculator."""
    
    def test_calculate_metrics_with_relevant_results(self, sample_search_results):
        """Test calculating metrics when there are relevant results."""
        relevant_ids = ["code1", "code3"]
        
        metrics = MetricsCalculator.calculate_metrics(sample_search_results, relevant_ids)
        
        # Check MRR
        assert metrics["mrr"] == 1.0  # First result is relevant
        
        # Check Recall@K - with 2 relevant items
        assert metrics["recall_at_1"] == 0.5  # 1 out of 2 relevant results found in top 1
        assert metrics["recall_at_5"] == 1.0  # All 2 out of 2 relevant results found in top 5
        assert metrics["recall_at_10"] == 1.0  # All 2 out of 2 relevant results found in top 10
        
        # Check NDCG@10
        assert metrics["ndcg_at_10"] > 0.0
    
    def test_calculate_metrics_with_no_relevant_results(self, sample_search_results):
        """Test calculating metrics when there are no relevant results."""
        relevant_ids = ["nonexistent"]
        
        metrics = MetricsCalculator.calculate_metrics(sample_search_results, relevant_ids)
        
        # Check MRR
        assert metrics["mrr"] == 0.0  # No relevant results
        
        # Check Recall@K
        assert metrics["recall_at_1"] == 0.0  # No relevant results
        assert metrics["recall_at_5"] == 0.0  # No relevant results
        
        # Check NDCG@10
        assert metrics["ndcg_at_10"] == 0.0  # No relevant results
    
    def test_calculate_metrics_with_some_relevant_results(self, sample_search_results):
        """Test calculating metrics when some results are relevant."""
        relevant_ids = ["code2"]  # Only the second result is relevant
        
        metrics = MetricsCalculator.calculate_metrics(sample_search_results, relevant_ids)
        
        # Check MRR
        assert metrics["mrr"] == 0.5  # Second result is relevant (1/2)
        
        # Check Recall@K - with 1 relevant item
        assert metrics["recall_at_1"] == 0.0  # 0 out of 1 relevant results found in top 1
        assert metrics["recall_at_5"] == 1.0  # 1 out of 1 relevant results found in top 5
        
        # Check NDCG@10
        assert 0.0 < metrics["ndcg_at_10"] < 1.0  # Some but not optimal ranking
    
    def test_calculate_metrics_with_empty_results(self):
        """Test calculating metrics with empty results."""
        metrics = MetricsCalculator.calculate_metrics([], ["code1"])
        
        # Check all metrics are zero with empty results
        assert metrics["mrr"] == 0.0
        assert metrics["recall_at_1"] == 0.0
        assert metrics["ndcg_at_10"] == 0.0
    
    def test_calculate_metrics_with_empty_relevant_ids(self, sample_search_results):
        """Test calculating metrics with no relevant ids specified."""
        metrics = MetricsCalculator.calculate_metrics(sample_search_results, [])
        
        # Check all metrics are zero with no relevant items
        assert metrics["mrr"] == 0.0
        assert metrics["recall_at_1"] == 0.0
        assert metrics["ndcg_at_10"] == 0.0