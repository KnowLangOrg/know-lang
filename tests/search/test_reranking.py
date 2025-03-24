from unittest.mock import MagicMock, patch, call
import pytest

from knowlang.configs import RerankerConfig
from knowlang.core.types import ModelProvider
from knowlang.search.base import SearchResult
from knowlang.search.reranking import KnowLangReranker


@pytest.fixture
def reranker_config():
    """Create a test reranker configuration."""
    return RerankerConfig(
        enabled=True,
        model_name="microsoft/graphcodebert-base",
        model_provider=ModelProvider.GRAPH_CODE_BERT,
        top_k=3,
        relevance_threshold=0.5
    )


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing."""
    return [
        SearchResult(
            id="result1",
            document="def search_code(query): pass",
            metadata={"path": "search.py", "type": "function"},
            score=0.7
        ),
        SearchResult(
            id="result2",
            document="class CodeSearcher: def __init__(self): pass",
            metadata={"path": "searcher.py", "type": "class"},
            score=0.6
        ),
        SearchResult(
            id="result3",
            document="# This file contains search utilities",
            metadata={"path": "utils.py", "type": "comment"},
            score=0.5
        ),
        SearchResult(
            id="result4",
            document="def irrelevant_function(): pass",
            metadata={"path": "other.py", "type": "function"},
            score=0.4
        )
    ]


@patch("knowlang.models.graph_code_bert.calculate_similarity")
def test_rerank_successful(mock_calculate_scores : MagicMock, reranker_config, sample_search_results):
    """Test successful reranking of search results."""
    # Configure mock to return some scores
    mock_calculate_scores.side_effect = [0.95, 0.85, 0.65, 0.45]
    
    # Create reranker
    reranker = KnowLangReranker(reranker_config)
    
    # Run reranker
    query = "how to search code"
    reranked_results = reranker.rerank(query, sample_search_results)
    
    # Verify results
    assert len(reranked_results) == 3  # top_k=3
    assert reranked_results[0].score == 0.95
    assert reranked_results[1].score == 0.85
    assert reranked_results[2].score == 0.65
    
    # Verify function calls
    mock_calculate_scores.assert_has_calls(call(query, result.document) for result in sample_search_results)



@patch("knowlang.models.graph_code_bert.calculate_similarity")
def test_reranker_threshold_filtering(mock_calculate_scores : MagicMock, reranker_config, sample_search_results):
    """Test that reranker filters results below the relevance threshold."""
    # Configure mock to return scores, with some below threshold
    mock_calculate_scores.side_effect = [0.95, 0.85, 0.45, 0.35]  # Last two below threshold (0.5)
    
    # Create reranker
    reranker = KnowLangReranker(reranker_config)
    
    # Run reranker
    reranked_results = reranker.rerank("search query", sample_search_results)
    
    # Verify only results above threshold are returned
    assert len(reranked_results) == 2
    assert reranked_results[0].score == 0.95
    assert reranked_results[1].score == 0.85


def test_reranker_disabled(reranker_config, sample_search_results):
    """Test that reranker returns original results when disabled."""
    # Disable reranker
    reranker_config.enabled = False
    
    # Create reranker
    reranker = KnowLangReranker(reranker_config)
    
    # Run reranker
    reranked_results = reranker.rerank("search query", sample_search_results)
    
    # Verify original results returned unchanged
    assert reranked_results == sample_search_results


def test_reranker_empty_results(reranker_config):
    """Test reranker handles empty result list."""
    # Create reranker
    reranker = KnowLangReranker(reranker_config)
    
    # Run reranker with empty list
    reranked_results = reranker.rerank("search query", [])
    
    # Verify empty list returned
    assert reranked_results == []


@patch("knowlang.models.graph_code_bert.calculate_similarity")
def test_reranker_result_ordering(mock_calculate_scores : MagicMock, reranker_config, sample_search_results):
    """Test that results are properly ordered by score."""
    # Configure mock to return scores in non-descending order
    mock_calculate_scores.side_effect = [0.75, 0.95, 0.85, 0.65]
    
    # Create reranker
    reranker = KnowLangReranker(reranker_config)
    
    # Run reranker
    reranked_results = reranker.rerank("search query", sample_search_results)
    
    # Verify results are ordered by descending score
    assert reranked_results[0].score == 0.95
    assert reranked_results[1].score == 0.85
    assert reranked_results[2].score == 0.75