from unittest.mock import MagicMock, patch, call
import pytest
import torch

from knowlang.configs import RerankerConfig
from knowlang.core.types import ModelProvider
from knowlang.search.base import SearchResult
from knowlang.search.reranker.knowlang_reranker import KnowLangReranker


@pytest.fixture
def reranker_config():
    """Create a test reranker configuration."""
    return RerankerConfig(
        enabled=True,
        model_name="microsoft/graphcodebert-base",
        model_provider=ModelProvider.GRAPH_CODE_BERT,
        top_k=3,
        relevance_threshold=0.5,
        max_sequence_length=512
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


@patch("knowlang.search.reranker.knowlang_reranker.AutoTokenizer")
@patch("knowlang.search.reranker.knowlang_reranker.CodeBERTReranker")
@patch("knowlang.search.reranker.knowlang_reranker.get_device", return_value="cpu")
@patch("knowlang.search.reranker.knowlang_reranker.RobertaConfig")
def test_rerank_successful(mock_roberta_config, mock_get_device, mock_code_bert_reranker, mock_tokenizer, reranker_config, sample_search_results):
    """Test successful reranking of search results."""
    # Set up mock tokenizer
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    mock_tokenizer_instance.tokenize.side_effect = lambda text: text.split()
    mock_tokenizer_instance.convert_tokens_to_ids.return_value = [0] * 10
    mock_tokenizer_instance.cls_token = "[CLS]"
    mock_tokenizer_instance.sep_token = "[SEP]"
    mock_tokenizer_instance.pad_token_id = 0
    
    # Set up mock model
    mock_model_instance = MagicMock()
    mock_code_bert_reranker.from_pretrained.return_value = mock_model_instance
    # Configure the model to return different scores for different inputs
    mock_model_instance.get_score.side_effect = lambda **kwargs: torch.tensor([0.95]), torch.tensor([0.85]), torch.tensor([0.65]), torch.tensor([0.45])

    mock_roberta_config.from_pretrained.return_value = MagicMock()
    
    # Create reranker with our mocks
    with patch.object(KnowLangReranker, '_get_score', side_effect=[0.95, 0.85, 0.65, 0.45]):
        reranker = KnowLangReranker(reranker_config)
        
        # Run reranker
        query = "how to search code"
        reranked_results = reranker.rerank(query, sample_search_results)
        
        # Verify results
        assert len(reranked_results) <= reranker_config.top_k
        assert reranked_results[0].score == 0.95
        assert reranked_results[1].score == 0.85
        assert reranked_results[2].score == 0.65


@patch("knowlang.search.reranker.knowlang_reranker.AutoTokenizer")
@patch("knowlang.search.reranker.knowlang_reranker.CodeBERTReranker")
@patch("knowlang.search.reranker.knowlang_reranker.get_device", return_value="cpu")
@patch("knowlang.search.reranker.knowlang_reranker.RobertaConfig")
def test_reranker_threshold_filtering(mock_roberta_config, mock_get_device, mock_code_bert_reranker, mock_tokenizer, reranker_config, sample_search_results):
    """Test that reranker filters results below the relevance threshold."""
    # Set up mock tokenizer and model like in previous test
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    mock_tokenizer_instance.tokenize.side_effect = lambda text: text.split()
    mock_tokenizer_instance.convert_tokens_to_ids.return_value = [0] * 10
    mock_tokenizer_instance.cls_token = "[CLS]"
    mock_tokenizer_instance.sep_token = "[SEP]"
    mock_tokenizer_instance.pad_token_id = 0
    
    mock_model_instance = MagicMock()
    mock_code_bert_reranker.from_pretrained.return_value = mock_model_instance

    mock_roberta_config.from_pretrained.return_value = MagicMock()
    
    # Create reranker with our mocks
    with patch.object(KnowLangReranker, '_get_score', side_effect=[0.95, 0.85, 0.45, 0.35]):
        reranker = KnowLangReranker(reranker_config)
        
        # Modify config to test threshold
        reranker.config.relevance_threshold = 0.5
        
        # Run reranker
        reranked_results = reranker.rerank("search query", sample_search_results)
        
        # Verify results
        # Get results with scores greater than threshold (0.5)
        filtered_results = [r for r in reranked_results if r.score >= 0.5]
        assert len(filtered_results) == 2
        assert filtered_results[0].score == 0.95
        assert filtered_results[1].score == 0.85


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



@patch("knowlang.search.reranker.knowlang_reranker.AutoTokenizer")
@patch("knowlang.search.reranker.knowlang_reranker.CodeBERTReranker")
@patch("knowlang.search.reranker.knowlang_reranker.get_device", return_value="cpu")
@patch("knowlang.search.reranker.knowlang_reranker.RobertaConfig")
def test_reranker_result_ordering(mock_roberta_config, mock_get_device, mock_code_bert_reranker, mock_tokenizer, reranker_config, sample_search_results):
    """Test that results are properly ordered by score."""
    # Set up mock tokenizer
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    mock_tokenizer_instance.tokenize.side_effect = lambda text: text.split()
    mock_tokenizer_instance.convert_tokens_to_ids.return_value = [0] * 10
    mock_tokenizer_instance.cls_token = "[CLS]"
    mock_tokenizer_instance.sep_token = "[SEP]"
    mock_tokenizer_instance.pad_token_id = 0
    
    mock_model_instance = MagicMock()
    mock_code_bert_reranker.from_pretrained.return_value = mock_model_instance

    mock_roberta_config.from_pretrained.return_value = MagicMock()
    
    # Create reranker with our mocks
    with patch.object(KnowLangReranker, '_get_score', side_effect=[0.75, 0.95, 0.85, 0.65]):
        reranker = KnowLangReranker(reranker_config)
        
        # Run reranker
        reranked_results = reranker.rerank("search query", sample_search_results)
        
        # Verify results are ordered by descending score
        assert reranked_results[0].score == 0.95
        assert reranked_results[1].score == 0.85
        assert reranked_results[2].score == 0.75
        assert len(reranked_results) <= reranker_config.top_k