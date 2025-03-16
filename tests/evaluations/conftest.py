import json
import pytest
from pathlib import Path
from unittest import mock

import numpy as np

from knowlang.evaluations.base import QueryCodePair, SearchConfiguration
from knowlang.configs import AppConfig, DBConfig, EmbeddingConfig
from knowlang.search.base import SearchResult

@pytest.fixture
def temp_dir(tmp_path):
    """Fixture to provide a temporary directory."""
    return tmp_path

@pytest.fixture
def sample_query_code_pairs():
    """Fixture to provide sample query-code pairs for testing."""
    return [
        QueryCodePair(
            query_id="query1",
            query="How to sort a list in Python",
            code_id="code1",
            code="def sort_list(lst):\n    return sorted(lst)",
            language="python",
            is_relevant=True,
            metadata={"repo": "sample_repo", "path": "sample_path"}
        ),
        QueryCodePair(
            query_id="query2",
            query="How to find elements in a list",
            code_id="code2",
            code="def find_element(lst, element):\n    return element in lst",
            language="python",
            is_relevant=True,
            metadata={"repo": "sample_repo", "path": "sample_path"}
        )
    ]

@pytest.fixture
def sample_search_configuration():
    """Fixture to provide a sample search configuration."""
    return SearchConfiguration(
        name="test_config",
        description="Test configuration",
        keyword_search_enabled=True,
        vector_search_enabled=True,
        reranking_enabled=True,
        keyword_search_threshold=0.1,
        vector_search_threshold=0.6,
        reranker_threshold=0.5,
        keyword_search_top_k=10,
        vector_search_top_k=10,
        reranker_top_k=5
    )

@pytest.fixture
def mock_app_config():
    """Fixture to provide a mock app configuration."""
    config = mock.MagicMock(spec=AppConfig)
    config.db = DBConfig()
    config.embedding = EmbeddingConfig()
    
    return config

@pytest.fixture
def sample_search_results():
    """Fixture to provide sample search results."""
    return [
        SearchResult(
            document="def sort_list(lst):\n    return sorted(lst)",
            metadata={"id": "code1", "language": "python"},
            score=0.95
        ),
        SearchResult(
            document="def find_element(lst, element):\n    return element in lst",
            metadata={"id": "code2", "language": "python"},
            score=0.85
        ),
        SearchResult(
            document="def append_element(lst, element):\n    lst.append(element)",
            metadata={"id": "code3", "language": "python"},
            score=0.75
        )
    ]