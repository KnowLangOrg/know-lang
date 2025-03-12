import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pydantic_graph import GraphRunContext, End

from knowlang.configs.config import AppConfig, LLMConfig, RerankerConfig
from knowlang.configs.retrieval_config import MultiStageRetrievalConfig, SearchConfig
from knowlang.core.types import ModelProvider
from knowlang.search.base import SearchResult 
from knowlang.search.search_graph.base import SearchState, SearchDeps, SearchOutputs
from knowlang.search.search_graph.graph import FirstStageNode, RerankerNode, search_graph


@pytest.fixture
def app_config():
    """Create a test app configuration."""
    config = AppConfig(
        llm=LLMConfig(
            model_provider=ModelProvider.TESTING,
            model_name="test-model"
        ),
        retrieval=MultiStageRetrievalConfig(
            keyword_search=SearchConfig(
                enabled=True,
                top_k=5,
                max_retries=1
            ),
            vector_search=SearchConfig(
                enabled=True,
                top_k=5,
                max_retries=1
            )
        ),
        reranker=RerankerConfig(
            enabled=True,
            model_name="microsoft/graphcodebert-base",
            model_provider=ModelProvider.GRAPH_CODE_BERT,
            top_k=3,
            relevance_threshold=0.5
        )
    )
    return config


@pytest.fixture
def mock_search_store():
    """Create a mock searchable store."""
    store = MagicMock()
    store.search = AsyncMock()
    return store


@pytest.fixture
def search_state():
    """Create a search state for testing."""
    return SearchState(query="How does the search work?")


@pytest.fixture
def search_deps(app_config, mock_search_store):
    """Create search dependencies for testing."""
    return SearchDeps(
        config=app_config,
        store=mock_search_store
    )


@pytest.fixture
def run_context(search_state, search_deps):
    """Create a graph run context for testing."""
    return GraphRunContext(state=search_state, deps=search_deps)


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing."""
    return [
        SearchResult(
            id="kw1",
            document="def keyword_search(): pass",
            metadata={"path": "search.py", "type": "function"},
            score=0.8
        ),
        SearchResult(
            id="vec1",
            document="class VectorSearch: pass",
            metadata={"path": "vector.py", "type": "class"},
            score=0.7
        )
    ]


@pytest.mark.asyncio
@patch("knowlang.search.search_graph.graph.Graph.run")
async def test_first_stage_node_success(mock_graph_run, run_context, sample_search_results):
    """Test FirstStageNode with successful search results."""
    # Set up mock to return some results for keyword search and vector search
    mock_graph_run.return_value = None  # We'll directly modify state instead
    
    # Configure initial state before running the node
    run_context.state.search_results = sample_search_results

    # Create and run the node
    node = FirstStageNode()
    result = await node.run(run_context)
    
    # Verify node returned RerankerNode 
    assert isinstance(result, RerankerNode)
    
    # Verify Graph.run was called twice (once for keyword search, once for vector search)
    assert mock_graph_run.call_count == 2


@pytest.mark.asyncio
@patch("knowlang.search.search_graph.graph.Graph.run")
async def test_first_stage_node_no_results(mock_graph_run, run_context):
    """Test FirstStageNode when no search results are found."""
    # Set up mock to return no results 
    mock_graph_run.return_value = None
    run_context.state.search_results = []  # Empty search results
    
    # Create and run the node
    node = FirstStageNode()
    result = await node.run(run_context)
    
    # Verify node returned End with empty search results
    assert isinstance(result, End)
    assert isinstance(result.data, SearchOutputs)
    assert len(result.data.search_results) == 0


@pytest.mark.asyncio
@patch("knowlang.search.search_graph.graph.Graph.run")
async def test_first_stage_node_exception(mock_graph_run, run_context):
    """Test FirstStageNode handles exceptions gracefully."""
    # Set up mock to raise an exception
    mock_graph_run.side_effect = Exception("Test exception")
    
    # Create and run the node
    node = FirstStageNode()
    result = await node.run(run_context)
    
    # Verify node returned End with empty search results
    assert isinstance(result, End)
    assert isinstance(result.data, SearchOutputs)
    assert len(result.data.search_results) == 0


@pytest.mark.asyncio
@patch("knowlang.search.reranking.GraphCodeBertReranker")
async def test_reranker_node_success(mock_reranker_class, run_context, sample_search_results):
    """Test RerankerNode with successful reranking."""
    # Configure initial state
    run_context.state.search_results = sample_search_results
    
    # Configure mock reranker
    mock_reranker_instance = mock_reranker_class.return_value
    reranked_results = [
        SearchResult(
            id="reranked1",
            document="Reranked code",
            metadata={"path": "reranked.py"},
            score=0.95
        )
    ]
    mock_reranker_instance.rerank = AsyncMock(return_value=reranked_results)
    
    # Create and run the node
    node = RerankerNode()
    result = await node.run(run_context)
    
    # Verify result
    assert isinstance(result, End)
    assert result.data.search_results == reranked_results
    
    # Verify reranker was called
    mock_reranker_instance.rerank.assert_called_once_with(
        query=run_context.state.query,
        raw_search_results=sample_search_results
    )


@pytest.mark.asyncio
async def test_reranker_node_no_results(run_context):
    """Test RerankerNode when no search results are available."""
    # Configure empty search results
    run_context.state.search_results = []
    
    # Create and run the node
    node = RerankerNode()
    result = await node.run(run_context)
    
    # Verify result
    assert isinstance(result, End)
    assert len(result.data.search_results) == 0


@pytest.mark.asyncio
@patch("knowlang.search.reranking.GraphCodeBertReranker")
async def test_reranker_node_disabled(mock_reranker_class, run_context, sample_search_results):
    """Test RerankerNode when reranking is disabled."""
    # Configure initial state
    run_context.state.search_results = sample_search_results
    
    # Disable reranker
    run_context.deps.config.reranker.enabled = False
    
    # Create and run the node
    node = RerankerNode()
    result = await node.run(run_context)
    
    # Verify original results returned
    assert isinstance(result, End)
    assert result.data.search_results == sample_search_results
    
    # Verify reranker was not instantiated
    mock_reranker_class.assert_not_called()


@pytest.mark.asyncio
@patch("knowlang.search.reranking.GraphCodeBertReranker")
async def test_reranker_node_exception(mock_reranker_class, run_context, sample_search_results):
    """Test RerankerNode handles exceptions gracefully."""
    # Configure initial state
    run_context.state.search_results = sample_search_results
    
    # Configure mock reranker to raise exception
    mock_reranker_instance = mock_reranker_class.return_value
    mock_reranker_instance.rerank = AsyncMock(side_effect=Exception("Test exception"))
    
    # Create and run the node
    node = RerankerNode()
    result = await node.run(run_context)
    
    # Verify original results returned
    assert isinstance(result, End)
    assert result.data.search_results == sample_search_results