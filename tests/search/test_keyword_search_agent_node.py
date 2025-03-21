from unittest.mock import AsyncMock, Mock, patch
import pytest
from pydantic_graph import GraphRunContext, End

from knowlang.configs.config import AppConfig, LLMConfig
from knowlang.configs.retrieval_config import MultiStageRetrievalConfig, SearchConfig 
from knowlang.core.types import ModelProvider
from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.search_graph.keyword_search_agent_node import KeywordSearchAgentNode, KeywordExtractionResult
from knowlang.search.search_graph.base import SearchState, SearchDeps, SearchOutputs
from knowlang.search.keyword_search import KeywordSearchableStore
from knowlang.search.query import KeywordQuery


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    return AppConfig(
        llm=LLMConfig(
            model_provider=ModelProvider.TESTING,
            model_name="test-model"
        ),
        retrieval=MultiStageRetrievalConfig(
            keyword_search=SearchConfig(
                top_k=10,
                max_retries=2,
                query_refinement=True  # Default to True for backward compatibility
            )
        )
    )


@pytest.fixture
def mock_keyword_store():
    """Create a mock store specifically for keyword search testing"""
    store = Mock(spec=KeywordSearchableStore)
    
    # Set up capabilities
    store.capabilities = {SearchMethodology.KEYWORD}
    store.has_capability = lambda methodology: methodology in store.capabilities
    
    # Mock the search method - we'll configure its return value in each test
    store.search = AsyncMock(return_value=[])  # Default to empty list for safety
    
    # Return our configured mock
    return store


@pytest.fixture
def search_state():
    """Create a search state for testing"""
    return SearchState(query="How does keyword search work?")


@pytest.fixture
def search_deps(mock_config, mock_keyword_store):
    """Create search dependencies for testing"""
    return SearchDeps(
        store=mock_keyword_store,
        config=mock_config
    )


@pytest.fixture
def run_context(search_state, search_deps):
    """Create a graph run context for testing"""
    return GraphRunContext(state=search_state, deps=search_deps)


# Reset the agent instance before each test to avoid cross-test contamination
@pytest.fixture(autouse=True)
def reset_agent_instance():
    KeywordSearchAgentNode._agent_instance = None


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.keyword_search_agent_node.Agent')
async def test_keyword_search_agent_node_success(mock_agent_class, run_context, mock_keyword_store):
    """Test that KeywordSearchAgentNode extracts keywords and performs search successfully"""
    # Explicitly set query_refinement to True
    run_context.deps.config.retrieval.keyword_search.query_refinement = True
    
    node = KeywordSearchAgentNode()
    
    # Mock keyword extraction
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="keyword & search")

    # Configure search to return specific results
    search_results = [
        SearchResult(
            document="def keyword_search(): pass", 
            metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
            score=0.9
        )
    ]
    mock_keyword_store.search.return_value = search_results

    # Run the node
    result = await node.run(run_context)

    # Verify result is an End node with SearchOutputs
    assert isinstance(result, End)
    output = result.data
    assert len(output.search_results) == 1
    assert output.search_results[0].document == "def keyword_search(): pass"
    
    # Verify state was updated correctly
    assert len(run_context.state.refined_queries[SearchMethodology.KEYWORD]) == 1
    assert run_context.state.refined_queries[SearchMethodology.KEYWORD][0] == "keyword & search"
    assert len(run_context.state.search_results) == 1
    
    # Verify agent and search were called correctly
    mock_agent.run.assert_called_once()
    mock_keyword_store.search.assert_called_once()
    
    # Verify the search call parameters
    args, kwargs = mock_keyword_store.search.call_args
    assert kwargs["strategy_name"] == SearchMethodology.KEYWORD
    assert isinstance(kwargs["query"], KeywordQuery)
    assert kwargs["query"].text == "keyword & search"


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.keyword_search_agent_node.Agent')
async def test_keyword_search_agent_node_recursive(mock_agent_class, run_context, mock_keyword_store):
    """Test that KeywordSearchAgentNode calls itself recursively if no results found"""
    # Explicitly set query_refinement to True
    run_context.deps.config.retrieval.keyword_search.query_refinement = True
    
    node = KeywordSearchAgentNode()
    
    # Mock keyword extraction
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="keyword & search")

    # Configure search to return empty results
    mock_keyword_store.search.return_value = []

    # Run the node
    next_node = await node.run(run_context)

    # Verify it's trying again with a new instance
    assert isinstance(next_node, KeywordSearchAgentNode)
    assert next_node.attempts == 1
    assert next_node.previous_query == "keyword & search"
    
    # Now set up search to return results for the second attempt
    mock_keyword_store.search.return_value = [
        SearchResult(
            document="def keyword_search(): pass", 
            metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
            score=0.9
        )
    ]
    
    # Reset the agent mock to give a different response for the second call
    mock_agent.run.reset_mock()
    mock_agent.run.return_value = Mock(data="keyword | search | broader")
    
    # Run the node again
    result = await next_node.run(run_context)
    
    # Verify it returns SearchOutputs with results
    assert isinstance(result, End)
    output = result.data
    assert len(output.search_results) == 1


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.keyword_search_agent_node.Agent')
async def test_keyword_search_agent_node_max_retries(mock_agent_class, run_context):
    """Test that KeywordSearchAgentNode stops recursing after max retries"""
    # Explicitly set query_refinement to True
    run_context.deps.config.retrieval.keyword_search.query_refinement = True
    
    node = KeywordSearchAgentNode(attempts=1, previous_query="first & query")
    
    # Mock keyword extraction
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="second | query")

    # Configure search to return no results
    run_context.deps.store.search.return_value = []

    # Run the node - this should be our second attempt (max_retries=2)
    result = await node.run(run_context)

    # Verify it returns End with empty SearchOutputs
    assert isinstance(result, End)
    output = result.data
    assert isinstance(output, SearchOutputs)
    assert len(output.search_results) == 0


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.keyword_search_agent_node.Agent')
async def test_keyword_search_agent_node_error_handling(mock_agent_class, run_context):
    """Test that KeywordSearchAgentNode handles errors gracefully"""
    node = KeywordSearchAgentNode()
    
    # Mock keyword extraction to raise an error
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(side_effect=Exception("Test error"))
    
    # Make sure search method is not called at all
    run_context.deps.store.search.side_effect = Exception("This should not be called")

    # Run the node and check it returns End with empty SearchOutputs
    result = await node.run(run_context)
    
    assert isinstance(result, End)
    output = result.data
    assert isinstance(output, SearchOutputs)
    assert len(output.search_results) == 0


@pytest.mark.asyncio
async def test_extract_keywords_method(mock_config, run_context):
    """Test the _extract_keywords method directly"""
    with patch('knowlang.search.search_graph.keyword_search_agent_node.Agent') as mock_agent_class:
        node = KeywordSearchAgentNode()
        
        # Mock the agent
        mock_agent = mock_agent_class.return_value
        mock_agent.run = AsyncMock()
        
        # Test with query_refinement=True
        run_context.deps.config.retrieval.keyword_search.query_refinement = True
        
        # Test AND logic detection
        mock_agent.run.return_value = Mock(data="keyword & search & strategy")
        result = await node._extract_keywords(run_context, "test")
        assert isinstance(result, KeywordExtractionResult)
        assert result.logic == "AND"
        assert mock_agent.run.called
        mock_agent.run.reset_mock()
        
        # Test OR logic detection
        mock_agent.run.return_value = Mock(data="keyword | search | strategy")
        result = await node._extract_keywords(run_context, "test")
        assert result.logic == "OR"
        assert mock_agent.run.called
        mock_agent.run.reset_mock()
        
        # Test mixed logic (should default to OR)
        mock_agent.run.return_value = Mock(data="keyword & (search | strategy)")
        result = await node._extract_keywords(run_context, "test")
        assert result.logic == "OR"
        assert mock_agent.run.called
        mock_agent.run.reset_mock()
        
        # Test with query_refinement=False
        run_context.deps.config.retrieval.keyword_search.query_refinement = False
        result = await node._extract_keywords(run_context, "original query")
        assert result.query == "original query"
        assert result.logic == "AND"
        assert not mock_agent.run.called


@pytest.mark.asyncio
async def test_perform_keyword_search_method(mock_keyword_store):
    """Test the _perform_keyword_search method directly"""
    node = KeywordSearchAgentNode()
    
    # Configure search to return specific results
    search_results = [
        SearchResult(
            document="def keyword_search(): pass", 
            metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
            score=0.9
        )
    ]
    mock_keyword_store.search.return_value = search_results
    
    # Call the method directly
    results = await node._perform_keyword_search(
        query="test query",
        vector_store=mock_keyword_store,
        top_k=10
    )
    
    # Verify results
    assert results == search_results
    
    # Verify search called with correct parameters
    mock_keyword_store.search.assert_called_once()
    args, kwargs = mock_keyword_store.search.call_args
    assert kwargs["strategy_name"] == "keyword_search"
    assert isinstance(kwargs["query"], KeywordQuery)
    assert kwargs["query"].text == "test query"
    assert kwargs["query"].top_k == 10


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.keyword_search_agent_node.Agent')
async def test_keyword_search_agent_node_no_refinement(mock_agent_class, run_context, mock_keyword_store):
    """Test that KeywordSearchAgentNode doesn't use LLM when query_refinement is False"""
    # Set query_refinement to False
    run_context.deps.config.retrieval.keyword_search.query_refinement = False
    
    node = KeywordSearchAgentNode()
    
    # Set up mock agent - this should not be called
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    
    # Configure search to return specific results
    search_results = [
        SearchResult(
            document="def keyword_search(): pass", 
            metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
            score=0.9
        )
    ]
    mock_keyword_store.search.return_value = search_results

    # Run the node
    result = await node.run(run_context)

    # Verify result is an End node with SearchOutputs
    assert isinstance(result, End)
    output = result.data
    assert len(output.search_results) == 1
    
    # Verify the original query was used
    assert len(run_context.state.refined_queries[SearchMethodology.KEYWORD]) == 1
    assert run_context.state.refined_queries[SearchMethodology.KEYWORD][0] == run_context.state.query
    
    # Verify agent was NOT called - critical test!
    mock_agent.run.assert_not_called()
    
    # Verify search was called with the right parameters
    args, kwargs = mock_keyword_store.search.call_args
    assert kwargs["strategy_name"] == SearchMethodology.KEYWORD
    assert isinstance(kwargs["query"], KeywordQuery)
    assert kwargs["query"].text == run_context.state.query  # Original query used