from unittest.mock import AsyncMock, Mock, patch
import pytest
from pydantic_graph import GraphRunContext

from knowlang.chat_bot.chat_graph import AnswerQuestionNode
from knowlang.chat_bot.nodes.base import ChatGraphDeps, ChatGraphState
from knowlang.configs.config import AppConfig, LLMConfig
from knowlang.core.types import ModelProvider
from knowlang.search import SearchResult
from knowlang.search.search_graph.keyword_search_agent_node import KeywordSearchAgentNode, KeywordExtractionResult
from knowlang.search.keyword_search import KeywordSearchableStore
from knowlang.search.query import KeywordQuery
from knowlang.search.base import SearchMethodology

@pytest.fixture
def mock_config():
    return AppConfig(
        llm = LLMConfig(
            model_provider=ModelProvider.TESTING
        ),
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

# KeywordSearchAgentNode._agent_instance, a class variable, is shared across tests
# This fixture resets it to None after each test to avoid the tests interfering with each other
@pytest.fixture(autouse=True)
def reset_agent_instance():
    KeywordSearchAgentNode._agent_instance = None

@pytest.mark.asyncio
@patch('knowlang.search.keyword_search_agent_node.Agent')
async def test_keyword_search_agent_node_success(mock_agent_class, mock_config, mock_keyword_store):
    """Test that KeywordSearchAgentNode extracts keywords and performs search successfully"""
    node = KeywordSearchAgentNode()
    state = ChatGraphState(
        original_question="How does keyword search work?",
    )
    deps = ChatGraphDeps(vector_store=mock_keyword_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    # Mock keyword extraction
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="keyword & search")

    # Configure search to return specific results
    mock_keyword_store.search.return_value = [
        SearchResult(
            document="def keyword_search(): pass", 
            metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
            score=0.9
        )
    ]

    # Run the node
    next_node = await node.run(ctx)

    # Verify behavior
    assert isinstance(next_node, AnswerQuestionNode)
    assert ctx.state.retrieved_context is not None
    assert len(ctx.state.retrieved_context) == 1
    assert ctx.state.retrieved_context[0].document == "def keyword_search(): pass"
    
    # Verify agent and search were called correctly
    mock_agent.run.assert_called_once()
    mock_keyword_store.search.assert_called_once()
    
    # Verify the search call parameters
    args, kwargs = mock_keyword_store.search.call_args
    assert kwargs["strategy_name"] == "keyword_search"
    assert isinstance(kwargs["query"], KeywordQuery)
    assert kwargs["query"].text == "keyword & search"


@pytest.mark.asyncio
@patch('knowlang.search.keyword_search_agent_node.Agent')
async def test_keyword_search_agent_node_recursive(mock_agent_class, mock_config, mock_keyword_store):
    """Test that KeywordSearchAgentNode calls itself recursively if no results found"""
    node = KeywordSearchAgentNode()
    state = ChatGraphState(
        original_question="How does keyword search work?",
    )
    deps = ChatGraphDeps(vector_store=mock_keyword_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    # Mock keyword extraction
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="keyword & search")

    # Configure search to return empty results
    mock_keyword_store.search.return_value = []

    # Run the node
    next_node = await node.run(ctx)

    # Verify it's trying again with a new instance
    assert isinstance(next_node, KeywordSearchAgentNode)
    assert next_node.attempts == 1
    assert next_node.previous_query == "keyword & search"
    
    # Now set up search to return results for the second attempt
    mock_keyword_store.search.return_value = [
        SearchResult(document="def keyword_search(): pass", 
                   metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
                   score=0.9)
    ]
    
    # Reset the agent mock to give a different response for the second call
    mock_agent.run.reset_mock()
    mock_agent.run.return_value = Mock(data="keyword | search | broader")
    
    # Run the node again
    next_node_2 = await next_node.run(ctx)
    
    # Verify it proceeds to answer node
    assert isinstance(next_node_2, AnswerQuestionNode)
    assert ctx.state.retrieved_context is not None
    assert len(ctx.state.retrieved_context) == 1


@pytest.mark.asyncio
@patch('knowlang.search.keyword_search_agent_node.Agent')
async def test_keyword_search_agent_node_max_retries(mock_agent_class, mock_config, mock_keyword_store):
    """Test that KeywordSearchAgentNode stops recursing after max retries"""
    # Set the max retries to 2 for faster testing
    mock_config.retrieval.keyword_search.max_retries = 2
    
    node = KeywordSearchAgentNode(attempts=1, previous_query="first & query")
    state = ChatGraphState(
        original_question="How does keyword search work?",
    )
    deps = ChatGraphDeps(vector_store=mock_keyword_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    # Mock keyword extraction
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="second | query")

    # Configure search to return no results
    mock_keyword_store.search.return_value = []

    # Run the node - this should be our second attempt
    next_node = await node.run(ctx)

    # Verify it proceeds to answer node despite no results
    # because we hit the max retries
    assert isinstance(next_node, AnswerQuestionNode)
    # Check that retrieved_context is an empty list (not None)
    assert isinstance(ctx.state.retrieved_context, list)
    assert len(ctx.state.retrieved_context) == 0

@pytest.mark.asyncio
@patch('knowlang.search.keyword_search_agent_node.Agent')
async def test_keyword_search_agent_node_error_handling(mock_agent_class, mock_config, mock_keyword_store):
    """Test that KeywordSearchAgentNode handles errors gracefully"""
    node = KeywordSearchAgentNode()
    state = ChatGraphState(
        original_question="How does keyword search work?",
    )
    deps = ChatGraphDeps(vector_store=mock_keyword_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    # Mock keyword extraction to raise an error
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(side_effect=Exception("Test error"))
    
    # Make sure mock_keyword_store.search is not called at all
    mock_keyword_store.search.side_effect = Exception("This should not be called")

    # Run the node and check it still proceeds to the answer node
    next_node = await node.run(ctx)
    
    assert isinstance(next_node, AnswerQuestionNode)
    assert isinstance(ctx.state.retrieved_context, list)
    assert len(ctx.state.retrieved_context) == 0


@pytest.mark.asyncio
async def test_extract_keywords_method(mock_config, mock_keyword_store):
    """Test the _extract_keywords method directly"""
    with patch('knowlang.search.keyword_search_agent_node.Agent') as mock_agent_class:
        node = KeywordSearchAgentNode()
        state = ChatGraphState(
            original_question="How does keyword search work?",
        )
        deps = ChatGraphDeps(vector_store=mock_keyword_store, config=mock_config)
        ctx = GraphRunContext(state=state, deps=deps)
        
        # Mock the agent
        mock_agent = mock_agent_class.return_value
        mock_agent.run = AsyncMock()
        
        # Test AND logic detection
        mock_agent.run.return_value = Mock(data="keyword & search & strategy")
        result = await node._extract_keywords(ctx, "test")
        assert isinstance(result, KeywordExtractionResult)
        assert result.logic == "AND"
        
        # Test OR logic detection
        mock_agent.run.return_value = Mock(data="keyword | search | strategy")
        result = await node._extract_keywords(ctx, "test")
        assert result.logic == "OR"
        
        # Test mixed logic (should default to OR)
        mock_agent.run.return_value = Mock(data="keyword & (search | strategy)")
        result = await node._extract_keywords(ctx, "test")
        assert result.logic == "OR"


@pytest.mark.asyncio
async def test_perform_keyword_search_method(mock_config, mock_keyword_store):
    """Test the _perform_keyword_search method directly"""
    node = KeywordSearchAgentNode()
    
    # Configure search to return specific results
    search_results = [
        SearchResult(document="def keyword_search(): pass", 
                   metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
                   score=0.9)
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