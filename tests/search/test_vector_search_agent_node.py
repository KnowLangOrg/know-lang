from unittest.mock import AsyncMock, Mock, patch
import pytest
from pydantic_graph import GraphRunContext, End

from knowlang.configs.config import AppConfig, LLMConfig, EmbeddingConfig
from knowlang.configs.retrieval_config import MultiStageRetrievalConfig, SearchConfig 
from knowlang.core.types import ModelProvider
from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.search_graph.vector_search_agent_node import VectorSearchAgentNode, QueryRefinementResult
from knowlang.search.search_graph.base import SearchState, SearchDeps, SearchOutputs
from knowlang.search.query import VectorQuery
from knowlang.vector_stores.base import VectorStore
from knowlang.models.types import EmbeddingInputType


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    return AppConfig(
        llm=LLMConfig(
            model_provider=ModelProvider.TESTING,
            model_name="test-model"
        ),
        embedding=EmbeddingConfig(
            model_provider=ModelProvider.TESTING,
            model_name="test_embedding_model"
        ),
        retrieval=MultiStageRetrievalConfig(
            vector_search=SearchConfig(
                top_k=10,
                score_threshold=0.5,
                max_retries=2,
                query_refinement=True  # Default to True for backward compatibility
            )
        )
    )


@pytest.fixture
def mock_vector_store():
    """Create a mock store for vector search testing"""
    store = Mock(spec=VectorStore)
    
    # Set up capabilities
    store.capabilities = {SearchMethodology.VECTOR}
    store.has_capability = lambda methodology: methodology in store.capabilities
    
    # Mock the search method
    store.search = AsyncMock(return_value=[])  # Default to empty list
    
    return store


@pytest.fixture
def search_state():
    """Create a search state for testing"""
    return SearchState(query="How does vector search work?")


@pytest.fixture
def search_deps(mock_config, mock_vector_store):
    """Create search dependencies for testing"""
    return SearchDeps(
        store=mock_vector_store,
        config=mock_config
    )


@pytest.fixture
def run_context(search_state, search_deps):
    """Create a graph run context for testing"""
    return GraphRunContext(state=search_state, deps=search_deps)


# Reset the agent instance before each test
@pytest.fixture(autouse=True)
def reset_agent_instance():
    VectorSearchAgentNode._agent_instance = None


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.vector_search_agent_node.Agent')
@patch('knowlang.search.search_graph.vector_search_agent_node.generate_embedding')
async def test_vector_search_agent_node_success(mock_generate_embedding, mock_agent_class, run_context, mock_vector_store):
    """Test that VectorSearchAgentNode refines query, generates embeddings, and performs search successfully"""
    # Explicitly set query_refinement to True
    run_context.deps.config.retrieval.vector_search.query_refinement = True
    
    node = VectorSearchAgentNode()
    
    # Mock query refinement
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="vector embedding similarity search implementation")

    # Mock embedding generation
    mock_generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]  # Sample embedding
    
    # Configure search to return results
    search_results = [
        SearchResult(
            document="def vector_search(): pass", 
            metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
            score=0.9
        )
    ]
    mock_vector_store.search.return_value = search_results

    # Run the node
    result = await node.run(run_context)

    # Verify result is End node with SearchOutputs
    assert isinstance(result, End)
    output = result.data
    assert isinstance(output, SearchOutputs)
    assert len(output.search_results) == 1
    assert output.search_results[0].document == "def vector_search(): pass"
    
    # Verify state was updated correctly
    assert len(run_context.state.refined_queries[SearchMethodology.VECTOR]) == 1
    assert run_context.state.refined_queries[SearchMethodology.VECTOR][0] == "vector embedding similarity search implementation"
    assert len(run_context.state.search_results) == 1
    
    # Verify agent was called
    mock_agent.run.assert_called_once()
    
    # Verify generate_embedding was called correctly
    mock_generate_embedding.assert_called_with(
        input="vector embedding similarity search implementation",
        config=run_context.deps.config.embedding,
        input_type=EmbeddingInputType.QUERY
    )
    
    # Verify search was called correctly
    mock_vector_store.search.assert_called_once()
    args, kwargs = mock_vector_store.search.call_args
    assert kwargs["strategy_name"] == SearchMethodology.VECTOR
    assert isinstance(kwargs["query"], VectorQuery)
    assert kwargs["query"].embedding == [0.1, 0.2, 0.3, 0.4]
    assert kwargs["query"].top_k == 10
    assert kwargs["query"].score_threshold == 0.5


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.vector_search_agent_node.Agent')
@patch('knowlang.search.search_graph.vector_search_agent_node.generate_embedding')
async def test_vector_search_agent_node_recursive(mock_generate_embedding, mock_agent_class, run_context, mock_vector_store):
    """Test that VectorSearchAgentNode calls itself recursively if no results found"""
    # Explicitly set query_refinement to True
    run_context.deps.config.retrieval.vector_search.query_refinement = True
    
    node = VectorSearchAgentNode()
    
    # Mock query refinement
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="vector search implementation")

    # Mock embedding generation
    mock_generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]  # Sample embedding
    
    # Configure search to return no results initially
    mock_vector_store.search.return_value = []

    # Run the node
    next_node = await node.run(run_context)

    # Verify it's trying again with a new instance
    assert isinstance(next_node, VectorSearchAgentNode)
    assert next_node.attempts == 1
    assert next_node.previous_query == "vector search implementation"
    
    # Now set up search to return results for the second attempt
    mock_vector_store.search.return_value = [
        SearchResult(
            document="def vector_search(): pass", 
            metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
            score=0.9
        )
    ]
    
    # Configure agent for the second attempt
    mock_agent.run.reset_mock()
    mock_agent.run.return_value = Mock(data="vector embedding similarity search broader")
    
    # Run the node again
    result = await next_node.run(run_context)
    
    # Verify it returns SearchOutputs with results
    assert isinstance(result, End)
    output = result.data
    assert isinstance(output, SearchOutputs)
    assert len(output.search_results) == 1


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.vector_search_agent_node.Agent')
@patch('knowlang.search.search_graph.vector_search_agent_node.generate_embedding')
async def test_vector_search_agent_node_max_retries(mock_generate_embedding, mock_agent_class, run_context):
    """Test that VectorSearchAgentNode stops recursing after max retries"""
    # Explicitly set query_refinement to True
    run_context.deps.config.retrieval.vector_search.query_refinement = True
    
    node = VectorSearchAgentNode(attempts=1, previous_query="first refined query")
    
    # Mock query refinement
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="second broader query")

    # Mock embedding generation
    mock_generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]  # Sample embedding
    
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
@patch('knowlang.search.search_graph.vector_search_agent_node.Agent')
@patch('knowlang.search.search_graph.vector_search_agent_node.generate_embedding')
async def test_vector_search_agent_node_embedding_error(mock_generate_embedding, mock_agent_class, run_context):
    """Test that VectorSearchAgentNode handles embedding generation errors"""
    node = VectorSearchAgentNode()
    
    # Mock query refinement
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    mock_agent.run.return_value = Mock(data="vector search implementation")

    # Mock embedding generation to fail
    mock_generate_embedding.side_effect = RuntimeError("Embedding error")
    
    # Run the node
    result = await node.run(run_context)
    
    # Verify it returns End with empty SearchOutputs
    assert isinstance(result, End)
    output = result.data
    assert isinstance(output, SearchOutputs)
    assert len(output.search_results) == 0


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.vector_search_agent_node.Agent')
async def test_vector_search_agent_node_general_error(mock_agent_class, run_context):
    """Test that VectorSearchAgentNode handles general errors gracefully"""
    node = VectorSearchAgentNode()
    
    # Mock agent to raise an error
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(side_effect=Exception("Test error"))
    
    # Run the node and check it returns End with empty SearchOutputs
    result = await node.run(run_context)
    
    assert isinstance(result, End)
    output = result.data
    assert isinstance(output, SearchOutputs)
    assert len(output.search_results) == 0


@pytest.mark.asyncio
async def test_refine_query_method(run_context):
    """Test the _refine_query method directly with both refinement settings"""
    with patch('knowlang.search.search_graph.vector_search_agent_node.Agent') as mock_agent_class:
        node = VectorSearchAgentNode()
        
        # Mock the agent
        mock_agent = mock_agent_class.return_value
        mock_agent.run = AsyncMock()
        mock_agent.run.return_value = Mock(data="refined query text")
        
        # Test with query_refinement=True
        run_context.deps.config.retrieval.vector_search.query_refinement = True
        
        # Test initial refinement
        result = await node._refine_query(run_context, "original query")
        assert isinstance(result, QueryRefinementResult)
        assert result.refined_query == "refined query text"
        assert result.explanation == "Initial query refinement"
        assert mock_agent.run.called
        mock_agent.run.reset_mock()
        
        # Test refinement when too few results
        node.previous_query = "previous query"
        result = await node._refine_query(run_context, "original query", too_few_results=True)
        assert result.explanation == "Generating broader query due to few results"
        assert mock_agent.run.called
        mock_agent.run.reset_mock()
        
        # Test with query_refinement=False
        run_context.deps.config.retrieval.vector_search.query_refinement = False
        result = await node._refine_query(run_context, "original query")
        assert result.refined_query == "original query"
        assert result.explanation == "Query refinement disabled"
        assert not mock_agent.run.called


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.vector_search_agent_node.generate_embedding')
async def test_generate_embeddings_method(mock_generate_embedding, run_context):
    """Test the _generate_embeddings method directly"""
    node = VectorSearchAgentNode()
    
    # Mock generate_embedding
    mock_generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
    
    # Call the method
    embedding = await node._generate_embeddings(run_context, "test query")
    
    # Verify result
    assert embedding == [0.1, 0.2, 0.3, 0.4]
    
    # Verify generate_embedding was called correctly
    mock_generate_embedding.assert_called_with(
        input="test query",
        config=run_context.deps.config.embedding,
        input_type=EmbeddingInputType.QUERY
    )
    
    # Test handling of empty embeddings
    mock_generate_embedding.return_value = []
    embedding = await node._generate_embeddings(run_context, "test query")
    assert embedding == []
    
    # Test error handling
    mock_generate_embedding.side_effect = RuntimeError("Test error")
    embedding = await node._generate_embeddings(run_context, "test query")
    assert embedding == []


@pytest.mark.asyncio
async def test_perform_vector_search_method(mock_vector_store):
    """Test the _perform_vector_search method directly"""
    node = VectorSearchAgentNode()
    
    # Configure search to return specific results
    search_results = [
        SearchResult(
            document="def vector_search(): pass", 
            metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
            score=0.9
        )
    ]
    mock_vector_store.search.return_value = search_results
    
    # Call the method directly
    results = await node._perform_vector_search(
        embedding=[0.1, 0.2, 0.3, 0.4],
        vector_store=mock_vector_store,
        top_k=10,
        score_threshold=0.5
    )
    
    # Verify results
    assert results == search_results
    
    # Verify search called with correct parameters
    mock_vector_store.search.assert_called_once()
    args, kwargs = mock_vector_store.search.call_args
    assert kwargs["strategy_name"] == SearchMethodology.VECTOR
    assert isinstance(kwargs["query"], VectorQuery)
    assert kwargs["query"].embedding == [0.1, 0.2, 0.3, 0.4]
    assert kwargs["query"].top_k == 10
    assert kwargs["query"].score_threshold == 0.5
    
    # Test error handling
    mock_vector_store.search.side_effect = Exception("Search error")
    results = await node._perform_vector_search(
        embedding=[0.1, 0.2, 0.3, 0.4],
        vector_store=mock_vector_store,
        top_k=10,
        score_threshold=0.5
    )
    assert results == []


@pytest.mark.asyncio
@patch('knowlang.search.search_graph.vector_search_agent_node.Agent')
@patch('knowlang.search.search_graph.vector_search_agent_node.generate_embedding')
async def test_vector_search_agent_node_no_refinement(mock_generate_embedding, mock_agent_class, run_context, mock_vector_store):
    """Test that VectorSearchAgentNode doesn't use LLM when query_refinement is False"""
    # Set query_refinement to False
    run_context.deps.config.retrieval.vector_search.query_refinement = False
    
    node = VectorSearchAgentNode()
    
    # Set up mock agent - this should not be called
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock()
    
    # Mock embedding generation
    mock_generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]  # Sample embedding
    
    # Configure search to return results
    search_results = [
        SearchResult(
            document="def vector_search(): pass", 
            metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2}, 
            score=0.9
        )
    ]
    mock_vector_store.search.return_value = search_results

    # Run the node
    result = await node.run(run_context)

    # Verify result is End node with SearchOutputs
    assert isinstance(result, End)
    output = result.data
    assert len(output.search_results) == 1
    
    # Verify the original query was used
    assert len(run_context.state.refined_queries[SearchMethodology.VECTOR]) == 1
    assert run_context.state.refined_queries[SearchMethodology.VECTOR][0] == run_context.state.query
    
    # Verify agent was NOT called - critical test!
    mock_agent.run.assert_not_called()
    
    # Verify generate_embedding was called with the original query
    mock_generate_embedding.assert_called_with(
        input=run_context.state.query,
        config=run_context.deps.config.embedding,
        input_type=EmbeddingInputType.QUERY
    )