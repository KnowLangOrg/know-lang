from unittest.mock import AsyncMock, Mock, patch
import pytest
from pydantic_graph import End, GraphRunContext

from knowlang.search import SearchResult
from knowlang.chat_bot.chat_graph import (
    AnswerQuestionNode,
    ChatGraphDeps,
    ChatGraphState,
    ChatResult,
    ChatStatus,
    RetrievalNode,
    stream_chat_progress,
)


@pytest.mark.asyncio
@patch("knowlang.chat_bot.chat_graph.Agent")
async def test_answer_question_node(
    mock_agent_class, mock_config, populated_mock_store
):
    """Test that AnswerQuestionNode generates appropriate answers with the new state structure"""
    node = AnswerQuestionNode()
    state = ChatGraphState(
        original_question="test question",
        retrieved_context=[
            SearchResult(
                document="def test_function(): pass",
                metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2},
                score=0.9,
            )
        ],
    )
    deps = ChatGraphDeps(chat_config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    mock_answer = Mock()
    mock_answer.output = "This is the answer based on the code context."
    mock_agent = mock_agent_class.return_value
    mock_agent.run = AsyncMock(return_value=mock_answer)

    result = await node.run(ctx)
    assert result.data.answer == "This is the answer based on the code context."
    assert result.data.retrieved_context == state.retrieved_context
    mock_agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_answer_question_node_no_context(mock_config, mock_vector_store):
    """Test AnswerQuestionNode behavior when no context is found with new state structure"""
    node = AnswerQuestionNode()
    state = ChatGraphState(
        original_question="test question",
        retrieved_context=[],  # Empty list instead of RetrievedContext
    )
    deps = ChatGraphDeps(chat_config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    result = await node.run(ctx)
    assert "couldn't find any relevant code context" in result.data.answer.lower()
    assert result.data.retrieved_context is None


class MockGraphRun:
    """Mock for GraphRun to simulate the behavior of graph.iter context manager"""

    def __init__(self, nodes_sequence):
        self.nodes_sequence = nodes_sequence
        self.current_index = 0
        self.history = []
        self.result = None
        self.next_node = nodes_sequence[0] if nodes_sequence else None

    async def next(self, node):
        """Simulate GraphRun.next method behavior"""
        self.history.append(node)
        self.current_index += 1
        if self.current_index < len(self.nodes_sequence):
            self.next_node = self.nodes_sequence[self.current_index]
            return self.next_node
        else:
            # End of sequence - should not happen in normal testing as we
            # expect an End node to terminate the iteration
            return None


@pytest.mark.asyncio
@patch("knowlang.chat_bot.chat_graph.chat_graph")
async def test_stream_chat_progress_success(
    mock_chat_graph, mock_config, populated_mock_store
):
    """Test successful streaming chat progress with proper graph.iter mocking"""
    # Mock the span context manager

    # Define the sequence of nodes to be returned
    retrieval_node = RetrievalNode()
    answer_node = AnswerQuestionNode()
    end_node = End(
        ChatResult(
            answer="Test answer",
            retrieved_context=[
                SearchResult(
                    document="def test_function(): pass",
                    metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2},
                    score=0.9,
                )
            ],
        )
    )

    # Create mock GraphRun instance
    mock_graph_run = MockGraphRun([retrieval_node, answer_node, end_node])

    # Set up the context manager mock for graph.iter
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_graph_run)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    # Configure chat_graph.iter to return our context manager mock
    mock_chat_graph.iter.return_value = mock_context_manager

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
    ):
        results.append(result)

    # Verify the sequence of streaming results
    assert len(results) >= 3  # Should have starting, retrieving, answering

    # Verify initial status
    assert results[0].status == ChatStatus.STARTING
    assert "Processing question: test question" in results[0].progress_message

    # Verify retrieval status
    assert results[1].status == ChatStatus.RETRIEVING
    assert "Searching codebase" in results[1].progress_message

    # Verify answering status
    assert results[2].status == ChatStatus.ANSWERING

    # Verify final result
    assert results[-1].status == ChatStatus.COMPLETE
    assert results[-1].answer == "Test answer"
    assert results[-1].retrieved_context is not None

    # Verify the chat_graph.iter call
    mock_chat_graph.iter.assert_called_once()
    mock_context_manager.__aenter__.assert_called_once()
    mock_context_manager.__aexit__.assert_called_once()


@pytest.mark.asyncio
@patch("knowlang.chat_bot.chat_graph.chat_graph")
async def test_stream_chat_progress_node_error(
    mock_chat_graph, mock_vector_store, mock_config
):
    """Test streaming chat progress when a node execution fails"""
    # Create a mock context manager for graph.iter that will raise an exception
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(
        side_effect=Exception("Test node error")
    )
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    # Configure chat_graph.iter to return our context manager mock
    mock_chat_graph.iter.return_value = mock_context_manager

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
    ):
        results.append(result)

    # Verify error handling
    assert len(results) == 2  # Initial status and error

    # Verify initial status
    assert results[0].status == ChatStatus.STARTING

    # Verify error status
    assert results[-1].status == ChatStatus.ERROR
    assert "Test node error" in results[-1].progress_message


@pytest.mark.asyncio
@patch("knowlang.chat_bot.chat_graph.chat_graph")
async def test_stream_chat_progress_invalid_node(
    mock_chat_graph, mock_vector_store, mock_config
):
    """Test streaming chat progress when an invalid node type is returned"""

    # Create mock GraphRun that will return an invalid node type
    retrieval_node = RetrievalNode()
    invalid_node = "invalid node"  # Not a valid BaseNode instance

    mock_graph_run = MockGraphRun([retrieval_node])
    mock_graph_run.next = AsyncMock(return_value=invalid_node)

    # Set up the context manager mock for graph.iter
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_graph_run)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    # Configure chat_graph.iter to return our context manager mock
    mock_chat_graph.iter.return_value = mock_context_manager

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
    ):
        results.append(result)

    # Verify error handling
    assert len(results) == 3  # Should have initial status, node status, and error
    assert results[-1].status == ChatStatus.ERROR
    assert "Invalid node type" in results[-1].progress_message


@pytest.mark.asyncio
@patch("knowlang.chat_bot.chat_graph.chat_graph")
async def test_stream_chat_progress_general_error(
    mock_chat_graph, mock_vector_store, mock_config
):
    """Test streaming chat progress when a general error occurs"""

    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.side_effect = Exception("Test general error")

    mock_chat_graph.iter.return_value = mock_context_manager

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
    ):
        results.append(result)

    # Verify error handling
    assert len(results) == 2  # Should have initial status and error
    assert results[-1].status == ChatStatus.ERROR
    assert "Test general error" in results[-1].progress_message
