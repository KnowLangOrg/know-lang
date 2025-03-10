# test_chat_graph.py
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
    stream_chat_progress
)
from knowlang.search.search_graph.keyword_search_agent_node import KeywordSearchAgentNode


@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.Agent')
async def test_answer_question_node(mock_agent_class, mock_config, populated_mock_store):
    """Test that AnswerQuestionNode generates appropriate answers with the new state structure"""
    node = AnswerQuestionNode()
    state = ChatGraphState(
        original_question="test question",
        retrieved_context=[
            SearchResult(document="def test_function(): pass", 
                       metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2},
                       score=0.9)
        ]
    )
    deps = ChatGraphDeps(vector_store=populated_mock_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    mock_answer = Mock()
    mock_answer.data = "This is the answer based on the code context."
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
        retrieved_context=[]  # Empty list instead of RetrievedContext
    )
    deps = ChatGraphDeps(vector_store=mock_vector_store, config=mock_config)
    ctx = GraphRunContext(state=state, deps=deps)

    result = await node.run(ctx)
    assert "couldn't find any relevant code context" in result.data.answer.lower()
    assert result.data.retrieved_context is None


@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.logfire')
@patch('knowlang.chat_bot.chat_graph.chat_graph')
async def test_stream_chat_progress_success(
    mock_chat_graph, 
    mock_logfire, 
    mock_config, 
    populated_mock_store
):
    """Test successful streaming chat progress with updated node flow"""
    # Mock the span context manager
    mock_span = Mock()
    mock_logfire.span.return_value.__enter__.return_value = mock_span
    
    # Set up mock chat graph behavior
    mock_chat_graph.next = AsyncMock()
    mock_chat_graph.next.side_effect = [
        # Only KeywordSearchAgentNode and AnswerQuestionNode now
        AnswerQuestionNode(),   # Move to answer node
        End(ChatResult(         # Finally return the result
            answer="Test answer",
            retrieved_context=[
                SearchResult(document="def test_function(): pass", 
                          metadata={"file_path": "test1.py", "start_line": 1, "end_line": 2},
                          score=0.9)
            ]
        ))
    ]

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
        vector_store=populated_mock_store,
        config=mock_config
    ):
        results.append(result)

    # Verify the sequence of streaming results
    assert len(results) >= 3  # Should have starting, retrieving, complete
    
    # Verify initial status
    assert results[0].status == ChatStatus.STARTING
    assert "Processing question: test question" in results[0].progress_message
    
    # Verify retrieval status
    assert results[1].status == ChatStatus.RETRIEVING
    assert "Searching codebase" in results[1].progress_message
    
    # Verify final result
    assert results[-1].status == ChatStatus.COMPLETE
    assert results[-1].answer == "Test answer"
    assert results[-1].retrieved_context is not None
    
    # Verify graph execution
    assert mock_chat_graph.next.call_count == 2
    mock_span.set_attribute.assert_called_once()


@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.logfire')
@patch('knowlang.chat_bot.chat_graph.chat_graph')
async def test_stream_chat_progress_node_error(mock_chat_graph, mock_logfire, mock_vector_store, mock_config):
    """Test streaming chat progress when a node execution fails"""
    # Mock the span context manager
    mock_span = Mock()
    mock_logfire.span.return_value.__enter__.return_value = mock_span
    
    # Set up mock chat graph to raise an error
    mock_chat_graph.next = AsyncMock()
    mock_chat_graph.next.side_effect = Exception("Test node error")

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
        vector_store=mock_vector_store,
        config=mock_config
    ):
        results.append(result)

    # Verify error handling
    assert len(results) == 3  # Should have initial status, pending first node, and error
    
    # Verify initial status
    assert results[0].status == ChatStatus.STARTING
    
    # Verify error status
    assert results[-1].status == ChatStatus.ERROR
    assert "Test node error" in results[-1].progress_message
    assert not results[-1].retrieved_context


@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.logfire')
@patch('knowlang.chat_bot.chat_graph.chat_graph')
async def test_stream_chat_progress_invalid_node(mock_chat_graph, mock_logfire, mock_vector_store, mock_config):
    """Test streaming chat progress when an invalid node type is returned"""
    # Mock the span context manager
    mock_span = Mock()
    mock_logfire.span.return_value.__enter__.return_value = mock_span
    
    # Set up mock chat graph to return invalid node type
    mock_chat_graph.next = AsyncMock()
    mock_chat_graph.next.return_value = "invalid node"  # Return invalid node type

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
        vector_store=mock_vector_store,
        config=mock_config
    ):
        results.append(result)

    # Verify error handling
    assert len(results) == 3  # Should have initial status, pending first node, and error
    assert results[-1].status == ChatStatus.ERROR
    assert "Invalid node type" in results[-1].progress_message


@pytest.mark.asyncio
@patch('knowlang.chat_bot.chat_graph.logfire')
async def test_stream_chat_progress_general_error(mock_logfire, mock_vector_store, mock_config):
    """Test streaming chat progress when a general error occurs"""
    # Mock the span context manager to raise an error
    mock_logfire.span.side_effect = Exception("Test general error")

    # Collect all streamed results
    results = []
    async for result in stream_chat_progress(
        question="test question",
        vector_store=mock_vector_store,
        config=mock_config
    ):
        results.append(result)

    # Verify error handling
    assert len(results) == 2  # Should have initial status and error
    assert results[-1].status == ChatStatus.ERROR
    assert "Test general error" in results[-1].progress_message