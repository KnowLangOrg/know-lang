import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect
from unittest.mock import AsyncMock, patch

# Assuming your FastAPI app instance is in knowlang.api.main
from knowlang.api.main import app
from knowlang.chat_bot.chat_graph import ChatStatus, StreamingChatResult
# ServerSentChatEvent import removed as per subtask instructions for this test modification
# from knowlang.api.chat.router import ServerSentChatEvent

# Configure all tests in this file to be treated as asyncio
pytestmark = pytest.mark.asyncio

client = TestClient(app)

# Placeholder for test functions to be added later
# Example:
# async def test_example():
#     assert True


async def test_websocket_chat_stream_success_unmocked():
    received_messages = []
    completed_received = False
    # The TestClient's websocket_connect is synchronous in its entry/exit,
    # but the operations within it (send_text, receive_text) are async.
    # Pytest's asyncio marker handles running this test function in an event loop.
    with client.websocket_connect("/api/v1/chat/ws/chat/stream") as websocket:
        await websocket.send_text("Hello")
        try:
            while True:  # Loop until WebSocketDisconnect
                data = await websocket.receive_text()
                # Assuming StreamingChatResult is the direct model being sent as JSON
                message_obj = StreamingChatResult.model_validate_json(data)
                received_messages.append(message_obj)
                if message_obj.status == ChatStatus.COMPLETE:
                    completed_received = True
                # For this basic test, we might not need to check all intermediate states,
                # just that it completes. The default vector store and model might produce
                # minimal messages (e.g. just START and COMPLETE).
        except WebSocketDisconnect:
            # This is expected when the server finishes streaming and closes the connection.
            pass

    assert len(received_messages) > 0, "Should receive at least one message (e.g., START, COMPLETE)"
    assert completed_received, "A message with status COMPLETE should be received"


@patch("knowlang.api.chat.router.stream_chat_progress")
async def test_websocket_chat_stream_mocked(mock_stream_chat_progress):
    mock_results = [
        StreamingChatResult(answer="", status=ChatStatus.STARTING, progress_message="Starting...", retrieved_context=None),
        StreamingChatResult(answer="Partial answer...", status=ChatStatus.ANSWERING, progress_message="Thinking...", retrieved_context=None),
        StreamingChatResult(answer="Complete answer.", status=ChatStatus.COMPLETE, progress_message="Done.", retrieved_context=[])
    ]

    async def mock_generator_func(*args, **kwargs):
        for result in mock_results:
            # Simulate async behavior if needed, though side_effect handles it for generators
            # await asyncio.sleep(0)
            yield result

    mock_stream_chat_progress.side_effect = mock_generator_func

    received_json_messages = []
    test_query = "test query for mocked stream"

    with client.websocket_connect("/api/v1/chat/ws/chat/stream") as websocket:
        await websocket.send_text(test_query)
        try:
            while True:
                data = await websocket.receive_text()
                received_json_messages.append(json.loads(data))
        except WebSocketDisconnect:
            # Expected when the mocked generator finishes and the server closes the connection
            pass

    assert len(received_json_messages) == len(mock_results), \
        f"Expected {len(mock_results)} messages, got {len(received_json_messages)}"

    for i, received_msg_json in enumerate(received_json_messages):
        # The router sends result.model_dump_json(), which by default excludes None values.
        # So, we should compare with model_dump() after converting to JSON and back, or ensure model_dump settings match.
        # For simplicity, we compare the JSON string representation after model_dump_json from the mock result.
        # Alternatively, convert received_msg_json back to StreamingChatResult and compare Pydantic models.
        expected_dict = mock_results[i].model_dump()
        assert received_msg_json == expected_dict, \
            f"Message {i} does not match. Expected: {expected_dict}, Got: {received_msg_json}"

    # Assert that the mocked function was called with the correct query.
    # The other arguments are AppConfig and VectorStore instances, which are harder to check directly
    # without more complex mocking or comparison logic. Checking the query is a good start.
    mock_stream_chat_progress.assert_called_once()
    args, kwargs = mock_stream_chat_progress.call_args
    assert args[0] == test_query
    # print(f"Mock called with: args={args}, kwargs={kwargs}") # For debugging


@patch("knowlang.api.chat.router.stream_chat_progress")
async def test_websocket_chat_stream_error_in_generator(mock_stream_chat_progress):
    class CustomTestError(Exception):
        pass

    simulated_error_message = "Simulated error in stream"
    first_result = StreamingChatResult(
        answer="First normal message.",
        status=ChatStatus.ANSWERING,
        progress_message="Processing...",
        retrieved_context=None
    )

    async def mock_generator_with_error(*args, **kwargs):
        yield first_result
        raise CustomTestError(simulated_error_message)

    mock_stream_chat_progress.side_effect = mock_generator_with_error

    received_raw_messages = []
    test_query_for_error = "test query for error handling"

    with client.websocket_connect("/api/v1/chat/ws/chat/stream") as websocket:
        await websocket.send_text(test_query_for_error)
        try:
            while True:
                data = await websocket.receive_text()
                received_raw_messages.append(data)
        except WebSocketDisconnect:
            # This is expected after the server sends the error message and closes the connection.
            pass

    assert len(received_raw_messages) == 2, \
        f"Expected 2 messages (1 normal, 1 error), got {len(received_raw_messages)}"

    # Validate the first message (normal stream item)
    # This one is sent as a direct StreamingChatResult JSON
    normal_message_obj = StreamingChatResult.model_validate_json(received_raw_messages[0])
    assert normal_message_obj.status == ChatStatus.ANSWERING
    assert normal_message_obj.answer == first_result.answer

    # Validate the second message (error message)
    # As per subtask instructions, expecting a direct StreamingChatResult model for errors.
    error_message_obj = StreamingChatResult.model_validate_json(received_raw_messages[1])

    assert error_message_obj.status == ChatStatus.ERROR

    # StreamingChatResult.error() formats the answer and progress_message
    expected_error_answer = f"Error: {simulated_error_message}"
    expected_error_progress = f"An error occurred: {simulated_error_message}"

    assert error_message_obj.answer == expected_error_answer
    assert error_message_obj.progress_message == expected_error_progress
    assert error_message_obj.retrieved_context == [] # Default from StreamingChatResult.error()

    mock_stream_chat_progress.assert_called_once_with(
        test_query_for_error,
        mock_stream_chat_progress.call_args.args[1], # config
        mock_stream_chat_progress.call_args.args[2]  # vector_store
    )
