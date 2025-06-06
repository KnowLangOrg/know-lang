from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from knowlang.api import ApiModelRegistry
from knowlang.chat_bot import (
    ChatStatus,
    StreamingChatResult,
    stream_chat_progress,
)
from knowlang.configs import AppConfig
from knowlang.utils import FancyLogger
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = FancyLogger(__name__)

router = APIRouter()
config = AppConfig()


@ApiModelRegistry.register
class ServerSentChatEvent(BaseModel):
    event: ChatStatus
    data: StreamingChatResult


# Dependency to get config
async def get_app_config():
    return config


# Dependency to get vector store
async def get_vector_store(config: AppConfig = Depends(get_app_config)):
    return VectorStoreFactory.get(config)


@router.get("/chat/stream")
async def stream_chat(
    query: str,
    config: AppConfig = Depends(get_app_config),
    vector_store=Depends(get_vector_store),
):
    """
    Streaming chat endpoint that uses server-sent events (SSE)
    """

    async def event_generator():
        # Process using the core logic from Gradio
        async for result in stream_chat_progress(query, vector_store, config):
            yield ServerSentChatEvent(event=result.status, data=result).model_dump()

    return EventSourceResponse(event_generator())


# To test this WebSocket endpoint for streaming chat:
#
# 1. Using `websocat`:
#    - Install websocat (e.g., `apt-get install websocat` or from https://github.com/vi/websocat).
#    - Connect to the endpoint (adjust URI if your server runs on a different host/port):
#      websocat ws://localhost:8000/api/v1/chat/ws/chat/stream
#    - Once connected, type your query (e.g., "What is retrieval augmented generation?") and press Enter.
#    - The server will stream back JSON responses, each representing a part of the chat answer.
#
# 2. Using a Python client (requires `websockets` library: `pip install websockets`):
#    ```python
#    import asyncio
#    import websockets
#    import json
#
#    async def test_chat_websocket():
#        # Adjust URI if your FastAPI app is served under a different path or port
#        uri = "ws://localhost:8000/api/v1/chat/ws/chat/stream"
#        try:
#            async with websockets.connect(uri) as websocket:
#                query = input("Enter your chat query: ")
#                await websocket.send(query) # Sending query as plain text
#
#                print("Waiting for responses...")
#                while True:
#                    try:
#                        response = await websocket.recv()
#                        # Assuming the response is a JSON string from StreamingChatResult
#                        print(f"<<< {json.loads(response)}")
#                    except websockets.exceptions.ConnectionClosedOK:
#                        print("Connection closed normally.")
#                        break
#                    except websockets.exceptions.ConnectionClosedError as e:
#                        print(f"Connection closed with error: {e}")
#                        break
#                    except Exception as e:
#                        print(f"An error occurred while receiving: {e}")
#                        break
#        except ConnectionRefusedError:
#            print(f"Connection to {uri} refused. Ensure the server is running.")
#        except Exception as e:
#            print(f"An overall error occurred: {e}")
#
#    if __name__ == "__main__":
#        # In a Jupyter notebook, you might need to run this with nest_asyncio
#        # import nest_asyncio
#        # nest_asyncio.apply()
#        asyncio.run(test_chat_websocket())
#    ```
@router.websocket("/ws/chat/stream")
async def websocket_chat_stream(
    websocket: WebSocket,
    config: AppConfig = Depends(get_app_config),
    vector_store=Depends(get_vector_store),
):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            LOG.info(f"Received query via WebSocket: {query}")
            async for result in stream_chat_progress(query, vector_store, config):
                await websocket.send_text(result.model_dump_json())
    except WebSocketDisconnect:
        LOG.info("Client disconnected from WebSocket chat stream.")
    except Exception as e:
        LOG.error(f"Error in WebSocket chat stream: {e}", exc_info=True)
        # Attempt to send an error message to the client
        try:
            await websocket.send_text(
                ServerSentChatEvent(
                    event=ChatStatus.ERROR, data=StreamingChatResult.error(str(e))
                ).model_dump_json()
            )
        except Exception as send_error:
            LOG.error(f"Failed to send error message to client: {send_error}", exc_info=True)
    finally:
        try:
            await websocket.close()
            LOG.info("WebSocket connection closed.")
        except Exception as close_error:
            LOG.error(f"Error closing WebSocket: {close_error}", exc_info=True)
