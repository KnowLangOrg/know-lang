from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from knowlang.api import ApiModelRegistry
from knowlang.chat_bot import (
    ChatStatus,
    StreamingChatResult,
    stream_chat_progress,
)
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

router = APIRouter()


@ApiModelRegistry.register
class ServerSentChatEvent(BaseModel):
    event: ChatStatus
    data: StreamingChatResult


@router.get("/chat/stream")
async def stream_chat(
    query: str,
):
    """
    Streaming chat endpoint that uses server-sent events (SSE)
    """

    async def event_generator():
        # Process using the core logic from Gradio
        async for result in stream_chat_progress(query):
            yield ServerSentChatEvent(event=result.status, data=result).model_dump()

    return EventSourceResponse(event_generator())


@router.websocket("/ws/chat/stream")
async def websocket_chat_stream(
    websocket: WebSocket,
):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            LOG.info(f"Received query via WebSocket: {query}")
            async for result in stream_chat_progress(query):
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
            LOG.error(
                f"Failed to send error message to client: {send_error}", exc_info=True
            )
    finally:
        try:
            await websocket.close()
            LOG.info("WebSocket connection closed.")
        except Exception as close_error:
            LOG.error(f"Error closing WebSocket: {close_error}", exc_info=True)
