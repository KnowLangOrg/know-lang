from fastapi import APIRouter, Depends
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
