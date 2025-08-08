import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from grpc_stub.unity.ui_generation_pb2 import (
    UIGenerationStreamResponse,
)

from knowlang.configs.chat_config import ChatConfig
from knowlang.utils import FancyLogger
from knowlang.agents.unity.ui_generation_graph import stream_ui_generation_progress
from knowlang.api import ApiModelRegistry

LOG = FancyLogger(__name__)

router = APIRouter()


@ApiModelRegistry.register
class ServerSentUIGenerationEvent(BaseModel):
    event: int  # UIGenerationStatus enum value
    data: dict  # UIGenerationStreamResponse as dict


@router.get("/unity/ui/stream")
async def stream_ui_generation(
    ui_description: str,
):
    """
    Streaming UI generation endpoint that uses server-sent events (SSE)
    For debugging purposes only
    """

    async def event_generator():
        # Process using the core UI generation logic
        async for result in stream_ui_generation_progress(ui_description):
            # Convert protobuf to dict for JSON serialization
            result_dict = {
                "uxml_content": result.uxml_content or "",
                "uss_content": result.uss_content or "",
                "csharp_content": result.csharp_content or "",
                "ui_description": result.ui_description,
                "status": result.status,
                "progress_message": result.progress_message,
                "error_message": result.error_message or "",
                "is_complete": result.is_complete,
            }

            yield ServerSentUIGenerationEvent(
                event=result.status, data=result_dict
            ).model_dump()

    return EventSourceResponse(event_generator())


@router.websocket("/ws/unity/ui/stream")
async def websocket_ui_generation_stream(
    websocket: WebSocket,
):
    """
    WebSocket UI generation endpoint for Unity client communication
    Supports generation requests and cancellation via WebSocket messages
    """
    await websocket.accept()
    LOG.info("Unity UI generation WebSocket connected")

    try:
        while True:
            # Receive UI generation request or control message
            message = await websocket.receive_text()
            LOG.info(
                f"Received UI generation request via WebSocket: {message[:100]}..."
            )

            # Handle different message types
            if message.startswith("CANCEL:"):
                # Handle cancellation request
                request_id = message.replace("CANCEL:", "").strip()
                LOG.info(f"Received cancellation request for: {request_id}")

                # Send cancellation confirmation
                cancel_response = UIGenerationStreamResponse(
                    ui_description="",
                    status=1,  # UIGenerationStatus.Error (using as cancelled status)
                    progress_message=f"Generation cancelled: {request_id}",
                    error_message="User cancelled generation",
                    is_complete=True,
                )
                await websocket.send_bytes(cancel_response.SerializeToString())
                continue

            # Parse UI generation request from JSON or direct text
            try:
                import json

                request_data = json.loads(message)
                ui_description = request_data.get("ui_description", message)
                chat_config_override = request_data.get("chat_config_override", {})
            except (json.JSONDecodeError, AttributeError):
                # Treat as direct UI description text
                ui_description = message
                chat_config_override = {}

            if not ui_description.strip():
                error_response = UIGenerationStreamResponse(
                    ui_description="",
                    status=1,  # UIGenerationStatus.Error
                    progress_message="UI description cannot be empty",
                    error_message="Empty UI description provided",
                    is_complete=True,
                )
                await websocket.send_bytes(error_response.SerializeToString())
                continue

            # Create chat config from override if provided
            chat_config = None
            if chat_config_override:
                chat_config = ChatConfig(**chat_config_override)

            # Generate unique request ID for this generation
            request_id = str(uuid.uuid4())
            LOG.info(f"Starting UI generation with ID: {request_id}")

            try:
                # Stream UI generation progress
                async for result in stream_ui_generation_progress(
                    ui_description=ui_description,
                    chat_config=chat_config,
                ):
                    if websocket.client_state.name != "CONNECTED":
                        LOG.info(
                            f"WebSocket disconnected during generation {request_id}"
                        )
                        break

                    # Send progress update as protobuf binary
                    await websocket.send_bytes(result.SerializeToString())

                    # If generation is complete, break and wait for next request
                    if result.is_complete:
                        LOG.info(f"UI generation completed for request {request_id}")
                        break

            except Exception as e:
                LOG.error(f"Error in UI generation {request_id}: {e}", exc_info=True)

                # Send error response
                error_response = UIGenerationStreamResponse(
                    ui_description=ui_description,
                    status=1,  # UIGenerationStatus.Error
                    progress_message=f"Error during UI generation: {str(e)}",
                    error_message=str(e),
                    is_complete=True,
                )
                await websocket.send_bytes(error_response.SerializeToString())

    except WebSocketDisconnect:
        LOG.info("Unity UI generation WebSocket disconnected")
    except Exception as e:
        LOG.error(f"Error in WebSocket UI generation stream: {e}", exc_info=True)
        # Attempt to send an error message to the client
        try:
            error_response = UIGenerationStreamResponse(
                ui_description="",
                status=1,  # UIGenerationStatus.Error
                progress_message=f"WebSocket error: {str(e)}",
                error_message=str(e),
                is_complete=True,
            )
            await websocket.send_bytes(error_response.SerializeToString())
        except Exception as send_error:
            LOG.error(
                f"Failed to send error message to Unity client: {send_error}",
                exc_info=True,
            )
