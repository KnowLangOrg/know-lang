import asyncio
import uuid
from concurrent import futures
from typing import AsyncGenerator, Dict

from grpc import aio

from grpc_stub.unity import ui_generation_pb2_grpc
from grpc_stub.unity.ui_generation_pb2 import (
    UIGenerationRequest,
    UIGenerationStreamResponse,
    UIGenerationStatusResponse,
    UIGenerationCancelRequest,
    UIGenerationCancelResponse,
    UIGENERATION_STATUS_COMPLETE,
    UIGENERATION_STATUS_ERROR,
    GetGenerationStatusRequest,
)

from knowlang.configs.chat_config import ChatConfig
from knowlang.utils import FancyLogger
from .ui_generation_graph import stream_ui_generation_progress

LOG = FancyLogger(__name__)


class UIGenerationServicer(ui_generation_pb2_grpc.UIGenerationServiceServicer):
    """gRPC service handler for Unity UI generation"""

    def __init__(self):
        self.active_generations: Dict[str, asyncio.Task] = {}
        self.generation_status: Dict[str, UIGenerationStatusResponse] = {}

    async def GenerateUIStream(
        self, request: UIGenerationRequest, context: aio.ServicerContext
    ) -> AsyncGenerator[UIGenerationStreamResponse, None]:
        """Generate UI with streaming progress updates"""
        request_id = str(uuid.uuid4())

        try:
            # Create chat config from override if provided
            chat_config = None
            if request.chat_config_override:
                chat_config = ChatConfig(**request.chat_config_override)

            # Start the generation process
            async for result in stream_ui_generation_progress(
                ui_description=request.ui_description,
                chat_config=chat_config,
                unity_project_path=request.unity_project_path or None,
                ui_style_preferences=dict(request.ui_style_preferences),
            ):
                # Update status
                status_response = UIGenerationStatusResponse(
                    request_id=request_id,
                    status=result.status,
                    progress_message=result.progress_message,
                    error_message=result.error_message or "",
                    is_complete=result.is_complete,
                )
                self.generation_status[request_id] = status_response

                # Yield the streaming response
                yield result

                # If this was the final response, clean up
                if result.is_complete:
                    break

        except Exception as e:
            LOG.error(f"Error in GenerateUIStream: {e}")
            error_response = UIGenerationStreamResponse(
                status=UIGENERATION_STATUS_ERROR,
                progress_message=f"Error during UI generation: {str(e)}",
                error_message=str(e),
                is_complete=True,
            )
            yield error_response

            # Update status with error
            status_response = UIGenerationStatusResponse(
                request_id=request_id,
                status=UIGENERATION_STATUS_ERROR,
                progress_message=f"Error: {str(e)}",
                error_message=str(e),
                is_complete=True,
            )
            self.generation_status[request_id] = status_response

    async def GetGenerationStatus(
        self, request: GetGenerationStatusRequest, context: aio.ServicerContext
    ) -> UIGenerationStatusResponse:
        """Get the status of a UI generation request"""
        return self.generation_status.get(
            request.request_id,
            UIGenerationStatusResponse(
                request_id=request.request_id,
                status=UIGENERATION_STATUS_ERROR,
                progress_message="Request ID not found",
                is_complete=True,
                error_message=f"No generation found with ID: {request.request_id}",
            ),
        )

    async def CancelGeneration(
        self, request: UIGenerationCancelRequest, context: aio.ServicerContext
    ) -> UIGenerationCancelResponse:
        """Cancel a UI generation request"""
        if request.request_id in self.active_generations:
            task = self.active_generations[request.request_id]
            if not task.done():
                task.cancel()
                return UIGenerationCancelResponse(
                    success=True,
                    message=f"Successfully cancelled generation {request.request_id}",
                )

        return UIGenerationCancelResponse(
            success=False,
            message=f"No active generation found with ID: {request.request_id}",
        )


async def serve(port: int = 50051) -> None:
    """Start the gRPC server"""
    server = aio.server(futures.ThreadPoolExecutor(max_workers=10))
    ui_generation_pb2_grpc.add_UIGenerationServiceServicer_to_server(
        UIGenerationServicer(), server
    )
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    LOG.info(f"Starting gRPC server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        LOG.info("Shutting down gRPC server...")
        await server.stop(5)
        LOG.info("gRPC server shut down successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Unity UI Generation gRPC server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="The port to listen on",
    )

    args = parser.parse_args()

    try:
        asyncio.run(serve(port=args.port))
    except KeyboardInterrupt:
        LOG.info("Server stopped by user")
