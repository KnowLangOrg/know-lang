import asyncio
import uuid
from typing import Dict, Optional
from knowlang.utils import FancyLogger
from knowlang.configs.chat_config import ChatConfig
from .ui_generation_graph import stream_ui_generation_progress
from .grpc_models import (
    UIGenerationRequest,
    UIGenerationResponse,
    UIGenerationStreamResponse,
    UIGenerationStatusResponse,
    UIGenerationCancelRequest,
    UIGenerationCancelResponse,
)

LOG = FancyLogger(__name__)


class UIGenerationService:
    """gRPC service handler for Unity UI generation"""
    
    def __init__(self):
        self.active_generations: Dict[str, asyncio.Task] = {}
        self.generation_results: Dict[str, dict] = {}
    
    async def generate_ui_stream(
        self, request: UIGenerationRequest
    ) -> UIGenerationStreamResponse:
        """Generate UI with streaming progress updates"""
        request_id = str(uuid.uuid4())
        
        try:
            # Create chat config from override if provided
            chat_config = None
            if request.chat_config_override:
                chat_config = ChatConfig(**request.chat_config_override)
            
            # Start the generation process
            generation_task = asyncio.create_task(
                self._run_generation(request_id, request, chat_config)
            )
            self.active_generations[request_id] = generation_task
            
            # Stream the results
            async for result in stream_ui_generation_progress(
                ui_description=request.ui_description,
                chat_config=chat_config,
                unity_project_path=request.unity_project_path,
                ui_style_preferences=request.ui_style_preferences,
            ):
                # Store the latest result
                self.generation_results[request_id] = {
                    "uxml_content": result.uxml_content,
                    "uss_content": result.uss_content,
                    "csharp_content": result.csharp_content,
                    "ui_description": result.ui_description,
                    "status": result.status,
                    "progress_message": result.progress_message,
                    "error_message": result.error_message,
                    "is_complete": result.status == UIGenerationStatus.COMPLETE,
                }
                
                # Yield the streaming response
                yield UIGenerationStreamResponse(
                    uxml_content=result.uxml_content,
                    uss_content=result.uss_content,
                    csharp_content=result.csharp_content,
                    ui_description=result.ui_description,
                    status=result.status,
                    progress_message=result.progress_message,
                    error_message=result.error_message,
                    is_complete=result.status == UIGenerationStatus.COMPLETE,
                )
                
                if result.status in [UIGenerationStatus.COMPLETE, UIGenerationStatus.ERROR]:
                    break
                    
        except Exception as e:
            LOG.error(f"Error in generate_ui_stream: {e}")
            yield UIGenerationStreamResponse(
                ui_description=request.ui_description,
                status=UIGenerationStatus.ERROR,
                progress_message=f"An error occurred: {str(e)}",
                error_message=str(e),
                is_complete=True,
            )
        finally:
            # Clean up
            if request_id in self.active_generations:
                del self.active_generations[request_id]
    
    async def generate_ui_sync(self, request: UIGenerationRequest) -> UIGenerationResponse:
        """Generate UI synchronously (returns final result only)"""
        try:
            # Create chat config from override if provided
            chat_config = None
            if request.chat_config_override:
                chat_config = ChatConfig(**request.chat_config_override)
            
            # Run the generation and collect the final result
            final_result = None
            async for result in stream_ui_generation_progress(
                ui_description=request.ui_description,
                chat_config=chat_config,
                unity_project_path=request.unity_project_path,
                ui_style_preferences=request.ui_style_preferences,
            ):
                if result.status.value == "complete":
                    final_result = result
                    break
                elif result.status == UIGenerationStatus.ERROR:
                    return UIGenerationResponse(
                        success=False,
                        ui_description=request.ui_description,
                        status=UIGenerationStatus.ERROR,
                        progress_message=result.progress_message,
                        error_message=result.error_message,
                    )
            
            if final_result:
                return UIGenerationResponse(
                    success=True,
                    uxml_content=final_result.uxml_content,
                    uss_content=final_result.uss_content,
                    csharp_content=final_result.csharp_content,
                    ui_description=final_result.ui_description,
                    status=UIGenerationStatus.COMPLETE,
                    progress_message="UI generation completed successfully",
                )
            else:
                return UIGenerationResponse(
                    success=False,
                    ui_description=request.ui_description,
                    status=UIGenerationStatus.ERROR,
                    progress_message="UI generation did not complete",
                    error_message="No final result received",
                )
                
        except Exception as e:
            LOG.error(f"Error in generate_ui_sync: {e}")
            return UIGenerationResponse(
                success=False,
                ui_description=request.ui_description,
                status=UIGenerationStatus.ERROR,
                progress_message=f"An error occurred: {str(e)}",
                error_message=str(e),
            )
    
    async def get_generation_status(
        self, request_id: str
    ) -> UIGenerationStatusResponse:
        """Get the status of a UI generation request"""
        if request_id not in self.generation_results:
            return UIGenerationStatusResponse(
                request_id=request_id,
                status=UIGenerationStatus.ERROR,  # Using ERROR for not_found
                progress_message="Request not found",
                is_complete=False,
            )
        
        result = self.generation_results[request_id]
        return UIGenerationStatusResponse(
            request_id=request_id,
            status=result["status"],
            progress_message=result["progress_message"],
            is_complete=result["is_complete"],
            error_message=result.get("error_message"),
        )
    
    async def cancel_generation(
        self, request: UIGenerationCancelRequest
    ) -> UIGenerationCancelResponse:
        """Cancel a UI generation request"""
        request_id = request.request_id
        
        if request_id not in self.active_generations:
            return UIGenerationCancelResponse(
                success=False,
                message="Request not found or already completed",
            )
        
        try:
            # Cancel the task
            task = self.active_generations[request_id]
            task.cancel()
            
            # Wait for cancellation to complete
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Clean up
            del self.active_generations[request_id]
            if request_id in self.generation_results:
                del self.generation_results[request_id]
            
            return UIGenerationCancelResponse(
                success=True,
                message="Generation cancelled successfully",
            )
            
        except Exception as e:
            LOG.error(f"Error cancelling generation {request_id}: {e}")
            return UIGenerationCancelResponse(
                success=False,
                message=f"Error cancelling generation: {str(e)}",
            )
    
    async def _run_generation(
        self, 
        request_id: str, 
        request: UIGenerationRequest, 
        chat_config: Optional[ChatConfig]
    ):
        """Internal method to run the generation process"""
        try:
            async for _ in stream_ui_generation_progress(
                ui_description=request.ui_description,
                chat_config=chat_config,
                unity_project_path=request.unity_project_path,
                ui_style_preferences=request.ui_style_preferences,
            ):
                pass
        except asyncio.CancelledError:
            LOG.info(f"Generation {request_id} was cancelled")
            raise
        except Exception as e:
            LOG.error(f"Error in generation {request_id}: {e}")
            raise 