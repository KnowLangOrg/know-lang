from __future__ import annotations

from typing import AsyncGenerator, List, Optional, Union

from pydantic import BaseModel
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from knowlang.configs.chat_config import ChatConfig
from knowlang.utils import FancyLogger
from .nodes.base import UIGenerationState, UIGenerationDeps, UIGenerationResult, UIGenerationStatus
from .nodes.uxml_generator import UXMLGeneratorNode
from .nodes.uss_generator import USSGeneratorNode
from .nodes.csharp_generator import CSharpGeneratorNode

LOG = FancyLogger(__name__)


class StreamingUIGenerationResult(BaseModel):
    """Extended UI generation result with streaming information"""

    uxml_content: Optional[str] = None
    uss_content: Optional[str] = None
    csharp_content: Optional[str] = None
    ui_description: str
    status: UIGenerationStatus
    progress_message: str
    error_message: Optional[str] = None

    @classmethod
    def from_node(cls, node: BaseNode, state: UIGenerationState) -> StreamingUIGenerationResult:
        """Create a StreamingUIGenerationResult from a node's current state"""
        if isinstance(node, UXMLGeneratorNode):
            return cls(
                ui_description=state.ui_description,
                status=UIGenerationStatus.GENERATING_UXML,
                progress_message=f"Generating UXML markup for: '{state.ui_description}'",
            )
        elif isinstance(node, USSGeneratorNode):
            return cls(
                ui_description=state.ui_description,
                uxml_content=state.uxml_content,
                status=UIGenerationStatus.GENERATING_USS,
                progress_message="Generating USS styling based on UXML structure",
            )
        elif isinstance(node, CSharpGeneratorNode):
            return cls(
                ui_description=state.ui_description,
                uxml_content=state.uxml_content,
                uss_content=state.uss_content,
                status=UIGenerationStatus.GENERATING_CSHARP,
                progress_message="Generating C# boilerplate code for event binding",
            )
        else:
            return cls(
                ui_description=state.ui_description,
                status=UIGenerationStatus.ERROR,
                progress_message=f"Unknown node type: {type(node).__name__}",
            )

    @classmethod
    def complete(cls, result: UIGenerationResult) -> StreamingUIGenerationResult:
        """Create a completed StreamingUIGenerationResult"""
        return cls(
            uxml_content=result.uxml_content,
            uss_content=result.uss_content,
            csharp_content=result.csharp_content,
            ui_description=result.ui_description,
            status=UIGenerationStatus.COMPLETE,
            progress_message="UI generation complete",
        )

    @classmethod
    def error(cls, ui_description: str, error_msg: str) -> StreamingUIGenerationResult:
        """Create an error StreamingUIGenerationResult"""
        return cls(
            ui_description=ui_description,
            status=UIGenerationStatus.ERROR,
            progress_message=f"An error occurred: {error_msg}",
            error_message=error_msg,
        )


# Create the graph
ui_generation_graph = Graph(nodes=[UXMLGeneratorNode, USSGeneratorNode, CSharpGeneratorNode])


async def stream_ui_generation_progress(
    ui_description: str,
    chat_config: Optional[ChatConfig] = None,
    unity_project_path: Optional[str] = None,
    ui_style_preferences: Optional[dict] = None,
) -> AsyncGenerator[StreamingUIGenerationResult, None]:
    """
    Stream UI generation progress through the graph.
    This is the main entry point for Unity UI generation.
    """
    state = UIGenerationState(ui_description=ui_description)
    
    if chat_config is None:
        chat_config = ChatConfig()
        
    deps = UIGenerationDeps(
        chat_config=chat_config,
        unity_project_path=unity_project_path,
        ui_style_preferences=ui_style_preferences
    )

    start_node = UXMLGeneratorNode()

    try:
        # Initial status
        yield StreamingUIGenerationResult(
            ui_description=ui_description,
            status=UIGenerationStatus.STARTING,
            progress_message=f"Starting UI generation for: {ui_description}",
        )

        graph_run_cm = ui_generation_graph.iter(
            start_node, state=state, deps=deps, infer_name=False
        )

        # we have to manually enter the context manager since this function itself is a AsyncGenerator
        graph_run = await graph_run_cm.__aenter__()
        next_node = graph_run.next_node

        while True:
            # Yield current node's status before processing
            yield StreamingUIGenerationResult.from_node(next_node, state)

            # Process the current node
            next_node = await graph_run.next(next_node)

            if isinstance(next_node, End):
                result: UIGenerationResult = next_node.data
                # Yield final result and break
                yield StreamingUIGenerationResult.complete(result)
                break
            elif not isinstance(next_node, BaseNode):
                # If the next node is not a valid BaseNode, raise an error
                raise TypeError(f"Invalid node type: {type(next_node)}")

    except Exception as e:
        LOG.error(f"Error in stream_ui_generation_progress: {e}")
        yield StreamingUIGenerationResult.error(ui_description, str(e))
    finally:
        await graph_run_cm.__aexit__(None, None, None)
        return


async def generate_ui_sync(
    ui_description: str,
    chat_config: Optional[ChatConfig] = None,
    unity_project_path: Optional[str] = None,
    ui_style_preferences: Optional[dict] = None,
) -> UIGenerationResult:
    """
    Synchronous version of UI generation that returns the final result directly.
    Useful for non-streaming use cases.
    """
    async for result in stream_ui_generation_progress(
        ui_description, chat_config, unity_project_path, ui_style_preferences
    ):
        if result.status == UIGenerationStatus.COMPLETE:
            return UIGenerationResult(
                uxml_content=result.uxml_content,
                uss_content=result.uss_content,
                csharp_content=result.csharp_content,
                ui_description=result.ui_description,
            )
        elif result.status == UIGenerationStatus.ERROR:
            raise RuntimeError(f"UI generation failed: {result.error_message}")
    
    raise RuntimeError("UI generation did not complete successfully") 