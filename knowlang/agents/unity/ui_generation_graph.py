from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from grpc_stub.unity.ui_generation_pb2 import (
    UIGenerationStreamResponse,
    UIGenerationStatus,
    UIGENERATION_STATUS_GENERATING_UXML,
    UIGENERATION_STATUS_GENERATING_USS,
    UIGENERATION_STATUS_GENERATING_CSHARP,
    UIGENERATION_STATUS_COMPLETE,
    UIGENERATION_STATUS_ERROR,
    UIGENERATION_STATUS_UNSPECIFIED,
)

from knowlang.configs.chat_config import ChatConfig
from knowlang.utils import FancyLogger
from .nodes.base import UIGenerationState, UIGenerationDeps, UIGenerationResult
from .nodes.uxml_generator import UXMLGeneratorNode
from .nodes.uss_generator import USSGeneratorNode
from .nodes.csharp_generator import CSharpGeneratorNode

LOG = FancyLogger(__name__)


def create_stream_response(
    node: BaseNode, state: UIGenerationState
) -> UIGenerationStreamResponse:
    """Create a streaming response from a node's current state"""
    response = UIGenerationStreamResponse(
        ui_description=state.ui_description,
        is_complete=False,
    )

    if isinstance(node, UXMLGeneratorNode):
        response.status = UIGENERATION_STATUS_GENERATING_UXML
        response.progress_message = (
            f"Generating UXML markup for: '{state.ui_description}'"
        )
    elif isinstance(node, USSGeneratorNode):
        response.uxml_content = state.uxml_content or ""
        response.status = UIGENERATION_STATUS_GENERATING_USS
        response.progress_message = (
            f"Generating USS styles for: '{state.ui_description}'"
        )
    elif isinstance(node, CSharpGeneratorNode):
        response.uxml_content = state.uxml_content or ""
        response.uss_content = state.uss_content or ""
        response.status = UIGENERATION_STATUS_GENERATING_CSHARP
        response.progress_message = (
            f"Generating C# script for: '{state.ui_description}'"
        )
    else:
        response.status = UIGENERATION_STATUS_UNSPECIFIED
        response.progress_message = "Starting UI generation..."

    return response


def create_error_response(
    error: Exception, state: UIGenerationState
) -> UIGenerationStreamResponse:
    """Create a streaming error response"""
    error_msg = f"Error during UI generation: {str(error)}"
    LOG.error(error_msg)
    response = UIGenerationStreamResponse(
        ui_description=state.ui_description,
        status=UIGENERATION_STATUS_ERROR,
        progress_message=error_msg,
        error_message=error_msg,
        is_complete=True,
    )
    return response


def create_complete_response(state: UIGenerationState) -> UIGenerationStreamResponse:
    """Create a streaming completion response"""
    response = UIGenerationStreamResponse(
        ui_description=state.ui_description,
        uxml_content=state.uxml_content or "",
        uss_content=state.uss_content or "",
        csharp_content=state.csharp_content or "",
        status=UIGENERATION_STATUS_COMPLETE,
        progress_message=f"Successfully generated UI for: '{state.ui_description}'",
        is_complete=True,
    )
    return response


# Create the graph
ui_generation_graph = Graph(
    nodes=[UXMLGeneratorNode, USSGeneratorNode, CSharpGeneratorNode]
)


async def stream_ui_generation_progress(
    ui_description: str,
    chat_config: Optional[ChatConfig] = None,
    unity_project_path: Optional[str] = None,
    ui_style_preferences: Optional[dict] = None,
) -> AsyncGenerator[UIGenerationStreamResponse, None]:
    """
    Stream UI generation progress through the graph.
    This is the main entry point for Unity UI generation.
    """
    if not chat_config:
        chat_config = ChatConfig()

    # Create initial state
    state = UIGenerationState(
        ui_description=ui_description,
        unity_project_path=unity_project_path,
        ui_style_preferences=ui_style_preferences or {},
        chat_config=chat_config,
    )

    # Create dependencies
    deps = UIGenerationDeps()

    try:
        # Run the graph and stream progress
        async for node in ui_generation_graph.run(state, deps):
            if node is not None and node is not End:
                # Yield progress update
                yield create_stream_response(node, state)

        # Yield final result
        yield create_complete_response(state)

    except Exception as e:
        # Yield error result
        yield create_error_response(e, state)
        raise
