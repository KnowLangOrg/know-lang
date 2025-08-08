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
from knowlang.agents.unity.nodes.base import UIGenerationState, UIGenerationDeps, UIGenerationResult
from knowlang.agents.unity.nodes.uxml_generator import UXMLGeneratorNode
from knowlang.agents.unity.nodes.uss_generator import USSGeneratorNode
from knowlang.agents.unity.nodes.csharp_generator import CSharpGeneratorNode

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
    )

    # Create dependencies
    deps = UIGenerationDeps(
        chat_config=chat_config
    )
    
    start_node = UXMLGeneratorNode()

    try:
        graph_run_context_manager = ui_generation_graph.iter(
            start_node, state=state, deps=deps, infer_name=False
        )
        graph_run = await graph_run_context_manager.__aenter__()
        next_node = graph_run.next_node

        # Run the graph and stream progress
        while True:
            yield create_stream_response(next_node, state)

            next_node = await graph_run.next(next_node)
            if isinstance(next_node, End):
                yield create_complete_response(state)
                break


    except Exception as e:
        yield create_error_response(e, state)
        raise
    finally:
        await graph_run_context_manager.__aexit__(None, None, None)
        return
