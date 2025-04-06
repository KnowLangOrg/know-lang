# __future__ annotations is necessary for the type hints to work in this file
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, List, Optional, Union
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import (
    BaseNode, 
    End, 
    EndStep, 
    Graph, 
    GraphRunContext,
    HistoryStep
)
from rich.console import Console
from knowlang.configs import AppConfig
from knowlang.utils import create_pydantic_model, truncate_chunk, FancyLogger
from knowlang.vector_stores import VectorStore
from knowlang.search import SearchResult
from knowlang.api import ApiModelRegistry
from knowlang.chat_bot.nodes.base import ChatGraphState, ChatGraphDeps, ChatResult
from knowlang.search.search_graph.graph import search_graph

LOG = FancyLogger(__name__)
console = Console()

@ApiModelRegistry.register
class ChatStatus(str, Enum):
    """Enum for tracking chat progress status"""
    STARTING = "starting"
    POLISHING = "polishing"
    RETRIEVING = "retrieving"
    ANSWERING = "answering"
    COMPLETE = "complete"
    ERROR = "error"

@ApiModelRegistry.register
class StreamingChatResult(BaseModel):
    """Extended chat result with streaming information"""
    answer: str
    retrieved_context: Optional[List[SearchResult]] = None
    status: ChatStatus
    progress_message: str
    
    @classmethod
    def from_node(cls, node: BaseNode, state: ChatGraphState) -> StreamingChatResult:
        """Create a StreamingChatResult from a node's current state"""
        if isinstance(node, RetrievalNode):
            return cls(
                answer="",
                status=ChatStatus.RETRIEVING,
                progress_message=f"Searching codebase with: '{state.original_question}'"
            )
        elif isinstance(node, AnswerQuestionNode):
            context_msg = f"Found {len(state.retrieved_context)} relevant segments" if state.retrieved_context else "No context found"
            return cls(
                answer="",
                retrieved_context=state.retrieved_context,
                status=ChatStatus.ANSWERING,
                progress_message=f"Generating answer... {context_msg}"
            )
        else:
            return cls(
                answer="",
                status=ChatStatus.ERROR,
                progress_message=f"Unknown node type: {type(node).__name__}"
            )
    
    @classmethod
    def complete(cls, result: ChatResult) -> StreamingChatResult:
        """Create a completed StreamingChatResult"""
        return cls(
            answer=result.answer,
            retrieved_context=result.retrieved_context,
            status=ChatStatus.COMPLETE,
            progress_message="Response complete"
        )
    
    @classmethod
    def error(cls, error_msg: str) -> StreamingChatResult:
        """Create an error StreamingChatResult"""
        return cls(
            answer=f"Error: {error_msg}",
            status=ChatStatus.ERROR,
            progress_message=f"An error occurred: {error_msg}"
        )

@dataclass
class RetrievalNode(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Base node for search operations"""
    async def run(self, ctx: GraphRunContext[ChatGraphState, ChatGraphDeps]) -> Union['AnswerQuestionNode']:
        from knowlang.search.search_graph.base import SearchDeps, SearchState
        from knowlang.search.search_graph.graph import FirstStageNode
        search_graph_result, _history = await search_graph.run(
            start_node=FirstStageNode(),
            state=SearchState(query=ctx.state.original_question),
            deps=SearchDeps(
                store=ctx.deps.vector_store,
                config=ctx.deps.config
        ))
        ctx.state.retrieved_context = search_graph_result.search_results
        return AnswerQuestionNode()



@dataclass
class AnswerQuestionNode(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Node that generates the final answer"""
    system_prompt = """
You are an expert code assistant helping developers understand complex codebases. Follow these rules strictly:

1. ALWAYS answer the user's question - this is your primary task
2. Base your answer ONLY on the provided code context, not on general knowledge
3. When referencing code:
   - ALWAYS use the format [description]("file_path: line_range") to link to code
   - Example: [incremental update]("src/knowlang/search.py: 12-16")
   - Quote relevant code snippets briefly after your reference only if necessary
   - Avoid quoting large code blocks
   - Explain why this code is relevant to the question
4. If you cannot find sufficient context to answer fully:
   - Clearly state what's missing
   - Explain what additional information would help
5. Focus on accuracy over comprehensiveness:
   - If you're unsure about part of your answer, explicitly say so
   - Better to acknowledge limitations than make assumptions

Remember: Your primary goal is answering the user's specific question, not explaining the entire codebase."""

    async def run(self, ctx: GraphRunContext[ChatGraphState, ChatGraphDeps]) -> End[ChatResult]:
        answer_agent = Agent(
            create_pydantic_model(
                model_provider=ctx.deps.config.llm.model_provider,
                model_name=ctx.deps.config.llm.model_name
            ),
            system_prompt=self.system_prompt
        )
        
        if not ctx.state.retrieved_context:
            return End(ChatResult(
                answer="I couldn't find any relevant code context for your question. "
                      "Could you please rephrase or be more specific?",
                retrieved_context=None,
            ))

        context = ctx.state.retrieved_context
        for single_context in context:
            chunk = truncate_chunk(single_context.document, ctx.deps.config.chat.max_length_per_chunk)

        prompt = f"""
Question: {ctx.state.original_question}

Relevant Code Context:
{context}

Provide a focused answer to the question based on the provided context.

Important: Stay focused on answering the specific question asked.
        """
        
        try:
            result = await answer_agent.run(prompt)
            return End(ChatResult(
                answer=result.data,
                retrieved_context=context,
            ))
        except Exception as e:
            LOG.error(f"Error generating answer: {e}")
            return End(ChatResult(
                answer="I encountered an error processing your question. Please try again.",
                retrieved_context=context,
            ))

# Create the graph
chat_graph = Graph(
    nodes=[RetrievalNode, AnswerQuestionNode]
)

async def process_chat(
    question: str,
    vector_store: VectorStore,
    config: AppConfig
) -> ChatResult:
    """
    Process a chat question through the graph.
    This is the main entry point for chat processing.
    """
    state = ChatGraphState(original_question=question)
    deps = ChatGraphDeps(vector_store=vector_store, config=config)
    
    try:
        result, _history = await chat_graph.run(
            RetrievalNode(),
            state=state,
            deps=deps
        )
    except Exception as e:
        LOG.error(f"Error processing chat in graph: {e}")
        console.print_exception()
        
        result = ChatResult(
            answer="I encountered an error processing your question. Please try again."
        )
    finally:
        return result
    
async def stream_chat_progress(
    question: str,
    vector_store: VectorStore,
    config: AppConfig
) -> AsyncGenerator[StreamingChatResult, None]:
    """
    Stream chat progress through the graph.
    This is the main entry point for chat processing.
    """
    state = ChatGraphState(original_question=question)
    deps = ChatGraphDeps(vector_store=vector_store, config=config)
    
    start_node = RetrievalNode()
    history: list[HistoryStep[ChatGraphState, ChatResult]] = []

    try:
        # Initial status
        yield StreamingChatResult(
            answer="",
            status=ChatStatus.STARTING,
            progress_message=f"Processing question: {question}"
        )

        with logfire.span(
            '{graph_name} run {start=}',
            graph_name='RAG_chat_graph',
            start=start_node,
        ) as run_span:
            current_node = start_node
            
            while True:
                # Yield current node's status before processing
                yield StreamingChatResult.from_node(current_node, state)
                
                try:
                    # Process the current node
                    next_node = await chat_graph.next(current_node, history, state=state, deps=deps, infer_name=False)
                    
                    if isinstance(next_node, End):
                        result: ChatResult = next_node.data
                        history.append(EndStep(result=next_node))
                        run_span.set_attribute('history', history)
                        # Yield final result
                        yield StreamingChatResult.complete(result)
                        return
                    elif isinstance(next_node, BaseNode):
                        current_node = next_node
                    else:
                        raise ValueError(f"Invalid node type: {type(next_node)}")
                        
                except Exception as node_error:
                    LOG.error(f"Error in node {current_node.__class__.__name__}: {node_error}")
                    yield StreamingChatResult.error(str(node_error))
                    return
                    
    except Exception as e:
        LOG.error(f"Error in stream_chat_progress: {e}")
        yield StreamingChatResult.error(str(e))
        return

    