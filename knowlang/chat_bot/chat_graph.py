from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, List, Optional, Union

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from rich.console import Console

from knowlang.api import ApiModelRegistry
from knowlang.assets.config import BaseDomainConfig
from knowlang.chat_bot.nodes.base import ChatGraphDeps, ChatGraphState, ChatResult
from knowlang.configs.chat_config import ChatConfig
from knowlang.search import SearchResult
from knowlang.utils import FancyLogger, create_pydantic_model, truncate_chunk

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
                progress_message=f"Searching codebase with: '{state.original_question}'",
            )
        elif isinstance(node, AnswerQuestionNode):
            context_msg = (
                f"Found {len(state.retrieved_context)} relevant segments"
                if state.retrieved_context
                else "No context found"
            )
            return cls(
                answer="",
                retrieved_context=state.retrieved_context,
                status=ChatStatus.ANSWERING,
                progress_message=f"Generating answer... {context_msg}",
            )
        else:
            return cls(
                answer="",
                status=ChatStatus.ERROR,
                progress_message=f"Unknown node type: {type(node).__name__}",
            )

    @classmethod
    def complete(cls, result: ChatResult) -> StreamingChatResult:
        """Create a completed StreamingChatResult"""
        return cls(
            answer=result.answer,
            retrieved_context=result.retrieved_context,
            status=ChatStatus.COMPLETE,
            progress_message="Response complete",
        )

    @classmethod
    def error(cls, error_msg: str) -> StreamingChatResult:
        """Create an error StreamingChatResult"""
        return cls(
            answer=f"Error: {error_msg}",
            status=ChatStatus.ERROR,
            progress_message=f"An error occurred: {error_msg}",
        )


@dataclass
class RetrievalNode(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Base node for search operations"""

    async def run(
        self, ctx: GraphRunContext[ChatGraphState, ChatGraphDeps]
    ) -> Union["AnswerQuestionNode"]:
        from knowlang.models.embeddings import generate_embedding
        from knowlang.search.query import VectorQuery
        from knowlang.vector_stores.factory import VectorStoreFactory

        for domain in ctx.deps.domain_configs:
            vector_store_config = domain.processor_config.vector_store
            search_config = domain.search_config
            vector_store = VectorStoreFactory.get(vector_store_config)
            embedding_vec = await generate_embedding(
                ctx.state.original_question, vector_store_config.embedding
            )
            vector_query = VectorQuery(
                embedding=embedding_vec,
                top_k=search_config.top_k,
            )
            ctx.state.retrieved_context += await vector_store.search(query=vector_query)

        return AnswerQuestionNode()


@dataclass
class AnswerQuestionNode(BaseNode[ChatGraphState, ChatGraphDeps, ChatResult]):
    """Node that generates the final answer"""

    default_system_prompt = """
You are an expert code assistant helping developers understand complex codebases. Follow these rules strictly:

1. ALWAYS answer the user's question - this is your primary task
2. Base your answer ONLY on the provided code context, not on general knowledge
3. When referencing code:
   - ALWAYS use the format [description]("file_path: line_range") to link to code
   - Example: [incremental update]("knowlang/search.py: 12-16")
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

    async def run(
        self, ctx: GraphRunContext[ChatGraphState, ChatGraphDeps]
    ) -> End[ChatResult]:
        chat_config = ctx.deps.chat_config
        answer_agent = Agent(
            create_pydantic_model(
                model_provider=chat_config.llm.model_provider,
                model_name=chat_config.llm.model_name,
            ),
            system_prompt=(
                self.default_system_prompt
                if chat_config.llm.system_prompt is None
                else chat_config.llm.system_prompt
            ),
        )

        if not ctx.state.retrieved_context:
            return End(
                ChatResult(
                    answer="I couldn't find any relevant code context for your question. "
                    "Could you please rephrase or be more specific?",
                    retrieved_context=None,
                )
            )

        context = ctx.state.retrieved_context
        for single_context in context:
            chunk = truncate_chunk(single_context.document)

        prompt = f"""
Question: {ctx.state.original_question}

Relevant Code Context:
{context}

Provide a focused answer to the question based on the provided context.

Important: Stay focused on answering the specific question asked.
        """

        try:
            result = await answer_agent.run(prompt)
            return End(
                ChatResult(
                    answer=result.output,
                    retrieved_context=context,
                )
            )
        except Exception as e:
            LOG.error(f"Error generating answer: {e}")
            return End(
                ChatResult(
                    answer="I encountered an error processing your question. Please try again.",
                    retrieved_context=context,
                )
            )


# Create the graph
chat_graph = Graph(nodes=[RetrievalNode, AnswerQuestionNode])


async def stream_chat_progress(
    question: str,
    domain_configs: Optional[List[BaseDomainConfig]] = None,
) -> AsyncGenerator[StreamingChatResult, None]:
    """
    Stream chat progress through the graph.
    This is the main entry point for chat processing.
    """
    state = ChatGraphState(original_question=question)
    if domain_configs is None:
        from knowlang.assets.registry import DomainRegistry, RegistryConfig

        config = RegistryConfig()
        registry = DomainRegistry(config)
        await registry.discover_and_register()
        domain_configs = registry.domain_configs.values()
    deps = ChatGraphDeps(domain_configs=domain_configs, chat_config=ChatConfig())

    start_node = RetrievalNode()

    try:
        # Initial status
        yield StreamingChatResult(
            answer="",
            status=ChatStatus.STARTING,
            progress_message=f"Processing question: {question}",
        )

        graph_run_cm = chat_graph.iter(
            start_node, state=state, deps=deps, infer_name=False
        )

        # we have to manually enter the context manager since this function itself is a AsyncGenerator
        graph_run = await graph_run_cm.__aenter__()
        next_node = graph_run.next_node

        while True:
            # Yield current node's status before processing
            yield StreamingChatResult.from_node(next_node, state)

            # Process the current node
            next_node = await graph_run.next(next_node)

            if isinstance(next_node, End):
                result: ChatResult = next_node.data
                # Yield final result and break
                yield StreamingChatResult.complete(result)
                break
            elif not isinstance(next_node, BaseNode):
                # If the next node is not a valid BaseNode, raise an error
                raise TypeError(f"Invalid node type: {type(next_node)}")

    except Exception as e:
        LOG.error(f"Error in stream_chat_progress: {e}")
        yield StreamingChatResult.error(str(e))
    finally:
        await graph_run_cm.__aexit__(None, None, None)
        return
