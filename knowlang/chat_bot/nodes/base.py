from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel
from knowlang.search import SearchResult
from knowlang.assets.config import BaseDomainConfig
from knowlang.configs.chat_config import ChatConfig


class ChatResult(BaseModel):
    """Final result from the chat graph"""

    answer: str
    retrieved_context: Optional[List[SearchResult]] = None


@dataclass
class ChatGraphState:
    """State maintained throughout the graph execution"""

    original_question: str
    polished_question: Optional[str] = None
    retrieved_context: Optional[List[SearchResult]] = field(default_factory=list)


@dataclass
class ChatGraphDeps:
    """Dependencies required by the graph"""

    domain_configs: List[BaseDomainConfig] = None
    chat_config: ChatConfig = None
