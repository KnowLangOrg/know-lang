from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel
from knowlang.vector_stores.base import VectorStore
from knowlang.search import SearchResult
from knowlang.configs import AppConfig
from knowlang.api.base import ApiModelRegistry

class ChatResult(BaseModel):
    """Final result from the chat graph"""
    answer: str
    retrieved_context: Optional[List[SearchResult]] = None

@dataclass
class ChatGraphState:
    """State maintained throughout the graph execution"""
    original_question: str
    polished_question: Optional[str] = None
    retrieved_context: Optional[List[SearchResult]] = None

@dataclass
class ChatGraphDeps:
    """Dependencies required by the graph"""
    vector_store: VectorStore
    config: AppConfig

