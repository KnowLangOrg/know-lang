from typing import List, Optional
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Base class for all search queries"""
    top_k: int = Field(default=10, description="Number of results to return")
    score_threshold: float = Field(default=0.0, description="Minimum score threshold")

class VectorQuery(SearchQuery):
    """Query for vector similarity search"""
    embedding: List[float]
    
class KeywordQuery(SearchQuery):
    """Query for keyword-based search"""
    text: str
    fields: Optional[List[str]] = None