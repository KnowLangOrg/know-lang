from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    """Configuration for retrieval"""
    enabled: bool = Field(
        default=True, 
        description="Enable retrieval"
    )
    top_k: int = Field(
        default=10, 
        description="Number of candidates to retrieve"
    )
    score_threshold: float = Field(
        default=0.0, 
        description="Minimum relevance score to include a document"
    )
    max_retries: int = Field(
        default=5, 
        description="Maximum number of retries for retrieval"
    )

class MultiStageRetrievalConfig(BaseModel):
    """Configuration for two-stage retrieval"""
    enabled: bool = Field(
        default=True, 
        description="Enable two-stage retrieval"
    )
    keyword_search : SearchConfig = Field(
        default_factory=SearchConfig,
        description="Keyword search configuration"
    )
    vector_search : SearchConfig = Field(
        default_factory=SearchConfig,
        description="Vector search configuration"
    )