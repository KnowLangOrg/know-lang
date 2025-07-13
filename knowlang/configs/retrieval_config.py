from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    """Configuration for retrieval"""

    enabled: bool = Field(default=True, description="Enable retrieval")
    top_k: int = Field(default=50, description="Number of candidates to retrieve")
    score_threshold: float = Field(
        default=0.0, description="Minimum relevance score to include a document"
    )
    query_refinement: bool = Field(default=False, description="Enable query refinement")
    max_retries: int = Field(
        default=2, description="Maximum number of retries for retrieval"
    )
    filter: dict = Field(default_factory=dict, description="Filter for retrieval")
