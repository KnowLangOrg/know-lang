from pydantic import BaseModel
from enum import Enum
from typing import List, Optional
from knowlang.configs.config import EmbeddingConfig

EmbeddingVector = List[float]


class EmbeddingInputType(Enum):
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingParams(BaseModel):
    cfg: EmbeddingConfig
    inputs: List[str]
    input_type: Optional[EmbeddingInputType] = None
