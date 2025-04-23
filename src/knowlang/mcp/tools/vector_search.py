from typing import Dict, List, Optional, Any

from knowlang.mcp.common import KnowLangTool, Singleton
from knowlang.models.types import EmbeddingInputType
from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.query import VectorQuery, SearchQuery
from knowlang.configs.retrieval_config import SearchConfig
from knowlang.configs.config import AppConfig
from knowlang.utils import FancyLogger
from knowlang.vector_stores.base import VectorStore

LOG = FancyLogger(__name__)

class VectorSearchTool(KnowLangTool, metaclass=Singleton):
    """MCP tool for vector-based search in Knowlang."""
    name: str = "vector search codebase"
    description = "Search code snippets through vector embeddings"
    config : AppConfig = None
    vector_store: VectorStore = None

    
    @classmethod
    def initialize(cls, config: AppConfig) -> 'VectorSearchTool':
        _instance = VectorSearchTool()

        from knowlang.vector_stores.factory import VectorStoreFactory
        _instance.config = config
        _instance.vector_store = VectorStoreFactory.get(config)

        return _instance


    @classmethod
    async def run(self, query: str) -> List[Dict[str, Any]]:
        from knowlang.models.embeddings import generate_embedding

        embedding = await generate_embedding(query, self.config.embedding, EmbeddingInputType.QUERY)

        vector_query = VectorQuery(
            embedding=embedding,
            top_k=self.config.retrieval.vector_search.top_k,
        )

        results = await self.vector_store.search(
            query=vector_query,
            strategy_name=SearchMethodology.VECTOR,
        )

        return [r.model_dump_json() for r in results]
