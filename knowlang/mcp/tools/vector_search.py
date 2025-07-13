from typing import Any, Dict, List, Optional

from knowlang.assets.config import BaseDomainConfig
from knowlang.mcp.common import KnowLangTool, Singleton
from knowlang.models.types import EmbeddingInputType
from knowlang.search.base import SearchMethodology
from knowlang.search.query import VectorQuery
from knowlang.utils import FancyLogger
from knowlang.vector_stores.base import VectorStore
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = FancyLogger(__name__)


class VectorSearchTool(KnowLangTool, metaclass=Singleton):
    """MCP tool for vector-based search in Knowlang."""

    name: str = "vector_search_codebase"
    description = """
Search code snippets through vector embeddings with natual language queries. 
It's a powerful tool for finding relevant code snippets fast based on their semantic meaning rather than going through the filesystem.
"""
    domain_configs: Optional[List[BaseDomainConfig]] = []

    @classmethod
    def initialize(
        cls, domain_configs: Optional[List[BaseDomainConfig]] = []
    ) -> "VectorSearchTool":
        _instance = VectorSearchTool()
        _instance.domain_configs = domain_configs

        return _instance

    @classmethod
    async def run(cls, query: str) -> List[Dict[str, Any]]:
        instance = VectorSearchTool()

        from knowlang.models.embeddings import generate_embedding

        results = []
        for domain_config in instance.domain_configs:
            vector_store_config = domain_config.processor_config.vector_store
            vector_store: VectorStore = VectorStoreFactory.get(vector_store_config)

            embedding = await generate_embedding(
                query, vector_store_config.embedding, EmbeddingInputType.QUERY
            )

            vector_query = VectorQuery(
                embedding=embedding,
                top_k=10,  # TODO: make this configurable
            )

            search_results = await vector_store.search(
                query=vector_query,
                strategy_name=SearchMethodology.VECTOR,
            )

            results.extend(search_results)

        return [r.model_dump_json() for r in results]
