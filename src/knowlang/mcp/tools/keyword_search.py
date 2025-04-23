from typing import Dict, List, Any
from knowlang.configs.config import AppConfig
from knowlang.search.base import SearchMethodology, SearchResult
from knowlang.search.query import KeywordQuery, SearchQuery
from knowlang.configs.retrieval_config import SearchConfig
from knowlang.search.search_graph.keyword_search_agent_node import KeywordSearchAgentNode
from knowlang.utils import FancyLogger
from mcp import Tool
from knowlang.mcp.common import KnowLangTool, Singleton
from knowlang.vector_stores.base import VectorStore

LOG = FancyLogger(__name__)

class KeywordSearchTool(KnowLangTool, metaclass=Singleton):
    """ Search code snippets through string keywords"""
    name: str = "keyword search codebase"
    description = "Search code snippets through string keywords"
    config : AppConfig = None
    vector_store: VectorStore = None

    
    @classmethod
    def initialize(cls, config: AppConfig) -> 'KeywordSearchTool':
        _instance = KeywordSearchTool()

        from knowlang.vector_stores.factory import VectorStoreFactory
        _instance.config = config
        _instance.vector_store = VectorStoreFactory.get(config)

        return _instance
        
    @classmethod
    async def run(cls, query: str) -> List[Dict[str, Any]]:
        instance = KeywordSearchTool()
        vector_query = KeywordQuery(
            text=query,
            top_k=instance.config.retrieval.vector_search.top_k,
        )

        results = await instance.vector_store.search(
            query=vector_query,
            strategy_name=SearchMethodology.KEYWORD,
        )

        return [r.model_dump_json() for r in results]