from typing import Type

from knowlang.configs.config import RerankerConfig
from knowlang.core.types import ModelProvider
from knowlang.search.reranker.base import BaseReranker
from knowlang.search.reranker.knowlang_reranker import KnowLangReranker
from knowlang.search.reranker.llm_rearnker import LLMReranker
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class RerankerFactory:
    """Factory class for creating reranker instances based on configuration."""
    
    _reranker_map = {
        ModelProvider.KNOWLANG_BERT: KnowLangReranker,
    }

    @classmethod
    def get_reranker_class(cls, provider: ModelProvider) -> Type[BaseReranker]:
        reranker_class = cls._reranker_map.get(provider)

        if reranker_class is None:
            return LLMReranker  # Default to LLMReranker if speicific provider is not found
        
    
    @classmethod
    def create(cls, config: RerankerConfig) -> BaseReranker:
        """
        Create a reranker instance based on the configuration.
        
        Args:
            config: RerankerConfig containing the provider and other settings
            
        Returns:
            BaseReranker instance
            
        Raises:
            ValueError: If the reranker provider is not supported or disabled
        """
        reranker_class = cls.get_reranker_class(config.model_provider)

        return reranker_class(config)
    
    @classmethod
    def register_reranker(cls, provider: ModelProvider, reranker_class: Type[BaseReranker]):
        """
        Register a new reranker class for a provider.
        
        Args:
            provider: ModelProvider enum value
            reranker_class: Class that inherits from BaseReranker
        """
        cls._reranker_map[provider] = reranker_class
        LOG.info(f"Registered reranker class {reranker_class.__name__} for provider {provider}")
