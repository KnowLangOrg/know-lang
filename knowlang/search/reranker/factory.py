from typing import Type

from knowlang.configs.config import RerankerConfig
from knowlang.core.types import RerankerProvider
from knowlang.search.reranker.base import BaseReranker
from knowlang.search.reranker.knowlang_reranker import KnowLangReranker
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class RerankerFactory:
    """Factory class for creating reranker instances based on configuration."""
    
    _reranker_map = {
        RerankerProvider.KNOWLANG_BERT: KnowLangReranker,
        RerankerProvider.LLM_AGENT: None,  # Will be implemented later
    }
    
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
        # Determine provider from config
        provider = config.model_provider
        
        # Get reranker class from map
        reranker_class = cls._reranker_map.get(provider)
        
        if reranker_class is None:
            raise ValueError(f"Unknown reranker provider: {provider}")
        
        LOG.info(f"Creating reranker with provider: {provider}")
        return reranker_class(config)
    
    @classmethod
    def register_reranker(cls, provider: RerankerProvider, reranker_class: Type[BaseReranker]):
        """
        Register a new reranker class for a provider.
        
        Args:
            provider: RerankerProvider enum value
            reranker_class: Class that inherits from BaseReranker
        """
        cls._reranker_map[provider] = reranker_class
        LOG.info(f"Registered reranker class {reranker_class.__name__} for provider {provider}")
