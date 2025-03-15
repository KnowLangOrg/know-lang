import json
from pathlib import Path
from typing import List, Optional

from knowlang.evaluations.base import SearchConfiguration
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


class SearchConfigurationManager:
    """Manages search configurations for evaluation."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_configuration(self, config: SearchConfiguration) -> None:
        """
        Save a search configuration to a file.
        
        Args:
            config: Search configuration to save
        """
        file_path = self.config_dir / f"{config.name}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2)
        
        LOG.info(f"Saved search configuration to {file_path}")
    
    def load_configuration(self, name: str) -> Optional[SearchConfiguration]:
        """
        Load a search configuration from a file.
        
        Args:
            name: Name of the configuration to load
            
        Returns:
            SearchConfiguration if found, None otherwise
        """
        file_path = self.config_dir / f"{name}.json"
        if not file_path.exists():
            LOG.warning(f"Configuration file not found: {file_path}")
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            return SearchConfiguration.model_validate(config_data)
        except Exception as e:
            LOG.error(f"Error loading configuration: {e}")
            return None
    
    def list_configurations(self) -> List[str]:
        """
        List all available configurations.
        
        Returns:
            List of configuration names
        """
        return [f.stem for f in self.config_dir.glob("*.json")]
    
    def create_default_configurations(self) -> List[SearchConfiguration]:
        """
        Create and save default search configurations.
        
        Returns:
            List of created configurations
        """
        default_configs = [
            SearchConfiguration(
                name="baseline",
                description="Baseline configuration with both search methods and reranking",
                keyword_search_enabled=True,
                vector_search_enabled=True,
                reranking_enabled=True,
                keyword_search_threshold=0.0,
                vector_search_threshold=0.7,
                reranker_threshold=0.5,
                keyword_search_top_k=50,
                vector_search_top_k=50,
                reranker_top_k=10,
            ),
            SearchConfiguration(
                name="keyword_only",
                description="Keyword search only without reranking",
                keyword_search_enabled=True,
                vector_search_enabled=False,
                reranking_enabled=False,
                keyword_search_threshold=0.0,
                keyword_search_top_k=100,
            ),
            SearchConfiguration(
                name="vector_only",
                description="Vector search only without reranking",
                keyword_search_enabled=False,
                vector_search_enabled=True,
                reranking_enabled=False,
                vector_search_threshold=0.6,
                vector_search_top_k=100,
            ),
            SearchConfiguration(
                name="no_reranking",
                description="Both search methods without reranking",
                keyword_search_enabled=True,
                vector_search_enabled=True,
                reranking_enabled=False,
                keyword_search_threshold=0.0,
                vector_search_threshold=0.6,
                keyword_search_top_k=50,
                vector_search_top_k=50,
            ),
            SearchConfiguration(
                name="high_precision",
                description="Higher thresholds for precision-focused search",
                keyword_search_enabled=True,
                vector_search_enabled=True,
                reranking_enabled=True,
                keyword_search_threshold=0.2,
                vector_search_threshold=0.8,
                reranker_threshold=0.7,
                keyword_search_top_k=20,
                vector_search_top_k=20,
                reranker_top_k=5,
            ),
            SearchConfiguration(
                name="high_recall",
                description="Lower thresholds for recall-focused search",
                keyword_search_enabled=True,
                vector_search_enabled=True,
                reranking_enabled=True,
                keyword_search_threshold=0.0,
                vector_search_threshold=0.5,
                reranker_threshold=0.3,
                keyword_search_top_k=100,
                vector_search_top_k=100,
                reranker_top_k=20,
            ),
        ]
        
        for config in default_configs:
            self.save_configuration(config)
        
        return default_configs