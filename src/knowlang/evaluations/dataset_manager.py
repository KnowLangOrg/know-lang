from pathlib import Path
from typing import Dict, List, Optional

from knowlang.evaluations.types import DatasetSplitType, DatasetType
from knowlang.evaluations.providers.codesearchnet_provider import CodeSearchNetProvider
from knowlang.evaluations.dataset_provider import DatasetProvider
from knowlang.evaluations.indexer import DatasetIndexer, QueryManager
from knowlang.configs import AppConfig
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


class DatasetManager:
    """
    High-level manager for dataset preparation.
    
    This class orchestrates the loading, indexing, and query mapping
    for benchmark datasets.
    """
    
    def __init__(self, config: AppConfig, data_dir: Path, output_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Initialize components
        self.indexer = DatasetIndexer(config)
        self.query_manager = QueryManager(output_dir)
        
        # Create dataset providers
        self.providers: Dict[DatasetType, DatasetProvider] = {
            DatasetType.CODESEARCHNET: CodeSearchNetProvider(data_dir),
            # DatasetType.COSQA: CoSQAProvider(data_dir / "cosqa"),
        }
    
    async def prepare_dataset(
        self, 
        dataset_type: DatasetType,
        splits: List[DatasetSplitType] = None,
        languages: Optional[List[str]] = None,
    ) -> Dict:
        """
        Prepare a dataset for evaluation.
        
        This method:
        1. Loads the dataset
        2. Indexes code snippets into the vector store
        3. Saves query mappings for evaluation
        
        Args:
            dataset_type: Type of dataset to prepare
            languages: Filter for programming languages
            split: Dataset split to use
            
        Returns:
            Dictionary with dataset statistics
        """
        provider = self.providers.get(dataset_type)
        if not provider:
            LOG.error(f"Unsupported dataset type: {dataset_type}")
            return {"error": f"Unsupported dataset type: {dataset_type}"}
        
        try:
            # Load dataset
            LOG.info(f"Loading {dataset_type} dataset (split: {splits})")
            pairs = await provider.load(languages=languages, splits=splits)
            
            if not pairs:
                LOG.warning(f"No data loaded for {dataset_type}")
                return {"error": "No data loaded"}
            
            # Index code snippets
            LOG.info(f"Indexing {len(pairs)} code snippets")
            indexed_ids = await self.indexer.index_dataset(pairs)
            
            # Save query mappings
            LOG.info(f"Saving query mappings for {len(pairs)} pairs")
            self.query_manager.save_query_mappings(pairs, str(dataset_type))
            
            return {
                "dataset_type": str(dataset_type),
                "pairs_loaded": len(pairs),
                "indexed_snippets": len(indexed_ids),
                "languages": list(set(p.language for p in pairs)),
                "splits": splits
            }
            
        except Exception as e:
            LOG.error(f"Error preparing dataset {dataset_type}: {e}")
            return {"error": str(e)}