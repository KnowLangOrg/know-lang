from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from knowlang.evaluations.base import QueryCodePair
from knowlang.evaluations.types import DatasetSplitType


class DatasetProvider(ABC):
    """Base class for dataset providers."""
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
    
    @abstractmethod
    async def load(self, 
                  languages: Optional[List[str]] = None, 
                  splits: Optional[List[DatasetSplitType]] = None) -> List[QueryCodePair]:
        """
        Load dataset and return list of query-code pairs.
        
        Args:
            languages: Optional filter for programming languages
            splits: Dataset splits to load (train, valid, test)
            
        Returns:
            List of QueryCodePair objects
        """
        pass