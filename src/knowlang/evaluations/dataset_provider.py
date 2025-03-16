from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from knowlang.evaluations.base import QueryCodePair


class DatasetProvider(ABC):
    """Base class for dataset providers."""
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
    
    @abstractmethod
    async def load(self, languages: Optional[List[str]] = None, split: str = "test") -> List[QueryCodePair]:
        """
        Load dataset and return list of query-code pairs.
        
        Args:
            languages: Optional filter for programming languages
            split: Dataset split to use (train, valid, test)
            
        Returns:
            List of QueryCodePair objects
        """
        pass