from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

from knowlang.database.config import VectorStoreConfig


class CodeProvider(ABC):
    """Abstract base class for code source providers"""

    @abstractmethod
    def __init__(self, source_path: Path, config: VectorStoreConfig):
        self.source_path = source_path
        self.config = config

    @abstractmethod
    def get_files(self) -> Generator[Path, None, None]:
        """Yield paths to code files that should be processed"""
        pass
