from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from tree_sitter import Language, Parser

from knowlang.configs import AppConfig, LanguageConfig
from knowlang.core.types import CodeChunk, LanguageEnum


class LanguageParser(ABC):
    """Abstract base class for language-specific parsers"""
    
    def __init__(self, config: AppConfig, current_codebase_root: Path):
        self.config : AppConfig = config
        self.current_codebase_root: Path = current_codebase_root
        self.language_name : LanguageEnum = None
        self.language : Language = None
        self.parser : Parser = None
        self.language_config : LanguageConfig = None
    
    @abstractmethod
    def setup(self) -> None:
        """Set up the parser (e.g., initialize tree-sitter)"""
        pass

    @abstractmethod
    def parse_file(self, file_path: Path, root_alias: str) -> List[CodeChunk]:
        """
        Parse a single file and return code chunks.
        file_path is the absolute path to the file.
        root_alias is the alias of the codebase source.
        """
        pass

    @abstractmethod
    def supports_extension(self, ext: str) -> bool:
        """Check if this parser supports a given file extension"""
        pass