from pathlib import Path
from typing import Dict, Optional, Type

from knowlang.configs import AppConfig
from knowlang.parser.base.parser import LanguageParser
from knowlang.core.types import LanguageEnum # Added
from knowlang.parser.languages.cpp.parser import CppParser
from knowlang.parser.languages.csharp.parser import CSharpParser # Added
from knowlang.parser.languages.python.parser import PythonParser
from knowlang.parser.languages.ts.parser import TypeScriptParser


class CodeParserFactory():
    """Concrete implementation of parser factory"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._parsers: Dict[str, LanguageParser] = {}
        self._parser_classes = self._register_parsers()
    
    def _register_parsers(self) -> Dict[str, Type[LanguageParser]]:
        """Register available parser implementations"""
        return {
            LanguageEnum.PYTHON.value: PythonParser,
            LanguageEnum.CPP.value: CppParser,
            LanguageEnum.TYPESCRIPT.value: TypeScriptParser,
            LanguageEnum.CSHARP.value: CSharpParser, # Added
            # Add more languages here
        }
    
    def get_parser(self, file_path: Path, current_codebase_root: Path) -> Optional[LanguageParser]:
        """Get appropriate parser for a file, specific to its codebase root."""
        extension = file_path.suffix
        
        # Find parser class for this extension
        for lang_name_str, parser_class in self._parser_classes.items():
            # Check if language is configured and enabled in AppConfig
            lang_config_from_app = self.config.parser.languages.get(lang_name_str)
            if not lang_config_from_app or not lang_config_from_app.enabled:
                continue

            # Now, check if this language's configured extensions match the file's extension.
            # This uses the file_extensions list from the AppConfig for this language.
            if extension not in lang_config_from_app.file_extensions:
                continue

            # If the extension matches, this is the correct parser class.
            # Instantiate it with the specific AppConfig and current_codebase_root.
            # No caching of parser instances is used here to ensure that
            # current_codebase_root is correctly associated with this parser instance.
            try:
                # Pass current_codebase_root to the parser's constructor
                parser_instance = parser_class(self.config, current_codebase_root)
                # Call setup on the new instance (might load language-specific resources)
                parser_instance.setup()

                # The supports_extension method on the instance can perform a more detailed check if needed,
                # but we've already matched based on AppConfig. If that's sufficient, we can return.
                # If parser_instance.supports_extension(extension) is more accurate, use it.
                # For now, assuming AppConfig's file_extensions is the primary check.
                return parser_instance
            except Exception as e:
                # Log error during parser instantiation or setup
                # Consider how to get a logger here if needed, or raise.
                # For now, printing to stderr or using a basic logger.
                print(f"Error instantiating or setting up parser for {lang_name_str} with root {current_codebase_root}: {e}")
                # Optionally, re-raise or return None if a parser cannot be created/setup
                return None
        
        return None