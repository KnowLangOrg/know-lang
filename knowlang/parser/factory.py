from pathlib import Path
from typing import Dict, Optional, Type

from knowlang.parser.base.parser import CodeProcessorConfig, LanguageParser
from knowlang.core.types import LanguageEnum 


class CodeParserFactory:
    """Concrete implementation of parser factory"""
    
    def __init__(self, config: CodeProcessorConfig):
        self.config = config
        self._parsers: Dict[str, LanguageParser] = {}
        self._parser_classes = self._register_parsers()

    def _register_parsers(self) -> Dict[str, Type[LanguageParser]]:
        """Register available parser implementations"""
        from knowlang.parser.languages.cpp.parser import CppParser
        from knowlang.parser.languages.csharp.parser import CSharpParser
        from knowlang.parser.languages.python.parser import PythonParser
        from knowlang.parser.languages.ts.parser import TypeScriptParser
        from knowlang.parser.languages.unity_asset.parser import UnityAssetParser

        return {
            LanguageEnum.PYTHON.value: PythonParser,
            LanguageEnum.CPP.value: CppParser,
            LanguageEnum.TYPESCRIPT.value: TypeScriptParser,
            LanguageEnum.CSHARP.value: CSharpParser,  # Added
            LanguageEnum.UNITYASSET.value: UnityAssetParser,  # Added Unity Asset parser
            # Add more languages here
        }

    def get_parser(self, file_path: Path) -> Optional[LanguageParser]:
        """Get appropriate parser for a file"""
        extension = file_path.suffix

        # Find parser class for this extension
        for lang, parser_class in self._parser_classes.items():
            if lang not in self.config.languages or not self.config.languages[lang].enabled:
                continue
            parser = self._parsers.get(lang)
            if parser is None:
                parser = parser_class(self.config)
                parser.setup()
                self._parsers[lang] = parser

            if parser.supports_extension(extension):
                return self._parsers[lang]

        return None
