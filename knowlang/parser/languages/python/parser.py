from pathlib import Path
from typing import List, Optional
import os
import aiofiles
from tree_sitter import Language, Node, Parser

from knowlang.core.types import (
    BaseChunkType, 
    CodeChunk, 
    CodeLocation,
    LanguageEnum
)
from knowlang.parser.base.parser import LanguageParser
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class PythonParser(LanguageParser):
    """Python-specific implementation of LanguageParser"""
    
    def setup(self) -> None:
        """Initialize tree-sitter with Python language support"""
        try:
            import tree_sitter_python
            LOG.info("Successfully loaded tree-sitter Python grammar from 'tree_sitter_python' package.")
        except ImportError as e:
            raise ImportError(
                "Failed to import 'tree_sitter_python'. Please install using pip install 'knowlang[python]'"
            ) from e
        self.language = Language(tree_sitter_python.language())
        self.language_name = LanguageEnum.PYTHON
        self.parser = Parser(self.language)
        self.language_config = self.config.languages[LanguageEnum.PYTHON.value]
    
    def _get_preceding_docstring(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract docstring from inside function or class body"""
        # For decorated definitions, get the actual function/class node
        if node.type == "decorated_definition":
            node = node.child_by_field_name("definition")
            if not node:
                return None

        # Find the body block
        body_node = node.child_by_field_name("body")
        if not body_node:
            return None

        # Look for the first expression statement containing a string literal
        for child in body_node.children:
            if child.type == "expression_statement":
                string_node = child.children[0] if child.children else None
                if string_node and string_node.type in ("string", "string_literal"):
                    return source_code[string_node.start_byte:string_node.end_byte].decode('utf-8')
        
        return None

    def _has_syntax_error(self, node: Node) -> bool:
        """Check if the node or its children contain syntax errors"""
        if node.type == "ERROR":
            return True
        if node.has_error:
            return True
        return any(self._has_syntax_error(child) for child in node.children)

    def _process_class(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a class node and return a CodeChunk"""
        name = next(
            (child.text.decode('utf-8') 
             for child in node.children 
             if child.type == "identifier"),
            None
        )
        
        if not name:
            raise ValueError(f"Could not find class name in node: {node.text}")
        
        return CodeChunk(
            language=self.language_name,
            type=BaseChunkType.CLASS,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            location=CodeLocation(
                file_path=str(file_path),
                start_line=node.start_point[0],
                end_line=node.end_point[0]
            ),
            docstring=self._get_preceding_docstring(node, source_code)
        )

    def _process_function(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a function node and return a CodeChunk"""
        name = next(
            (child.text.decode('utf-8') 
             for child in node.children 
             if child.type == "identifier"),
            None
        )

        if not name:
            raise ValueError(f"Could not find function name in node: {node.text}")
        
        # Determine if this is a method within a class
        parent_node = node.parent
        parent_name = None
        if parent_node and parent_node.type == "class_definition":
            parent_name = next(
                (child.text.decode('utf-8') 
                 for child in parent_node.children 
                 if child.type == "identifier"),
                None
            )
        
        return CodeChunk(
            language=self.language_name,
            type=BaseChunkType.FUNCTION,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            location=CodeLocation(
                file_path=str(file_path),
                start_line=node.start_point[0],
                end_line=node.end_point[0]
            ),
            parent_name=parent_name,
            docstring=self._get_preceding_docstring(node, source_code)
        )

    def _process_decorated_function(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a decorated function node and return a CodeChunk"""
        # Get the actual function definition node
        function_node = node.child_by_field_name("definition")
        if not function_node:
            raise ValueError("Could not find function definition in decorated node")

        # Get function name
        name = next(
            (child.text.decode('utf-8') 
             for child in function_node.children 
             if child.type == "identifier"),
            None
        )

        if not name:
            raise ValueError(f"Could not find function name in node: {function_node.text}")
        
        # Get decorators
        decorators = [
            child.text.decode('utf-8').strip('@')
            for child in node.children
            if child.type == "decorator"
        ]
        
        # Determine if this is a method within a class
        parent_node = node.parent
        parent_name = None
        if parent_node and parent_node.type == "class_definition":
            parent_name = next(
                (child.text.decode('utf-8') 
                 for child in parent_node.children 
                 if child.type == "identifier"),
                None
            )
        
        return CodeChunk(
            language=self.language_name,
            type=BaseChunkType.FUNCTION,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            location=CodeLocation(
                file_path=str(file_path),
                start_line=node.start_point[0],
                end_line=node.end_point[0]
            ),
            parent_name=parent_name,
            docstring=self._get_preceding_docstring(node, source_code),
        )

    def _process_decorated_class(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a decorated class node and return a CodeChunk"""
        # Get the actual class definition node
        class_node = node.child_by_field_name("definition")
        if not class_node:
            raise ValueError("Could not find class definition in decorated node")

        # Get class name
        name = next(
            (child.text.decode('utf-8') 
             for child in class_node.children 
             if child.type == "identifier"),
            None
        )
        
        if not name:
            raise ValueError(f"Could not find class name in node: {class_node.text}")
        
        # Get decorators
        decorators = [
            child.text.decode('utf-8').strip('@')
            for child in node.children
            if child.type == "decorator"
        ]
        
        return CodeChunk(
            language=self.language_name,
            type=BaseChunkType.CLASS,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            location=CodeLocation(
                file_path=str(file_path),
                start_line=node.start_point[0],
                end_line=node.end_point[0]
            ),
            docstring=self._get_preceding_docstring(node, source_code),
        )

    async def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single Python file and return list of code chunks"""
        if not self.supports_extension(file_path.suffix):
            LOG.debug(f"Skipping file {file_path}: unsupported extension")
            return []

        try:
            # Check file size limit
            if file_path.stat().st_size > self.language_config.max_file_size:
                LOG.warning(f"Skipping file {file_path}: exceeds size limit of {self.language_config.max_file_size} bytes")
                return []

            async with aiofiles.open(file_path, 'rb') as f:
                source_code = await f.read()
            
            if not self.parser:
                raise RuntimeError("Parser not initialized. Call setup() first.")
                
            tree = self.parser.parse(source_code)

            # Check for overall syntax validity
            if self._has_syntax_error(tree.root_node):
                LOG.warning(f"Syntax errors found in {file_path}")
                return []

            chunks: List[CodeChunk] = []
            relative_path = os.path.relpath(file_path, self.config.directory_path)
            
            # Process the syntax tree
            for node in tree.root_node.children:
                if node.type == "decorated_definition":
                    if node.child_by_field_name("definition").type == "function_definition":
                        chunks.append(self._process_decorated_function(node, source_code, relative_path))
                    elif node.child_by_field_name("definition").type == "class_definition":
                        chunks.append(self._process_decorated_class(node, source_code, relative_path))
                elif node.type == "class_definition":
                    chunks.append(self._process_class(node, source_code, relative_path))
                elif node.type == "function_definition":
                    chunks.append(self._process_function(node, source_code, relative_path))
                else:
                    # Skip other node types for now
                    pass
            
            return chunks

        except Exception as e:
            LOG.error(f"Error parsing file {file_path}: {str(e)}")
            return []
    
    def supports_extension(self, ext: str) -> bool:
        """Check if this parser supports a given file extension"""
        return ext in self.language_config.file_extensions