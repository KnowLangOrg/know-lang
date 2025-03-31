from pathlib import Path
from typing import List, Optional
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from knowlang.core.types import (BaseChunkType, CodeChunk, CodeLocation,
                                CodeMetadata, LanguageEnum)
from knowlang.parser.base.parser import LanguageParser
from knowlang.utils import convert_to_relative_path, FancyLogger

LOG = FancyLogger(__name__)

class TypescriptChunkType(BaseChunkType):
    """TypeScript-specific chunk types"""
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"


class TypeScriptParser(LanguageParser):
    """TypeScript-specific implementation of LanguageParser"""
    
    def setup(self) -> None:
        """Initialize tree-sitter with TypeScript language support for both TS and TSX"""
        self.language_name = LanguageEnum.TYPESCRIPT
        # Initialize two different parsers for TS and TSX
        self.language_ts = Language(tree_sitter_typescript.language_typescript())
        self.language_tsx = Language(tree_sitter_typescript.language_tsx())
        
        # Create two separate parsers
        self.parser_ts = Parser(self.language_ts)
        self.parser_tsx = Parser(self.language_tsx)
        
        self.language_config = self.config.parser.languages["typescript"]
    
    def _get_parser_for_file(self, file_path: Path) -> Parser:
        """Get the appropriate parser based on file extension"""
        if file_path.suffix.lower() == ".tsx":
            return self.parser_tsx
        else:
            return self.parser_ts
    
    def _get_preceding_docstring(self, node: Node, source_code: bytes) -> Optional[str]:
        """Extract docstring from JSDoc comments"""
        docstring_parts = []
        current_node = node.prev_sibling

        while current_node:
            if current_node.type == "comment":
                comment = source_code[current_node.start_byte:current_node.end_byte].decode('utf-8')
                # Handle JSDoc style comments (/** ... */)
                if comment.startswith('/**') and comment.endswith('*/'):
                    # Clean up the comment by removing /** and */ and * at the beginning of lines
                    lines = comment[3:-2].strip().split('\n')
                    cleaned_lines = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith('*'):
                            line = line[1:].strip()
                        cleaned_lines.append(line)
                    docstring = '\n'.join(cleaned_lines)
                    docstring_parts.insert(0, docstring)
                    break
            # Stop if we encounter a non-comment node
            elif current_node.type not in ("expression_statement", "empty_statement"):
                break
            current_node = current_node.prev_sibling
        
        return '\n'.join(docstring_parts) if docstring_parts else None

    def _get_namespace_context(self, node: Node, source_code: bytes) -> Optional[str]:
        """Get the namespace (module) context of a node"""
        current = node.parent
        namespaces = []
        
        while current:
            if current.type == "module":
                for child in current.children:
                    if child.type == "identifier":
                        namespaces.insert(0, source_code[child.start_byte:child.end_byte].decode('utf-8'))
                        break
            elif current.type == "namespace_declaration":
                for child in current.children:
                    if child.type == "identifier":
                        namespaces.insert(0, source_code[child.start_byte:child.end_byte].decode('utf-8'))
                        break
            current = current.parent
            
        return ".".join(namespaces) if namespaces else None

    def _is_generic_type(self, node: Node) -> bool:
        """Check if a node has generic type parameters"""
        for child in node.children:
            if child.type == "type_parameters":
                return True
        return False

    def _process_class(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a class node and return a CodeChunk"""
        name = node.child_by_field_name("name").text.decode('utf-8')
                
        if not name:
            raise ValueError(f"Could not find class name in node: {node.__str__()}")
        
        # Check if this is a decorated class
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                decorator_text = source_code[child.start_byte:child.end_byte].decode('utf-8')
                decorators.append(decorator_text)
        
        return CodeChunk(
            language=self.language_name,
            type=TypescriptChunkType.CLASS,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            location=CodeLocation(
                file_path=str(file_path),
                start_line=node.start_point[0],
                end_line=node.end_point[0]
            ),
            docstring=self._get_preceding_docstring(node, source_code),
            metadata=CodeMetadata(
                namespace=self._get_namespace_context(node, source_code),
                is_generic=self._is_generic_type(node),
                decorators=decorators if decorators else None
            )
        )
    
    def _process_interface(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process an interface node and return a CodeChunk"""
        name = node.child_by_field_name("name").text.decode('utf-8')
                
        if not name:
            raise ValueError(f"Could not find interface name in node: {node.__str__()}")
        
        return CodeChunk(
            language=self.language_name,
            type=TypescriptChunkType.INTERFACE,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            location=CodeLocation(
                file_path=str(file_path),
                start_line=node.start_point[0],
                end_line=node.end_point[0]
            ),
            docstring=self._get_preceding_docstring(node, source_code),
            metadata=CodeMetadata(
                namespace=self._get_namespace_context(node, source_code),
                is_generic=self._is_generic_type(node)
            )
        )
    
    def _process_type_alias(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a type alias node and return a CodeChunk"""
        name = node.child_by_field_name("name").text.decode('utf-8')
                
        if not name:
            raise ValueError(f"Could not find type alias name in node: {node.__str__()}")
        
        return CodeChunk(
            language=self.language_name,
            type=TypescriptChunkType.TYPE_ALIAS,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            location=CodeLocation(
                file_path=str(file_path),
                start_line=node.start_point[0],
                end_line=node.end_point[0]
            ),
            docstring=self._get_preceding_docstring(node, source_code),
            metadata=CodeMetadata(
                namespace=self._get_namespace_context(node, source_code),
                is_generic=self._is_generic_type(node)
            )
        )

    def _process_function(self, node: Node, source_code: bytes, file_path: Path) -> CodeChunk:
        """Process a function node and return a CodeChunk"""
        # In TypeScript, functions can be declarations, expressions, or methods
        name = None
        
        # Handle different function types
        if node.type == "function_declaration":
            name = node.child_by_field_name("name").text.decode('utf-8')
        elif node.type == "method_definition":
            for child in node.children:
                if child.type in ("property_identifier", "identifier"):
                    name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                    break
        elif node.type == "arrow_function":
            # For arrow functions assigned to variables, try to find the variable name
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                for child in parent.children:
                    if child.type == "identifier":
                        name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                        break
                # For property assignments, use the property name
                if not name and parent.parent and parent.parent.type == "pair":
                    for child in parent.parent.children:
                        if child.type == "property_identifier":
                            name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                            break
        elif node.type == "lexical_declaration":
            # For const/let declarations with arrow functions
            for child in node.children:
                if child.type == "variable_declarator":
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            name = source_code[subchild.start_byte:subchild.end_byte].decode('utf-8')
                            break
        
        if not name:
            # If we still can't find a name, use a placeholder
            name = f"anonymous_function_{node.start_point[0]}_{node.start_point[1]}"
        
        # Check for decorators
        decorators = []
        # For method_definition, decorators are typically at the parent level
        current_node = node
        if node.type == "method_definition" and node.parent and node.parent.type == "class_body":
            for child in node.parent.children:
                if child.type == "decorator" and child.next_sibling == node:
                    decorator_text = source_code[child.start_byte:child.end_byte].decode('utf-8')
                    decorators.append(decorator_text)
        
        # For function_declaration, decorators are directly associated
        if node.type == "function_declaration":
            for child in node.children:
                if child.type == "decorator":
                    decorator_text = source_code[child.start_byte:child.end_byte].decode('utf-8')
                    decorators.append(decorator_text)
        
        parent_name = None

        
        return CodeChunk(
            language=self.language_name,
            type=TypescriptChunkType.FUNCTION,
            name=name,
            content=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            location=CodeLocation(
                file_path=str(file_path),
                start_line=node.start_point[0],
                end_line=node.end_point[0]
            ),
            parent_name=parent_name,
            docstring=self._get_preceding_docstring(node, source_code),
            metadata=CodeMetadata(
                namespace=self._get_namespace_context(node, source_code),
                is_generic=self._is_generic_type(node),
                decorators=decorators if decorators else None
            )
        )

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single TypeScript file and return list of code chunks"""
        if not self.supports_extension(file_path.suffix):
            LOG.debug(f"Skipping file {file_path}: unsupported extension")
            return []

        try:
            # Check file size limit
            if file_path.stat().st_size > self.language_config.max_file_size:
                LOG.warning(f"Skipping file {file_path}: exceeds size limit of {self.language_config.max_file_size} bytes")
                return []

            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            # Select the appropriate parser based on file extension
            parser = self._get_parser_for_file(file_path)
            if not parser:
                raise RuntimeError("Parser not initialized. Call setup() first.")
                
            tree = parser.parse(source_code)
            chunks: List[CodeChunk] = []
            
            # Get the relative path for location information
            relative_path = convert_to_relative_path(file_path, self.config.db)
            
            def traverse_node(node: Node):
                """Recursively traverse the syntax tree"""
                try:
                    if node.type == "class_declaration":
                        chunks.append(self._process_class(node, source_code, relative_path))
                        return
                    elif node.type == "interface_declaration":
                        chunks.append(self._process_interface(node, source_code, relative_path))
                        return
                    elif node.type == "type_alias_declaration":
                        chunks.append(self._process_type_alias(node, source_code, relative_path))
                        return
                    elif node.type in ("function_declaration", "method_definition"):
                        chunks.append(self._process_function(node, source_code, relative_path))
                        return
                    elif node.type == "variable_declaration" or node.type == "lexical_declaration":
                        # Look for arrow functions assigned to variables
                        for declarator in node.children:
                            if declarator.type == "variable_declarator":
                                for child in declarator.children:
                                    if child.type == "arrow_function":
                                        chunks.append(self._process_function(declarator, source_code, relative_path))
                                        return
                except ValueError as e:
                    LOG.warning(f"Failed to process node: {str(e)}")
                
                # Recursively process children for nested structures
                for child in node.children:
                    traverse_node(child)
            
            # Start traversal from root
            traverse_node(tree.root_node)
            return chunks

        except Exception as e:
            LOG.error(f"Error parsing file {file_path}: {str(e)}")
            return []
    
    def supports_extension(self, ext: str) -> bool:
        """Check if this parser supports a given file extension"""
        return ext.lower() in self.language_config.file_extensions