from pathlib import Path
from typing import List, Optional
import aiofiles
import os

from tree_sitter import Language, Node, Parser
from knowlang.assets.codebase.models import CodeProcessorConfig
from knowlang.core.types import CodeChunk, CodeLocation, CodeMetadata, BaseChunkType, LanguageEnum
from knowlang.parser.base.parser import LanguageParser
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

# Tree-sitter node types for C#
# Reference: https://github.com/tree-sitter/tree-sitter-c-sharp/blob/master/src/node-types.json
NODE_TYPE_CLASS_DECLARATION = "class_declaration"
NODE_TYPE_METHOD_DECLARATION = "method_declaration"
NODE_TYPE_IDENTIFIER = "identifier"
NODE_TYPE_NAMESPACE_DECLARATION = "namespace_declaration"
NODE_TYPE_INTERFACE_DECLARATION = "interface_declaration"
NODE_TYPE_RECORD_DECLARATION = "record_declaration" 
NODE_TYPE_COMMENT = "comment"

# For navigating class/method bodies
NODE_TYPE_BLOCK = "block" # Common body for methods
NODE_TYPE_DECLARATION_LIST = "declaration_list" # Often found in class bodies C#
NODE_TYPE_CLASS_BODY = "class_body" # C# specific for class contents

class CSharpChunkType(BaseChunkType):
    INTERFACE = "interface"

class CSharpParser(LanguageParser):
    def setup(self):

        try:
            import tree_sitter_c_sharp
            LOG.info("Successfully loaded tree-sitter C# grammar from 'tree_sitter_c_sharp' package.")
        except ImportError as e:
            raise ImportError(
                "Failed to import 'tree_sitter_c_sharp'. please install using pip install 'knowlang[csharp]'"
            ) from e

        self.language_name = LanguageEnum.CSHARP
        self.language = Language(tree_sitter_c_sharp.language())
        self.parser = Parser(self.language)
        self.language_config = self.config.languages["csharp"]

    def supports_extension(self, ext: str) -> bool:
        return ext.lower() in self.language_config.file_extensions

    def _get_node_text(self, node: Node, source_code: bytes) -> str:
        return source_code[node.start_byte:node.end_byte].decode('utf-8')

    def _get_code_location(self, node: Node) -> CodeLocation:
        return CodeLocation(start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1)

    def _get_preceding_docstring(self, node: Node, source_code: bytes) -> Optional[str]:
        docstrings = []
        sibling = node.prev_named_sibling
        while sibling:
            if sibling.type == NODE_TYPE_COMMENT:
                docstrings.append(self._get_node_text(sibling, source_code))
            else:
                # Not a comment node, stop searching
                break

            sibling = sibling.prev_named_sibling

        return "\n".join(reversed(docstrings)) if docstrings else None

    def _get_namespace_context(self, node: Node, source_code: bytes) -> Optional[str]:
        namespaces = []
        current = node.parent
        while current:
            if current.type == NODE_TYPE_NAMESPACE_DECLARATION:
                name_node = next((child for child in current.named_children if child.type == NODE_TYPE_IDENTIFIER or child.type == "qualified_name"), None)
                if name_node:
                    namespaces.append(self._get_node_text(name_node, source_code))
            current = current.parent
        return ".".join(reversed(namespaces)) if namespaces else None
    
    def _chunk_type_from_node(self, node: Node) -> CSharpChunkType:
        if node.type == NODE_TYPE_CLASS_DECLARATION:
            return CSharpChunkType.CLASS
        elif node.type == NODE_TYPE_INTERFACE_DECLARATION:
            return CSharpChunkType.INTERFACE
        elif node.type == NODE_TYPE_RECORD_DECLARATION:
            return CSharpChunkType.CLASS
        else:
            return CSharpChunkType.OTHER
    
    def _process_declaration_block(self, node: Node, source_code: bytes, file_path: Path) -> Optional[CodeChunk]:
        name = node.child_by_field_name("name").text.decode('utf-8') 

        if not name:
            raise ValueError(f"Could not find class name in node: {node.text}")

        docstring = self._get_preceding_docstring(node, source_code)
        namespace = self._get_namespace_context(node, source_code)


        return CodeChunk(
            language=self.language_name,
            type=self._chunk_type_from_node(node),
            name=name,
            content=node.text,
            location=CodeLocation(
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                file_path=os.path.relpath(file_path, self.config.directory_path)
            ),
            docstring=docstring,
            metadata=CodeMetadata(
                namespace=namespace,
            )
        )

    async def parse_file(self, file_path: Path) -> List[CodeChunk]:
        if not self.supports_extension(file_path.suffix):
            LOG.debug(f"Skipping file {file_path}: unsupported extension")
            return []

        if not self.parser:
            LOG.error(f"Parser not initialized for C#. Skipping parsing for {file_path}.")
            return []

        LOG.info(f"Parsing C# file: {file_path}")

        async with aiofiles.open(file_path, "rb") as f:
            source_code = await f.read()

        try:
            tree = self.parser.parse(source_code)
        except Exception as e:
            LOG.error(f"Error parsing file {file_path} with tree-sitter: {e}")
            return []

        chunks: List[CodeChunk] = []

        # For now, stop at the top level of class, interface, etc
        def traverse_node(node: Node):
            try:
                if node.type in [
                    NODE_TYPE_CLASS_DECLARATION,
                    NODE_TYPE_INTERFACE_DECLARATION,
                    NODE_TYPE_RECORD_DECLARATION
                ]:
                    chunks.append(self._process_declaration_block(node, source_code, file_path))
                    return
            except Exception as e:
                LOG.error(f"Failed to process code {str(e)}")
                return

            for child in node.children:
                traverse_node(child)
        
        traverse_node(tree.root_node)

        return chunks