from pathlib import Path
from typing import List, Optional, Tuple

from tree_sitter import Language, Node, Parser
# Assuming LanguageEnum will be updated to include CSHARP
# from knowlang.core.types import LanguageEnum
from knowlang.core.types import CodeChunk, CodeLocation, CodeMetadata, BaseChunkType, LanguageEnum
from knowlang.parser.base.parser import LanguageParser
from knowlang.configs import AppConfig
from knowlang.utils import FancyLogger, convert_to_relative_path

LOG = FancyLogger(__name__)

# Tree-sitter node types for C#
# Reference: https://github.com/tree-sitter/tree-sitter-c-sharp/blob/master/src/node-types.json
NODE_TYPE_CLASS_DECLARATION = "class_declaration"
NODE_TYPE_METHOD_DECLARATION = "method_declaration"
NODE_TYPE_IDENTIFIER = "identifier"
NODE_TYPE_NAMESPACE_DECLARATION = "namespace_declaration"
NODE_TYPE_COMMENT = "comment" # Covers // and /* */

# For navigating class/method bodies
NODE_TYPE_BLOCK = "block" # Common body for methods
NODE_TYPE_DECLARATION_LIST = "declaration_list" # Often found in class bodies C#
NODE_TYPE_CLASS_BODY = "class_body" # C# specific for class contents


class CSharpParser(LanguageParser):
    def __init__(self, config: AppConfig):
        super().__init__(config)
        # LOG.info("CSharpParser initialized")

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
        self.language_config = self.config.parser.languages["csharp"]

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

    def _process_class(self, node: Node, source_code: bytes, file_path: Path) -> Optional[CodeChunk]:
        name = node.child_by_field_name("name").text.decode('utf-8') 

        if not name:
            raise ValueError(f"Could not find class name in node: {node.text}")

        docstring = self._get_preceding_docstring(node, source_code)
        namespace = self._get_namespace_context(node, source_code)


        return CodeChunk(
            language=self.language_name,
            type=BaseChunkType.CLASS,
            name=name,
            content=node.text,
            location=CodeLocation(
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                file_path=str(file_path)
            ),
            docstring=docstring,
            metadata=CodeMetadata(
                namespace=namespace,
            )
        )

    def _process_method(self, node: Node, source_code: bytes, file_path: Path, relative_path: Path, class_name: Optional[str] = None, class_namespace: Optional[str] = None) -> Optional[CodeChunk]:
        name_node = next((child for child in node.named_children if child.type == NODE_TYPE_IDENTIFIER), None)
        if not name_node:
            LOG.debug(f"Method declaration without an identifier: {self._get_node_text(node, source_code)}")
            return None

        method_name = self._get_node_text(name_node, source_code)
        docstring = self._get_preceding_docstring(node, source_code)

        # Namespace for method can be inherited from class or resolved if global (less common in C#)
        namespace = class_namespace if class_namespace else self._get_namespace_context(node, source_code)

        content = self._get_node_text(node, source_code)

        metadata = CodeMetadata(
            file_path=str(file_path),
            relative_path=str(relative_path),
            namespace=namespace,
            class_name=class_name, # Could be None for global methods if C# had them like C++
            type=BaseChunkType.FUNCTION, # Using FUNCTION for methods
            name=method_name,
            docstring=docstring,
            code_location=self._get_code_location(node),
            additional_info={}
        )
        return CodeChunk(content=content, metadata=metadata)

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        if not self.supports_extension(file_path.suffix):
            LOG.debug(f"Skipping file {file_path}: unsupported extension")
            return []

        if not self.parser:
            LOG.error(f"Parser not initialized for C#. Skipping parsing for {file_path}.")
            return []

        LOG.info(f"Parsing C# file: {file_path}")

        try:
            with open(file_path, "rb") as f:
                source_code = f.read()
        except Exception as e:
            LOG.error(f"Error reading file {file_path}: {e}")
            return []

        try:
            tree = self.parser.parse(source_code)
        except Exception as e:
            LOG.error(f"Error parsing file {file_path} with tree-sitter: {e}")
            return []

        chunks: List[CodeChunk] = []
        relative_path = convert_to_relative_path(file_path, self.config.db)

        # Using a list as a queue for BFS-like traversal (node, current_namespace, current_class_name)
        # We only go one level deep for methods within classes.

        # queue: List[Tuple[Node, Optional[str], Optional[str]]] = [(tree.root_node, None, None)]

        # visited_nodes = set() # To avoid processing nodes multiple times if graph is complex

        # while queue:
        #     node, current_namespace_str, current_class_name_str = queue.pop(0)

        #     if node.id in visited_nodes:
        #         continue
        #     visited_nodes.add(node.id)

        #     if node.type == NODE_TYPE_CLASS_DECLARATION:
        #         class_chunk = self._process_class(node, source_code, file_path, relative_path)
        #         if class_chunk:
        #             chunks.append(class_chunk)
        #             # Now look for direct methods within this class
        #             # C# class body can be 'declaration_list' or 'class_body'
        #             body_node = next((child for child in node.children if child.type in [NODE_TYPE_DECLARATION_LIST, NODE_TYPE_CLASS_BODY, NODE_TYPE_BLOCK]), None)
        #             if body_node:
        #                 for child in body_node.children:
        #                     if child.type == NODE_TYPE_METHOD_DECLARATION:
        #                         method_chunk = self._process_method(
        #                             child, source_code, file_path, relative_path,
        #                             class_name=class_chunk.name,
        #                             class_namespace=class_chunk.metadata.namespace
        #                         )
        #                         if method_chunk:
        #                             chunks.append(method_chunk)
        #                     # Do not recurse into nested classes or other structures within class body
        #         # Do not add children of class_declaration to the main queue here to control depth
        #         continue # Processed this class and its direct methods

        #     elif node.type == NODE_TYPE_METHOD_DECLARATION:
        #         # This handles methods outside classes (e.g. in C# 9+ top-level statements, though less common for full methods)
        #         # Or if traversal logic changes. For now, methods are primarily processed via classes.
        #         if not current_class_name_str: # Only if not already processed as part of a class
        #             method_chunk = self._process_method(node, source_code, file_path, relative_path, class_namespace=current_namespace_str)
        #             if method_chunk:
        #                 chunks.append(method_chunk)
        #         # Do not recurse into method body
        #         continue

        #     elif node.type == NODE_TYPE_NAMESPACE_DECLARATION:
        #         name_node = next((child for child in node.named_children if child.type == NODE_TYPE_IDENTIFIER or child.type == "qualified_name"), None)
        #         ns_name = self._get_node_text(name_node, source_code) if name_node else ""
        #         # If current_namespace_str already exists, append to it.
        #         effective_ns = f"{current_namespace_str}.{ns_name}" if current_namespace_str and ns_name else ns_name or current_namespace_str
        #         for child in node.children:
        #             queue.append((child, effective_ns, None)) # Reset class name when entering new namespace scope
        #         continue # Namespace processed, continue with its children

        #     # Default traversal for other node types to find top-level elements or namespaces
        #     for child in node.children:
        #         # Propagate current namespace and class name if not overridden by child type
        #         queue.append((child, current_namespace_str, current_class_name_str))
    
        def traverse_node(node: Node):
            try:
                if node.type in [NODE_TYPE_CLASS_DECLARATION]:
                    chunks.append(self._process_class(node, source_code, file_path))
                    # Stop at top-level
                    return
                elif node.type == "":
                    # Stop at top-level
                    return
            except Exception as e:
                LOG.error(f"Failed to process code {str(e)}")
                return

            for child in node.children:
                traverse_node(child)
        
        traverse_node(tree.root_node)

        return chunks