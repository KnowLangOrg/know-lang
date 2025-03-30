import tempfile
from pathlib import Path
from typing import List

import pytest

from knowlang.configs import AppConfig
from knowlang.core.types import BaseChunkType, CodeChunk
from knowlang.parser.languages.ts.parser import TypeScriptParser
from tests.parser.languages.ts.ts_files import (
    COMPLEX_FILE_EXPECTATIONS,
    INVALID_TS,
    REACT_FILE_EXPECTATIONS,
    SIMPLE_FILE_EXPECTATIONS,
    TEST_FILES,
)


@pytest.fixture
def typescript_parser(test_config):
    """Provides initialized TypeScript parser"""
    parser = TypeScriptParser(test_config)
    parser.setup()
    return parser


def find_chunk_by_criteria(chunks: List[CodeChunk], **criteria) -> CodeChunk:
    """Helper function to find a chunk matching given criteria"""
    for chunk in chunks:
        if all(getattr(chunk, k) == v for k, v in criteria.items()):
            return chunk
    return None


def verify_chunk_matches_expectation(
    chunk: CodeChunk,
    expected_name: str,
    expected_docstring: str,
    expected_content_snippet: str
) -> bool:
    """Verify that a chunk matches expected values"""
    return (
        chunk.name == expected_name and
        expected_content_snippet in chunk.content and
        expected_docstring in (chunk.docstring or "")
    )


class TestTypeScriptParser:
    """Test suite for TypeScriptParser"""

    def test_parser_initialization(self, typescript_parser: TypeScriptParser):
        """Test parser initialization"""
        assert typescript_parser.parser is not None
        assert typescript_parser.language is not None
        assert typescript_parser.language_name == "typescript"
        assert typescript_parser.supports_extension(".ts")
        assert typescript_parser.supports_extension(".tsx")

    def test_simple_file_parsing(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test parsing a simple TypeScript file with function, class, interface, and type alias"""
        chunks = typescript_parser.parse_file(test_config.db.codebase_directory / "simple.ts")
        
        # Test function extraction
        function_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.FUNCTION,
            name="helloWorld"
        )
        assert function_chunk is not None
        expected = SIMPLE_FILE_EXPECTATIONS['helloWorld']
        assert verify_chunk_matches_expectation(
            function_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )

        # Test class extraction
        class_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.CLASS,
            name="Counter"
        )
        assert class_chunk is not None
        expected = SIMPLE_FILE_EXPECTATIONS['Counter']
        assert verify_chunk_matches_expectation(
            class_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Test method extraction
        method_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.FUNCTION,
            name="increment"
        )
        assert method_chunk is not None
        expected = SIMPLE_FILE_EXPECTATIONS['increment']
        assert verify_chunk_matches_expectation(
            method_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        assert method_chunk.parent_name == "Counter"
        
        # Test interface extraction
        interface_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.INTERFACE,
            name="Person"
        )
        assert interface_chunk is not None
        expected = SIMPLE_FILE_EXPECTATIONS['Person']
        assert verify_chunk_matches_expectation(
            interface_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Test type alias extraction
        type_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.TYPE_ALIAS,
            name="User"
        )
        assert type_chunk is not None
        expected = SIMPLE_FILE_EXPECTATIONS['User']
        assert verify_chunk_matches_expectation(
            type_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )

    def test_complex_file_parsing(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test parsing a complex TypeScript file with generics, namespaces, and decorators"""
        chunks = typescript_parser.parse_file(test_config.db.codebase_directory / "complex.ts")
        
        # Test decorator function
        decorator_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.FUNCTION,
            name="log"
        )
        assert decorator_chunk is not None
        expected = COMPLEX_FILE_EXPECTATIONS['log']
        assert verify_chunk_matches_expectation(
            decorator_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Test generic class in namespace
        repository_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.CLASS,
            name="Repository"
        )
        assert repository_chunk is not None
        expected = COMPLEX_FILE_EXPECTATIONS['Repository']
        assert verify_chunk_matches_expectation(
            repository_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        assert repository_chunk.metadata.namespace == "Utils"
        assert repository_chunk.metadata.is_generic
        
        # Test decorated method
        method_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.FUNCTION,
            name="getAll"
        )
        assert method_chunk is not None
        expected = COMPLEX_FILE_EXPECTATIONS['getAll']
        assert verify_chunk_matches_expectation(
            method_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        assert method_chunk.parent_name == "Repository"
        
        # Test generic interface
        interface_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.INTERFACE,
            name="ApiConfig"
        )
        assert interface_chunk is not None
        expected = COMPLEX_FILE_EXPECTATIONS['ApiConfig']
        assert verify_chunk_matches_expectation(
            interface_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Test generic type alias
        type_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.TYPE_ALIAS,
            name="ApiResult"
        )
        assert type_chunk is not None
        expected = COMPLEX_FILE_EXPECTATIONS['ApiResult']
        assert verify_chunk_matches_expectation(
            type_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        assert type_chunk.metadata.is_generic
        
        # Test arrow function
        arrow_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.FUNCTION,
            name="fetchData"
        )
        assert arrow_chunk is not None
        expected = COMPLEX_FILE_EXPECTATIONS['fetchData']
        assert verify_chunk_matches_expectation(
            arrow_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )

    def test_react_file_parsing(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test parsing a TypeScript React file (.tsx)"""
        chunks = typescript_parser.parse_file(test_config.db.codebase_directory / "component.tsx")
        
        # Test interface for props
        props_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.INTERFACE,
            name="CounterProps"
        )
        assert props_chunk is not None
        expected = REACT_FILE_EXPECTATIONS['CounterProps']
        assert verify_chunk_matches_expectation(
            props_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Test function component
        component_chunk = find_chunk_by_criteria(
            chunks,
            type=BaseChunkType.FUNCTION,
            name="Counter"
        )
        assert component_chunk is not None
        expected = REACT_FILE_EXPECTATIONS['Counter']
        assert verify_chunk_matches_expectation(
            component_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )

    def test_error_handling(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test error handling for various error cases"""
        # Test invalid syntax
        invalid_file = Path(test_config.db.codebase_directory) / "invalid.ts"
        chunks = typescript_parser.parse_file(invalid_file)
        # Should still try to extract what it can from invalid files
        assert chunks != []
        
        # Test non-existent file
        nonexistent = Path(test_config.db.codebase_directory) / "nonexistent.ts"
        chunks = typescript_parser.parse_file(nonexistent)
        assert chunks == []
        
        # Test non-TypeScript file
        non_ts = Path(test_config.db.codebase_directory) / "readme.md"
        non_ts.write_text("# README")
        chunks = typescript_parser.parse_file(non_ts)
        assert chunks == []

    def test_file_size_limits(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test file size limit enforcement"""
        large_file = Path(test_config.db.codebase_directory) / "large.ts"
        # Create a file larger than the limit
        large_file.write_text("const x = 1;\n" * 1_000_000)
        
        chunks = typescript_parser.parse_file(large_file)
        assert chunks == []