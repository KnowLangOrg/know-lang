from pathlib import Path
from typing import List
import pytest
from knowlang.configs import AppConfig
from knowlang.core.types import CodeChunk
from knowlang.parser.languages.ts.parser import TypeScriptParser, TypescriptChunkType
from tests.parser.languages.ts.ts_files import (
    COMPLEX_FILE_EXPECTATIONS,
    COMPLEX_TSX_EXPECTATIONS,
    INVALID_TS,
    INVALID_TSX,
    SIMPLE_FILE_EXPECTATIONS,
    SIMPLE_TSX_EXPECTATIONS,
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
        assert typescript_parser.parser_ts is not None
        assert typescript_parser.parser_tsx is not None
        assert typescript_parser.language_ts is not None
        assert typescript_parser.language_tsx is not None
        assert typescript_parser.language_name == "typescript"
        assert typescript_parser.supports_extension(".ts")
        assert typescript_parser.supports_extension(".tsx")

    def test_get_parser_for_file(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test the parser selection based on file extension"""
        ts_file = Path(test_config.db.codebase_directory) / "test.ts"
        tsx_file = Path(test_config.db.codebase_directory) / "test.tsx"
        
        assert typescript_parser._get_parser_for_file(ts_file) == typescript_parser.parser_ts
        assert typescript_parser._get_parser_for_file(tsx_file) == typescript_parser.parser_tsx

    @pytest.mark.asyncio
    async def test_simple_ts_file_parsing(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test parsing a simple TypeScript file with function, class, interface, and type alias"""
        chunks = await typescript_parser.parse_file(test_config.db.codebase_directory / "simple.ts")
        
        # Test function extraction
        function_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.FUNCTION,
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
            type=TypescriptChunkType.CLASS,
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
        
        # Test interface extraction
        interface_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.INTERFACE,
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
            type=TypescriptChunkType.TYPE_ALIAS,
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
        
        # Verify we don't extract methods from classes since we stop traversal
        increment_method = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.FUNCTION,
            name="increment"
        )
        assert increment_method is None, "Methods inside classes should not be extracted"

    @pytest.mark.asyncio
    async def test_complex_ts_file_parsing(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test parsing a complex TypeScript file with generics, namespaces, and decorators"""
        chunks = await typescript_parser.parse_file(test_config.db.codebase_directory / "complex.ts")
        
        
        # Test deprecated decorator function
        deprecated_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.FUNCTION,
            name="deprecated"
        )
        assert deprecated_chunk is not None
        expected = COMPLEX_FILE_EXPECTATIONS['deprecated']
        assert verify_chunk_matches_expectation(
            deprecated_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        

        # Test decorated class at top level
        decorated_class = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.CLASS,
            name="DecoratedService"
        )
        assert decorated_class is not None
        expected = COMPLEX_FILE_EXPECTATIONS['DecoratedService']
        assert verify_chunk_matches_expectation(
            decorated_class,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        # Verify decorator is included in the content
        assert '@deprecated("Use NewService instead")' in decorated_class.content
        
        # Test generic class in namespace - should be found since we traverse namespaces
        repository_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.CLASS,
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
        
        # Test generic interface
        interface_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.INTERFACE,
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
            type=TypescriptChunkType.TYPE_ALIAS,
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
        
        # Test arrow function
        arrow_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.FUNCTION,
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

    @pytest.mark.asyncio
    async def test_simple_tsx_file_parsing(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test parsing a simple React TSX file"""
        chunks = await typescript_parser.parse_file(test_config.db.codebase_directory / "simple.tsx")
        
        # Test interface for props
        props_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.INTERFACE,
            name="ButtonProps"
        )
        assert props_chunk is not None
        expected = SIMPLE_TSX_EXPECTATIONS['ButtonProps']
        assert verify_chunk_matches_expectation(
            props_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Test React functional component - this is a top-level variable declaration 
        # with an arrow function, should still be found
        component_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.FUNCTION,
            name="Button"
        )
        assert component_chunk is not None
        expected = SIMPLE_TSX_EXPECTATIONS['Button']
        assert verify_chunk_matches_expectation(
            component_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )

    @pytest.mark.asyncio
    async def test_complex_tsx_file_parsing(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test parsing a complex React TSX file"""
        chunks = await typescript_parser.parse_file(test_config.db.codebase_directory / "complex.tsx")
        
        # Test interface for user data
        user_interface_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.INTERFACE,
            name="User"
        )
        assert user_interface_chunk is not None
        expected = COMPLEX_TSX_EXPECTATIONS['User']
        assert verify_chunk_matches_expectation(
            user_interface_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Test interface for props
        props_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.INTERFACE,
            name="UserListProps"
        )
        assert props_chunk is not None
        expected = COMPLEX_TSX_EXPECTATIONS['UserListProps']
        assert verify_chunk_matches_expectation(
            props_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Test React functional component - top-level variable declaration
        component_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.FUNCTION,
            name="UserList"
        )
        assert component_chunk is not None
        expected = COMPLEX_TSX_EXPECTATIONS['UserList']
        assert verify_chunk_matches_expectation(
            component_chunk,
            expected.name,
            expected.docstring,
            expected.content_snippet
        )
        
        # Hook functions defined inside the component function should not be found
        # since we stop traversal at the component function
        hook_chunk = find_chunk_by_criteria(
            chunks,
            type=TypescriptChunkType.FUNCTION,
            name="loadUsers"
        )
        assert hook_chunk is None, "Functions inside other functions should not be extracted"

    @pytest.mark.asyncio
    async def test_error_handling(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test error handling for various error cases"""
        # Test invalid TS syntax
        invalid_ts_file = Path(test_config.db.codebase_directory) / "invalid.ts"
        chunks_ts = await typescript_parser.parse_file(invalid_ts_file)
        # Should still try to extract what it can from invalid files
        assert chunks_ts != []
        
        # Test invalid TSX syntax
        invalid_tsx_file = Path(test_config.db.codebase_directory) / "invalid.tsx"
        chunks_tsx = await typescript_parser.parse_file(invalid_tsx_file)
        # Should still try to extract what it can from invalid files
        assert chunks_tsx != []
        
        # Test non-existent file
        nonexistent = Path(test_config.db.codebase_directory) / "nonexistent.ts"
        chunks = await typescript_parser.parse_file(nonexistent)
        assert chunks == []
        
        # Test non-TypeScript file
        non_ts = Path(test_config.db.codebase_directory) / "readme.md"
        non_ts.write_text("# README")
        chunks = await typescript_parser.parse_file(non_ts)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_file_size_limits(self, typescript_parser: TypeScriptParser, test_config: AppConfig):
        """Test file size limit enforcement"""
        # Test TS file size limit
        large_ts_file = Path(test_config.db.codebase_directory) / "large.ts"
        large_ts_file.write_text("const x = 1;\n" * 1_000_000)
        chunks_ts = await typescript_parser.parse_file(large_ts_file)
        assert chunks_ts == []
        
        # Test TSX file size limit
        large_tsx_file = Path(test_config.db.codebase_directory) / "large.tsx"
        large_tsx_file.write_text("const x = <div>Test</div>;\n" * 1_000_000)
        chunks_tsx = await typescript_parser.parse_file(large_tsx_file)
        assert chunks_tsx == []