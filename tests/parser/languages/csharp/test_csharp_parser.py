from pathlib import Path
from typing import Any, List

import pytest

from knowlang.assets.codebase.models import CodeProcessorConfig
from knowlang.configs import LanguageConfig
from knowlang.core.types import CodeChunk
from knowlang.parser.languages.csharp.parser import CSharpParser

# Import the new fixtures
from .fixtures.expectations import (
    ChunkExpectation,
    get_test_file_path,
    load_test_expectations,
)


# Helper function to find chunks (similar to C++ tests)
def find_chunk_by_criteria(
    chunks: List[CodeChunk], **criteria: Any
) -> CodeChunk | None:
    """
    Finds the first CodeChunk in a list that matches all given criteria.
    Criteria are checked against the chunk's metadata.
    """
    for chunk in chunks:
        match = True
        for key, value in criteria.items():
            # Handle direct attributes of CodeChunk (like 'content') and metadata attributes
            if hasattr(chunk, key) and getattr(chunk, key) == value:
                continue
            if hasattr(chunk.metadata, key) and getattr(chunk.metadata, key) == value:
                continue
            match = False
            break
        if match:
            return chunk
    return None


def validate_chunk_against_expectation(
    chunk: CodeChunk, expectation: ChunkExpectation
) -> None:
    """Helper to validate a chunk against its expectation"""
    assert chunk.name == expectation.name, (
        f"Expected name {expectation.name}, got {chunk.name}"
    )

    if expectation.docstring is not None:
        assert expectation.docstring in chunk.docstring, (
            f"Expected docstring '{expectation.docstring}', got '{chunk.docstring}'"
        )
    else:
        assert chunk.docstring is None, (
            f"Expected no docstring, but got '{chunk.docstring}'"
        )

    assert expectation.content_snippet in chunk.content, (
        f"Expected content snippet '{expectation.content_snippet}' not found in chunk content"
    )

    if expectation.namespace is not None:
        assert chunk.metadata.namespace == expectation.namespace, (
            f"Expected namespace {expectation.namespace}, got {chunk.metadata.namespace}"
        )

    if expectation.class_name is not None:
        assert chunk.name == expectation.class_name, (
            f"Expected class_name {expectation.class_name}, got {chunk.name}"
        )


@pytest.fixture
def csharp_parser() -> CSharpParser:
    processor_config = CodeProcessorConfig(
        directory_path=str(Path(__file__).parent),
        languages={
            "csharp": LanguageConfig(
                file_extensions=[".cs"],
                tree_sitter_language="csharp",
                chunk_types=[
                    "class_declaration",
                    "method_declaration",
                ],  # Using common tree-sitter type names
                max_file_size=1_000_000,
            )
        },
    )
    parser = CSharpParser(processor_config)
    parser.setup()
    return parser


@pytest.fixture(scope="class")
def test_expectations():
    """Fixture to load test expectations"""
    return load_test_expectations()


class TestCSharpParser:
    def test_parser_initialization(self, csharp_parser: CSharpParser):
        """Test that the CSharpParser initializes correctly."""
        assert csharp_parser.parser is not None, (
            "Parser should be initialized if grammar is loaded"
        )
        assert csharp_parser.language is not None, (
            "Language should be set if grammar is loaded"
        )
        assert csharp_parser.language_name == "csharp"

        # Check if language_config was loaded (even if default)
        assert hasattr(csharp_parser, "language_config")
        assert ".cs" in csharp_parser.language_config.file_extensions

        assert csharp_parser.supports_extension(".cs") is True
        assert csharp_parser.supports_extension(".txt") is False
        # Test case-insensitivity if specified by config (assuming lowercase in config)
        assert csharp_parser.supports_extension(".CS") is True

    @pytest.mark.asyncio
    async def test_simple_file_parsing(
        self, csharp_parser: CSharpParser, test_expectations
    ):
        """Test parsing the simple.cs file with basic classes and methods."""
        test_file = get_test_file_path("simple.cs")
        chunks = await csharp_parser.parse_file(test_file)

        expectations = test_expectations["simple.cs"]

        # We expect at least the major chunks (HelloWorld class, Person class, and their methods)
        assert len(chunks) == 2, (
            f"Expected 2 chunks, got {len(chunks)}. Chunks: {[c.name for c in chunks]}"
        )

        # Validate each expected chunk
        for expected_name, expectation in expectations.items():
            chunk = find_chunk_by_criteria(chunks, name=expected_name)
            assert chunk is not None, f"Expected chunk '{expected_name}' not found"
            validate_chunk_against_expectation(chunk, expectation)

    @pytest.mark.asyncio
    async def test_complex_file_parsing(
        self, csharp_parser: CSharpParser, test_expectations
    ):
        """Test parsing the complex.cs file with namespaces, generics, and attributes."""
        test_file = get_test_file_path("complex.cs")
        chunks = await csharp_parser.parse_file(test_file)

        expectations = test_expectations["complex.cs"]

        # We expect multiple chunks across different namespaces
        assert len(chunks) == len(expectations), (
            f"Expected at least {len(expectations)} chunks, got {len(chunks)}"
        )

        # Validate each expected chunk
        for expected_name, expectation in expectations.items():
            chunk = find_chunk_by_criteria(chunks, name=expected_name)
            assert chunk is not None, f"Expected chunk '{expected_name}' not found"
            validate_chunk_against_expectation(chunk, expectation)

    @pytest.mark.asyncio
    async def test_nested_file_parsing(
        self, csharp_parser: CSharpParser, test_expectations
    ):
        """Test parsing the nested.cs file with nested classes and static classes."""
        test_file = get_test_file_path("nested.cs")
        chunks = await csharp_parser.parse_file(test_file)

        expectations = test_expectations["nested.cs"]

        assert len(chunks) >= len(expectations), (
            f"Expected at least {len(expectations)} chunks, got {len(chunks)}"
        )

        # Validate each expected chunk
        for expected_name, expectation in expectations.items():
            chunk = find_chunk_by_criteria(chunks, name=expected_name)
            assert chunk is not None, f"Expected chunk '{expected_name}' not found"
            validate_chunk_against_expectation(chunk, expectation)

    @pytest.mark.asyncio
    async def test_advanced_features_file_parsing(
        self, csharp_parser: CSharpParser, test_expectations
    ):
        """Test parsing the advanced_features.cs file with events, delegates, and properties."""
        test_file = get_test_file_path("advanced_features.cs")
        chunks = await csharp_parser.parse_file(test_file)

        expectations = test_expectations["advanced_features.cs"]

        assert len(chunks) >= len(expectations), (
            f"Expected at least {len(expectations)} chunks, got {len(chunks)}"
        )

        # Validate each expected chunk
        for expected_name, expectation in expectations.items():
            chunk = find_chunk_by_criteria(chunks, name=expected_name)
            assert chunk is not None, f"Expected chunk '{expected_name}' not found"
            validate_chunk_against_expectation(chunk, expectation)

    @pytest.mark.asyncio
    async def test_modern_csharp_file_parsing(
        self, csharp_parser: CSharpParser, test_expectations
    ):
        """Test parsing the modern.cs file with C# 9+ features like records."""
        test_file = get_test_file_path("modern.cs")
        chunks = await csharp_parser.parse_file(test_file)

        expectations = test_expectations["modern.cs"]

        assert len(chunks) >= len(expectations), (
            f"Expected at least {len(expectations)} chunks, got {len(chunks)}"
        )

        # Validate each expected chunk
        for expected_name, expectation in expectations.items():
            chunk = find_chunk_by_criteria(chunks, name=expected_name)
            assert chunk is not None, f"Expected chunk '{expected_name}' not found"
            validate_chunk_against_expectation(chunk, expectation)

    @pytest.mark.asyncio
    async def test_invalid_file_handling(self, csharp_parser: CSharpParser):
        """Test that invalid C# files are handled gracefully."""
        test_file = get_test_file_path("invalid.cs")
        chunks = await csharp_parser.parse_file(test_file)

        # Invalid files should either return empty list or handle errors gracefully
        # The exact behavior depends on tree-sitter's error recovery
        assert isinstance(chunks, list), "Should return a list even for invalid files"

    @pytest.mark.asyncio
    async def test_minimal_file_parsing(self, csharp_parser: CSharpParser):
        """Test parsing a minimal file with only using statements and empty namespace."""
        test_file = get_test_file_path("minimal.cs")
        chunks = await csharp_parser.parse_file(test_file)

        # Should produce no chunks since there are no classes or methods
        assert len(chunks) == 0, (
            f"Expected 0 chunks for minimal file, got {len(chunks)}"
        )

    @pytest.mark.parametrize(
        "test_file",
        ["simple.cs", "complex.cs", "nested.cs", "advanced_features.cs", "modern.cs"],
    )
    @pytest.mark.asyncio
    async def test_all_files_parse_successfully(
        self, csharp_parser: CSharpParser, test_file: str, test_expectations
    ):
        """Test that all major test files parse without errors."""
        file_path = get_test_file_path(test_file)
        chunks = await csharp_parser.parse_file(file_path)

        # Should return a list (even if empty)
        assert isinstance(chunks, list), f"Failed to parse {test_file}"

        # If we have expectations for this file, check basic structure
        if test_file in test_expectations:
            expectations = test_expectations[test_file]
            if expectations:  # If we expect chunks
                assert len(chunks) > 0, f"Expected chunks for {test_file} but got none"

    @pytest.mark.asyncio
    async def test_namespace_handling(
        self, csharp_parser: CSharpParser, test_expectations
    ):
        """Test that namespaces are correctly identified and assigned to chunks."""
        test_file = get_test_file_path("complex.cs")
        chunks = await csharp_parser.parse_file(test_file)

        # Check that chunks in MyApplication.Services namespace are correctly identified
        repo_chunk = find_chunk_by_criteria(chunks, name="IRepository")
        assert repo_chunk is not None
        assert repo_chunk.metadata.namespace == "MyApplication.Services"

        # Check that chunks in MyApplication.Models namespace are correctly identified
        user_chunk = find_chunk_by_criteria(chunks, name="User")
        assert user_chunk is not None
        assert user_chunk.metadata.namespace == "MyApplication.Models"

    @pytest.mark.asyncio
    async def test_generic_types_handling(self, csharp_parser: CSharpParser):
        """Test that generic types are handled correctly."""
        test_file = get_test_file_path("complex.cs")
        chunks = await csharp_parser.parse_file(test_file)

        # Find the generic repository
        generic_repo_chunk = find_chunk_by_criteria(chunks, name="GenericRepository")
        assert generic_repo_chunk is not None
        assert "GenericRepository<T>" in generic_repo_chunk.content

        # Find the interface
        interface_chunk = find_chunk_by_criteria(chunks, name="IRepository")
        assert interface_chunk is not None
        assert "IRepository<T>" in interface_chunk.content

    def test_extension_support(self, csharp_parser: CSharpParser):
        """Test that the parser correctly identifies supported file extensions."""
        assert csharp_parser.supports_extension(".cs")
        assert csharp_parser.supports_extension(".CS")  # Case insensitive
        assert not csharp_parser.supports_extension(".txt")
        assert not csharp_parser.supports_extension(".py")
        assert not csharp_parser.supports_extension(".js")

    @pytest.mark.asyncio
    async def test_empty_file(self, csharp_parser: CSharpParser):
        """Test parsing an empty C# file."""
        import os
        import tempfile

        temp_fd, temp_file = tempfile.mkstemp(suffix=".cs", dir=Path(__file__).parent)
        with os.fdopen(temp_fd, "w") as f:
            f.write("")  # Create an empty file
        chunks = await csharp_parser.parse_file(Path(temp_file))
        assert chunks == [], "Empty file should produce no chunks"

        # Clean up the temporary file
        os.remove(temp_file)

    def test_all_test_files_exist(self):
        """Test that all expected test files exist."""
        expected_files = [
            "simple.cs",
            "complex.cs",
            "nested.cs",
            "advanced_features.cs",
            "invalid.cs",
            "minimal.cs",
            "modern.cs",
        ]

        for filename in expected_files:
            file_path = get_test_file_path(filename)
            assert file_path.exists(), (
                f"Test file {filename} does not exist at {file_path}"
            )
            assert file_path.is_file(), f"Test file {filename} is not a file"

    def test_expectations_completeness(self, test_expectations):
        """Test that expectations are defined for all major test files."""
        expected_files_with_expectations = [
            "simple.cs",
            "complex.cs",
            "nested.cs",
            "advanced_features.cs",
            "modern.cs",
        ]

        for filename in expected_files_with_expectations:
            assert filename in test_expectations, (
                f"No expectations defined for {filename}"
            )
            expectations = test_expectations[filename]
            assert len(expectations) > 0, f"Empty expectations for {filename}"
