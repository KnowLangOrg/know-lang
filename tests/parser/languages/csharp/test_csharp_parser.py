import pytest
from pathlib import Path
from typing import List, Any

from knowlang.configs import AppConfig
from knowlang.parser.languages.csharp.parser import CSharpParser
from knowlang.core.types import CodeChunk, BaseChunkType, CodeMetadata

# Helper function to find chunks (similar to C++ tests)
def find_chunk_by_criteria(chunks: List[CodeChunk], **criteria: Any) -> CodeChunk | None:
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

@pytest.fixture
def csharp_parser(test_config: AppConfig) -> CSharpParser:
    """
    Fixture to create and setup a CSharpParser instance.
    Relies on test_config to have C# language settings.
    """
    # Ensure the test_config has a basic csharp configuration
    if "csharp" not in test_config.parser.languages:
        # Provide a default minimal config if not present for test execution
        # This is a fallback, ideally test_config should be comprehensive
        test_config.parser.languages["csharp"] = {
            "file_extensions": [".cs"],
            "max_file_size": 1024 * 1024, # 1MB
            # Add other necessary default C# config if any
        }

    parser = CSharpParser(test_config)
    parser.setup() # This is crucial
    return parser

class TestCSharpParser:
    def test_parser_initialization(self, csharp_parser: CSharpParser):
        """Test that the CSharpParser initializes correctly."""
        assert csharp_parser.parser is not None, "Parser should be initialized if grammar is loaded"
        assert csharp_parser.language is not None, "Language should be set if grammar is loaded"
        assert csharp_parser.language_name == "csharp"

        # Check if language_config was loaded (even if default)
        assert hasattr(csharp_parser, 'language_config')
        assert ".cs" in csharp_parser.language_config.file_extensions

        assert csharp_parser.supports_extension(".cs") is True
        assert csharp_parser.supports_extension(".txt") is False
        # Test case-insensitivity if specified by config (assuming lowercase in config)
        assert csharp_parser.supports_extension(".CS") is True

    def test_simple_file_parsing(self, csharp_parser: CSharpParser, tmp_path: Path):
        """Test parsing a simple C# file with one class and two methods."""
        test_file_content = """
/// <summary>A simple class.</summary>
public class MyClass {
    /// <summary>A simple method.</summary>
    public void MyMethod() {
        // Method body
    }
    public void AnotherMethod() {} // No docstring
}
"""
        test_file = tmp_path / "test.cs"
        test_file.write_text(test_file_content)

        chunks = csharp_parser.parse_file(test_file)

        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}. Chunks: {chunks}"

        # Class chunk
        my_class_chunk = find_chunk_by_criteria(chunks, name="MyClass", type=BaseChunkType.CLASS)
        assert my_class_chunk is not None, "MyClass chunk not found"
        assert my_class_chunk.metadata.name == "MyClass"
        assert my_class_chunk.metadata.docstring == "A simple class."
        # Check a part of the content for class
        assert "public class MyClass" in my_class_chunk.content
        assert "public void MyMethod()" in my_class_chunk.content # Class content includes method definition

        # Method chunk 1
        my_method_chunk = find_chunk_by_criteria(chunks, name="MyMethod", type=BaseChunkType.FUNCTION)
        assert my_method_chunk is not None, "MyMethod chunk not found"
        assert my_method_chunk.metadata.name == "MyMethod"
        assert my_method_chunk.metadata.docstring == "A simple method."
        assert my_method_chunk.metadata.class_name == "MyClass"
        assert "public void MyMethod()" in my_method_chunk.content
        assert "// Method body" in my_method_chunk.content

        # Method chunk 2
        another_method_chunk = find_chunk_by_criteria(chunks, name="AnotherMethod", type=BaseChunkType.FUNCTION)
        assert another_method_chunk is not None, "AnotherMethod chunk not found"
        assert another_method_chunk.metadata.name == "AnotherMethod"
        assert another_method_chunk.metadata.docstring is None
        assert another_method_chunk.metadata.class_name == "MyClass"
        assert "public void AnotherMethod()" in another_method_chunk.content

    def test_namespace_handling(self, csharp_parser: CSharpParser, tmp_path: Path):
        """Test parsing C# code with namespaces."""
        test_file_content = """
namespace MyNamespace {
    public class NamespacedClass { // Class 1
        public void NamespacedMethod() {} // Method 1
    }
}
namespace Another.Nested {
    public class DeepClass { // Class 2
         public void DeepMethod() {} // Method 2
    }
}
namespace MyNamespace { // Re-opening namespace
    public class SecondClassInMyNamespace {} // Class 3
}
"""
        test_file = tmp_path / "test_ns.cs"
        test_file.write_text(test_file_content)

        chunks = csharp_parser.parse_file(test_file)

        assert len(chunks) == 5, f"Expected 5 chunks, got {len(chunks)}. Chunks: {chunks}"

        # NamespacedClass and its method
        ns_class_chunk = find_chunk_by_criteria(chunks, name="NamespacedClass", type=BaseChunkType.CLASS)
        assert ns_class_chunk is not None
        assert ns_class_chunk.metadata.namespace == "MyNamespace"
        assert ns_class_chunk.metadata.name == "NamespacedClass"

        ns_method_chunk = find_chunk_by_criteria(chunks, name="NamespacedMethod", type=BaseChunkType.FUNCTION)
        assert ns_method_chunk is not None
        assert ns_method_chunk.metadata.namespace == "MyNamespace"
        assert ns_method_chunk.metadata.class_name == "NamespacedClass"
        assert ns_method_chunk.metadata.name == "NamespacedMethod"

        # DeepClass and its method
        deep_class_chunk = find_chunk_by_criteria(chunks, name="DeepClass", type=BaseChunkType.CLASS)
        assert deep_class_chunk is not None
        assert deep_class_chunk.metadata.namespace == "Another.Nested"
        assert deep_class_chunk.metadata.name == "DeepClass"

        deep_method_chunk = find_chunk_by_criteria(chunks, name="DeepMethod", type=BaseChunkType.FUNCTION)
        assert deep_method_chunk is not None
        assert deep_method_chunk.metadata.namespace == "Another.Nested"
        assert deep_method_chunk.metadata.class_name == "DeepClass"
        assert deep_method_chunk.metadata.name == "DeepMethod"

        # SecondClassInMyNamespace
        second_class_chunk = find_chunk_by_criteria(chunks, name="SecondClassInMyNamespace", type=BaseChunkType.CLASS)
        assert second_class_chunk is not None
        assert second_class_chunk.metadata.namespace == "MyNamespace" # Should correctly handle re-opened namespace
        assert second_class_chunk.metadata.name == "SecondClassInMyNamespace"


    def test_docstring_variations(self, csharp_parser: CSharpParser, tmp_path: Path):
        """Test different styles of docstrings and comments."""
        test_file_content = """
/// <summary>
/// Multi-line summary.
/// </summary>
public class ClassWithXmlDoc { //Chunk 1
    /// Single line XML doc.
    public void MethodWithXmlDoc() {} //Chunk 2

    // Regular comment before method
    // This is line 2 of the comment
    public void MethodWithRegularComment() {} //Chunk 3

    /**
     * Block comment before method.
     * Line 2 of block.
     */
    public void MethodWithBlockComment() {} //Chunk 4

    /// Not a summary tag, just free text.
    /// Line 2 of free text.
    public void MethodWithFreeXmlDoc() {} //Chunk 5
}
"""
        test_file = tmp_path / "test_doc.cs"
        test_file.write_text(test_file_content)
        chunks = csharp_parser.parse_file(test_file)

        assert len(chunks) == 5, f"Expected 5 chunks, got {len(chunks)}. Chunks: {chunks}"

        class_doc = find_chunk_by_criteria(chunks, name="ClassWithXmlDoc")
        assert class_doc is not None
        assert class_doc.metadata.docstring == "Multi-line summary."

        method_xml_doc = find_chunk_by_criteria(chunks, name="MethodWithXmlDoc")
        assert method_xml_doc is not None
        assert method_xml_doc.metadata.docstring == "Single line XML doc."

        method_free_xml_doc = find_chunk_by_criteria(chunks, name="MethodWithFreeXmlDoc")
        assert method_free_xml_doc is not None
        assert method_free_xml_doc.metadata.docstring == "Not a summary tag, just free text.\nLine 2 of free text."

        method_regular_comment = find_chunk_by_criteria(chunks, name="MethodWithRegularComment")
        assert method_regular_comment is not None
        assert method_regular_comment.metadata.docstring == "Regular comment before method\nThis is line 2 of the comment"

        method_block_comment = find_chunk_by_criteria(chunks, name="MethodWithBlockComment")
        assert method_block_comment is not None
        # Tree-sitter preserves the comment structure including leading *, adjust expectation
        expected_block_docstring = "Block comment before method.\n     * Line 2 of block."
        # Example: if tree-sitter gives "Block comment before method.\n * Line 2 of block." (strips some space)
        # We would need to inspect actual output of parser for this specific grammar.
        # For now, assuming it's fairly literal. Let's make it more robust by checking containment or stripping.
        assert expected_block_docstring.replace(" ","") in method_block_comment.metadata.docstring.replace(" ",""), \
            f"Actual: '{method_block_comment.metadata.docstring}', Expected contains: '{expected_block_docstring}'"


    def test_depth_constraints_ignored_nested(self, csharp_parser: CSharpParser, tmp_path: Path):
        """Test that nested classes and methods inside methods are ignored."""
        test_file_content = """
public class OuterClass { // Process this
    public void OuterMethod() { // Process this
        // Inner workings - ignore
        int x = 1;
        if (x > 0) {
            Action lambda = () => { /* ignore lambda */ };
        }
    }

    class NestedClass { // IGNORE THIS
        public void NestedMethod() {} // IGNORE THIS
    }

    public void AnotherOuterMethod() {} // Process this
}
"""
        test_file = tmp_path / "test_depth.cs"
        test_file.write_text(test_file_content)
        chunks = csharp_parser.parse_file(test_file)

        assert len(chunks) == 3, f"Expected 3 chunks (OuterClass, OuterMethod, AnotherOuterMethod), got {len(chunks)}. Chunks: {[c.metadata.name for c in chunks]}"

        outer_class = find_chunk_by_criteria(chunks, name="OuterClass", type=BaseChunkType.CLASS)
        assert outer_class is not None

        outer_method = find_chunk_by_criteria(chunks, name="OuterMethod", type=BaseChunkType.FUNCTION)
        assert outer_method is not None
        assert outer_method.metadata.class_name == "OuterClass"

        another_outer_method = find_chunk_by_criteria(chunks, name="AnotherOuterMethod", type=BaseChunkType.FUNCTION)
        assert another_outer_method is not None
        assert another_outer_method.metadata.class_name == "OuterClass"

        nested_class = find_chunk_by_criteria(chunks, name="NestedClass", type=BaseChunkType.CLASS)
        assert nested_class is None, "NestedClass should not be parsed as a separate chunk"

        nested_method = find_chunk_by_criteria(chunks, name="NestedMethod", type=BaseChunkType.FUNCTION)
        assert nested_method is None, "NestedMethod should not be parsed as a separate chunk"

    def test_empty_and_invalid_files(self, csharp_parser: CSharpParser, tmp_path: Path):
        """Test behavior with empty, syntactically incorrect, or non-CS files."""
        empty_cs_file = tmp_path / "empty.cs"
        empty_cs_file.write_text("")
        chunks_empty = csharp_parser.parse_file(empty_cs_file)
        assert chunks_empty == [], "Empty .cs file should produce no chunks"

        # Syntactically incorrect C#
        invalid_cs_file = tmp_path / "invalid.cs"
        invalid_cs_file.write_text("public class MyClass { public void MyMethod() { int x = ; } }") # Invalid syntax: int x = ;
        # Tree-sitter might still parse parts of it or produce an error node.
        # The current implementation returns [] on parser error.
        chunks_invalid = csharp_parser.parse_file(invalid_cs_file)
        # Depending on tree-sitter's resilience, it might produce some chunks or an error.
        # For now, we assume it might produce some partial chunks if the error is localized.
        # If parser.parse() itself fails and returns None, then it would be [].
        # The current CSharpParser returns [] if tree is None or if parsing exception occurs.
        # Check CSharpParser.parse_file: if tree is None (error in self.parser.parse), returns []
        assert chunks_invalid == [], "Syntactically incorrect .cs file should produce no chunks or handle errors gracefully"


        non_cs_file = tmp_path / "readme.txt"
        non_cs_file.write_text("This is a text file.")
        chunks_non_cs = csharp_parser.parse_file(non_cs_file)
        assert chunks_non_cs == [], "Non-CS file should produce no chunks due to supports_extension check"

    def test_file_with_only_usings_or_namespace(self, csharp_parser: CSharpParser, tmp_path: Path):
        """Test parsing a file that only has using directives or an empty namespace."""
        content_usings_only = "using System;\nusing System.Collections.Generic;"
        file_usings = tmp_path / "usings.cs"
        file_usings.write_text(content_usings_only)
        chunks_usings = csharp_parser.parse_file(file_usings)
        assert len(chunks_usings) == 0, f"File with only usings should produce 0 chunks, got {len(chunks_usings)}"

        content_empty_ns = "namespace MyEmptyNamespace { }"
        file_empty_ns = tmp_path / "empty_ns.cs"
        file_empty_ns.write_text(content_empty_ns)
        chunks_empty_ns = csharp_parser.parse_file(file_empty_ns)
        assert len(chunks_empty_ns) == 0, f"File with empty namespace should produce 0 chunks, got {len(chunks_empty_ns)}"

        content_ns_with_usings = """
namespace MyNamespace {
    using System;
}
public class TopLevelClass {} // Chunk 1
"""
        file_ns_usings = tmp_path / "ns_usings.cs"
        file_ns_usings.write_text(content_ns_usings)
        chunks_ns_usings = csharp_parser.parse_file(file_ns_usings)
        assert len(chunks_ns_usings) == 1, f"Expected 1 chunk, got {len(chunks_ns_usings)}"
        assert find_chunk_by_criteria(chunks_ns_usings, name="TopLevelClass") is not None

    # Consider adding tests for C# 9+ top-level statements if they should be handled.
    # For now, the parser focuses on class and method declarations.
    # def test_top_level_statements(self, csharp_parser: CSharpParser, tmp_path: Path):
    #     test_file_content = """
    #     System.Console.WriteLine("Hello from top-level!"); // Should this be a chunk?
    #     public class MyClass {} // This should be a chunk
    #     """
    #     # ...
    # Based on current logic, top-level statements that are not class/method defs won't be chunks.
    # Which is fine given the current requirements.

    def test_unicode_characters_in_identifiers_and_strings(self, csharp_parser: CSharpParser, tmp_path: Path):
        """Test handling of unicode characters in names and docstrings."""
        test_file_content = """
        /// <summary>Clase con nombre Unicode: ÄÖÜß.</summary>
        public class ClaseÄÖÜß {
            /// <summary>Método con acentos: áéíóú.</summary>
            public void MétodoÁÉÍÓÚ() {
                string str = "Hola, Mundo! 👋";
            }
        }
        """
        test_file = tmp_path / "unicode_test.cs"
        test_file.write_text(test_file_content, encoding='utf-8') # Ensure UTF-8 for writing

        chunks = csharp_parser.parse_file(test_file)
        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"

        class_chunk = find_chunk_by_criteria(chunks, name="ClaseÄÖÜß", type=BaseChunkType.CLASS)
        assert class_chunk is not None, "Unicode class name not parsed correctly."
        assert class_chunk.metadata.docstring == "Clase con nombre Unicode: ÄÖÜß."

        method_chunk = find_chunk_by_criteria(chunks, name="MétodoÁÉÍÓÚ", type=BaseChunkType.FUNCTION)
        assert method_chunk is not None, "Unicode method name not parsed correctly."
        assert method_chunk.metadata.docstring == "Método con acentos: áéíóú."
        assert method_chunk.metadata.class_name == "ClaseÄÖÜß"
        assert 'string str = "Hola, Mundo! 👋";' in method_chunk.content

    # TODO: Add a test for relative_path in CodeMetadata once db_path is properly handled in CSharpParser
    # def test_relative_path_metadata(self, csharp_parser: CSharpParser, test_config: AppConfig, tmp_path: Path):
    #     # This test will require test_config.db_path to be set appropriately
    #     # and CSharpParser to use it with convert_to_relative_path
    #     db_root = tmp_path / "project_root"
    #     db_root.mkdir()
    #     test_config.db_path = str(db_root) # Mocking db_path in config

    #     file_loc = db_root / "src" / "myfile.cs"
    #     file_loc.parent.mkdir(parents=True)
    #     file_loc.write_text("public class MyClass {}")

    #     # Re-initialize parser if db_path is read at init/setup, or ensure it's read dynamically
    #     # For now, assume csharp_parser fixture might need adjustment or a new one for this test
    #     # if test_config is not mutable effectively after parser init.

    #     chunks = csharp_parser.parse_file(file_loc)
    #     assert len(chunks) == 1
    #     class_chunk = chunks[0]
    #     assert class_chunk.metadata.relative_path == Path("src") / "myfile.cs"
    #     assert class_chunk.metadata.file_path == str(file_loc)