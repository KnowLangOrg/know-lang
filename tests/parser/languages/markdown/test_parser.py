import asyncio
from pathlib import Path
import pytest
from knowlang.assets.codebase.models import CodeProcessorConfig
from knowlang.parser.languages.markdown.parser import MarkdownParser

@pytest.fixture
def markdown_parser() -> MarkdownParser:
    config = CodeProcessorConfig()
    parser = MarkdownParser(config)
    parser.setup()
    return parser

@pytest.mark.asyncio
async def test_parse_markdown_file(markdown_parser: MarkdownParser, tmp_path: Path):
    content = """\
# Header 1

This is the first section.

## Header 2

This is the second section.
It has two lines.

# Header 3

This is the third section.
"""
    md_file = tmp_path / "test.md"
    md_file.write_text(content)

    chunks = await markdown_parser.parse_file(md_file)

    assert len(chunks) == 3

    # Check first chunk
    assert chunks[0].name == "Header 1"
    assert chunks[0].location.start_line == 1
    assert chunks[0].location.end_line == 3
    assert "This is the first section." in chunks[0].content

    # Check second chunk
    assert chunks[1].name == "Header 2"
    assert chunks[1].location.start_line == 5
    assert chunks[1].location.end_line == 8
    assert "This is the second section." in chunks[1].content
    assert "It has two lines." in chunks[1].content

    # Check third chunk
    assert chunks[2].name == "Header 3"
    assert chunks[2].location.start_line == 10
    assert chunks[2].location.end_line == 12
    assert "This is the third section." in chunks[2].content
