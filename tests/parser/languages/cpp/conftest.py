import tempfile
from typing import Generator

import pytest

from knowlang.assets.codebase.models import CodeProcessorConfig
from knowlang.configs import LanguageConfig


@pytest.fixture
def test_config() -> Generator[CodeProcessorConfig, None, None]:
    """Provides test configuration"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield CodeProcessorConfig(
            languages={
                "cpp": LanguageConfig(
                    file_extensions=[".cpp", ".hpp", ".h"],
                    tree_sitter_language="cpp",
                    max_file_size=1_000_000,
                    chunk_types=["class_definition", "function_definition"],
                )
            },
            directory_path=str(temp_dir),
        )
