import tempfile
from pathlib import Path
from typing import Generator

import git
import pytest

from knowlang.assets.codebase.models import CodeProcessorConfig
from knowlang.configs import LanguageConfig
from knowlang.database.config import VectorStoreConfig
from tests.parser.languages.ts.ts_files import TEST_FILES


@pytest.fixture
def test_config() -> Generator[VectorStoreConfig, None, None]:
    """Provides test configuration"""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = git.Repo.init(temp_dir)

        for filename, content in TEST_FILES.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content)
            repo.index.add([str(file_path)])

        repo.index.commit("Initial commit")

        yield CodeProcessorConfig(
            languages={
                "typescript": LanguageConfig(
                    file_extensions=[".ts", ".tsx"],
                    tree_sitter_language="typescript",
                    max_file_size=1_000_000,
                    chunk_types=[
                        "class_declaration",
                        "function_declaration",
                        "interface_declaration",
                        "type_alias_declaration",
                    ],
                )
            },
            directory_path=str(temp_dir),
        )
