import tempfile
from pathlib import Path
from typing import Generator
import git
import pytest
from knowlang.configs import AppConfig, DBConfig, LanguageConfig, ParserConfig
from knowlang.assets.codebase.models import CodeProcessorConfig
from tests.parser.languages.python.python_files import TEST_FILES


@pytest.fixture
def test_config() -> Generator[CodeProcessorConfig, None, None]:
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
                "python": LanguageConfig(
                    file_extensions=[".py"],
                    tree_sitter_language="python",
                    max_file_size=1_000_000,
                    chunk_types=["class_definition", "function_definition"]
                )
            },
            directory_path=str(temp_dir),
        )
