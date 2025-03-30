import tempfile
from pathlib import Path
from typing import Generator
import git
import pytest
from knowlang.configs import AppConfig, DBConfig, LanguageConfig, ParserConfig
from tests.parser.languages.ts.ts_files import TEST_FILES


@pytest.fixture
def test_config() -> Generator[AppConfig, None, None]:
    """Provides test configuration"""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = git.Repo.init(temp_dir)
        
        for filename, content in TEST_FILES.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content)
            repo.index.add([str(file_path)])
        
        repo.index.commit("Initial commit")
        
        yield AppConfig(
            parser=ParserConfig(
                languages={
                    "typescript": LanguageConfig(
                        file_extensions=[".ts", ".tsx"],
                        tree_sitter_language="typescript",
                        max_file_size=1_000_000,
                        chunk_types=["class_declaration", "function_declaration", "interface_declaration", "type_alias_declaration"]
                    )
                }
            ),
            db=DBConfig(
                codebase_directory=Path(temp_dir)
            )
        )