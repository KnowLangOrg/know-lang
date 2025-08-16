from pathlib import Path
from typing import List
import aiofiles

from knowlang.core.types import BaseChunkType, CodeChunk, CodeLocation, LanguageEnum
from knowlang.parser.base.parser import LanguageParser
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


class MarkdownParser(LanguageParser):
    """Markdown-specific implementation of LanguageParser"""

    def setup(self) -> None:
        """No setup required for Markdown parser"""
        self.language_name = LanguageEnum.MARKDOWN
        self.language_config = self.config.languages[LanguageEnum.MARKDOWN.value]

    async def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single Markdown file and return list of chunks"""
        if not self.supports_extension(file_path.suffix):
            LOG.debug(f"Skipping file {file_path}: unsupported extension")
            return []

        try:
            if file_path.stat().st_size > self.language_config.max_file_size:
                LOG.warning(
                    f"Skipping file {file_path}: exceeds size limit of {self.language_config.max_file_size} bytes"
                )
                return []

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                lines = await f.readlines()

            chunks: List[CodeChunk] = []
            current_chunk_content = []
            current_chunk_name = "Introduction"
            current_chunk_start_line = 1

            for i, line in enumerate(lines):
                if line.startswith("#"):
                    if current_chunk_content:
                        chunks.append(
                            CodeChunk(
                                language=self.language_name,
                                type=BaseChunkType.SECTION,
                                name=current_chunk_name,
                                content="".join(current_chunk_content),
                                location=CodeLocation(
                                    file_path=str(file_path),
                                    start_line=current_chunk_start_line,
                                    end_line=i - 1,
                                ),
                            )
                        )
                    current_chunk_name = line.strip().lstrip("#").strip()
                    current_chunk_content = [line]
                    current_chunk_start_line = i + 1
                else:
                    current_chunk_content.append(line)

            if current_chunk_content:
                chunks.append(
                    CodeChunk(
                        language=self.language_name,
                        type=BaseChunkType.SECTION,
                        name=current_chunk_name,
                        content="".join(current_chunk_content),
                        location=CodeLocation(
                            file_path=str(file_path),
                            start_line=current_chunk_start_line,
                            end_line=len(lines),
                        ),
                    )
                )

            return chunks

        except Exception as e:
            LOG.error(f"Error parsing file {file_path}: {str(e)}")
            return []

    def supports_extension(self, ext: str) -> bool:
        """Check if this parser supports a given file extension"""
        return ext in self.language_config.file_extensions
