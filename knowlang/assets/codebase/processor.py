from typing import List, AsyncGenerator, TypeAlias
import os
import aiofiles
from knowlang.assets.processor import (
    DomainAssetSourceMixin,
    DomainAssetIndexingMixin,
    DomainAssetParserMixin,
    DomainContext,
)
from knowlang.assets.codebase.models import (
    CodebaseMetaData,
    CodeAssetMetaData,
    CodeAssetChunkMetaData,
    CodebaseManagerData,
    CodeProcessorConfig,
)
from knowlang.assets.models import (
    GenericAssetData,
    GenericAssetChunkData,
)
from knowlang.parser.factory import CodeParserFactory

# Type aliases to eliminate repetition
CodebaseDomainType: TypeAlias = CodebaseManagerData[CodebaseMetaData]
CodebaseAssetType: TypeAlias = GenericAssetData[CodeAssetMetaData]
CodebaseChunkType: TypeAlias = GenericAssetChunkData[CodeAssetChunkMetaData]
CodebaseConfigType: TypeAlias = CodeProcessorConfig

# Main context type alias
CodebaseDomainContext: TypeAlias = DomainContext[
    CodebaseDomainType,
    CodebaseAssetType,
    CodebaseChunkType,
    CodebaseConfigType,
]


class CodebaseAssetSource(
    DomainAssetSourceMixin[
        CodebaseDomainType,
        CodebaseAssetType,
        CodebaseChunkType,
        CodebaseConfigType,
    ]
):
    """Handles source management for codebase assets."""

    async def yield_all_assets(
        self,
        ctx: CodebaseDomainContext,
    ) -> AsyncGenerator[CodebaseAssetType, None]:
        """Get all assets for the codebase."""

        import zlib
        from git import Repo, InvalidGitRepositoryError

        domain = ctx.domain
        dir_path = ctx.config.directory_path
        try:
            repo = Repo(dir_path)
        except InvalidGitRepositoryError:
            repo = None

        for top, dirs, files in os.walk(dir_path):
            for file in files:
                if repo and repo.ignored(file):
                    continue

                async with aiofiles.open(os.path.join(top, file), "rb") as f:
                    file_content = await f.read()
                    file_hash = zlib.crc32(file_content)

                file_path = os.path.join(top, file)
                relative_path = os.path.relpath(file_path, dir_path)
                asset_data = GenericAssetData(
                    domain_id=domain.id,
                    id=relative_path,
                    name=file,
                    asset_manager_id=domain.id,
                    asset_hash=str(file_hash),
                    meta=CodeAssetMetaData(
                        file_path=file_path,
                    ),
                )
                yield asset_data


class CodebaseAssetIndexing(
    DomainAssetIndexingMixin[
        CodebaseDomainType,
        CodebaseAssetType,
        CodebaseChunkType,
        CodebaseConfigType,
    ]
):
    """Handles indexing of codebase assets."""

    async def index_assets(
        self,
        ctx: CodebaseDomainContext,
    ) -> None:
        """Index the given codebase assets."""
        pass


class CodebaseAssetParser(
    DomainAssetParserMixin[
        CodebaseDomainType,
        CodebaseAssetType,
        CodebaseChunkType,
        CodebaseConfigType,
    ]
):
    """Handles parsing of codebase assets."""

    def __init__(
        self,
        ctx: CodebaseDomainContext,
    ) -> None:
        super().__init__(ctx)
        self.code_parser_factory = CodeParserFactory(ctx.config)

    async def parse_assets(
        self,
        ctx: CodebaseDomainContext,
    ) -> List[CodebaseChunkType]:
        """Parse the given codebase assets."""

        for assets in ctx.assets:
            file_path = assets.meta.file_path
            parser = self.code_parser_factory.get_parser(file_path)
            await parser.parse_file(file_path)

        return []
