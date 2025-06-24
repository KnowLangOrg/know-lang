from typing import List, AsyncGenerator
import os
from knowlang.assets.processor import (
    DomainAssetSourceMixin,
    DomainAssetIndexingMixin,
    DomainAssetParserMixin,
    DomainContext,
)
from knowlang.assets.codebase.models import (
    CodeAssetMetaData,
    CodebaseManagerData,
    CodeProcessorConfig,
)
from knowlang.assets.models import (
    GenericAssetData,
    GenericAssetChunkData,
)


class CodebaseAssetSource(
    DomainAssetSourceMixin[
        CodebaseManagerData,
        GenericAssetData,
        GenericAssetChunkData,
        CodeProcessorConfig,
    ]
):
    """Handles source management for codebase assets."""

    async def yield_all_assets(
        self,
        ctx: DomainContext[
            CodebaseManagerData,
            GenericAssetData,
            GenericAssetChunkData,
            CodeProcessorConfig,
        ],
    ) -> AsyncGenerator[GenericAssetData, None]:
        """Get all assets for the codebase."""

        import zlib
        import aiofiles

        domain = ctx.domain
        dir_path = ctx.config.directory_path

        for top, dirs, files in os.walk(dir_path):
            for file in files:
                async with aiofiles.open(os.path.join(top, file), 'rb') as f:
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
        CodebaseManagerData,
        GenericAssetData,
        GenericAssetChunkData,
        CodeProcessorConfig,
    ]
):
    """Handles indexing of codebase assets."""

    async def index_assets(
        self,
        ctx: DomainContext[
            CodebaseManagerData,
            GenericAssetData,
            GenericAssetChunkData,
            CodeProcessorConfig,
        ],
    ) -> None:
        """Index the given codebase assets."""
        pass


class CodebaseAssetParser(
    DomainAssetParserMixin[
        CodebaseManagerData,
        GenericAssetData,
        GenericAssetChunkData,
        CodeProcessorConfig,
    ]
):
    """Handles parsing of codebase assets."""

    async def parse_assets(
        self,
        ctx: DomainContext[
            CodebaseManagerData,
            GenericAssetData,
            GenericAssetChunkData,
            CodeProcessorConfig,
        ],
    ) -> List[GenericAssetChunkData]:
        """Parse the given codebase assets."""
        return []
