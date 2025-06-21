from typing import List, AsyncGenerator
import os
from knowlang.assets.processor import (
    DomainAssetSourceMixin,
    DomainAssetIndexingMixin,
    DomainAssetParserMixin,
)
from knowlang.assets.codebase.models import (
    CodebaseManagerData,
)
from knowlang.assets.models import (
    GenericAssetData,
    GenericAssetChunkData,
)


class CodebaseAssetSource(
    DomainAssetSourceMixin[CodebaseManagerData, GenericAssetData]
):
    """Handles source management for codebase assets."""

    async def yield_all_assets(
        self, domain: CodebaseManagerData
    ) -> AsyncGenerator[GenericAssetData, None]:
        """Get all assets for the codebase."""

        metadata = domain.metadata

        for top, dirs, files in os.walk(metadata.directory_path):
            for file in files:
                file_path = os.path.join(top, file)
                relative_path = os.path.relpath(file_path, metadata.directory_path)
                asset_data = GenericAssetData(
                    id=relative_path,
                    name=file,
                    asset_manager_id=domain.id,
                    metadata={
                        "absolute_path": file_path,
                        "relative_path": relative_path,
                    },
                )
                yield asset_data


class CodebaseAssetIndexing(
    DomainAssetIndexingMixin[CodebaseManagerData, GenericAssetData]
):
    """Handles indexing of codebase assets."""

    async def index_assets(
        self, domain: CodebaseManagerData, assets: List[GenericAssetData]
    ) -> None:
        """Index the given codebase assets."""
        pass


class CodebaseAssetParser(
    DomainAssetParserMixin[CodebaseManagerData, GenericAssetData, GenericAssetChunkData]
):
    """Handles parsing of codebase assets."""

    async def parse_assets(self, assets: List[GenericAssetData]) -> List[GenericAssetChunkData]:
        """Parse the given codebase assets."""
        return []
    
    