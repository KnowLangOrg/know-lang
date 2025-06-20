from typing import List, AsyncGenerator
import os
from knowlang.assets.processor import (
    DomainContextMixin,
    DomainAssetSourceMixin,
    DomainAssetIndexingMixin,
    DomainAssetParserMixin,
)
from knowlang.assets.codebase.models import (
    CodebaseManagerData,
    CodebaseMetaData,
    CodeAssetMetaData,
    CodeAssetChunkMetaData,
)
from knowlang.assets.models import (
    GenericAssetData,
    GenericAssetChunkData,
)


class CodebaseContext(
    DomainContextMixin[CodebaseMetaData, CodeAssetMetaData, CodeAssetChunkMetaData]
):
    pass
    

class CodebaseAssetSource(
    CodebaseContext,
    DomainAssetSourceMixin[CodebaseManagerData, GenericAssetData]
):
    """Handles source management for codebase assets."""

    async def yield_all_assets(self, domain: CodebaseManagerData) -> AsyncGenerator[GenericAssetData, None]:
        """Get all assets for the codebase."""

        for top, dirs, files in os.walk(domain.directory_path):
            for file in files:
                file_path = os.path.join(top, file)
                relative_path = os.path.relpath(file_path, domain.directory_path)
                asset_data = GenericAssetData(
                    id=relative_path,
                    name=file,
                    asset_manager_id=domain.id,
                    metadata={
                        "absolute_path": file_path,
                        "relative_path": relative_path,
                    }
                )
                yield asset_data


class CodebaseAssetIndexing(
    CodebaseContext,
    DomainAssetIndexingMixin[CodebaseManagerData, GenericAssetData]
):
    """Handles indexing of codebase assets."""

    async def index_asset(self, assets: List[GenericAssetData]) -> None:
        """Index the given codebase assets."""
        pass

    async def get_dirty_assets(self, assets: List[GenericAssetData]) -> List[GenericAssetData]:
        """Check if the codebase assets are dirty (i.e., need re-indexing)."""
        return []


class CodebaseAssetParser(
    CodebaseContext,
    DomainAssetParserMixin[CodebaseManagerData, GenericAssetData, GenericAssetChunkData]
):
    """Handles parsing of codebase assets."""

    async def parse_assets(self, assets: List[GenericAssetData]) -> List[GenericAssetChunkData]:
        """Parse the given codebase assets."""
        return []
    
    