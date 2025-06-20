from knowlang.assets.processor import (
    DomainContextMixin,
    DomainAssetSourceMixin,
    DomainAssetIndexingMixin,
    DomainAssetParserMixin,
)
from knowlang.assets.codebase.models import (
    CodebaseAssetManagerData,
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
    DomainAssetSourceMixin[CodebaseAssetManagerData, GenericAssetData]
):
    """Handles source management for codebase assets."""

    async def get_asset_source(self, asset_id: str) -> GenericAssetData:
        """Get the asset source by its ID."""
        raise NotImplementedError("This method should be implemented in subclasses.")


class CodebaseAssetIndexing(
    CodebaseContext,
    DomainAssetIndexingMixin[CodebaseAssetManagerData, GenericAssetData]
):
    """Handles indexing of codebase assets."""

    async def index_asset(self, assets: list[GenericAssetData]) -> None:
        """Index the given codebase assets."""
        pass

    async def get_dirty_assets(self, assets: list[GenericAssetData]) -> list[GenericAssetData]:
        """Check if the codebase assets are dirty (i.e., need re-indexing)."""
        return []


class CodebaseAssetParser(
    CodebaseContext,
    DomainAssetParserMixin[CodebaseAssetManagerData, GenericAssetData, GenericAssetChunkData]
):
    """Handles parsing of codebase assets."""

    async def parse_assets(self, assets: list[GenericAssetData]) -> list[GenericAssetChunkData]:
        """Parse the given codebase assets."""
        return []
    
    