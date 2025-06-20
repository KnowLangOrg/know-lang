from typing import List
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

    async def get_all_assets(self) -> List[GenericAssetData]:
        """Get all assets for the codebase."""
        raise NotImplementedError("This method should be implemented in subclasses.")


class CodebaseAssetIndexing(
    CodebaseContext,
    DomainAssetIndexingMixin[CodebaseAssetManagerData, GenericAssetData]
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
    DomainAssetParserMixin[CodebaseAssetManagerData, GenericAssetData, GenericAssetChunkData]
):
    """Handles parsing of codebase assets."""

    async def parse_assets(self, assets: List[GenericAssetData]) -> List[GenericAssetChunkData]:
        """Parse the given codebase assets."""
        return []
    
    