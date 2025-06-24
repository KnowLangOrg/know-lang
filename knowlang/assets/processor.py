from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Generic, List, AsyncGenerator
from knowlang.assets.models import (
    DomainDataT,
    AssetDataT,
    AssetChunkDataT,
    MixinConfigT,
)

@dataclass
class DomainContext(Generic[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]):
    domain: DomainDataT
    assets: List[AssetDataT]
    asset_chunks: List[AssetChunkDataT]
    config: MixinConfigT


class DomainAssetSourceMixin(
    ABC, Generic[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]
):
    """Base class for domain asset source managers."""

    @abstractmethod
    async def yield_all_assets(
        self, ctx: DomainContext[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]
    ) -> AsyncGenerator[AssetDataT, None]:
        """Get all assets for the given asset ID."""
        pass


class DomainAssetIndexingMixin(
    ABC, Generic[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]
):
    """Base class for domain asset indexing managers."""

    @abstractmethod
    async def index_assets(
        self, ctx: DomainContext[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]
    ) -> None:
        """Index the given assets."""
        pass


class DomainAssetParserMixin(
    ABC, Generic[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]
):
    """Base class for domain asset parsers."""

    @abstractmethod
    async def parse_assets(
        self, ctx: DomainContext[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]
    ) -> List[AssetChunkDataT]:
        """Parse the given assets."""
        pass


class DomainProcessor(Generic[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]):
    source_mixin: DomainAssetSourceMixin[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]
    indexing_mixin: DomainAssetIndexingMixin[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]
    parser_mixin: DomainAssetParserMixin[DomainDataT, AssetDataT, AssetChunkDataT, MixinConfigT]