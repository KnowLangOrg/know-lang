from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Generic, List, AsyncGenerator
from knowlang.assets.models import (
    DomainDataT,
    AssetDataT,
    AssetChunkDataT,
)
from knowlang.assets.config import ProcessorConfigT

@dataclass
class DomainContext(Generic[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]):
    domain: DomainDataT
    assets: List[AssetDataT]
    asset_chunks: List[AssetChunkDataT]
    config: ProcessorConfigT

class DomainContextInit(DomainContext[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]):
    def __init__(
        self,
        ctx: DomainContext[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]
    ) -> None:
        self.ctx = ctx

class DomainAssetSourceMixin(
    ABC, DomainContextInit[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]
):
    """Base class for domain asset source managers."""

    @abstractmethod
    async def yield_all_assets(
        self, ctx: DomainContext[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]
    ) -> AsyncGenerator[AssetDataT, None]:
        """Get all assets for the given asset ID."""
        pass


class DomainAssetIndexingMixin(
    ABC, DomainContextInit[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]
):
    """Base class for domain asset indexing managers."""

    @abstractmethod
    async def index_assets(
        self, ctx: DomainContext[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]
    ) -> None:
        """Index the given assets."""
        pass


class DomainAssetParserMixin(
    ABC, DomainContextInit[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]
):
    """Base class for domain asset parsers."""

    @abstractmethod
    async def parse_assets(
        self, ctx: DomainContext[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]
    ) -> List[AssetChunkDataT]:
        """Parse the given assets."""
        pass


class DomainProcessor(Generic[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]):
    source_mixin: DomainAssetSourceMixin[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]
    indexing_mixin: DomainAssetIndexingMixin[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]
    parser_mixin: DomainAssetParserMixin[DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT]