from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Generic, List, AsyncGenerator, TypeAlias
from knowlang.assets.models import (
    DomainDataT,
    AssetDataT,
    AssetChunkDataT,
)
from knowlang.assets.config import ProcessorConfigT

DomainGenericT: TypeAlias = Generic[
    DomainDataT, AssetDataT, AssetChunkDataT, ProcessorConfigT
]


@dataclass
class DomainContext(DomainGenericT):
    domain: DomainDataT
    assets: List[AssetDataT]
    asset_chunks: List[AssetChunkDataT]
    config: ProcessorConfigT


DomainCtxT: TypeAlias = DomainContext[DomainDataT, AssetDataT, AssetChunkDataT]


class DomainContextInit:
    def __init__(self, ctx: DomainCtxT) -> None:
        self.ctx = ctx


class DomainAssetSourceMixin(ABC, DomainContextInit):
    """Base class for domain asset source managers."""

    @abstractmethod
    async def yield_all_assets(
        self,
        ctx: DomainCtxT = None,
    ) -> AsyncGenerator[AssetDataT, None]:
        """Get all assets for the given asset ID."""
        pass


class DomainAssetParserMixin(ABC, DomainContextInit):
    """Base class for domain asset parsers."""

    @abstractmethod
    async def parse_assets(
        self, assets: List[AssetDataT], ctx: DomainCtxT = None
    ) -> List[AssetChunkDataT]:
        """Parse the given assets."""
        pass


class DomainAssetIndexingMixin(ABC, DomainContextInit):
    """Base class for domain asset indexing managers."""

    @abstractmethod
    async def index_chunks(
        self, chunks: List[AssetChunkDataT], ctx: DomainCtxT = None
    ) -> None:
        """Index the given assets."""
        pass

    @abstractmethod
    async def delete_chunks(
        self, chunks: List[AssetChunkDataT], ctx: DomainCtxT = None
    ) -> None:
        """Delete the given asset chunks."""
        pass


class DomainProcessor:
    source_mixin: DomainAssetSourceMixin
    indexing_mixin: DomainAssetIndexingMixin
    parser_mixin: DomainAssetParserMixin
