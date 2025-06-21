from abc import ABC, abstractmethod
from typing_extensions import Generic, List, TypeVar, AsyncGenerator
from knowlang.assets.models import (
    GenericAssetChunkData,
    GenericAssetData,
    DomainManagerData,
)

# Covariant type variables: allow being more specific
DomainDataT = TypeVar("DomainDataT", covariant=True, bound=DomainManagerData)
AssetDataT = TypeVar("AssetDataT", covariant=True, bound=GenericAssetData)
AssetChunkDataT = TypeVar(
    "AssetChunkDataT", covariant=True, bound=GenericAssetChunkData
)

class DomainAssetSourceMixin(ABC, Generic[DomainDataT, AssetDataT]):
    """Base class for domain asset source managers."""

    @abstractmethod
    async def yield_all_assets(self, domain: DomainDataT) -> AsyncGenerator[AssetDataT, None]:
        """Get all assets for the given asset ID."""
        pass


class DomainAssetIndexingMixin(ABC, Generic[DomainDataT, AssetDataT]):
    """Base class for domain asset indexing managers."""

    @abstractmethod
    async def index_assets(self, domain: DomainDataT, assets: List[AssetDataT]) -> None:
        """Index the given assets."""
        pass


class DomainAssetParserMixin(ABC, Generic[DomainDataT, AssetDataT, AssetChunkDataT]):
    """Base class for domain asset parsers."""

    @abstractmethod
    async def parse_assets(self, assets: List[AssetDataT]) -> List[AssetChunkDataT]:
        """Parse the given assets."""
        pass


class DomainProcessor():
    source_mixin: DomainAssetSourceMixin
    indexing_mixin: DomainAssetIndexingMixin
    parser_mixin: DomainAssetParserMixin