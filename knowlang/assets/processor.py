from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from knowlang.assets.models import (
    GenericAssetChunkData,
    GenericAssetData,
    DomainAssetManagerData,
)

# Covariant type variables: allow being more specific
DomainDataT = TypeVar('DomainDataT', covariant=True, bound=DomainAssetManagerData)
AssetDataT = TypeVar('AssetDataT', covariant=True, bound=GenericAssetData)
AssetChunkDataT = TypeVar('AssetChunkDataT', covariant=True, bound=GenericAssetChunkData)

# Contravariant type variables: allow being more general
DomainMetaDataT = TypeVar('DomainMetaDataT', default=None, contravariant=True)
AssetMetadataT = TypeVar('AssetMetadataT', default=None, contravariant=True)
AssetChunkMetadataT = TypeVar('AssetChunkMetadataT', default=None, contravariant=True)

@dataclass
class DomainContextContext(Generic[DomainMetaDataT, AssetMetadataT, AssetChunkMetadataT]):
    """Context for domain asset processing."""
    domain_metadata: DomainMetaDataT
    asset_metadata: AssetMetadataT
    asset_chunk_metadata: AssetChunkMetadataT

class DomainContextMixin(ABC):
    def __init__(self, ctx: DomainContextContext):
        self.ctx = ctx

class DomainAssetSourceMixin(ABC, Generic[DomainDataT, AssetDataT]):
    """Base class for domain asset source managers."""

    @abstractmethod
    async def get_all_assets(self) -> List[AssetDataT]:
        """Get all assets for the given asset ID."""
        pass

class DomainAssetIndexingMixin(ABC, Generic[DomainDataT, AssetDataT]):
    """Base class for domain asset indexing managers."""

    @abstractmethod
    async def index_asset(self, assets: List[AssetDataT]) -> None:
        """Index the given assets."""
        pass

    @abstractmethod
    async def get_dirty_assets(self, assets: List[AssetDataT]) -> List[AssetDataT]:
        """Check if the assets are dirty (i.e., need re-indexing)."""
        pass

class DomainAssetParserMixin(ABC, Generic[DomainDataT, AssetDataT, AssetChunkDataT]):
    """Base class for domain asset parsers."""

    @abstractmethod
    async def parse_assets(self, assets: List[AssetDataT]) -> List[AssetChunkDataT]:
        """Parse the given assets."""
        pass
