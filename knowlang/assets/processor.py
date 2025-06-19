from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

# Covariant type variables: allow being more specific
AssetDataT = TypeVar('AssetDataT', covariant=True)
AssetChunkDataT = TypeVar('AssetChunkDataT', covariant=True)

# Contravariant type variables: allow being more general
DomainConfigDataT = TypeVar('DomainConfigDataT', default=None, contravariant=True)
AssetMetadataT = TypeVar('AssetMetadataT', default=None, contravariant=True)
AssetChunkMetadataT = TypeVar('AssetChunkMetadataT', default=None, contravariant=True)

@dataclass
class DomainContext(Generic[DomainConfigDataT, AssetMetadataT, AssetChunkMetadataT]):
    """Context for domain asset processing."""
    config: DomainConfigDataT
    asset_metadata: AssetMetadataT
    asset_chunk_metadata: AssetChunkMetadataT


class Domain(ABC):
    def __init__(self, ctx: DomainContext):
        self.ctx = ctx

class DomainAssetSourceBase(ABC, Generic[AssetDataT], Domain):
    """Base class for domain asset source managers."""

    @abstractmethod
    async def get_asset_source(self, asset_id: str) -> AssetDataT:
        """Get the asset source by its ID."""
        pass

class DomainAssetIndexingBase(ABC, Generic[AssetDataT], Domain):
    """Base class for domain asset indexing managers."""

    @abstractmethod
    async def index_asset(self, assets: List[AssetDataT]) -> None:
        """Index the given assets."""
        pass

    @abstractmethod
    async def get_dirty_assets(self, assets: List[AssetDataT]) -> List[AssetDataT]:
        """Check if the assets are dirty (i.e., need re-indexing)."""
        pass

class DomainAssetParserBase(ABC, Generic[AssetDataT, AssetChunkDataT], Domain):
    """Base class for domain asset parsers."""

    @abstractmethod
    async def parse_assets(self, assets: List[AssetDataT]) -> List[AssetChunkDataT]:
        """Parse the given assets."""
        pass
