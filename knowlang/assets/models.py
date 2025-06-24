from pydantic import BaseModel, Field
from enum import Enum
from typing_extensions import Optional, Generic, TypeVar, TYPE_CHECKING


if TYPE_CHECKING:
    from knowlang.database.db import DomainManagerOrm, GenericAssetOrm, GenericAssetChunkOrm

# Covariant type variables: allow being more specific
MetaDataT = TypeVar('MetaDataT', default=BaseModel, bound=BaseModel, covariant=True)
DomainDataT = TypeVar("DomainDataT", default='DomainManagerData', covariant=True, bound="DomainManagerData")
AssetDataT = TypeVar("AssetDataT", default='GenericAssetData', covariant=True, bound="GenericAssetData")
AssetChunkDataT = TypeVar("AssetChunkDataT", default='GenericAssetChunkData', covariant=True, bound="GenericAssetChunkData")



class MetaDataMixin(BaseModel, Generic[MetaDataT]):
    """Mixin for metadata in domain asset models."""
    meta: MetaDataT = Field(
        default=None,
        description="Additional metadata about the asset",
    )

    class Config:
        from_attributes = True

class DomainManagerData(MetaDataMixin, Generic[MetaDataT]):
    """Base class for domain asset manager data."""
    id: str = Field(..., description="Unique identifier for the asset manager")
    name: str = Field(..., description="Name of the asset manager")
    assets: Optional[list['GenericAssetData[MetaDataT]']] = Field(
        default=None,
        description="List of assets managed by this asset manager",
    )

    def to_orm(self) -> 'DomainManagerOrm':
        from knowlang.database.db import DomainManagerOrm
        return DomainManagerOrm(
            id=self.id,
            name=self.name,
            meta=self.meta.model_dump_json(),
        )

class GenericAssetData(MetaDataMixin, Generic[MetaDataT]):
    """Base class for generic asset data."""
    id: str = Field(..., description="Unique identifier for the asset")
    name: str = Field(..., description="Name of the asset")
    domain_id: str = Field(..., description="ID of the domain that manages this asset")
    domain: Optional[DomainManagerData[MetaDataT]] = Field(
        default=None,
        description="Domain manager data for the asset",
    )
    asset_chunks: Optional[list['GenericAssetChunkData']] = Field(
        default=None,
        description="List of chunks that make up this asset",
    )
    asset_hash: Optional[str] = Field(
        default=None,
        description="Hash of the asset file for integrity checks",
    )

    def to_orm(self) -> 'GenericAssetOrm':
        from knowlang.database.db import GenericAssetOrm
        return GenericAssetOrm(
            id=self.id,
            name=self.name,
            domain_id=self.domain_id,
            asset_hash=self.asset_hash,
            meta=self.meta.model_dump_json(),
        )

class GenericAssetChunkData(MetaDataMixin, Generic[MetaDataT]):
    """Base class for generic asset chunk data."""
    chunk_id: str = Field(..., description="Unique identifier for the asset chunk")
    asset_id: str = Field(..., description="ID of the parent asset")
    asset: Optional[GenericAssetData[MetaDataT]] = Field(
        default=None,
        description="Parent asset data for this chunk",
    )

    def to_orm(self) -> 'GenericAssetChunkOrm':
        from knowlang.database.db import GenericAssetChunkOrm
        return GenericAssetChunkOrm(
            id=self.chunk_id,
            asset_id=self.asset_id,
            meta=self.meta.model_dump_json(),
        )

class KnownDomainTypes(str, Enum):
    """Known domain types for asset management."""
    CODEBASE = "codebase"
    DOCUMENT = "document"