from pydantic import BaseModel, Field
from enum import Enum
from typing_extensions import Optional, Generic, TypeVar


# Covariant type variables: allow being more specific
MetaDataT = TypeVar('MetaDataT', bound=BaseModel, covariant=True)
DomainDataT = TypeVar("DomainDataT", covariant=True, bound="DomainManagerData")
AssetDataT = TypeVar("AssetDataT", covariant=True, bound="GenericAssetData")
AssetChunkDataT = TypeVar("AssetChunkDataT", covariant=True, bound="GenericAssetChunkData")
MixinConfigT = TypeVar("MixinConfigT", bound=BaseModel, covariant=True)



class MetaDataMixin(BaseModel, Generic[MetaDataT]):
    """Mixin for metadata in domain asset models."""
    metadata: Optional[MetaDataT] = Field(
        default=None,
        description="Additional metadata about the asset",
        alias="metadata_"
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

class GenericAssetChunkData(MetaDataMixin, Generic[MetaDataT]):
    """Base class for generic asset chunk data."""
    chunk_id: str = Field(..., description="Unique identifier for the asset chunk")
    asset_id: str = Field(..., description="ID of the parent asset")
    asset: Optional[GenericAssetData[MetaDataT]] = Field(
        default=None,
        description="Parent asset data for this chunk",
    )

class KnownDomainTypes(str, Enum):
    """Known domain types for asset management."""
    CODEBASE = "codebase"
    DOCUMENT = "document"