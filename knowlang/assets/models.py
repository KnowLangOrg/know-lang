from pydantic import BaseModel
from typing_extensions import Generic, TypeVar, List
from sqlmodel import SQLModel, Field, Relationship

MetaDataT = TypeVar('MetaDataT', bound=BaseModel, covariant=True)


class MetaData(BaseModel, Generic[MetaDataT]):
    meta_data: str = Field(
        default=None,
        description="Additional metadata"
    )

    @property
    def MetaData(self) -> MetaDataT:
        """Return the metadata for this asset manager."""
        return MetaDataT.model_validate_strings(self.meta_data)

class DomainManagerData(MetaData, SQLModel, table=True):
    """Base class for domain asset manager data."""
    id: str = Field(..., description="Unique identifier for the asset manager", primary_key=True)
    name: str = Field(..., description="Name of the asset manager", index=True)
    assets: List['GenericAssetData'] = Relationship(
        back_populates="domain",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

class GenericAssetData(MetaData, SQLModel, table=True):
    """Base class for generic asset data."""
    id: str = Field(..., description="Unique identifier for the asset", primary_key=True)
    name: str = Field(..., description="Name of the asset", index=True)
    domain_id: str = Field(..., description="ID of the domain that manages this asset" ,foreign_key="DomainManagerData.id")
    asset_hash: str = Field(..., description="Hash of the asset content for change detection")
    asset_chunks: List['GenericAssetChunkData'] = Relationship(
        back_populates="asset",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    domain: DomainManagerData = Relationship(
        back_populates="assets",
        sa_relationship_kwargs={"lazy": "joined"}
    )

class GenericAssetChunkData(MetaData, SQLModel, table=True):
    """Base class for generic asset chunk data."""
    id: str = Field(..., description="Unique identifier for the asset chunk", primary_key=True)
    asset_id: str = Field(..., description="ID of the parent asset", foreign_key="GenericAssetData.id")
    asset: GenericAssetData = Relationship(
        back_populates="asset_chunks",
        sa_relationship_kwargs={"lazy": "joined"}
    )