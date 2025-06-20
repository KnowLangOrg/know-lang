from pydantic import BaseModel, Field
from typing import Optional, Dict


class DomainAssetMangerData(BaseModel):
    """Base class for domain asset manager data."""
    id: str = Field(..., description="Unique identifier for the asset manager")
    name: str = Field(..., description="Name of the asset manager")
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        default_factory=dict,
        description="Additional metadata about the asset manager"
    )

class GenericAssetData(BaseModel):
    """Base class for generic asset data."""
    id: str = Field(..., description="Unique identifier for the asset")
    name: str = Field(..., description="Name of the asset")
    asset_manager_id: str = Field(..., description="ID of the asset manager that manages this asset")
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        default_factory=dict,
        description="Additional metadata about the asset"
    )

class GenericAssetChunkData(BaseModel):
    """Base class for generic asset chunk data."""
    chunk_id: str = Field(..., description="Unique identifier for the asset chunk")
    asset_id: str = Field(..., description="ID of the parent asset")
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        default_factory=dict,
        description="Additional metadata about the asset chunk"
    )