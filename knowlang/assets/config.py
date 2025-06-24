from pydantic import BaseModel
from typing_extensions import TypeVar, Generic
from knowlang.assets.models import DomainManagerData 
from knowlang.database.config import VectorStoreConfig


ProcessorConfigT = TypeVar("ProcessorConfigT", bound='ProcessorConfigBase', covariant=True)


class DomainMixinConfig(BaseModel,  Generic[ProcessorConfigT]):
    source_cls: str             # Class identifier (e.g., "CodebaseAssetSource")
    indexer_cls: str            # Class identifier
    parser_cls: str             # Class identifier
    mixin_config: ProcessorConfigT  # Mixin configuration


class BaseDomainConfig(BaseModel, Generic[ProcessorConfigT]):
    domain_type: str
    enabled: bool = True
    domain_data: DomainManagerData
    mixins: DomainMixinConfig[ProcessorConfigT]


class DatabaseConfig(BaseModel):
    """Configuration for database connections"""
    provider: str = "sqlite"
    connection_url: str = "sqlite+aiosqlite:///database.db"

class ProcessorConfigBase(BaseModel):
    """Base class for processor configurations."""
    vector_store: VectorStoreConfig
