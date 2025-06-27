from pydantic import BaseModel
from typing_extensions import TypeVar, Generic
from knowlang.assets.models import DomainManagerData, MetaDataT 
from knowlang.configs.defaults import DEFAULT_SQLITE_DB_CONNECTION_URL_ASYNC
from knowlang.database.config import VectorStoreConfig

ProcessorConfigT = TypeVar('ProcessorConfigT', bound='ProcessorConfigBase', covariant=True, default='ProcessorConfigBase')

class ProcessorConfigBase(BaseModel):
    """Base class for processor configurations."""
    vector_store: VectorStoreConfig

class DomainMixinConfig(BaseModel):
    source_cls: str             # Class identifier (e.g., "CodebaseAssetSource")
    indexer_cls: str            # Class identifier
    parser_cls: str             # Class identifier

class BaseDomainConfig(BaseModel, Generic[MetaDataT, ProcessorConfigT]):
    domain_type: str
    enabled: bool = True
    domain_data: DomainManagerData[MetaDataT]
    mixins: DomainMixinConfig
    processor_config: ProcessorConfigT

class DatabaseConfig(BaseModel):
    """Configuration for database connections"""
    provider: str = "sqlite"
    connection_url: str = DEFAULT_SQLITE_DB_CONNECTION_URL_ASYNC

