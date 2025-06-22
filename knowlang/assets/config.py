from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Generic

from knowlang.assets.models import DomainManagerData, MetaDataT


class DomainMixinConfig(BaseModel):
    source_cls: str             # Class identifier (e.g., "CodebaseAssetSource")
    indexer_cls: str            # Class identifier
    parser_cls: str             # Class identifier

class BaseDomainConfig(BaseModel, Generic[MetaDataT]):
    domain_type: str
    domain_id: str
    enabled: bool = True
    domain_data: DomainManagerData[MetaDataT]
    mixins: DomainMixinConfig
    

class DatabaseConfig(BaseSettings):
    """Configuration for database connections"""
    provider: str = "sqlite"
    connection_url: str = "sqlite+aiosqlite:///database.db"