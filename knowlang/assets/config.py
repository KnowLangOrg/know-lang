from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Generic

from knowlang.assets.models import DomainManagerData, MixinConfigT


class DomainMixinConfig(BaseModel,  Generic[MixinConfigT]):
    source_cls: str             # Class identifier (e.g., "CodebaseAssetSource")
    indexer_cls: str            # Class identifier
    parser_cls: str             # Class identifier
    mixin_config: MixinConfigT  # Mixin configuration


class BaseDomainConfig(BaseModel, Generic[MixinConfigT]):
    domain_type: str
    enabled: bool = True
    domain_data: DomainManagerData
    mixins: DomainMixinConfig[MixinConfigT]


class DatabaseConfig(BaseSettings):
    """Configuration for database connections"""
    provider: str = "sqlite"
    connection_url: str = "sqlite+aiosqlite:///database.db"