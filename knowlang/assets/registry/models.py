from __future__ import annotations
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict
from knowlang.assets.models import DomainManagerData
from knowlang.assets.config import YamlConfigMixin

class BaseDomainConfig(YamlConfigMixin):
    domain_type: str
    domain_id: str
    enabled: bool = True
    manager_data: DomainManagerData
    mixins: DomainMixinConfig
    
class DomainMixinConfig(BaseModel):
    source_cls: str             # Class identifier (e.g., "CodebaseAssetSource")
    indexer_cls: str            # Class identifier
    parser_cls: str             # Class identifier

class RegistryConfig(YamlConfigMixin):
    """Configuration for the domain registry."""
    discovery_path: str = 'settings/'
    model_config = SettingsConfigDict(
        yaml_file='settings/registry.yaml',
    )