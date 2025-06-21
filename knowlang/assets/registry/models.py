from __future__ import annotations
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource
)

from knowlang.assets.models import DomainManagerData

class BaseDomainConfig(BaseModel):
    domain_type: str
    domain_id: str
    enabled: bool = True
    manager_data: DomainManagerData
    mixins: DomainMixinConfig
    
class DomainMixinConfig(BaseModel):
    source_cls: str             # Class identifier (e.g., "CodebaseAssetSource")
    indexer_cls: str            # Class identifier
    parser_cls: str             # Class identifier

class RegistryConfig(BaseSettings):
    """Configuration for the domain registry."""
    discovery_path: str = 'settings/'
    model_config = SettingsConfigDict(
        yaml_file='settings/registry.yaml',
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)