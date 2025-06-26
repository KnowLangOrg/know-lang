from enum import Enum
from typing import Dict, Type, Any 
import glob
import os
import yaml
import aiofiles
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource
)
from knowlang.assets.models import KnownDomainTypes, DomainManagerData
from knowlang.assets.processor import DomainProcessor, DomainContext
from knowlang.assets.config import BaseDomainConfig, DatabaseConfig


class DataModelTarget(str, Enum):
    DOMAIN = "domain"
    ASSET = "asset"
    CHUNK = "chunk"
    PROCESSOR = "processor"

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

class TypeRegistry:
    """Registry for mapping domain types to their metadata classes."""

    def __init__(self):
        self._data_model_types: Dict[str, Dict[str, Type[BaseModel]]] = {}

    def register_data_models(
        self,
        domain_type: str,
        domain_meta: Type[BaseModel],
        asset_meta: Type[BaseModel],
        chunk_meta: Type[BaseModel],
        processor_cfg: Type[BaseModel]
    ) -> None:
        """Register all data model classes for a domain."""
        self._data_model_types[domain_type] = {
            DataModelTarget.DOMAIN: domain_meta,
            DataModelTarget.ASSET: asset_meta,
            DataModelTarget.CHUNK: chunk_meta,
            DataModelTarget.PROCESSOR: processor_cfg,
        }

    def get_data_models(self, domain_type: str, target: DataModelTarget) -> Type[BaseModel]:
        """Get all data model classes for domain type."""
        try:
            return self._data_model_types[domain_type][target]
        except KeyError:
            raise ValueError(f"Invalid target '{target}' for domain type: {domain_type}")


class MixinRegistry:
    """Registry for domain processor mixins."""

    def __init__(self):
        self._mixins: Dict[str, Type] = {}

    def register_mixin(self, name: str, mixin_class: Type) -> None:
        """Register a mixin class."""
        self._mixins[name] = mixin_class

    def get_mixin(self, name: str) -> Type:
        """Get mixin class by name."""
        if name not in self._mixins:
            raise ValueError(f"No mixin registered with name: {name}")
        return self._mixins[name]

    def create_mixin_instance(self, name: str) -> Any:
        """Create an instance of the mixin."""
        mixin_class = self.get_mixin(name)
        return mixin_class()


# Main registry class
class DomainRegistry:
    """Centralized registry for all domain-related components."""

    def __init__(self, config: RegistryConfig):
        self.type_registry = TypeRegistry()
        self.mixin_registry = MixinRegistry()
        self.registry_config = config

        self._processors: Dict[str, DomainProcessor] = {}
        self._configs: Dict[str, BaseDomainConfig] = {}

        # Initialize with built-in types
        self._register_builtin_types()

    def _register_builtin_types(self) -> None:
        """Register built-in domain types."""
        # Register codebase types
        from knowlang.assets.codebase.models import (
            CodebaseMetaData,
            CodeAssetMetaData,
            CodeAssetChunkMetaData,
            CodeProcessorConfig,
        )

        self.type_registry.register_data_models(
            KnownDomainTypes.CODEBASE, 
            CodebaseMetaData,
            CodeAssetMetaData,
            CodeAssetChunkMetaData,
            CodeProcessorConfig,
        )

        # Register mixins
        from knowlang.assets.codebase.processor import (
            CodebaseAssetSource,
            CodebaseAssetIndexing,
            CodebaseAssetParser,
        )

        self.mixin_registry.register_mixin(CodebaseAssetSource.__name__, CodebaseAssetSource)
        self.mixin_registry.register_mixin(CodebaseAssetIndexing.__name__, CodebaseAssetIndexing)
        self.mixin_registry.register_mixin(CodebaseAssetParser.__name__, CodebaseAssetParser)


    def register_processor_mixins(
        self,
        mixinTypes: list[Type],
    ) -> None:
        """Register processor mixin classes."""
        for mixin_class in mixinTypes:
            self.mixin_registry.register_mixin(mixin_class.__name__, mixin_class)

    def _resolve_domain_context(self, domain_config: BaseDomainConfig) -> DomainContext:
        """Resolve the domain context based on the configuration."""
        domain_meta_t = self.type_registry.get_data_models(
            domain_config.domain_type, DataModelTarget.DOMAIN
        )
        processor_cfg_t = self.type_registry.get_data_models(
            domain_config.domain_type, DataModelTarget.PROCESSOR
        )

        return DomainContext(
            domain=DomainManagerData[domain_meta_t].model_validate(domain_config.domain_data),
            assets=[],
            asset_chunks=[],
            config=processor_cfg_t.model_validate(domain_config.processor_config),
        )

    def create_processor(self, config: BaseDomainConfig) -> DomainProcessor:
        """Create a domain processor based on configuration."""
        from knowlang.assets.processor import DomainContextInit
        try:
            ctx = self._resolve_domain_context(config)

            source_cls = self.mixin_registry.get_mixin(config.mixins.source_cls)
            indexer_cls = self.mixin_registry.get_mixin(config.mixins.indexer_cls)
            parser_cls = self.mixin_registry.get_mixin(config.mixins.parser_cls)
            assert issubclass(source_cls, DomainContextInit)
            assert issubclass(indexer_cls, DomainContextInit)
            assert issubclass(parser_cls, DomainContextInit)

            processor = DomainProcessor()
            processor.source_mixin = source_cls(ctx)
            processor.indexing_mixin = indexer_cls(ctx)
            processor.parser_mixin = parser_cls(ctx)

            return processor
        except Exception as e:
            raise ValueError(
                f"Failed to create processor for [{config.domain_type}]: {str(e)}"
            )

    async def discover_and_register(self, discovery_path: str = None) -> None:
        """Discover and register all domain processors from configuration files."""
        if discovery_path is None:
            discovery_path = self.registry_config.discovery_path

        for file in glob.glob(os.path.join(discovery_path, "*.yaml")):
            await self._load_domain_file(file)
    
    def _resolve_cfg_type(self, config_dict: dict) -> BaseDomainConfig:
        domain_config = BaseDomainConfig.model_validate(config_dict)
        domain_data_t = self.type_registry.get_data_models(domain_config.domain_type, DataModelTarget.DOMAIN)
        processor_config_t = self.type_registry.get_data_models(domain_config.domain_type, DataModelTarget.PROCESSOR)
        domain_config = BaseDomainConfig[domain_data_t, processor_config_t].model_validate(config_dict)

        return domain_config

    async def _load_domain_file(self, file_path: str) -> None:
        """Load and register configuration from a single file."""
        async with aiofiles.open(file_path, mode="r") as f:
            content = await f.read()
            config_dict = yaml.safe_load(content)

            domain_config = self._resolve_cfg_type(config_dict)

            # Create and register processor
            processor = self.create_processor(domain_config)

            self._processors[domain_config.domain_data.id] = processor
            self._configs[domain_config.domain_data.id] = domain_config

    def get_processor(self, domain_id: str) -> DomainProcessor:
        """Get processor by domain ID."""
        if domain_id not in self._processors:
            raise ValueError(f"No processor registered for domain: {domain_id}")
        return self._processors[domain_id]

    def get_config(self, domain_id: str) -> BaseDomainConfig:
        """Get configuration by domain ID."""
        if domain_id not in self._configs:
            raise ValueError(f"No config registered for domain: {domain_id}")
        return self._configs[domain_id]

    def list_domains(self) -> list[str]:
        """List all registered domain IDs."""
        return list(self._processors.keys())

    async def process_all_domains(self) -> None:
        """Process all registered domains."""
        from knowlang.database.db import KnowledgeSqlDatabase
        from knowlang.assets.processor import DomainContext
        from knowlang.vector_stores.factory import VectorStoreFactory

        db = KnowledgeSqlDatabase(config=DatabaseConfig())
        await db.create_schema()

        for domain_id, processor in self._processors.items():
            domain_config = self._configs[domain_id]

            async for asset in processor.source_mixin.yield_all_assets():
                await db.index_assets([asset.to_orm()])
                chunks = await processor.parser_mixin.parse_assets([asset])
                await processor.indexing_mixin.index_chunks(chunks)