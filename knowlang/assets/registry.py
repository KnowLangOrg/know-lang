from enum import Enum
from typing import Dict, List, Optional, Set, Type, Any 
import glob
import os
import yaml
import aiofiles
from pydantic import BaseModel, ValidationInfo, field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource
)
from knowlang.assets.models import GenericAssetData, KnownDomainTypes, DomainManagerData, GenericAssetChunkData
from knowlang.assets.processor import DomainProcessor, DomainContext
from knowlang.assets.config import BaseDomainConfig
from knowlang.database.db import KnowledgeSqlDatabase
from knowlang.database.config import DatabaseConfig
from knowlang.configs.base import get_resource_path
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__file__)


class DataModelTarget(str, Enum):
    DOMAIN = "domain"
    ASSET = "asset"
    CHUNK = "chunk"
    PROCESSOR = "processor"

class RegistryConfig(BaseSettings):
    """Configuration for the domain registry."""
    discovery_path: str = 'settings/'
    model_config = SettingsConfigDict(
        yaml_file=get_resource_path('settings/registry.yaml'),
    )

    @field_validator("discovery_path", mode="after")
    @classmethod
    def resolve_discovery_path(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Resolve the discovery path to an absolute path."""
        return str(get_resource_path(v))

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


    async def process_all_domains(self, batch_size: int = 200) -> None:
        """Process all registered domains with efficient batching."""
        
        db = KnowledgeSqlDatabase(config=DatabaseConfig())
        await db.create_schema()

        for domain_id, processor in self._processors.items():
            # 1. Ensure domain is registered in DB
            LOG.info(f"Start processing domain: {domain_id}")
            async with db.get_session() as session:
                await db.upsert_domains(session, [processor.source_mixin.ctx.domain.to_orm()])

            # 2. Process domain assets with batching
            await self._process_domain_with_batching(db, processor, batch_size)
    
    async def _process_domain_with_batching(
        self,
        db: KnowledgeSqlDatabase,
        processor: DomainProcessor,
        batch_size: int
    ) -> None:
        """Process a single domain with batching."""
        
        asset_batch = []
        seen_asset_ids: Set[str] = set()
        
        async for asset in processor.source_mixin.yield_all_assets():
            assert isinstance(asset, GenericAssetData), "Asset must be of type GenericAssetData"
            asset_batch.append(asset)
            seen_asset_ids.add(asset.id)

            if len(asset_batch) >= batch_size:
                await self._process_asset_batch(db, processor, asset_batch)
                asset_batch.clear()
        
        # Process remaining assets
        if asset_batch:
            await self._process_asset_batch(db, processor, asset_batch)
        
        domain_id = processor.source_mixin.ctx.domain.id
        await self._cleanup_deleted_assets(db, processor, domain_id, seen_asset_ids)
    
    async def _cleanup_deleted_assets(
        self,
        db: KnowledgeSqlDatabase,
        processor: DomainProcessor,
        domain_id: str,
        seen_asset_ids: Set[str]
    ) -> None:
        """Remove assets that exist in DB but were not seen during scan."""
        
        # Find assets that are in DB but not seen during scan
        async with db.get_session() as session:
            stored_asset_ids = await db.get_all_asset_ids_for_domain(session, domain_id)
            deleted_asset_ids = set(stored_asset_ids) - set(seen_asset_ids)
        
            if not deleted_asset_ids:
                LOG.info(f"No deleted assets found for domain: {domain_id}")
                return
            
            LOG.info(f"Found {len(deleted_asset_ids)} deleted assets for domain: {domain_id}")
        
            if not deleted_asset_ids:
                LOG.info(f"No deleted assets found for domain: {domain_id}")
                return
            
            LOG.info(f"Found {len(deleted_asset_ids)} deleted assets for domain: {domain_id}")
            
            deleted_list = list(deleted_asset_ids)
                
            # Get chunks that will be deleted (for vector store cleanup)
            chunks_to_delete = await db.get_chunks_given_assets(session, deleted_list)
            
            # Clean up from vector store first
            chunk_data = [GenericAssetChunkData.model_validate(chunk) for chunk in chunks_to_delete]
            await processor.indexing_mixin.delete_chunks(chunk_data)
            
            # Delete from database (will cascade to chunks)
            await db.delete_assets_by_ids(session, deleted_list)
    
    async def _process_asset_batch(
        self,
        db: KnowledgeSqlDatabase,
        processor: DomainProcessor,
        assets: List[GenericAssetData],
    ) -> None:
        """Process a batch of assets efficiently."""

        # 1. Bulk dirty checking
        async with db.get_session() as session:
            existing_hashes = await db.get_asset_hash(session, [GenericAssetData(
                id=asset.id,
                name=asset.name,
                domain_id=asset.domain_id,
                asset_hash=asset.asset_hash,
                meta=asset.meta,
                domain=None  # Avoid implicit asyncIO by sqlalchemy
            ) for asset in assets])
            dirty_assets = [
                asset for asset in assets 
                if asset.id not in existing_hashes or existing_hashes[asset.id] != asset.asset_hash
            ]
            
            if not dirty_assets:
                # Nothing to process
                return  
            
            # 2. Bulk database update for dirty assets
            dirty_chunks = await db.get_chunks_given_assets(session, [asset.id for asset in dirty_assets])

            await processor.indexing_mixin.delete_chunks([
                GenericAssetChunkData(
                    id=chunk.id,
                    asset_id=chunk.asset_id,
                    meta=chunk.meta,
                    # excluding the asset to avoid implicit asyncIO by sqlalchemy
                    # https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#asyncio-orm-avoid-lazyloads
                    asset=None
                ) for chunk in dirty_chunks
            ])

            chunks = await processor.parser_mixin.parse_assets(dirty_assets)
            # Filter out assets that have no chunks
            dirty_assets = [
                a for a in dirty_assets if a.id in set(c.asset_id for c in chunks)
            ]
            await db.upsert_assets(session, [asset.to_orm() for asset in dirty_assets])

        # 3. Parse and index
        # Restart the session to avoid sqlite3 lock
        async with db.get_session() as session:
            await processor.indexing_mixin.index_chunks(chunks)
            await db.index_asset_chunks(session, [chunk.to_orm() for chunk in chunks])

            LOG.info(f"Processed {len(dirty_assets)} dirty assets in batch of size {len(assets)}")