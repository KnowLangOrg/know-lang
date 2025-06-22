from typing import Dict, Type, List
import glob
import os
import yaml
import aiofiles
from knowlang.assets.processor import DomainProcessor
from knowlang.assets.registry.config import RegistryConfig
from knowlang.assets.config import BaseDomainConfig, DatabaseConfig

class DomainRegistry():
    """Registry for domain processors and mixins."""
    _mixin_registry : Dict[str, type] = {}
    _processor_registry: List[DomainProcessor] = []

    def __init__(self):
        """Initialize the domain processor factory."""
        self._mixin_registry = {}

        from knowlang.assets.codebase.processor import (
            CodebaseAssetSource,
            CodebaseAssetIndexing,
            CodebaseAssetParser
        )
        self._register_mixin([
            CodebaseAssetSource,
            CodebaseAssetIndexing,
            CodebaseAssetParser
        ])

    def _register_mixin(self, cls_list: List[Type]) -> None:
        """Register domain processor mixin classes."""
        for cls in cls_list:
            self._mixin_registry[cls.__name__] = cls


    def create_processor(self, config: BaseDomainConfig) -> DomainProcessor:
        """Create a domain processor based on the provided configuration."""
        
        try:
            processor = DomainProcessor()
            processor.source_mixin = self._mixin_registry[config.mixins.source_cls]()
            processor.indexing_mixin = self._mixin_registry[config.mixins.indexer_cls]()
            processor.parser_mixin = self._mixin_registry[config.mixins.parser_cls]()

            return processor

        except Exception as e:
            raise ValueError(f"Failed to create processor for [{config.domain_type}]{config.domain_id}: {str(e)}")

    async def discover_and_register(self, config: RegistryConfig) -> None:
        """Discover and register all domain processors and mixins."""
        
        for file in glob.glob(os.path.join(config.discovery_path, '*.yaml')):
            async with aiofiles.open(file, mode='r') as f:
                content = await f.read()
                domain_config = BaseDomainConfig.model_validate(yaml.safe_load(content))

                self._processor_registry.append(self.create_processor(domain_config))
    
    async def process_all_domains(self) -> None:
        """Process all registered domains."""
        from knowlang.assets.db import KnowledgeSqlDatabase
        db = KnowledgeSqlDatabase(config=DatabaseConfig())

        for processor in self._processor_registry:
            async for asset in processor.source_mixin.yield_all_assets():
                db.index_assets([asset])
                # processor.indexing_mixin.index_assets([asset])