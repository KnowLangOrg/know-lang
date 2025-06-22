from typing import Dict, Type, List
import glob
import os
import yaml
import aiofiles
import functools
from knowlang.assets.processor import DomainProcessor
from knowlang.assets.registry.config import RegistryConfig
from knowlang.assets.config import BaseDomainConfig, DatabaseConfig

_mixin_registry : Dict[str, type] = {}
_domain_processor_registry: Dict[str, DomainProcessor] = {}
_domain_config_registry: Dict[str, BaseDomainConfig] = {}

def register_mixin() -> None:
    """Register domain processor mixin class."""

    from  knowlang.assets.codebase.processor import (
        CodebaseAssetSource,
        CodebaseAssetIndexing,
        CodebaseAssetParser
    )
    _mixin_registry[CodebaseAssetSource.__name__] =  CodebaseAssetSource
    _mixin_registry[CodebaseAssetIndexing.__name__] = CodebaseAssetIndexing
    _mixin_registry[CodebaseAssetParser.__name__] = CodebaseAssetParser


def create_processor(config: BaseDomainConfig) -> DomainProcessor:
    """Create a domain processor based on the provided configuration."""
    
    try:
        processor = DomainProcessor()
        processor.source_mixin = _mixin_registry[config.mixins.source_cls]()
        processor.indexing_mixin = _mixin_registry[config.mixins.indexer_cls]()
        processor.parser_mixin = _mixin_registry[config.mixins.parser_cls]()

        return processor

    except Exception as e:
        raise ValueError(f"Failed to create processor for [{config.domain_type}]{config.domain_id}: {str(e)}")

async def discover_and_register(config: RegistryConfig) -> None:
    """Discover and register all domain processors and mixins."""
    
    for file in glob.glob(os.path.join(config.discovery_path, '*.yaml')):
        async with aiofiles.open(file, mode='r') as f:
            content = await f.read()
            domain_config = BaseDomainConfig.model_validate(yaml.safe_load(content))

            _domain_processor_registry[domain_config.domain_id] = create_processor(domain_config)
            _domain_config_registry[domain_config.domain_id] = domain_config

async def process_all_domains() -> None:
    """Process all registered domains."""
    from knowlang.assets.db import KnowledgeSqlDatabase
    db = KnowledgeSqlDatabase(config=DatabaseConfig())
    await db.create_schema()

    for domain_name, processor in _domain_processor_registry.items():
        domain_config = _domain_config_registry[domain_name]
        async for asset in processor.source_mixin.yield_all_assets(domain_config.domain_data):
            await db.index_assets([asset])