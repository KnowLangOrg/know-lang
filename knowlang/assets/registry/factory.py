from typing import Dict, Type
import glob
import os
import yaml
import aiofiles
from knowlang.assets.registry.models import BaseDomainConfig, RegistryConfig

class DomainProcessorFactory():
    _mixin_registry : Dict[str, type] = {}

    @staticmethod
    def register_mixin():
        """Register a domain processor mixin."""
        def func(cls: type) -> type:
            DomainProcessorFactory._mixin_registry[cls.__name__] = cls
            return cls
        
        return func

    @staticmethod
    def get_mixin(mixin_name: str) -> Type | None:
        """Get a registered domain processor mixin by name."""

        try:
            return DomainProcessorFactory._mixin_registry[mixin_name]
        except KeyError:
            raise ValueError(f"Mixin '{mixin_name}' is not registered.")


class DomainRegistry():
    """Registry for domain processors and mixins."""
    async def discovery_and_register(self, config: RegistryConfig) -> None:
        """Discover and register all domain processors and mixins."""
        
        for file in glob.glob(os.path.join(config.discovery_path, '*.yaml')):
            async with aiofiles.open(file, mode='r') as f:
                content = await f.read()
                domain_config = BaseDomainConfig.model_validate(yaml.safe_load(content))

                print(domain_config.model_dump_json(indent=2))