
from knowlang.assets.registry import DomainRegistry, RegistryConfig
import asyncio


async def main():
    config = RegistryConfig()
    registry = DomainRegistry(config)
    await registry.discover_and_register()

if __name__ == "__main__":
    asyncio.run(main())
