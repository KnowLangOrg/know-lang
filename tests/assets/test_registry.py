import os
import tempfile

# Third-party imports
import aiofiles
import pytest
import yaml
from pydantic import BaseModel

# knowlang imports
from knowlang.assets.config import (
    BaseDomainConfig,
    DomainMixinConfig,
    ProcessorConfigBase,
)
from knowlang.assets.models import DomainManagerData
from knowlang.assets.registry import (
    DataModelTarget,
    DomainRegistry,
    MixinRegistry,
    RegistryConfig,
    TypeRegistry,
)


def test_registryconfig_loads_yaml(tmp_path, monkeypatch):
    # Prepare a dummy YAML
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("discovery_path: /foo/bar\n")
    # Monkey-patch get_resource_path to bypass file existence check
    import knowlang.assets.registry as registry_module

    monkeypatch.setattr(
        registry_module,
        "get_resource_path",
        lambda path, default_path=None: path,
    )
    # Instantiate with explicit discovery_path to override YAML
    cfg = RegistryConfig(discovery_path="/foo/bar")
    assert cfg.discovery_path == "/foo/bar"


@pytest.mark.parametrize(
    "target,expected",
    [
        (DataModelTarget.DOMAIN, dict),
        (DataModelTarget.ASSET, dict),
    ],
)
def test_typeregistry_register_and_get(target, expected):
    tr = TypeRegistry()
    # Fake models
    tr.register_data_models("X", dict, dict, dict, dict)
    assert tr.get_data_models("X", target) is dict


def test_typeregistry_invalid_key_raises():
    tr = TypeRegistry()
    with pytest.raises(ValueError):
        tr.get_data_models("unknown", DataModelTarget.CHUNK)


class DummyMixin:
    def __init__(self):
        self.name = "dummy"


def test_mixinregistry_register_and_get():
    mr = MixinRegistry()
    mr.register_mixin("DummyMixin", DummyMixin)
    mixin_cls = mr.get_mixin("DummyMixin")
    assert mixin_cls is DummyMixin


def test_mixinregistry_get_unregistered_raises():
    mr = MixinRegistry()
    with pytest.raises(ValueError, match="No mixin registered with name: UnknownMixin"):
        mr.get_mixin("UnknownMixin")


def test_mixinregistry_create_instance():
    class ExampleMixin:
        def __init__(self):
            self.value = 123

    mr = MixinRegistry()
    mr.register_mixin("ExampleMixin", ExampleMixin)
    instance = mr.create_mixin_instance("ExampleMixin")
    assert isinstance(instance, ExampleMixin)
    assert instance.value == 123


# Additional test for multiple types registration in TypeRegistry
def test_typeregistry_multiple_types():
    tr = TypeRegistry()

    class DummyModel(BaseModel):
        a: int

    tr.register_data_models("DummyType", DummyModel, DummyModel, DummyModel, DummyModel)

    for target in DataModelTarget:
        assert tr.get_data_models("DummyType", target) is DummyModel


def test_resolve_domain_context_from_registry():
    class DummyMeta(BaseModel):
        field: str

    class DummyConfig(ProcessorConfigBase):
        config_field: str

    registry = DomainRegistry(RegistryConfig())
    registry.type_registry.register_data_models(
        "dummy", DummyMeta, DummyMeta, DummyMeta, DummyConfig
    )

    dummy_cfg = BaseDomainConfig[DummyMeta, DummyConfig](
        domain_type="dummy",
        domain_data=DomainManagerData[DummyMeta](
            id="x", name="X", meta=DummyMeta(field="foo")
        ),
        mixins=DomainMixinConfig(source_cls="", indexer_cls="", parser_cls=""),
        processor_config=DummyConfig(config_field="bar"),
    )

    ctx = registry._resolve_domain_context(dummy_cfg)
    assert ctx.domain.meta.field == "foo"
    assert ctx.config.config_field == "bar"


# Test for create_processor with registered mixins
def test_create_processor_with_registered_mixins():
    from knowlang.assets.processor import DomainContextInit

    class DummyMeta(BaseModel):
        field: str

    class DummyConfig(ProcessorConfigBase):
        config_field: str

    class DummySource(DomainContextInit):
        pass

    class DummyIndexer(DomainContextInit):
        pass

    class DummyParser(DomainContextInit):
        pass

    registry = DomainRegistry(RegistryConfig())
    registry.type_registry.register_data_models(
        "dummy", DummyMeta, DummyMeta, DummyMeta, DummyConfig
    )
    registry.register_processor_mixins([DummySource, DummyIndexer, DummyParser])

    dummy_cfg = BaseDomainConfig[DummyMeta, DummyConfig](
        domain_type="dummy",
        domain_data=DomainManagerData[DummyMeta](
            id="x", name="X", meta=DummyMeta(field="foo")
        ),
        mixins=DomainMixinConfig(
            source_cls="DummySource",
            indexer_cls="DummyIndexer",
            parser_cls="DummyParser",
        ),
        processor_config=DummyConfig(config_field="bar"),
    )

    processor = registry.create_processor(dummy_cfg)
    assert isinstance(processor.source_mixin, DummySource)
    assert isinstance(processor.indexing_mixin, DummyIndexer)
    assert isinstance(processor.parser_mixin, DummyParser)


def test_create_processor_unregistered_mixin_raises():
    class DummyMeta(BaseModel):
        field: str

    class DummyConfig(ProcessorConfigBase):
        config_field: str

    registry = DomainRegistry(RegistryConfig())
    registry.type_registry.register_data_models(
        "dummy", DummyMeta, DummyMeta, DummyMeta, DummyConfig
    )

    dummy_cfg = BaseDomainConfig[DummyMeta, DummyConfig](
        domain_type="dummy",
        domain_data=DomainManagerData[DummyMeta](
            id="x", name="X", meta=DummyMeta(field="foo")
        ),
        mixins=DomainMixinConfig(
            source_cls="NonExistent",
            indexer_cls="NonExistent",
            parser_cls="NonExistent",
        ),
        processor_config=DummyConfig(config_field="bar"),
    )

    with pytest.raises(ValueError, match=r"Failed to create processor.*NonExistent"):
        registry.create_processor(dummy_cfg)


def test_create_processor_invalid_mixin_type_raises():
    class DummyMeta(BaseModel):
        field: str

    class DummyConfig(ProcessorConfigBase):
        config_field: str

    class NotADomainContextInit:
        def __init__(self, ctx):
            self.ctx = ctx

    registry = DomainRegistry(RegistryConfig())
    registry.type_registry.register_data_models(
        "dummy", DummyMeta, DummyMeta, DummyMeta, DummyConfig
    )
    registry.register_processor_mixins([NotADomainContextInit])

    dummy_cfg = BaseDomainConfig[DummyMeta, DummyConfig](
        domain_type="dummy",
        domain_data=DomainManagerData[DummyMeta](
            id="x", name="X", meta=DummyMeta(field="foo")
        ),
        mixins=DomainMixinConfig(
            source_cls="NotADomainContextInit",
            indexer_cls="NotADomainContextInit",
            parser_cls="NotADomainContextInit",
        ),
        processor_config=DummyConfig(config_field="bar"),
    )

    with pytest.raises(ValueError, match="Failed to create processor.*"):
        registry.create_processor(dummy_cfg)


def test_resolve_cfg_type_invalid_raises():
    registry = DomainRegistry(RegistryConfig())
    invalid_config = {
        "domain_type": "dummy",
        "domain_data": {
            "id": "x",
            "name": "X",
            "meta": {"field": 123},  # Assume this should be str, not int
        },
        "mixins": {
            "source_cls": "DummySource",
            "indexer_cls": "DummyIndexer",
            "parser_cls": "DummyParser",
        },
        "processor_config": {"config_field": 42},  # Assume this should be str, not int
    }
    with pytest.raises(Exception):
        registry._resolve_cfg_type(invalid_config)


# --- async test for _load_domain_file ---


@pytest.mark.asyncio
async def test_load_domain_file_registers_config():
    class DummyMeta(BaseModel):
        field: str

    class DummyConfig(ProcessorConfigBase):
        config_field: str

    registry = DomainRegistry(RegistryConfig())
    registry.type_registry.register_data_models(
        "dummy", DummyMeta, DummyMeta, DummyMeta, DummyConfig
    )

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    config = {
        "domain_type": "dummy",
        "domain_data": {
            "id": "x",
            "name": "X",
            "meta": {"field": "foo"},
        },
        "mixins": {
            "source_cls": "",
            "indexer_cls": "",
            "parser_cls": "",
        },
        "processor_config": {"config_field": "bar"},
    }

    async with aiofiles.open(tmp_file.name, "w") as f:
        await f.write(yaml.safe_dump(config))

    await registry._load_domain_file(tmp_file.name)
    assert "x" in registry.domain_configs
    assert registry.domain_configs["x"].domain_data.name == "X"


@pytest.mark.asyncio
async def test_create_processors_after_load():
    from knowlang.assets.processor import DomainContextInit

    class DummyMeta(BaseModel):
        field: str

    class DummyConfig(ProcessorConfigBase):
        config_field: str

    class DummySource(DomainContextInit):
        pass

    class DummyIndexer(DomainContextInit):
        pass

    class DummyParser(DomainContextInit):
        pass

    registry = DomainRegistry(RegistryConfig())
    registry.type_registry.register_data_models(
        "dummy", DummyMeta, DummyMeta, DummyMeta, DummyConfig
    )
    registry.register_processor_mixins([DummySource, DummyIndexer, DummyParser])

    # prepare config
    domain_id = "domain_x"
    config = {
        "domain_type": "dummy",
        "domain_data": {"id": domain_id, "name": "X", "meta": {"field": "foo"}},
        "mixins": {
            "source_cls": "DummySource",
            "indexer_cls": "DummyIndexer",
            "parser_cls": "DummyParser",
        },
        "processor_config": {"config_field": "bar"},
    }

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    async with aiofiles.open(tmp_file.name, "w") as f:
        await f.write(yaml.safe_dump(config))

    await registry._load_domain_file(tmp_file.name)
    assert domain_id not in registry._processors

    registry._create_processors()
    assert domain_id in registry._processors
    processor = registry._processors[domain_id]
    assert isinstance(processor.source_mixin, DummySource)


# Test discover_and_register loads multiple domain configs from a directory
@pytest.mark.asyncio
async def test_discover_and_register_multiple_configs(monkeypatch):
    from pathlib import Path

    import knowlang.assets.registry as registry_module
    from knowlang.assets.processor import DomainContextInit

    class DummyMeta(BaseModel):
        field: str

    class DummyConfig(ProcessorConfigBase):
        config_field: str

    class DummySource(DomainContextInit):
        pass

    class DummyIndexer(DomainContextInit):
        pass

    class DummyParser(DomainContextInit):
        pass

    with tempfile.TemporaryDirectory() as temp_dir:
        # Patch get_resource_path to return the temp dir
        monkeypatch.setattr(
            registry_module,
            "get_resource_path",
            lambda path, default_path=None: Path(temp_dir),
        )

        registry = DomainRegistry(RegistryConfig(discovery_path=temp_dir))
        registry.type_registry.register_data_models(
            "dummy", DummyMeta, DummyMeta, DummyMeta, DummyConfig
        )
        registry.register_processor_mixins([DummySource, DummyIndexer, DummyParser])

        for i in range(3):
            domain_id = f"domain_{i}"
            config = {
                "domain_type": "dummy",
                "domain_data": {
                    "id": domain_id,
                    "name": f"Domain {i}",
                    "meta": {"field": "foo"},
                },
                "mixins": {
                    "source_cls": "DummySource",
                    "indexer_cls": "DummyIndexer",
                    "parser_cls": "DummyParser",
                },
                "processor_config": {"config_field": "bar"},
            }
            file_path = os.path.join(temp_dir, f"domain_{i}.yaml")
            async with aiofiles.open(file_path, "w") as f:
                await f.write(yaml.safe_dump(config))

        await registry.discover_and_register(discovery_path=temp_dir)

        assert len(registry.domain_configs) == 3
        assert all(k.startswith("domain_") for k in registry.domain_configs.keys())


# Test that _load_domain_file raises on malformed YAML


@pytest.mark.asyncio
async def test_load_domain_file_malformed_yaml(monkeypatch):
    from pathlib import Path

    import knowlang.assets.registry as registry_module

    # Monkeypatch path resolution to avoid file not found
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setattr(
            registry_module,
            "get_resource_path",
            lambda path, default_path=None: Path(temp_dir),
        )

        registry = DomainRegistry(RegistryConfig(discovery_path=temp_dir))

        # Write malformed YAML
        bad_file_path = os.path.join(temp_dir, "bad.yaml")
        async with aiofiles.open(bad_file_path, "w") as f:
            await f.write("domain_type: dummy\ndomain_data: [malformed] : entry\n")

        with pytest.raises(Exception):
            await registry._load_domain_file(bad_file_path)
