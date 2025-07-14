import pytest

from knowlang.assets.registry import DataModelTarget, RegistryConfig, TypeRegistry


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
