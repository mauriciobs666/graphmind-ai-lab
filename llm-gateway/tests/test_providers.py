"""Provider contract and the lazy registry (plan §3.2, S7; FR-4, FR-15, FR-18, AC-8)."""

import sys

import pytest
from gwstandin import FakeProvider

from llmgateway.providers import BUILTIN_KINDS, ProviderRegistry, UnknownProviderKind
from llmgateway.providers.base import SettingUnsupported, encode_body, map_settings
from llmgateway.types import CallFailed, FailureClass


def test_builtin_kinds_are_the_four_with_module_paths():
    assert BUILTIN_KINDS == {
        "lmstudio": "llmgateway.providers.lmstudio",
        "ollama": "llmgateway.providers.ollama",
        "openai_compat": "llmgateway.providers.openai_compat",
        "anthropic": "llmgateway.providers.anthropic",
    }


def test_unwritten_builtin_module_is_a_named_config_error():
    reg = ProviderRegistry({"ghost": "llmgateway.providers.no_such_module"})
    with pytest.raises(UnknownProviderKind, match="ghost") as e:
        reg.get("ghost")
    assert e.value.failure_class is FailureClass.CONFIG


def test_unregistered_kind():
    with pytest.raises(UnknownProviderKind, match="not registered"):
        ProviderRegistry({}).get("nope")


@pytest.fixture
def modules(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in ("zz_good_prov", "zz_bad_prov", "zz_empty_prov", "zz_wrongkind_prov"):
        sys.modules.pop(name, None)


GOOD = """
from gwstandin import FakeProvider
PROVIDER = FakeProvider("zkind")
"""


def test_registry_is_lazy_and_caches(modules):
    (modules / "zz_good_prov.py").write_text(GOOD)
    reg = ProviderRegistry({"zkind": "zz_good_prov"})
    assert "zz_good_prov" not in sys.modules and reg.loaded() == frozenset()
    first = reg.get("zkind")
    assert "zz_good_prov" in sys.modules and reg.loaded() == {"zkind"}
    assert first.kind == "zkind" and reg.get("zkind") is first


def test_other_kinds_are_not_imported_when_one_is_used(modules):
    (modules / "zz_good_prov.py").write_text(GOOD)
    reg = ProviderRegistry({"zkind": "zz_good_prov", "other": "zz_bad_prov"})
    reg.get("zkind")
    assert "zz_bad_prov" not in sys.modules


def test_broken_import_inside_a_provider_module_propagates(modules):
    (modules / "zz_bad_prov.py").write_text("import a_module_that_does_not_exist_zz\n")
    with pytest.raises(ModuleNotFoundError, match="a_module_that_does_not_exist_zz"):
        ProviderRegistry({"bad": "zz_bad_prov"}).get("bad")


def test_module_without_provider_or_with_wrong_kind(modules):
    (modules / "zz_empty_prov.py").write_text("X = 1\n")
    (modules / "zz_wrongkind_prov.py").write_text(
        "from gwstandin import FakeProvider\nPROVIDER = FakeProvider('other')\n"
    )
    reg = ProviderRegistry({"a": "zz_empty_prov", "b": "zz_wrongkind_prov"})
    with pytest.raises(UnknownProviderKind, match="no PROVIDER"):
        reg.get("a")
    with pytest.raises(UnknownProviderKind, match="not a Provider of that kind"):
        reg.get("b")


def test_register_seam_adds_a_kind_without_touching_the_registry(modules):
    """AC-8: a fifth kind, by module path or instance."""
    reg = ProviderRegistry()
    inst = FakeProvider("fifth")
    reg.register("fifth", inst)
    assert reg.get("fifth") is inst and "fifth" in reg.kinds()
    (modules / "zz_good_prov.py").write_text(GOOD)
    reg.register("zkind", "zz_good_prov")
    assert reg.get("zkind").kind == "zkind"


def test_map_settings_passes_values_unchanged_in_given_order():
    table = {"seed": "seed", "temperature": "temperature"}
    out = map_settings("p", {"temperature": 0.0, "seed": 7}, table)
    assert list(out.items()) == [("temperature", 0.0), ("seed", 7)]
    assert type(out["seed"]) is int


def test_map_settings_unsupported_required_names_it_and_optional_is_skipped():
    with pytest.raises(SettingUnsupported, match="top_k") as e:
        map_settings("p", {"top_k": 5}, {"top_k": None})
    assert isinstance(e.value, CallFailed) and e.value.setting == "top_k"
    assert map_settings("p", {"top_k": 5, "seed": 1}, {"seed": "seed"}, {"top_k"}) == {"seed": 1}
    with pytest.raises(SettingUnsupported):  # absent from the table = unsupported too
        map_settings("p", {"min_p": 0.1}, {})


def test_encode_body_is_compact_utf8_and_preserves_values():
    schema = {"type": "object", "properties": {"é": {"type": "integer"}}}
    raw = encode_body({"seed": 42, "response_format": schema})
    want = '{"seed":42,"response_format":{"type":"object","properties":{"é":{"type":"integer"}}}}'
    assert raw == want.encode()


def test_map_settings_never_rounds_a_float():
    table = {"temperature": "temperature", "top_p": "top_p"}
    out = map_settings("p", {"temperature": 0.123456789, "top_p": 0.955}, table)
    assert out == {"temperature": 0.123456789, "top_p": 0.955}
