"""Unit tests for the model-resolution seam (K-042 Landing 1, FR-1..FR-6/FR-11..FR-15).

Entirely offline: two hand-edited JSON files in, a `Resolution`/client out — no
network, no FalkorDB. Fixtures live in `tests/data/`; the `_model_config_env`
autouse fixture (`conftest.py`) never lets this suite depend on a developer's real
`~/.config/opencode/opencode.json`.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

import pytest

from falkorchat import config
from falkorchat.modelconfig import (
    KINDS,
    ModelConfigError,
    ModelGateway,
    ModelResolutionError,
    Overlay,
    ProviderCatalog,
    StaticModelGateway,
    _camel_to_snake,
    _normalize_base_url,
)

_DATA = Path(__file__).resolve().parent / "data"


def _write_json(tmp_path: Path, name: str, doc: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _catalog(tmp_path: Path, providers: dict) -> ProviderCatalog:
    path = _write_json(tmp_path, "opencode.json", {"provider": providers})
    return ProviderCatalog.load(path)


def _overlay(tmp_path: Path, doc: dict | None = None) -> Overlay:
    path = _write_json(tmp_path, "models.json", doc or {})
    return Overlay.load(path)


_LMSTUDIO = {
    "lmstudio": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "LM Studio",
        "options": {"baseURL": "http://localhost:1234/v1"},
        "models": {"qwen/qwen3-4b-2507": {"name": "Qwen3 4B"}},
    }
}


def _gateway(tmp_path: Path, *, providers=None, overlay_doc=None) -> ModelGateway:
    catalog = _catalog(tmp_path, providers if providers is not None else _LMSTUDIO)
    overlay = _overlay(tmp_path, overlay_doc)
    return ModelGateway(catalog, overlay)


# ── {env:}/{file:} substitution (FR-12, §4.8) ──────────────────────────────────────


def test_env_apikey_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-secret-123")
    providers = {
        "cloud": {
            "npm": "@ai-sdk/openai",
            "options": {"baseURL": "https://api.example.com/v1", "apiKey": "{env:TEST_API_KEY}"},
        }
    }
    gw = _gateway(
        tmp_path, providers=providers,
        overlay_doc={"defaults": {"agent": "cloud/gpt-4o"}},
    )
    resolution = gw.resolve("agent")
    assert resolution.primary.api_key is not None
    assert resolution.primary.api_key.get() == "sk-secret-123"


def test_env_apikey_missing_raises_naming_variable_and_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_KEY_VAR", raising=False)
    providers = {
        "cloud": {
            "options": {"baseURL": "https://api.example.com/v1", "apiKey": "{env:MISSING_KEY_VAR}"},
        }
    }
    catalog_path = _write_json(tmp_path, "opencode.json", {"provider": providers})
    catalog = ProviderCatalog.load(catalog_path)  # parsing alone never substitutes
    overlay = _overlay(tmp_path, {"defaults": {"agent": "cloud/gpt-4o"}})
    with pytest.raises(ModelConfigError) as excinfo:
        ModelGateway(catalog, overlay)  # substitution happens while building providers
    message = str(excinfo.value)
    assert "MISSING_KEY_VAR" in message
    assert catalog_path in message


def test_secret_never_appears_in_repr_log_or_error(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TEST_API_KEY", "sk-very-secret-xyz")
    providers = {
        "cloud": {
            "options": {"baseURL": "https://api.example.com/v1", "apiKey": "{env:TEST_API_KEY}"},
        }
    }
    with caplog.at_level(logging.INFO):
        gw = _gateway(
            tmp_path, providers=providers,
            overlay_doc={"defaults": {"agent": "cloud/gpt-4o"}},
        )
        resolution = gw.resolve("agent")

    assert "sk-very-secret-xyz" not in repr(resolution.primary)
    assert "sk-very-secret-xyz" not in repr(resolution.primary.api_key)
    assert "sk-very-secret-xyz" not in str(resolution.primary.api_key)
    for record in caplog.records:
        assert "sk-very-secret-xyz" not in record.getMessage()

    try:
        raise ModelResolutionError(f"boom near {resolution.primary!r}")
    except ModelResolutionError as exc:
        assert "sk-very-secret-xyz" not in str(exc)


def test_file_substitution_resolves_and_strips_whitespace(tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("sk-from-file\n", encoding="utf-8")
    providers = {
        "cloud": {
            "options": {
                "baseURL": "https://api.example.com/v1",
                "apiKey": f"{{file:{key_file}}}",
            },
        }
    }
    gw = _gateway(
        tmp_path, providers=providers,
        overlay_doc={"defaults": {"agent": "cloud/gpt-4o"}},
    )
    resolution = gw.resolve("agent")
    assert resolution.primary.api_key.get() == "sk-from-file"


def test_file_substitution_missing_file_raises(tmp_path):
    providers = {
        "cloud": {
            "options": {
                "baseURL": "https://api.example.com/v1",
                "apiKey": "{file:/nonexistent/path/key.txt}",
            },
        }
    }
    catalog = _catalog(tmp_path, providers)
    overlay = _overlay(tmp_path, {"defaults": {"agent": "cloud/gpt-4o"}})
    with pytest.raises(ModelConfigError) as excinfo:
        ModelGateway(catalog, overlay)
    assert "/nonexistent/path/key.txt" in str(excinfo.value)


# ── the §4.3 /v1 normalization table — every row, including both rejects ──────────


@pytest.mark.parametrize(
    "declared, expected",
    [
        ("http://localhost:1234/v1", "http://localhost:1234/v1"),
        ("http://192.168.0.69:1234", "http://192.168.0.69:1234/v1"),
        ("http://host:1234/", "http://host:1234/v1"),
        ("https://api.openai.com/v1", "https://api.openai.com/v1"),
        ("https://api.anthropic.com", "https://api.anthropic.com/v1"),
        ("https://api.anthropic.com/v1/", "https://api.anthropic.com/v1"),
        ("http://gw.lan/openai/v1", "http://gw.lan/openai/v1"),
    ],
)
def test_v1_normalization_table(declared, expected):
    resolved, _source = _normalize_base_url(declared, provider_id="p", source="f")
    assert resolved == expected


@pytest.mark.parametrize(
    "declared", ["192.168.0.69:1234", "localhost:1234/v1"],
)
def test_v1_normalization_rejects_scheme_typos(declared):
    with pytest.raises(ModelConfigError):
        _normalize_base_url(declared, provider_id="p", source="f")


def test_v1_rule_reports_source_rule_vs_verbatim():
    _, source_appended = _normalize_base_url("http://host:1234", provider_id="p", source="f")
    _, source_verbatim = _normalize_base_url("http://host:1234/v1", provider_id="p", source="f")
    assert source_appended == "rule"
    assert source_verbatim == "verbatim"


# ── overlay provider baseURL override — the escape hatch (§4.3) ───────────────────


def test_overlay_provider_override_wins_over_file_and_rule(tmp_path):
    gw = _gateway(
        tmp_path,
        overlay_doc={
            "defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"},
            "providers": {"lmstudio": {"baseURL": "http://overridden:9999/custom"}},
        },
    )
    resolution = gw.resolve("agent")
    assert resolution.primary.base_url == "http://overridden:9999/custom"


def test_overlay_provider_override_is_used_verbatim_no_v1_autoappend(tmp_path):
    # The override wins over the /v1 rule too — a bare host with no /v1 stays bare
    # if that's what the admin wrote (§4.3: "wins outright ... over the rule").
    gw = _gateway(
        tmp_path,
        overlay_doc={
            "defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"},
            "providers": {"lmstudio": {"baseURL": "http://overridden:9999"}},
        },
    )
    resolution = gw.resolve("agent")
    assert resolution.primary.base_url == "http://overridden:9999"


def test_startup_info_line_names_the_winning_source(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _gateway(
            tmp_path,
            overlay_doc={"providers": {"lmstudio": {"baseURL": "http://overridden:9999/x"}}},
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("lmstudio" in m and "override" in m for m in messages)


# ── ref grammar: split-on-first-/, no-/ rejected with the Landing-2 message ────────


def test_ref_splits_on_first_slash_only(tmp_path):
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"}})
    resolved = gw.resolve("agent").primary
    assert resolved.provider == "lmstudio"
    assert resolved.model == "qwen/qwen3-4b-2507"
    assert resolved.ref == "lmstudio/qwen/qwen3-4b-2507"


def test_ref_with_no_slash_is_rejected_with_landing_2_message(tmp_path):
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"agent": "just-a-role-name"}})
    with pytest.raises(ModelConfigError) as excinfo:
        gw.resolve("agent")
    message = str(excinfo.value).lower()
    assert "landing 2" in message or "role" in message


# ── both real sample files parse unmodified (AC-1) ─────────────────────────────────


def test_severino_sample_file_parses_unmodified():
    path = str(_DATA / "opencode_severino_sample.json")
    catalog = ProviderCatalog.load(path)
    assert "lmstudio" in catalog.provider_ids()
    raw = catalog.raw("lmstudio")
    assert raw["options"]["baseURL"] == "http://localhost:1234/v1"


def test_stakeholder_sample_file_parses_unmodified():
    # Mirrors the real ~/.config/opencode/opencode.json content (plan §2.3/§2.4) as
    # a committed fixture — the real file lives outside the repo and outside any
    # machine this suite must run on (M-2: no dependency on a real home dir).
    path = str(_DATA / "opencode_stakeholder_sample.json")
    catalog = ProviderCatalog.load(path)
    assert "lmstudio" in catalog.provider_ids()
    raw = catalog.raw("lmstudio")
    assert raw["options"]["baseURL"] == "http://192.168.0.69:1234"
    assert "apiKey" not in raw["options"]


def test_stakeholder_sample_resolves_the_declared_v1_gap(tmp_path):
    # §2.4: LM Studio serves 7 models, none of which appear in the file's `models`
    # map — the map is metadata, not an allow-list (§4.4). A ref naming a model
    # absent from the map still resolves.
    catalog = ProviderCatalog.load(str(_DATA / "opencode_stakeholder_sample.json"))
    overlay = _overlay(tmp_path, {"defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"}})
    gw = ModelGateway(catalog, overlay)
    resolved = gw.resolve("agent").primary
    assert resolved.base_url == "http://192.168.0.69:1234/v1"
    assert resolved.model == "qwen/qwen3-4b-2507"  # absent from the file's `models` map


# ── unknown top-level keys accepted (shared file: silent; overlay: logged) ────────


def test_shared_file_unknown_top_level_keys_ignored_without_error(tmp_path):
    doc = {
        "$schema": "https://opencode.ai/config.schema.json",
        "provider": _LMSTUDIO,
        "agent": {"severino": {"model": "lmstudio/qwen/qwen3-4b-2507"}},
        "mcp": {"foo": "bar"},
    }
    path = _write_json(tmp_path, "opencode.json", doc)
    catalog = ProviderCatalog.load(path)
    assert catalog.provider_ids() == ["lmstudio"]


def test_overlay_unknown_top_level_keys_accepted_and_logged(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _gateway(tmp_path, overlay_doc={"someFutureThing": {"x": 1}})
    messages = [r.getMessage() for r in caplog.records]
    assert any("someFutureThing" in m for m in messages)


def test_overlay_roles_and_agents_reserved_and_logged(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _gateway(
            tmp_path,
            overlay_doc={
                "roles": {"cheap": {"models": ["lmstudio/x"]}},
                "agents": {"assistant": "lmstudio/qwen/qwen3-4b-2507"},
            },
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("roles" in m and "Landing 2" in m for m in messages)
    assert any("agents" in m and "Landing 2" in m for m in messages)


# ── per-kind defaults (AC-5) ─────────────────────────────────────────────────────


def test_four_differing_per_kind_defaults_each_resolve(tmp_path):
    providers = {
        "lmstudio": _LMSTUDIO["lmstudio"],
        "second": {
            "npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": "http://second-host:5000/v1"},
        },
    }
    gw = _gateway(
        tmp_path, providers=providers,
        overlay_doc={
            "defaults": {
                "agent": "lmstudio/agent-model",
                "step": "lmstudio/step-model",
                "guard": "second/guard-model",
                "embedding": "second/embedding-model",
            }
        },
    )
    resolved_by_kind = {kind: gw.resolve(kind).primary for kind in KINDS}
    assert resolved_by_kind["agent"].model == "agent-model"
    assert resolved_by_kind["step"].model == "step-model"
    assert resolved_by_kind["guard"].model == "guard-model"
    assert resolved_by_kind["embedding"].model == "embedding-model"
    assert resolved_by_kind["guard"].base_url == "http://second-host:5000/v1"


# ── per-model settings: timeout/dim reserved, everything else passes through ──────


def test_per_model_timeout_overrides_the_kind_default(tmp_path):
    gw = _gateway(
        tmp_path,
        overlay_doc={
            "defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"},
            "timeouts": {"agent": 180},
            "models": {"lmstudio/qwen/qwen3-4b-2507": {"timeout": 12.5}},
        },
    )
    assert gw.resolve("agent").primary.timeout == 12.5


def test_kind_default_timeout_used_when_no_per_model_override(tmp_path):
    gw = _gateway(
        tmp_path,
        overlay_doc={
            "defaults": {"embedding": "lmstudio/qwen/qwen3-4b-2507"},
            "timeouts": {"embedding": 30},
        },
    )
    assert gw.resolve("embedding").primary.timeout == 30.0


def test_dim_is_honoured(tmp_path):
    gw = _gateway(
        tmp_path,
        overlay_doc={
            "defaults": {"embedding": "lmstudio/qwen/qwen3-4b-2507"},
            "models": {"lmstudio/qwen/qwen3-4b-2507": {"dim": 1024}},
        },
    )
    assert gw.resolve("embedding").primary.dim == 1024


def test_unknown_passthrough_key_reaches_params_camel_to_snake(tmp_path):
    gw = _gateway(
        tmp_path,
        overlay_doc={
            "defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"},
            "models": {
                "lmstudio/qwen/qwen3-4b-2507": {
                    "topP": 0.9, "maxCompletionTokens": 512, "reasoningEffort": "low",
                }
            },
        },
    )
    params = gw.resolve("agent").primary.params
    assert params == {"top_p": 0.9, "max_completion_tokens": 512, "reasoning_effort": "low"}


def test_reserved_keys_do_not_leak_into_params(tmp_path):
    gw = _gateway(
        tmp_path,
        overlay_doc={
            "defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"},
            "models": {
                "lmstudio/qwen/qwen3-4b-2507": {
                    "timeout": 10, "dim": 4, "protocol": "openai", "topP": 0.5,
                }
            },
        },
    )
    params = gw.resolve("agent").primary.params
    assert params == {"top_p": 0.5}


@pytest.mark.parametrize(
    "camel, snake",
    [("topP", "top_p"), ("maxCompletionTokens", "max_completion_tokens"),
     ("reasoningEffort", "reasoning_effort"), ("already_snake", "already_snake")],
)
def test_camel_to_snake(camel, snake):
    assert _camel_to_snake(camel) == snake


# ── unknown provider — resolution-time, loud (FR-9/FR-10 asymmetry, §4.4) ─────────


def test_unknown_provider_raises_resolution_error(tmp_path):
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"agent": "nope/thing"}})
    with pytest.raises(ModelResolutionError) as excinfo:
        gw.resolve("agent")
    assert "nope" in str(excinfo.value)


def test_no_default_and_no_requested_raises_resolution_error(tmp_path):
    gw = _gateway(tmp_path, overlay_doc={})
    with pytest.raises(ModelResolutionError):
        gw.resolve("agent")


# ── protocol gate (§4.7) ────────────────────────────────────────────────────────


def test_unsupported_protocol_raises_at_construction(tmp_path):
    providers = {
        "claude": {
            "npm": "@ai-sdk/anthropic",
            "options": {"baseURL": "https://api.anthropic.com/v1", "apiKey": "{env:X}"},
        }
    }
    import os

    os.environ.setdefault("X", "unused")
    catalog = _catalog(tmp_path, providers)
    overlay = _overlay(tmp_path, {})
    with pytest.raises(ModelConfigError) as excinfo:
        ModelGateway(catalog, overlay)
    assert "claude" in str(excinfo.value)
    assert "anthropic" in str(excinfo.value).lower()


def test_openai_protocol_is_the_default_when_npm_unnamed(tmp_path):
    providers = {"x": {"options": {"baseURL": "http://host:1"}}}
    gw = _gateway(tmp_path, providers=providers, overlay_doc={"defaults": {"agent": "x/m"}})
    assert gw.resolve("agent").primary.protocol == "openai"


def test_overlay_protocol_override(tmp_path):
    providers = {"x": {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": "http://host:1"}}}
    # override to a still-unsupported protocol should still raise
    catalog = _catalog(tmp_path, providers)
    overlay = _overlay(tmp_path, {"providers": {"x": {"protocol": "openai"}}})
    gw = ModelGateway(catalog, overlay)
    # no default configured, but constructing succeeded (protocol accepted)
    assert gw is not None


# ── the {no-file}, no-network offline import guarantee ────────────────────────────


def test_importing_modelconfig_touches_no_file_and_no_network(monkeypatch):
    # Sanity: importing the module (already done at collection time) must not have
    # required any env var or file. Constructing nothing here is the point.
    import falkorchat.modelconfig  # noqa: F401


# ── StaticModelGateway sugar (§3, A-4) ─────────────────────────────────────────────


def test_static_model_gateway_ignores_requested_and_returns_injected_llm():
    class StubLLM:
        pass

    stub = StubLLM()
    gw = StaticModelGateway(llm=stub)
    assert gw.has_chat() is True
    assert gw.has_embedder() is False
    assert gw.llm("step", requested="whatever/model") is stub


def test_static_model_gateway_warns_once_per_kind_ref(caplog):
    class StubLLM:
        pass

    gw = StaticModelGateway(llm=StubLLM())
    with caplog.at_level(logging.WARNING):
        gw.resolve("step", requested="a/b")
        gw.resolve("step", requested="a/b")
        gw.resolve("step", requested="a/c")
        gw.resolve("guard", requested="a/b")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3  # (step,a/b) once, (step,a/c) once, (guard,a/b) once


def test_static_model_gateway_raises_when_nothing_injected():
    gw = StaticModelGateway()
    assert gw.has_chat() is False
    with pytest.raises(ModelResolutionError):
        gw.llm("step")


# ── AC-13 tripwire (FR-20) — exercised at the config layer ────────────────────────


def test_from_env_requires_opencode_config_path(monkeypatch):
    monkeypatch.delenv("FALKORCHAT_OPENCODE_CONFIG", raising=False)
    from falkorchat import config as config_mod

    monkeypatch.setattr(config_mod, "OPENCODE_CONFIG_PATH", None)
    with pytest.raises(ModelConfigError) as excinfo:
        ModelGateway.from_env()
    assert "FALKORCHAT_OPENCODE_CONFIG" in str(excinfo.value)


def test_setting_a_legacy_env_var_aborts_via_assert_no_legacy_model_env(monkeypatch):
    # Code-review Major 1: `assert_no_legacy_model_env()` itself, driven with an
    # actual legacy var SET (not merely absent) — the AC-13 behavior the plan's L1-5
    # done-condition and §10 item 14 both name explicitly.
    from falkorchat import config as config_mod

    monkeypatch.setenv("FALKORCHAT_LLM_MODEL", "qwen/qwen3-4b-2507")
    with pytest.raises(RuntimeError) as excinfo:
        config_mod.assert_no_legacy_model_env()
    message = str(excinfo.value)
    assert "FALKORCHAT_LLM_MODEL" in message
    assert "FALKORCHAT_OPENCODE_CONFIG" in message  # names the replacement, too
    assert "FALKORCHAT_MODEL_CONFIG" in message


def test_setting_multiple_legacy_env_vars_names_all_of_them(monkeypatch):
    from falkorchat import config as config_mod

    monkeypatch.setenv("FALKORCHAT_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("FALKORCHAT_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b")
    with pytest.raises(RuntimeError) as excinfo:
        config_mod.assert_no_legacy_model_env()
    message = str(excinfo.value)
    assert "FALKORCHAT_LLM_BASE_URL" in message
    assert "FALKORCHAT_EMBEDDING_MODEL" in message


@pytest.mark.parametrize("legacy_var", list(config.LEGACY_MODEL_ENV_VARS))
def test_each_legacy_env_var_individually_aborts_startup(monkeypatch, legacy_var):
    monkeypatch.setenv(legacy_var, "x")
    with pytest.raises(RuntimeError) as excinfo:
        config.assert_no_legacy_model_env()
    assert legacy_var in str(excinfo.value)


def test_from_env_raises_when_a_legacy_env_var_is_set(monkeypatch):
    # The end-to-end path: ModelGateway.from_env() calls assert_no_legacy_model_env()
    # FIRST, before ever reading FALKORCHAT_OPENCODE_CONFIG/FALKORCHAT_MODEL_CONFIG —
    # a legacy var aborts even though the two replacement files (the conftest.py
    # autouse fixtures) are perfectly valid.
    monkeypatch.setenv("FALKORCHAT_EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    with pytest.raises(RuntimeError) as excinfo:
        ModelGateway.from_env()
    assert "FALKORCHAT_EMBEDDING_BASE_URL" in str(excinfo.value)


# ── end-to-end through a real gateway + a fake HTTP layer (AC-4/AC-5/AC-12) ────────
#
# `ModelGateway._build_llm`/`_build_embedder` build a transport via
# `transport.make_http_transport` at call time (looked up fresh off the `transport`
# module on every `.llm()`/`.embedder()` call, not bound at import time) — so
# monkeypatching that one factory function intercepts the URL/model/timeout every
# resolved client would actually send, with no real network touched.


def _fake_transport_factory(calls: list[dict]):
    def fake_make_http_transport(*, timeout, headers=None, opener=None, provider="?", model="?"):
        def _transport(url, payload):
            calls.append({"url": url, "model": payload.get("model"), "timeout": timeout})
            return {"choices": [{"message": {"content": "ok"}}], "data": [{"embedding": [0.0]}]}
        return _transport
    return fake_make_http_transport


def test_two_steps_naming_different_models_hit_different_urls_and_models(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(calls)
    )
    providers = {
        "lmstudio": _LMSTUDIO["lmstudio"],
        "second": {"options": {"baseURL": "http://second-host:5000/v1"}},
    }
    gw = _gateway(tmp_path, providers=providers, overlay_doc={})

    gw.llm("step", requested="lmstudio/model-a").complete([{"role": "user", "content": "hi"}])
    gw.llm("step", requested="second/model-b").complete([{"role": "user", "content": "hi"}])

    assert calls[0]["url"] == "http://localhost:1234/v1/chat/completions"
    assert calls[0]["model"] == "model-a"
    assert calls[1]["url"] == "http://second-host:5000/v1/chat/completions"
    assert calls[1]["model"] == "model-b"
    assert calls[0]["url"] != calls[1]["url"]


def test_four_per_kind_defaults_each_reach_a_different_url_or_model(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(calls)
    )
    providers = {
        "lmstudio": _LMSTUDIO["lmstudio"],
        "second": {"options": {"baseURL": "http://second-host:5000/v1"}},
    }
    gw = _gateway(
        tmp_path, providers=providers,
        overlay_doc={
            "defaults": {
                "agent": "lmstudio/agent-model", "step": "lmstudio/step-model",
                "guard": "second/guard-model", "embedding": "second/embedding-model",
            }
        },
    )

    gw.llm("agent").complete([{"role": "user", "content": "hi"}])
    gw.llm("step").complete([{"role": "user", "content": "hi"}])
    gw.llm("guard").complete([{"role": "user", "content": "hi"}])
    gw.embedder("embedding").embed("hi")

    models_seen = [c["model"] for c in calls]
    assert models_seen == ["agent-model", "step-model", "guard-model", "embedding-model"]
    assert len(set(models_seen)) == 4


def test_per_model_timeout_reaches_the_transport_factory(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(calls)
    )
    gw = _gateway(
        tmp_path,
        overlay_doc={
            "defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"},
            "models": {"lmstudio/qwen/qwen3-4b-2507": {"timeout": 7.5}},
        },
    )

    gw.llm("agent").complete([{"role": "user", "content": "hi"}])

    assert calls[0]["timeout"] == 7.5


def test_kind_default_timeout_reaches_the_transport_factory_when_no_per_model_override(
    tmp_path, monkeypatch,
):
    calls: list[dict] = []
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(calls)
    )
    gw = _gateway(
        tmp_path,
        overlay_doc={
            "defaults": {"embedding": "lmstudio/qwen/qwen3-4b-2507"},
            "timeouts": {"embedding": 30},
        },
    )

    gw.embedder("embedding").embed("hi")

    assert calls[0]["timeout"] == 30.0


# ── FR-4 enforcement: only modelconfig.py/tests/ construct the raw clients ────────


_FALKORCHAT_PKG = Path(__file__).resolve().parents[1] / "falkorchat"
_ALLOWED_MODULES = {"modelconfig.py"}
_FORBIDDEN_NAMES = {"OpenAICompatibleLLM", "OpenAICompatibleEmbedder"}


def test_fr4_only_modelconfig_constructs_openai_compatible_clients_directly():
    offenders: list[str] = []
    for path in sorted(_FALKORCHAT_PKG.glob("*.py")):
        if path.name in _ALLOWED_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in _FORBIDDEN_NAMES:
                offenders.append(f"{path.relative_to(_FALKORCHAT_PKG)}:{node.lineno}: {name}(...)")
    assert not offenders, (
        "FR-4 violation — direct client construction outside modelconfig.py:\n"
        + "\n".join(offenders)
    )
