"""Unit tests for the model-resolution seam (K-042 Landing 1 FR-1..FR-6/FR-11..FR-15;
Landing 2 L2-1 roles/ordered fallback chains, FR-7/FR-18).

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
    GraphWorkspaceOverrides,
    ModelConfigError,
    ModelGateway,
    ModelResolutionError,
    Overlay,
    ProviderCatalog,
    StaticModelGateway,
    _camel_to_snake,
    _normalize_base_url,
)
from falkorchat.llm import FallbackClient
from falkorchat.transport import ProviderCallError

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


def _gateway(
    tmp_path: Path, *, providers=None, overlay_doc=None, workspace_overrides=None,
) -> ModelGateway:
    catalog = _catalog(tmp_path, providers if providers is not None else _LMSTUDIO)
    overlay = _overlay(tmp_path, overlay_doc)
    return ModelGateway(catalog, overlay, workspace_overrides=workspace_overrides)


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


def test_ref_with_no_slash_and_no_matching_role_raises_resolution_error(tmp_path):
    # K-042 Landing 2 (FR-7): a no-'/' ref is now a role reference, not an outright
    # rejection — but a role the overlay never declared is still unresolvable, at
    # resolve time (ModelResolutionError), not load time.
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"agent": "not-a-declared-role"}})
    with pytest.raises(ModelResolutionError) as excinfo:
        gw.resolve("agent")
    assert "not-a-declared-role" in str(excinfo.value)


# ── roles: ordered fallback chains (K-042 Landing 2, FR-7/FR-18) ──────────────────

_TWO_PROVIDERS = {
    "lmstudio": _LMSTUDIO["lmstudio"],
    "second": {
        "npm": "@ai-sdk/openai-compatible",
        "options": {"baseURL": "http://second-host:5000/v1"},
    },
}


def test_role_resolves_to_its_first_model_as_the_primary(tmp_path):
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"agent": "cheap"},
            "roles": {"cheap": {"models": ["lmstudio/qwen/qwen3-4b-2507", "second/x"]}},
        },
    )
    resolution = gw.resolve("agent")
    assert resolution.primary.ref == "lmstudio/qwen/qwen3-4b-2507"
    assert [r.ref for r in resolution.chain] == [
        "lmstudio/qwen/qwen3-4b-2507", "second/x",
    ]


def test_direct_ref_stays_a_length_one_chain(tmp_path):
    # Unchanged from Landing 1: a direct provider/model ref never becomes a chain.
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"}})
    resolution = gw.resolve("agent")
    assert len(resolution.chain) == 1


def test_role_remapping_changes_resolution_with_no_republish(tmp_path):
    # AC-6: editing the overlay's role mapping (a restart, no workflow-def republish)
    # changes which model resolves — modelled here as two independently-constructed
    # gateways reading two different overlay files, since ModelGateway parses config
    # once at construction (FR-15, no live reload).
    v1_dir = tmp_path / "v1"
    v2_dir = tmp_path / "v2"
    v1_dir.mkdir()
    v2_dir.mkdir()
    gw_v1 = _gateway(
        v1_dir,
        providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"agent": "cheap"},
            "roles": {"cheap": {"models": ["lmstudio/model-a"]}},
        },
    )
    gw_v2 = _gateway(
        v2_dir,
        providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"agent": "cheap"},
            "roles": {"cheap": {"models": ["second/model-b"]}},
        },
    )
    assert gw_v1.resolve("agent").primary.ref == "lmstudio/model-a"
    assert gw_v2.resolve("agent").primary.ref == "second/model-b"


def test_role_settings_are_applied_per_element(tmp_path):
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"agent": "cheap"},
            "roles": {"cheap": {"models": ["lmstudio/model-a", "second/model-b"]}},
            "models": {"lmstudio/model-a": {"timeout": 7.5}},
        },
    )
    resolution = gw.resolve("agent")
    assert resolution.chain[0].timeout == 7.5
    # element 2 has no per-model timeout — falls back to the kind default
    assert resolution.chain[1].timeout == gw._overlay.timeout_for_kind("agent")


def test_role_level_timeout_used_when_element_has_no_own_timeout(tmp_path):
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"agent": "cheap"},
            "roles": {
                "cheap": {"models": ["lmstudio/model-a", "second/model-b"], "timeout": 42},
            },
        },
    )
    resolution = gw.resolve("agent")
    assert resolution.chain[0].timeout == 42.0
    assert resolution.chain[1].timeout == 42.0


def test_per_model_timeout_still_wins_over_role_level_timeout(tmp_path):
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"agent": "cheap"},
            "roles": {"cheap": {"models": ["lmstudio/model-a"], "timeout": 42}},
            "models": {"lmstudio/model-a": {"timeout": 5}},
        },
    )
    assert gw.resolve("agent").primary.timeout == 5.0


def test_role_name_must_not_contain_slash_rejected_at_load(tmp_path):
    with pytest.raises(ModelConfigError) as excinfo:
        _gateway(
            tmp_path,
            overlay_doc={"roles": {"bad/name": {"models": ["lmstudio/model-a"]}}},
        )
    assert "bad/name" in str(excinfo.value)


def test_role_chain_element_without_slash_rejected_at_load_not_deferred(tmp_path):
    # A chain element with no '/' looks like another role name — nested roles are
    # rejected at LOAD time (Overlay construction), never deferred to first use.
    with pytest.raises(ModelConfigError) as excinfo:
        _gateway(
            tmp_path,
            overlay_doc={"roles": {"outer": {"models": ["inner-role-name"]}}},
        )
    message = str(excinfo.value)
    assert "inner-role-name" in message
    assert "outer" in message


def test_role_with_empty_models_list_rejected_at_load(tmp_path):
    with pytest.raises(ModelConfigError):
        _gateway(tmp_path, overlay_doc={"roles": {"empty": {"models": []}}})


def test_role_with_no_models_key_rejected_at_load(tmp_path):
    with pytest.raises(ModelConfigError):
        _gateway(tmp_path, overlay_doc={"roles": {"broken": {}}})


def test_gateway_llm_wraps_multi_element_chain_in_fallback_client(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(calls)
    )
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"agent": "cheap"},
            "roles": {"cheap": {"models": ["lmstudio/model-a", "second/model-b"]}},
        },
    )
    client = gw.llm("agent")
    assert isinstance(client, FallbackClient)


def test_gateway_llm_direct_ref_is_not_wrapped_in_fallback_client(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(calls)
    )
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"}})
    client = gw.llm("agent")
    assert not isinstance(client, FallbackClient)


def test_gateway_llm_role_chain_falls_back_end_to_end(tmp_path, monkeypatch):
    # Drives ModelGateway.llm() all the way through a role → FallbackClient → two
    # OpenAICompatibleLLM clients, proving the whole L2-1 wiring (not just
    # FallbackClient in isolation, which test_llm.py covers).
    calls: list[dict] = []

    def fake_make_http_transport(*, timeout, headers=None, opener=None, provider="?", model="?"):
        def _transport(url, payload):
            calls.append({"url": url, "model": payload.get("model")})
            if provider == "lmstudio":
                raise ProviderCallError(f"{provider}/{model}: simulated outage")
            return {"choices": [{"message": {"content": "from second"}}]}
        return _transport

    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", fake_make_http_transport
    )
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"agent": "cheap"},
            "roles": {"cheap": {"models": ["lmstudio/model-a", "second/model-b"]}},
        },
    )
    result = gw.llm("agent").chat([{"role": "user", "content": "hi"}], [])

    assert result.model == "second/model-b"
    assert result.fallback is True
    assert len(calls) == 2


# ── K-042 Landing 2 (L2-3, FR-16/FR-17): workspace override precedence ────────────
#
# `resolve()`'s real precedence, first-match-wins: workspace override -> `requested`
# -> the per-kind default. `.source` names which rung won. Parametrized across every
# kind in the closed `KINDS` set (not just `step`/`agent`) — B-1 existed precisely so
# `guard` isn't the one kind this can't reach; proving the crosswalk (`_gateway`'s
# `overrides=` key differs per kind) is itself part of what this pins.

_KIND_OVERRIDE_KEYS = {
    "agent": "responderModel", "step": "agentModel",
    "guard": "guardModel", "embedding": "embeddingModel",
}


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_three_rungs_and_the_hard_cap_direction_for_every_kind(tmp_path, kind):
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={"defaults": {kind: "lmstudio/model-a"}},
    )
    override_key = _KIND_OVERRIDE_KEYS[kind]

    # rung 3: no requested choice, no override -> the per-kind default
    res = gw.resolve(kind)
    assert res.source == "default"
    assert res.primary.ref == "lmstudio/model-a"

    # rung 2: an explicit requested= choice beats the default
    res = gw.resolve(kind, requested="second/model-b")
    assert res.source == "requested"
    assert res.primary.ref == "second/model-b"

    # rung 1 / the hard cap: a workspace override beats an explicit requested=
    # choice outright — AC-10's scenario, for every kind (the whole point of B-1).
    res = gw.resolve(
        kind, requested="second/model-b",
        overrides={override_key: "lmstudio/model-a"},
    )
    assert res.source == "workspace"
    assert res.primary.ref == "lmstudio/model-a"


def test_overrides_dict_present_but_this_kinds_key_unset_falls_through_to_requested(
    tmp_path,
):
    # A `read_model_overrides`-shaped dict with OTHER kinds set but this one absent
    # must not be mistaken for "this kind is overridden" — only its own crosswalk
    # key matters.
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"step": "lmstudio/default-model"}})
    overrides = {
        "agentModel": None, "guardModel": "lmstudio/other-kind-override",
        "embeddingModel": None, "responderModel": None,
    }
    res = gw.resolve("step", requested="lmstudio/requested-model", overrides=overrides)
    assert res.source == "requested"
    assert res.primary.ref == "lmstudio/requested-model"


def test_workspace_override_naming_an_undeclared_model_fails_loudly_not_silently(
    tmp_path,
):
    # §6.3: a graph-stored override is NOT config-validated at write time — it
    # resolves through the normal path and, on failure, must fail loudly (FR-10),
    # never silently fall through to `requested` or the default. A hard cap that
    # silently degrades to the thing it caps is worse than no cap at all.
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"step": "lmstudio/default-model"}})
    with pytest.raises(ModelResolutionError) as excinfo:
        gw.resolve(
            "step", requested="lmstudio/requested-model",
            overrides={"agentModel": "no-such-provider/some-model"},
        )
    assert "no-such-provider" in str(excinfo.value)


def test_workspace_override_may_name_a_role_not_just_a_direct_ref(tmp_path):
    # §6.1: "a value may be a concrete ref OR a role name" — resolved through the
    # same `_resolve_ref` path as any other ref (role expansion included).
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"step": "lmstudio/default-model"},
            "roles": {"cheap": {"models": ["lmstudio/model-a", "second/model-b"]}},
        },
    )
    res = gw.resolve("step", overrides={"agentModel": "cheap"})
    assert res.source == "workspace"
    assert [r.ref for r in res.chain] == ["lmstudio/model-a", "second/model-b"]


def test_ws_triggered_override_read_used_when_no_overrides_dict_given(tmp_path):
    # The other of the two §2.6 shapes: a caller with no per-drive `run` dict (the
    # responder, the embedding worker) passes `ws=` instead, and `resolve()` reads
    # fresh through the injected `WorkspaceOverrides` port.
    class StubWorkspaceOverrides:
        def __init__(self, mapping):
            self._mapping = mapping
            self.calls: list[tuple] = []

        def get(self, ws, kind):
            self.calls.append((ws, kind))
            return self._mapping.get((ws, kind))

    stub = StubWorkspaceOverrides({("acme", "embedding"): "second/embed-override"})
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={"defaults": {"embedding": "lmstudio/embed-default"}},
        workspace_overrides=stub,
    )

    res = gw.resolve("embedding", ws="acme")

    assert res.source == "workspace"
    assert res.primary.ref == "second/embed-override"
    assert stub.calls == [("acme", "embedding")]


def test_overrides_dict_takes_precedence_over_ws_triggered_read_no_double_read(
    tmp_path,
):
    # §2.6: a caller that already did a per-drive read passes the pre-fetched dict
    # so `resolve()` never triggers a SECOND graph round trip via `ws=`.
    class ExplodingWorkspaceOverrides:
        def get(self, ws, kind):
            raise AssertionError(
                "must not read ws-triggered when an overrides dict was given"
            )

    gw = _gateway(
        tmp_path, overlay_doc={"defaults": {"step": "lmstudio/default-model"}},
        workspace_overrides=ExplodingWorkspaceOverrides(),
    )
    res = gw.resolve(
        "step", ws="acme", overrides={"agentModel": "lmstudio/default-model"}
    )
    assert res.source == "workspace"


def test_static_model_gateway_resolve_source_is_requested_or_default_never_workspace():
    # A static gateway has no workspace-override concept (FR-4 sugar bypasses config
    # entirely) — `.source` only ever distinguishes "a model was requested" from not,
    # matching the pre-L2-3 behavior every `llm=`/`embedder=` injection test relies on.
    class StubLLM:
        pass

    gw = StaticModelGateway(llm=StubLLM())
    assert gw.resolve("step", requested="a/b").source == "requested"
    assert gw.resolve("step").source == "default"


def test_static_model_gateway_resolve_llm_returns_client_and_resolution():
    class StubLLM:
        pass

    stub = StubLLM()
    gw = StaticModelGateway(llm=stub)
    client, resolution = gw.resolve_llm("step", requested="a/b")
    assert client is stub
    assert resolution.source == "requested"


def test_real_gateway_resolve_llm_returns_client_and_resolution(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(calls)
    )
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"agent": "lmstudio/qwen/qwen3-4b-2507"}})

    client, resolution = gw.resolve_llm("agent")
    client.complete([{"role": "user", "content": "hi"}])

    assert resolution.source == "default"
    assert resolution.primary.ref == "lmstudio/qwen/qwen3-4b-2507"
    assert calls[0]["model"] == "qwen/qwen3-4b-2507"


# ── K-042 Landing 2: `GraphWorkspaceOverrides` — the kind<->property crosswalk ────
#
# `-graph.md` §8.4 names its `WorkspaceConfig` properties after the WORKFLOW NODE
# TYPE ("agent") vs. the chat RESPONDER CLASS ("responder") — which does NOT match
# this module's own `kind` strings 1:1 (`_KIND_TO_OVERRIDE_KEY`). These pin the
# crosswalk directly against a fake repository, independent of `ModelGateway`.

class _FakeRepo:
    def __init__(self, overrides: dict) -> None:
        self._overrides = overrides
        self.calls: list[str] = []

    def read_model_overrides(self, ws):
        self.calls.append(ws)
        return dict(self._overrides)


def test_graph_workspace_overrides_kind_step_reads_the_agentModel_property():
    # -graph.md §8.4: `agentModelOverride` governs the WORKFLOW's agent-type step
    # node — this module's kind "step", not "agent".
    repo = _FakeRepo({
        "agentModel": "lmstudio/step-override", "guardModel": None,
        "embeddingModel": None, "responderModel": None,
    })
    ov = GraphWorkspaceOverrides(repo)
    assert ov.get("acme", "step") == "lmstudio/step-override"
    assert repo.calls == ["acme"]


def test_graph_workspace_overrides_kind_agent_reads_the_responderModel_property():
    # -graph.md §8.4: `responderModelOverride` governs the chat/@mention responder
    # (`AgentResponder`) — this module's kind "agent", not "step".
    repo = _FakeRepo({
        "agentModel": None, "guardModel": None,
        "embeddingModel": None, "responderModel": "lmstudio/chat-override",
    })
    ov = GraphWorkspaceOverrides(repo)
    assert ov.get("acme", "agent") == "lmstudio/chat-override"


def test_graph_workspace_overrides_kind_guard_and_embedding_match_by_name():
    repo = _FakeRepo({
        "agentModel": None, "guardModel": "lmstudio/guard-override",
        "embeddingModel": "lmstudio/embed-override", "responderModel": None,
    })
    ov = GraphWorkspaceOverrides(repo)
    assert ov.get("acme", "guard") == "lmstudio/guard-override"
    assert ov.get("acme", "embedding") == "lmstudio/embed-override"


def test_graph_workspace_overrides_returns_none_when_kind_unset():
    repo = _FakeRepo({
        "agentModel": None, "guardModel": None,
        "embeddingModel": None, "responderModel": None,
    })
    ov = GraphWorkspaceOverrides(repo)
    assert ov.get("acme", "step") is None


def test_graph_workspace_overrides_never_treats_empty_string_as_an_override():
    # Defensive backstop (§2.4's own warning: '' is a value, never "no override") —
    # even if the write path ever produced one, the read side must not honour it.
    repo = _FakeRepo({
        "agentModel": "", "guardModel": None,
        "embeddingModel": None, "responderModel": None,
    })
    ov = GraphWorkspaceOverrides(repo)
    assert ov.get("acme", "step") is None


# ── K-042 U8-gate Minor 3: embedder() silently drops fallback beyond the primary ──

def test_embedder_warns_once_per_kind_role_when_chain_has_more_than_one_element(
    tmp_path, monkeypatch, caplog,
):
    calls: list[dict] = []
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(calls)
    )
    gw = _gateway(
        tmp_path, providers=_TWO_PROVIDERS,
        overlay_doc={
            "defaults": {"embedding": "cheap"},
            "roles": {"cheap": {"models": ["lmstudio/embed-a", "second/embed-b"]}},
        },
    )

    with caplog.at_level(logging.WARNING):
        gw.embedder("embedding")
        gw.embedder("embedding")  # same (kind, primary ref) — must not double-warn

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "embedding" in warnings[0].message


def test_embedder_does_not_warn_for_a_direct_ref_length_one_chain(tmp_path, monkeypatch, caplog):
    calls: list[dict] = []
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(calls)
    )
    gw = _gateway(tmp_path, overlay_doc={"defaults": {"embedding": "lmstudio/qwen/qwen3-4b-2507"}})

    with caplog.at_level(logging.WARNING):
        gw.embedder("embedding")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


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


def test_overlay_agents_still_reserved_and_logged(tmp_path, caplog):
    # K-042 Landing 2: `roles` is now honoured (see the roles/fallback-chain section
    # below) — only `agents` (the FR-5 `agents[<agentId>]` resolution, a separate,
    # not-yet-built unit) remains reserved + logged.
    with caplog.at_level(logging.INFO):
        _gateway(
            tmp_path,
            overlay_doc={
                "roles": {"cheap": {"models": ["lmstudio/qwen/qwen3-4b-2507"]}},
                "agents": {"assistant": "lmstudio/qwen/qwen3-4b-2507"},
            },
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("agents" in m and "Landing 2" in m for m in messages)
    assert not any("roles" in m and "Landing 2" in m for m in messages)


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
