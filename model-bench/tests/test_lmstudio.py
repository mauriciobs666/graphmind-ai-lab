"""`modelbench.lmstudio` — offline, against stubbed HTTP and recorded/derived payloads.

Design: `docs/plans/small-model-benchmarking.md` §3.4.4a (capture order, `callSurface`) and §3.6
(the four operations, the unit boundary, the eligibility gate); `-ml` is
`docs/plans/small-model-benchmarking-ml.md`.

Every fixture under `tests/fixtures/lmstudio/` carries a `_provenance` key naming exactly where
each field came from — a live capture cited by line number, an established precedent elsewhere in
this repo's own fixtures (`tests/conftest.py`), or a plainly-labelled placeholder/synthetic value.
No literal 19-entry `GET /api/v0/models` capture exists anywhere in this repo's documentation
(checked: plan §2.5, review Pass 1 Appendix A.2, review Pass 4 Appendix D.3 — all narrative, not a
saved payload), so `catalog.json` holds every entry the docs record a field for rather than a
padded, partly-invented 19.

This suite never opens a real socket: `LMStudio`'s `opener` constructor parameter is injected with
a fake, the same pattern `falkorchat/transport.py` uses for the same reason.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from modelbench.lmstudio import (
    ChatResult,
    LMStudio,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    LMStudioUnreachable,
    ToolCallingIneligible,
    check_tool_calling_eligibility,
    tool_calling_eligible,
)

FIXTURES = Path(__file__).parent / "fixtures" / "lmstudio"


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj).encode("utf-8")


class _FakeResponse:
    """Just enough of `http.client.HTTPResponse` for `_raw_get`/`_raw_post` to use."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        pass


def make_opener(routes: dict[str, tuple[int, bytes] | Exception]) -> Callable[..., Any]:
    """Build a fake `urlopen` replacement. `routes` maps a URL *suffix* (e.g. `/api/v0/models`)
    to either `(status, body_bytes)` or an `Exception` instance to raise. A status >= 400 is
    raised as `urllib.error.HTTPError`, exactly as the real `urlopen` would."""

    def opener(req: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        url = req.full_url
        for suffix, outcome in routes.items():
            if url.endswith(suffix):
                if isinstance(outcome, Exception):
                    raise outcome
                status, body = outcome
                if status >= 400:
                    raise urllib.error.HTTPError(url, status, "error", None, io.BytesIO(body))
                return _FakeResponse(status, body)
        raise AssertionError(f"no stubbed route for {url}")

    return opener


def client(routes: dict[str, tuple[int, bytes] | Exception]) -> LMStudio:
    return LMStudio("http://localhost:1234", opener=make_opener(routes))


# --- probe() — §3.4.4a's two-step probe, all three named outcomes -------------------------------


def test_probe_returns_api_v0_when_the_native_catalog_answers():
    catalog_body = _json_bytes({"data": _load("catalog.json")["data"]})
    c = client({"/api/v0/models": (200, catalog_body)})
    assert c.probe() == "api-v0"


def test_probe_returns_v1_only_when_v0_404s_but_v1_answers():
    v1_body = _json_bytes(_load("v1_models_response.json"))
    c = client(
        {
            "/api/v0/models": (404, b"{}"),
            "/v1/models": (200, v1_body),
        }
    )
    assert c.probe() == "v1-only"


def test_probe_returns_unreachable_when_nothing_listens():
    refused = urllib.error.URLError(OSError("Connection refused"))
    c = client(
        {
            "/api/v0/models": refused,
            "/v1/models": refused,
        }
    )
    assert c.probe() == "unreachable"


# --- catalog() ------------------------------------------------------------------------------


def test_catalog_parses_every_fixture_entry_into_model_info():
    catalog_body = _json_bytes({"data": _load("catalog.json")["data"]})
    c = client({"/api/v0/models": (200, catalog_body)})
    infos = c.catalog()
    assert len(infos) == 7
    by_id = {m.id: m for m in infos}
    assert set(by_id) == {
        "text-embedding-qwen3-embedding-0.6b",
        "google/gemma-3-4b",
        "google/gemma-3-12b",
        "text-embedding-nomic-embed-text-v1.5",
        "mistralai/ministral-3-3b",
        "mistralai_ministral-3-3b-instruct-2512",
        "qwen/qwen3-4b-2507",
    }


def test_catalog_keeps_capabilities_absent_distinct_from_present_and_empty():
    # google/gemma-3-4b (review Pass 1 Appendix A.2): no `capabilities` key at all.
    catalog_body = _json_bytes({"data": _load("catalog.json")["data"]})
    c = client({"/api/v0/models": (200, catalog_body)})
    by_id = {m.id: m for m in c.catalog()}
    assert by_id["google/gemma-3-4b"].capabilities is None

    # A synthetic entry with `"capabilities": []` — present, just empty. Not in catalog.json
    # because no documented entry has this shape; this is a code-path test, not a payload claim.
    body = {
        "data": [
            {
                "id": "x",
                "object": "model",
                "type": "llm",
                "publisher": "p",
                "arch": "a",
                "compatibility_type": "gguf",
                "quantization": "Q4",
                "state": "not-loaded",
                "max_context_length": 100,
                "capabilities": [],
            }
        ]
    }
    c2 = client({"/api/v0/models": (200, _json_bytes(body))})
    [info] = c2.catalog()
    assert info.capabilities == ()


def test_catalog_raises_unreachable_when_nothing_listens():
    c = client({"/api/v0/models": urllib.error.URLError(OSError("Connection refused"))})
    with pytest.raises(LMStudioUnreachable):
        c.catalog()


# --- residency() — §3.4.4a: catalog filtered on state != "not-loaded", {id, state} -------------


def test_residency_is_empty_on_an_all_not_loaded_catalog():
    catalog_body = _json_bytes({"data": _load("catalog.json")["data"]})
    c = client({"/api/v0/models": (200, catalog_body)})
    assert c.residency() == []


def test_residency_maps_loaded_entries_to_id_and_literal_state_string():
    catalog_body = _json_bytes({"data": _load("catalog_one_loaded.json")["data"]})
    c = client({"/api/v0/models": (200, catalog_body)})
    resident = c.residency()
    assert len(resident) == 1
    assert resident[0].id == "qwen/qwen3-4b-2507"
    # the literal state string, never booleanised (plan §3.4.4a)
    assert resident[0].state == "loaded"
    assert resident[0].state is not True


# --- The §3.6 eligibility gate, on the three real catalog entries that break the naive rule ----


def _model_info(catalog_file: str, model_id: str):
    data = _load(catalog_file)["data"]
    c = client({"/api/v0/models": (200, _json_bytes({"data": data}))})
    by_id = {m.id: m for m in c.catalog()}
    return by_id[model_id]


def test_gate_refuses_an_embeddings_model_advertising_tool_use_on_a_tool_caller_pack():
    info = _model_info("catalog.json", "text-embedding-qwen3-embedding-0.6b")
    eligible, reason = tool_calling_eligible(info)
    assert eligible is False
    assert "embeddings" in reason
    with pytest.raises(ToolCallingIneligible):
        check_tool_calling_eligibility("tool-caller", info)


def test_gate_admits_an_entry_with_no_capabilities_key_on_a_tool_caller_pack():
    info = _model_info("catalog.json", "google/gemma-3-4b")
    assert info.capabilities is None
    eligible, reason = tool_calling_eligible(info)
    assert eligible is True
    assert reason is None
    check_tool_calling_eligibility("tool-caller", info)  # must not raise


def test_gate_admits_an_llm_with_tool_use_on_a_tool_caller_pack():
    info = _model_info("catalog.json", "qwen/qwen3-4b-2507")
    eligible, reason = tool_calling_eligible(info)
    assert eligible is True
    assert reason is None
    check_tool_calling_eligibility("tool-caller", info)  # must not raise


def test_gate_does_not_run_on_an_embedder_pack_so_the_run_proceeds():
    """v1.11, plan-gate P5-1: the same embeddings-model-advertising-tool_use entry that the
    tool-caller gate refuses must be admitted when the pack's role is not `tool-caller` — the
    gate must not run at all. This is the negative assertion the note says would have caught an
    unscoped gate, because every positive case above passes even with the scope missing."""
    info = _model_info("catalog.json", "text-embedding-qwen3-embedding-0.6b")
    check_tool_calling_eligibility("embedder", info)  # must not raise — the gate never runs


def test_check_tool_calling_eligibility_role_is_a_plain_string_not_a_pack():
    """U71 is building `Pack`/`load_pack`/`validate_pack` concurrently this wave; this unit's
    gate must never import or depend on that shape (neither exists yet). Pinned two ways: the
    parameter's own type annotation, and a source-level check that this module never imports
    `modelbench.packs`."""
    import inspect

    sig = inspect.signature(check_tool_calling_eligibility)
    # `from __future__ import annotations` in lmstudio.py stringifies annotations, so this
    # compares the string form rather than the `str` object.
    assert sig.parameters["role"].annotation == "str"

    import ast

    import modelbench.lmstudio as lmstudio_module

    tree = ast.parse(Path(lmstudio_module.__file__).read_text())
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("packs" in name for name in imported_modules)


# --- ChatResult: the unit boundary (§3.6, plan-gate P4-1/P5-8) ---------------------------------


def _chat(fixture_name: str, *, timeout_s: float = 30.0) -> ChatResult:
    body = _json_bytes(_load(fixture_name))
    c = client({"/api/v0/chat/completions": (200, body)})
    return c.chat(
        [{"role": "user", "content": "hi"}],
        model="qwen/qwen3-4b-2507",
        temperature=0.0,
        max_tokens=10,
        timeout_s=timeout_s,
    )


def test_chat_result_converts_seconds_to_milliseconds():
    """LM Studio's own v0 REST doc example (review Pass 4 Appendix D.2):
    time_to_first_token=0.111s, generation_time=0.954s."""
    result = _chat("chat_response_with_stats.json")
    assert result.ttftMs == pytest.approx(111.0)
    assert result.generationMs == pytest.approx(954.0)
    # the raw seconds values survive verbatim, for auditability, beside the converted ones
    assert result.stats["time_to_first_token"] == 0.111
    assert result.stats["generation_time"] == 0.954


def test_chat_result_tokens_per_second_is_the_one_unconverted_figure():
    result = _chat("chat_response_with_stats.json")
    assert result.tokensPerSecond == 51.43709529007664  # exact — not multiplied by 1000


def test_chat_result_derived_fields_are_none_not_zero_when_stats_is_absent():
    result = _chat("chat_response_no_stats.json")
    assert result.stats is None
    assert result.ttftMs is None
    assert result.generationMs is None
    assert result.tokensPerSecond is None
    assert result.model_info is None
    assert result.runtime is None


def test_chat_result_construction_never_raises_on_a_partial_stats_object():
    """A `stats` object present but missing every timing key — never observed live, but the rule
    (plan-gate P5-8) is unconditional: no exception, and every derived field lands `None`."""
    result = ChatResult(
        message={"role": "assistant", "content": "x"},
        tool_calls=(),
        toolCallForm="prose",
        stats={"stop_reason": "eosFound"},  # no ttft/generation_time/tokens_per_second keys
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=12.0,
    )
    assert result.ttftMs is None
    assert result.generationMs is None
    assert result.tokensPerSecond is None
    assert result.stats == {"stop_reason": "eosFound"}  # kept verbatim


def test_chat_result_tool_call_form_is_native_when_tool_calls_present():
    result = _chat("chat_response_native_tool_call.json")
    assert result.toolCallForm == "native"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "lookup_product_fact"


def test_chat_result_tool_call_form_is_prose_when_no_native_tool_calls():
    result = _chat("chat_response_with_stats.json")
    assert result.toolCallForm == "prose"
    assert result.tool_calls == ()


# --- timeout_s: required, no default, on chat/embed/warm_up (§3.6's two budgets) ---------------


def test_chat_requires_timeout_s_with_no_default():
    c = client({})
    with pytest.raises(TypeError):
        c.chat([{"role": "user", "content": "hi"}], model="m", temperature=0.0, max_tokens=10)


def test_embed_requires_timeout_s_with_no_default():
    c = client({})
    with pytest.raises(TypeError):
        c.embed(["hello"], model="m")


def test_warm_up_requires_timeout_s_with_no_default():
    catalog_body = _json_bytes({"data": _load("catalog.json")["data"]})
    c = client({"/api/v0/models": (200, catalog_body)})
    with pytest.raises(TypeError):
        c.warm_up("m", call_surface="chat", system_prompt=None)


# --- transport-boundary failure classification: timeout vs. everything else --------------------


def test_chat_call_that_times_out_raises_lmstudio_call_timeout():
    c = client({"/api/v0/chat/completions": TimeoutError("timed out")})
    with pytest.raises(LMStudioCallTimeout):
        c.chat(
            [{"role": "user", "content": "hi"}],
            model="m",
            temperature=0.0,
            max_tokens=10,
            timeout_s=0.01,
        )


def test_chat_call_with_dropped_connection_raises_lmstudio_call_failed_not_timeout():
    c = client(
        {"/api/v0/chat/completions": urllib.error.URLError(OSError("Connection reset by peer"))}
    )
    with pytest.raises(LMStudioCallFailed):
        c.chat(
            [{"role": "user", "content": "hi"}],
            model="m",
            temperature=0.0,
            max_tokens=10,
            timeout_s=30.0,
        )


def test_chat_call_with_http_error_raises_lmstudio_call_failed_not_timeout():
    c = client({"/api/v0/chat/completions": (500, b"internal error")})
    with pytest.raises(LMStudioCallFailed):
        c.chat(
            [{"role": "user", "content": "hi"}],
            model="m",
            temperature=0.0,
            max_tokens=10,
            timeout_s=30.0,
        )


def test_embed_call_that_times_out_raises_lmstudio_call_timeout():
    c = client({"/api/v0/embeddings": TimeoutError("timed out")})
    with pytest.raises(LMStudioCallTimeout):
        c.embed(["hi"], model="m", timeout_s=0.01)


# --- embed() ----------------------------------------------------------------------------------


def test_embed_records_dimension_from_the_first_vector():
    body = _json_bytes(_load("embed_response.json"))
    c = client({"/api/v0/embeddings": (200, body)})
    result = c.embed(["a", "b"], model="text-embedding-qwen3-embedding-0.6b", timeout_s=10.0)
    assert result.dimension == 4
    assert len(result.vectors) == 2
    assert result.vectors[0] == (0.1, 0.2, 0.3, 0.4)


def test_embed_dimension_is_none_not_zero_on_an_empty_batch():
    body = _json_bytes({"object": "list", "model": "m", "data": [], "usage": {}})
    c = client({"/api/v0/embeddings": (200, body)})
    result = c.embed([], model="m", timeout_s=10.0)
    assert result.vectors == ()
    assert result.dimension is None


# --- warm_up() — §3.6: call-surface aware, content discarded, metadata kept --------------------


def test_warm_up_on_chat_surface_reads_runtime_and_stats_from_the_response():
    chat_body = _json_bytes(_load("chat_response_with_stats.json"))
    catalog_body = _json_bytes({"data": _load("catalog.json")["data"]})  # all not-loaded
    c = client(
        {
            "/api/v0/models": (200, catalog_body),
            "/api/v0/chat/completions": (200, chat_body),
        }
    )
    result = c.warm_up(
        "qwen/qwen3-4b-2507", call_surface="chat", system_prompt="be terse", timeout_s=300.0
    )
    assert result.wasResidentBefore is False  # cold: catalog.json has it not-loaded
    assert result.runtime == {
        "name": "llama.cpp",
        "version": "1.52.0",
        "supported_formats": ["gguf"],
    }
    assert result.stats["time_to_first_token"] == 0.111
    assert result.wallClockMs is not None


def test_warm_up_was_resident_before_is_true_when_the_model_is_already_loaded():
    catalog_body = _json_bytes({"data": _load("catalog_one_loaded.json")["data"]})
    chat_body = _json_bytes(_load("chat_response_with_stats.json"))
    c = client(
        {
            "/api/v0/models": (200, catalog_body),
            "/api/v0/chat/completions": (200, chat_body),
        }
    )
    result = c.warm_up(
        "qwen/qwen3-4b-2507", call_surface="chat", system_prompt=None, timeout_s=300.0
    )
    assert result.wasResidentBefore is True


def test_warm_up_checks_residency_before_issuing_the_call_not_after():
    """§3.6/§3.4.4a: `wasResidentBefore` must come from a residency probe taken *before* the
    warm-up's own request fires, because under JIT the warm-up **is** the load — probing after
    would see the model it had just loaded and misreport every cold warm-up as already resident.

    The stub's `/api/v0/models` route answers 'not-loaded' *until* the chat call has actually
    happened, and 'loaded' from that point on — keyed on whether the chat call fired, not on how
    many times `/api/v0/models` itself is hit, so this fails if `warm_up` reads residency even
    once *after* issuing the chat request instead of only before it."""
    state = {"chat_happened": False}
    cold_catalog = _json_bytes({"data": _load("catalog.json")["data"]})
    warm_catalog = _json_bytes({"data": _load("catalog_one_loaded.json")["data"]})
    chat_body = _json_bytes(_load("chat_response_with_stats.json"))

    def opener(req, timeout=None):
        url = req.full_url
        if url.endswith("/api/v0/models"):
            return _FakeResponse(200, warm_catalog if state["chat_happened"] else cold_catalog)
        if url.endswith("/api/v0/chat/completions"):
            state["chat_happened"] = True
            return _FakeResponse(200, chat_body)
        raise AssertionError(f"no stubbed route for {url}")

    c = LMStudio("http://localhost:1234", opener=opener)
    result = c.warm_up(
        "qwen/qwen3-4b-2507", call_surface="chat", system_prompt=None, timeout_s=300.0
    )
    assert result.wasResidentBefore is False


def test_warm_up_on_embeddings_surface_has_no_runtime_or_stats():
    catalog_body = _json_bytes({"data": _load("catalog.json")["data"]})
    embed_body = _json_bytes(_load("embed_response.json"))
    c = client(
        {
            "/api/v0/models": (200, catalog_body),
            "/api/v0/embeddings": (200, embed_body),
        }
    )
    result = c.warm_up(
        "text-embedding-qwen3-embedding-0.6b",
        call_surface="embeddings",
        system_prompt=None,
        timeout_s=300.0,
    )
    assert result.runtime is None
    assert result.stats is None
    assert result.wallClockMs is not None


# --- one -m live test, per §4 S2's done-condition. Written and NEVER run by this suite ---------
# (`pyproject.toml`'s `addopts = '-ra -m "not live"'` deselects it by default; it needs a model
# actually loaded in LM Studio, and agents are not authorised to load one.)


@pytest.mark.live
def test_live_catalog_and_chat_stats_against_a_real_lm_studio():
    c = LMStudio("http://localhost:1234")
    models = c.catalog()
    assert len(models) > 0
    model_id = models[0].id
    result = c.chat(
        [{"role": "user", "content": "Say hello in one word."}],
        model=model_id,
        temperature=0.0,
        max_tokens=10,
        timeout_s=300.0,
    )
    assert result.stats is not None
    assert "time_to_first_token" in result.stats
