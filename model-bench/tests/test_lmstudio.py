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

import http.client
import io
import json
import time
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


class _ReadFailsResponse:
    """A response that is obtained successfully (a status exists) but whose `.read()` raises —
    the body-read-phase failure class Pass 12 P12-1 found escaping the adapter's taxonomy
    entirely, because `resp.read()` used to sit outside `_raw_get`/`_raw_post`'s try/except
    ladder. An opener-level `Exception` (the `routes` dict's other failure shape) cannot express
    this: it fails before a response object ever exists, which is a different phase."""

    def __init__(self, status: int, exc: Exception) -> None:
        self.status = status
        self._exc = exc

    def read(self) -> bytes:
        raise self._exc

    def close(self) -> None:
        pass


class _SlowReadResponse:
    """A response whose `.read()` takes measurable wall-clock time — pins Pass 12 P12-4:
    `wallClockMs` must be measured to the last byte of the body, not to the response headers."""

    def __init__(self, status: int, body: bytes, delay_s: float) -> None:
        self.status = status
        self._body = body
        self._delay_s = delay_s

    def read(self) -> bytes:
        time.sleep(self._delay_s)
        return self._body

    def close(self) -> None:
        pass


def make_opener(routes: dict[str, tuple[int, bytes] | Exception | Any]) -> Callable[..., Any]:
    """Build a fake `urlopen` replacement. `routes` maps a URL *suffix* (e.g. `/api/v0/models`)
    to one of: `(status, body_bytes)` (a status >= 400 is raised as `urllib.error.HTTPError`,
    exactly as the real `urlopen` would), an `Exception` instance to raise from the opener call
    itself (a connect-phase failure), or an already-constructed response-like object — anything
    with `.read()`/`.close()`, e.g. `_ReadFailsResponse`/`_SlowReadResponse` — returned verbatim,
    for a failure or a delay that only happens once the caller reaches `.read()`."""

    def opener(req: urllib.request.Request, timeout: float | None = None) -> Any:
        url = req.full_url
        for suffix, outcome in routes.items():
            if url.endswith(suffix):
                if isinstance(outcome, Exception):
                    raise outcome
                if isinstance(outcome, tuple):
                    status, body = outcome
                    if status >= 400:
                        raise urllib.error.HTTPError(url, status, "error", None, io.BytesIO(body))
                    return _FakeResponse(status, body)
                return outcome  # an already-constructed response-like object
        raise AssertionError(f"no stubbed route for {url}")

    return opener


def client(routes: dict[str, tuple[int, bytes] | Exception | Any]) -> LMStudio:
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


def test_catalog_reports_http_status_even_when_the_error_body_is_not_json():
    """Pass 12 P12-10: `catalog()` used to parse the body before checking the status, so a
    non-2xx response with a non-JSON body (an HTML error page, a proxy's plain-text error)
    reported a "not valid JSON" cause instead of the real HTTP status."""
    c = client({"/api/v0/models": (500, b"<html>Internal Server Error</html>")})
    with pytest.raises(LMStudioCallFailed, match="HTTP 500"):
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
    c = client({})
    with pytest.raises(TypeError):
        c.warm_up("m", call_surface="chat", system_prompt=None, was_resident_before=False)


def test_warm_up_requires_was_resident_before_with_no_default():
    """§3.6: `coldLoadSeconds` is recorded only when the model was not resident at start, so a
    default here would silently pick a disposition rather than refuse to guess one (AGENTS.md's
    "nothing that shapes a decision carries a default")."""
    c = client({})
    with pytest.raises(TypeError):
        c.warm_up("m", call_surface="chat", system_prompt=None, timeout_s=300.0)


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


# --- wall-clock window (§3.6 FR-11, Pass 12 P12-4) ---------------------------------------------


def test_chat_wall_clock_is_measured_to_the_last_byte_of_the_body_not_the_headers():
    """§3.6 FR-11: `wallClockMs` is "measured around the HTTP call from just before the request
    to the last byte of the body." Pass 12 P12-4: it used to stop the clock before `resp.read()`,
    so a slow body read off a fast-responding server was invisible — a response that spent 250ms
    in `read()` reported `wallClockMs = 0.001`."""
    slow = _SlowReadResponse(200, _json_bytes(_load("chat_response_with_stats.json")), 0.2)
    c = client({"/api/v0/chat/completions": slow})
    result = c.chat(
        [{"role": "user", "content": "hi"}],
        model="m",
        temperature=0.0,
        max_tokens=10,
        timeout_s=5.0,
    )
    assert result.wallClockMs >= 150.0  # 200ms sleep in read(), generous margin for jitter


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
#
# Pass 12 P12-5: `was_resident_before` is supplied by the caller (from `residentModelsAtStart`,
# §3.4.4a capture-order step 3) rather than re-probed inside `warm_up` — the method no longer
# calls `residency()`/`catalog()` itself, so none of these routes stub `/api/v0/models`.


def test_warm_up_on_chat_surface_reads_runtime_and_stats_from_the_response():
    chat_body = _json_bytes(_load("chat_response_with_stats.json"))
    c = client({"/api/v0/chat/completions": (200, chat_body)})
    result = c.warm_up(
        "qwen/qwen3-4b-2507",
        call_surface="chat",
        system_prompt="be terse",
        was_resident_before=False,
        timeout_s=300.0,
    )
    assert result.wasResidentBefore is False
    assert result.runtime == {
        "name": "llama.cpp",
        "version": "1.52.0",
        "supported_formats": ["gguf"],
    }
    assert result.stats["time_to_first_token"] == 0.111
    assert result.wallClockMs is not None


@pytest.mark.parametrize("call_surface", ["chat", "embeddings"])
def test_warm_up_passes_was_resident_before_through_verbatim(call_surface):
    """A plain contract test: `LoadResult.wasResidentBefore` is exactly the caller's value,
    neither inverted nor ignored — on **both** call surfaces. Parametrized deliberately: a first
    version of this test covered only the chat branch, and a mutation inverting the value on the
    embeddings branch alone (`warm_up`'s second `return LoadResult(...)`) passed the full suite
    unnoticed — the class this coordination has already hit five times, caught here by mutation
    before review rather than after."""
    if call_surface == "chat":
        route = "/api/v0/chat/completions"
        body = _json_bytes(_load("chat_response_with_stats.json"))
    else:
        route = "/api/v0/embeddings"
        body = _json_bytes(_load("embed_response.json"))
    c = client({route: (200, body)})
    result = c.warm_up(
        "qwen/qwen3-4b-2507",
        call_surface=call_surface,
        system_prompt=None,
        was_resident_before=True,
        timeout_s=300.0,
    )
    assert result.wasResidentBefore is True


def test_warm_up_never_probes_residency_itself():
    """Pass 12 P12-5: an earlier version of `warm_up` called `residency()` (hence `catalog()`,
    hence `GET /api/v0/models`) immediately before the timed call, on a false justification —
    `residentModelsAtStart` is `[]` on a cold run, and `model in set()` is correctly `False`
    ("not resident"), not a misreport, so there was nothing wrong for the extra probe to fix.
    No `/api/v0/models` route is stubbed here, so this reddens with 'no stubbed route' if
    `warm_up` ever re-probes residency itself again."""
    chat_body = _json_bytes(_load("chat_response_with_stats.json"))
    c = client({"/api/v0/chat/completions": (200, chat_body)})
    c.warm_up(
        "qwen/qwen3-4b-2507",
        call_surface="chat",
        system_prompt=None,
        was_resident_before=False,
        timeout_s=300.0,
    )


def test_warm_up_on_embeddings_surface_has_no_runtime_or_stats():
    embed_body = _json_bytes(_load("embed_response.json"))
    c = client({"/api/v0/embeddings": (200, embed_body)})
    result = c.warm_up(
        "text-embedding-qwen3-embedding-0.6b",
        call_surface="embeddings",
        system_prompt=None,
        was_resident_before=False,
        timeout_s=300.0,
    )
    assert result.runtime is None
    assert result.stats is None
    assert result.wallClockMs is not None


# --- §4B coverage probe (review Pass 12, `docs/reviews/small-model-benchmarking-impl.md`) ------
#
# A probe over (operation) x (failure phase) x (failure kind): the six public operations
# {catalog, residency, probe, chat, embed, warm_up} x {connect/headers, body-read} x {timeout,
# non-2xx, connection drop, unparseable body}. Every cell must land in exactly one of
# `LMStudioCallTimeout` / `LMStudioCallFailed` / `LMStudioUnreachable`, or — for `probe()` — one
# of its three literal outcomes. No cell may raise anything outside `LMStudioError`. The
# reviewer ran the body-read row and found 9 of 9 cells escaping (three exception kinds x
# chat/catalog/probe); this probe is the regression net, over every cell rather than the three
# sampled by hand.
#
# Two of the eight (phase, kind) combinations are not reachable through `urllib.request`'s own
# contract — not a judgement call to make silently, so they are named constants asserted below
# rather than simply absent from the parametrized cases:
#   - (connect, unparseable_body): no body exists to be unparseable before a response object —
#     with a status — has even been obtained.
#   - (read, non_2xx): `urlopen()` raises `HTTPError` for any status >= 400 *before* ever handing
#     back a response object (see `make_opener`), so a non-2xx status cannot be observed once
#     `.read()` is reachable — it is a connect/headers-phase fact by construction.

_CONNECT_KINDS: dict[str, Callable[[], Any]] = {
    "timeout": lambda: TimeoutError("timed out"),
    "non_2xx": lambda: (500, b"upstream error"),
    "connection_drop": lambda: urllib.error.URLError(ConnectionResetError("reset by peer")),
}
_READ_KINDS: dict[str, Callable[[], Any]] = {
    "timeout": lambda: _ReadFailsResponse(200, TimeoutError("timed out mid-body")),
    "connection_drop": lambda: _ReadFailsResponse(
        200, http.client.IncompleteRead(b"partial")
    ),
    "unparseable_body": lambda: (200, b"not json{"),
}

# The six reachable cells and the exception each must raise, for a GET-based operation
# (catalog/residency) and a POST-based one (chat/embed/warm_up) respectively. Both taxonomies
# cover the same six cells — the exemption below is structural, not per-taxonomy.
_EXPECTED_FOR_GET: dict[tuple[str, str], type[Exception]] = {
    ("connect", "timeout"): LMStudioUnreachable,
    ("connect", "non_2xx"): LMStudioCallFailed,
    ("connect", "connection_drop"): LMStudioUnreachable,
    ("read", "timeout"): LMStudioUnreachable,
    ("read", "connection_drop"): LMStudioUnreachable,
    ("read", "unparseable_body"): LMStudioCallFailed,
}
_EXPECTED_FOR_POST: dict[tuple[str, str], type[Exception]] = {
    ("connect", "timeout"): LMStudioCallTimeout,
    ("connect", "non_2xx"): LMStudioCallFailed,
    ("connect", "connection_drop"): LMStudioCallFailed,
    ("read", "timeout"): LMStudioCallTimeout,
    ("read", "connection_drop"): LMStudioCallFailed,
    ("read", "unparseable_body"): LMStudioCallFailed,
}

_ALL_KINDS = ("timeout", "non_2xx", "connection_drop", "unparseable_body")

_EXEMPT_CELLS = frozenset({("connect", "unparseable_body"), ("read", "non_2xx")})


def _route_outcome(phase: str, kind: str) -> Any:
    return (_CONNECT_KINDS if phase == "connect" else _READ_KINDS)[kind]()


def test_probe_cell_exemptions_are_exactly_the_structurally_unreachable_ones():
    """The guard against the exemption silently growing (or shrinking) without anyone deciding
    it: re-derive the full 8-cells-per-operation grid from the four failure kinds §4B names, and
    assert the two cells this module does not exercise are exactly, and only, `_EXEMPT_CELLS` —
    not implied by their absence from `_EXPECTED_FOR_GET`/`_POST`."""
    all_cells = {(phase, kind) for phase in ("connect", "read") for kind in _ALL_KINDS}
    exercised = set(_EXPECTED_FOR_GET)
    assert exercised == set(_EXPECTED_FOR_POST)  # both taxonomies cover the same six cells
    assert all_cells - exercised == _EXEMPT_CELLS


@pytest.mark.parametrize("phase,kind", sorted(_EXPECTED_FOR_GET))
def test_catalog_lands_in_the_taxonomy_on_every_reachable_cell(phase, kind):
    c = client({"/api/v0/models": _route_outcome(phase, kind)})
    with pytest.raises(_EXPECTED_FOR_GET[(phase, kind)]):
        c.catalog()


@pytest.mark.parametrize("phase,kind", sorted(_EXPECTED_FOR_GET))
def test_residency_lands_in_the_taxonomy_on_every_reachable_cell(phase, kind):
    c = client({"/api/v0/models": _route_outcome(phase, kind)})
    with pytest.raises(_EXPECTED_FOR_GET[(phase, kind)]):
        c.residency()


@pytest.mark.parametrize("phase,kind", sorted(_EXPECTED_FOR_GET))
def test_probe_never_raises_and_returns_a_literal_outcome_on_every_reachable_cell(phase, kind):
    c = client(
        {
            "/api/v0/models": _route_outcome(phase, kind),
            "/v1/models": urllib.error.URLError(OSError("connection refused")),
        }
    )
    assert c.probe() == "unreachable"  # no exception, and it names a real literal outcome


@pytest.mark.parametrize("phase,kind", sorted(_EXPECTED_FOR_POST))
def test_chat_lands_in_the_taxonomy_on_every_reachable_cell(phase, kind):
    c = client({"/api/v0/chat/completions": _route_outcome(phase, kind)})
    with pytest.raises(_EXPECTED_FOR_POST[(phase, kind)]):
        c.chat(
            [{"role": "user", "content": "hi"}],
            model="m",
            temperature=0.0,
            max_tokens=10,
            timeout_s=5.0,
        )


@pytest.mark.parametrize("phase,kind", sorted(_EXPECTED_FOR_POST))
def test_embed_lands_in_the_taxonomy_on_every_reachable_cell(phase, kind):
    c = client({"/api/v0/embeddings": _route_outcome(phase, kind)})
    with pytest.raises(_EXPECTED_FOR_POST[(phase, kind)]):
        c.embed(["hi"], model="m", timeout_s=5.0)


@pytest.mark.parametrize("phase,kind", sorted(_EXPECTED_FOR_POST))
def test_warm_up_chat_surface_lands_in_the_taxonomy_on_every_reachable_cell(phase, kind):
    c = client({"/api/v0/chat/completions": _route_outcome(phase, kind)})
    with pytest.raises(_EXPECTED_FOR_POST[(phase, kind)]):
        c.warm_up(
            "m",
            call_surface="chat",
            system_prompt=None,
            was_resident_before=False,
            timeout_s=5.0,
        )


@pytest.mark.parametrize("phase,kind", sorted(_EXPECTED_FOR_POST))
def test_warm_up_embeddings_surface_lands_in_the_taxonomy_on_every_reachable_cell(phase, kind):
    c = client({"/api/v0/embeddings": _route_outcome(phase, kind)})
    with pytest.raises(_EXPECTED_FOR_POST[(phase, kind)]):
        c.warm_up(
            "m",
            call_surface="embeddings",
            system_prompt=None,
            was_resident_before=False,
            timeout_s=5.0,
        )


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
