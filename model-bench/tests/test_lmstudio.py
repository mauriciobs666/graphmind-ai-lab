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

import ast
import http.client
import io
import json
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from modelbench import lmstudio
from modelbench.lmstudio import (
    _REQUIRED_MODEL_INFO_KEYS,
    ChatResult,
    LMStudio,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    LMStudioUnreachable,
    ToolCallingIneligible,
    _model_info_from_raw,
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


class _RaisingFp:
    """A minimal file-like object whose `.read()` raises — used as `HTTPError`'s own `fp`, to
    express the failure phase Pass 13 found the grid could not: reading the *error* response's
    body (`exc.read()`, inside `except urllib.error.HTTPError`), as opposed to
    `_ReadFailsResponse` above, which fails reading a *success* response's body. `HTTPError` is
    itself a response object — a status plus a `.read()` — and `probe()`'s own `v1-only`
    diagnosis calls `exc.read()` on every 404 a non-LM-Studio server returns, so this is
    normal-path code, not an edge case (review Pass 13, P13-1)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def read(self, *args: Any, **kwargs: Any) -> bytes:
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
    itself — a connect-phase failure, *or* a pre-built `urllib.error.HTTPError` (itself an
    `Exception`) raised exactly as real `urlopen` raises one for any non-2xx status, letting its
    own `fp`/`.read()` behave arbitrarily (`_RaisingFp`, above — Pass 13 P13-1's error-body
    phase) — or an already-constructed response-like object — anything with `.read()`/`.close()`,
    e.g. `_ReadFailsResponse`/`_SlowReadResponse` — returned verbatim, for a failure or a delay
    that only happens once the caller reaches `.read()`."""

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


# A full, valid `/api/v0/models` entry — every field `_model_info_from_raw` reads, required and
# optional alike (mirrors the synthetic entry in the capabilities test above).
_FULL_MODEL_INFO_RAW: dict[str, Any] = {
    "id": "x",
    "object": "model",
    "type": "llm",
    "publisher": "p",
    "arch": "a",
    "compatibility_type": "gguf",
    "quantization": "Q4",
    "state": "not-loaded",
    "max_context_length": 100,
    "capabilities": ["tool_use"],
    "loaded_context_length": 50,
}


# The two fields `_model_info_from_raw` reads with `.get(...)` rather than `raw[...]` — named
# directly, independently of `_REQUIRED_MODEL_INFO_KEYS`, so a widen that marks one of them
# required *in the constant* still has something outside the constant to disagree with (review
# Pass 15 §4: the missing-key check is a pure membership test, so "does the constant's own
# member get rejected when absent" is tautological — true for any member — and cannot alone
# catch a widen onto an already-optional field; P14-1's probe below only catches the file's
# *shrink* mutation because a second, independent mechanism, the bare `raw["quantization"]`
# access, diverges from the missing-key check under that specific mutation).
_OPTIONAL_MODEL_INFO_KEYS: frozenset[str] = frozenset({"capabilities", "loaded_context_length"})


def test_required_model_info_keys_partitions_every_catalog_field():
    """Form (i) (review Pass 15 §4): binds `_REQUIRED_MODEL_INFO_KEYS` to an independent
    declaration of the same partition — the full field set named in `_FULL_MODEL_INFO_RAW` minus
    the two named above as optional — rather than to anything computed from the constant itself.
    A widen that marks `"capabilities"` or `"loaded_context_length"` required, or a shrink that
    drops any of the nine, disagrees with this fixed right-hand side either way."""
    assert set(_REQUIRED_MODEL_INFO_KEYS) == set(_FULL_MODEL_INFO_RAW) - _OPTIONAL_MODEL_INFO_KEYS


def test_required_model_info_keys_is_exactly_what_model_info_from_raw_rejects():
    """Form (ii) (review Pass 15 §4): a *behavioural* consequence per member, complementing the
    declarative pin above rather than duplicating it — this one drives `_model_info_from_raw`
    itself and would catch, for instance, a typo'd `in`/`not in` in the missing-key check that
    the declaration-only pin above cannot see.

    Review Pass 14 P14-1: `_REQUIRED_MODEL_INFO_KEYS` is a hand-maintained list, kept in sync by
    hand with the `raw[key]` accesses `_model_info_from_raw` makes to build `ModelInfo`. Dropping
    `"quantization"` from the constant leaves the missing-key check blind to it: a catalog entry
    genuinely missing that field no longer gets the named `LMStudioCallFailed` — it raises a bare
    `KeyError` out of `catalog()` instead, an untyped escape from this adapter's declared
    exception taxonomy (`LMStudioError` and its three subclasses), the same defect class as
    P13-1 one layer down.

    Driving every field of a full, valid entry, deleted one at a time, is what catches that: a
    key whose absence raises anything other than `LMStudioCallFailed` (a bare `KeyError`,
    notably) is **not** caught here and so is **not** counted as rejected — the `except` clause
    names the typed refusal specifically, never a bare `except Exception`, which is what makes
    the equality below fail loudly on that drift rather than silently matching a shrunken
    constant (verified: an untyped `KeyError` propagates out of this test rather than being
    absorbed as *some* refusal)."""
    rejected: set[str] = set()
    for key in _FULL_MODEL_INFO_RAW:
        mutated = dict(_FULL_MODEL_INFO_RAW)
        del mutated[key]
        try:
            _model_info_from_raw(mutated, index=0)
        except LMStudioCallFailed:
            rejected.add(key)
    assert rejected == set(_REQUIRED_MODEL_INFO_KEYS)


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


def test_chat_result_construction_never_raises_when_stats_is_not_a_mapping():
    """P12-11: the class docstring's "never raises" promise is unconditional, but the only thing
    that held it was `chat()`'s own `isinstance(..., Mapping)` guard one level up — a direct
    construction with a malformed `stats` (here, a list) bypasses that guard entirely and raised
    `AttributeError` from `stats.get(...)`. Closure (a): the mechanism is widened so the promise is
    true of the class itself, not just of its one caller. `stats` itself is kept verbatim (the raw
    payload is retained for auditability regardless of shape); only the three fields *derived from*
    it degrade to `None` when there is nothing `Mapping`-shaped to derive them from."""
    result = ChatResult(
        message={"role": "assistant", "content": "x"},
        tool_calls=(),
        toolCallForm="prose",
        stats=[1],  # malformed: not a Mapping
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=12.0,
    )
    assert result.ttftMs is None
    assert result.generationMs is None
    assert result.tokensPerSecond is None
    assert result.stats == [1]  # kept verbatim, not coerced


def test_chat_result_tokens_per_second_coerces_a_numeric_string_source_to_float():
    """P12-11's other half: `tokensPerSecond` used to be `stats.get("tokens_per_second")`
    verbatim, so a string-valued source (e.g. from a hand-built or malformed payload) survived
    into a field typed `float | None` as a `str`. Closure decision: coerce, the same tolerance
    `ttftMs`/`generationMs` already apply via `_seconds_to_ms`'s own `float(...)` conversion — not
    reject and not pass through untyped."""
    result = ChatResult(
        message={"role": "assistant", "content": "x"},
        tool_calls=(),
        toolCallForm="prose",
        stats={"tokens_per_second": "51.4"},
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=12.0,
    )
    assert result.tokensPerSecond == 51.4
    assert isinstance(result.tokensPerSecond, float)


def test_chat_result_tokens_per_second_is_none_when_source_is_not_numeric():
    """The other side of the same coercion: a source that cannot be read as a number at all (not
    just a numeric string) yields `None`, mirroring `_seconds_to_ms`'s own bad-type tolerance,
    rather than raising or passing the un-coercible value through."""
    result = ChatResult(
        message={"role": "assistant", "content": "x"},
        tool_calls=(),
        toolCallForm="prose",
        stats={"tokens_per_second": "fast"},
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=12.0,
    )
    assert result.tokensPerSecond is None


def test_chat_result_derived_fields_are_none_not_a_number_when_source_is_non_finite():
    """P13-3: `float()` accepts `NaN`/`Infinity` without error, so `_as_float`/`_seconds_to_ms`
    used to let a non-finite source land as an actual `nan`/`inf` *float* rather than degrading to
    `None` — and this needs no malformed transport to reach: `json.loads` parses the bare tokens
    `NaN`/`Infinity` by default, so a server serialising a 0/0 rate is enough (Pass 13, P13-3,
    Appendix M.3). Downstream this is not cosmetic: `statistics.median` over a list containing
    `nan` returns an arbitrary element with no error, and `-ml` §11.5.1's gap
    (`latencyMs - (ttftMs + generationMs)`) goes `-inf`, so the in-call reload detector can never
    fire on that item."""
    result = ChatResult(
        message={"role": "assistant", "content": "x"},
        tool_calls=(),
        toolCallForm="prose",
        stats={
            "time_to_first_token": float("nan"),
            "generation_time": float("inf"),
            "tokens_per_second": float("nan"),
        },
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=12.0,
    )
    assert result.ttftMs is None
    assert result.generationMs is None
    assert result.tokensPerSecond is None


def test_chat_result_derived_fields_are_none_when_a_bare_nan_survives_json_loads_through_chat():
    """The same boundary, reached the way it happens live: `json.loads` parses a body carrying the
    bare tokens `NaN`/`Infinity` without raising (Python's default `parse_constant`), so the
    non-finite value must be rejected at the coercion boundary (`ChatResult.__post_init__`), not
    at parse time — `_parse_json` is deliberately left alone; the raw `stats` mapping still carries
    the non-finite value verbatim, for auditability."""
    body = json.dumps(
        {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}}],
            "stats": {
                "time_to_first_token": float("nan"),
                "generation_time": float("inf"),
                "tokens_per_second": float("nan"),
            },
        }
    ).encode("utf-8")
    c = client({"/api/v0/chat/completions": (200, body)})
    result = c.chat(
        [{"role": "user", "content": "hi"}],
        model="m",
        temperature=0.0,
        max_tokens=10,
        timeout_s=5.0,
    )
    assert result.ttftMs is None
    assert result.generationMs is None
    assert result.tokensPerSecond is None
    import math

    assert math.isnan(result.stats["time_to_first_token"])  # kept verbatim, for auditability


def test_chat_result_derived_fields_are_none_not_1000_or_0_when_source_is_a_bool():
    """`float(True) == 1.0`, so an unguarded coercion turns `True` into `ttftMs=1000.0` and
    `tokensPerSecond=1.0` (Pass 13, P13-3, Appendix M.3). This component excludes `bool` from
    numeric coercion deliberately elsewhere — `packs._row_count_identity_field_valid` and
    `results`' bool guard added at Pass 2 P2-2 — and the unit boundary now does the same."""
    result = ChatResult(
        message={"role": "assistant", "content": "x"},
        tool_calls=(),
        toolCallForm="prose",
        stats={
            "time_to_first_token": True,
            "generation_time": False,
            "tokens_per_second": True,
        },
        model_info=None,
        runtime=None,
        usage=None,
        wallClockMs=12.0,
    )
    assert result.ttftMs is None
    assert result.generationMs is None
    assert result.tokensPerSecond is None


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


# --- §4B coverage probe (review Pass 12, `docs/reviews/small-model-benchmarking-impl.md`;
#     extended at Pass 13, P13-1, to the phase the original grid could not express) -------------
#
# A probe over (operation) x (failure phase) x (failure kind): the six public operations
# {catalog, residency, probe, chat, embed, warm_up} x {connect/headers, success-body-read,
# error-body-read} x {timeout, non-2xx, connection drop, unparseable body}. Every cell must land
# in exactly one of `LMStudioCallTimeout` / `LMStudioCallFailed` / `LMStudioUnreachable`, or — for
# `probe()` — one of its three literal outcomes. No cell may raise anything outside
# `LMStudioError`. Pass 12's reviewer ran the success-body-read row by hand and found 9 of 9
# cells escaping; Pass 13's reviewer then ran the error-body-read row and found 21 of 21
# escaping, because `_EXEMPT_CELLS` declared `("read", "non_2xx")` structurally unreachable on
# the grounds that `HTTPError` is raised *before* a response object exists — which is false:
# `HTTPError` **is** a response object, with its own status and `.read()`, and reading *it* can
# fail exactly like reading a normal response can (`probe()`'s own `v1-only` diagnosis calls
# `exc.read()` on every 404 a non-LM-Studio server returns). This probe is the regression net
# over every cell the grid can name, not the ones a reviewer happened to sample by hand.
#
# Each phase has its own domain of applicable kinds — not every kind applies to every phase, and
# that is a structural fact this module asserts rather than silently encodes by omission:
#   - **connect**: three kinds apply — the opener call itself can time out, return a non-2xx
#     status, or drop the connection. `unparseable_body` is not a fourth cell in this phase's
#     domain at all (Pass 14, P14-6, applying P13-1's own reasoning to its sibling): no body
#     exists to be unparseable before a response object, with a status, has even been obtained —
#     the concept has no referent here, the same shape as `non_2xx` having none in `read` below.
#     It used to be modelled as the older treatment, an exemption (`_EXEMPT_CELLS`) rather than a
#     domain absence; that was inconsistent with how `("read", "non_2xx")` was fixed and is why
#     `_EXEMPT_CELLS` is empty now rather than carrying it.
#   - **read** (a *success* response's body): three kinds — `timeout`, `connection_drop`,
#     `unparseable_body`. `non_2xx` does not apply here at all, and it is not a fourth exempt
#     cell in this phase's domain: `urlopen()` raises `HTTPError` for any status >= 400 *before*
#     ever handing back a response object (`make_opener`), so "the response I'm reading has a
#     non-2xx status" cannot arise while reading a *success* response's body — the concept has no
#     referent in this phase, which is why P13-1 needed a third phase rather than a corrected
#     cell.
#   - **error-body** (an `HTTPError`'s own body, `exc.read()`): one kind — `non_2xx` is the only
#     way to be here at all, since reaching this phase presupposes the non-2xx status that raised
#     the `HTTPError` in the first place.

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
_ERROR_BODY_KINDS: dict[str, Callable[[], Any]] = {
    "non_2xx": lambda: urllib.error.HTTPError(
        "http://localhost:1234/x",
        404,
        "Not Found",
        None,
        _RaisingFp(http.client.IncompleteRead(b"partial")),
    ),
}

# Each phase's domain of applicable kinds (the comment block above states the reasons).
_PHASE_KINDS: dict[str, tuple[str, ...]] = {
    "connect": ("timeout", "non_2xx", "connection_drop"),
    "read": ("timeout", "connection_drop", "unparseable_body"),
    "error-body": ("non_2xx",),
}

# The seven reachable cells and the exception each must raise, for a GET-based operation
# (catalog/residency) and a POST-based one (chat/embed/warm_up) respectively. Both taxonomies
# cover the same seven cells — the exemption below is structural, not per-taxonomy.
#
# `error-body` GET: `_raw_get` folds a failed error-body read to `None` — exactly how it already
# folds a failed *success*-body read (`("read", "timeout")`/`("read", "connection_drop")` below)
# — so both `catalog()`/`residency()` see "no usable response" and raise `LMStudioUnreachable`,
# never a wrong-but-plausible `LMStudioCallFailed` built on a body that was never actually read.
# `error-body` POST: `_raw_post` already had the status in hand (`exc.code`) before attempting
# the read, so it degrades the message (empty body) rather than the exception type — the cell
# stays `LMStudioCallFailed`, identical to every other non-2xx POST cell.
_EXPECTED_FOR_GET: dict[tuple[str, str], type[Exception]] = {
    ("connect", "timeout"): LMStudioUnreachable,
    ("connect", "non_2xx"): LMStudioCallFailed,
    ("connect", "connection_drop"): LMStudioUnreachable,
    ("read", "timeout"): LMStudioUnreachable,
    ("read", "connection_drop"): LMStudioUnreachable,
    ("read", "unparseable_body"): LMStudioCallFailed,
    ("error-body", "non_2xx"): LMStudioUnreachable,
}
_EXPECTED_FOR_POST: dict[tuple[str, str], type[Exception]] = {
    ("connect", "timeout"): LMStudioCallTimeout,
    ("connect", "non_2xx"): LMStudioCallFailed,
    ("connect", "connection_drop"): LMStudioCallFailed,
    ("read", "timeout"): LMStudioCallTimeout,
    ("read", "connection_drop"): LMStudioCallFailed,
    ("read", "unparseable_body"): LMStudioCallFailed,
    ("error-body", "non_2xx"): LMStudioCallFailed,
}

# No cell is exempt (Pass 14, P14-6): `("connect", "unparseable_body")` — the last remaining
# member — held exactly the reasoning that took `("read", "non_2xx")` out of `_PHASE_KINDS["read"]`
# at Pass 13 rather than exempting it (see `_PHASE_KINDS` and the comment block above), and stayed
# in `_PHASE_KINDS["connect"]` under the older treatment only by inconsistency, not by a real
# difference in kind. `_EXEMPT_CELLS` is kept as a named constant, asserted below, rather than
# deleted: a future structurally-unreachable cell is a decision recorded here, not a silent
# absence from the parametrized cases.
_EXEMPT_CELLS: frozenset[tuple[str, str]] = frozenset()

# Pass 14, P14-5: `_PHASE_KINDS` names each phase's domain, but the per-phase outcome-factory
# registries above (`_CONNECT_KINDS`/`_READ_KINDS`/`_ERROR_BODY_KINDS`) are a second, independent
# declaration of the same domains, and nothing bound them to each other — adding a kind to
# `_CONNECT_KINDS` alone (so `_route_outcome` can produce it, but no `_PHASE_KINDS`-derived
# parametrization ever asks for it) left the full suite green, never exercised, never noticed.
_REGISTRY_BY_PHASE: dict[str, dict[str, Callable[[], Any]]] = {
    "connect": _CONNECT_KINDS,
    "read": _READ_KINDS,
    "error-body": _ERROR_BODY_KINDS,
}


def test_outcome_registries_match_their_phase_kinds_domain():
    """The pin: each registry's key set must equal its `_PHASE_KINDS` domain minus that phase's
    exempt kinds — in both directions, so a registry that gains an unexercised kind reddens
    exactly as one that silently loses a kind `_route_outcome` needs already does (via the
    `KeyError` `_route_outcome` raises)."""
    for phase, registry in _REGISTRY_BY_PHASE.items():
        exempt_kinds = {kind for (p, kind) in _EXEMPT_CELLS if p == phase}
        assert set(registry) == set(_PHASE_KINDS[phase]) - exempt_kinds


def _route_outcome(phase: str, kind: str) -> Any:
    kinds = {"connect": _CONNECT_KINDS, "read": _READ_KINDS, "error-body": _ERROR_BODY_KINDS}
    return kinds[phase][kind]()


def test_probe_cell_exemptions_are_exactly_the_structurally_unreachable_ones():
    """The guard against the exemption silently growing (or shrinking) without anyone deciding
    it: re-derive the full per-phase grid from `_PHASE_KINDS` — each phase's *own* domain, not a
    flat cross product that would manufacture cells no phase can express (P13-1's own lesson) —
    and assert the one cell this module does not exercise is exactly, and only, `_EXEMPT_CELLS`,
    not implied by its absence from `_EXPECTED_FOR_GET`/`_POST`.

    Written as a **partition of the whole grid** — the two lines below — and not as
    `all_cells - exercised == _EXEMPT_CELLS` (impl review Pass 16, P16-4). A subtraction
    constrains only the cells the domains declare and never exercise, so a cell exercised here
    that *no* phase domain declares cancels out of the left-hand side and the assertion stays
    green; that direction was caught only incidentally, by the GET/POST symmetry beside it and
    by `_route_outcome`'s `KeyError`, which is coverage by accident.

    **The union alone is not the fix**, and swapping one form for the other is a trade, not a
    repair: `all_cells == exercised | _EXEMPT_CELLS` is satisfied by a cell that is exercised
    *and* exempt at once — an exemption claiming a cell this file demonstrably reaches, which
    the subtraction did refuse (measured: it drops that cell's kill from 2 tests to 1). The
    disjointness line is the half that keeps it, so the assertion holds in every direction:
    nothing undeclared, nothing unaccounted for, nothing both.
    """
    all_cells = {(phase, kind) for phase, kinds in _PHASE_KINDS.items() for kind in kinds}
    exercised = set(_EXPECTED_FOR_GET)
    assert exercised == set(_EXPECTED_FOR_POST)  # both taxonomies cover the same seven cells
    assert all_cells == exercised | _EXEMPT_CELLS
    assert not (exercised & _EXEMPT_CELLS)


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


# --- `LMStudioCallFailed.status` — the partition plan §3.8.4's four-row table keys on ----------
#
# §3.8.4 splits a failed call two ways: `server-rejected` (the server answered and refused — an
# HTTP status is in hand) and `no-response` (the call never completed — a dropped connection, a
# socket error, or a body that could not be read as a response). Before this field both raised a
# bare `LMStudioCallFailed`, so the two were indistinguishable to any caller: an HTTP 400 (`-ml`
# §4.1's own example) and a dropped connection differed only in a message string. `status` is
# what `drive` will partition on — `status is not None` iff `server-rejected` — so it is the
# adapter's job, not a caller's, to get every raise site's side of that partition right.
#
# **A 2xx whose body is unusable carries `status=None`**, deliberately. The server did not
# *refuse*: §3.8.4 files a "body error" under `no-response`, and reporting `200` here would make
# `status is not None` read as a refusal that never happened. `status` is the *refusal* status,
# not "the last status seen".
#
# The grid above pins which exception *class* each transport cell lands in; this table pins the
# `status` of every `raise LMStudioCallFailed` in the module — all twelve, held to that by
# `test_every_raise_site_of_lmstudio_call_failed_is_covered_here` below, which compares the raise
# lines these scenarios actually reach against the raise lines the source actually contains.

_MISSING_KEY_ENTRY: dict[str, Any] = {
    k: v for k, v in _FULL_MODEL_INFO_RAW.items() if k != "arch"
}


def _catalog_returning(body: Any) -> Callable[[], Any]:
    return lambda: client({"/api/v0/models": (200, _json_bytes(body))}).catalog()


def _chat_over(make_outcome: Callable[[], Any]) -> Callable[[], Any]:
    """Takes a *factory*, not an outcome, matching `_CONNECT_KINDS`/`_READ_KINDS` above and for
    the same reason with teeth here: a `urllib.error.HTTPError` is single-use. `_raw_post` closes
    it (`exc.close()`), so raising one prebuilt instance a second time — and the completeness pin
    below re-runs every scenario — fails inside `tempfile` with "I/O operation on closed file"
    rather than in the adapter. Each call builds its own."""

    def trigger() -> Any:
        return client({"/api/v0/chat/completions": make_outcome()}).chat(
            [{"role": "user", "content": "hi"}],
            model="m",
            temperature=0.0,
            max_tokens=10,
            timeout_s=5.0,
        )

    return trigger


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://localhost:1234/api/v0/chat/completions",
        status,
        "error",
        None,
        io.BytesIO(b"refused"),
    )


#: scenario name -> (trigger, the `status` that scenario's raise site must carry). Two scenarios
#: share `_raw_post`'s rung 1 on purpose, with different statuses: one status value alone cannot
#: tell "read from the response" from "hardcoded".
_STATUS_SCENARIOS: dict[str, tuple[Callable[[], Any], int | None]] = {
    "_parse_json: a 2xx body that is not JSON": (_chat_over(lambda: (200, b"not json{")), None),
    "_model_info_from_raw: a catalog entry that is not an object": (
        _catalog_returning({"data": [42]}),
        None,
    ),
    "_model_info_from_raw: a catalog entry missing a required key": (
        _catalog_returning({"data": [_MISSING_KEY_ENTRY]}),
        None,
    ),
    "catalog: a non-2xx status — the server answered and refused": (
        lambda: client({"/api/v0/models": (500, b"<html>nope</html>")}).catalog(),
        500,
    ),
    "catalog: a 2xx body with no 'data' list": (_catalog_returning({"ok": True}), None),
    "_raw_post rung 1: HTTP 400 — `-ml` §4.1's own example, and §8.4's six lost conversations": (
        _chat_over(lambda: _http_error(400)),
        400,
    ),
    "_raw_post rung 1: HTTP 503 — the same rung, a different status": (
        _chat_over(lambda: _http_error(503)),
        503,
    ),
    "_raw_post rung 3: a dropped connection (URLError)": (
        _chat_over(lambda: urllib.error.URLError(ConnectionResetError("reset by peer"))),
        None,
    ),
    "_raw_post rung 4: a truncated body (http.client.HTTPException)": (
        _chat_over(lambda: _ReadFailsResponse(200, http.client.IncompleteRead(b"partial"))),
        None,
    ),
    "_raw_post rung 5: a bare socket error (OSError)": (
        _chat_over(lambda: OSError("host is unreachable")),
        None,
    ),
    "chat: a 2xx body with no usable 'choices'": (
        _chat_over(lambda: (200, _json_bytes({"choices": []}))),
        None,
    ),
    "chat: a choice with no 'message'": (
        _chat_over(lambda: (200, _json_bytes({"choices": [{"finish_reason": "stop"}]}))),
        None,
    ),
    "embed: a 2xx body with no 'data' list": (
        lambda: client({"/api/v0/embeddings": (200, _json_bytes({"ok": True}))}).embed(
            ["hi"], model="m", timeout_s=5.0
        ),
        None,
    ),
}


def _raise_line_of(trigger: Callable[[], Any]) -> int:
    """Run a scenario and report the source line of the `raise` it reached — the last traceback
    frame belongs to the raise statement itself, which is what lets the completeness pin below
    compare scenarios to source without a hand-maintained line table drifting under either."""
    try:
        trigger()
    except LMStudioCallFailed as exc:
        return traceback.extract_tb(exc.__traceback__)[-1].lineno or -1
    raise AssertionError("scenario did not raise LMStudioCallFailed")


@pytest.mark.parametrize("name", sorted(_STATUS_SCENARIOS))
def test_lmstudio_call_failed_carries_the_refusal_status_and_nothing_else(name):
    trigger, expected_status = _STATUS_SCENARIOS[name]
    with pytest.raises(LMStudioCallFailed) as caught:
        trigger()
    assert caught.value.status == expected_status


def test_every_raise_site_of_lmstudio_call_failed_is_covered_here():
    """The completeness pin, computed on both sides rather than transcribed: the set of source
    lines the scenarios above actually reach must equal the set of lines the module actually
    raises `LMStudioCallFailed` from. A thirteenth raise site added without a scenario reddens
    here — the `status=` keyword being required already forces that site's author to *decide*,
    and this is what forces the decision to be *tested*."""
    source = Path(lmstudio.__file__).read_text()
    raised_in_source = {
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "LMStudioCallFailed"
    }
    covered = {_raise_line_of(trigger) for trigger, _ in _STATUS_SCENARIOS.values()}
    assert covered == raised_in_source


def test_lmstudio_call_failed_docstring_states_what_status_distinguishes():
    """The class docstring used to fold "a non-2xx status, a dropped connection, or an
    unparseable body" into one disposition, which is exactly the conflation §3.8.4 split. A
    reader reaching for the class must find the partition named on it."""
    doc = LMStudioCallFailed.__doc__ or ""
    assert "server-rejected" in doc
    assert "no-response" in doc

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
