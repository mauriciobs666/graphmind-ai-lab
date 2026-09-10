"""The LM Studio adapter: the one place this tool talks to the outside world.

Design: `docs/plans/small-model-benchmarking.md` §3.4.4a (source of truth, capture order,
`callSurface`) and §3.6 (the four HTTP operations, the unit boundary, the eligibility gate).
`-ml` refers to `docs/plans/small-model-benchmarking-ml.md`.

Four operations, all against `GET/POST {base_url}/api/v0/...` — never `/v1/...` for anything but
`probe()`'s reachability check, because `/v1/models` returns `{id, object, owned_by}` and nothing
else (§2.5) while every fingerprint field and every `stats`/`runtime` object lives on the `/api/v0`
surface. **No `load`/`unload`/`ps`** — v1.7's three `lms.exe` operations are gone with the CLI
(§2.5, §3.4.4a), and nothing on either HTTP surface can unload a model: the harness cannot force a
cold state, by design, not by omission.

**The unit boundary (§3.6, plan-gate P4-1) lives here and nowhere else.** LM Studio's `stats`
object reports `time_to_first_token` and `generation_time` in **seconds**; every `...Ms` field
this plan names is **milliseconds**. `ChatResult` converts once, on construction, so no caller
ever sees a raw seconds value. Four rules, all load-bearing (plan-gate P5-8; the fourth is review
Pass 13, P13-3): each of `ttftMs`/`generationMs`/`tokensPerSecond` is `None` when its source key is
absent, **never `0`**; `tokensPerSecond` is the one figure *not unit-converted* (no ×1000 — a
per-second rate already is what its name says) but it **is** type-coerced, same as the other two;
a `bool` source or a non-finite one (`nan`/`inf`/`-inf` — `float()` accepts these, and
`json.loads` parses the bare `NaN`/`Infinity` tokens without error, so no malformed transport is
needed) is rejected at this coercion boundary rather than surviving as a number, deliberately
**not** at `_parse_json`, which parses every body this adapter reads and keeps `stats` verbatim
for auditability; and construction **never raises**, full stop, regardless of what `stats` holds —
not just on a missing or partial `stats` object, but on one of the wrong type entirely (a list, a
string, anything not `Mapping`-shaped).
A chat response without usable `stats` is an expected state (`-ml` §11.5.1 governs it), not an
error, and nothing about that guarantee should depend on `chat()`'s own `isinstance` check one
level up — a caller that constructs `ChatResult` directly gets the same promise (review Pass 12,
P12-11). The raw `stats` value is kept beside the derived fields verbatim, exactly as given, for
auditability only; nothing downstream may read a timing figure out of it.

`timeout_s` is **required, with no default**, on `chat`/`embed`/`warm_up` — §3.6's two budgets
(`firstCallTimeoutSeconds` for the warm-up, `requestTimeoutSeconds` for every scored call) are the
runner's to size and pass; this module does not choose a default that would silently paper over a
missing one.
"""

from __future__ import annotations

import http.client
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

# `catalog()`/`residency()`/`probe()` are not one of §3.6's two timing budgets — they are ~ms
# calls (§2.5 measured 1.6-2.3 ms) used before and between model calls, never around one. This is
# a generous socket safety net, not a tuned value; nothing in the plan sizes it.
_DEFAULT_CATALOG_TIMEOUT_S = 10.0

# Fixed, pack-independent warm-up payloads (§3.6): content is discarded, never scored, never
# written to a transcript. The literal strings are this module's own choice — the plan requires
# only that the call be fixed and pack-independent, not what it says.
_WARMUP_CHAT_MESSAGE = "Hello."
_WARMUP_EMBED_TEXT = "warm-up"

CallSurface = Literal["chat", "embeddings"]
ProbeResult = Literal["api-v0", "v1-only", "unreachable"]
ToolCallForm = Literal["native", "prose"]


class LMStudioError(RuntimeError):
    """Base for every adapter-level failure. Never raised directly."""


class LMStudioUnreachable(LMStudioError):
    """Neither `/api/v0/models` nor `/v1/models` answered (§3.4.4a's `unreachable` probe outcome),
    or a `GET /api/v0/models` call (`catalog`/`residency`) got no response at all."""


class LMStudioCallTimeout(LMStudioError):
    """A `chat`/`embed`/`warm_up` call exceeded its `timeout_s` budget — a *censored*
    observation, distinct from a connection or protocol failure (`-ml` §11.5.1; plan §3.6's
    timeout disposition). Decided here, at the transport boundary, because this is the one place
    that knows which of the two actually happened."""


class LMStudioCallFailed(LMStudioError):
    """A `chat`/`embed`/`warm_up` call returned no usable response for a reason other than a
    timeout. This is plan §3.6's fourth disposition, `"no_response"` — a *missing* observation,
    not a censored one, and the runner is expected to score it `fail` and continue rather than
    treat it as a crash.

    **`status` is what distinguishes the two mechanisms this class covers** (plan §3.8.4's
    four-row `turnDisposition` table, v1.26): the HTTP status when the server answered and
    refused — §3.8.4's `server-rejected`, `-ml` §4.1's `unrunnable` count — and `None` when the
    call never completed: a dropped connection, a socket error, or a body that could not be read
    as a response, which is §3.8.4's `no-response` and scores `fail`. The two used to be
    indistinguishable to any caller (this docstring itself folded "a non-2xx status, a dropped
    connection, or an unparseable body" into one disposition), and `drive` partitions on
    `status is not None`, so the distinction is decided here, at the boundary that has the
    evidence, exactly as `LMStudioCallTimeout` already is.

    A **2xx whose body is unusable carries `None`**, deliberately: unparseable JSON, a missing
    `choices`/`data` list, a malformed catalog entry. The server did not *refuse* — §3.8.4 files
    a body error under `no-response` — so this is the refusal status, never "the last status
    seen". `status` is a **required** keyword argument for the same reason: a new raise site must
    decide which side of that partition it is on rather than inheriting a default that silently
    scores `fail`.
    """

    def __init__(self, message: str, *, status: int | None) -> None:
        super().__init__(message)
        self.status = status


class ToolCallingIneligible(LMStudioError):
    """§3.6's tool-calling eligibility gate refused a model on a `tool-caller` pack. Raised only
    by `check_tool_calling_eligibility`, and only when its `role` argument is `"tool-caller"`."""


@dataclass(frozen=True)
class ModelInfo:
    """One `/api/v0/models` entry, verbatim (§2.3, re-probed §2.5; plan Appendix A).

    `capabilities` is `None` when the raw entry has no `capabilities` key at all, and a (possibly
    empty) tuple when it does — the catalog's own absent-versus-empty distinction (§3.4.4a),
    carried down to this type rather than collapsed by a `.get(..., [])` default. Two real catalog
    entries turn on exactly this: `text-embedding-qwen3-embedding-0.6b` has `capabilities` present
    with `tool_use`, while `google/gemma-3-4b` has no `capabilities` key at all.
    """

    id: str
    object: str
    type: str
    publisher: str
    arch: str
    compatibility_type: str
    quantization: str
    state: str
    max_context_length: int
    capabilities: tuple[str, ...] | None
    loaded_context_length: int | None


@dataclass(frozen=True)
class ResidentModel:
    """One `/api/v0/models` row surviving `state != "not-loaded"` — `{id, state}`, the literal
    `state` string kept rather than normalised to a boolean (§3.4.4a: the 2026-09-03 probe saw
    only `"not-loaded"`, so any other value this build can report is one the plan has not seen)."""

    id: str
    state: str


@dataclass(frozen=True)
class ChatResult:
    """`POST /api/v0/chat/completions`'s response, plus the derived timing trio (§3.6's unit
    boundary) and `toolCallForm` (FR-8(b)).

    `toolCallForm` is decided here, at the transport boundary, on the one fact only this layer
    can observe directly: did the response use LM Studio's *native* tool-calling mechanism
    (`message.tool_calls` non-empty) or not. `"native"` when it did; `"prose"` otherwise — meaning
    any call-shaped content the model produced, if any, is necessarily embedded in `message`'s
    text rather than expressed through the API's own mechanism. Whether that prose *looks like* an
    attempted call is a scoring judgement against pack-labelled examples (`scoring/toolcalls.py`,
    a later unit) and is deliberately not decided here — this field only ever states the mechanism.

    Construction **never raises**, on any `stats` this class is handed — absent, partial, or the
    wrong type entirely (not `Mapping`-shaped at all). That is a property of `__post_init__`
    itself, not of any caller's guard: `chat()` happens to only ever pass `None` or a `Mapping`,
    but a direct construction with a malformed `stats` (a list, a string, ...) degrades the same
    way rather than raising `AttributeError` (review Pass 12, P12-11). `ttftMs`/`generationMs`/
    `tokensPerSecond` are each `None` when there is nothing usable to derive them from, never `0`
    and never the raw un-coerced value; `tokensPerSecond` skips the ×1000 the other two apply (a
    per-second rate needs no unit conversion) but is coerced to `float` the same way they are, so a
    string- or otherwise wrong-typed source lands `None` rather than surviving untyped into a field
    declared `float | None`. A `bool` source, or one that coerces to a non-finite `float`
    (`nan`/`inf`/`-inf` — reachable with no malformed transport at all, since `json.loads` parses
    the bare tokens by default), lands `None` the same way (review Pass 13, P13-3) — the coercion
    boundary rejects both; `_parse_json` rejects neither. `stats` itself is kept exactly as given,
    whatever its shape, purely for auditability — nothing downstream may read a timing figure out
    of it.
    """

    message: Mapping[str, Any]
    tool_calls: tuple[Mapping[str, Any], ...]
    toolCallForm: ToolCallForm
    stats: Mapping[str, Any] | None
    model_info: Mapping[str, Any] | None
    runtime: Mapping[str, Any] | None
    usage: Mapping[str, Any] | None
    wallClockMs: float | None
    ttftMs: float | None = field(init=False)
    generationMs: float | None = field(init=False)
    tokensPerSecond: float | None = field(init=False)

    def __post_init__(self) -> None:
        stats = self.stats if isinstance(self.stats, Mapping) else {}
        object.__setattr__(self, "ttftMs", _seconds_to_ms(stats.get("time_to_first_token")))
        object.__setattr__(self, "generationMs", _seconds_to_ms(stats.get("generation_time")))
        object.__setattr__(self, "tokensPerSecond", _as_float(stats.get("tokens_per_second")))


@dataclass(frozen=True)
class EmbedResult:
    """`POST /api/v0/embeddings`'s response. `dimension` is the length of the *first* vector
    (AC-5) — `None` when the batch came back empty, never `0`."""

    vectors: tuple[tuple[float, ...], ...]
    dimension: int | None
    model: str | None
    usage: Mapping[str, Any] | None
    wallClockMs: float | None


@dataclass(frozen=True)
class LoadResult:
    """The warm-up call's outcome (§3.6). `wasResidentBefore` is the fact that decides whether the
    runner may treat `wallClockMs` as `coldLoadSeconds`; `runtime`/`stats` are populated on the
    chat surface only — they are the sole source of `runtimeName`/`runtimeVersion` (§3.4.4a step
    5) — and are `None` on an embeddings warm-up, which returns neither."""

    wallClockMs: float | None
    wasResidentBefore: bool
    runtime: Mapping[str, Any] | None
    stats: Mapping[str, Any] | None


def _coerce_finite_float(value: Any) -> float | None:
    """The shared half of both `_seconds_to_ms` and `_as_float`: `None` when `value` is absent,
    a `bool`, not coercible to `float` at all, or coercible but non-finite (`nan`/`inf`/`-inf`) —
    never the raw value surviving untyped or non-finite into a `float | None` field (review
    Pass 13, P13-3).

    `bool` is excluded before coercion, not after: `float(True) == 1.0`, so an unguarded coercion
    would turn a stray boolean into a real-looking number (`1000.0`, `1.0`) rather than `None`.
    This mirrors the same deliberate exclusion elsewhere in the component
    (`packs._row_count_identity_field_valid`, `results`' bool guard from review Pass 2 P2-2).

    Non-finite is rejected **here, at the coercion boundary — not at `_parse_json`/`json.loads`.**
    `json.loads` parses the bare tokens `NaN`/`Infinity`/`-Infinity` without error by default
    (Python's `parse_constant`), so a chat body serialising e.g. a 0/0 rate needs no malformed
    transport to reach this class at all. The boundary is deliberately *not* moved into
    `_parse_json`: that function parses every response body this adapter reads (catalog, chat,
    embed), and the raw `stats` mapping is kept verbatim beside the derived fields "for
    auditability" — narrowing the fix to the two functions that actually produce a typed timing
    figure keeps that verbatim guarantee intact and leaves every other body's parsing unchanged.
    A non-finite value degrading silently downstream is not cosmetic: `statistics.median` over a
    list containing `nan` returns an arbitrary element with no error, and `-ml` §11.5.1's gap
    (`latencyMs - (ttftMs + generationMs)`) would go `-inf`, so the in-call reload detector could
    never fire on that item.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _seconds_to_ms(value: Any) -> float | None:
    """§3.6's unit boundary: `None` when absent, never `0`, and no exception on a bad type or a
    non-finite one either — a chat response without `stats` (or with a partial or non-finite one)
    is an expected state (review Pass 13, P13-3)."""
    v = _coerce_finite_float(value)
    return None if v is None else 1000.0 * v


def _as_float(value: Any) -> float | None:
    """`tokensPerSecond`'s half of §3.6's unit boundary: no ×1000 (already a per-second rate), but
    the same tolerance `_seconds_to_ms` applies — `None` when absent, a `bool`, not coercible to
    `float` (review Pass 12, P12-11), or coercible but non-finite (review Pass 13, P13-3) — never
    the raw value surviving untyped into a `float | None` field."""
    return _coerce_finite_float(value)


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise LMStudioCallFailed(
            f"{context}: response body is not valid JSON: {exc}", status=None
        ) from exc


_REQUIRED_MODEL_INFO_KEYS = (
    "id",
    "object",
    "type",
    "publisher",
    "arch",
    "compatibility_type",
    "quantization",
    "state",
    "max_context_length",
)


def _model_info_from_raw(raw: Any, *, index: int) -> ModelInfo:
    if not isinstance(raw, Mapping):
        raise LMStudioCallFailed(
            f"GET /api/v0/models: catalog entry [{index}] is not a JSON object", status=None
        )
    missing = [k for k in _REQUIRED_MODEL_INFO_KEYS if k not in raw]
    if missing:
        raise LMStudioCallFailed(
            f"GET /api/v0/models: catalog entry [{index}] (id={raw.get('id')!r}) "
            f"missing required key(s) {missing}",
            status=None,
        )
    capabilities = raw.get("capabilities")
    return ModelInfo(
        id=raw["id"],
        object=raw["object"],
        type=raw["type"],
        publisher=raw["publisher"],
        arch=raw["arch"],
        compatibility_type=raw["compatibility_type"],
        quantization=raw["quantization"],
        state=raw["state"],
        max_context_length=raw["max_context_length"],
        capabilities=tuple(capabilities) if capabilities is not None else None,
        loaded_context_length=raw.get("loaded_context_length"),
    )


def tool_calling_eligible(model_info: ModelInfo) -> tuple[bool, str | None]:
    """§3.6's eligibility rule, verbatim:

    > eligible iff `type in {"llm", "vlm"}` and (`capabilities` is absent or contains `tool_use`)

    Returns `(eligible, reason)` — `reason` names *which half* failed, `None` when eligible, so a
    refusal can say why rather than just that (§3.6: "a refusal names which half failed").
    """
    if model_info.type not in ("llm", "vlm"):
        return False, f"type {model_info.type!r} is not in {{'llm', 'vlm'}}"
    if model_info.capabilities is not None and "tool_use" not in model_info.capabilities:
        return False, f"capabilities {list(model_info.capabilities)!r} do not include 'tool_use'"
    return True, None


def check_tool_calling_eligibility(role: str, model_info: ModelInfo) -> None:
    """Runs §3.6's gate only on a `tool-caller` pack (v1.11, plan-gate P5-1) — scope is
    load-bearing, not tidy: run unscoped, the gate refuses `type == "embeddings"` and so refuses
    every `model:embeddings` arm before the adapter is ever called (the defect P5-1 closed).

    `role` is the pack's plain role string — **never a `Pack` object**. `packs.py` (`Pack`,
    `load_pack`, ...) is a concurrent unit this wave and its shape is not final; the later wiring
    unit is expected to call this as `check_tool_calling_eligibility(pack.role, model_info)`.
    """
    if role != "tool-caller":
        return
    eligible, reason = tool_calling_eligible(model_info)
    if not eligible:
        raise ToolCallingIneligible(
            f"{model_info.id!r} is not eligible for a tool-caller pack ({reason})"
        )


class LMStudio:
    """The adapter. `base_url` is a plain constructor parameter (plan §3.4.4: it comes from
    `host.json`'s `apiBaseUrl`, a later unit's file to read — this module never reads it itself).

    `opener` is injected the way `falkorchat/transport.py`'s own HTTP transport is: it defaults to
    `urllib.request.urlopen` and every offline test overrides it with a fake, so this class never
    opens a real socket outside the one `-m live` test (§4 S2's done-condition)."""

    def __init__(
        self,
        base_url: str,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._opener = opener

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    # --- GET: catalog / residency / probe -----------------------------------------------------

    def _raw_get(self, path: str, timeout_s: float) -> tuple[int, bytes] | None:
        """`None` on any connect- or read-phase failure (no trustworthy response at all);
        `(status, body)` otherwise, including a non-2xx status — the caller decides what each
        status means. The body read sits inside the same ladder as the connect (Pass 12 P12-1):
        a response whose `read()` raises (`http.client.IncompleteRead`, a dropped connection, a
        read-phase timeout) is exactly as unreachable as one that never connected, so both
        phases fold to the same `None` — `IncompleteRead` is not an `OSError`, so it needs its
        own rung rather than riding the socket catch.

        The *error* body's own read (`exc.read()`, below) gets the identical treatment, and it
        needs its own guard rather than inheriting the surrounding `try`'s (Pass 13 P13-1):
        `HTTPError` **is** a response object — a status plus a `.read()` — so reading its body can
        fail exactly like reading a success response's body can, and `probe()`'s own `v1-only`
        diagnosis calls `exc.read()` on every 404 a non-LM-Studio server returns, which makes this
        normal-path code rather than an edge case. Folding it to `None` here (rather than
        surfacing `LMStudioCallFailed` with the known status) matches the read-phase fold just
        above: neither branch has a trustworthy body to report a message from, so both degrade to
        the same "no usable response" outcome, one rung apart."""
        req = urllib.request.Request(self._url(path), method="GET")
        try:
            resp = self._opener(req, timeout=timeout_s)
            try:
                return resp.status, resp.read()
            finally:
                resp.close()
        except urllib.error.HTTPError as exc:  # a subclass of URLError — must precede it
            try:
                try:
                    return exc.code, exc.read()
                except (TimeoutError, http.client.HTTPException, OSError):
                    return None
            finally:
                exc.close()
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError):
            return None

    def probe(self) -> ProbeResult:
        """§3.4.4a's two-step probe, against stubbed HTTP in every offline test:

        - `/api/v0/models` answers with a well-formed catalog body -> `"api-v0"`.
        - it does not, but `/v1/models` answers -> `"v1-only"` (a server that speaks the
          OpenAI-compatible surface but not LM Studio's own — an older build, a proxy, or the
          wrong port).
        - neither answers -> `"unreachable"`.
        """
        v0 = self._raw_get("/api/v0/models", _DEFAULT_CATALOG_TIMEOUT_S)
        if v0 is not None and v0[0] == 200 and _looks_like_catalog_body(v0[1]):
            return "api-v0"
        v1 = self._raw_get("/v1/models", _DEFAULT_CATALOG_TIMEOUT_S)
        if v1 is not None and v1[0] == 200:
            return "v1-only"
        return "unreachable"

    def catalog(self) -> list[ModelInfo]:
        """`GET /api/v0/models` — the only source for 13 of the fingerprint's 26 auto-captured
        fields (§3.4.4a's five-source table)."""
        result = self._raw_get("/api/v0/models", _DEFAULT_CATALOG_TIMEOUT_S)
        if result is None:
            raise LMStudioUnreachable(
                f"GET /api/v0/models: no response from {self._base_url}"
            )
        status, raw = result
        if status != 200:
            # Checked before parsing (Pass 12 P12-10): an error body is often not JSON at all
            # (an HTML error page, a proxy's plain-text response), and parsing it first reported
            # "not valid JSON" instead of the real HTTP status the operator actually needs.
            raise LMStudioCallFailed(f"GET /api/v0/models: HTTP {status}", status=status)
        body = _parse_json(raw, context="GET /api/v0/models")
        entries = body.get("data") if isinstance(body, Mapping) else None
        if not isinstance(entries, list):
            raise LMStudioCallFailed(
                "GET /api/v0/models: response has no 'data' list", status=None
            )
        return [_model_info_from_raw(e, index=i) for i, e in enumerate(entries)]

    def residency(self) -> list[ResidentModel]:
        """The catalog filtered on `state != "not-loaded"` (§3.4.4a). `[]` on an all-not-loaded
        catalog is the correct, informative clean-box answer — it means a probe succeeded and
        found nothing, not that no probe ran."""
        return [
            ResidentModel(id=m.id, state=m.state)
            for m in self.catalog()
            if m.state != "not-loaded"
        ]

    # --- POST: chat / embed / warm-up ---------------------------------------------------------

    def _raw_post(
        self, path: str, payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[float, bytes]:
        """Returns `(wallClockMs, body)` on a 2xx response, the clock stopped at the *last byte
        of the body* (§3.6 FR-11: "measured around the HTTP call from just before the request to
        the last byte of the body" — Pass 12 P12-4: it used to stop at the response headers, so
        a slow body read off a fast-responding server was invisible). Raises
        `LMStudioCallTimeout` when the budget was exhausted, at either the connect or the read
        phase, and `LMStudioCallFailed` for every other failure (HTTP error, connection drop,
        unparseable transport, a body read that never completes) — the distinction §3.6's two
        withholding dispositions ("timeout" vs "no_response") both need, decided here where the
        evidence is. The body read sits inside the same ladder as the connect (Pass 12 P12-1): a
        response whose `read()` raises used to escape this method's taxonomy entirely.

        The *error* body's read (`exc.read()`, rung 1 below) gets the same guard, separately
        (Pass 13 P13-1): `HTTPError` is itself a response object, so reading its body can fail
        exactly like a success response's can, and it used to escape this method's taxonomy the
        same way the success-body read once did. Unlike the GET side, the status is already in
        hand (`exc.code`) before the read is attempted, so a failed error-body read degrades the
        *message* (an empty body) rather than the exception type — the cell stays
        `LMStudioCallFailed`, identical to every other non-2xx POST outcome.
        """
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(path),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        try:
            resp = self._opener(req, timeout=timeout_s)
            try:
                raw = resp.read()
            finally:
                resp.close()
        except urllib.error.HTTPError as exc:  # rung 1 — a URLError subclass, must precede it
            try:
                try:
                    body = exc.read()
                except (TimeoutError, http.client.HTTPException, OSError):
                    body = b""
            finally:
                exc.close()
            raise LMStudioCallFailed(
                f"POST {path}: HTTP {exc.code}: {_truncate(body)}", status=exc.code
            ) from exc
        except TimeoutError as exc:  # rung 2 — NOT a URLError; must be named explicitly. Covers
            # a connect-phase timeout and a read-phase one alike (`socket.timeout` has been an
            # alias of `TimeoutError` since 3.10), because `resp.read()` sits inside this try.
            raise LMStudioCallTimeout(f"POST {path}: timed out after {timeout_s}s") from exc
        except urllib.error.URLError as exc:  # rung 3
            if isinstance(exc.reason, TimeoutError):
                raise LMStudioCallTimeout(f"POST {path}: timed out after {timeout_s}s") from exc
            raise LMStudioCallFailed(
                f"POST {path}: connection failed: {exc.reason}", status=None
            ) from exc
        except http.client.HTTPException as exc:  # rung 4 — a dropped/truncated body, e.g.
            # `IncompleteRead`; not an `OSError`, so the socket rung below would not catch it.
            raise LMStudioCallFailed(
                f"POST {path}: {type(exc).__name__}: {exc}", status=None
            ) from exc
        except OSError as exc:  # rung 5 — any other socket error, connect or read phase
            raise LMStudioCallFailed(
                f"POST {path}: {type(exc).__name__}: {exc}", status=None
            ) from exc
        wall_clock_ms = (time.monotonic() - start) * 1000.0
        return wall_clock_ms, raw

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> ChatResult:
        """`POST /api/v0/chat/completions` — not `/v1/...`: `stats`/`model_info`/`runtime` exist
        only on this route (§2.3, §3.6)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = list(tools)
        wall_clock_ms, raw = self._raw_post("/api/v0/chat/completions", payload, timeout_s)
        body = _parse_json(raw, context="POST /api/v0/chat/completions")
        choices = body.get("choices") if isinstance(body, Mapping) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise LMStudioCallFailed(
                "POST /api/v0/chat/completions: response has no usable 'choices'", status=None
            )
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise LMStudioCallFailed(
                "POST /api/v0/chat/completions: choice has no 'message'", status=None
            )
        raw_tool_calls = message.get("tool_calls")
        tool_calls = tuple(raw_tool_calls) if isinstance(raw_tool_calls, list) else ()
        tool_call_form: ToolCallForm = "native" if tool_calls else "prose"
        stats = body.get("stats")
        model_info = body.get("model_info")
        runtime = body.get("runtime")
        usage = body.get("usage")
        return ChatResult(
            message=message,
            tool_calls=tool_calls,
            toolCallForm=tool_call_form,
            stats=stats if isinstance(stats, Mapping) else None,
            model_info=model_info if isinstance(model_info, Mapping) else None,
            runtime=runtime if isinstance(runtime, Mapping) else None,
            usage=usage if isinstance(usage, Mapping) else None,
            wallClockMs=wall_clock_ms,
        )

    def embed(
        self, texts: Sequence[str], *, model: str, timeout_s: float
    ) -> EmbedResult:
        """`POST /api/v0/embeddings` — batched; `dimension` from the first vector (AC-5)."""
        payload: dict[str, Any] = {"model": model, "input": list(texts)}
        wall_clock_ms, raw = self._raw_post("/api/v0/embeddings", payload, timeout_s)
        body = _parse_json(raw, context="POST /api/v0/embeddings")
        data = body.get("data") if isinstance(body, Mapping) else None
        if not isinstance(data, list):
            raise LMStudioCallFailed(
                "POST /api/v0/embeddings: response has no 'data' list", status=None
            )
        vectors = tuple(
            tuple(entry["embedding"])
            for entry in data
            if isinstance(entry, Mapping) and isinstance(entry.get("embedding"), list)
        )
        dimension = len(vectors[0]) if vectors else None
        usage = body.get("usage")
        response_model = body.get("model")
        return EmbedResult(
            vectors=vectors,
            dimension=dimension,
            model=response_model if isinstance(response_model, str) else None,
            usage=usage if isinstance(usage, Mapping) else None,
            wallClockMs=wall_clock_ms,
        )

    def warm_up(
        self,
        model: str,
        *,
        call_surface: CallSurface,
        system_prompt: str | None,
        was_resident_before: bool,
        timeout_s: float,
    ) -> LoadResult:
        """The mandatory per-arm warm-up (§3.6). Under JIT auto-load this call **is** the load;
        its content is discarded (never an `ItemResult`, never written to a transcript) but its
        metadata is kept — `runtime`/`stats` on the chat surface are the sole source of
        `runtimeName`/`runtimeVersion` (§3.4.4a step 5).

        `was_resident_before` is supplied by the caller, from `residentModelsAtStart`
        (§3.4.4a capture-order step 3) — this method does not probe `residency()` itself (fixed
        at S2 U75; `docs/reviews/small-model-benchmarking-impl.md` Pass 12 P12-5). §3.6 names
        `residentModelsAtStart` as `coldLoadSeconds`'s source in so many words — "recorded only
        when the model was not resident at start" — and on a cold run that snapshot is `[]` by
        construction, so `model in set()` is `False`: "not resident", the *correct* answer, not
        a misreport. An earlier version of this method re-probed `residency()` immediately
        before the timed call instead, and justified it by inverting that same fact; the
        substitution was undeclared and cost an unlisted extra catalog GET on every warm-up.
        """
        if call_surface == "chat":
            messages: list[Mapping[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": _WARMUP_CHAT_MESSAGE})
            chat_result = self.chat(
                messages,
                model=model,
                temperature=0.0,
                max_tokens=10,
                timeout_s=timeout_s,
            )
            return LoadResult(
                wallClockMs=chat_result.wallClockMs,
                wasResidentBefore=was_resident_before,
                runtime=chat_result.runtime,
                stats=chat_result.stats,
            )
        if call_surface == "embeddings":
            embed_result = self.embed([_WARMUP_EMBED_TEXT], model=model, timeout_s=timeout_s)
            return LoadResult(
                wallClockMs=embed_result.wallClockMs,
                wasResidentBefore=was_resident_before,
                runtime=None,
                stats=None,
            )
        raise ValueError(f"warm_up: unknown call_surface {call_surface!r}")


def _looks_like_catalog_body(raw: bytes) -> bool:
    try:
        body = json.loads(raw)
    except ValueError:
        return False
    return isinstance(body, Mapping) and isinstance(body.get("data"), list)


def _truncate(raw: bytes, limit: int = 500) -> str:
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover — defensive; a body read must never mask the real error
        return ""
    return text if len(text) <= limit else text[:limit] + "…"
