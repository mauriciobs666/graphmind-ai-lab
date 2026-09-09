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
ever sees a raw seconds value. Three rules, all load-bearing (plan-gate P5-8): each of
`ttftMs`/`generationMs`/`tokensPerSecond` is `None` when its source key is absent, **never `0`**;
`tokensPerSecond` is the one figure *not* converted, because a per-second rate already is what its
name says; and construction **never raises** on a missing or partial `stats` object — a chat
response without `stats` is an expected state (`-ml` §11.5.1 governs it), not an error. The raw
`stats` mapping is kept beside the derived fields for auditability only; nothing downstream may
read a timing figure out of it.

`timeout_s` is **required, with no default**, on `chat`/`embed`/`warm_up` — §3.6's two budgets
(`firstCallTimeoutSeconds` for the warm-up, `requestTimeoutSeconds` for every scored call) are the
runner's to size and pass; this module does not choose a default that would silently paper over a
missing one.
"""

from __future__ import annotations

import json
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
    timeout: a non-2xx status, a dropped connection, or an unparseable body. This is plan §3.6's
    fourth disposition, `"no_response"` — a *missing* observation, not a censored one, and the
    runner is expected to score it `fail` and continue rather than treat it as a crash."""


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
        stats = self.stats or {}
        object.__setattr__(self, "ttftMs", _seconds_to_ms(stats.get("time_to_first_token")))
        object.__setattr__(self, "generationMs", _seconds_to_ms(stats.get("generation_time")))
        object.__setattr__(self, "tokensPerSecond", stats.get("tokens_per_second"))


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


def _seconds_to_ms(value: Any) -> float | None:
    """§3.6's unit boundary: `None` when absent, never `0`, and no exception on a bad type either
    — a chat response without `stats` (or with a partial one) is an expected state."""
    if value is None:
        return None
    try:
        return 1000.0 * float(value)
    except (TypeError, ValueError):
        return None


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise LMStudioCallFailed(f"{context}: response body is not valid JSON: {exc}") from exc


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
            f"GET /api/v0/models: catalog entry [{index}] is not a JSON object"
        )
    missing = [k for k in _REQUIRED_MODEL_INFO_KEYS if k not in raw]
    if missing:
        raise LMStudioCallFailed(
            f"GET /api/v0/models: catalog entry [{index}] (id={raw.get('id')!r}) "
            f"missing required key(s) {missing}"
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
        """`None` on a connection-level failure (no response at all); `(status, body)` otherwise,
        including a non-2xx status — the caller decides what each status means."""
        req = urllib.request.Request(self._url(path), method="GET")
        try:
            resp = self._opener(req, timeout=timeout_s)
        except urllib.error.HTTPError as exc:  # a subclass of URLError — must precede it
            try:
                return exc.code, exc.read()
            finally:
                exc.close()
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
        try:
            return resp.status, resp.read()
        finally:
            resp.close()

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
        body = _parse_json(raw, context="GET /api/v0/models")
        if status != 200:
            raise LMStudioCallFailed(f"GET /api/v0/models: HTTP {status}")
        entries = body.get("data") if isinstance(body, Mapping) else None
        if not isinstance(entries, list):
            raise LMStudioCallFailed("GET /api/v0/models: response has no 'data' list")
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
        """Returns `(wallClockMs, body)` on a 2xx response. Raises `LMStudioCallTimeout` when the
        budget was exhausted and `LMStudioCallFailed` for every other failure (HTTP error,
        connection drop, unparseable transport) — the distinction §3.6's two withholding
        dispositions ("timeout" vs "no_response") both need, decided here where the evidence is.
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
        except urllib.error.HTTPError as exc:  # rung 1 — a URLError subclass, must precede it
            try:
                body = exc.read()
            finally:
                exc.close()
            raise LMStudioCallFailed(
                f"POST {path}: HTTP {exc.code}: {_truncate(body)}"
            ) from exc
        except TimeoutError as exc:  # rung 2 — NOT a URLError; must be named explicitly
            raise LMStudioCallTimeout(f"POST {path}: timed out after {timeout_s}s") from exc
        except urllib.error.URLError as exc:  # rung 3
            if isinstance(exc.reason, TimeoutError):
                raise LMStudioCallTimeout(f"POST {path}: timed out after {timeout_s}s") from exc
            raise LMStudioCallFailed(f"POST {path}: connection failed: {exc.reason}") from exc
        except OSError as exc:  # rung 4 — any other socket error
            raise LMStudioCallFailed(f"POST {path}: {type(exc).__name__}: {exc}") from exc
        wall_clock_ms = (time.monotonic() - start) * 1000.0
        try:
            raw = resp.read()
        finally:
            resp.close()
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
                "POST /api/v0/chat/completions: response has no usable 'choices'"
            )
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise LMStudioCallFailed(
                "POST /api/v0/chat/completions: choice has no 'message'"
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
            raise LMStudioCallFailed("POST /api/v0/embeddings: response has no 'data' list")
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
        timeout_s: float,
    ) -> LoadResult:
        """The mandatory per-arm warm-up (§3.6). Under JIT auto-load this call **is** the load;
        its content is discarded (never an `ItemResult`, never written to a transcript) but its
        metadata is kept — `runtime`/`stats` on the chat surface are the sole source of
        `runtimeName`/`runtimeVersion` (§3.4.4a step 5).

        `wasResidentBefore` is read from `residency()` **immediately before** issuing the timed
        call, which is what makes it correct on a cold start: `residentModelsAtStart` (queried
        earlier, at capture-order step 3, before this call) is `[]` by construction on a cold run
        and would misreport every cold warm-up as "already resident" if used here instead.
        """
        resident_ids = {rm.id for rm in self.residency()}
        was_resident_before = model in resident_ids
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
