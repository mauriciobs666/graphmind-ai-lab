"""Load and residency (docs/plans/llm-gateway.md §3.8, S10; FR-16, AC-13).

`Residency` talks to one provider endpoint: `loaded()` lists what is resident, `load(model,
settings)` asks the provider to load one. `LoadResidency` is the engine's `ResidencyHook`: before
each attempt it reads `loaded()` (always first, over the same transport the call uses) and applies
the three modes.

Rules, each pinned by a test:
  * `on_demand` (the default) loads a model only when `loaded()` does not list it; `strict` and
    `explicit` never load inside a call: an absent model fails `model_not_loaded`, nothing sent
    beyond the `loaded()` read. `explicit` loading is `Gateway.load(call_site)`, which records an
    attempt carrying the loaded state.
  * A model that is present is never loaded again (no second copy), never reloaded and never
    unloaded; there is no unload code path at all.
  * Configured load settings and a present model: `context_length` is the one setting `loaded()`
    reports, so it is the one compared; a differing (or unreported) value fails `config` naming
    the mismatch, in every mode. With none configured the loaded model is accepted and recorded.
    The other settings (flash attention, `keep_alive`, ...) apply only to a load this gateway
    performs; they cannot be verified on a model it did not load.
  * `bind_loaded = true` on a call site binds the loaded chat model (embedding models are skipped;
    the configured model is preferred), failing `model_not_loaded` when none is loaded; it never
    loads.
  * Load settings are validated before anything is sent, so `parallel` on Ollama fails naming it
    even when the model is already resident. They are refused on a hosted step.

Call-site / provider configuration read here (shapes the plan leaves open):
  `load_mode = "on_demand" | "explicit" | "strict"` (a request's `load_mode` overrides it),
  `load = {context_length = N, ...}` (call site first, else the provider table), and
  `bind_loaded = true`.
A provider kind supports residency by defining `loaded_request()`, `parse_loaded(body)` and
`load_request(...)` (LM Studio, Ollama); other kinds pass through untouched.

The recorded loaded state is `loaded:<id>` (resident, reused or accepted), `loaded_on_demand:<id>`,
`loaded_explicit:<id>`, or on a failed attempt `not_loaded:<id>` / `mismatch:<id>`, lower-cased
and cut to the record's 64-character `[a-z0-9_:.-]` alphabet.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from llmgateway.engine import PreparedStep, StepContext
from llmgateway.failures import Condition, classify_condition, classify_exception
from llmgateway.providers.base import BuiltRequest, Provider
from llmgateway.transport import Transport, TransportFailure, WireRequest, WireResponse
from llmgateway.types import CallFailed, FailureClass, HostedFromCallerProcess, LoadMode

EMBEDDING_TYPES = frozenset({"embeddings", "embedding"})


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """One resident model. `raw` is the provider's verbatim record (ccg rebuilds `fingerprint`
    and `quantization` from it)."""

    id: str
    context_length: int | None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_chat(self) -> bool:
        return str(self.raw.get("type", "")).lower() not in EMBEDDING_TYPES


class ResidencyFailed(CallFailed):
    """A residency failure that also states the loaded state to record on the attempt."""

    def __init__(self, failure_class: FailureClass, message: str, loaded_state: str) -> None:
        super().__init__(failure_class, message)
        self.loaded_state = loaded_state


def _same_model(a: str, b: str) -> bool:
    return a == b or f"{a}:latest" == b or a == f"{b}:latest"


def find_loaded(loaded: Iterable[LoadedModel], model: str | None) -> LoadedModel | None:
    if not model:
        return None
    return next((m for m in loaded if _same_model(m.id, model)), None)


def bind_loaded(loaded: Iterable[LoadedModel], prefer_model: str | None = None) -> LoadedModel:
    """The loaded chat model, preferring `prefer_model`; never loads, fails when none (FR-16)."""
    chat = [m for m in loaded if m.is_chat]
    if not chat:
        raise CallFailed(FailureClass.MODEL_NOT_LOADED, "no chat model is loaded to bind to")
    return find_loaded(chat, prefer_model) or chat[0]


def state_label(prefix: str, model_id: str) -> str:
    return re.sub(r"[^a-z0-9_:.-]", "_", f"{prefix}:{model_id}".lower())[:64]


class Residency:
    """`loaded()` and `load()` against one provider endpoint, over the call's own transport."""

    def __init__(
        self,
        transport: Transport,
        provider: Provider,
        provider_id: str,
        config: Mapping[str, Any],
        *,
        component: str = "",
        scopes: tuple[str, ...] = (),
        run_id: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._transport, self._provider = transport, provider
        self._id, self._config = provider_id, config
        self._wire = {
            "component": component,
            "scopes": scopes,
            "run_id": run_id,
            "timeout_s": timeout_s,
        }

    def loaded(self) -> list[LoadedModel]:
        built = self._provider.loaded_request()  # type: ignore[attr-defined]
        reply = self._exchange(built)
        try:
            return [
                LoadedModel(r["id"], r.get("context_length"), r["raw"])
                for r in self._provider.parse_loaded(reply.body)  # type: ignore[attr-defined]
            ]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise CallFailed(
                FailureClass.CONFIG, f"{self._id}: the loaded-models reply is not valid"
            ) from exc

    def check_settings(self, model: str, settings: Mapping[str, Any]) -> BuiltRequest:
        """The load request for `settings`, built (and so validated) without sending it."""
        build = self._provider.load_request  # type: ignore[attr-defined]
        if "config" in inspect.signature(build).parameters:
            return build(self._id, model, settings, self._config)
        return build(self._id, model, settings)

    def load(self, model: str, settings: Mapping[str, Any]) -> None:
        self._exchange(self.check_settings(model, settings))

    def _exchange(self, built: BuiltRequest) -> WireResponse:
        wire = WireRequest(
            provider_id=self._id,
            kind=self._provider.kind,
            base_url=str(self._config.get("base_url", "")),
            path=built.path,
            body=built.body,
            hosted=False,
            method=built.method,
            headers=built.headers,
            **self._wire,
        )
        try:
            reply = self._transport.exchange(wire)
        except (HostedFromCallerProcess, CallFailed):
            raise
        except TransportFailure as exc:
            raise CallFailed(classify_condition(exc.condition), f"{self._id}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - classify_exception owns the mapping
            raise CallFailed(classify_exception(exc), f"{self._id}: {type(exc).__name__}") from exc
        if not 200 <= reply.status < 300:
            fc = (
                FailureClass.REJECTED
                if reply.status < 400
                else self._provider.classify(
                    reply.status, reply.content_type, reply.body, self._config
                )
            )
            raise CallFailed(fc, f"{self._id}: {fc.value}, HTTP {reply.status} on {built.path}")
        return reply


def _config_error(message: str) -> CallFailed:
    return CallFailed(FailureClass.CONFIG, message)


def load_mode(ctx: StepContext) -> LoadMode:
    if ctx.request.load_mode is not None:
        return ctx.request.load_mode
    raw = ctx.site.get("load_mode")
    if raw is None:
        return LoadMode.ON_DEMAND
    try:
        return LoadMode(raw)
    except ValueError:
        raise _config_error(
            f"call site load_mode must be one of {', '.join(m.value for m in LoadMode)}"
        ) from None


def load_settings(ctx: StepContext) -> dict[str, Any]:
    pconf = ctx.config.providers[ctx.step.provider]
    raw = ctx.site.get("load", pconf.get("load", {}))
    if not isinstance(raw, Mapping):
        raise _config_error("load settings must be a table")
    return dict(raw)


def _supports_residency(provider: Provider) -> bool:
    return all(hasattr(provider, n) for n in ("loaded_request", "parse_loaded", "load_request"))


class LoadResidency:
    """The engine's residency hook (`engine.ResidencyHook`) plus `load` for `Gateway.load`."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def prepare(self, ctx: StepContext) -> PreparedStep:
        return self._run(ctx, may_load=load_mode(ctx) is LoadMode.ON_DEMAND, label="on_demand")

    def load(self, ctx: StepContext) -> PreparedStep:
        """`explicit` mode: the caller's own load. Loads when absent, reuses what is resident."""
        return self._run(ctx, may_load=True, label="explicit")

    def _run(self, ctx: StepContext, *, may_load: bool, label: str) -> PreparedStep:
        pid = ctx.step.provider
        settings = load_settings(ctx)
        if ctx.step.hosted:
            if settings:
                raise _config_error(f"provider {pid!r}: load settings are refused on a hosted step")
            return PreparedStep(ctx.model)
        bind = ctx.site.get("bind_loaded") is True
        if not _supports_residency(ctx.provider):
            if bind or settings:
                raise _config_error(f"provider kind {ctx.provider.kind!r} has no residency")
            return PreparedStep(ctx.model)
        model = ctx.model
        if not bind and not model:
            return PreparedStep(model)  # the engine fails "no model configured"
        res = Residency(
            self._transport,
            ctx.provider,
            pid,
            ctx.config.providers[pid],
            component=ctx.component,
            scopes=ctx.scopes,
            run_id=ctx.request.run_id,
            timeout_s=ctx.timeout_s,
        )
        if settings:
            res.check_settings(model or "-", settings)  # parallel and unknown names fail here
        loaded = res.loaded()
        present = bind_loaded(loaded, model) if bind else find_loaded(loaded, model)
        if present is not None:
            self._check_context(pid, present, settings)
            return PreparedStep(present.id, state_label("loaded", present.id))
        assert model is not None
        if not may_load:
            raise ResidencyFailed(
                classify_condition(Condition.STRICT_MODE_MISS),
                f"provider {pid!r}: model {model!r} is not loaded and this mode never loads it",
                state_label("not_loaded", model),
            )
        res.load(model, settings)
        return PreparedStep(model, state_label(f"loaded_{label}", model))

    @staticmethod
    def _check_context(pid: str, present: LoadedModel, settings: Mapping[str, Any]) -> None:
        want = settings.get("context_length")
        if want is not None and present.context_length != want:
            have = "unreported" if present.context_length is None else present.context_length
            raise ResidencyFailed(
                FailureClass.CONFIG,
                f"provider {pid!r}: model {present.id!r} is loaded with context_length {have}, "
                f"configured {want}; it is not reloaded",
                state_label("mismatch", present.id),
            )
