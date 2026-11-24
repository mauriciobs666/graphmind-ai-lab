"""The engine: `Gateway.call` (docs/plans/llm-gateway.md §3.2, §3.3, §3.7, §3.9; FR-1, FR-4,
FR-5, FR-7, FR-11, FR-18, FR-19, AC-3, AC-12, AC-22, AC-23, AC-24).

Order per call: resolve call site and chain (policy) -> library run bound -> per chain step:
residency hook -> provider `build` -> transport `exchange` -> `parse`/`classify` -> one attempt
record -> retry/fallback. Retry and fallback exist only where the call site lists them (`retry`
table; a chain longer than one step) and only on temporary failures; the caller can switch
either off and gets the first failure unmasked.

Shapes this module defines where the plan leaves them open (all in `[call_sites.<name>]`):
  * `retry = {max = N, on = [<failure class>, ...]}` (same step, per step; absent: no retry);
  * `settings` (neutral name -> value), `provider_settings.<provider id>` (per-provider
    overrides) and `optional_settings = [names]`: the settings profile; the request's own
    settings override the profile's per name; every name is required unless listed optional;
  * `timeout_s` (also accepted on the provider table);
  * the model of a step is its `model`, else the provider table's `model`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from llmgateway.config import ConfigSource, GatewayConfig
from llmgateway.failures import (
    HOLDER_REFUSALS,
    REFUSAL_CONTENT_TYPE,
    classify_condition,
    classify_exception,
)
from llmgateway.policy import ScopeMarkerSource, Step, check_run_bound, resolve
from llmgateway.providers import REGISTRY, ProviderRegistry
from llmgateway.providers.base import BuildContext, Provider
from llmgateway.records import Attempt, RecordSink, RecordStore, new_call_id
from llmgateway.transport import Transport, TransportFailure, WireRequest, WireResponse
from llmgateway.types import (
    NEUTRAL_SETTINGS,
    CallFailed,
    EmbeddingsNotAvailable,
    FailureClass,
    HostedFromCallerProcess,
    Request,
    Response,
)

DEFAULT_TIMEOUT_S = 120.0


class AttemptsFailed(CallFailed):
    """Every attempt the call was allowed failed (or one failed permanently). `failure_class` and
    `code` are those of the last attempt; `classes` lists every attempt's class in order."""

    def __init__(
        self, call_id: str, classes: tuple[FailureClass, ...], message: str, code: str | None
    ) -> None:
        super().__init__(classes[-1], message, code)
        self.call_id = call_id
        self.classes = classes


@dataclass(frozen=True, slots=True)
class StepContext:
    """What the residency hook (D6, §3.8) sees for one attempt."""

    config: GatewayConfig
    site: Mapping[str, Any]
    step: Step
    provider: Provider
    request: Request
    model: str | None


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """The hook's answer: the model to call and the loaded-model state to record."""

    model: str | None
    loaded_state: str | None = None


class ResidencyHook(Protocol):
    def prepare(self, ctx: StepContext) -> PreparedStep:
        """May raise `CallFailed` (e.g. `model_not_loaded`); that is the attempt's failure."""


class NoResidency:
    """S6's no-op hook: the step's own model, no loaded state."""

    def prepare(self, ctx: StepContext) -> PreparedStep:
        return PreparedStep(ctx.model)


@dataclass(frozen=True, slots=True)
class _Retry:
    max: int = 0
    on: frozenset[FailureClass] = frozenset()


@dataclass(frozen=True, slots=True)
class _Failure:
    failure_class: FailureClass
    message: str
    code: str | None = None


def _retry_policy(site: Mapping[str, Any]) -> _Retry:
    raw = site.get("retry")
    if raw is None:
        return _Retry()
    try:
        count, on = raw["max"], raw["on"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("max")
        if not isinstance(on, list):
            raise ValueError("on")
        return _Retry(count, frozenset(FailureClass(c) for c in on))
    except (KeyError, TypeError, ValueError) as exc:
        raise CallFailed(
            FailureClass.CONFIG, "call site retry is invalid: want {max = N, on = [classes]}"
        ) from exc


def _effective_settings(
    site: Mapping[str, Any], provider_id: str, request: Request
) -> tuple[dict[str, Any], frozenset[str]]:
    merged: dict[str, Any] = {}
    for layer in (
        site.get("settings"),
        (site.get("provider_settings") or {}).get(provider_id),
        request.settings,
    ):
        if layer is None:
            continue
        if not isinstance(layer, Mapping):
            raise CallFailed(FailureClass.CONFIG, "call site settings must be a table")
        merged.update(layer)
    optional = site.get("optional_settings", [])
    if not isinstance(optional, list):
        raise CallFailed(FailureClass.CONFIG, "optional_settings must be a list")
    unknown = (set(merged) | set(optional)) - NEUTRAL_SETTINGS
    if unknown:
        raise CallFailed(
            FailureClass.CONFIG, f"unknown settings in call site: {', '.join(sorted(unknown))}"
        )
    return merged, frozenset(optional)


def _timeout(site: Mapping[str, Any], provider: Mapping[str, Any]) -> float:
    value = site.get("timeout_s", provider.get("timeout_s", DEFAULT_TIMEOUT_S))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise CallFailed(FailureClass.CONFIG, "timeout_s must be a positive number")
    return float(value)


def _refusal_code(body: bytes) -> str | None:
    try:
        code = json.loads(body).get("holderRefusal")
    except (ValueError, AttributeError):
        return None
    return code if isinstance(code, str) and code in HOLDER_REFUSALS else None


class Gateway:
    """The in-process gateway. `config` is a `ConfigSource` (re-read and owner-checked per call)
    or a fixed `GatewayConfig`."""

    def __init__(
        self,
        config: ConfigSource | GatewayConfig,
        transport: Transport,
        records: RecordSink | RecordStore,
        *,
        registry: ProviderRegistry = REGISTRY,
        marker_source: ScopeMarkerSource | None = None,
        residency: ResidencyHook | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._get_config = config.get if isinstance(config, ConfigSource) else (lambda: config)
        self._transport = transport
        self._records = records
        self._registry = registry
        self._markers = marker_source
        self._residency: ResidencyHook = residency or NoResidency()
        self._clock = clock

    def embed(self, *args: object, **kwargs: object) -> None:
        raise EmbeddingsNotAvailable(
            "embeddings are not available in this gateway release (chat completions only)"
        )

    def call(self, request: Request) -> Response:
        config = self._get_config()
        resolution = resolve(config, request, self._markers)
        if hasattr(self._records, "tokens_used"):  # the default store; custom sinks opt in
            check_run_bound(config, request.run_id, self._records)  # type: ignore[arg-type]
        if resolution.has_hosted and not self._transport.reaches_hosted:
            raise HostedFromCallerProcess(
                f"call site {request.call_site!r} has a hosted step; a hosted call goes only "
                "through the key-holding process, not this transport"
            )
        site = config.call_sites[request.call_site]
        retry = _retry_policy(site)
        steps = resolution.chain if request.options.fallback else resolution.chain[:1]
        call_id = new_call_id()
        classes: list[FailureClass] = []
        kind: str = "first"
        reason: FailureClass | None = None
        last: _Failure | None = None
        for step in steps:
            retries_left = retry.max if request.options.retry else 0
            while True:
                number = len(classes) + 1
                result = self._attempt(
                    config, site, resolution, step, request, call_id, number, kind, reason
                )
                if isinstance(result, Response):
                    return result
                last = result
                classes.append(result.failure_class)
                fc = result.failure_class
                if fc.temporary and retries_left > 0 and fc in retry.on:
                    retries_left -= 1
                    kind, reason = "retry", fc
                    continue
                break
            if not fc.temporary:
                break
            kind, reason = "fallback", fc
        assert last is not None
        raise AttemptsFailed(call_id, tuple(classes), last.message, last.code)

    def _attempt(
        self,
        config: GatewayConfig,
        site: Mapping[str, Any],
        resolution: Any,
        step: Step,
        request: Request,
        call_id: str,
        number: int,
        kind: str,
        reason: FailureClass | None,
    ) -> Response | _Failure:
        started = self._clock()
        model = step.model or config.providers[step.provider].get("model")
        loaded_state: str | None = None
        outcome: Response | _Failure
        try:
            provider = self._registry.get(config.providers[step.provider].get("kind", ""))
            prepared = self._residency.prepare(
                StepContext(config, site, step, provider, request, model)
            )
            model, loaded_state = prepared.model, prepared.loaded_state
            outcome = self._send(config, site, resolution, step, provider, request, call_id, model)
        except CallFailed as exc:
            outcome = _Failure(exc.failure_class, str(exc), exc.code)
        duration_ms = max(0, int((self._clock() - started) * 1000))
        self._record(
            resolution, step, request, call_id, number, kind, reason, outcome, model,
            loaded_state, duration_ms,
        )  # fmt: skip
        return outcome

    def _send(
        self,
        config: GatewayConfig,
        site: Mapping[str, Any],
        resolution: Any,
        step: Step,
        provider: Provider,
        request: Request,
        call_id: str,
        model: str | None,
    ) -> Response | _Failure:
        pconf = config.providers[step.provider]
        if not model:
            raise CallFailed(
                FailureClass.CONFIG, f"no model configured for provider {step.provider!r}"
            )
        settings, optional = _effective_settings(site, step.provider, request)
        ctx = BuildContext(
            provider_id=step.provider,
            config=pconf,
            model=model,
            messages=request.messages,
            settings=settings,
            optional=optional,
            tools=request.tools,
            response_format=request.response_format,
        )
        built = provider.build(ctx)
        wire = WireRequest(
            provider_id=step.provider,
            kind=provider.kind,
            base_url=str(pconf.get("base_url", "")),
            path=built.path,
            body=built.body,
            hosted=step.hosted,
            method=built.method,
            headers=built.headers,
            run_id=request.run_id,
            component=resolution.component,
            scopes=resolution.scopes,
            timeout_s=_timeout(site, pconf),
        )
        try:
            reply = self._transport.exchange(wire)
        except HostedFromCallerProcess:
            raise
        except CallFailed:
            raise
        except TransportFailure as exc:
            return _Failure(classify_condition(exc.condition), f"{step.provider}: {exc}")
        except Exception as exc:  # noqa: BLE001 - classify_exception owns the mapping
            fc = classify_exception(exc)
            return _Failure(fc, f"{step.provider}: {type(exc).__name__}")
        return self._interpret(provider, ctx, reply, call_id, step)

    def _interpret(
        self, provider: Provider, ctx: BuildContext, reply: WireResponse, call_id: str, step: Step
    ) -> Response | _Failure:
        if not 200 <= reply.status < 300:
            if reply.status < 400:
                fc = FailureClass.REJECTED
            else:
                fc = provider.classify(reply.status, reply.content_type, reply.body, ctx.config)
            code = (
                _refusal_code(reply.body)
                if (reply.content_type or "").split(";", 1)[0].strip().lower()
                == REFUSAL_CONTENT_TYPE
                else None
            )
            suffix = f" ({code})" if code else ""
            return _Failure(fc, f"{step.provider}: {fc.value}, HTTP {reply.status}{suffix}", code)
        try:
            parsed = provider.parse(reply.body, ctx)
        except (ValueError, KeyError, TypeError, AttributeError):
            return _Failure(
                FailureClass.CONFIG, f"{step.provider}: reply is not a valid {provider.kind} reply"
            )
        return Response(
            call_id=call_id,
            provider=step.provider,
            model=parsed.model,
            hosted=step.hosted,
            message=parsed.message,
            usage=parsed.usage,
            provider_detail=parsed.detail,
            raw=parsed.raw,
        )

    def _record(
        self,
        resolution: Any,
        step: Step,
        request: Request,
        call_id: str,
        number: int,
        kind: str,
        reason: FailureClass | None,
        outcome: Response | _Failure,
        model: str | None,
        loaded_state: str | None,
        duration_ms: int,
    ) -> None:
        ok = isinstance(outcome, Response)
        text = resolution.record_text
        self._records.write(
            Attempt(
                call_id=call_id,
                attempt=number,
                component=resolution.component,
                scopes=resolution.scopes,
                call_site=resolution.call_site,
                provider=step.provider,
                model=outcome.model if isinstance(outcome, Response) else (model or "unknown"),
                hosted=step.hosted,
                outcome="ok" if ok else "failed",
                run_id=request.run_id,
                kind=kind,
                reason=reason,
                input_tokens=outcome.usage.input_tokens if isinstance(outcome, Response) else None,
                output_tokens=(
                    outcome.usage.output_tokens if isinstance(outcome, Response) else None
                ),
                duration_ms=duration_ms,
                failure_class=None if isinstance(outcome, Response) else outcome.failure_class,
                loaded_state=loaded_state,
                request_text=json.dumps(list(request.messages)) if text else None,
                response_text=(
                    json.dumps(outcome.message) if text and isinstance(outcome, Response) else None
                ),
            ),
            text_allowed=text,
        )
