"""Call-time scope policy: chain resolution and the hosted-use refusals
(docs/plans/llm-gateway.md §3.6; FR-3, FR-6, FR-19 bound, AC-2, AC-4, AC-21).

`resolve` is pure over (configuration, request, marker source): it reads the markers afresh on
every call (no cache) and returns the chain the engine may walk, or raises `PolicyRefused`. It
never filters a hosted step out of a chain silently: a chain with a hosted step is either allowed
whole or refused (or replaced whole by the call site's `local_chain`).

Shapes this module defines where the plan leaves them open:
  * a chain step is a provider id string, or a mapping with a `provider` key (optional `model`);
  * a `[[scope_overrides]]` entry is `{scope, chain[, call_site]}`; overrides are tried in file
    order and the first whose `scope` is among the effective scopes (and whose `call_site`, when
    given, matches) supplies the chain.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmgateway.config import MARKERS, SCOPE_ID, GatewayConfig
from llmgateway.records import RecordStore
from llmgateway.types import CallFailed, FailureClass, Request

RESTRICTED = "restricted"
UNRESTRICTED = "unrestricted"


class PolicyRefused(CallFailed):
    """The call is refused before anything is sent. `reason` is a stable machine code."""

    def __init__(
        self,
        reason: str,
        message: str,
        failure_class: FailureClass = FailureClass.POLICY_REFUSED,
    ) -> None:
        super().__init__(failure_class, message)
        self.reason = reason


class ScopeMarkerSource(Protocol):
    """Answers a source-backed scope's marker, read at call time and never cached by the
    gateway (ccg's adapter reads the installed manifest). It raises, or returns anything but
    `"restricted"`/`"unrestricted"`, when it cannot say: that refuses a hosted call."""

    def marker(self, scope_id: str) -> str: ...


@dataclass(frozen=True, slots=True)
class Step:
    provider: str
    hosted: bool
    model: str | None = None


@dataclass(frozen=True, slots=True)
class Resolution:
    call_site: str
    component: str
    scopes: tuple[str, ...]
    chain: tuple[Step, ...]
    source: str  # "override" | "default" | "local_chain"
    markers: Mapping[str, str | None] = field(default_factory=dict)  # None: unreadable
    record_text: bool = False

    @property
    def has_hosted(self) -> bool:
        return any(s.hosted for s in self.chain)


def valid_scope_id(scope_id: object) -> bool:
    return isinstance(scope_id, str) and SCOPE_ID.fullmatch(scope_id) is not None


def _source_prefix(config: GatewayConfig, scope_id: str) -> str | None:
    for src in config.scope_sources.values():
        prefix = src.get("prefix")
        if isinstance(prefix, str) and prefix and scope_id.startswith(prefix):
            return prefix
    return None


def is_registered(config: GatewayConfig, scope_id: str) -> bool:
    """Registered statically, or under a recognised `[scope_sources]` prefix (§3.6)."""
    return scope_id in config.scopes or _source_prefix(config, scope_id) is not None


def read_marker(
    config: GatewayConfig, scope_id: str, source: ScopeMarkerSource | None
) -> str | None:
    """The scope's marker now, or None when unreadable/ill-typed. A registered scope with no
    marker is `unrestricted`. Not cached: every call reads again."""
    if _source_prefix(config, scope_id) is not None:
        if scope_id in config.scopes and "marker" in config.scopes[scope_id]:
            return None  # a static marker under a source prefix: the loader refuses it too
        if source is None:
            return None
        try:
            value = source.marker(scope_id)
        except Exception:  # noqa: BLE001 - any failure of the source means "unreadable"
            return None
        return value if isinstance(value, str) and value in MARKERS else None
    scope = config.scopes.get(scope_id)
    if scope is None:
        return None
    if "marker" not in scope:
        return UNRESTRICTED
    value = scope["marker"]
    return value if isinstance(value, str) and value in MARKERS else None


def _refuse(reason: str, message: str) -> PolicyRefused:
    return PolicyRefused(reason, message)


def _step(config: GatewayConfig, raw: object, where: str) -> Step:
    provider = raw if isinstance(raw, str) else None
    model = None
    if isinstance(raw, Mapping):
        provider = raw.get("provider") if isinstance(raw.get("provider"), str) else None
        model = raw.get("model") if isinstance(raw.get("model"), str) else None
    if provider is None or provider not in config.providers:
        raise PolicyRefused(
            "unknown_provider",
            f"{where}: step {raw!r} does not name a configured provider",
            FailureClass.CONFIG,
        )
    return Step(provider, config.provider_is_hosted(provider), model)


def _steps(config: GatewayConfig, raw: object, where: str) -> tuple[Step, ...]:
    if not isinstance(raw, list) or not raw:
        raise PolicyRefused(
            "empty_chain", f"{where}: no usable chain configured", FailureClass.CONFIG
        )
    return tuple(_step(config, r, where) for r in raw)


def _override_chain(config: GatewayConfig, call_site: str, scopes: tuple[str, ...]) -> object:
    for entry in config.scope_overrides:
        site = entry.get("call_site")
        if site is not None and site != call_site:
            continue
        if entry.get("scope") in scopes:
            # A matching entry without a usable `chain` is a config error (`_steps` refuses an
            # empty or non-list chain), never a fall-through to the default chain.
            return entry.get("chain", [])
    return None


def resolve(
    config: GatewayConfig, request: Request, source: ScopeMarkerSource | None = None
) -> Resolution:
    """Pick the chain for `request` or refuse (§3.6)."""
    site = config.call_sites.get(request.call_site)
    if site is None:
        raise PolicyRefused(
            "unknown_call_site",
            f"call site {request.call_site!r} is not configured",
            FailureClass.CONFIG,
        )
    bound = site.get("scopes", [])
    bound = tuple(bound) if isinstance(bound, list) else ()
    # A request can only add: the effective set is a superset of the call site's bound scopes.
    scopes = tuple(dict.fromkeys((*bound, *request.scopes)))
    for scope_id in scopes:
        if not valid_scope_id(scope_id):
            raise _refuse("scope_malformed", f"scope id {scope_id!r} is not [a-z0-9:._-]{{1,128}}")

    override = _override_chain(config, request.call_site, scopes)
    if override is not None:
        chain, origin = _steps(config, override, f"override for {request.call_site!r}"), "override"
    else:
        chain = _steps(config, site.get("chain"), f"call site {request.call_site!r}")
        origin = "default"

    markers: dict[str, str | None] = {s: read_marker(config, s, source) for s in scopes}

    if any(s.hosted for s in chain):
        chain, origin = _gate_hosted(
            config, request.call_site, scopes, markers, chain, origin, site
        )

    return Resolution(
        call_site=request.call_site,
        component=str(site.get("component", request.call_site)),
        scopes=scopes,
        chain=chain,
        source=origin,
        markers=markers,
        record_text=_record_text(config, scopes, markers),
    )


def _gate_hosted(
    config: GatewayConfig,
    call_site: str,
    scopes: tuple[str, ...],
    markers: Mapping[str, str | None],
    chain: tuple[Step, ...],
    origin: str,
    site: Mapping[str, Any],
) -> tuple[tuple[Step, ...], str]:
    if not config.hosted_enabled:
        raise _refuse("hosted_disabled", "hosted use is disabled (configuration is unprotected)")
    if not scopes:
        raise _refuse("scope_required", "hosted use requires a registered scope")
    for scope_id in scopes:
        if not is_registered(config, scope_id):
            raise _refuse("scope_unregistered", f"scope {scope_id!r} is not registered")
    for scope_id in scopes:
        if markers[scope_id] is None:
            raise _refuse("marker_unreadable", f"marker of scope {scope_id!r} is unreadable")
    restricted = [s for s in scopes if markers[s] == RESTRICTED]
    if not restricted:
        return chain, origin
    if origin == "override":
        raise _refuse(
            "restricted_hosted_override",
            f"restricted scope {restricted[0]!r} cannot use a hosted override",
        )
    local = site.get("local_chain")
    if local is None:
        raise _refuse(
            "restricted_hosted",
            f"restricted scope {restricted[0]!r} cannot use the hosted chain of {call_site!r}",
        )
    steps = _steps(config, local, f"local_chain of {call_site!r}")
    if any(s.hosted for s in steps):
        raise _refuse("local_chain_hosted", f"local_chain of {call_site!r} names a hosted step")
    return steps, "local_chain"


def _record_text(
    config: GatewayConfig, scopes: tuple[str, ...], markers: Mapping[str, str | None]
) -> bool:
    """Text recording needs every bound scope `record_text` and none restricted; an unreadable
    marker or an unregistered scope means no text (never a refusal: nothing leaves)."""
    if not scopes:
        return False
    for scope_id in scopes:
        scope = config.scopes.get(scope_id)
        if scope is None or scope.get("record_text") is not True:
            return False
        if markers.get(scope_id) != UNRESTRICTED:
            return False
    return True


def check_run_bound(config: GatewayConfig, run_id: str | None, records: RecordStore) -> None:
    """Library-side early check (§3.9): a run matching a `run_bounds` entry (first match wins)
    whose recorded tokens reached its `max_tokens` is refused `run_bound_reached`. Runs matching
    no entry, and calls without a run id, are unbounded here (the holder bounds hosted ones)."""
    if run_id is None:
        return
    for entry in config.run_bounds:
        pattern, limit = entry.get("match"), entry.get("max_tokens")
        if not isinstance(pattern, str) or not fnmatch.fnmatchcase(run_id, pattern):
            continue
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise PolicyRefused(
                "run_bound_invalid",
                f"run_bounds {pattern!r}: max_tokens invalid",
                FailureClass.CONFIG,
            )
        if records.tokens_used(run_id) >= limit:
            raise PolicyRefused(
                "run_bound_reached",
                f"run {run_id!r} reached its token bound",
                FailureClass.RUN_BOUND,
            )
        return
