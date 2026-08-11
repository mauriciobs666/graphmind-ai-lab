"""The model-resolution seam (K-042 Landing 1, FR-1..FR-6/FR-11..FR-15/FR-20).

Every LLM/embedding consumer in falkor-chat holds a `ModelGateway` and asks it for a
client — `self._models.llm(kind, ...)` / `.embedder(kind, ...)` — instead of
constructing `OpenAICompatibleLLM`/`OpenAICompatibleEmbedder` itself (FR-4, enforced
by an AST check in `test_modelconfig.py`: no module outside this one and `tests/`
constructs those clients directly).

Two hand-edited files feed the gateway:

  * the **shared** file (`FALKORCHAT_OPENCODE_CONFIG`) — a pristine, unmodified
    OpenCode `opencode.json`. Only its `provider.*` subtree is read (§4.2); falkor-chat
    writes nothing to it, ever.
  * falkor-chat's own **overlay** (`FALKORCHAT_MODEL_CONFIG`, default
    `<falkor-chat>/config/models.json`) — per-kind defaults, per-model settings,
    per-provider overrides, and (reserved, not yet honoured) `roles`/`agents`.

`ModelGateway.resolve(kind, *, requested=None, ws=None, overrides=None)` turns a ref
(`"<provider>/<model-id>"`) into a `Resolution` (`.primary` — a `ResolvedModel`; a
`Resolution.chain` of length 1 in Landing 1, the FR-18 fallback seam). `.llm(...)`/
`.embedder(...)` build the transport-backed client directly.

A directly-injected client (the `llm=`/`embedder=` constructor kwargs every consumer
still accepts) is sugar `__init__` wraps into a `StaticModelGateway` — dependency
injection for tests, never a configuration route (§3, A-4).

Landing-2 status (§6.1): a ref with **no** `/` resolves as a **role name** (FR-7) —
`Overlay.roles[name]` — to an ordered, settings-applied fallback chain (FR-18): on a
`ProviderCallError` from one chain element, `FallbackClient` (`llm.py`) tries the next,
naming every model tried if all fail. The answering model (and whether it came from a
fallback) travels on the return value, `ChatResult.model`/`.fallback` — never on any
client-held state. The overlay's `agents` key is parsed and logged, not honoured (a
separate, not-yet-built unit).

`resolve()` now implements FR-16/FR-17's real precedence (L2-3), first-match-wins:
**workspace override → the caller's own `requested=` choice → the per-kind default**.
The workspace override is a **hard cap** — it wins even over an explicit `requested=`
choice, and it is resolved through the *same* role-expansion / provider-lookup path as
any other ref, so a failing workspace override fails loudly (`ModelResolutionError`,
FR-10) rather than silently falling through to a lower rung. `Resolution.source` names
which rung won: `"workspace"` | `"requested"` | `"default"` — callers that persist a
rung name (the executor's `StepRun.modelSource`) translate this generic vocabulary into
their own terms locally; this module stays kind-and-caller-agnostic.

Two ways a caller supplies the workspace override, matched to the two shapes `-graph.md`
§2.6 identifies: a caller that already did one per-drive read (the executor, via
`run["modelOverrides"]`) passes the pre-fetched `{agentModel, guardModel,
embeddingModel, responderModel}` dict as `overrides=`, so `resolve()` never re-reads the
graph; a caller with no such per-drive read (the `@mention` responder, the embedding
worker) passes `ws=` and `resolve()` reads fresh via the injected `WorkspaceOverrides`
port (`GraphWorkspaceOverrides` in production, `NullWorkspaceOverrides` — Landing 1's
no-op — by default). `_KIND_TO_OVERRIDE_KEY` is the crosswalk between this module's
`kind` strings and `-graph.md`'s `WorkspaceConfig` property names, which do **not**
match 1:1 by name (see its own docstring).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

_log = logging.getLogger(__name__)

# The only protocol Landing 1 implements (§4.7) — a resolution naming any other
# protocol (e.g. `anthropic`) fails loudly at load time rather than sending a
# wrong-shaped payload. A native Anthropic Messages client is future work (K-043).
SUPPORTED_PROTOCOLS: frozenset[str] = frozenset({"openai"})

# Per-kind timeout floor (m-2, §9.2): `embedding` is short/predictable and on the hot
# path; `agent`/`step`/`guard` get headroom for a slower reasoning turn. Used only when
# neither the overlay's `timeouts.<kind>` nor a per-model `timeout` says otherwise —
# the shipped `config/models.json` sets these explicitly, so this is a defensive floor
# for a minimal/hand-written overlay (e.g. a test fixture).
_DEFAULT_KIND_TIMEOUTS: dict[str, float] = {
    "agent": 180.0, "step": 180.0, "guard": 180.0, "embedding": 30.0,
}
_DEFAULT_TIMEOUT_FALLBACK = 180.0

# The four closed consumer kinds (§3.1). Adding a fifth means adding its own override
# property (FR-17's hard cap) — not a change to this set casually.
KINDS: frozenset[str] = frozenset({"agent", "step", "embedding", "guard"})

# The crosswalk between this module's `kind` strings and `-graph.md` §2.2's
# `WorkspaceConfig` property names (surfaced by `repository.read_model_overrides` as
# `{agentModel, guardModel, embeddingModel, responderModel}`, §6.1). **These do NOT
# match 1:1 by name** — verified against `-graph.md` §8.4's own worked example
# ("the real requirement behind 'this workspace's assistant should use a different
# model' is FR-16, and §2.2's `responderModelOverride` covers it"): `-graph.md`'s
# "responder" is this module's kind `"agent"` (`AgentResponder`, the `@mention`
# responder — plan §3.1's "agent" consumer row); by elimination `-graph.md`'s "agent"
# (its property `agentModelOverride`) is this module's kind `"step"` (the workflow
# engine's `type:'agent'` node, plan §3.1's "step" consumer row). `-graph.md` names its
# properties after the WORKFLOW NODE TYPE ("agent") vs. the chat RESPONDER CLASS
# ("responder") — a distinction this module's own `kind` enum does not make the same
# way. `guard`/`embedding` match by name on both sides, no crosswalk needed.
_KIND_TO_OVERRIDE_KEY: dict[str, str] = {
    "agent": "responderModel",
    "step": "agentModel",
    "guard": "guardModel",
    "embedding": "embeddingModel",
}

_ENV_RE = re.compile(r"\{env:([^{}]+)\}")
_FILE_RE = re.compile(r"\{file:([^{}]+)\}")
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")

# Per-model settings keys with a defined meaning to the gateway itself — reserved so
# they never leak into the request-payload passthrough (L1-2 passthrough rule, m-3).
_RESERVED_MODEL_SETTING_KEYS = frozenset({"timeout", "dim", "protocol"})

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../falkor-chat
DEFAULT_MODEL_CONFIG_PATH = _REPO_ROOT / "config" / "models.json"


class ModelConfigError(Exception):
    """The configuration itself is broken: unparseable/unreadable file, an invalid
    `baseURL`, an unresolved `{env:}`/`{file:}` substitution, an unsupported protocol,
    or a model reference with no `/` (the Landing-2 role namespace)."""


class ModelResolutionError(Exception):
    """A specific `resolve()`/`.llm()`/`.embedder()` call could not produce a client:
    no default configured for a kind and none was requested, an unknown provider, or
    no client is available (an empty `StaticModelGateway` sugar wrapper) — FR-9/FR-10's
    "unresolvable at use time" failure."""


class Secret:
    """Wraps a credential so it can never leak into a log line, trace payload, error
    message, or `repr()` — the only way to read the real value is `.get()` (§4.8)."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def __repr__(self) -> str:  # pragma: no cover — trivial
        return "Secret(***)"

    def __str__(self) -> str:  # pragma: no cover — trivial
        return "***"


# ── value objects ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProviderSpec:
    """One provider, fully resolved: shared-file `options.*` merged with the
    overlay's `providers.<id>` override, `{env:}`/`{file:}`-substituted, and
    baseURL-normalized (§4.3). `repr()` never exposes `api_key`/`headers` values."""

    provider_id: str
    base_url: str
    declared_base_url: str
    resolution_source: str  # "rule" | "override" | "verbatim"
    api_key: Secret | None
    headers: dict[str, str]
    protocol: str
    name: str | None = None

    def __repr__(self) -> str:
        return (
            f"ProviderSpec(provider_id={self.provider_id!r}, base_url={self.base_url!r}, "
            f"protocol={self.protocol!r}, resolution_source={self.resolution_source!r})"
        )


@dataclass(frozen=True)
class ResolvedModel:
    """One fully-resolved model: enough to build a transport-backed client. `ref` is
    exactly the string FR-8 persists as `StepRun.resolvedModel` in Landing 2. `repr()`
    never exposes `api_key`/`headers` values (§4.8 secret hygiene)."""

    ref: str
    provider: str
    model: str
    base_url: str
    api_key: Secret | None
    headers: dict[str, str]
    protocol: str
    timeout: float
    params: dict[str, Any]
    dim: int | None = None

    def __repr__(self) -> str:
        return (
            f"ResolvedModel(ref={self.ref!r}, base_url={self.base_url!r}, "
            f"protocol={self.protocol!r}, timeout={self.timeout!r})"
        )


@dataclass(frozen=True)
class Resolution:
    """The outcome of one `resolve()` call. `chain` is a tuple — length 1 for a direct
    `provider/model` ref, potentially multi-element for a role-backed ref (FR-7/FR-18,
    Landing 2): an ordered, settings-applied fallback chain, ordered first-to-last-try.

    `source` (L2-3, FR-16/FR-17) names which precedence rung produced `chain`:
    `"workspace"` — a workspace override (the hard cap) won, even over an explicit
    caller choice; `"requested"` — the caller's own `requested=` ref won, no override
    was in effect; `"default"` — neither an override nor a caller choice was given, the
    per-kind default won. This is deliberately the generic three-rung vocabulary, not
    any one caller's persisted terminology (e.g. `StepRun.modelSource`'s `'step'` for
    the middle rung) — callers that persist a rung name translate locally."""

    kind: str
    chain: tuple[ResolvedModel, ...]
    source: str

    @property
    def primary(self) -> ResolvedModel:
        return self.chain[0]


@dataclass(frozen=True)
class RoleSpec:
    """One `roles.<name>` overlay entry (FR-7): an ordered fallback chain of
    `<provider>/<model-id>` refs (never another role name — rejected at load, §Overlay),
    plus an optional role-level `timeout` fallback. Precedence for an element's timeout:
    its own `models.<ref>.timeout` first, else this role's `timeout`, else the kind
    default (`ModelGateway._resolve_element`)."""

    models: tuple[str, ...]
    timeout: float | None = None


class WorkspaceOverrides(Protocol):
    """FR-16/FR-17 seam (Landing 2): a per-kind workspace override read. Threaded from
    every `resolve()` call from day one (§6.1 item 2) so Landing 2 swaps the null port
    below for a graph-backed one with no call-site changes."""

    def get(self, ws: str, kind: str) -> str | None: ...


class NullWorkspaceOverrides:
    """The Landing-1 no-op `WorkspaceOverrides` — every workspace has no override."""

    def get(self, ws: str, kind: str) -> str | None:
        return None


class GraphWorkspaceOverrides:
    """The Landing-2 `WorkspaceOverrides` implementation (FR-16/FR-17): reads the
    per-workspace `WorkspaceConfig` singleton via the injected repository's
    `read_model_overrides(ws)` (`-graph.md` §2.5/§6.1).

    `repo` is duck-typed (`Any`, not imported as `Repository`) so `modelconfig.py`
    never depends on `repository.py` — the two only ever meet at the wiring site
    (`app.py`), keeping the resolver importable with no graph driver at all (the
    offline-import guarantee `test_importing_modelconfig_touches_no_file_and_no_network`
    pins).

    Used only on the two per-call-read paths that have no `run` dict to carry a
    pre-fetched `overrides=` (§2.6) — the `@mention` responder and the embedding
    worker/retrieval tool. The executor and the guard judge instead read once per
    drive (`executor._drive`) and pass the pre-fetched dict as `overrides=`, so this
    class's `.get()` is never called from either of those two paths.
    """

    def __init__(self, repo: Any) -> None:
        self._repo = repo

    def get(self, ws: str, kind: str) -> str | None:
        key = _KIND_TO_OVERRIDE_KEY.get(kind)
        if key is None:  # pragma: no cover — kind outside the closed KINDS set
            return None
        value = self._repo.read_model_overrides(ws).get(key)
        return value if isinstance(value, str) and value else None


# ── {env:}/{file:} substitution (FR-12, §4.8) ──────────────────────────────────────

def _substitute(value: str, *, source: str) -> str:
    """Replace every `{env:NAME}`/`{file:PATH}` occurrence embedded in `value`
    (in place, not just whole-value). An unresolved reference is a startup error
    naming the variable/path and `source` (the file it appeared in) — never a silent
    empty string (§4.8: a blank `Authorization: Bearer ` reaches a cloud provider as
    a 401 with no local diagnosis)."""

    def _env_sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in os.environ:
            raise ModelConfigError(
                f"{source}: environment variable {name!r} referenced via "
                f"{{env:{name}}} is not set"
            )
        return os.environ[name]

    def _file_sub(m: re.Match[str]) -> str:
        raw_path = m.group(1)
        path = Path(raw_path).expanduser()
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ModelConfigError(
                f"{source}: file {raw_path!r} referenced via {{file:{raw_path}}} "
                f"could not be read: {exc}"
            ) from exc

    value = _ENV_RE.sub(_env_sub, value)
    return _FILE_RE.sub(_file_sub, value)


# ── the /v1 normalization rule (AC-1, §4.3) ────────────────────────────────────────

def _validate_url(raw: str, *, provider_id: str, source: str) -> None:
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ModelConfigError(
            f"provider {provider_id!r} ({source}): invalid baseURL {raw!r} — must "
            f"include an http(s) scheme and host, e.g. 'http://host:1234'"
        )


def _normalize_base_url(raw: str, *, provider_id: str, source: str) -> tuple[str, str]:
    """The §4.3 rule, in three ordered steps: validate, strip (all trailing `/`),
    then normalize (append `/v1` only when the path is now empty). Order matters —
    stripping AFTER normalizing turns an empty-path URL into a double slash."""
    _validate_url(raw, provider_id=provider_id, source=source)
    stripped = raw.rstrip("/")
    if urlparse(stripped).path == "":
        return f"{stripped}/v1", "rule"
    return stripped, "verbatim"


def _override_base_url(raw: str, *, provider_id: str, source: str) -> str:
    """The overlay `providers.<id>.baseURL` escape hatch (§4.3): wins outright over
    the shared file AND the /v1 rule — used exactly as declared (only validated +
    trailing-slash-stripped), never auto-suffixed."""
    _validate_url(raw, provider_id=provider_id, source=source)
    return raw.rstrip("/")


def _infer_protocol(npm: Any) -> str:
    if isinstance(npm, str) and "anthropic" in npm:
        return "anthropic"
    return "openai"


def _camel_to_snake(key: str) -> str:
    return _CAMEL_RE.sub("_", key).lower()


def _read_json(path: str, *, var_name: str, example_path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelConfigError(
            f"{var_name}={path!r} could not be read ({exc}); see {example_path} "
            f"for the expected shape"
        ) from exc
    try:
        doc = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ModelConfigError(f"{var_name}={path!r} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ModelConfigError(f"{var_name}={path!r} must be a JSON object")
    return doc


# ── the shared OpenCode file (FR-1/FR-2/FR-11) ─────────────────────────────────────

class ProviderCatalog:
    """The parsed `provider.*` subtree of the shared, pristine OpenCode file. Every
    other top-level key (`agent`, `mcp`, `$schema`, `theme`, …) is ignored without
    comment (§4.2) — falkor-chat writes nothing to this file, ever."""

    def __init__(self, providers: dict[str, dict[str, Any]], *, path: str) -> None:
        self._providers = providers
        self.path = path

    @classmethod
    def load(cls, path: str) -> "ProviderCatalog":
        doc = _read_json(
            path, var_name="FALKORCHAT_OPENCODE_CONFIG",
            example_path="config/opencode.example.json",
        )
        raw_providers = doc.get("provider")
        providers = raw_providers if isinstance(raw_providers, dict) else {}
        return cls(providers, path=path)

    def provider_ids(self) -> list[str]:
        return list(self._providers)

    def raw(self, provider_id: str) -> dict[str, Any] | None:
        entry = self._providers.get(provider_id)
        return entry if isinstance(entry, dict) else None


# ── the falkor-chat overlay (FR-11, defaults/models/roles) ─────────────────────────

class Overlay:
    """The parsed falkor-chat overlay file: per-kind `defaults`, per-kind `timeouts`,
    per-model `models.<ref>` settings, per-provider `providers.<id>` overrides, `roles`
    (FR-7, Landing 2 — honoured), and (still reserved, Landing 2's `agents[<agentId>]`
    resolution is a separate, not-yet-built unit) `agents`. Unknown top-level keys are
    accepted and logged, never rejected (§4.1) — an admin's Landing-2-ready file never
    fails a Landing-1 build."""

    _KNOWN_KEYS = frozenset(
        {"defaults", "timeouts", "models", "providers", "roles", "agents"}
    )

    def __init__(self, doc: dict[str, Any], *, path: str) -> None:
        self.path = path
        self.defaults = _as_dict(doc.get("defaults"))
        self.timeouts = _as_dict(doc.get("timeouts"))
        self.models = _as_dict(doc.get("models"))
        self.providers = _as_dict(doc.get("providers"))
        self.roles = _build_roles(_as_dict(doc.get("roles")), path=path)

        if "agents" in doc:
            _log.info(
                "overlay %s declares 'agents' — reserved, parsed, not honoured until "
                "Landing 2 (§4.6)", path,
            )
        unknown = sorted(set(doc) - self._KNOWN_KEYS)
        if unknown:
            _log.info(
                "overlay %s has unrecognized top-level key(s) %s — accepted, ignored",
                path, unknown,
            )

    @classmethod
    def load(cls, path: str) -> "Overlay":
        doc = _read_json(
            path, var_name="FALKORCHAT_MODEL_CONFIG", example_path="config/models.json"
        )
        return cls(doc, path=path)

    def default_for(self, kind: str) -> str | None:
        value = self.defaults.get(kind)
        return value if isinstance(value, str) and value else None

    def timeout_for_kind(self, kind: str) -> float:
        value = self.timeouts.get(kind)
        if isinstance(value, (int, float)):
            return float(value)
        return _DEFAULT_KIND_TIMEOUTS.get(kind, _DEFAULT_TIMEOUT_FALLBACK)

    def model_settings(self, ref: str) -> dict[str, Any]:
        return _as_dict(self.models.get(ref))

    def provider_override(self, provider_id: str) -> dict[str, Any]:
        return _as_dict(self.providers.get(provider_id))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# ── roles: ordered fallback chains (FR-7/FR-18, §7 L2-1) ──────────────────────────

def _build_roles(raw_roles: dict[str, Any], *, path: str) -> dict[str, RoleSpec]:
    """Parse + validate the overlay's `roles` map at **load** time (not deferred to
    first use): a role name must not contain `/` (that syntax is reserved for a direct
    `provider/model` ref), each role must declare a non-empty `models` list, and every
    element of that list must itself be a direct `provider/model` ref — a chain element
    with no `/` would be another role name, and role-resolves-to-role is rejected here
    rather than left to recurse at resolve time."""
    roles: dict[str, RoleSpec] = {}
    for name, raw_spec in raw_roles.items():
        if "/" in name:
            raise ModelConfigError(
                f"{path}: role name {name!r} must not contain '/' — that syntax is "
                f"reserved for a direct '<provider>/<model-id>' reference"
            )
        spec = _as_dict(raw_spec)
        models = spec.get("models")
        if not isinstance(models, list) or not models or not all(
            isinstance(m, str) and m for m in models
        ):
            raise ModelConfigError(
                f"{path}: role {name!r} must declare a non-empty 'models' list of "
                f"'<provider>/<model-id>' strings"
            )
        for m in models:
            if "/" not in m:
                raise ModelConfigError(
                    f"{path}: role {name!r} chain element {m!r} has no provider "
                    f"prefix — a role cannot resolve to another role"
                )
        raw_timeout = spec.get("timeout")
        timeout = float(raw_timeout) if isinstance(raw_timeout, (int, float)) else None
        roles[name] = RoleSpec(models=tuple(models), timeout=timeout)
    return roles


# ── provider resolution (shared file + overlay override, merged) ──────────────────

def _build_provider_spec(
    provider_id: str, catalog: ProviderCatalog, overlay: Overlay
) -> ProviderSpec:
    raw = catalog.raw(provider_id) or {}
    override = overlay.provider_override(provider_id)
    options = _as_dict(raw.get("options"))

    override_base = override.get("baseURL")
    if isinstance(override_base, str) and override_base:
        declared = _substitute(override_base, source=overlay.path)
        base_url = _override_base_url(declared, provider_id=provider_id, source=overlay.path)
        source = "override"
    else:
        raw_base = options.get("baseURL")
        if not isinstance(raw_base, str) or not raw_base:
            raise ModelConfigError(
                f"provider {provider_id!r} has no options.baseURL in "
                f"{catalog.path} and no overlay override in {overlay.path}"
            )
        declared = _substitute(raw_base, source=catalog.path)
        base_url, source = _normalize_base_url(
            declared, provider_id=provider_id, source=catalog.path
        )

    api_key: Secret | None = None
    raw_key = options.get("apiKey")
    if isinstance(raw_key, str) and raw_key:
        api_key = Secret(_substitute(raw_key, source=catalog.path))

    headers: dict[str, str] = {}
    raw_headers = options.get("headers")
    if isinstance(raw_headers, dict):
        for k, v in raw_headers.items():
            headers[k] = _substitute(v, source=catalog.path) if isinstance(v, str) else v

    protocol = override.get("protocol") or _infer_protocol(raw.get("npm"))
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ModelConfigError(
            f"provider {provider_id!r}: protocol {protocol!r} is not implemented in "
            f"this build (only 'openai' is supported — plan §4.7); a native "
            f"Anthropic Messages client is future work (K-043 if wanted)"
        )

    _log.info(
        "model provider %s: baseURL %s -> %s (%s)",
        provider_id, declared, base_url, source,
    )
    name = raw.get("name")
    return ProviderSpec(
        provider_id=provider_id, base_url=base_url, declared_base_url=declared,
        resolution_source=source, api_key=api_key, headers=headers, protocol=protocol,
        name=name if isinstance(name, str) else None,
    )


def _build_providers(
    catalog: ProviderCatalog, overlay: Overlay
) -> dict[str, ProviderSpec]:
    ids = set(catalog.provider_ids()) | set(overlay.providers)
    return {pid: _build_provider_spec(pid, catalog, overlay) for pid in sorted(ids)}


# ── clients (deferred imports — modelconfig is the seam llm.py/embedding.py sit behind,
#    so a top-level import here would cycle with embedding.py's StaticModelGateway use) ──

def _auth_headers(resolved: ResolvedModel) -> dict[str, str]:
    headers = dict(resolved.headers)
    if resolved.api_key is not None:
        headers.setdefault("Authorization", f"Bearer {resolved.api_key.get()}")
    return headers


def _build_llm(resolved: ResolvedModel) -> Any:
    from .llm import OpenAICompatibleLLM
    from .transport import make_http_transport

    transport = make_http_transport(
        timeout=resolved.timeout, headers=_auth_headers(resolved),
        provider=resolved.provider, model=resolved.model,
    )
    return OpenAICompatibleLLM(
        resolved.base_url, resolved.model, transport=transport,
        params=resolved.params or None, ref=resolved.ref,
    )


def _build_embedder(resolved: ResolvedModel) -> Any:
    from .embedding import OpenAICompatibleEmbedder
    from .transport import make_http_transport

    transport = make_http_transport(
        timeout=resolved.timeout, headers=_auth_headers(resolved),
        provider=resolved.provider, model=resolved.model,
    )
    return OpenAICompatibleEmbedder(
        resolved.base_url, resolved.model, transport=transport, params=resolved.params or None,
    )


# ── the gateway ─────────────────────────────────────────────────────────────────────

class ModelGateway:
    """The one internal model-resolution seam (FR-4). Parses both config files once at
    construction (`from_env()` — FR-15, no reload path); `.resolve()`/`.llm()`/
    `.embedder()` are cheap, offline, per-call lookups over the already-parsed data —
    resolving per call (not at construction) is what lets Landing 2's workspace
    override apply with no signature changes (§3.1)."""

    def __init__(
        self,
        catalog: ProviderCatalog,
        overlay: Overlay,
        *,
        workspace_overrides: WorkspaceOverrides | None = None,
    ) -> None:
        self._catalog = catalog
        self._overlay = overlay
        self._providers = _build_providers(catalog, overlay)
        self._ws_overrides = workspace_overrides or NullWorkspaceOverrides()
        # Minor 3 (U8 gate): warn once per (kind, ref) when `embedder()` silently
        # drops every fallback-chain element beyond the primary.
        self._embedder_fallback_warned: set[tuple[str, str]] = set()

    @classmethod
    def from_env(
        cls, *, workspace_overrides: WorkspaceOverrides | None = None
    ) -> "ModelGateway":
        """Parse both config files (FR-15, no reload path). `workspace_overrides`
        (L2-3) is the production seam for FR-16/FR-17: the caller (`app.py`) builds a
        `GraphWorkspaceOverrides(repo)` bound to the same `Repository`/connection the
        rest of the app shares and passes it here; omitted, every workspace resolves
        as override-free (`NullWorkspaceOverrides`, Landing 1's behavior, still the
        default for offline construction and every existing caller)."""
        from . import config as _config

        _config.assert_no_legacy_model_env()
        opencode_path = _config.OPENCODE_CONFIG_PATH
        if not opencode_path:
            raise ModelConfigError(
                "FALKORCHAT_OPENCODE_CONFIG is not set — set it to the shared, "
                "pristine opencode.json path (see config/opencode.example.json) "
                "before enabling an LLM consumer"
            )
        overlay_path = _config.MODEL_CONFIG_PATH
        catalog = ProviderCatalog.load(opencode_path)
        overlay = Overlay.load(overlay_path)
        return cls(catalog, overlay, workspace_overrides=workspace_overrides)

    def has_chat(self) -> bool:
        return True

    def has_embedder(self) -> bool:
        return True

    def _resolve_element(
        self, ref: str, *, kind: str, role_timeout: float | None = None
    ) -> ResolvedModel:
        """Resolve one direct `provider/model` ref to a `ResolvedModel`. `role_timeout`
        is the role-level `timeout` fallback (§ `RoleSpec`) — used only when this
        element has no `models.<ref>.timeout` of its own; irrelevant for a direct
        (non-role) resolution, where it is always `None`."""
        provider_id, model_id = ref.split("/", 1)
        spec = self._providers.get(provider_id)
        if spec is None:
            raise ModelResolutionError(
                f"unknown provider {provider_id!r} (ref {ref!r}, kind {kind!r}) — "
                f"declare it in {self._catalog.path} or {self._overlay.path}"
            )

        settings = dict(self._overlay.model_settings(ref))
        timeout = settings.pop("timeout", None)
        if not isinstance(timeout, (int, float)):
            timeout = (
                role_timeout if role_timeout is not None
                else self._overlay.timeout_for_kind(kind)
            )
        dim = settings.pop("dim", None)
        settings.pop("protocol", None)  # reserved; provider-level protocol governs L1
        params = {_camel_to_snake(k): v for k, v in settings.items()}

        return ResolvedModel(
            ref=ref, provider=provider_id, model=model_id, base_url=spec.base_url,
            api_key=spec.api_key, headers=spec.headers, protocol=spec.protocol,
            timeout=float(timeout), params=params,
            dim=dim if isinstance(dim, int) else None,
        )

    def _resolve_ref(self, ref: str, *, kind: str) -> tuple[ResolvedModel, ...]:
        """Resolve one ref to an ordered chain. A direct `provider/model` ref (has a
        `/`) is a length-1 chain, unchanged from Landing 1. A ref with **no** `/` is a
        role name (FR-7): looked up in the overlay's `roles` map and expanded to its
        ordered, settings-applied fallback chain (FR-18) — the role's own load-time
        validation (`_build_roles`) already guarantees every chain element has a `/`,
        so this never recurses."""
        if "/" in ref:
            return (self._resolve_element(ref, kind=kind),)
        role = self._overlay.roles.get(ref)
        if role is None:
            raise ModelResolutionError(
                f"unknown role {ref!r} (kind={kind!r}) — declare it in "
                f"{self._overlay.path}'s roles map, or use '<provider>/<model-id>'"
            )
        return tuple(
            self._resolve_element(m, kind=kind, role_timeout=role.timeout)
            for m in role.models
        )

    def _workspace_override_ref(
        self, kind: str, *, ws: str | None, overrides: Any
    ) -> str | None:
        """FR-16/FR-17: the workspace override ref for `kind`, or `None` when there
        isn't one. Two mutually-exclusive sources (§2.6), `overrides` preferred so a
        caller that already did a per-drive read never triggers a second graph round
        trip: a pre-fetched `{agentModel, guardModel, embeddingModel, responderModel}`
        dict (`overrides=`, translated via `_KIND_TO_OVERRIDE_KEY`), or a fresh
        per-call read through the injected `WorkspaceOverrides` port (`ws=`, only when
        `overrides` was not given). An empty string is never treated as an override
        (`-graph.md` §2.4: `''` is a value, never "no override") — this is a defensive
        backstop, not a substitute for the write path never producing one."""
        if overrides is not None:
            key = _KIND_TO_OVERRIDE_KEY.get(kind)
            value = overrides.get(key) if key else None
        elif ws is not None:
            value = self._ws_overrides.get(ws, kind)
        else:
            value = None
        return value if isinstance(value, str) and value else None

    def resolve(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Resolution:
        """FR-16/FR-17's real precedence (L2-3), first-match-wins: **workspace
        override → `requested` → the per-kind default**. The workspace override is a
        hard cap: when present it wins outright, even over an explicit `requested=`
        choice (AC-10) — and it resolves through the exact same `_resolve_ref` path as
        any other ref (role expansion, provider lookup), so an override naming a model
        or role the config does not declare fails loudly here (`ModelResolutionError`,
        FR-10/§6.3) rather than silently falling through to `requested` or the
        default. A hard cap that silently degrades to the thing it caps would be worse
        than no cap at all."""
        override_ref = self._workspace_override_ref(kind, ws=ws, overrides=overrides)
        if override_ref:
            chain = self._resolve_ref(override_ref, kind=kind)
            return Resolution(kind=kind, chain=chain, source="workspace")

        ref = requested or self._overlay.default_for(kind)
        if not ref:
            raise ModelResolutionError(
                f"no model configured for kind {kind!r}: no default in "
                f"{self._overlay.path} and none was requested"
            )
        chain = self._resolve_ref(ref, kind=kind)
        source = "requested" if requested else "default"
        return Resolution(kind=kind, chain=chain, source=source)

    def _client_for_chain(self, resolution: Resolution) -> Any:
        if len(resolution.chain) == 1:
            return _build_llm(resolution.primary)
        from .llm import FallbackClient

        clients = tuple(_build_llm(r) for r in resolution.chain)
        refs = tuple(r.ref for r in resolution.chain)
        return FallbackClient(clients, refs)

    def resolve_llm(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> tuple[Any, Resolution]:
        """Like `.llm()`, but also returns the `Resolution` that produced the client —
        for a caller that needs to know which precedence rung won (FR-8's
        `modelSource`, §7 L2-3's `-graph.md` §6.2 binding requirement 3), in **one**
        gateway call rather than a separate `.resolve()` + `.llm()` pair (so a
        call-counting test double records exactly one entry per logical resolution,
        same as `.llm()` always has). `.llm()` is the simple form for callers that
        only need the client (`guard`, `agent`/responder, `embedding`)."""
        resolution = self.resolve(kind, requested=requested, ws=ws, overrides=overrides)
        return self._client_for_chain(resolution), resolution

    def llm(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Any:
        return self.resolve_llm(kind, requested=requested, ws=ws, overrides=overrides)[0]

    def embedder(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Any:
        resolution = self.resolve(kind, requested=requested, ws=ws, overrides=overrides)
        if len(resolution.chain) > 1:
            self._warn_embedder_fallback_ignored(kind, resolution)
        return _build_embedder(resolution.primary)

    def _warn_embedder_fallback_ignored(self, kind: str, resolution: Resolution) -> None:
        """Minor 3 (U8 gate): `embedder()` only ever builds `resolution.primary` — a
        role resolving to more than one model for kind `embedding` has its fallback
        elements silently ignored (no error, no `FallbackClient` wrapping, unlike
        `llm()`). Warn once per `(kind, primary ref)` — mirrors
        `StaticModelGateway`'s existing once-per-`(kind, ref)` WARNING shape."""
        key = (kind, resolution.primary.ref)
        if key in self._embedder_fallback_warned:
            return
        self._embedder_fallback_warned.add(key)
        _log.warning(
            "embedder() for kind %r resolved a %d-element fallback chain but only "
            "the primary %r is ever used — embedder() honours no fallback (Landing 2 "
            "Minor 3); llm() wraps a multi-element chain in FallbackClient, "
            "embedder() does not",
            kind, len(resolution.chain), resolution.primary.ref,
        )


# The degenerate `ResolvedModel` a `StaticModelGateway` reports — never used to build a
# real client (its `.llm()`/`.embedder()` return the injected object directly).
_STATIC_RESOLVED = ResolvedModel(
    ref="(static)", provider="(static)", model="(static)", base_url="(static)",
    api_key=None, headers={}, protocol="openai", timeout=0.0, params={}, dim=None,
)


class StaticModelGateway:
    """The FR-4 sugar wrapper: `__init__` wraps a directly-injected `llm=`/`embedder=`
    client into one of these instead of reading any config file. Dependency injection
    for tests, never a configuration route (A-4) — this is what keeps the 38 `llm=` /
    24 `guard_judge=` test injections working untouched.

    `.resolve()` ignores `requested` (a static client cannot honour a per-call model
    choice) and logs a WARNING **once per `(kind, ref)`** naming the ignored ref —
    without this, AC-4 passes under a real gateway and silently regresses to one
    model under any `llm=` wiring.
    """

    def __init__(self, *, llm: Any = None, embedder: Any = None) -> None:
        self._llm = llm
        self._embedder = embedder
        self._warned: set[tuple[str, str]] = set()

    def has_chat(self) -> bool:
        return self._llm is not None

    def has_embedder(self) -> bool:
        return self._embedder is not None

    def _warn_once(self, kind: str, ref: str) -> None:
        key = (kind, ref)
        if key in self._warned:
            return
        self._warned.add(key)
        _log.warning(
            "StaticModelGateway ignoring requested model %r for kind %r — a "
            "statically-injected client is in use (FR-4 test sugar); wire a real "
            "ModelGateway for per-call resolution", ref, kind,
        )

    def resolve(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Resolution:
        self._warn_once(kind, requested if requested is not None else "(default)")
        # A static gateway has no workspace-override concept (it bypasses config
        # entirely) — `.source` only ever distinguishes "the caller named a model" from
        # "it didn't", mirroring the pre-L2-3 behavior every existing `llm=`/`embedder=`
        # injection test already depends on (`requested` truthy ⇒ that choice "won").
        source = "requested" if requested is not None else "default"
        return Resolution(kind=kind, chain=(_STATIC_RESOLVED,), source=source)

    def resolve_llm(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> tuple[Any, Resolution]:
        resolution = self.resolve(kind, requested=requested, ws=ws, overrides=overrides)
        if self._llm is None:
            raise ModelResolutionError(
                f"no static llm client was injected for kind {kind!r}"
            )
        return self._llm, resolution

    def llm(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Any:
        return self.resolve_llm(kind, requested=requested, ws=ws, overrides=overrides)[0]

    def embedder(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Any:
        self.resolve(kind, requested=requested, ws=ws, overrides=overrides)
        if self._embedder is None:
            raise ModelResolutionError(
                f"no static embedder client was injected for kind {kind!r}"
            )
        return self._embedder
