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

Landing-2 seams intentionally left as no-ops/stubs here (§6.1): `ws=`/`overrides=` are
accepted and threaded through but not yet applied (`WorkspaceOverrides` resolves to a
null port); `roles`/`agents` overlay keys are parsed and logged, not honoured; a ref
with no `/` is rejected naming Landing 2 rather than silently misresolving.
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
    """The outcome of one `resolve()` call. `chain` is a tuple — length 1 in Landing 1
    — so FR-18's ordered fallback chain (Landing 2) is a wrapper swap, not a signature
    change (§6.1 item 1)."""

    kind: str
    chain: tuple[ResolvedModel, ...]

    @property
    def primary(self) -> ResolvedModel:
        return self.chain[0]


class WorkspaceOverrides(Protocol):
    """FR-16/FR-17 seam (Landing 2): a per-kind workspace override read. Threaded from
    every `resolve()` call from day one (§6.1 item 2) so Landing 2 swaps the null port
    below for a graph-backed one with no call-site changes."""

    def get(self, ws: str, kind: str) -> str | None: ...


class NullWorkspaceOverrides:
    """The Landing-1 no-op `WorkspaceOverrides` — every workspace has no override."""

    def get(self, ws: str, kind: str) -> str | None:
        return None


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
    per-model `models.<ref>` settings, per-provider `providers.<id>` overrides, and
    (reserved, Landing 2) `roles`/`agents`. Unknown top-level keys are accepted and
    logged, never rejected (§4.1) — an admin's Landing-2-ready file never fails a
    Landing-1 build."""

    _KNOWN_KEYS = frozenset(
        {"defaults", "timeouts", "models", "providers", "roles", "agents"}
    )

    def __init__(self, doc: dict[str, Any], *, path: str) -> None:
        self.path = path
        self.defaults = _as_dict(doc.get("defaults"))
        self.timeouts = _as_dict(doc.get("timeouts"))
        self.models = _as_dict(doc.get("models"))
        self.providers = _as_dict(doc.get("providers"))

        if "roles" in doc:
            _log.info(
                "overlay %s declares 'roles' — reserved, parsed, not honoured until "
                "Landing 2 (FR-7)", path,
            )
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

    @classmethod
    def from_env(cls) -> "ModelGateway":
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
        return cls(catalog, overlay)

    def has_chat(self) -> bool:
        return True

    def has_embedder(self) -> bool:
        return True

    def _resolve_ref(self, ref: str, *, kind: str) -> ResolvedModel:
        if "/" not in ref:
            raise ModelConfigError(
                f"model reference {ref!r} (kind={kind!r}) has no provider prefix — "
                f"bare role names are not available until Landing 2 (FR-7); use "
                f"'<provider>/<model-id>'"
            )
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
            timeout = self._overlay.timeout_for_kind(kind)
        dim = settings.pop("dim", None)
        settings.pop("protocol", None)  # reserved; provider-level protocol governs L1
        params = {_camel_to_snake(k): v for k, v in settings.items()}

        return ResolvedModel(
            ref=ref, provider=provider_id, model=model_id, base_url=spec.base_url,
            api_key=spec.api_key, headers=spec.headers, protocol=spec.protocol,
            timeout=float(timeout), params=params,
            dim=dim if isinstance(dim, int) else None,
        )

    def resolve(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Resolution:
        # FR-16/FR-17 seam (Landing 2): the read happens, the result is unused — a
        # graph-backed WorkspaceOverrides swaps in with no call-site change (§6.1 item 2).
        if ws is not None:
            self._ws_overrides.get(ws, kind)

        ref = requested or self._overlay.default_for(kind)
        if not ref:
            raise ModelResolutionError(
                f"no model configured for kind {kind!r}: no default in "
                f"{self._overlay.path} and none was requested"
            )
        resolved = self._resolve_ref(ref, kind=kind)
        return Resolution(kind=kind, chain=(resolved,))

    def llm(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Any:
        resolution = self.resolve(kind, requested=requested, ws=ws, overrides=overrides)
        return _build_llm(resolution.primary)

    def embedder(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Any:
        resolution = self.resolve(kind, requested=requested, ws=ws, overrides=overrides)
        return _build_embedder(resolution.primary)


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
        return Resolution(kind=kind, chain=(_STATIC_RESOLVED,))

    def llm(
        self, kind: str, *, requested: str | None = None, ws: str | None = None,
        overrides: Any = None,
    ) -> Any:
        self.resolve(kind, requested=requested, ws=ws, overrides=overrides)
        if self._llm is None:
            raise ModelResolutionError(
                f"no static llm client was injected for kind {kind!r}"
            )
        return self._llm

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
