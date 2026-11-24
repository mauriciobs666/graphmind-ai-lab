"""The one gateway-configuration parser (docs/plans/llm-gateway.md §3.5).

Library, service and holder all load `gateway.toml` through `ConfigSource`. The file is checked
by `ownercheck.read_owner_file` on every `get()` (the parse alone is cached, by the
`(dev, ino, mtime_ns, size)` of the descriptor that was read). A file that fails the check is
*unprotected*: its `local_endpoints`, `scopes`, `scope_sources`, `scope_overrides`, `run_bounds`
and hosted steps are ignored, hosted is disabled, and the local allow-list is loopback IP
literals only. `LLMGATEWAY_CONFIG` overrides the path in the library only (`resolve_path`).
"""

from __future__ import annotations

import ipaddress
import os
import re
import tomllib
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from llmgateway.ownercheck import read_owner_file
from llmgateway.types import GatewayError

DEFAULT_PATH = "/etc/llmgateway/gateway.toml"
ENV_VAR = "LLMGATEWAY_CONFIG"
PRODUCTION_TRUSTED_UIDS: frozenset[int] = frozenset({0})

SCOPE_ID = re.compile(r"[a-z0-9:._-]{1,128}")
MARKERS = frozenset({"restricted", "unrestricted"})
_LOCAL_KINDS = frozenset({"lmstudio", "ollama", "openai_compat"})

# (scheme, canonical IP literal, port)
Origin = tuple[str, str, int]


class ConfigError(GatewayError):
    """A protected configuration that cannot be used (ill-formed, or a forbidden construct)."""


def resolve_path(env: Mapping[str, str] | None = None, *, library: bool = False) -> str:
    """The configuration path. Only the library honours `LLMGATEWAY_CONFIG`; service and holder
    pass `library=False` and always read the default (§3.5, §3.4: root config only, env ignored)."""
    if library:
        override = (os.environ if env is None else env).get(ENV_VAR)
        if override:
            return override
    return DEFAULT_PATH


def parse_origin(url: object) -> Origin | None:
    """Canonical origin of an http(s) URL with an IP-literal host, else None.

    Hostnames, userinfo, zone ids, whitespace/control characters, backslashes and malformed
    ports are all refused (§3.5), so no spelling of a name can pass for a local address.
    """
    if not isinstance(url, str) or not url.isascii() or not url.isprintable() or " " in url:
        return None
    if "\\" in url:
        return None
    scheme, sep, rest = url.partition("://")
    scheme = scheme.lower()
    if not sep or scheme not in ("http", "https"):
        return None
    authority = re.split(r"[/?#]", rest, maxsplit=1)[0]
    if not authority or "@" in authority:
        return None
    if authority.startswith("["):
        host, bracket, tail = authority[1:].partition("]")
        if not bracket or (tail and not tail.startswith(":")):
            return None
        port_text = tail[1:] if tail else None
    else:
        host, colon, port_text = authority.partition(":")
        if not colon:
            port_text = None
    if "%" in host:
        return None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if port_text is None:
        port = 443 if scheme == "https" else 80
    elif port_text.isdecimal() and port_text.isascii() and 0 < int(port_text) < 65536:
        port = int(port_text)
    else:
        return None
    return (scheme, str(ip), port)


def _is_loopback_literal(host: str) -> bool:
    ip = ipaddress.ip_address(host)
    if ip.version == 4:
        return ip in ipaddress.ip_network("127.0.0.0/8")
    return ip == ipaddress.IPv6Address("::1")


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """One load of the configuration. Sections are plain mappings; policy interprets them."""

    protected: bool
    reason: str = ""
    deployment: Mapping[str, Any] = field(default_factory=dict)
    holder: Mapping[str, Any] = field(default_factory=dict)
    providers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    scopes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    scope_sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    call_sites: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    scope_overrides: tuple[Mapping[str, Any], ...] = ()
    run_bounds: tuple[Mapping[str, Any], ...] = ()
    local_origins: frozenset[Origin] = frozenset()
    warnings: tuple[str, ...] = ()

    @property
    def hosted_enabled(self) -> bool:
        return self.protected

    @property
    def loopback_only(self) -> bool:
        return not self.protected

    def effective_local_endpoints(self) -> frozenset[Origin]:
        """Owner-listed local origins; empty when unprotected (then `loopback_only` applies:
        any loopback IP literal, any port)."""
        return self.local_origins if self.protected else frozenset()

    def is_local(self, url: object) -> bool:
        origin = parse_origin(url)
        if origin is None:
            return False
        if self.loopback_only:
            return _is_loopback_literal(origin[1])
        return origin in self.local_origins

    def provider_is_hosted(self, provider_id: str) -> bool:
        prov = self.providers[provider_id]
        return prov.get("kind") not in _LOCAL_KINDS or not self.is_local(prov.get("base_url"))


def _table(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _array(raw: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value = raw.get(key, [])
    return tuple(v for v in value if isinstance(v, Mapping)) if isinstance(value, list) else ()


def _named_tables(raw: Mapping[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {k: dict(v) for k, v in _table(raw, key).items() if isinstance(v, Mapping)}


def _validate_protected(raw: Mapping[str, Any]) -> None:
    sources = _named_tables(raw, "scope_sources")
    prefixes = []
    for name, src in sources.items():
        prefix = src.get("prefix")
        if not isinstance(prefix, str) or not prefix:
            raise ConfigError(f"scope_sources.{name}: prefix must be a non-empty string")
        prefixes.append(prefix)
    for scope_id, scope in _named_tables(raw, "scopes").items():
        if not SCOPE_ID.fullmatch(scope_id):
            raise ConfigError(f"scope id {scope_id!r} is not ASCII [a-z0-9:._-]{{1,128}}")
        if "marker" in scope:
            if any(scope_id.startswith(p) for p in prefixes):
                raise ConfigError(
                    f"scope {scope_id!r} is source-backed; a static marker is not allowed"
                )
            if scope["marker"] not in MARKERS:
                raise ConfigError(f"scope {scope_id!r}: marker must be one of {sorted(MARKERS)}")


def _build_protected(raw: Mapping[str, Any]) -> GatewayConfig:
    _validate_protected(raw)
    deployment = _table(raw, "deployment")
    origins: set[Origin] = set()
    warnings: list[str] = []
    listed = deployment.get("local_endpoints", [])
    for entry in listed if isinstance(listed, list) else []:
        origin = parse_origin(entry)
        if origin is None:
            warnings.append(f"local_endpoints entry {entry!r} ignored: not an IP-literal origin")
        else:
            origins.add(origin)
    return GatewayConfig(
        protected=True,
        deployment=deployment,
        holder=_table(raw, "holder"),
        providers=_named_tables(raw, "providers"),
        scopes=_named_tables(raw, "scopes"),
        scope_sources=_named_tables(raw, "scope_sources"),
        call_sites=_named_tables(raw, "call_sites"),
        scope_overrides=_array(raw, "scope_overrides"),
        run_bounds=_array(raw, "run_bounds"),
        local_origins=frozenset(origins),
        warnings=tuple(warnings),
    )


def _step_provider(step: object) -> str | None:
    if isinstance(step, str):
        return step
    if isinstance(step, Mapping) and isinstance(step.get("provider"), str):
        return step["provider"]
    return None


def _build_unprotected(reason: str, untrusted: bytes) -> GatewayConfig:
    """Fail closed: only loopback-literal local providers and their call sites survive; every
    hosted step, scope, override, bound, deployment and holder setting is dropped (§3.5)."""
    try:
        raw = tomllib.loads(untrusted.decode("utf-8")) if untrusted else {}
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        raw = {}
    shell = GatewayConfig(protected=False)
    providers = {
        pid: prov
        for pid, prov in _named_tables(raw, "providers").items()
        if prov.get("kind") in _LOCAL_KINDS and shell.is_local(prov.get("base_url"))
    }
    call_sites: dict[str, dict[str, Any]] = {}
    for name, site in _named_tables(raw, "call_sites").items():
        kept = dict(site)
        kept["scopes"] = []
        for chain_key in ("chain", "local_chain"):
            steps = site.get(chain_key)
            if isinstance(steps, list):
                kept[chain_key] = [s for s in steps if _step_provider(s) in providers]
        call_sites[name] = kept
    return GatewayConfig(
        protected=False,
        reason=reason,
        providers=providers,
        call_sites=call_sites,
    )


class ConfigSource:
    """Loads the configuration for one path; the owner check runs on every `get()`."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        trusted_uids: Collection[int] = PRODUCTION_TRUSTED_UIDS,
        *,
        anchor: str | os.PathLike[str] | None = None,
    ) -> None:
        self._path = os.fspath(path)
        self._trusted = frozenset(trusted_uids)
        self._anchor = anchor
        self._cache: tuple[tuple[int, int, int, int], GatewayConfig] | None = None

    def get(self) -> GatewayConfig:
        res = read_owner_file(self._path, self._trusted, anchor=self._anchor)
        if not res.ok:
            self._cache = None
            return _build_unprotected(res.reason, res.untrusted_data)
        if self._cache is not None and self._cache[0] == res.key:
            return self._cache[1]
        try:
            raw = tomllib.loads(res.data.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"{self._path}: not valid TOML: {exc}") from exc
        config = _build_protected(raw)
        self._cache = (res.key, config)
        return config
