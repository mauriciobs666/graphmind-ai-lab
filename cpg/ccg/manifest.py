"""The system manifest loader and the egress section's checks (C-1001 P1-03).

Plan: ``docs/plans/connected-code-graph-p1-extraction.md`` § P1-03. Egress section, S1-S5 and ``canonical_endpoint``:
``docs/plans/connected-code-graph-egress-spec3.md`` §4. ``repoId`` normalization: bundle spec3 §3.4.

A manifest is one YAML file: ``schemaVersion``, ``system`` (checked by ``ccg.sysname``), ``egress`` (the system's
only allow-list), and ``repos``, each with ``repoId``, ``path``, optional ``exclusions`` and a mandatory
``policy.egress``. ``groupings`` (P2) and a repository's ``remote`` (git fetch, deferred) are refused with their own
messages. S6 (the local endpoint is on this machine) is the broker's check, not this module's.
"""
from __future__ import annotations

import ipaddress
import json
import posixpath
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import jsonschema
import yaml

from ccg import sysname

SCHEMAS = Path(__file__).resolve().parent / "schemas"
DEFAULT_PREVIEW_CAP = 2  # proposal Q-R8

_PORTS = {"http": 80, "https": 443}


class ManifestError(ValueError):
    """A manifest, or its egress section, is refused. ``codes`` lists the S-checks that fired (may be empty)."""

    def __init__(self, message: str, codes: tuple[str, ...] = ()):
        super().__init__(message)
        self.codes = codes


@dataclass(frozen=True)
class Policy:
    egress: str
    rca_window: int | None = None
    secrets_window: int | None = None
    preview_cap: int = DEFAULT_PREVIEW_CAP


@dataclass(frozen=True)
class Repo:
    repo_id: str
    path: str
    policy: Policy
    exclusion_paths: tuple[str, ...] = ()
    content_check_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class Manifest:
    system: str
    repos: tuple[Repo, ...]
    egress: Mapping[str, Any] = field(repr=False, default_factory=dict)


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text())


def _validate(instance: Any, schema_name: str, what: str) -> None:
    validator = jsonschema.Draft202012Validator(_schema(schema_name))
    errors = sorted(validator.iter_errors(instance), key=lambda e: [str(p) for p in e.absolute_path])
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.absolute_path) or "(root)"
        raise ManifestError(f"{what}: {where}: {e.message}")


# --- repoId (bundle spec3 §3.4) --------------------------------------------------------------------------------


def normalize_repo_id(name: str) -> str:
    """NFC, strip, lowercase, each run outside ``[a-z0-9._-]`` to ``-``, trim ``-``. An empty result is an error."""
    out = re.sub(r"[^a-z0-9._-]+", "-", unicodedata.normalize("NFC", name).strip().lower()).strip("-")
    if not out:
        raise ManifestError(f"repository name {name!r} normalizes to an empty repoId")
    return out


# --- canonical endpoint and the egress checks (egress-spec3 §4.3) -----------------------------------------------


def _bad_char(c: str) -> bool:
    return ord(c) <= 0x1F or ord(c) == 0x7F


def canonical_endpoint(endpoint: str) -> str:
    """``scheme://host:port``: lowercase scheme and host, explicit port, no path/query/user-info, RFC 5952 IPv6.

    An endpoint equals its canonical form only if it already is exactly that shape, so the broker's ``R0`` check
    (``endpoint == canonical_endpoint(endpoint)``) refuses every other spelling. Unparseable input raises.
    """
    if not isinstance(endpoint, str) or not endpoint:
        raise ManifestError(f"endpoint must be a non-empty string, got {endpoint!r}")
    # urlsplit silently drops tab, CR and LF and tolerates edge spaces, so refuse them first.
    if any(_bad_char(c) or c == " " for c in endpoint):
        raise ManifestError(f"endpoint {endpoint!r} contains a control character or a space")
    try:
        parts = urlsplit(endpoint)
        host, port = parts.hostname, parts.port
    except ValueError as exc:
        raise ManifestError(f"unparseable endpoint {endpoint!r}: {exc}") from None
    scheme = parts.scheme.lower()
    if scheme not in _PORTS:
        raise ManifestError(f"endpoint scheme must be http or https: {endpoint!r}")
    if not host:
        raise ManifestError(f"endpoint has no host: {endpoint!r}")
    if not host.isascii() or "%" in host:
        raise ManifestError(f"endpoint host must be plain ASCII without a zone id: {endpoint!r}")
    if port is None:
        port = _PORTS[scheme]
    if not 1 <= port <= 65535:
        raise ManifestError(f"endpoint port out of range: {endpoint!r}")
    if ":" in host:
        try:
            host = f"[{ipaddress.IPv6Address(host).compressed}]"
        except ValueError:
            raise ManifestError(f"endpoint has an invalid IPv6 literal: {endpoint!r}") from None
    return f"{scheme}://{host}:{port}"


_ENDPOINT_RE = re.compile(r"\A(?P<scheme>https?)://(?P<host>\[[^\]]*\]|[^:/\[\]]+):(?P<port>[0-9]+)\Z")


def _ip_literal(host: str):
    inner = host[1:-1] if host.startswith("[") else host
    try:
        return ipaddress.ip_address(inner)
    except ValueError:
        return None


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():  # keys are fixed by the schema (additionalProperties: false)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def check_egress(egress: Mapping[str, Any]) -> None:
    """Validate the ``egress`` section against ``egress-v1.schema.json``, then apply S1-S5; raise `ManifestError`.

    Every violated check is reported (``codes``), not only the first. S3 subsumes S5 for the same endpoint: an
    endpoint with an over-range port or a non-compressed IPv6 literal is reported as S3 only.
    """
    _validate(egress, "egress-v1.schema.json", "egress section")
    entries = [(sec, e) for sec in ("local", "outbound", "hosted") for e in egress[sec]]
    found: dict[str, str] = {}

    ids = [e["id"] for _, e in entries]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        found["S1"] = f"ids repeated: {dup}"
    eps = [e["endpoint"] for _, e in entries]
    dup = sorted({i for i in eps if eps.count(i) > 1})
    if dup:
        found["S2"] = f"endpoints in more than one entry: {dup}"

    s3, s4, s5 = [], [], []
    for sec, e in entries:
        ep = e["endpoint"]
        m = _ENDPOINT_RE.match(ep)
        subsumed = False
        if m:
            host = m.group("host")
            if int(m.group("port")) > 65535:
                s3.append(f"{ep}: port above 65535")
                subsumed = True
            if host.startswith("["):
                try:
                    ok = ipaddress.IPv6Address(host[1:-1]).compressed == host[1:-1]
                except ValueError:
                    ok = False
                if not ok:
                    s3.append(f"{ep}: IPv6 literal is not its RFC 5952 compressed form")
                    subsumed = True
            ip = _ip_literal(host) if sec == "local" else None
            if ip is not None and (ip.is_unspecified or ip.is_multicast):
                s4.append(f"{ep}: unspecified or multicast")
        if not subsumed:
            try:
                if canonical_endpoint(ep) != ep:
                    s5.append(f"{ep!r}: not its canonical form")
            except ManifestError as exc:
                s5.append(str(exc))
    for s in _strings(egress):
        if any(_bad_char(c) for c in s):
            s5.append(f"{s!r}: contains a character <= U+001F or U+007F")
    for code, msgs in (("S3", s3), ("S4", s4), ("S5", s5)):
        if msgs:
            found[code] = "; ".join(msgs)
    if found:
        raise ManifestError(
            "egress section refused: " + "; ".join(f"{c}: {found[c]}" for c in sorted(found)), tuple(sorted(found))
        )


# --- the loader ---------------------------------------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """``SafeLoader`` that refuses a duplicate mapping key instead of keeping the last one."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ManifestError(f"duplicate key {key!r} in the manifest (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _check_relative(path: str, what: str) -> None:
    if path.startswith("/") or "\\" in path or posixpath.normpath(path).startswith("..") or any(
        _bad_char(c) for c in path
    ):
        raise ManifestError(f"{what} {path!r} must be a relative path inside the repository")


def load_manifest(path: str | Path) -> Manifest:
    try:
        doc = yaml.load(Path(path).read_text(), Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ManifestError(f"manifest is not valid YAML: {exc}") from None
    if not isinstance(doc, dict):
        raise ManifestError("manifest must be a YAML mapping")
    if "groupings" in doc:
        raise ManifestError("groupings are not supported until P2; remove the `groupings` key")
    for repo in doc.get("repos") if isinstance(doc.get("repos"), list) else []:
        if isinstance(repo, dict) and "remote" in repo:
            raise ManifestError("git fetch is not enabled in P1: a repository must not name a `remote`")
    _validate(doc, "system-manifest-v1.schema.json", "manifest")
    try:
        system = sysname.check_system_name(doc["system"])
    except sysname.SystemNameError as exc:
        raise ManifestError(f"system name: {exc}") from None
    check_egress(doc["egress"])

    repos, seen = [], set()
    for raw in doc["repos"]:
        repo_id = normalize_repo_id(raw["repoId"])
        if repo_id in seen:
            raise ManifestError(f"two repositories normalize to the repoId {repo_id!r}")
        seen.add(repo_id)
        rpath = raw["path"]
        _check_relative(rpath, "repository path")
        if posixpath.normpath(rpath) != rpath:
            raise ManifestError(f"repository path {rpath!r} is not normalized (expected {posixpath.normpath(rpath)!r})")
        excl = raw.get("exclusions", {})
        paths = tuple(excl.get("paths", ()))
        files = tuple(excl.get("contentCheck", {}).get("files", ()))
        for p in paths + files:
            _check_relative(p, "exclusion")
        pol = raw["policy"]
        repos.append(
            Repo(
                repo_id=repo_id,
                path=rpath,
                policy=Policy(
                    egress=pol["egress"],
                    rca_window=pol.get("rcaWindow"),
                    secrets_window=pol.get("secretsWindow"),
                    preview_cap=pol.get("previewCap", DEFAULT_PREVIEW_CAP),
                ),
                exclusion_paths=paths,
                content_check_files=files,
            )
        )
    return Manifest(system=system, repos=tuple(repos), egress=doc["egress"])
