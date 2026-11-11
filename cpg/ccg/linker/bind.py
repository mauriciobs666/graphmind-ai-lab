"""The linker's binding provider (P1-17 dispatch 2; plan ``docs/plans/connected-code-graph-p1-linking.md`` P1-17 behaviours (4), (5),
(6), (13) and trust rules (a)-(e); chain shape ``docs/plans/connected-code-graph-p1-graph.md`` §2, ``-answer-spec2.md`` §3.2; the parts
comparison ``-bundle-spec5.md`` §5.4, §7, §9).

:class:`Binder` is a :class:`ccg.linker.link.Bindings`: ``link(..., bindings=Binder(...))`` asks it once per consumer site which
units the site's *URL or launch chain* selects, and with what chain of ``BOUND_BY`` entries (``ord`` 0 the entry nearest the consumer).

**What a consumer's URL is.** (a) An ``http`` consumer's own absolute URL: the ``alternatives`` of its ``PackFacts.lifted`` leaf (the key
drops the host and port, the leaf keeps them). (b) A pack's ``urlBindings`` ``callsite`` source (``scope`` file or unit) for the pack's
``kinds``. (c) A pack's ``urlBindings`` ``config`` source: an ``mcpServers.<server>.url`` entry binds a prompt mention
``mcp__<server>__<tool>`` of that server, which names its server and so takes no ``callsite`` URL source (the file's client URL is another server's). (d) A consumer that reads an environment variable (a Python ``os.environ.get``/``getenv``
call flowing into its URL slot): the variable's default text, and the URL a Compose ``services.<svc>.environment.<VAR>`` entry sets
(behaviour (5)). No other form of environment read is in any bundle record (a JavaScript ``import.meta.env.X`` leaves no trace),
so a consumer reading one is left unbound, which is the unbound outcome, not an error.

**Which units a URL selects.** Every URL becomes ``bindParts`` (spec5 §5.3): the bundle's own when its key epoch is usable, else
parts derived here from the ``url``-form ``bindValue`` or the URL text, under the system key; one code path (spec5 §7). A
``loopback`` host (``localhost``, ``127/8``, ``::1``, ``host.docker.internal``) matches its port against the packs' ``port`` launch
facts (explicit ``--port``, ``envPorts``, ``frameworkDefaults``) and against Compose ``port`` facts (trust rule e below); a ``service``
host, or an ``external`` one whose ``hostHmac`` equals the hash of a Compose service name declared anywhere in the system (and only
a single label, BL-16), selects that service's unit through its ``build.context`` (behaviour (5), the unit index is system-wide) or its
launched command. Then the path: a unit that has mount paths of a pack whose ``urlBindings`` cover the consumer's kind is selected only
when the URL's ``pathHmacs`` carry one of them as a prefix (O-5), through :func:`ccg.valuehash.mount_prefix_hmac`. A unit with none
is not restricted by path. ``ip`` hosts and unmatched ``external`` ones bind nothing. Several units on one port make ``port-shared``.

**Launch binding (6).** A ``.mcp.json`` server with a command and no URL: its ``command`` launch fact, a repo-relative script path named in
its argv (lexical, ``$CLAUDE_PROJECT_DIR`` is the repo root), that script's own launch facts, the ``Dockerfile`` ``CMD`` beside the
script, and the unit the packs' ``targetArg`` resolved for a launch (or, failing that, the unit rooted at the script's directory).
A launch the argv gate redacted names no script and breaks the chain there, as every unresolved hop does.

**Trust rules (they read untrusted repo files).** (a) Nothing here opens, stats, resolves or dials a path: only :mod:`posixpath`
string work (the module imports no ``os``/``pathlib``/``socket``). (b) A ``mountPath`` is joined to its compose file's directory
(system-relative: the repository's manifest path plus the file's), normalised lexically and must lie under a manifest repository, else the
site is ``possible``/``target-not-mapped`` through ``Unmapped(None, tuple_id(["unmapped", normalised path]))``; an absolute path or
one that leaves every repository is never followed. (c) A service is ``(origin, repoId, path, service)``: a ``.mcp.json`` server and a
Compose service of one name are two services, and the two lookups never meet. (d) A clear host the writer forbids (link-local and cloud-metadata
targets, in every form ``forbidden_target`` reads: ``inet_aton`` forms, embedded IPv4) is never bindable, whatever wrote the bundle: the
one definition is ``valuehash.forbidden_target`` (:func:`forbidden_host`). (e) ``port`` facts carry no published/container distinction
(``portKind`` is a later spec update): a loopback match through any Compose ``port`` fact (``ports: "A:B"``, a lone ``"N"``, ``expose``,
the long syntax) is counted per site in ``counts()["loopback_port_residual"]`` and stays confirmed.

**Untrusted records.** The binder re-checks what it reads (it does not rely on ``bundle.validate`` having run): a record or ``bindParts`` of
the wrong shape is skipped alone and binds nothing; a record whose id its own fields do not derive, or whose ``repoId`` is not its bundle's, is rejected alone; a record id that two *different* records
claim is dropped for both and counted in ``counts()["binding_duplicate_ids"]`` (the same record listed twice is one). A ``mcpServers.<server>.url`` entry
binds only from an ``mcp-json`` file. Env binding takes the value of the variable from every Compose service that sets it (the consumer's own
service cannot be told: ``-p0-compose-binding.md``); two different units make the site ``port-shared``. Each unit of a binding carries its own
chain (:meth:`~ccg.linker.link.Binding.entries_for`), so a link shows only the entries that reach its provider's unit.

**Mapped over unmapped.** When some chains reach a built unit and others end outside the built code, only the built units bind; the site is
``target-not-mapped`` only when nothing built is reached.

**Example role.** A chain through a file whose role is ``example`` (or an entry flagged ``example``) weakens the site with
``("possible", "example-config")``; when a chain through a primary file also exists, only the primary chains count.

**Key epoch (13).** ``bind_parts_usable`` gates every bundle's parts; a bundle that carries ``bindParts`` or ``bindKeyCheck`` and is
not usable yields one value-free ``bind-key-mismatch <unitId>`` line (:meth:`Binder.key_mismatches`); its URL entries bind by parts
derived from ``bindValue``, an entry with only parts stays unbound (BR-2, BR-3).
"""
from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ccg import bundle, ids, valuehash
from ccg.units import is_mcp_json
from ccg.linker.link import Binding, BoundEntry, Unmapped
from ccg.packrt.extract import Corpus
from ccg.packrt.match import clause_hit, text_hit
from ccg.packrt.run import PackFacts

__all__ = ["Binder", "forbidden_host"]

_DEFAULT_PORT = {"http": 80, "https": 443, "ws": 80, "wss": 443}
_SERVICE_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_SERVER_OF = re.compile(r"mcpServers\.([^.\[]+)\.")
_COMPOSE_ENV = re.compile(r"services\.([^.\[]+)\.environment\.([^.\[]+)")
_COMPOSE_SERVICE = re.compile(r"services\.([^.\[]+)\.")
_ENV_READS = ("environ.get", "getenv")
_PATHISH = re.compile(r"[A-Za-z0-9._/~+-]+")
_PROJECT_DIR = re.compile(r"^\$\{?CLAUDE_PROJECT_DIR\}?/")


def forbidden_host(host: str | None) -> bool:
    """True iff ``host`` (clear, as ``bindParts.host``) is one the writer withholds (trust rule d). The decision is
    ``ccg.valuehash.forbidden_target`` (the writer imports the same function), read at call time: there is no second set here to drift; what it cannot judge is forbidden."""
    if not isinstance(host, str):
        return False
    try:
        return bool(valuehash.forbidden_target(host))
    except Exception:
        return True


# -- records with their bundle context ---------------------------------------------------------------
@dataclass(frozen=True)
class _Rec:
    id: str
    repo: str
    path: str
    unit: str | None
    example: bool
    raw: dict = field(compare=False, hash=False)


@dataclass(frozen=True)
class _Url:
    """``bindParts`` of one URL (spec5 §5.3), plus where it came from: ``entry`` is a config entry id, ``None`` for a URL in code."""
    scheme: str
    port: int | None
    host_kind: str
    host: str | None
    host_hmac: str
    path_hmacs: tuple[str, ...]
    entry: str | None = None
    example: bool = False


@dataclass
class _Chain:
    """One way a URL or launch reaches a unit: ``unit`` (built) or ``unmapped``, with the ``BOUND_BY`` entries in order from the consumer."""
    unit: str | None
    unmapped: Unmapped | None
    entries: list[BoundEntry]
    example: bool = False


def _bound(rec_id: str, port: int | None = None, origin: str | None = None, pack: str | None = None) -> BoundEntry:
    return BoundEntry(rec_id, port, origin if port is not None else None, pack if origin == "pack-default" else None)


_HOST_KINDS = frozenset({"loopback", "ip", "service", "external"})


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _entry_ok(r: dict) -> bool:
    """A ``configEntry`` the binder can read: the fields it indexes are strings (trust: it does not rely on ``bundle.validate``)."""
    return all(isinstance(r.get(k), str) for k in ("id", "path", "keyPath", "format"))


def _launch_ok(r: dict) -> bool:
    """A ``launchFact`` the binder can read: its kind, and what that kind carries, have the type the indexes use."""
    if not (isinstance(r.get("id"), str) and isinstance(r.get("path"), str) and isinstance(r.get("kind"), str)):
        return False
    if not all(_int(r.get(k, 0)) for k in ("line", "ordinal")) or not isinstance(r.get("service", ""), str):
        return False
    kind = r["kind"]
    if kind == "port":
        return _int(r.get("port"))
    if kind == "mount":
        return isinstance(r.get("mountPath"), str)
    if kind == "command":
        args = r.get("args", ())
        return isinstance(r.get("command", ""), str) and isinstance(args, (list, tuple)) and all(isinstance(a, str) for a in args)
    return True


def _clean_parts(parts: Any) -> dict | None:
    """``parts`` if it has the shape of a ``bindParts`` object (spec5 §5.3), else ``None``: an entry that does not is unbound."""
    if not isinstance(parts, dict):
        return None
    paths, port, host = parts.get("pathHmacs"), parts.get("port"), parts.get("host")
    if not (isinstance(parts.get("scheme"), str) and parts.get("hostKind") in _HOST_KINDS and isinstance(parts.get("hostHmac"), str)
            and isinstance(paths, (list, tuple)) and all(isinstance(p, str) for p in paths)
            and (port is None or _int(port)) and (host is None or isinstance(host, str))):
        return None
    return parts


def _id_ok(r: dict, repo: str) -> bool:
    """Is ``r['id']`` the id its own fields derive (``ids.config_entry_id`` / ``ids.launch_fact_id``, which need no key) and is its
    ``repoId`` its bundle's? A record that is not cannot be the one its id names, so it never claims that id."""
    if r.get("repoId", repo) != repo:
        return False
    try:
        want = (ids.config_entry_id(repo, r["path"], r["keyPath"]) if r["type"] == "configEntry"
                else ids.launch_fact_id(repo, r["path"], r["kind"], r["ordinal"]))
    except (KeyError, TypeError, ValueError):
        return False
    return r["id"] == want


def _launch_order(r: "_Rec") -> tuple:
    return (r.example, r.repo, r.path, r.raw.get("line", 0), r.raw.get("ordinal", 0), r.id)


class Binder:
    """The :class:`ccg.linker.link.Bindings` provider; see the module docstring.

    ``key``: the system HMAC key (read by the caller through ``ccg.atrest.hmac_key``; this class never reads one). ``repos``: manifest
    ``repoId -> path`` (system-relative, ``"."`` for the root). ``verify_ids`` (default on): accept a ``configEntry``/``launchFact`` only if its ``repoId`` is its bundle's and its id is the one its fields derive (hand-made
    test records with readable ids pass ``False``). ``units``: optional discovered units (anything with ``unitId``,
    ``repoId``, ``rootPath``), so a unit that was never built is ``target-not-mapped`` rather than unknown. ``packs``: loaded packs, for
    ``urlBindings`` and the kinds their mount paths serve (a ``config`` clause naming anything but ``keyPath`` is refused here, at build
    time). The launch facts are indexed once, here: a site costs the services it matches, not every service and fact.
    """

    def __init__(self, facts: PackFacts, bundles: list[list[dict]], *, key: bytes, repos: Mapping[str, str], packs: Iterable[dict] = (),
                 units: Iterable[Any] = (), verify_ids: bool = True):
        self._key, self._facts, self._packs = key, facts, list(packs)
        for pack in self._packs:
            for ub in pack.get("urlBindings", ()):
                for src in ub["urlSources"]:
                    if "config" in src and set(src["config"]) != {"keyPath"}:
                        raise ValueError(f"a urlBindings config clause may only name keyPath, got {sorted(src['config'])}")
        self._repos = {r: posixpath.normpath(p) for r, p in repos.items()}
        self._corpus = Corpus(bundles)
        self._mismatch: set[str] = set()
        self._residual_sites: set[str] = set()
        self._cache: dict[str, Binding | None] = {}
        self._units: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)     # (repo, rootPath) -> {unitId: built}
        self._files: set[tuple[str, str]] = set()
        self._file_repo: dict[str, str] = {}                                # file id -> its repository
        self._entries: dict[str, _Rec] = {}
        self._launch: dict[str, _Rec] = {}
        for u in units:
            self._units[(u.repoId, u.rootPath)].setdefault(u.unitId, False)
        pending: list[tuple[_Rec, bool]] = []
        seen: dict[str, dict] = {}
        clashes: set[str] = set()
        for records in bundles:
            records = [r for r in records if isinstance(r, dict)]
            header = next((r for r in records if r.get("type") == "header"), None)
            try:
                unit, repo, root = header["unitId"], header["unit"]["repoId"], header["unit"]["rootPath"]
            except (KeyError, TypeError):
                continue
            if not all(isinstance(v, str) for v in (unit, repo, root)):
                continue
            self._units[(repo, root)][unit] = True
            files = [f for f in records if f.get("type") == "file" and isinstance(f.get("path"), str)]
            roles = {(f.get("repoId", repo), f["path"]): f.get("role") for f in files}
            self._files |= set(roles)
            self._file_repo.update({f["id"]: f.get("repoId", repo) for f in files if isinstance(f.get("id"), str)})
            usable = valuehash.bind_parts_usable(header, key)
            carries = "bindKeyCheck" in header or any(r.get("type") == "configEntry" and "bindParts" in r for r in records)
            if carries and not usable:
                self._mismatch.add(unit)
            for r in records:
                kind = r.get("type")
                if kind not in ("configEntry", "launchFact") or not (_entry_ok(r) if kind == "configEntry" else _launch_ok(r)):
                    continue
                if verify_ids and not _id_ok(r, repo):                       # not the record its id names: it claims nothing
                    continue
                if r["id"] in seen and seen[r["id"]] != r:                   # a different record claims the id: neither is trusted
                    clashes.add(r["id"])                                     # (the same record in an overlapping unit's bundle is one)
                    continue
                seen[r["id"]] = r
                rrepo = r.get("repoId", repo)
                rec = _Rec(r["id"], rrepo, r["path"], unit, bool(r.get("example")) or roles.get((rrepo, r["path"])) == "example", r)
                (self._entries if kind == "configEntry" else self._launch)[r["id"]] = rec
                if kind == "configEntry":
                    pending.append((rec, usable))
        for cid in clashes:
            self._entries.pop(cid, None)
            self._launch.pop(cid, None)
        self._clashes = len(clashes)
        pending = [(rec, usable) for rec, usable in pending if rec.id in self._entries]
        self._index_launch()
        self._declared = self._declared_services()
        self._host_of_hash = {valuehash.entry_hmac(key, valuehash.BIND_HOST_TAG, s): s for s in self._declared if _SERVICE_LABEL.fullmatch(s)}
        self._parts: dict[str, _Url] = {}
        for rec, usable in sorted(pending, key=lambda p: p[0].id):
            url = self._entry_url(rec, usable)
            if url is not None:
                self._parts[rec.id] = url
        self._index_entries()
        self._launch_unit = {f["launchId"]: f["unitId"] for f in facts.launch_facts if f["fact"] == "launchUnit"}
        self._mounts_of: dict[str, list[dict]] = defaultdict(list)
        for f in facts.launch_facts:
            if f["fact"] == "mountPath":
                self._mounts_of[f["unitId"]].append(f)
        self._pack_ports: dict[int, list[dict]] = defaultdict(list)
        for f in facts.launch_facts:
            if f["fact"] == "port":
                self._pack_ports[f["port"]].append(f)
        self._compose_ports = self._index_compose_ports()
        self._pack_kinds: dict[str, set[str]] = defaultdict(set)
        for pack in self._packs:
            for ub in pack.get("urlBindings", ()):
                self._pack_kinds[pack["id"]] |= set(ub["kinds"])
        self._callsite_ids = sorted(self._corpus.callsites)

    # -- indexes (built once) ----------------------------------------------------------------------
    def _index_launch(self) -> None:
        self._svc: dict[str, list[_Rec]] = defaultdict(list)            # compose service (lowercase) -> its launch facts, in order
        self._servers_by: dict[str, list[_Rec]] = defaultdict(list)     # explicit (.mcp.json) server -> its command facts, in order
        self._commands_at: dict[tuple[str, str], list[_Rec]] = defaultdict(list)
        for rec in sorted(self._launch.values(), key=_launch_order):
            raw = rec.raw
            if raw.get("origin") == "compose":
                self._svc[(raw.get("service") or "").lower()].append(rec)
            if raw.get("origin") == "explicit" and raw["kind"] == "command":
                self._servers_by[raw.get("service")].append(rec)
            if raw["kind"] == "command":
                self._commands_at[(rec.repo, rec.path)].append(rec)
        for recs in self._servers_by.values():
            recs.sort(key=lambda r: (r.example, r.repo, r.path, r.id))
        for recs in self._commands_at.values():
            recs.sort(key=lambda r: r.id)

    def _index_entries(self) -> None:
        self._server_urls: dict[str, list[_Rec]] = defaultdict(list)    # mcp-json server -> its url entries that have parts
        self._compose_env: dict[str, list[_Rec]] = defaultdict(list)    # environment variable -> compose entries that set it, with parts
        for rec in sorted(self._entries.values(), key=lambda r: (r.example, r.repo, r.path, r.id)):
            if rec.id not in self._parts:
                continue
            raw = rec.raw
            if raw["format"] == "mcp-json" and (m := _SERVER_OF.match(raw["keyPath"])):
                self._server_urls[m.group(1)].append(rec)
            if raw["format"] == "compose" and (m := _COMPOSE_ENV.fullmatch(raw["keyPath"])):
                self._compose_env[m.group(2)].append(rec)

    def _index_compose_ports(self) -> dict[int, list[tuple[str, tuple[str, str], _Rec]]]:
        """port -> ``(service, (repoId, path), first port fact)`` of every declared service that lists it, per compose file."""
        out: dict[int, list[tuple[str, tuple[str, str], _Rec]]] = defaultdict(list)
        for name in sorted(self._declared):
            done: set[tuple[str, str, int]] = set()
            for f in self._svc.get(name, ()):
                key = (f.repo, f.path, f.raw["port"]) if f.raw["kind"] == "port" else None
                if key is not None and key not in done:
                    done.add(key)
                    out[f.raw["port"]].append((name, (f.repo, f.path), f))
        return out

    # -- the system's services ---------------------------------------------------------------------
    def _declared_services(self) -> set[str]:
        out: set[str] = set()
        for name, recs in self._svc.items():
            if name and any(not r.example for r in recs):
                out.add(name)
        for rec in self._entries.values():
            m = _COMPOSE_SERVICE.match(rec.raw["keyPath"])
            if rec.raw["format"] == "compose" and m and not rec.example:
                out.add(m.group(1).lower())
        return out

    def _service_launch(self, name: str, kind: str | None = None, ident: tuple[str, str] | None = None) -> list[_Rec]:
        """Launch facts of the Compose services called ``name``; with ``ident`` (``(repoId, path)`` of the compose file) only that service's.
        A service is ``(origin, repoId, path, service)`` (trust rule c): facts of two files never mix."""
        return [r for r in self._svc.get(name, ()) if (kind is None or r.raw["kind"] == kind) and (ident is None or (r.repo, r.path) == ident)]

    def _idents(self, name: str) -> list[tuple[str, str]]:
        """The compose files that define a service ``name`` (any launch fact of it), primary files first."""
        recs = self._svc.get(name, ())
        return sorted({(r.repo, r.path) for r in recs}, key=lambda i: (any(r.example for r in recs if (r.repo, r.path) == i), i))

    # -- URL parts ---------------------------------------------------------------------------------
    def _entry_url(self, rec: _Rec, usable: bool) -> _Url | None:
        raw = rec.raw
        parts = raw.get("bindParts") if usable else None
        if parts is None and isinstance(raw.get("bindValue"), str):
            parts = valuehash.bind_parts(self._key, raw["bindValue"], self._declared)
        return None if parts is None else self._from_parts(parts, rec.id, rec.example)          # a malformed ``parts`` binds nothing, alone

    def _text_url(self, text: str) -> _Url | None:
        parts = valuehash.bind_parts(self._key, text, self._declared) if "\x00" not in text else None
        return None if parts is None else self._from_parts(parts, None, False)

    @staticmethod
    def _from_parts(parts: dict, entry: str | None, example: bool) -> _Url | None:
        parts = _clean_parts(parts)
        if parts is None or forbidden_host(parts.get("host")):
            return None
        return _Url(parts["scheme"], parts.get("port"), parts["hostKind"], parts.get("host"), parts["hostHmac"],
                    tuple(parts["pathHmacs"]), entry, example)

    # -- paths, lexically (trust rules a, b) -------------------------------------------------------
    def _system_dir(self, rec: _Rec) -> str | None:
        base = self._repos.get(rec.repo)
        if base is None or rec.path.startswith("/") or "\x00" in rec.path:              # an absolute path would discard the base in `join`
            return None
        return posixpath.dirname(posixpath.normpath(posixpath.join(base, rec.path)))

    def _locate(self, target: str) -> tuple[str, str] | None:
        """``(repoId, repo-relative path)`` of a normalised system-relative path under a manifest repository (the deepest wins)."""
        best: tuple[str, str] | None = None
        for repo, base in self._repos.items():
            if base == ".":
                inside, rel = not (target == ".." or target.startswith(("../", "/"))), target
            elif target == base or target.startswith(base + "/"):
                inside, rel = True, "." if target == base else target[len(base) + 1:]
            else:
                inside = False
            if inside and (best is None or len(self._repos[best[0]]) < len(base)):
                best = (repo, rel)
        return best

    def _resolve_context(self, rec: _Rec) -> tuple[str, str] | Unmapped | None:
        """A compose ``mount`` (build context) as ``(repoId, rel)`` inside a repository, or the :class:`Unmapped` it ends in."""
        path = rec.raw["mountPath"]
        base = self._system_dir(rec)
        if base is None or path.startswith("/") or "\x00" in path:
            return Unmapped(None, ids.tuple_id(["unmapped", posixpath.normpath(path) if path else "."]))
        target = posixpath.normpath(posixpath.join(base, path))
        where = self._locate(target)
        return Unmapped(None, ids.tuple_id(["unmapped", target])) if where is None else where

    def _units_at(self, repo: str, rel: str) -> list[tuple[str, bool]]:
        return sorted(self._units.get((repo, rel), {}).items())

    # -- a service to its units --------------------------------------------------------------------
    def _service_chains(self, name: str, ident: tuple[str, str], port_entry: _Rec | None, port: int | None) -> list[_Chain]:
        chains: list[_Chain] = []
        for mount in self._service_launch(name, "mount", ident):
            got = self._resolve_context(mount)
            lead = [_bound(port_entry.id, port, "compose")] if port_entry is not None else []
            if isinstance(got, Unmapped):
                chains.append(_Chain(None, got, lead + [_bound(mount.id)], mount.example))
                continue
            for uid, built in self._units_at(*got):
                chains.append(_Chain(uid if built else None, None if built else Unmapped(uid, uid), lead + [_bound(mount.id)], mount.example))
        if chains:
            return chains
        for cmd in self._service_launch(name, "command", ident):
            uid = self._launch_unit.get(cmd.id)
            if uid is not None:
                lead = [_bound(port_entry.id, port, "compose")] if port_entry is not None else []
                chains.append(_Chain(uid, None, lead + [_bound(cmd.id)], cmd.example))
        return chains

    def _port_match(self, port: int) -> tuple[list[_Chain], bool]:
        """Loopback: units that listen on ``port`` by the packs' facts and by Compose, and whether a Compose match was made (trust rule e)."""
        chains: list[_Chain] = []
        for f in self._pack_ports.get(port, ()):
            entries = []
            for lid in f["launchIds"]:
                carries = (self._launch[lid].raw["kind"] == "env-default") == (f["origin"] == "env-default") if lid in self._launch else False
                entries.append(_bound(lid, port, f["origin"], f["packId"]) if carries else _bound(lid))
            ex = any(self._launch[l].example for l in f["launchIds"] if l in self._launch)
            chains.append(_Chain(f["unitId"], None, entries, ex))
        compose = False
        for name, ident, hit in self._compose_ports.get(port, ()):
            found = self._service_chains(name, ident, hit, port)
            chains += found
            compose |= bool(found)
        return chains, compose

    # -- one URL to its chains ---------------------------------------------------------------------
    def _url_chains(self, url: _Url, kind: str) -> tuple[list[_Chain], bool]:
        if url.host_kind == "ip":                          # a metadata host never gets here: `_from_parts` drops it (trust rule d)
            return [], False
        port = url.port if url.port is not None else _DEFAULT_PORT.get(url.scheme)
        residual = False
        if url.host_kind == "loopback":
            host = url.host
            if port is None:
                return [], False
            chains, residual = self._port_match(port)
        else:
            name = url.host if url.host_kind == "service" else self._host_of_hash.get(url.host_hmac)
            if name is None or name.lower() not in self._declared:
                return [], False
            host, chains = name.lower(), []
            for ident in self._idents(host):
                ports = self._service_launch(host, "port", ident)
                known = {f.raw["port"] for f in ports}
                if url.port is not None and known and url.port not in known:        # the service publishes or exposes other ports only
                    continue
                hit = next((f for f in ports if f.raw["port"] == url.port), None)
                chains += self._service_chains(host, ident, hit, url.port)
        chains = [c for c in chains if self._path_ok(c, url, host, kind)]
        lead = [_bound(url.entry)] if url.entry is not None else []
        for c in chains:
            c.entries = lead + c.entries
            c.example = c.example or url.example
        return chains, residual and bool(chains)

    def _path_ok(self, chain: _Chain, url: _Url, host: str, kind: str) -> bool:
        if chain.unit is None:
            return True
        mounts = [m for m in self._mounts_of.get(chain.unit, ()) if kind in self._pack_kinds.get(m["packId"], ())]
        if not mounts:
            return True
        for m in mounts:
            got = valuehash.mount_prefix_hmac(self._key, host, m["path"])
            if got is None:
                continue
            depth, digest = got
            if depth == 0:
                return True
            if len(url.path_hmacs) >= depth and url.path_hmacs[depth - 1] == digest and (m["pathMatch"] != "exact" or len(url.path_hmacs) == depth):
                return True
        return False

    # -- consumers to URLs -------------------------------------------------------------------------
    def _file_of(self, anchor: str) -> str | None:
        sym = self._corpus.symbols.get(anchor)
        return None if sym is None else sym["fileId"]

    def _own_urls(self, consumer: dict) -> list[_Url]:
        out: list[_Url] = []
        for item in self._facts.lifted:
            if item["siteId"] != consumer["siteId"] or consumer["kind"] != "http":
                continue
            for alt in item["alternatives"]:
                if bundle_url(alt) and (u := self._text_url(alt)) is not None:
                    out.append(u)
        return out

    def _env_urls(self, consumer: dict) -> list[_Url]:
        """Behaviour (5): the default text and the Compose-set values of the environment variable a Python consumer reads into its URL."""
        out: list[_Url] = []
        c = self._corpus
        for item in self._facts.lifted:
            if item["siteId"] != consumer["siteId"] or consumer["kind"] != "http":
                continue
            for d, _ in c.flows_into(item["siteId"], item["urlPath"]):
                cs = c.callsites.get(d["def"])
                if cs is None or not cs["calleeName"].endswith(_ENV_READS):
                    continue
                for var in c.arg_alternatives(cs, "a0") or ():
                    for text in c.arg_alternatives(cs, "a1") or ():
                        if bundle_url(text) and (u := self._text_url(text)) is not None:
                            out.append(u)
                    out += [self._parts[rec.id] for rec in self._compose_env.get(var, ())]
        return out

    def _pack_urls(self, consumer: dict) -> list[_Url]:
        out: list[_Url] = []
        c = self._corpus
        for pack in self._packs:
            for ub in pack.get("urlBindings", ()):
                if consumer["kind"] not in ub["kinds"]:
                    continue
                for src in ub["urlSources"]:
                    if "callsite" in src:
                        if consumer.get("server") is None:                  # a mention that names its server is bound by that server's entries only
                            out += self._callsite_urls(consumer, src)
                    elif "config" in src:
                        out += self._config_urls(consumer, src["config"])
        return out

    def _callsite_urls(self, consumer: dict, src: dict) -> list[_Url]:
        c, out = self._corpus, []
        my_file = self._file_of(consumer["anchorId"])
        if my_file is None:
            return out
        for cid in self._callsite_ids:
            site = c.callsites[cid]
            if not clause_hit("callsite", src["callsite"], site, c):
                continue
            site_file = c.file_id_of(site)
            same = site_file == my_file if src.get("scope", "file") == "file" else c.unit_of_file(site_file) == c.unit_of_file(my_file)
            if not same:
                continue
            for alt in c.arg_alternatives(site, src["arg"]) or ():
                if bundle_url(alt) and (u := self._text_url(alt)) is not None:
                    out.append(u)
        return out

    def _config_urls(self, consumer: dict, clause: dict) -> list[_Url]:
        """An ``mcp-json`` ``mcpServers.<server>.url`` entry binds the mentions of that server (the clause is checked when the binder is built)."""
        server = consumer.get("server")
        if server is None:
            return []
        return [self._parts[rec.id] for rec in self._mention_urls(consumer, server) if text_hit(clause["keyPath"], rec.raw["keyPath"])]

    def _mention_urls(self, consumer: dict, server: str) -> list[_Rec]:
        """The server's url entries a mention may bind through: an ``.mcp.json`` entry of the mention's own repository, no other (architect ruling, m5)."""
        repo = self._repo_of(consumer["anchorId"])
        return [r for r in self._server_urls.get(server, ()) if r.repo == repo and is_mcp_json(r.path)]

    # -- launch chain (6) --------------------------------------------------------------------------
    def _repo_of(self, anchor: str) -> str | None:
        """The repository a symbol (a consumer's anchor) lives in."""
        fid = self._file_of(anchor)
        return self._file_repo.get(fid) if fid is not None else None

    def _servers(self, name: str, repo: str | None) -> list[_Rec]:
        """The ``command`` launch facts of an ``.mcp.json`` server of that name in the mention's own repository (architect ruling, m5)."""
        return [r for r in self._servers_by.get(name, []) if r.repo == repo and is_mcp_json(r.path)]

    def _script_of(self, rec: _Rec) -> tuple[str, str] | None:
        base = self._system_dir(rec)
        if base is None:
            return None
        for text in [rec.raw.get("command", ""), *rec.raw.get("args", ())]:
            for tok in text.split():
                tok = _PROJECT_DIR.sub("", tok.strip("\"'"))
                if (bundle.REDACTED in tok or tok.startswith("/") or not _PATHISH.fullmatch(tok) or not ("/" in tok or tok.endswith(".sh"))):
                    continue
                where = self._locate(posixpath.normpath(posixpath.join(base, tok)))
                if where is not None and where in self._files:
                    return where
        return None

    def _launch_chains(self, server: str, repo: str | None) -> list[_Chain]:
        """Server command -> script -> the script's own launches -> the ``Dockerfile`` ``CMD`` beside it -> the unit; the chain stops at the
        first hop whose launch the packs resolved to a unit (``targetArg``), else ends at the unit rooted at the script's directory."""
        out: list[_Chain] = []
        for first in self._servers(server, repo):
            entries, uid = [_bound(first.id)], self._launch_unit.get(first.id)
            script = self._script_of(first)
            if uid is None and script is not None:
                repo, spath = script
                sdir = posixpath.dirname(spath) or "."
                docker = (repo, posixpath.join(sdir, "Dockerfile") if sdir != "." else "Dockerfile")
                for place in (script, docker):
                    found = self._commands_at.get(place, ())
                    for lf in found:
                        entries.append(_bound(lf.id))
                        uid = uid or self._launch_unit.get(lf.id)
                    if uid is not None:
                        break
                    if place == docker and found:
                        built = [u for u, ok in self._units_at(repo, sdir) if ok]       # a container was reached: the unit rooted at its directory
                        uid = built[0] if len(built) == 1 else None
            if uid is not None:
                out.append(_Chain(uid, None, entries, first.example))
        return out

    # -- the seam ----------------------------------------------------------------------------------
    def bind(self, consumer: dict) -> Binding | None:
        sid = consumer["siteId"]
        if sid not in self._cache:
            self._cache[sid] = self._bind(consumer)
        return self._cache[sid]

    def _bind(self, consumer: dict) -> Binding | None:
        chains: list[_Chain] = []
        residual = False
        urls = self._own_urls(consumer) + self._env_urls(consumer) + self._pack_urls(consumer)
        for url in urls:
            found, res = self._url_chains(url, consumer["kind"])
            chains += found
            residual |= res
        if consumer.get("server") is not None and consumer["kind"] == "mcp_tool" and not self._mention_urls(consumer, consumer["server"]):
            chains += self._launch_chains(consumer["server"], self._repo_of(consumer["anchorId"]))
        primary = [c for c in chains if not c.example]
        chains = primary or chains                                       # a chain through a primary file wins over example-role ones
        mapped = [c for c in chains if c.unit is not None]
        if mapped:
            by_unit: dict[str, list[BoundEntry]] = {}
            for c in mapped:
                by_unit.setdefault(c.unit, []).extend(c.entries)
            entries = tuple(e for c in mapped for e in c.entries)
            if residual:
                self._residual_sites.add(consumer["siteId"])
            return Binding(tuple(sorted(by_unit)), entries, weaken=None if primary else ("possible", "example-config"),
                           unit_entries=tuple((u, tuple(es)) for u, es in sorted(by_unit.items())))
        unmapped = sorted((c for c in chains if c.unmapped is not None), key=lambda c: c.unmapped.ref)
        if unmapped:
            return Binding((), tuple(unmapped[0].entries), unmapped=unmapped[0].unmapped)
        return None

    def key_mismatches(self) -> list[str]:
        """``bind-key-mismatch <unitId>`` per unit whose bundle carries ``bindParts`` it cannot use (spec5 §5.4): no hash, no value."""
        return [f"bind-key-mismatch {u}" for u in sorted(self._mismatch)]

    def counts(self) -> dict:
        """Figures merged into ``Links.counts``: the consumer sites bound by a loopback port through a Compose port fact (trust rule e), and the
        record ids dropped because two records claimed them."""
        return {"loopback_port_residual": len(self._residual_sites), "binding_duplicate_ids": self._clashes}


def bundle_url(text: str) -> bool:
    """True iff ``text`` looks like an absolute URL worth parsing (no unresolved piece, a scheme and an authority)."""
    return "\x00" not in text and "://" in text
