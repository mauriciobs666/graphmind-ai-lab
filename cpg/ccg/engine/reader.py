"""The reader layer (C-1001 P1-21): the one place the engine reads the store.

Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §5 (the query texts, character for character), §7 (the ``ccg_engine_<s>`` role: ``+graph.ro_query
+graph.explain +info`` on ``~ccg_<s>`` only), §8 P1-21; ``docs/plans/connected-code-graph-p1-answers.md`` P1-21 behaviour (1), (6), (7).

What it guarantees, by construction:

* **Read-only.** Every query goes out as ``GRAPH.RO_QUERY`` (the server itself refuses a write there), under a credential that holds no write command at all,
  and its text is one of the closed :data:`QUERIES` table, each passed through :func:`assert_read_only` at import and again at every read. A name that is
  not in the table is refused before any call, so no caller can send a text of its own.
* **No key or query text is ever built from input.** The graph key is ``ccg.sysname.live_key`` of a name ``check_system_name`` accepted; every caller value
  travels as a parameter (``$ids``, ``$pairs``, ``$repo``, ...); a ``<Label>`` in a templated name is a token of the fixed tables below, never caller text.
* **Bounded.** The timeout is sent on every call; ids and pairs chunk at ``config.CHUNK``; a reply that fills ``config.RESULTSET_SIZE`` halves the chunk and,
  at one id (or on an unchunked read), raises :class:`ResultTooLarge`. A reply is never truncated.
* **Everything read is untrusted text** (it originates in repos): rows come back as plain tuples of ``None``/bool/int/float/str/tuple, never interpreted, never
  formatted into a text or key, and an error message names the query and the exception class, never a value read.

Rows come back in a total order (``None`` last), whatever the server sent: a fact must not depend on read order (FR-19).
Consumed by ``ccg.engine.bfs`` (P1-22): the walk calls :meth:`CypherReader.read` with an expansion name and sorts nothing; ``count(x)`` of ``name_in``/``name_out`` is interpreted there.
``snapshot`` and ``units`` are registered and returned raw; the stamp and the staleness block are built from them in ``ccg.engine.staleness`` and ``ccg.engine.answer_build``.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ccg import sysname
from ccg.engine import config
from ccg.store import acl
from ccg.store import loader as L


class ReaderError(RuntimeError):
    """A refusal or a failed read. The message names the query and the failure class, never a value read from the graph."""


class GraphDataError(ReaderError):
    """A row the graph returned does not have the shape its query promises. The store is untrusted input: a malformed row is refused, never repaired."""


class ReadTimeout(ReaderError):
    """The server's per-query timeout fired."""


class ResultTooLarge(ReaderError):
    """A reply reached ``RESULTSET_SIZE`` and could not be split further: it may have been cut short, so it is refused, never truncated."""


# ---------------------------------------------------------------------------------------------
# The texts (``-p1-graph.md`` §5). ``<Label>``/``<idProp>``/``<HAS_REL>`` come from these fixed tables only.
# ---------------------------------------------------------------------------------------------

_POINT_ID = {"CallSite": "callSiteId", "Param": "pointId", "Ret": "pointId", "Cond": "pointId"}
_OWNER_REL = {"CallSite": "HAS_CALL", "Param": "HAS_PARAM", "Ret": "HAS_RETURN", "Cond": "HAS_COND"}
_DATA_IN = ("Param", "CallSite")
_DATA_OUT = ("CallSite", "Ret", "Cond")
_CONTROLLED_OUT = ("CallSite", "Ret", "Cond", "Param")
_D8_STATUS = "coalesce(e.status, 'confirmed'), coalesce(e.reason, 'resolved-call')"
_D8_DATA = "coalesce(d.status, 'confirmed'), coalesce(d.reason, 'intra-flow')"

#: the string literals a text may hold: the D8 defaults, coalesced in the query and nowhere else (``-p1-graph.md`` §5)
ALLOWED_LITERALS = frozenset({"confirmed", "resolved-call", "intra-flow"})

_ANCHORS = {
    "repo_exists": "MATCH (r:Repo {repoId: $repo}) RETURN r.repoId",
    "file_by_path": "MATCH (f:File {path: $path}) WHERE f.repoId = $repo RETURN f.fileId, f.unitId",
    "skipped_by_path": "MATCH (s:SkippedFile {path: $path}) WHERE s.repoId = $repo RETURN s.unitId, s.reason",
    "symbols_of_file": "MATCH (f:File {path: $path})-[:DEFINES]->(s:Symbol) WHERE f.repoId = $repo RETURN s.symbolId, s.qname, s.lineStart, s.lineEnd, s.ownedRegions, s.depth",
    "symbol_by_qname": "MATCH (s:Symbol {qname: $qname}) WHERE s.repoId = $repo RETURN s.symbolId, s.unitId, s.path, s.lineStart, s.lineEnd, s.ownedRegions",
    "symbol_by_name": "MATCH (s:Symbol {nameNorm: $name}) RETURN s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart",
    # the engine drops the rows that carry a launchKind (a launch entry is never an anchor); a query filter would need IS NULL
    "config_by_path": "MATCH (e:ConfigEntry {path: $path}) WHERE e.repoId = $repo RETURN e.entryId, e.unitId, e.line, e.keyPath, e.launchKind",
    "cp_by_key": "MATCH (cp:ContractPoint {keyNorm: $key}) RETURN cp.cpId, cp.kind, cp.keyNorm",
    # a label scan by design: the client-side prefix/template match needs every key of the kind (ContractPoint is small)
    "cp_keys": "MATCH (cp:ContractPoint) WHERE cp.kind = $kind RETURN cp.cpId, cp.keyNorm",
    "snapshot": "MATCH (s:Snapshot) RETURN s.snapshotId, s.verdictLogOffset, s.verdictLogHash, s.graphModelVersion, s.bundleSchemaVersion",
    "units": "UNWIND $ids AS i MATCH (u:Unit {unitId: i}) RETURN u.unitId, u.repoId, u.rootPath, u.language, u.buildStatus, u.builtCommit, u.builtAt, u.failedAt, u.failReason, u.flowAvailable",
}

_IMPACT = {
    "calls_in": f"UNWIND $ids AS i MATCH (c:CallSite)-[e:CALLS]->(s:Symbol {{symbolId: i}}) RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, {_D8_STATUS}",
    "name_in": "UNWIND $ids AS i MATCH (c:CallSite)-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(s:Symbol {symbolId: i}) OPTIONAL MATCH (c)-[r:CALLS]->(s) WITH i, c, s, r WHERE r IS NULL OPTIONAL MATCH (c)-[d:DISMISSED]->(s) WITH i, c, d WHERE d IS NULL OPTIONAL MATCH (c)-[x:CALLS]->() RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, count(x)",
    "controlled_in": "UNWIND $ids AS i MATCH (p)-[:CONTROLLED_BY]->(k:Cond {pointId: i}) RETURN i, labels(p)[0], coalesce(p.callSiteId, p.pointId), p.line",
    "to_in": "UNWIND $ids AS i MATCH (l:Link)-[:TO]->(s:Symbol {symbolId: i}) RETURN i, l.linkId, l.kind, l.status, l.reason, l.matcher",
    "via_in": "UNWIND $ids AS i MATCH (c:CallSite)-[v:VIA]->(l:Link {linkId: i}) RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, v.status, v.reason, v.consumerKey",
    "consumes_in": "UNWIND $ids AS i MATCH (c:CallSite)-[e:CONSUMES]->(cp:ContractPoint {cpId: i}) RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, e.status, e.reason",
    "on_in": "UNWIND $ids AS i MATCH (l:Link)-[:ON]->(cp:ContractPoint {cpId: i}) RETURN i, l.linkId, l.kind, l.status, l.reason, l.matcher",
    "provides_in": "UNWIND $ids AS i MATCH (s:Symbol)-[e:PROVIDES]->(cp:ContractPoint {cpId: i}) RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, e.status, e.reason",
    "bound_in": "UNWIND $ids AS i MATCH (l:Link)-[b:BOUND_BY]->(e:ConfigEntry {entryId: i}) RETURN i, l.linkId, l.kind, l.status, l.reason, b.siteId, b.ord",
}

_DEPENDS_ON = {
    "calls_out": f"UNWIND $ids AS i MATCH (c:CallSite {{callSiteId: i}})-[e:CALLS]->(s:Symbol) RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, {_D8_STATUS}",
    "name_out": "UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(s:Symbol) OPTIONAL MATCH (c)-[r:CALLS]->(s) WITH i, c, s, r WHERE r IS NULL OPTIONAL MATCH (c)-[d:DISMISSED]->(s) WITH i, c, s, d WHERE d IS NULL OPTIONAL MATCH (c)-[x:CALLS]->() RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, count(x)",
    "has_out:CallSite": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_CALL]->(p:CallSite) RETURN i, p.callSiteId, p.line, p.calleeName",
    "has_out:Param": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_PARAM]->(p:Param) RETURN i, p.pointId, p.line, p.index, p.name",
    "has_out:Ret": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_RETURN]->(p:Ret) RETURN i, p.pointId, p.line",
    "has_out:Cond": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_COND]->(p:Cond) RETURN i, p.pointId, p.line, p.code",
    "consumes_out": "UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[e:CONSUMES]->(cp:ContractPoint) RETURN i, cp.cpId, cp.kind, cp.keyNorm, e.status, e.reason",
    "provides_out": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[e:PROVIDES]->(cp:ContractPoint) RETURN i, cp.cpId, cp.kind, cp.keyNorm, e.status, e.reason",
    "via_out": "UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[v:VIA]->(l:Link) RETURN i, l.linkId, l.kind, l.status, l.reason, l.matcher, v.status, v.reason, v.consumerKey",
    "to_out": "UNWIND $ids AS i MATCH (l:Link {linkId: i})-[:TO]->(s:Symbol) RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart",
    "to_unit_out": "UNWIND $ids AS i MATCH (l:Link {linkId: i})-[:TO_UNIT]->(u:Unit) RETURN i, u.unitId, u.repoId, u.rootPath, u.buildStatus",
}

_EVIDENCE = {
    "lifted_of": "UNWIND $pairs AS q MATCH (l:Link {linkId: q[0]})-[r:LIFTED_VIA]->(t) WHERE r.siteId = q[1] RETURN q[0], q[1], r.ord, labels(t)[0], coalesce(t.callSiteId, t.symbolId), t.repoId, t.unitId, t.path, r.line, r.urlPath, r.verbPath, r.viaSiteId, r.viaLine, coalesce(t.calleeName, t.qname) ORDER BY q[0], q[1], r.ord",
    "bound_out": "UNWIND $pairs AS q MATCH (l:Link {linkId: q[0]})-[b:BOUND_BY]->(e:ConfigEntry) WHERE b.siteId = q[1] RETURN q[0], q[1], b.ord, e.entryId, e.repoId, e.unitId, e.path, e.line, e.keyPath, e.launchKind, e.protocol, e.port, e.envVar, e.mountPath, e.service, e.argv0, b.port, b.portOrigin, b.packId ORDER BY q[0], q[1], b.ord",
    "bound_of_link": "UNWIND $ids AS i MATCH (l:Link {linkId: i})-[b:BOUND_BY]->(e:ConfigEntry) RETURN i, e.entryId, b.siteId, b.ord, e.repoId, e.unitId, e.path, e.line, e.keyPath, e.launchKind, b.port, b.portOrigin, b.packId ORDER BY i, e.entryId, b.siteId, b.ord",
    "candidate_count": "UNWIND $pairs AS q MATCH (c:CallSite {callSiteId: q[0]})-[v:VIA]->(l:Link {linkId: q[1]}) MATCH (s:Symbol)-[:HAS_CALL]->(c) MATCH (s)-[:HAS_CALL]->(:CallSite)-[v2:VIA]->(l2:Link) WHERE v2.consumerKey = v.consumerKey RETURN q[0], q[1], count(DISTINCT l2)",
}


def _templated() -> dict[str, str]:
    out: dict[str, str] = {}
    for lab, rel in _OWNER_REL.items():
        out[f"owner_of:{lab}"] = (f"UNWIND $ids AS i MATCH (s:Symbol)-[:{rel}]->(p:{lab} {{{_POINT_ID[lab]}: i}}) "
                                  "RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, s.lineEnd, s.ownedRegions")
    for lab in _DATA_IN:
        out[f"data_in:{lab}"] = (f"UNWIND $ids AS i MATCH (u)-[d:DATA]->(p:{lab} {{{_POINT_ID[lab]}: i}}) "
                                 f"RETURN i, labels(u)[0], coalesce(u.callSiteId, u.pointId), u.line, d.fromPath, d.toPath, {_D8_DATA}")
    for lab in _DATA_OUT:
        out[f"data_out:{lab}"] = (f"UNWIND $ids AS i MATCH (p:{lab} {{{_POINT_ID[lab]}: i}})-[d:DATA]->(t) "
                                  f"RETURN i, labels(t)[0], coalesce(t.callSiteId, t.pointId), t.line, d.fromPath, d.toPath, {_D8_DATA}")
    for lab in _CONTROLLED_OUT:
        out[f"controlled_out:{lab}"] = (f"UNWIND $ids AS i MATCH (p:{lab} {{{_POINT_ID[lab]}: i}})-[:CONTROLLED_BY]->(k:Cond) "
                                        "RETURN i, k.pointId, k.line, k.code")
    return out


#: name -> Cypher text. The closed table: nothing outside it is ever sent (the tests replace an entry to prove the read-time lint; production code never does).
QUERIES: dict[str, str] = {**_ANCHORS, **_IMPACT, **_DEPENDS_ON, **_EVIDENCE, **_templated()}

_NAME_RE = re.compile(r"\A[a-z_]+(:[A-Za-z]+)?\Z")


# ---------------------------------------------------------------------------------------------
# The read-only lint.
# ---------------------------------------------------------------------------------------------

#: any of these as a whole word (case-insensitive) is not a read. ``OR`` is on the list by the plan's rule (no ``OR`` filter: it defeats the index seek).
_FORBIDDEN_WORDS = frozenset({"CREATE", "MERGE", "SET", "DELETE", "DETACH", "REMOVE", "DROP", "CALL", "FOREACH", "LOAD", "CSV", "OR"})
_PRINTABLE = re.compile(r"\A[ -~]+\Z")
_LITERAL = re.compile(r"'([^'\\]*)'")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def assert_read_only(text: str) -> None:
    """Raise :class:`ReaderError` unless ``text`` is one read-only statement of the shape this engine sends.

    Strict by design (it fails closed): printable ASCII on one line, one statement (no ``;``), no comment, no backtick or double quote, string literals
    only from :data:`ALLOWED_LITERALS`, and none of :data:`_FORBIDDEN_WORDS` as a whole word outside those literals.
    """
    if not isinstance(text, str) or not text.strip() or not _PRINTABLE.match(text):
        raise ReaderError("query text is not read-only: it is not one line of printable ASCII")
    if ";" in text or "//" in text or "/*" in text or "`" in text or '"' in text:
        raise ReaderError("query text is not read-only: a statement separator, comment, quoted identifier or double quote")
    bad = [m for m in _LITERAL.findall(text) if m not in ALLOWED_LITERALS]
    if bad:
        raise ReaderError("query text is not read-only: a string literal outside the D8 defaults")
    rest = _LITERAL.sub("", text)
    if "'" in rest:
        raise ReaderError("query text is not read-only: an unterminated string literal")
    words = {w.upper() for w in _WORD.findall(rest)}
    if hit := sorted(words & _FORBIDDEN_WORDS):
        raise ReaderError(f"query text is not read-only: it holds {hit[0]}")


def _validate_table() -> None:
    for name, text in QUERIES.items():
        if not _NAME_RE.match(name):
            raise ReaderError(f"query name {name!r} is not a recipe name")
        assert_read_only(text)


_validate_table()


# ---------------------------------------------------------------------------------------------
# Parameters and rows.
# ---------------------------------------------------------------------------------------------

_PARAM = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_SCALARS = (str, int, float, bool, type(None))


def _check_params(name: str, text: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ReaderError(f"query {name!r}: parameters are a mapping")
    wanted = set(_PARAM.findall(text))
    if set(params) != wanted:
        raise ReaderError(f"query {name!r}: parameters are {sorted(wanted)}")
    for key, value in params.items():
        if key == "ids":
            ok = isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)
        elif key == "pairs":
            ok = isinstance(value, (list, tuple)) and all(
                isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, str) for v in p) for p in value)
        else:
            ok = isinstance(value, str)
        if not ok:
            raise ReaderError(f"query {name!r}: parameter {key!r} has the wrong shape")
    return dict(params)


def _cell(name: str, value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_cell(name, v) for v in value)
    if isinstance(value, _SCALARS):
        return value
    raise GraphDataError(f"query {name!r} returned a value of type {type(value).__name__}")


def _cell_key(value: Any) -> tuple:
    """A total order over cells: ``None`` last, otherwise by type name then value (a list as the tuple of its keys)."""
    if value is None:
        return (1, "", 0)
    if isinstance(value, tuple):
        return (0, "tuple", tuple(_cell_key(v) for v in value))
    return (0, type(value).__name__, value)


def _row_key(row: tuple) -> tuple:
    return tuple(_cell_key(c) for c in row)


@dataclass(frozen=True)
class BoundTarget:
    """One ``configEntry``-origin ``BOUND_BY`` target of a link, with the evidence of the edge the item takes (``-answer-spec2.md`` §3.3)."""
    link_id: str
    entry_id: str
    site_id: str
    ord: int
    repo_id: str | None
    unit_id: str | None
    path: str | None
    line: int | None
    key_path: str
    port: int | None
    port_origin: str | None
    pack_id: str | None


class Reader(Protocol):
    """What the walk and the anchors use. ``read`` returns the rows of one registered query as plain tuples in a total order."""
    system: str

    def read(self, name: str, params: Mapping[str, Any] | None = None) -> list[tuple]: ...
    def bound_of_link(self, link_ids: Sequence[str]) -> list[BoundTarget]: ...
    def recipe_queries(self) -> list[dict[str, str]]: ...


class CypherReader:
    """The live reader over one falkordb-py client: ``GRAPH.RO_QUERY`` on ``live_key(system)``, as whatever credential the client holds (``ccg_engine_<s>``)."""

    def __init__(self, db, system: str, *, timeout_ms: int | None = None):
        try:
            self.system = sysname.check_system_name(system)
        except sysname.SystemNameError as exc:
            raise ReaderError(f"invalid system name ({exc})") from None
        timeout_ms = config.QUERY_TIMEOUT_MS if timeout_ms is None else timeout_ms
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 1:
            raise ReaderError("the query timeout is a positive integer number of milliseconds")
        self._timeout_ms = timeout_ms
        self._graph = db.select_graph(sysname.live_key(self.system))
        self._ran: dict[str, str] = {}

    @classmethod
    def for_system(cls, system: str, host: str, port: int, *, timeout_ms: int | None = None) -> CypherReader:
        """Connect as ``ccg_engine_<system>`` with the password ``ccg.store.acl.read_credentials`` returns (``CCG_CREDENTIALS_DIR``). Port 6379 is refused."""
        try:
            sysname.check_system_name(system)
        except sysname.SystemNameError as exc:
            raise ReaderError(f"invalid system name ({exc})") from None
        password = acl.read_credentials(system)["engine"]
        try:
            db = L.connect(host, port, username=sysname.acl_user("engine", system), password=password)
        except L.LoadError as exc:
            raise ReaderError(str(exc)) from None
        return cls(db, system, timeout_ms=timeout_ms)

    # -- the one read path ------------------------------------------------------------------------
    def _once(self, name: str, text: str, params: dict[str, Any]) -> list[tuple]:
        self._ran[name] = text
        try:
            result = self._graph.ro_query(text, params, self._timeout_ms)
        except Exception as exc:
            # ``from None`` on all three: the server's text may quote graph data, so it is neither in the message nor kept in the traceback chain
            msg = str(exc).lower()
            if "timed out" in msg:
                raise ReadTimeout(f"query {name!r} timed out after {self._timeout_ms} ms") from None
            if "timeout_max" in msg:
                raise ReaderError(f"query {name!r}: the timeout of {self._timeout_ms} ms exceeds the server's TIMEOUT_MAX") from None
            raise ReaderError(f"query {name!r} failed ({type(exc).__name__})") from None
        return [tuple(_cell(name, c) for c in row) for row in result.result_set]

    def read(self, name: str, params: Mapping[str, Any] | None = None) -> list[tuple]:
        text = QUERIES.get(name) if isinstance(name, str) else None
        if text is None:
            raise ReaderError(f"unknown query {name!r}" if isinstance(name, str) and len(name) < 64 else "unknown query")
        assert_read_only(text)
        params = _check_params(name, text, params)
        chunked = next((k for k in ("ids", "pairs") if k in params), None)
        if chunked is None:
            rows = self._once(name, text, params)
            if len(rows) >= config.RESULTSET_SIZE:
                raise ResultTooLarge(f"query {name!r} filled RESULTSET_SIZE ({config.RESULTSET_SIZE}); it may be truncated")
            return sorted(rows, key=_row_key)
        items, out, chunk, i = list(params[chunked]), [], config.CHUNK, 0
        while i < len(items):
            part = items[i:i + chunk]
            rows = self._once(name, text, {**params, chunked: part})
            if len(rows) >= config.RESULTSET_SIZE:
                if len(part) <= 1:
                    raise ResultTooLarge(f"query {name!r} filled RESULTSET_SIZE ({config.RESULTSET_SIZE}) for one id; it may be truncated")
                chunk = max(1, len(part) // 2)
                continue
            out.extend(rows)
            i += len(part)
        return sorted(out, key=_row_key)

    def recipe_queries(self) -> list[dict[str, str]]:
        """Every text run so far, once, by name: the answer's ``recipe.queries`` (``{"name", "cypher"}``, sorted by name)."""
        return [{"name": n, "cypher": t} for n, t in sorted(self._ran.items())]

    # -- the one typed read: the engine's pick (``-p1-graph.md`` §5, ``-answer-spec2.md`` §3.3) --------------------
    def bound_of_link(self, link_ids: Sequence[str]) -> list[BoundTarget]:
        """Per (link, entry) the first row, in ``(siteId, ord)`` order, of an entry that has a ``keyPath`` and no ``launchKind`` (launch entries are never items)."""
        if isinstance(link_ids, str):
            raise ReaderError("bound_of_link: ids are a list")
        picked: dict[tuple[str, str], BoundTarget] = {}
        for row in self.read("bound_of_link", {"ids": list(link_ids)}):
            if len(row) != 13:
                raise GraphDataError("query 'bound_of_link' returned a malformed row")
            link, entry, site, ord_, repo, unit, path, line, key_path, launch, port, origin, pack = row
            if not (isinstance(link, str) and isinstance(entry, str) and isinstance(site, str) and isinstance(ord_, int)):
                raise GraphDataError("query 'bound_of_link' returned a malformed row")
            if not isinstance(key_path, str) or launch is not None:
                continue
            picked.setdefault((link, entry), BoundTarget(link, entry, site, ord_, repo, unit, path, line, key_path, port, origin, pack))
        return list(picked.values())
