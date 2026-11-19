"""What every ``ccg`` client shares (C-1001 P1-25a): system and endpoint resolution, the asker's identity, the four client operations, and the one raw connection.

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-25 behaviours (1)-(5), (8), (9); ``-answer-spec2.md`` §2 (``cacheBypass``), §4.1 (no depth bound),
§5.1 (the flow notice is never dropped); ``-egress-spec3.md`` §10.1 (this is the one client module that opens ``raw_query``'s ``ccg_raw_<s>`` connection).
The CLI (``clients/cli/main.py``) and the MCP server (P1-25b) are thin wrappers over :class:`Client`; neither re-implements a rule below.

* **Structured request.** ``{"useCase": "impact" | "depends-on", "anchor": {...}}`` (:func:`parse_request`; ``scope`` and a depth bound are not part of P1). The anchor is one of
  ``{"kind": "symbol", "repo", "qname"}``, ``{"kind": "name", "name", "repo"?}``, ``{"kind": "location", "repo", "path", "line"? | "symbol"?}``,
  ``{"kind": "contract", "key", "contractKind"?}`` or ``{"kind": "config", "repo", "path", "keyPath"?}``. Any other key, use case or kind is a typed error, never a default.
  An anchor that does not resolve to exactly one candidate is a typed ``anchor-ambiguous``/``anchor-unresolved`` error carrying the candidates (``resolve`` lists them);
  the answer's own ambiguous/unresolved documents belong to the plain-language front (P1-26).
* **Plain-language question.** :meth:`Client.ask` takes ``question=`` and hands it to the ``front`` seam, ``front(question, reader=..., cache_bypass=...) -> canonical answer``.
  The default seam is ``None``, so a question returns the typed ``not-built`` error. The front is injected: ``Client.__init__`` only stores ``front=``, and the CLI's default factory passes none, so P1-26 must also edit this module (or the CLI factory) to wire its front in. ``cache_bypass`` reaches the
  front and has no effect on a structured request, nothing records it there (spec2 §2).
* **Validate first, render only a valid document** (8): :meth:`Client.ask` validates what the builder or the front returned before it leaves, and :func:`render_answer` validates
  before it calls a renderer.
* **The flow notice is never dropped** (9): the canonical document is returned unchanged, so ``notices`` is in it; the text and Markdown forms are ``ccg.render.render_text`` and
  ``render_markdown``, which print it.
* **raw_query** (4): ``GRAPH.RO_QUERY`` on the live graph as ``ccg_raw_<system>`` and no other credential; the database, not this module, refuses a write (the role holds no
  ``GRAPH.QUERY`` or ``GRAPH.PROFILE``). A time limit is sent on every call (default ``ccg.engine.config.QUERY_TIMEOUT_MS``, the pinned instance's ``TIMEOUT_MAX``); the server refuses
  a larger one, and the raw role cannot read ``TIMEOUT_MAX`` to check. The connection's socket timeout is the limit plus :data:`SOCKET_MARGIN_S`, so the server's own timeout
  answers first; a reply of ``RESULTSET_SIZE`` rows may have been cut short by the server and is a typed error, never presented as complete. A time limit bounds wall time, not memory.
* **Identity** (5): the asker is the launching OS user (:func:`asker`). Access control is P4 (FR-24): in P1 ``raw_query`` is open to the local user.
* **Network** (7): the client opens a socket to FalkorDB only (``CCG_FALKORDB_HOST``/``CCG_FALKORDB_PORT``, default ``127.0.0.1:6390``; port 6379 is refused).
"""
from __future__ import annotations

import dataclasses
import os
import pwd
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ccg import answer, atrest, sysname
from ccg.engine import anchors as A
from ccg.engine import config
from ccg.engine.answer_build import build_answer
from ccg.engine.reader import CypherReader, ReaderError
from ccg.manifest import ManifestError, load_manifest
from ccg.render import render_markdown, render_text
from ccg.store import acl
from ccg.store import loader as L

#: the default system manifest (where ``ccg.refresh`` takes it from); ``ccg.refresh`` itself is imported inside :meth:`Client.refresh` so the clients load no broker module at run time
DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "ccg" / "system.yaml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6390
USE_CASES = ("impact", "depends-on")
FORMATS = ("json", "text", "markdown")
_ANCHOR_KEYS = {
    "symbol": ({"repo", "qname"}, set()),
    "name": ({"name"}, {"repo"}),
    "location": ({"repo", "path"}, {"line", "symbol"}),
    "contract": ({"key"}, {"contractKind"}),
    "config": ({"repo", "path"}, {"keyPath"}),
}
#: seconds added to the query time limit to form the raw connection's socket timeout (the server's timeout must fire before the socket's)
SOCKET_MARGIN_S = 5


class ClientError(RuntimeError):
    """A typed, value-free failure every client reports the same way: ``code`` is the machine-readable kind, ``detail`` extra JSON-able context."""

    code = "client-error"

    def __init__(self, message: str, *, detail: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.detail = dict(detail or {})

    def payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": str(self), **({"detail": self.detail} if self.detail else {})}}


class BadRequest(ClientError):
    code = "bad-request"


class NotBuilt(ClientError):
    """A use case, input or surface P1 does not build. Never answered with an empty answer."""
    code = "not-built"


class AnchorNotResolved(ClientError):
    """The anchor is not exactly one candidate; ``detail`` lists the status and the candidates (what ``resolve`` returns)."""

    def __init__(self, message: str, *, status: str, detail: Mapping[str, Any]):
        super().__init__(message, detail=detail)
        self.code = f"anchor-{status}"


class StoreError(ClientError):
    code = "store-error"


class NoCredentials(ClientError):
    code = "no-credentials"


class RawRefused(ClientError):
    """The database refused the raw query (a write, ``GRAPH.QUERY``, ``GRAPH.PROFILE``, a permission, a time limit above ``TIMEOUT_MAX``)."""
    code = "raw-refused"


class ResultTooLarge(ClientError):
    """The reply filled the server's ``RESULTSET_SIZE``, so it may be truncated (as ``CypherReader`` treats it); narrow the query or add ``LIMIT``."""
    code = "result-too-large"


class RefreshFailed(ClientError):
    code = "refresh-failed"


# --------------------------------------------------------------------------------------------- system, endpoint, identity

def resolve_system(explicit: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """The system the client works on: the explicit name, else ``CCG_SYSTEM``, else the default manifest's ``system``. Always a name ``check_system_name`` accepted."""
    env = os.environ if env is None else env
    name = explicit or env.get("CCG_SYSTEM")
    if not name:
        try:
            name = load_manifest(DEFAULT_MANIFEST).system
        except (ManifestError, OSError) as exc:
            raise BadRequest(f"no system given and the default manifest is unusable ({type(exc).__name__})") from None
    try:
        return sysname.check_system_name(name)
    except sysname.SystemNameError as exc:
        raise BadRequest(f"invalid system name ({exc})"[:200]) from None


def endpoint(env: Mapping[str, str] | None = None) -> tuple[str, int]:
    """``(host, port)`` of FalkorDB from ``CCG_FALKORDB_HOST``/``CCG_FALKORDB_PORT`` (default ``127.0.0.1:6390``); port 6379 (``falkordb-dev``) is refused."""
    env = os.environ if env is None else env
    host = env.get("CCG_FALKORDB_HOST") or DEFAULT_HOST
    raw = env.get("CCG_FALKORDB_PORT") or str(DEFAULT_PORT)
    try:
        port = int(raw)
    except ValueError:
        raise BadRequest("CCG_FALKORDB_PORT is an integer") from None
    if not 1 <= port <= 65535:
        raise BadRequest("CCG_FALKORDB_PORT is a port number")
    if port == L.FORBIDDEN_PORT:
        raise BadRequest(f"port {L.FORBIDDEN_PORT} is falkordb-dev; ccg never connects to it")
    return host, port


def asker() -> str:
    """The launching OS user (P1's whole identity model; access control is P4, FR-24)."""
    uid = os.getuid()
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return f"uid:{uid}"


# --------------------------------------------------------------------------------------------- request parsing

def parse_request(request: Any) -> tuple[str, dict[str, Any]]:
    """``(use case, anchor spec)`` of a structured request, strictly: unknown keys, use cases and anchor kinds are errors (never ignored, never defaulted)."""
    if not isinstance(request, Mapping):
        raise BadRequest("a structured request is an object")
    unknown = set(request) - {"useCase", "anchor"}
    if "scope" in unknown:
        raise NotBuilt("a request scope is not built in this phase")
    if unknown:
        raise BadRequest("a structured request holds useCase and anchor only")
    use_case = request.get("useCase")
    if not isinstance(use_case, str):
        raise BadRequest("useCase is a string")
    if use_case not in USE_CASES:
        raise NotBuilt(f"the use case {use_case[:40]!r} is not built in this phase", detail={"built": list(USE_CASES)})
    spec = request.get("anchor")
    if not isinstance(spec, Mapping):
        raise BadRequest("anchor is an object")
    return use_case, dict(spec)


def _resolve_anchor(reader, spec: Mapping[str, Any]) -> A.Resolution:
    kind = spec.get("kind")
    if kind not in _ANCHOR_KEYS:
        raise BadRequest(f"anchor kind is one of {sorted(_ANCHOR_KEYS)}")
    need, may = _ANCHOR_KEYS[kind]
    keys = set(spec) - {"kind"}
    if not need <= keys or keys - need - may:
        raise BadRequest(f"a {kind} anchor holds {sorted(need)} and optionally {sorted(may)}")
    try:
        if kind == "symbol":
            return A.resolve_symbol(reader, spec["repo"], spec["qname"])
        if kind == "name":
            return A.resolve_symbol_by_name(reader, spec["name"], spec.get("repo"))
        if kind == "location":
            return A.resolve_location(reader, spec["repo"], spec["path"], spec.get("line"), spec.get("symbol"))
        if kind == "contract":
            return A.resolve_contract(reader, spec["key"], spec.get("contractKind"))
        return A.resolve_config(reader, spec["repo"], spec["path"], spec.get("keyPath"))
    except A.AnchorError as exc:
        raise BadRequest(f"unusable anchor: {exc}"[:200]) from None


def anchor_json(anchor: Any) -> dict[str, Any]:
    """One resolved candidate as a plain JSON object with its ids (``type`` is the answer's anchor type)."""
    kind = {A.SymbolAnchor: "symbol", A.ContractAnchor: "contract", A.ConfigAnchor: "config"}[type(anchor)]
    body = dataclasses.asdict(anchor)
    return {"type": kind, **{_camel(k): v for k, v in body.items()}}


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)


def resolution_json(res: A.Resolution) -> dict[str, Any]:
    out: dict[str, Any] = {"status": res.status, "candidates": [anchor_json(a) for a in (res.anchors or res.options)]}
    if res.status == "ambiguous":
        out["candidatesTotal"] = res.options_total
    for key, value in (("state", res.state), ("unitId", res.unit_id), ("reason", res.reason), ("via", res.via), ("detail", res.detail)):
        if value is not None:
            out[key] = value
    return out


# --------------------------------------------------------------------------------------------- render

def render_answer(doc: dict, fmt: str) -> str:
    """``doc`` as ``json`` (the canonical bytes), ``text`` (``ccg.render.render_text``) or ``markdown``. The document is validated FIRST: an invalid one raises ``AnswerError``
    and no renderer runs. The notice is in every form (JSON: the field; text/markdown: the renderers' notice)."""
    answer.validate(doc)
    if fmt == "json":
        return answer.dumps(doc).decode("utf-8")
    if fmt == "text":
        return render_text(doc)
    if fmt == "markdown":
        return render_markdown(doc)
    raise BadRequest(f"format is one of {list(FORMATS)}")


# --------------------------------------------------------------------------------------------- the raw connection (the only one a client opens)

def _open_raw(host: str, port: int, *, username: str, password: str, socket_timeout: float = config.QUERY_TIMEOUT_MS / 1000 + SOCKET_MARGIN_S):
    """A falkordb-py client for ``raw_query``: one connection, zero retries, ``socket_timeout`` (``raw_query`` passes the query time limit plus :data:`SOCKET_MARGIN_S`).
    Port 6379 is refused before any socket exists."""
    if port == L.FORBIDDEN_PORT:
        raise BadRequest(f"port {L.FORBIDDEN_PORT} is falkordb-dev; ccg never connects to it")
    from falkordb import FalkorDB
    return FalkorDB(host=host, port=port, username=username, password=password, socket_timeout=socket_timeout, socket_connect_timeout=10, retry=L.zero_retry())


def _plain(value: Any) -> Any:
    """A result cell as plain JSON data: scalars and lists as they are, a node/edge/path as an object, anything else as its text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "labels") and hasattr(value, "properties"):
        return {"labels": list(value.labels or []), "properties": _plain(value.properties)}
    if hasattr(value, "relation") and hasattr(value, "properties"):
        return {"relation": value.relation, "properties": _plain(value.properties)}
    if hasattr(value, "nodes") and hasattr(value, "edges") and callable(value.nodes):
        return {"nodes": _plain(value.nodes()), "edges": _plain(value.edges())}
    return str(value)


def _safe(value: object, cap: int = 120) -> str:
    """``value`` as one printable-ASCII token (control characters and non-ASCII backslash-escaped), cut at ``cap``; same rule as ``ccg.refresh.safe_text``."""
    text = str(value).encode("unicode_escape", "backslashreplace").decode("ascii")
    return text if len(text) <= cap else text[:cap - 3] + "..."


def _credential(system: str, role: str) -> str:
    try:
        return acl.read_credentials(system)[role]
    except (OSError, atrest.AtRestError, acl.AclError, KeyError) as exc:
        raise NoCredentials(f"the {role} credential of {system!r} is not readable ({type(exc).__name__})") from None


# --------------------------------------------------------------------------------------------- the client

class Client:
    """The four operations of one system. Everything that opens a connection or builds an answer is injectable, so a test runs offline.

    ``reader_factory(system, host, port) -> Reader`` (default: ``CypherReader.for_system``, the ``ccg_engine_<s>`` role); ``connect`` opens the ``build``/``verdict``
    connections of a refresh (default ``ccg.store.loader.connect``); ``raw_connect`` opens ``raw_query``'s (default :func:`_open_raw`); ``builder`` is ``build_answer``;
    ``front`` is the plain-language seam (module docstring)."""

    def __init__(self, system: str | None = None, *, host: str | None = None, port: int | None = None, env: Mapping[str, str] | None = None,
                 reader_factory: Callable[..., Any] | None = None, connect: Callable[..., Any] | None = None, raw_connect: Callable[..., Any] | None = None,
                 builder: Callable[..., dict] | None = None, front: Callable[..., dict] | None = None):
        self.system = resolve_system(system, env)
        env_host, env_port = endpoint(env) if host is None or port is None else (host, port)
        self.host, self.port = host or env_host, port or env_port
        if self.port == L.FORBIDDEN_PORT:
            raise BadRequest(f"port {L.FORBIDDEN_PORT} is falkordb-dev; ccg never connects to it")
        self._reader_factory = reader_factory or self._engine_reader
        self._connect = connect or L.connect
        self._raw_connect = raw_connect or _open_raw
        self._builder = builder or build_answer
        self._front = front

    def _engine_reader(self, system: str, host: str, port: int):
        return CypherReader.for_system(system, host, port)

    def _reader(self):
        try:
            return self._reader_factory(self.system, self.host, self.port)
        except ReaderError as exc:
            raise StoreError(str(exc)) from None
        except (atrest.AtRestError, acl.AclError, FileNotFoundError) as exc:
            raise NoCredentials(f"the engine credential of {self.system!r} is not readable ({type(exc).__name__})") from None
        except Exception as exc:  # noqa: BLE001 - an unreachable or refusing store (redis/falkordb errors are not ReaderErrors): typed, value-free, never a traceback
            raise StoreError(f"the store is not reachable ({type(exc).__name__})") from None

    # (2)
    def resolve(self, anchor: Mapping[str, Any]) -> dict[str, Any]:
        """The candidates of ``anchor`` with their ids (code location, symbol name or qname, contract point, config entry), whatever their number."""
        if not isinstance(anchor, Mapping):
            raise BadRequest("anchor is an object")
        try:
            return resolution_json(_resolve_anchor(self._reader(), anchor))
        except ReaderError as exc:
            raise StoreError(str(exc)) from None

    # (1)
    def ask(self, request: Mapping[str, Any] | None = None, *, question: str | None = None, cache_bypass: bool = False) -> dict:
        """The validated canonical answer to a structured ``request`` or a plain-language ``question`` (exactly one). ``cache_bypass`` is for the question path."""
        if (request is None) == (question is None):
            raise BadRequest("ask takes a structured request or a question, one of them")
        if not isinstance(cache_bypass, bool):
            raise BadRequest("cacheBypass is a boolean")
        if question is not None:
            if not isinstance(question, str) or not question.strip():
                raise BadRequest("a question is a non-empty string")
            if self._front is None:
                raise NotBuilt("the plain-language front is not built yet (P1-26); send a structured request",
                               detail={"surface": "plain-language-question"})
            doc = self._front(question, reader=self._reader(), cache_bypass=cache_bypass)
        else:
            use_case, spec = parse_request(request)
            try:
                res = _resolve_anchor(self._reader(), spec)
                if res.status != "resolved":
                    raise AnchorNotResolved(f"the anchor is {res.status}; resolve it to pick one candidate", status=res.status, detail=resolution_json(res))
                doc = self._builder(self._reader(), res.anchors[0], question=use_case)     # a fresh reader: ``recipe.queries`` lists the walk's reads, not the resolver's (it is in ``facts_bytes``, AC-16)
            except ReaderError as exc:
                raise StoreError(str(exc)) from None
        try:
            answer.validate(doc)
        except answer.AnswerError as exc:
            raise StoreError(f"the answer does not validate ({str(exc)[:80]})") from None
        return doc

    # (3)
    def refresh(self, scope: Sequence[str] | None = None) -> dict[str, Any]:
        """Rebuild the system from its local clones (``ccg.refresh.refresh``) as the ``build`` and ``verdict`` users; ``scope`` is a list of repoIds to rebuild (None: all)."""
        if scope is not None and (isinstance(scope, str) or not all(isinstance(s, str) for s in scope)):
            raise BadRequest("scope is a list of repoIds")
        from ccg import refresh as refresh_mod                    # lazy: a client that never refreshes loads no broker module (``-egress-spec3.md`` §2.1)
        creds = {role: _credential(self.system, role) for role in ("build", "verdict")}
        try:
            build_db, verdict_db = (self._connect(self.host, self.port, username=sysname.acl_user(role, self.system), password=creds[role]) for role in ("build", "verdict"))
            report = refresh_mod.refresh(self.system, scope, target=refresh_mod.Target(L.LiveStore(build_db, self.system), build_db, verdict_db))
        except (refresh_mod.RefreshError, L.LoadError) as exc:
            raise RefreshFailed(str(exc)[:200]) from None
        except Exception as exc:  # noqa: BLE001 - a connection or server failure of the refresh: typed, value-free
            raise StoreError(f"the refresh failed against the store ({type(exc).__name__})") from None
        return {"system": report.system, "snapshotId": report.snapshot_id, "scope": list(report.scope) if report.scope else None,
                "units": [{"unitId": u.unit_id, "repoId": u.repo_id, "language": u.language, "status": u.status, "failReason": u.fail_reason} for u in report.units],
                "report": report.lines()}

    # (4), (5)
    def raw_query(self, cypher: str, *, timeout_ms: int | None = None) -> dict[str, Any]:
        """``GRAPH.RO_QUERY`` of ``cypher`` on the live graph as ``ccg_raw_<system>``, with a time limit (default ``config.QUERY_TIMEOUT_MS``). A write is refused by the database."""
        if not isinstance(cypher, str) or not cypher.strip():
            raise BadRequest("cypher is a non-empty string")
        timeout_ms = config.QUERY_TIMEOUT_MS if timeout_ms is None else timeout_ms
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 1:
            raise BadRequest("the time limit is a positive integer number of milliseconds")
        password = _credential(self.system, "raw")
        try:
            db = self._raw_connect(self.host, self.port, username=sysname.acl_user("raw", self.system), password=password,
                                   socket_timeout=timeout_ms / 1000 + SOCKET_MARGIN_S)
            result = db.select_graph(sysname.live_key(self.system)).ro_query(cypher, None, timeout_ms)
        except ClientError:
            raise
        except Exception as exc:  # noqa: BLE001 - the server's refusal arrives as whatever the driver raises; it is classified below, never swallowed
            raise self._raw_error(exc, timeout_ms) from None
        if len(result.result_set) >= config.RESULTSET_SIZE:
            raise ResultTooLarge(f"the reply filled RESULTSET_SIZE ({config.RESULTSET_SIZE}) and may be truncated; narrow the query or add LIMIT")
        return {"asker": asker(), "system": self.system, "timeoutMs": timeout_ms, "columns": [h[1] for h in result.header],
                "rows": [[_plain(c) for c in row] for row in result.result_set]}

    @staticmethod
    def _raw_error(exc: Exception, timeout_ms: int) -> ClientError:
        text = " ".join(str(exc).split())[:200]
        low = text.lower()
        if "timeout_max" in low or "timeout parameter" in low:
            return RawRefused(f"the time limit of {timeout_ms} ms is above the server's TIMEOUT_MAX", detail={"server": _safe(text)})
        if "timed out" in low:
            return StoreError(f"the query timed out after {timeout_ms} ms")
        if any(s in low for s in ("read-only", "readonly", "noperm", "no permission", "not allowed", "permission")):
            return RawRefused("the database refused the query for the raw role", detail={"server": _safe(text)})
        return StoreError(f"the raw query failed ({type(exc).__name__})", detail={"server": _safe(text)})
