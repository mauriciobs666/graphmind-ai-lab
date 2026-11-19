"""The ``ccg`` MCP server (C-1001 P1-25b): stdio, the four tools ``ask``, ``resolve``, ``refresh``, ``raw_query``.

    cd cpg && .venv/bin/python -m clients.mcp.server

Configuration is the environment the client reads (``clients/common.py``): ``CCG_SYSTEM`` (default: the default manifest's ``system``), ``CCG_FALKORDB_HOST``/``CCG_FALKORDB_PORT``
(default ``127.0.0.1:6390``; 6379 is refused), and the credentials directory of ``ccg.store.acl``. A tool's optional ``system`` argument overrides ``CCG_SYSTEM`` for that call,
as the CLI's ``--system`` does. A configuration the client refuses stops the server at start, exit 2, with the typed error on stderr.

Each tool is a thin wrapper over :class:`clients.common.Client` (one per system, kept for the server's life: the interpretation cache of the plain-language front lives in it) (``docs/plans/connected-code-graph-p1-answers.md`` P1-25 (1)-(9)); no rule is re-implemented here:

* ``ask`` returns the canonical answer bytes (``ccg.answer.dumps``) as its one text content, validated first (``render_answer``), so ``notices`` is in it (``-answer-spec2.md`` §5.1).
* ``resolve``, ``refresh`` and ``raw_query`` return :func:`clients.common.dump_json` of the client's result: the bytes the CLI prints, without its newline.
* A typed failure is a tool error (``isError``) whose one text content is the CLI's error payload ``{"error": {"code", "message", "detail"?}}``; an invalid answer is
  ``invalid-answer`` and anything unanticipated is ``internal-error`` naming only the exception type. Nothing leaves as a traceback.
* **The arguments are declared ``Any``**, so FastMCP does no type coercion (``timeoutMs: true`` never becomes 1, ``cacheBypass: "yes"`` never becomes ``True``) and
  ``Client`` is the one validator: a wrong type is the typed ``bad-request``. Two FastMCP behaviours remain: a missing required argument (``anchor``, ``cypher``) is
  FastMCP's own error text, and an argument sent as a JSON *string* holding an object or array is parsed first (FastMCP's ``pre_parse_json``; a scalar string is not).
* The transport is stdio only: the server opens no listening socket (``-egress-spec3.md`` §10.1; ``tests/test_no_network.py``).
"""
from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from ccg import answer
from clients import common
from clients.common import Client, ClientError

NAME = "ccg"
INSTRUCTIONS = ("Connected code graph (P1): impact and depends-on answers over the local store. ask takes a structured request {useCase, anchor} "
                "(useCase impact or depends-on); resolve lists an anchor's candidates; refresh rebuilds the system; raw_query runs read-only Cypher as the "
                "ccg_raw_<system> role. Results are canonical JSON; errors are {\"error\": {code, message}}.")

ClientFactory = Callable[[str | None], Client]
#: the most systems whose client the server keeps (a tool's ``system`` argument is caller-chosen); past it the least recently used goes. The interpretation cache each client holds is
#: bounded on its own (``ccg.front.cache.MAX_ENTRIES``).
MAX_CLIENTS = 8


class _Clients:
    """One client per system for the life of the server, so the plain-language front's in-memory interpretation cache survives between calls (a client per call would
    never hit it). Nothing else of a client is call state: each ``ask`` opens its own reader and model run. A factory that raises memoizes nothing."""

    def __init__(self, factory: ClientFactory, limit: int = MAX_CLIENTS):
        self._factory, self._limit = factory, limit
        self._lock = threading.Lock()
        self._clients: dict[str | None, Client] = {}

    def __call__(self, system: Any) -> Client:
        if not (system is None or isinstance(system, str)):
            return self._factory(system)                        # the client refuses it (bad-request); an unhashable value is never a key
        with self._lock:
            client = self._clients.pop(system, None)           # popped and re-inserted: a hit is the newest, so the least recently used goes first
            if client is None:
                client = self._factory(system)
            self._clients[system] = client
            while len(self._clients) > self._limit:
                del self._clients[next(iter(self._clients))]
            return client


def _ok(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


def _error(payload: dict[str, Any]) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=common.dump_json(payload))], isError=True)


def _call(produce: Callable[[], str]) -> CallToolResult:
    """Run one client operation and map every failure to the typed error payload the CLI prints (same codes, same value-free messages)."""
    try:
        return _ok(produce())
    except ClientError as exc:
        return _error(exc.payload())
    except answer.AnswerError:
        return _error({"error": {"code": "invalid-answer", "message": "the answer does not validate; nothing was returned"}})
    except Exception as exc:  # noqa: BLE001 - the backstop: a tool never answers with a traceback or a driver's text
        return _error({"error": {"code": "internal-error", "message": f"unexpected failure ({type(exc).__name__})"}})


def build_server(client_factory: ClientFactory = Client) -> FastMCP:
    """The FastMCP server whose four tools use ``client_factory(system)`` (default: ``Client``, configured from the environment), built once per system and kept
    (:class:`_Clients`)."""
    server = FastMCP(NAME, instructions=INSTRUCTIONS)
    client_factory = _Clients(client_factory)

    @server.tool()
    def ask(request: Any = None, question: Any = None, cacheBypass: Any = False, system: Any = None) -> CallToolResult:  # noqa: N803 - the answer spec's field name
        """Answer a structured request {"useCase": "impact" | "depends-on", "anchor": {...}} or a plain-language question (exactly one). Returns the canonical
        answer JSON. cacheBypass (default false) is for the question path and has no effect on a structured request. An anchor that is not exactly one candidate is
        the typed error anchor-ambiguous/anchor-unresolved carrying the candidates; another use case, or a request scope, is not-built."""
        return _call(lambda: common.render_answer(client_factory(system).ask(request, question=question, cache_bypass=cacheBypass), "json"))

    @server.tool()
    def resolve(anchor: Any, system: Any = None) -> CallToolResult:
        """List the candidates of an anchor with their ids. The anchor is {"kind": "symbol", "repo", "qname"}, {"kind": "name", "name", "repo"?},
        {"kind": "location", "repo", "path", "line"? | "symbol"?}, {"kind": "contract", "key", "contractKind"?} or {"kind": "config", "repo", "path", "keyPath"?}."""
        return _call(lambda: common.dump_json(client_factory(system).resolve(anchor)))

    @server.tool()
    def refresh(scope: Any = None, system: Any = None) -> CallToolResult:
        """Rebuild the system from its local clones; scope is a list of repoIds to rebuild (default: all). Returns the refresh summary."""
        return _call(lambda: common.dump_json(client_factory(system).refresh(scope)))

    @server.tool()
    def raw_query(cypher: Any, timeoutMs: Any = None, system: Any = None) -> CallToolResult:  # noqa: N803 - camelCase like every request field
        """Read-only Cypher (GRAPH.RO_QUERY) on the live graph as the ccg_raw_<system> role; the database refuses a write. timeoutMs defaults to the pinned
        TIMEOUT_MAX and the server refuses a larger one; a time limit bounds wall time, not memory. Available to the local OS user in P1 (access control is P4)."""
        return _call(lambda: common.dump_json(client_factory(system).raw_query(cypher, timeout_ms=timeoutMs)))

    return server


def main() -> int:
    try:
        Client()                                   # validate the configuration (system, endpoint) before serving; opens no connection
    except ClientError as exc:
        sys.stderr.write(common.dump_json(exc.payload()) + "\n")
        return 2
    build_server().run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
