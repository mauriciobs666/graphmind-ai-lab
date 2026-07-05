"""FalkorDB connection + graph selection (DESIGN §14.2 `db.py`).

Thin wrapper over `falkordb-py`. The only responsibilities here are opening a
connection (fast-failing when FalkorDB is unreachable — DEF-2) and selecting
the per-workspace graph (`ws:{id}`); all Cypher lives in `repository.py`.
"""

from __future__ import annotations

import threading

from falkordb import FalkorDB, Graph
from redis import exceptions as redis_exceptions

from . import config


class FalkorDBUnreachableError(ConnectionError):
    """FalkorDB could not be reached within the connect budget (DEF-2).

    The message names host:port and the timeout so a misconfigured or down
    instance is diagnosable from the first line of output.
    """


def connect(host: str | None = None, port: int | None = None) -> FalkorDB:
    """Open a FalkorDB connection, failing fast when it is unreachable.

    Config is resolved at call time (not import time) so tests/deploys can
    repoint `config.FALKORDB_*` without re-importing this module.

    The socket timeouts are load-bearing (DEF-2): the falkordb-py constructor
    issues an eager command, and on WSL2 a dead port can blackhole instead of
    refusing — without `socket_connect_timeout` this call hangs for minutes.
    Unreachable → `FalkorDBUnreachableError` within the connect budget.
    """
    host = host if host is not None else config.FALKORDB_HOST
    port = port if port is not None else config.FALKORDB_PORT
    try:
        return FalkorDB(
            host=host,
            port=port,
            socket_connect_timeout=config.FALKORDB_CONNECT_TIMEOUT,
            socket_timeout=config.FALKORDB_SOCKET_TIMEOUT,
        )
    except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError, OSError) as exc:
        raise FalkorDBUnreachableError(
            f"FalkorDB unreachable at {host}:{port} "
            f"(connect timeout {config.FALKORDB_CONNECT_TIMEOUT}s): {exc} — "
            f"is it running? start it with ./scripts/start_falkordb.sh or point "
            f"FALKORDB_HOST/FALKORDB_PORT at a live instance"
        ) from exc


class LazyFalkorDB:
    """Deferred-connection handle (DEF-2).

    Quacks like `FalkorDB` for the repository's needs (`select_graph`), but
    opens the real connection on first use — so importing/building the app
    (`app = create_app()` runs at module scope) never touches the network.
    The first real use (app lifespan startup) inherits `connect`'s fast-fail
    behavior. Lock-guarded: FastAPI runs sync endpoints on a threadpool.
    """

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self._host = host
        self._port = port
        self._db: FalkorDB | None = None
        self._lock = threading.Lock()

    def select_graph(self, graph_id: str) -> Graph:
        with self._lock:
            if self._db is None:
                self._db = connect(self._host, self._port)
        return self._db.select_graph(graph_id)


def workspace_graph(db: FalkorDB | LazyFalkorDB, ws: str) -> Graph:
    """Select the per-workspace graph for workspace id `ws` (key ``ws:{ws}``).

    One graph per workspace is a locked decision (AGENTS.md) — never filter by a
    `workspaceId` property inside a shared graph.
    """
    return db.select_graph(f"ws:{ws}")
