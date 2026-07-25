#!/usr/bin/env python3
"""The `cpg` MCP server — one tool, `query`, over a named FalkorDB graph.

Design and rationale: ``docs/plans/cpg-query-access.md`` §4.4 (the frozen tool
contract). The short version:

* **One tool, exactly two parameters** (``graph``, ``cypher``) — FR-2/FR-4. No
  ``limit``, no ``params``, no ``mode``; every knob is an environment variable.
* **Read-only via ``GRAPH.RO_QUERY``.** That is what bounds the blast radius on
  an instance that also holds non-CPG graphs: writes are rejected server-side,
  *and* a typo'd graph name cannot materialise an empty key (plain
  ``GRAPH.QUERY`` would create one). Safe by construction, not by validation.
* **``EXPLAIN`` is honoured, ``PROFILE`` is actively refused.** Raw FalkorDB
  *ignores* both prefixes and executes the query, so a passive drop would return
  results to a caller that asked for a profile — a wrong answer rather than an
  error. ``GRAPH.PROFILE`` additionally executes writes, which is why it is not
  offered at all. The ``EXPLAIN`` path is preceded by a ``GRAPH.LIST`` check
  because ``GRAPH.EXPLAIN`` is not a read-only command and would otherwise
  materialise a key for a mistyped graph name.
* **Display-only truncation.** The full result set is materialised before
  formatting; the caps below shape the *rendering*, so the reported row count is
  always exact and memory/latency are bounded by the query, not by the caps.

Operational constraints this module obeys, all load-bearing for a stdio server:
importing it must not connect; nothing is ever written to stdout (the transport
owns it — diagnostics go to stderr); and the tool never raises, because a stdio
server that dies mid-session is not reconnected by the harness.
"""

from __future__ import annotations

import os
import re
import sys
import time

from falkordb import FalkorDB
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

# --------------------------------------------------------------------------
# Configuration (environment only — FR-2 forbids extra tool parameters)
# --------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on anything odd.

    A bad value must not take the server down at import time: an unparseable or
    non-positive override is reported on stderr and the default is used.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        _log(f"{name}={raw!r} is not an integer; using {default}")
        return default
    if value <= 0:
        _log(f"{name}={value} must be positive; using {default}")
        return default
    return value


def _log(message: str) -> None:
    """Diagnostics go to stderr only — stdout carries the MCP protocol."""
    print(f"[cpg-mcp] {message}", file=sys.stderr, flush=True)


FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "127.0.0.1")
FALKORDB_PORT = _env_int("FALKORDB_PORT", 6379)

MAX_ROWS = _env_int("CPG_MCP_MAX_ROWS", 200)
MAX_CELL = _env_int("CPG_MCP_MAX_CELL", 300)
MAX_CHARS = _env_int("CPG_MCP_MAX_CHARS", 30000)
TIMEOUT_MS = _env_int("CPG_MCP_TIMEOUT_MS", 30000)

#: Declared to the harness as ``_meta["anthropic/maxResultSizeChars"]``. Claude
#: Code otherwise estimates a *token* budget and, above it, persists the result
#: to disk and replaces it with a file reference — which would swallow the
#: truncation notice on exactly the runs that are truncated. Pinning it in the
#: same unit as our own cap (with 2x headroom for the stats/header/notice lines)
#: keeps the two from ever disagreeing. Hard ceiling: 500 000 chars.
MAX_RESULT_SIZE_CHARS = min(2 * MAX_CHARS, 500_000)

# --------------------------------------------------------------------------
# Fixed strings (frozen by the plan — tests pin them)
# --------------------------------------------------------------------------

TOOL_DESCRIPTION = (
    "Run a read-only OpenCypher query against a named FalkorDB graph, typically a loaded "
    "Joern CPG. `graph` is the graph key (caller-supplied, e.g. cpg_<component>); `cypher` "
    "is the query text, sent verbatim — multi-line welcome, no shell quoting. Prefix "
    "EXPLAIN for a query plan; PROFILE is not supported (it executes the query). FalkorDB "
    "is OpenCypher: no APOC, no GDS. CPG schema: "
    "skills/joern-cpg/references/cpg-model.md."
)

SERVER_INSTRUCTIONS = (
    "The `cpg` server exposes a single tool, `query`: read-only OpenCypher against a named "
    "FalkorDB graph — typically a Joern Code Property Graph loaded as `cpg_<component>`. "
    "Use it to answer call-graph, data-flow, impact-analysis and test-gap questions about a "
    "codebase without reading files. Graph names are always supplied by the caller; a query "
    "against an unknown graph answers with the list of loaded graphs."
)

PROFILE_REFUSAL = (
    "PROFILE is not available through this tool: GRAPH.PROFILE executes the query, "
    "including writes. Use EXPLAIN for the plan. For measured profiling use the fallback: "
    "redis-cli -p 6379 GRAPH.PROFILE <graph> '<cypher>' --no-raw."
)

SCHEMA_POINTER = (
    "FalkorDB is OpenCypher (no APOC/GDS); CPG property keys are UPPER_CASE — see "
    "skills/joern-cpg/references/cpg-model.md."
)

# --------------------------------------------------------------------------
# Directive sniffing (plan §4.4 D5a) — classification only, never a rewrite
# --------------------------------------------------------------------------

#: ``str.lstrip()`` semantics, spelled out so the scan below stays explicit.
_WHITESPACE = " \t\r\n\f\v"

#: ``\b`` matters: ``EXPLAIN_ME`` / ``PROFILER`` / ``PROFILEDATA`` are ordinary
#: identifiers and must classify as plain queries, while ``EXPLAIN(``,
#: ``EXPLAIN\n`` and ``explain\tMATCH`` are directives.
_DIRECTIVE_RE = re.compile(r"(EXPLAIN|PROFILE)\b", re.IGNORECASE)


def _scan_leading_trivia(cypher: str) -> int:
    """Return the index of the first character that is not leading trivia.

    Trivia is whitespace, ``//``-to-end-of-line comments and ``/* … */`` blocks,
    in any order and any number. An **unterminated** ``/*`` stops the scan where
    it starts: the statement is malformed, so it classifies as a plain query and
    FalkorDB gets to return its own error verbatim.

    The scan stops at the first real character, so comment markers *inside* a
    string literal later in the query are never seen — no lexer is needed.
    """
    i = 0
    n = len(cypher)
    while i < n:
        j = i
        while j < n and cypher[j] in _WHITESPACE:
            j += 1
        if cypher.startswith("//", j):
            newline = cypher.find("\n", j + 2)
            j = n if newline == -1 else newline + 1
        elif cypher.startswith("/*", j):
            # Cypher block comments do not nest — match the first `*/`.
            end = cypher.find("*/", j + 2)
            if end == -1:
                return j
            j = end + 2
        if j == i:  # nothing consumed this round: we are at real text
            return i
        i = j
    return i


def split_directive(cypher: str) -> tuple[str, str]:
    """Classify a statement as ``"query"``, ``"explain"`` or ``"profile"``.

    Returns ``(kind, cypher_to_send)``:

    * ``"query"`` — the caller's string **byte-for-byte**. The normalisation
      above is used for classification only; FR-3 promises verbatim
      transmission, and stripping comments would corrupt e.g.
      ``CONTAINS '// x'``.
    * ``"explain"`` — the text after the keyword, left-stripped. The consumed
      leading trivia is dropped; it cannot change a plan.
    * ``"profile"`` — the empty string. Nothing is ever sent.
    """
    start = _scan_leading_trivia(cypher)
    match = _DIRECTIVE_RE.match(cypher, start)
    if match is None:
        return "query", cypher
    if match.group(1).upper() == "PROFILE":
        return "profile", ""
    return "explain", cypher[match.end():].lstrip()


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_cell(value: object, max_chars: int) -> str:
    """Render one result cell as a single line of at most ~``max_chars`` chars.

    ``None`` becomes ``null`` (distinguishable from an empty string), strings
    pass through, nodes/edges use their FalkorDB ``__str__``, lists and maps use
    ``repr``. Newlines and tabs are escaped so that one row is always one line;
    pipes are deliberately *not* escaped (return a single column when a value
    must be copied out exactly).
    """
    if value is None:
        text = "null"
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple, dict)):
        text = repr(value)
    else:
        text = str(value)

    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    if len(text) > max_chars:
        dropped = len(text) - max_chars
        text = f"{text[:max_chars]}…(+{dropped} chars)"
    return text


def _column_names(header: list) -> list[str]:
    """Column names from a ``QueryResult.header`` (``[[type, name], …]``).

    Tolerates a plain list of names, which is what hand-built fixtures use.
    """
    names: list[str] = []
    for column in header or []:
        if isinstance(column, (list, tuple)) and len(column) >= 2:
            names.append(str(column[1]))
        else:
            names.append(str(column))
    return names


def truncation_notice(shown: int, total: int, reason: str) -> str:
    """The one honest truncation sentence, emitted first *and* last, verbatim.

    The "unordered" clause is not padding: the first N rows of an unordered
    result set are arbitrary, and an agent reading "showing 200 of 79581" may
    otherwise draw a conclusion from a non-deterministic sample.
    """
    return (
        f"… truncated: showing {shown} of {total} rows ({reason}) — results are unordered "
        "unless the query has ORDER BY; narrow with LIMIT, a projection, or an aggregate."
    )


def format_result(
    graph: str,
    header: list,
    rows: list,
    elapsed_ms: float,
    max_rows: int = MAX_ROWS,
    max_cell: int = MAX_CELL,
    max_chars: int = MAX_CHARS,
) -> str:
    """Render a result set as plain text (JSON roughly doubles the token cost).

    Layout, with the notice lines present only when something was actually cut::

        … truncated: showing 200 of 79581 rows (row cap) — …
        graph=cpg_falkorchat · rows=79581 · 812.4ms
        caller | file | line
        …rows…
        … truncated: showing 200 of 79581 rows (row cap) — …

    ``rows=`` is always the **true** total. When the char cap binds, whole rows
    are dropped from the tail (never a partial row) and the notice names that
    cap instead.
    """
    total = len(rows)
    stats = f"graph={graph} · rows={total} · {elapsed_ms:.1f}ms"

    names = _column_names(header)
    if not names:
        # A statement with no RETURN (not expected under RO) has no table.
        return stats

    head = " | ".join(names)
    if total == 0:
        return f"{stats}\n{head}\n(no rows)"

    kept = rows[:max_rows]
    body = [" | ".join(render_cell(cell, max_cell) for cell in row) for row in kept]
    notice = truncation_notice(len(kept), total, "row cap") if total > len(kept) else None

    def payload(lines: list[str], note: str | None) -> str:
        parts = [stats, head, *lines]
        if note is not None:
            parts = [note, *parts, note]
        return "\n".join(parts)

    rendered = payload(body, notice)
    if len(rendered) > max_chars:
        # Char cap binds: drop whole rows from the tail until the payload —
        # including *both* copies of the notice — fits.
        reason = f"char cap {max_chars}"
        keep = len(body)
        while keep > 0:
            note = truncation_notice(keep, total, reason)
            if len(payload(body[:keep], note)) <= max_chars:
                break
            keep -= 1
        body = body[:keep]
        notice = truncation_notice(keep, total, reason)
        rendered = payload(body, notice)
    return rendered


def format_plan(graph: str, plan: object) -> str:
    """Render an ``EXPLAIN`` result. Nothing was executed to produce it."""
    return f"graph={graph} · EXPLAIN (plan only — nothing was executed)\n{plan}"


# --------------------------------------------------------------------------
# Errors — every failure is a curated, actionable message, never a traceback
# --------------------------------------------------------------------------


def _is_missing_graph(exc: Exception) -> bool:
    return isinstance(exc, ResponseError) and "empty key" in str(exc).lower()


def graph_not_found_message(graph: str, graphs: list[str] | None) -> str:
    loaded = ", ".join(graphs) if graphs else "(could not read GRAPH.LIST)"
    return (
        f"Graph '{graph}' does not exist. Loaded graphs: {loaded}. If no CPG is loaded, "
        "building and loading one is the joern agent's job (joern-cpg pipeline) — this tool "
        "only queries."
    )


def explain_error(
    exc: Exception,
    graph: str,
    host: str,
    port: int,
    graphs: list[str] | None = None,
) -> str:
    """Map an exception to the curated message for its condition (plan §4.4)."""
    if isinstance(exc, ResponseError):
        message = str(exc)
        lowered = message.lower()
        if "empty key" in lowered:
            return graph_not_found_message(graph, graphs)
        if "ro_query" in lowered:
            return (
                "This tool is read-only (GRAPH.RO_QUERY). Loading/writing a CPG goes through "
                "the joern pipeline, or redis-cli for ad-hoc writes."
            )
        if "timeout" in lowered or "timed out" in lowered:
            return (
                f"Query exceeded {TIMEOUT_MS} ms. Bound variable-length traversals (*1..N), "
                "add LIMIT, or prefix EXPLAIN to inspect the plan first."
            )
        # Syntax/semantic error: FalkorDB's own message carries line, column and
        # context — it is better than anything we could write. Pass it through.
        return f"{message}\n{SCHEMA_POINTER}"

    if isinstance(exc, (RedisConnectionError, RedisTimeoutError, ConnectionError, TimeoutError)):
        return (
            f"FalkorDB unreachable at {host}:{port}. Start it "
            "(falkor-chat/scripts/start_falkordb.sh, or docker start falkordb-dev) and retry."
        )

    return f"Unexpected error: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Connection — one lazily created client, reused (redis-py pools and reconnects)
# --------------------------------------------------------------------------

_client: FalkorDB | None = None


def get_client() -> FalkorDB:
    """Return the shared client, creating it on first use.

    Import must not connect: a server whose module-level code talks to FalkorDB
    would fail to start whenever the database is down, and a stdio server that
    fails to start is simply absent from the session.
    """
    global _client
    if _client is None:
        _client = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    return _client


def _list_graphs(client: FalkorDB) -> list[str] | None:
    try:
        return list(client.list_graphs())
    except Exception:  # noqa: BLE001 — best-effort enrichment of an error message
        return None


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------

mcp = FastMCP(name="cpg", instructions=SERVER_INSTRUCTIONS)


@mcp.tool(
    name="query",
    description=TOOL_DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True),
    # `-> str` alone is NOT enough on mcp 1.28.x: FastMCP would synthesise an
    # `outputSchema {result: string}` and then return the payload twice (text
    # content *and* structuredContent), doubling every capped result.
    structured_output=False,
    meta={"anthropic/maxResultSizeChars": MAX_RESULT_SIZE_CHARS},
)
def query(graph: str, cypher: str) -> str:
    """Run read-only Cypher against `graph` and return a plain-text table."""
    return run_query(graph, cypher)


def run_query(graph: str, cypher: str) -> str:
    """The tool body, importable and callable without the MCP plumbing.

    Never raises: every failure path returns a curated message instead, because
    a crashed stdio server is not reconnected mid-session.
    """
    try:
        kind, to_send = split_directive(cypher)

        # Refused before any server call: FalkorDB would silently ignore the
        # prefix and return results, which is a wrong answer, not an error.
        if kind == "profile":
            return PROFILE_REFUSAL

        client = get_client()
        target = client.select_graph(graph)

        if kind == "explain":
            # GRAPH.EXPLAIN is not a read-only command, so a mistyped name would
            # materialise an empty key. Check first; the list doubles as the
            # not-found message's content. A failing GRAPH.LIST propagates to the
            # curated handler rather than letting an unverified explain() through.
            graphs = list(client.list_graphs())
            if graph not in graphs:
                return graph_not_found_message(graph, graphs)
            return format_plan(graph, target.explain(to_send))

        started = time.perf_counter()
        result = target.ro_query(to_send, timeout=TIMEOUT_MS)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return format_result(graph, result.header, result.result_set, elapsed_ms)
    except Exception as exc:  # noqa: BLE001 — the curated path is the only exit
        graphs = _list_graphs(get_client()) if _is_missing_graph(exc) else None
        return explain_error(exc, graph, FALKORDB_HOST, FALKORDB_PORT, graphs)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
