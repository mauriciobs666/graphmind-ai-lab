"""Unit tests for the `cypher` MCP server (plan §5 S2, §7.1).

Everything here except the `live`-marked tests runs offline: the pure functions
(`split_directive`, `render_cell`, `format_result`, `explain_error`) need no
database, and the paths that would touch one are exercised through a recording
fake so that "no server call was made" is itself assertable.

The `live` tests run against a REAL FalkorDB and are deselected by default
(`pytest.ini`). They create and delete their own scratch graph and never touch
`cpg_*`, `ws:*` or `reference`.
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import pytest

import server

# --------------------------------------------------------------------------
# Fakes — a client that records every call it receives
# --------------------------------------------------------------------------


class FakeResult:
    def __init__(self, header, rows):
        self.header = header
        self.result_set = rows


class FakeWriteResult:
    """Stands in for `falkordb.query_result.QueryResult`'s write path: every
    mutation-counter attribute the real (1.6.2, pinned) client exposes,
    defaulting to 0 unless overridden — mirrors `server._WRITE_STAT_ATTRS`
    exactly, so a test can assert on the same names `format_write_result()`
    reads."""

    def __init__(self, **stats):
        for attr in server._WRITE_STAT_ATTRS:
            setattr(self, attr, stats.get(attr, 0))


class FakeGraph:
    def __init__(
        self,
        calls,
        *,
        header=None,
        rows=None,
        error=None,
        plan="PLAN",
        query_error=None,
        write_stats=None,
    ):
        self._calls = calls
        self._header = header if header is not None else [[2, "n"]]
        self._rows = rows if rows is not None else [["x"]]
        self._error = error
        self._plan = plan
        self._query_error = query_error
        self._write_stats = write_stats or {}

    def ro_query(self, cypher, timeout=None):
        self._calls.append(("ro_query", cypher, timeout))
        if self._error is not None:
            raise self._error
        return FakeResult(self._header, self._rows)

    def explain(self, cypher):
        self._calls.append(("explain", cypher))
        if self._error is not None:
            raise self._error
        return self._plan

    def query(self, cypher, timeout=None):
        """Records the call so "was a real write attempted" is directly
        assertable, then returns a `FakeWriteResult` with the configured
        mutation counters (all 0 unless `write_stats` was supplied)."""
        self._calls.append(("query", cypher, timeout))
        if self._query_error is not None:
            raise self._query_error
        return FakeWriteResult(**self._write_stats)


class FakeClient:
    def __init__(self, *, graphs=(), **graph_kwargs):
        self.calls: list[tuple] = []
        self._graphs = list(graphs)
        self.graph = FakeGraph(self.calls, **graph_kwargs)

    def select_graph(self, name):
        self.calls.append(("select_graph", name))
        return self.graph

    def list_graphs(self):
        self.calls.append(("list_graphs",))
        return list(self._graphs)


@pytest.fixture
def fake_client(monkeypatch):
    """Install a recording client and hand it back for assertions."""

    def _install(**kwargs):
        client = FakeClient(**kwargs)
        monkeypatch.setattr(server, "get_client", lambda: client)
        return client

    return _install


# --------------------------------------------------------------------------
# 1 — tool contract
# --------------------------------------------------------------------------


def _tools():
    return asyncio.run(server.mcp.list_tools())


def test_exactly_one_tool_named_query():
    tools = _tools()
    assert [t.name for t in tools] == ["query"]


def test_input_schema_has_two_required_params_and_one_optional_agent():
    """FR-2's original two-required-param shape holds; FR-1 (generic-cypher-mcp)
    widens it with exactly one new, optional third parameter — never a second
    required one, which would break every existing read-only caller."""
    schema = _tools()[0].inputSchema
    assert set(schema["properties"]) == {"graph", "cypher", "agent"}
    assert sorted(schema["required"]) == ["cypher", "graph"]
    assert schema["properties"]["graph"]["type"] == "string"
    assert schema["properties"]["cypher"]["type"] == "string"
    # `agent: str | None = None` — optional, accepts a string or null.
    agent_types = {branch["type"] for branch in schema["properties"]["agent"]["anyOf"]}
    assert agent_types == {"string", "null"}
    assert schema["properties"]["agent"]["default"] is None


def test_output_schema_is_absent():
    """Regression guard for the double-payload bug (review M-3).

    `-> str` alone makes FastMCP synthesise `outputSchema {result: string}` on
    mcp 1.28.x, and with an outputSchema present the payload is returned twice —
    text content *and* structuredContent — so every capped result would arrive at
    ~2x its budgeted size. `structured_output=False` is what prevents it.
    """
    assert _tools()[0].outputSchema is None


def test_tool_declares_read_only_and_the_max_result_size_meta():
    tool = _tools()[0]
    assert tool.annotations.readOnlyHint is True
    assert tool.meta == {"anthropic/maxResultSizeChars": 2 * server.MAX_CHARS}
    assert tool.meta == {"anthropic/maxResultSizeChars": 60000}
    # It serialises as `_meta`, which is what the harness reads.
    assert tool.model_dump(by_alias=True, exclude_none=True)["_meta"] == tool.meta


def test_default_caps():
    """Guards against a silent bump (N-2 lowered the char cap 60 000 → 30 000)."""
    assert server.MAX_ROWS == 200
    assert server.MAX_CELL == 300
    assert server.MAX_CHARS == 30000
    assert server.TIMEOUT_MS == 30000
    assert server.MAX_RESULT_SIZE_CHARS == 60000


def test_server_instructions_are_present_and_bounded():
    """C-318: `mcp.instructions` is the only unguarded part of the tool contract.

    It is what MCP tool search reads before a cold session ever calls `query`
    (tools are deferred by default), so a missing or oversized string is a
    silent defect nothing else here would catch.
    """
    instructions = server.mcp.instructions
    assert instructions
    assert len(instructions) <= 2000


# --------------------------------------------------------------------------
# 2 — directive splitting (plan §4.4 D5a)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cypher, kind, to_send",
    [
        # plain
        ("MATCH (n) RETURN n", "query", "MATCH (n) RETURN n"),
        ("  \n\tMATCH (n) RETURN n", "query", "  \n\tMATCH (n) RETURN n"),
        # a query merely *containing* the words
        (
            "MATCH (n) WHERE n.NAME = 'explain' RETURN n.PROFILE",
            "query",
            "MATCH (n) WHERE n.NAME = 'explain' RETURN n.PROFILE",
        ),
        # explain, in several spellings
        ("EXPLAIN MATCH (n) RETURN n", "explain", "MATCH (n) RETURN n"),
        ("explain\nMATCH (n) RETURN n", "explain", "MATCH (n) RETURN n"),
        ("   EXPLAIN   MATCH (n) RETURN n", "explain", "MATCH (n) RETURN n"),
        ("explain\tMATCH (n) RETURN n", "explain", "MATCH (n) RETURN n"),
        # profile never sends anything
        ("PROFILE MATCH (n) RETURN n", "profile", ""),
        ("profile\tMATCH (n) RETURN n", "profile", ""),
        # comment-blind (n-3): raw GRAPH.RO_QUERY accepts these and returns ROWS
        ("/* hi */ PROFILE MATCH (n) RETURN n", "profile", ""),
        ("// hi\nPROFILE MATCH (n) RETURN n", "profile", ""),
        (
            "  /* a */\n// b\n  explain match (n) return n",
            "explain",
            "match (n) return n",
        ),
        # the \b boundary: identifiers that merely start with the keyword
        ("EXPLAIN_ME MATCH (n) RETURN n", "query", "EXPLAIN_ME MATCH (n) RETURN n"),
        ("PROFILER MATCH (n) RETURN n", "query", "PROFILER MATCH (n) RETURN n"),
        ("PROFILEDATA MATCH (n) RETURN n", "query", "PROFILEDATA MATCH (n) RETURN n"),
        # EXPLAIN( is a directive — `(` is a word boundary
        ("EXPLAIN(MATCH (n) RETURN n)", "explain", "(MATCH (n) RETURN n)"),
    ],
)
def test_split_directive_classification(cypher, kind, to_send):
    assert server.split_directive(cypher) == (kind, to_send)


def test_unterminated_block_comment_classifies_plain_and_is_sent_verbatim():
    """Malformed input is FalkorDB's to reject, with its own error message."""
    cypher = "/* unterminated PROFILE MATCH (n) RETURN n"
    assert server.split_directive(cypher) == ("query", cypher)


def test_string_literal_decoy_is_not_a_directive_and_survives_byte_for_byte():
    """FR-3: the plain path transmits the caller's string unmodified.

    The prefix scan stops at the first real character, so comment markers inside
    a later string literal are never even looked at.
    """
    cypher = "MATCH (n) WHERE n.CODE CONTAINS '// PROFILE' RETURN n"
    kind, to_send = server.split_directive(cypher)
    assert kind == "query"
    assert to_send == cypher


def test_comment_only_input_is_plain():
    assert server.split_directive("// nothing here") == ("query", "// nothing here")


# --------------------------------------------------------------------------
# 3 — PROFILE is refused before any server call
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cypher",
    [
        "PROFILE MATCH (n) RETURN n",
        "profile\tMATCH (n) RETURN n",
        "/* hi */ PROFILE MATCH (n) RETURN n",
        "// hi\nPROFILE MATCH (n) RETURN n",
    ],
)
def test_profile_is_refused_without_touching_the_server(fake_client, cypher):
    client = fake_client()
    out = server.run_query("g", cypher)
    assert out == server.PROFILE_REFUSAL
    assert "GRAPH.PROFILE executes the query" in out
    assert "redis-cli -p 6379 GRAPH.PROFILE" in out
    assert client.calls == []  # not even select_graph


# --------------------------------------------------------------------------
# 4 — cell rendering
# --------------------------------------------------------------------------


class NodeLike:
    """Stands in for falkordb.node.Node, which renders through __str__."""

    def __str__(self):
        return "(:CpgNode:METHOD{NAME:\"post_message\"})"


def test_render_cell_none_is_distinguishable_from_empty_string():
    assert server.render_cell(None, 300) == "null"
    assert server.render_cell("", 300) == ""


def test_render_cell_passes_strings_and_scalars_through():
    assert server.render_cell("plain", 300) == "plain"
    assert server.render_cell(42, 300) == "42"


def test_render_cell_renders_booleans_lowercase_like_cypher_json_not_python():
    # C-315: booleans previously fell through to str(), producing Python's
    # capitalized `True`/`False`. Render lowercase to match Cypher/JSON
    # convention and the rest of this function's scalar rendering.
    assert server.render_cell(True, 300) == "true"
    assert server.render_cell(False, 300) == "false"


def test_render_cell_escapes_newlines_and_tabs_so_a_row_stays_one_line():
    assert server.render_cell("a\nb\tc\r\n", 300) == "a\\nb\\tc\\r\\n"


def test_render_cell_uses_str_for_nodes_and_repr_for_collections():
    assert server.render_cell(NodeLike(), 300) == '(:CpgNode:METHOD{NAME:"post_message"})'
    assert server.render_cell(["a", 1], 300) == "['a', 1]"
    assert server.render_cell({"k": "v"}, 300) == "{'k': 'v'}"


def test_render_cell_renders_map_valued_cells_without_leaking_client_type():
    # C-314: falkordb 1.6.2 returns map cells as OrderedDict, whose repr
    # includes the class name (e.g. `OrderedDict({'a': 1, 'b': 'x'})`). The
    # rendering must look like a plain dict regardless of the concrete
    # mapping type the client library hands back.
    from collections import OrderedDict

    out = server.render_cell(OrderedDict([("a", 1), ("b", "x")]), 300)
    assert out == "{'a': 1, 'b': 'x'}"
    assert "OrderedDict" not in out


def test_render_cell_normalizes_booleans_and_maps_at_any_nesting_depth():
    # Review finding on the flat-only fix above: `RETURN properties(m)`-shaped
    # results nest one map's values inside another, and C-315's own
    # motivating example is a boolean *property* (`IS_EXTERNAL`) inside such
    # a map, not a top-level scalar. Both the bool-lowercasing and the
    # OrderedDict-hiding must apply recursively, not just to the cell's own
    # top-level value.
    from collections import OrderedDict

    # A property map, as `properties(m)` would return it, containing the
    # exact `IS_EXTERNAL` boolean field the backlog item cites.
    out = server.render_cell({"NAME": "foo", "IS_EXTERNAL": False}, 300)
    assert out == "{'NAME': 'foo', 'IS_EXTERNAL': false}"

    # A map nested inside a map — OrderedDict must not leak at any depth.
    nested = OrderedDict([("a", 1), ("nested", OrderedDict([("b", 2)]))])
    out = server.render_cell(nested, 300)
    assert out == "{'a': 1, 'nested': {'b': 2}}"
    assert "OrderedDict" not in out

    # A map inside a list.
    out = server.render_cell([1, OrderedDict([("a", 1)])], 300)
    assert out == "[1, {'a': 1}]"
    assert "OrderedDict" not in out

    # A tuple containing a bool and a nested map with a bool value; the
    # tuple's own type (round parens) must be preserved.
    out = server.render_cell((1, True, {"x": False}), 300)
    assert out == "(1, true, {'x': false})"


def test_render_cell_marks_how_much_it_cut():
    out = server.render_cell("x" * 50, 10)
    assert out == "x" * 10 + "…(+40 chars)"


# --------------------------------------------------------------------------
# 5 — table formatting and truncation
# --------------------------------------------------------------------------

HEADER = [[2, "caller"], [2, "file"]]


def test_format_result_renders_stats_header_and_rows():
    out = server.format_result("g", HEADER, [["a", "b"], ["c", "d"]], 12.34)
    assert out.splitlines() == [
        "graph=g · rows=2 · 12.3ms",
        "caller | file",
        "a | b",
        "c | d",
    ]


def test_format_result_empty_result_is_not_mistakable_for_a_failure():
    out = server.format_result("g", HEADER, [], 1.0)
    assert out.splitlines() == ["graph=g · rows=0 · 1.0ms", "caller | file", "(no rows)"]


def test_format_result_statement_without_a_projection_reports_stats_only():
    assert server.format_result("g", [], [], 1.0) == "graph=g · rows=0 · 1.0ms"


def test_untruncated_payload_carries_no_notice():
    out = server.format_result("g", HEADER, [["a", "b"]], 1.0)
    assert "… truncated:" not in out


def _notice_lines(payload):
    return [line for line in payload.splitlines() if line.startswith("… truncated:")]


def test_row_cap_notice_is_exact_and_emitted_first_and_last():
    rows = [[f"r{i}", "f"] for i in range(5)]
    out = server.format_result("g", HEADER, rows, 5.0, max_rows=2)
    lines = out.splitlines()
    expected = (
        "… truncated: showing 2 of 5 rows (row cap) — results are unordered unless the "
        "query has ORDER BY; narrow with LIMIT, a projection, or an aggregate."
    )
    assert lines[0] == expected
    assert lines[-1] == expected  # byte-identical: it must survive a tail-side cut
    assert _notice_lines(out) == [expected, expected]
    # the true total is still reported, and only the kept rows are rendered
    assert "rows=5" in lines[1]
    assert lines[3:5] == ["r0 | f", "r1 | f"]


def test_char_cap_drops_whole_rows_from_the_tail_and_names_the_cap():
    rows = [[f"row{i:03d}", "x" * 40] for i in range(50)]
    cap = 600
    out = server.format_result("g", HEADER, rows, 5.0, max_rows=200, max_chars=cap)
    lines = out.splitlines()

    assert len(out) <= cap  # the notice's own two copies are budgeted for
    assert lines[0] == lines[-1]
    assert lines[0].startswith("… truncated: showing ")
    assert f"(char cap {cap})" in lines[0]
    assert "of 50 rows" in lines[0]
    assert _notice_lines(out) == [lines[0], lines[0]]

    shown = len(lines) - 4  # two notices + stats + header
    assert 0 < shown < 50
    assert f"showing {shown} of 50 rows" in lines[0]
    # whole rows only, dropped from the tail
    body = lines[3:-1]
    assert body == [f"row{i:03d} | {'x' * 40}" for i in range(shown)]


def test_notice_text_carries_the_unordered_caveat_on_both_paths():
    rows = [[f"r{i}", "y" * 60] for i in range(40)]
    row_capped = server.format_result("g", HEADER, rows, 1.0, max_rows=2)
    char_capped = server.format_result("g", HEADER, rows, 1.0, max_rows=200, max_chars=700)
    for payload in (row_capped, char_capped):
        assert "results are unordered unless the query has ORDER BY" in payload.splitlines()[0]


def test_cells_are_capped_before_the_char_cap_is_measured():
    out = server.format_result("g", HEADER, [["z" * 500, "b"]], 1.0, max_cell=10)
    assert out.splitlines()[2] == "z" * 10 + "…(+490 chars) | b"


# --------------------------------------------------------------------------
# 6 — error classification
# --------------------------------------------------------------------------


def _response_error(message):
    from redis.exceptions import ResponseError

    return ResponseError(message)


def test_error_falkordb_unreachable():
    from redis.exceptions import ConnectionError as RedisConnectionError

    out = server.explain_error(
        RedisConnectionError("Error 111 connecting"), "g", "127.0.0.1", 6379
    )
    assert out.startswith("FalkorDB unreachable at 127.0.0.1:6379.")
    assert "start_falkordb.sh" in out


def test_error_missing_graph_lists_loaded_graphs_and_routes_to_joern():
    out = server.explain_error(
        _response_error("Invalid graph operation on empty key"),
        "typo",
        "127.0.0.1",
        6379,
        ["a", "b"],
    )
    assert "Graph 'typo' does not exist." in out
    assert "Loaded graphs: a, b." in out
    assert "graph-dba agent" in out


def test_error_missing_graph_without_a_readable_graph_list():
    out = server.explain_error(
        _response_error("Invalid graph operation on empty key"), "typo", "h", 1, None
    )
    assert "could not read GRAPH.LIST" in out


def test_error_write_attempt_names_the_read_only_mode():
    out = server.explain_error(
        _response_error("graph.RO_QUERY is to be executed only on read-only queries"),
        "g",
        "h",
        1,
    )
    assert out.startswith("This tool is read-only (GRAPH.RO_QUERY).")


def test_error_timeout_suggests_bounding_the_traversal():
    out = server.explain_error(_response_error("Query timed out"), "g", "h", 1)
    assert out.startswith(f"Query exceeded {server.TIMEOUT_MS} ms.")
    assert "*1..N" in out


def test_error_syntax_passes_falkordb_message_through_verbatim_plus_a_pointer():
    message = "errMsg: Invalid input 'R': expected … line: 1, column: 17 errCtx: MATCH (m:METHOD"
    out = server.explain_error(_response_error(message), "g", "h", 1)
    assert out.splitlines()[0] == message
    assert out.splitlines()[-1] == server.SCHEMA_POINTER


def test_error_unknown_exception_is_curated_not_a_traceback():
    out = server.explain_error(ValueError("boom"), "g", "h", 1)
    assert out == "Unexpected error: ValueError: boom"
    assert "Traceback" not in out


def test_run_query_never_raises_on_an_unexpected_client_failure(fake_client):
    fake_client(graphs=["g"], error=ValueError("boom"))
    assert server.run_query("g", "MATCH (n) RETURN n") == "Unexpected error: ValueError: boom"


def test_run_query_enriches_a_missing_graph_error_with_the_live_graph_list(fake_client):
    missing = _response_error("Invalid graph operation on empty key")
    fake_client(graphs=["one", "two"], error=missing)
    out = server.run_query("typo", "MATCH (n) RETURN n")
    assert "Graph 'typo' does not exist." in out
    assert "Loaded graphs: one, two." in out


# --------------------------------------------------------------------------
# 7 — the paths through run_query
# --------------------------------------------------------------------------


def test_plain_path_sends_the_query_verbatim_with_the_timeout(fake_client):
    client = fake_client(graphs=["g"], header=HEADER, rows=[["a", "b"]])
    cypher = "  MATCH (n)\n  WHERE n.CODE CONTAINS '// PROFILE'\n  RETURN n"
    out = server.run_query("g", cypher)
    assert ("ro_query", cypher, server.TIMEOUT_MS) in client.calls
    assert ("list_graphs",) not in client.calls  # no pre-check needed on the RO path
    assert out.splitlines()[1] == "caller | file"


def test_explain_precheck_skips_explain_entirely_for_an_unknown_graph(fake_client):
    """GRAPH.EXPLAIN is not read-only: an unchecked name would materialise a key."""
    client = fake_client(graphs=["other"])
    out = server.run_query("missing", "EXPLAIN MATCH (n) RETURN n")
    assert "Graph 'missing' does not exist." in out
    assert "Loaded graphs: other." in out
    assert [c for c in client.calls if c[0] == "explain"] == []


def test_explain_path_sends_the_stripped_query_and_renders_the_plan(fake_client):
    client = fake_client(graphs=["g"], plan="Results\n    Node By Label Scan | (n:T)")
    out = server.run_query("g", "  /* c */ explain\n  MATCH (n:T) RETURN n")
    assert ("explain", "MATCH (n:T) RETURN n") in client.calls
    assert out.splitlines()[0] == "graph=g · EXPLAIN (plan only — nothing was executed)"
    assert "Node By Label Scan" in out


# --------------------------------------------------------------------------
# 8 — write authorization (FR-8, docs/plans/generic-cypher-mcp.md §3.1/§3.2)
# --------------------------------------------------------------------------

_READ_ONLY_ERROR = "graph.RO_QUERY is to be executed only on read-only queries"
_EMPTY_KEY_ERROR = "Invalid graph operation on empty key"

# The real §3.4 migration shape: one shared `author: 'graph-dba'` literal in
# the CREATE clause itself, not per-row inside the UNWIND list (the per-row
# maps never carry an `author` key at all).
_MIGRATION_CYPHER = (
    "UNWIND [{entryId:'1', date:'2026-08-16', fact:'a', evidence:'e1', context:'c1', "
    "suggestedHome:'unsure', createdAt:'t'}, "
    "{entryId:'2', date:'2026-08-16', fact:'b', evidence:'e2', context:'c2', "
    "suggestedHome:'unsure', createdAt:'t'}] AS e "
    "CREATE (k:KaizenEntry {entryId: e.entryId, date: e.date, fact: e.fact, "
    "evidence: e.evidence, context: e.context, suggestedHome: e.suggestedHome, "
    "author: 'graph-dba', createdAt: e.createdAt})"
)


def _writes_never_ran(client) -> bool:
    return not any(call[0] == "query" for call in client.calls)


def test_read_path_unchanged_without_agent(fake_client):
    """1 — a plain MATCH...RETURN, no `agent` → unchanged behavior, regression pin."""
    client = fake_client(graphs=["g"], header=HEADER, rows=[["a", "b"]])
    out = server.run_query("g", "MATCH (n) RETURN n.caller, n.file")
    assert out.splitlines()[1] == "caller | file"
    assert _writes_never_ran(client)


def test_write_without_agent_is_rejected(fake_client):
    """2 — write Cypher, `agent` omitted, graph exists → curated "no agent
    supplied" message; the real write is never attempted."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    cypher = "CREATE (k:KaizenEntry {entryId:'x', author:'graph-dba'})"
    out = server.run_query("g", cypher)
    assert "no `agent` parameter supplied" in out
    assert _writes_never_ran(client)


def test_author_write_with_matching_agent_succeeds(fake_client):
    """3 — a real CREATE with a matching `author:` literal is authorized and
    the write summary is rendered."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_created": 1, "properties_set": 8},
    )
    cypher = (
        "CREATE (k:KaizenEntry {entryId:'e1', date:'2026-08-17', fact:'x', "
        "evidence:'y', context:'z', suggestedHome:'unsure', author:'graph-dba', "
        "createdAt:'2026-08-17T00:00:00Z'})"
    )
    out = server.run_query("kaizen_graph_dba", cypher, "graph-dba")
    assert ("query", cypher, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out
    assert "nodes_created=1" in out
    assert "properties_set=8" in out


def test_author_write_with_mismatched_agent_is_rejected(fake_client):
    """4 — `author: 'cobb'` literal, `agent='graph-dba'` → rejected; no
    partial write (AC-6)."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    cypher = "CREATE (k:KaizenEntry {entryId:'e1', author:'cobb'})"
    out = server.run_query("kaizen_graph_dba", cypher, "graph-dba")
    assert "author 'cobb'" in out
    assert "agent='graph-dba'" in out
    assert _writes_never_ran(client)


def test_curator_clear_shape_with_cobb_succeeds(fake_client):
    """5 — the exact curator-clear skeleton, `agent='cobb'` → allowed."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_deleted": 1},
    )
    cypher = "MATCH (e:KaizenEntry {entryId: 'e1'}) DETACH DELETE e"
    out = server.run_query("kaizen_graph_dba", cypher, "cobb")
    assert ("query", cypher, server.TIMEOUT_MS) in client.calls
    assert "nodes_deleted=1" in out


def test_curator_clear_shape_with_non_curator_is_rejected(fake_client):
    """6 — same skeleton, `agent='graph-dba'` → rejected (curator-only)."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    cypher = "MATCH (e:KaizenEntry {entryId: 'e1'}) DETACH DELETE e"
    out = server.run_query("kaizen_graph_dba", cypher, "graph-dba")
    assert "not a recognized curator" in out
    assert _writes_never_ran(client)


def test_unrecognized_write_shape_is_rejected(fake_client):
    """7 — neither shape (no `author:` literal, not the curator skeleton) →
    rejected regardless of `agent`."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    cypher = "MATCH (n) DETACH DELETE n"
    out = server.run_query("g", cypher, "graph-dba")
    assert "neither an author-write" in out
    assert _writes_never_ran(client)


def test_migration_shaped_batch_with_single_shared_author_literal_succeeds(fake_client):
    """8 — the real §3.4 shape: one claim found, matches → allowed."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_created": 2, "properties_set": 16},
    )
    out = server.run_query("kaizen_graph_dba", _MIGRATION_CYPHER, "graph-dba")
    assert ("query", _MIGRATION_CYPHER, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out


def test_write_to_nonexistent_graph_without_agent_is_graph_not_found(fake_client):
    """9 — "empty key", no `agent` → today's exact graph-not-found message,
    unchanged (regression)."""
    client = fake_client(graphs=["one", "two"], error=_response_error(_EMPTY_KEY_ERROR))
    cypher = "CREATE (k:KaizenEntry {entryId:'e1', author:'graph-dba'})"
    out = server.run_query("kaizen_graph_dba", cypher)
    assert "Graph 'kaizen_graph_dba' does not exist." in out
    assert _writes_never_ran(client)


def test_write_to_nonexistent_graph_with_agent_and_matching_author_creates_it(fake_client):
    """10 — "empty key", `agent='graph-dba'`, the §3.4-shaped CREATE with a
    matching `author:` literal → the write runs (the migration path)."""
    client = fake_client(
        error=_response_error(_EMPTY_KEY_ERROR),
        write_stats={"nodes_created": 6, "properties_set": 48},
    )
    out = server.run_query("kaizen_graph_dba", _MIGRATION_CYPHER, "graph-dba")
    assert ("query", _MIGRATION_CYPHER, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out


def test_write_to_nonexistent_graph_with_agent_but_non_write_text_is_graph_not_found(
    fake_client,
):
    """11 — Pass-1 review M3. "empty key", `agent='graph-dba'` (supplied out of
    habit), plain read text (no write keyword) → routed straight to
    graph-not-found; `authorize_write()` never runs, `query()` never called."""
    client = fake_client(graphs=["one"], error=_response_error(_EMPTY_KEY_ERROR))
    out = server.run_query(
        "kaizen_graph_dba", "MATCH (e:KaizenEntry) RETURN e.fact", "graph-dba"
    )
    assert "Graph 'kaizen_graph_dba' does not exist." in out
    assert _writes_never_ran(client)


def test_profile_still_refused_regardless_of_agent(fake_client):
    """12 — PROFILE, any `agent` → unchanged refusal, no ro_query/query call at
    all (PROFILE can't be used to dodge §3.2)."""
    client = fake_client()
    out = server.run_query("g", "PROFILE MATCH (n) RETURN n", "graph-dba")
    assert out == server.PROFILE_REFUSAL
    assert client.calls == []


def test_explain_still_works_on_write_cypher(fake_client):
    """13 — EXPLAIN + a write statement → plan returned via the unchanged
    explain() path, regardless of `agent`; nothing executed."""
    client = fake_client(graphs=["g"], plan="Create")
    cypher = "CREATE (k:KaizenEntry {entryId:'e1', author:'graph-dba'})"
    out = server.run_query("g", f"EXPLAIN {cypher}", "graph-dba")
    assert ("explain", cypher) in client.calls
    assert _writes_never_ran(client)
    assert "Create" in out


def test_author_write_succeeds_despite_decoy_author_substring_in_evidence(fake_client):
    """14 — Pass-1 review M1. A decoy `author:`-shaped substring sitting inside
    `evidence`'s own string-literal span is excluded; only the real top-level
    `author: 'graph-dba'` is found → allowed."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_created": 1},
    )
    cypher = (
        "CREATE (k:KaizenEntry {fact: 'x', evidence: \"the pattern author: "
        "'someone-else' shows up in the log line\", context: 'c', "
        "suggestedHome: 'unsure', author: 'graph-dba', createdAt: 't'})"
    )
    out = server.run_query("kaizen_graph_dba", cypher, "graph-dba")
    assert ("query", cypher, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out


def test_set_based_author_reassignment_is_always_rejected(fake_client):
    """15 — Pass-1 review M2. `SET e.author = 'graph-dba'` against a node the
    caller doesn't own, even with a matching `agent` → still rejected, because
    `_kaizen_entry_create_map_spans()` finds no CREATE clause at all. Proves
    the SET path is categorically unreachable, not just unmatched by
    coincidence."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    cypher = "MATCH (e:KaizenEntry {entryId:'not-mine'}) SET e.author = 'graph-dba'"
    out = server.run_query("kaizen_graph_dba", cypher, "graph-dba")
    assert "neither an author-write" in out
    assert _writes_never_ran(client)


def test_set_map_merge_author_reassignment_is_always_rejected(fake_client):
    """15b — analyst's diff-scoped code re-gate (2026-08-18, Major M-A). Test
    15's dot-assignment spelling (`.author = '<value>'`) never matches
    `_AUTHOR_LITERAL_RE` (map-KEY colon form only), with or without
    CREATE-scoping — so it doesn't actually pin the M2 fix (CREATE-only
    scoping excluding SET). This uses the colon-form property-merge SET
    (`SET e += {author: '<value>'}`) instead, against a MATCHed (not
    newly-created) node: a form `_AUTHOR_LITERAL_RE` *would* match if
    CREATE-scoping were ever removed (the literal itself is spelled exactly
    like a CREATE map's `author:` key), so this genuinely dies under that
    mutation, while the real, shipped `_kaizen_entry_create_map_spans()`
    finds no CREATE clause here at all and correctly rejects it."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    cypher = "MATCH (e:KaizenEntry {entryId:'not-mine'}) SET e += {author: 'graph-dba'}"
    out = server.run_query("kaizen_graph_dba", cypher, "graph-dba")
    assert "neither an author-write" in out
    assert _writes_never_ran(client)


def test_nested_create_decoy_in_free_text_is_excluded(fake_client):
    """16 — Pass-2 review, M1-residual. Two sub-cases, both copied verbatim
    from the review's own verified reproductions."""
    # Over-rejection: a free-text field quotes a COMPLETE decoy CREATE clause,
    # but the real top-level `author: 'graph-dba'` is correct — must still be
    # allowed (before the fix, the decoy's own `author: 'evil'` also
    # registered as a claim and wrongly rejected this).
    over_rejection_cypher = (
        "CREATE (real:KaizenEntry {fact: 'x', "
        "evidence: \"example: CREATE (k:KaizenEntry {author: 'evil'})\", "
        "context: 'c', suggestedHome: 'unsure', author: 'graph-dba', createdAt: 't'})"
    )
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_created": 1},
    )
    out = server.run_query("kaizen_graph_dba", over_rejection_cypher, "graph-dba")
    assert ("query", over_rejection_cypher, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out

    # Under-enforcement: the real top-level map has NO `author:` property at
    # all; only a free-text-embedded decoy CREATE carries one. The decoy must
    # NOT satisfy authorization (before the fix, it wrongly did — allowing an
    # un-attributed node to be created).
    under_enforcement_cypher = (
        "CREATE (real:KaizenEntry {fact: 'x', "
        "evidence: \"example: CREATE (k:KaizenEntry {author: 'graph-dba'})\", "
        "context: 'c', suggestedHome: 'unsure', createdAt: 't'})"
    )
    client2 = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out2 = server.run_query("kaizen_graph_dba", under_enforcement_cypher, "graph-dba")
    assert "neither an author-write" in out2
    assert _writes_never_ran(client2)


# --------------------------------------------------------------------------
# 8 (cont'd) — M8 kaizen-agent-ontology: 4 new write shapes (plan §5, items 2-23)
# docs/plans/kaizen-agent-ontology.md, Version 3, approved Pass 3
# --------------------------------------------------------------------------

_PRODUCER_WRITE_CYPHER = (
    "MERGE (a:Agent {agentId: 'graph-dba'}) "
    "CREATE (a)-[:PRODUCED {sessionId: 's1'}]->(k:KaizenEntry {"
    "entryId: 'e1', date: '2026-08-22', fact: 'f', evidence: 'e', context: 'c', "
    "suggestedHome: 'unsure', createdAt: 't'})"
)

_MENTIONS_WRITE_CYPHER = (
    "MATCH (k:KaizenEntry {entryId: 'e1'}) "
    "MERGE (a:Agent {agentId: 'security-expert'}) "
    "MERGE (k)-[:MENTIONS]->(a)"
)

_PRODUCER_EDGE_RESOLVE_CYPHER = (
    "MATCH (:Agent)-[p:PRODUCED]->(k:KaizenEntry {entryId: 'e1'}) DELETE p"
)

_MENTION_EDGE_RESOLVE_CYPHER = (
    "MATCH (k:KaizenEntry {entryId: 'e1'})-[m:MENTIONS]->(:Agent {agentId: 'security-expert'}) "
    "DELETE m"
)

# Finding-1 closure regression fixtures — reproduced verbatim from
# docs/reviews/kaizen-agent-ontology.md (Attacks A/B, Pass 1; Attack C, Pass 2/3).
_ATTACK_A_CYPHER = (
    "CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', "
    "context:'c', suggestedHome:'unsure', author:'analyst', createdAt:'t'}) "
    "MATCH (victim:KaizenEntry {entryId:'not-mine'}) DETACH DELETE victim"
)

_ATTACK_B_CYPHER = (
    "CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', "
    "context:'c', suggestedHome:'unsure', author:'analyst', createdAt:'t'}) "
    "MERGE (a:Agent {agentId: 'cobb'}) "
    "CREATE (a)-[:PRODUCED {sessionId:'s'}]->(k:KaizenEntry {entryId:'forged', "
    "date:'2026-08-22', fact:'fabricated, attributed to cobb', evidence:'e', context:'c', "
    "suggestedHome:'unsure', createdAt:'t'})"
)

_ATTACK_C_SET_CYPHER = (
    "CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', "
    "context:'c', suggestedHome:'unsure', author:'analyst', createdAt:'t'}) "
    "MATCH (victim:KaizenEntry {entryId:'not-mine'}) "
    "SET victim.author = 'nobody', victim.fact = 'tampered'"
)

_ATTACK_C_REMOVE_CYPHER = (
    "CREATE (junk:KaizenEntry {entryId:'z1', date:'2026-08-22', fact:'f', evidence:'e', "
    "context:'c', suggestedHome:'unsure', author:'analyst', createdAt:'t'}) "
    "MATCH (victim:KaizenEntry {entryId:'not-mine'}) REMOVE victim.author"
)


def test_producer_write_with_matching_agent_succeeds(fake_client):
    """17 (plan item 2) — a real producer-write, `agent` matching the MERGE's
    `agentId` → authorized, write summary rendered (mirrors test 3's shape)."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_created": 1, "relationships_created": 1, "properties_set": 8},
    )
    out = server.run_query("kaizen_team", _PRODUCER_WRITE_CYPHER, "graph-dba")
    assert ("query", _PRODUCER_WRITE_CYPHER, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out
    assert "relationships_created=1" in out


def test_producer_write_with_mismatched_agent_is_rejected(fake_client):
    """18 (plan item 3) — `agent` mismatched against the MERGE's `agentId` →
    rejected, no partial write (mirrors test 4)."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", _PRODUCER_WRITE_CYPHER, "cobb")
    assert "MERGE (:Agent {agentId: 'graph-dba'})" in out
    assert "agent='cobb'" in out
    assert _writes_never_ran(client)


def test_producer_write_with_session_id_map_present_succeeds(fake_client):
    """19 (plan item 4) — the optional `{sessionId: ...}` property map present
    → authorized."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_created": 1, "relationships_created": 1},
    )
    assert "sessionId" in _PRODUCER_WRITE_CYPHER
    out = server.run_query("kaizen_team", _PRODUCER_WRITE_CYPHER, "graph-dba")
    assert ("query", _PRODUCER_WRITE_CYPHER, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out


def test_producer_write_with_session_id_map_omitted_succeeds(fake_client):
    """20 (plan item 5) — the `sessionId` map omitted entirely (the template's
    "or omit this key" case) → still authorized."""
    cypher = (
        "MERGE (a:Agent {agentId: 'graph-dba'}) "
        "CREATE (a)-[:PRODUCED]->(k:KaizenEntry {"
        "entryId: 'e1', date: '2026-08-22', fact: 'f', evidence: 'e', context: 'c', "
        "suggestedHome: 'unsure', createdAt: 't'})"
    )
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_created": 1, "relationships_created": 1},
    )
    out = server.run_query("kaizen_team", cypher, "graph-dba")
    assert ("query", cypher, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out


def test_producer_write_with_literal_braces_in_free_text_succeeds(fake_client):
    """21 (plan item 6) — free-text `fact`/`evidence` containing literal
    `{`/`}` characters → still authorized (mirrors tests 14/16's decoy-
    robustness intent, now for the new anchor)."""
    cypher = (
        "MERGE (a:Agent {agentId: 'graph-dba'}) "
        "CREATE (a)-[:PRODUCED {sessionId: 's1'}]->(k:KaizenEntry {"
        "entryId: 'e1', date: '2026-08-22', fact: 'contains { and } braces', "
        "evidence: 'also has { nested } text', context: 'c', "
        "suggestedHome: 'unsure', createdAt: 't'})"
    )
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_created": 1, "relationships_created": 1},
    )
    out = server.run_query("kaizen_team", cypher, "graph-dba")
    assert ("query", cypher, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out


def test_producer_write_with_trailing_extra_clause_is_rejected(fake_client):
    """22 (plan item 7) — a trailing extra clause after the CREATE's closing
    `)` → rejected — pins the "nothing else follows" structural check (§3.1
    step 2e)."""
    cypher = _PRODUCER_WRITE_CYPHER + " SET k.extra = 'x'"
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", cypher, "graph-dba")
    assert "Rejected" in out
    assert _writes_never_ran(client)


def test_producer_write_with_trailing_return_clause_gives_specific_trailer_message(
    fake_client,
):
    """New (2026-08-23) — cobb's Finding 1: a producer-write with a trailing
    `RETURN` clause after the `KaizenEntry` map's closing brace is still
    rejected (unchanged outcome, mirrors test 22), but the message now names
    the trailing-clause trap specifically instead of falling through to the
    generic FR-8 fallback that lists all 6 shapes with no signal about how
    close the statement came."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    cypher = _PRODUCER_WRITE_CYPHER + " RETURN k.entryId"
    out = server.run_query("kaizen_team", cypher, "graph-dba")
    assert "Rejected" in out
    assert "KaizenEntry map's closing brace" in out
    assert "RETURN k.entryId" in out
    assert "neither an author-write" not in out  # not the generic FR-8 fallback
    assert _writes_never_ran(client)


def test_producer_write_with_trailing_return_k_gives_specific_trailer_message(fake_client):
    """New (2026-08-23) — same trap, `RETURN k` spelling (no property
    accessor) — proves the near-miss check isn't keyed to `.entryId`
    specifically."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    cypher = _PRODUCER_WRITE_CYPHER + " RETURN k"
    out = server.run_query("kaizen_team", cypher, "graph-dba")
    assert "Rejected" in out
    assert "KaizenEntry map's closing brace" in out
    assert "RETURN k" in out
    assert _writes_never_ran(client)


def test_producer_write_with_mismatched_create_variable_is_rejected(fake_client):
    """23 (plan item 8) — the `CREATE`'s referenced variable not matching the
    `MERGE`'s bound variable → rejected — pins the var-binding check (§3.1
    step 2b)."""
    cypher = (
        "MERGE (a:Agent {agentId: 'graph-dba'}) "
        "CREATE (b)-[:PRODUCED {sessionId: 's1'}]->(k:KaizenEntry {"
        "entryId: 'e1', date: '2026-08-22', fact: 'f', evidence: 'e', context: 'c', "
        "suggestedHome: 'unsure', createdAt: 't'})"
    )
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", cypher, "graph-dba")
    assert "Rejected" in out
    assert _writes_never_ran(client)


def test_producer_write_with_reversed_clause_order_is_rejected(fake_client):
    """24 (plan item 9) — `CREATE`/`MERGE` in reversed order → rejected (not
    the one recognized skeleton)."""
    cypher = (
        "CREATE (a)-[:PRODUCED {sessionId: 's1'}]->(k:KaizenEntry {"
        "entryId: 'e1', date: '2026-08-22', fact: 'f', evidence: 'e', context: 'c', "
        "suggestedHome: 'unsure', createdAt: 't'}) "
        "MERGE (a:Agent {agentId: 'graph-dba'})"
    )
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", cypher, "graph-dba")
    assert "Rejected" in out
    assert _writes_never_ran(client)


def test_mentions_write_with_cobb_succeeds(fake_client):
    """25 (plan item 10) — MENTIONS-write, `agent='cobb'` → authorized;
    mirrors test 5's shape but for the new regex."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"relationships_created": 1},
    )
    out = server.run_query("kaizen_team", _MENTIONS_WRITE_CYPHER, "cobb")
    assert ("query", _MENTIONS_WRITE_CYPHER, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out


def test_mentions_write_with_non_curator_is_rejected(fake_client):
    """26 (plan item 11) — MENTIONS-write, non-curator `agent` → rejected
    (mirrors test 6)."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", _MENTIONS_WRITE_CYPHER, "graph-dba")
    assert "not a recognized curator" in out
    assert _writes_never_ran(client)


def test_mentions_write_with_mismatched_backreference_is_rejected(fake_client):
    """27 (plan item 12) — MENTIONS-write with the second `MERGE`'s
    referenced variables not matching the first `MATCH`/`MERGE`'s bound
    variables → rejected — pins the backreference check."""
    cypher = (
        "MATCH (k:KaizenEntry {entryId: 'e1'}) "
        "MERGE (a:Agent {agentId: 'security-expert'}) "
        "MERGE (other)-[:MENTIONS]->(a)"
    )
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", cypher, "cobb")
    assert "Rejected" in out
    assert _writes_never_ran(client)


def test_producer_edge_resolve_with_cobb_succeeds(fake_client):
    """28 (plan item 13) — producer-edge-resolve, `agent='cobb'` →
    authorized (`relationships_deleted=1`, mirroring test 5's
    counter-assertion style)."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"relationships_deleted": 1},
    )
    out = server.run_query("kaizen_team", _PRODUCER_EDGE_RESOLVE_CYPHER, "cobb")
    assert ("query", _PRODUCER_EDGE_RESOLVE_CYPHER, server.TIMEOUT_MS) in client.calls
    assert "relationships_deleted=1" in out


def test_producer_edge_resolve_with_non_curator_is_rejected(fake_client):
    """29 (plan item 14) — producer-edge-resolve, non-curator → rejected."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", _PRODUCER_EDGE_RESOLVE_CYPHER, "graph-dba")
    assert "not a recognized curator" in out
    assert _writes_never_ran(client)


def test_mention_edge_resolve_with_cobb_succeeds(fake_client):
    """30 (plan item 15) — mention-edge-resolve, `agent='cobb'` →
    authorized."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"relationships_deleted": 1},
    )
    out = server.run_query("kaizen_team", _MENTION_EDGE_RESOLVE_CYPHER, "cobb")
    assert ("query", _MENTION_EDGE_RESOLVE_CYPHER, server.TIMEOUT_MS) in client.calls
    assert "relationships_deleted=1" in out


def test_mention_edge_resolve_with_non_curator_is_rejected(fake_client):
    """31 (plan item 16) — mention-edge-resolve, non-curator → rejected."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", _MENTION_EDGE_RESOLVE_CYPHER, "graph-dba")
    assert "not a recognized curator" in out
    assert _writes_never_ran(client)


def test_curator_clear_shape_zero_space_entryid_succeeds_for_cobb(fake_client):
    """New (2026-08-23) — cobb's Finding 2: `entryId:'e1'` with NO space after
    the colon (today rejected by a hardcoded literal single space in
    `_CURATOR_CLEAR_RE`) now authorizes for a recognized curator agent —
    whitespace has no semantic meaning in Cypher here, so this only widens
    formatting tolerance, not the set of authorized statements."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"nodes_deleted": 1},
    )
    cypher = "MATCH (e:KaizenEntry {entryId:'e1'}) DETACH DELETE e"
    out = server.run_query("kaizen_graph_dba", cypher, "cobb")
    assert ("query", cypher, server.TIMEOUT_MS) in client.calls
    assert "nodes_deleted=1" in out


def test_curator_clear_shape_zero_space_entryid_rejected_for_non_curator(fake_client):
    """New (2026-08-23) — same zero-space statement, non-curator agent → still
    rejected; the tolerance widening changes nothing about *who* is
    authorized, only whitespace shape recognition."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    cypher = "MATCH (e:KaizenEntry {entryId:'e1'}) DETACH DELETE e"
    out = server.run_query("kaizen_graph_dba", cypher, "graph-dba")
    assert "not a recognized curator" in out
    assert _writes_never_ran(client)


def test_mentions_write_zero_space_entryid_and_agentid_succeeds_for_cobb(fake_client):
    """New (2026-08-23) — Finding 2, `_MENTIONS_WRITE_RE`: zero space after
    both `entryId:` and `agentId:` now authorizes for `cobb`."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"relationships_created": 1},
    )
    cypher = (
        "MATCH (k:KaizenEntry {entryId:'e1'}) "
        "MERGE (a:Agent {agentId:'security-expert'}) "
        "MERGE (k)-[:MENTIONS]->(a)"
    )
    out = server.run_query("kaizen_team", cypher, "cobb")
    assert ("query", cypher, server.TIMEOUT_MS) in client.calls
    assert "write ok" in out


def test_producer_edge_resolve_zero_space_entryid_succeeds_for_cobb(fake_client):
    """New (2026-08-23) — Finding 2, `_PRODUCER_EDGE_RESOLVE_RE`: zero space
    after `entryId:` now authorizes for `cobb`."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"relationships_deleted": 1},
    )
    cypher = "MATCH (:Agent)-[p:PRODUCED]->(k:KaizenEntry {entryId:'e1'}) DELETE p"
    out = server.run_query("kaizen_team", cypher, "cobb")
    assert ("query", cypher, server.TIMEOUT_MS) in client.calls
    assert "relationships_deleted=1" in out


def test_mention_edge_resolve_zero_space_entryid_and_agentid_succeeds_for_cobb(fake_client):
    """New (2026-08-23) — Finding 2, `_MENTION_EDGE_RESOLVE_RE`: zero space
    after both `entryId:` and `agentId:` now authorizes for `cobb`."""
    client = fake_client(
        error=_response_error(_READ_ONLY_ERROR),
        write_stats={"relationships_deleted": 1},
    )
    cypher = (
        "MATCH (k:KaizenEntry {entryId:'e1'})-[m:MENTIONS]->(:Agent {agentId:'security-expert'}) "
        "DELETE m"
    )
    out = server.run_query("kaizen_team", cypher, "cobb")
    assert ("query", cypher, server.TIMEOUT_MS) in client.calls
    assert "relationships_deleted=1" in out


def test_all_four_new_shapes_rejected_without_agent(fake_client):
    """32 (plan item 17) — all 4 new shapes still correctly rejected without
    `agent` supplied (mirrors test 2's coverage, extended)."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    for cypher in (
        _PRODUCER_WRITE_CYPHER,
        _MENTIONS_WRITE_CYPHER,
        _PRODUCER_EDGE_RESOLVE_CYPHER,
        _MENTION_EDGE_RESOLVE_CYPHER,
    ):
        out = server.run_query("kaizen_team", cypher, None)
        assert "no `agent` parameter supplied" in out, cypher
    assert _writes_never_ran(client)


def test_finding1_attack_a_is_rejected(fake_client):
    """33 (plan item 18) — Finding-1 closure, Attack A (analyst review,
    docs/reviews/kaizen-agent-ontology.md, reproduced verbatim): the exact
    self-attributed-decoy-CREATE-plus-DETACH-DELETE compound statement
    (§3.1a), called with `agent='analyst'` → rejected in full; no partial
    write."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", _ATTACK_A_CYPHER, "analyst")
    assert "Rejected" in out
    assert _writes_never_ran(client)


def test_finding1_attack_b_is_rejected(fake_client):
    """34 (plan item 19) — Finding-1 closure, Attack B (same source,
    reproduced verbatim): the exact self-attributed-decoy-CREATE-plus-forged-
    MERGE(:Agent{agentId:'cobb'})-PRODUCED compound statement (§3.1a), called
    with `agent='analyst'` → rejected in full; no partial write."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", _ATTACK_B_CYPHER, "analyst")
    assert "Rejected" in out
    assert _writes_never_ran(client)


def test_migration_batch_and_plain_author_write_contain_no_foreign_trigger_keywords():
    """35 (plan item 20) — a legitimate single-clause author-write and the
    legitimate migration-batch shape both still succeed after
    `_has_foreign_trigger_outside_strings()` is added (already covered by not
    modifying tests 3/8/10) — add one explicit, documentation-grade
    regression pin independent of the end-to-end test: a bare-word scan of
    both fixtures' exact text finds neither `MERGE`, `DELETE`, `SET` nor
    `REMOVE`."""
    assert server._has_foreign_trigger_outside_strings(_MIGRATION_CYPHER) is False
    plain_author_write = (
        "CREATE (k:KaizenEntry {entryId:'e1', date:'2026-08-17', fact:'x', "
        "evidence:'y', context:'z', suggestedHome:'unsure', author:'graph-dba', "
        "createdAt:'2026-08-17T00:00:00Z'})"
    )
    assert server._has_foreign_trigger_outside_strings(plain_author_write) is False


def test_author_write_succeeds_with_decoy_foreign_trigger_words_quoted_in_free_text(
    fake_client,
):
    """36 (plan item 21, extended per Pass 3's non-blocking nit) — a decoy
    that itself quotes "MERGE"/"DELETE" (Pass-1 era) — and, symmetrically,
    "SET"/"REMOVE" (Pass-3 nit) — inside a free-text field of an otherwise-
    legitimate single-clause author-write → still authorized (string-literal
    exclusion in `_has_foreign_trigger_outside_strings()` correctly ignores
    it) — extends the existing decoy-robustness style (tests 14/16) to the
    new check specifically, symmetrically across all 4 widened keywords."""
    for keyword in ("MERGE", "DELETE", "SET", "REMOVE"):
        cypher = (
            "CREATE (k:KaizenEntry {entryId:'e1', date:'2026-08-17', fact:'x', "
            f"evidence: \"a decoy quoting {keyword} (n) right here\", context:'z', "
            "suggestedHome:'unsure', author:'graph-dba', createdAt:'t'})"
        )
        client = fake_client(
            error=_response_error(_READ_ONLY_ERROR),
            write_stats={"nodes_created": 1},
        )
        out = server.run_query("kaizen_graph_dba", cypher, "graph-dba")
        assert ("query", cypher, server.TIMEOUT_MS) in client.calls, keyword
        assert "write ok" in out, keyword


def test_finding1_attack_c_set_chained_variant_is_rejected(fake_client):
    """37 (plan item 22) — Finding-1 closure, Attack C, `SET`-chained variant
    (analyst review Pass 2, reproduced verbatim, §3.1a): the self-attributed-
    decoy-CREATE-plus-MATCH...SET compound statement, called with
    `agent='analyst'` → rejected in full; no partial write. Explicitly
    includes the `victim.author = '<other-agent>'` sub-case (not just
    `victim.fact = 'tampered'`) — the sub-case that reopens tests 15/15b's
    exact concern in chained form."""
    assert "victim.author = 'nobody'" in _ATTACK_C_SET_CYPHER
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", _ATTACK_C_SET_CYPHER, "analyst")
    assert "Rejected" in out
    assert _writes_never_ran(client)


def test_finding1_attack_c_remove_chained_variant_is_rejected(fake_client):
    """38 (plan item 23) — Finding-1 closure, `REMOVE`-chained variant
    (symmetric to 37, per the review's own note that a `REMOVE
    victim.author` variant is symmetric and equally unguarded): the same
    decoy-CREATE chained with a `MATCH (victim:KaizenEntry {...}) REMOVE
    victim.author` clause, `agent='analyst'` → rejected in full; no partial
    write."""
    client = fake_client(error=_response_error(_READ_ONLY_ERROR))
    out = server.run_query("kaizen_team", _ATTACK_C_REMOVE_CYPHER, "analyst")
    assert "Rejected" in out
    assert _writes_never_ran(client)


# --------------------------------------------------------------------------
# live — against a REAL FalkorDB, on a scratch graph this module owns
# --------------------------------------------------------------------------


def _scratch_graph_name() -> str:
    """Key for the live suite's throwaway scratch graph (see `live_graph` below).

    C-321: must not derive from `os.getpid()` — that's `1` inside every
    container's PID namespace, so two concurrent containerized `-m live` runs
    (or an interrupted one leaving residue) would collide on the same key
    against the shared FalkorDB. `uuid4` is unique regardless of PID namespace.
    """
    return f"_cypher_mcp_selftest_{uuid4().hex[:8]}"


def test_scratch_graph_name_is_unique_across_calls():
    """C-321 regression pin: offline, no FalkorDB needed.

    `os.getpid()` was constant for the whole test process (and is `1` inside
    every container's PID namespace regardless), so two calls used to collide
    on the same key. The uuid4-derived key must not.
    """
    assert _scratch_graph_name() != _scratch_graph_name()


@pytest.fixture(scope="module")
def live_graph():
    """Create a throwaway graph, yield its name, delete it afterwards.

    Deliberately self-contained: the tests below must not depend on the contents
    of `cpg_*`, `ws:*` or `reference`, and must not modify them.
    """
    client = server.get_client()
    name = _scratch_graph_name()
    graph = client.select_graph(name)
    graph.query(
        "CREATE (:CypherMcpSelfTest {name:'alpha', note:'line1\nline2'}), "
        "(:CypherMcpSelfTest {name:'beta', note:'plain'})"
    )
    try:
        yield name
    finally:
        graph.delete()


@pytest.mark.live
def test_live_two_column_query(live_graph):
    out = server.run_query(
        live_graph,
        "MATCH (n:CypherMcpSelfTest) RETURN n.name AS name, n.note AS note ORDER BY name",
    )
    lines = out.splitlines()
    assert lines[0].startswith(f"graph={live_graph} · rows=2 · ")
    assert lines[1] == "name | note"
    assert lines[2:] == ["alpha | line1\\nline2", "beta | plain"]


@pytest.fixture(scope="module")
def live_kaizen_graph():
    """A throwaway scratch graph with `Agent.agentId`'s index+constraint
    freshly provisioned (mirrors `kaizen_team`'s own real DDL, plan §5 "Live,
    automated") — provisioned directly via the raw client, never through
    `mcp__cypher__query` (schema DDL is unconditionally rejected there, same
    restriction as `kaizen_team`'s own S0). No dependency on S0 having
    already landed against the real `kaizen_team` graph."""
    client = server.get_client()
    name = _scratch_graph_name()
    graph = client.select_graph(name)
    graph.create_node_range_index("Agent", "agentId")
    graph.create_node_unique_constraint("Agent", "agentId")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        constraints = graph.list_constraints()
        if constraints and all(c.get("status") == "OPERATIONAL" for c in constraints):
            break
        time.sleep(0.1)
    try:
        yield name
    finally:
        graph.delete()


@pytest.mark.live
def test_live_producer_write_then_traversal_read(live_kaizen_graph):
    """New producer-write shape, executed for real, then read back via
    graph-dba's own §5 traversal recipe."""
    entry_id = f"e-{uuid4().hex[:8]}"
    cypher = (
        "MERGE (a:Agent {agentId: 'graph-dba'}) "
        f"CREATE (a)-[:PRODUCED {{sessionId: 's1'}}]->(k:KaizenEntry {{entryId: '{entry_id}', "
        "date: '2026-08-22', fact: 'live producer-write test', evidence: 'e', context: 'c', "
        "suggestedHome: 'unsure', createdAt: 't'})"
    )
    out = server.run_query(live_kaizen_graph, cypher, "graph-dba")
    assert "write ok" in out
    assert "relationships_created=1" in out

    read = server.run_query(
        live_kaizen_graph,
        "MATCH (a:Agent {agentId: 'graph-dba'})-[:PRODUCED]->(k:KaizenEntry) "
        "RETURN k.entryId AS entryId",
    )
    assert entry_id in read


@pytest.mark.live
def test_live_producer_write_mismatched_agent_is_rejected(live_kaizen_graph):
    """Mismatched `agent` against a real graph — still rejected server-side,
    nothing written."""
    entry_id = f"e-{uuid4().hex[:8]}"
    cypher = (
        "MERGE (a:Agent {agentId: 'graph-dba'}) "
        f"CREATE (a)-[:PRODUCED]->(k:KaizenEntry {{entryId: '{entry_id}', "
        "date: '2026-08-22', fact: 'f', evidence: 'e', context: 'c', "
        "suggestedHome: 'unsure', createdAt: 't'})"
    )
    out = server.run_query(live_kaizen_graph, cypher, "cobb")
    assert "Rejected" in out
    read = server.run_query(
        live_kaizen_graph, f"MATCH (k:KaizenEntry {{entryId: '{entry_id}'}}) RETURN k"
    )
    assert "(no rows)" in read


@pytest.mark.live
def test_live_mentions_write_then_producer_and_mention_edge_resolve(live_kaizen_graph):
    """MENTIONS-write, both edge-resolve shapes, all executed for real
    against the same disposable entry, ending with zero residue."""
    entry_id = f"e-{uuid4().hex[:8]}"
    producer_cypher = (
        "MERGE (a:Agent {agentId: 'graph-dba'}) "
        f"CREATE (a)-[:PRODUCED {{sessionId: 's1'}}]->(k:KaizenEntry {{entryId: '{entry_id}', "
        "date: '2026-08-22', fact: 'f', evidence: 'e', context: 'c', "
        "suggestedHome: 'unsure', createdAt: 't'})"
    )
    out = server.run_query(live_kaizen_graph, producer_cypher, "graph-dba")
    assert "write ok" in out

    mentions_cypher = (
        f"MATCH (k:KaizenEntry {{entryId: '{entry_id}'}}) "
        "MERGE (a:Agent {agentId: 'security-expert'}) "
        "MERGE (k)-[:MENTIONS]->(a)"
    )
    out = server.run_query(live_kaizen_graph, mentions_cypher, "cobb")
    assert "write ok" in out

    resolve_producer_cypher = (
        f"MATCH (:Agent)-[p:PRODUCED]->(k:KaizenEntry {{entryId: '{entry_id}'}}) DELETE p"
    )
    out = server.run_query(live_kaizen_graph, resolve_producer_cypher, "cobb")
    assert "relationships_deleted=1" in out

    resolve_mention_cypher = (
        f"MATCH (k:KaizenEntry {{entryId: '{entry_id}'}})-[m:MENTIONS]->"
        "(:Agent {agentId: 'security-expert'}) DELETE m"
    )
    out = server.run_query(live_kaizen_graph, resolve_mention_cypher, "cobb")
    assert "relationships_deleted=1" in out

    # Both edges resolved, node not yet cleared — confirm it's still there,
    # then clear it via the unchanged curator-clear shape, leaving no residue.
    clear_cypher = f"MATCH (e:KaizenEntry {{entryId: '{entry_id}'}}) DETACH DELETE e"
    out = server.run_query(live_kaizen_graph, clear_cypher, "cobb")
    assert "nodes_deleted=1" in out
    read = server.run_query(
        live_kaizen_graph, f"MATCH (k:KaizenEntry {{entryId: '{entry_id}'}}) RETURN k"
    )
    assert "(no rows)" in read


@pytest.mark.live
def test_live_multiline_query_is_accepted_verbatim(live_graph):
    single = server.run_query(
        live_graph, "MATCH (n:CypherMcpSelfTest) RETURN n.name AS name ORDER BY name"
    )
    multi = server.run_query(
        live_graph,
        "MATCH (n:CypherMcpSelfTest)\n"
        "    // no shell quoting, no escaping\n"
        "RETURN n.name AS name\n"
        "ORDER BY name",
    )
    assert single.splitlines()[1:] == multi.splitlines()[1:]


@pytest.mark.live
def test_live_missing_graph_does_not_materialise_a_key(live_graph):
    client = server.get_client()
    before = set(client.list_graphs())
    absent = f"{live_graph}_absent"
    for cypher in ("MATCH (n) RETURN n", "EXPLAIN MATCH (n) RETURN n"):
        out = server.run_query(absent, cypher)
        assert f"Graph '{absent}' does not exist." in out
        assert "graph-dba agent" in out
    assert set(client.list_graphs()) == before


@pytest.mark.live
def test_live_syntax_error_keeps_falkordbs_line_and_column(live_graph):
    out = server.run_query(live_graph, "MATCH (n:CypherMcpSelfTest RETURN n")
    assert "line: 1" in out and "column:" in out
    assert out.splitlines()[-1] == server.SCHEMA_POINTER


@pytest.mark.live
def test_live_write_without_agent_is_rejected_server_side(live_graph):
    """A write against a REAL, existing graph is still refused with no `agent`
    supplied — updated for the generic-cypher-mcp write path (`docs/plans/
    generic-cypher-mcp.md` §3.1): `GRAPH.RO_QUERY` really does reject it
    server-side first (proving the "ro_query" classification branch fires
    against live FalkorDB, not just the fake), and `authorize_write()` then
    curates the more specific "no `agent` supplied" message rather than the
    old generic read-only refusal. Either way, nothing is written."""
    client = server.get_client()
    out = server.run_query(live_graph, "CREATE (:CypherMcpWriteProbe)")
    assert "no `agent` parameter supplied" in out
    check = client.select_graph(live_graph).ro_query(
        "MATCH (n:CypherMcpWriteProbe) RETURN count(n)"
    )
    assert check.result_set[0][0] == 0


@pytest.mark.live
def test_live_explain_returns_a_plan_not_results(live_graph):
    out = server.run_query(live_graph, "EXPLAIN MATCH (n:CypherMcpSelfTest) RETURN n.name")
    assert out.splitlines()[0] == f"graph={live_graph} · EXPLAIN (plan only — nothing was executed)"
    assert "Scan" in out
    assert "alpha" not in out


@pytest.mark.live
def test_live_profile_is_refused_even_though_falkordb_would_return_rows(live_graph):
    """Raw GRAPH.RO_QUERY accepts `/* c */ PROFILE …` and returns ROWS.

    That is the wrong-answer class the refusal exists to prevent, so assert both
    halves: FalkorDB really does answer, and the tool really does not.
    """
    client = server.get_client()
    raw = client.select_graph(live_graph).ro_query(
        "/* c */ PROFILE MATCH (n:CypherMcpSelfTest) RETURN count(n)"
    )
    assert raw.result_set[0][0] == 2  # FalkorDB ignored the directive

    for cypher in (
        "PROFILE MATCH (n:CypherMcpSelfTest) RETURN count(n)",
        "/* c */ PROFILE MATCH (n:CypherMcpSelfTest) RETURN count(n)",
        "// c\nPROFILE MATCH (n:CypherMcpSelfTest) RETURN count(n)",
    ):
        assert server.run_query(live_graph, cypher) == server.PROFILE_REFUSAL
