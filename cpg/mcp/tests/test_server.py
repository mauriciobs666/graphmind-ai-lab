"""Unit tests for the `cpg` MCP server (plan §5 S2, §7.1).

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


class FakeGraph:
    def __init__(self, calls, *, header=None, rows=None, error=None, plan="PLAN"):
        self._calls = calls
        self._header = header if header is not None else [[2, "n"]]
        self._rows = rows if rows is not None else [["x"]]
        self._error = error
        self._plan = plan

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


def test_input_schema_has_exactly_two_required_string_parameters():
    """FR-2 is a hard stakeholder requirement — a third parameter is a defect."""
    schema = _tools()[0].inputSchema
    assert set(schema["properties"]) == {"graph", "cypher"}
    assert sorted(schema["required"]) == ["cypher", "graph"]
    assert {p["type"] for p in schema["properties"].values()} == {"string"}


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
# live — against a REAL FalkorDB, on a scratch graph this module owns
# --------------------------------------------------------------------------


def _scratch_graph_name() -> str:
    """Key for the live suite's throwaway scratch graph (see `live_graph` below).

    C-321: must not derive from `os.getpid()` — that's `1` inside every
    container's PID namespace, so two concurrent containerized `-m live` runs
    (or an interrupted one leaving residue) would collide on the same key
    against the shared FalkorDB. `uuid4` is unique regardless of PID namespace.
    """
    return f"_cpg_mcp_selftest_{uuid4().hex[:8]}"


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
        "CREATE (:CpgMcpSelfTest {name:'alpha', note:'line1\nline2'}), "
        "(:CpgMcpSelfTest {name:'beta', note:'plain'})"
    )
    try:
        yield name
    finally:
        graph.delete()


@pytest.mark.live
def test_live_two_column_query(live_graph):
    out = server.run_query(
        live_graph,
        "MATCH (n:CpgMcpSelfTest) RETURN n.name AS name, n.note AS note ORDER BY name",
    )
    lines = out.splitlines()
    assert lines[0].startswith(f"graph={live_graph} · rows=2 · ")
    assert lines[1] == "name | note"
    assert lines[2:] == ["alpha | line1\\nline2", "beta | plain"]


@pytest.mark.live
def test_live_multiline_query_is_accepted_verbatim(live_graph):
    single = server.run_query(
        live_graph, "MATCH (n:CpgMcpSelfTest) RETURN n.name AS name ORDER BY name"
    )
    multi = server.run_query(
        live_graph,
        "MATCH (n:CpgMcpSelfTest)\n"
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
    out = server.run_query(live_graph, "MATCH (n:CpgMcpSelfTest RETURN n")
    assert "line: 1" in out and "column:" in out
    assert out.splitlines()[-1] == server.SCHEMA_POINTER


@pytest.mark.live
def test_live_write_is_rejected_server_side(live_graph):
    client = server.get_client()
    out = server.run_query(live_graph, "CREATE (:CpgMcpWriteProbe)")
    assert out.startswith("This tool is read-only (GRAPH.RO_QUERY).")
    check = client.select_graph(live_graph).ro_query(
        "MATCH (n:CpgMcpWriteProbe) RETURN count(n)"
    )
    assert check.result_set[0][0] == 0


@pytest.mark.live
def test_live_explain_returns_a_plan_not_results(live_graph):
    out = server.run_query(live_graph, "EXPLAIN MATCH (n:CpgMcpSelfTest) RETURN n.name")
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
        "/* c */ PROFILE MATCH (n:CpgMcpSelfTest) RETURN count(n)"
    )
    assert raw.result_set[0][0] == 2  # FalkorDB ignored the directive

    for cypher in (
        "PROFILE MATCH (n:CpgMcpSelfTest) RETURN count(n)",
        "/* c */ PROFILE MATCH (n:CpgMcpSelfTest) RETURN count(n)",
        "// c\nPROFILE MATCH (n:CpgMcpSelfTest) RETURN count(n)",
    ):
        assert server.run_query(live_graph, cypher) == server.PROFILE_REFUSAL
