"""Offline tests of `coordgraph.store`: the statements, the write boundary, the endpoint guard,
bootstrap ordering. A recording fake stands in for the graph; the live behaviour is
`test_store_live.py`'s (ephemeral container only)."""
import sys

import pytest

from coordgraph import commands_store, store
from coordgraph.keys import KeyHygieneError

LEDGER_STATEMENT = """OPTIONAL MATCH (old:Source {path: $path})
OPTIONAL MATCH (old)-[]->(child)
WITH old, collect(child) AS children
FOREACH (c IN children | DETACH DELETE c)
FOREACH (o IN CASE WHEN old IS NULL THEN [] ELSE [old] END | DETACH DELETE o)
WITH count(*) AS _
CREATE (s:Source:Ledger)
SET s = $meta
WITH s
UNWIND $rows AS row
CREATE (s)-[:HAS_UNIT]->(u:Unit)
SET u = row
RETURN count(u) AS written"""


class Result:
    def __init__(self, rows=(), deleted=0):
        self.result_set = list(rows)
        self.nodes_deleted = deleted


class FakeGraph:
    """Records every call; `script` maps a query prefix to a callable/list/exception."""

    def __init__(self, script=None):
        self.calls: list[tuple[str, dict | None]] = []
        self.script = script or {}

    def query(self, q, params=None):
        self.calls.append((q, params))
        for prefix, answer in self.script.items():
            if q.startswith(prefix):
                value = answer(q, params) if callable(answer) else answer
                if isinstance(value, Exception):
                    raise value
                return value
        return Result([[1]])


class FakeDb:
    def __init__(self):
        self.commands: list[tuple] = []

    def execute_command(self, *args):
        self.commands.append(args)
        return "PENDING"


# --- the statements are the note's, verbatim ---------------------------------------------------

def test_ledger_statement_is_the_note_s_verbatim():
    assert store.REPLACE["ledger"] == LEDGER_STATEMENT


def test_backlog_statement_differs_only_in_labels():
    expected = (LEDGER_STATEMENT.replace("Source:Ledger", "Source:Backlog")
                .replace("HAS_UNIT", "HAS_ITEM").replace("u:Unit", "u:Item"))
    assert store.REPLACE["backlog"] == expected


@pytest.mark.parametrize("kind,label", [("review", "Review"), ("requirement", "Requirement")])
def test_childless_kinds_stop_after_set_meta(kind, label):
    stmt = store.REPLACE[kind]
    assert f"CREATE (s:Source:{label})\nSET s = $meta\nRETURN count(s) AS written" in stmt
    assert "UNWIND" not in stmt and "HAS_" not in stmt


def test_delete_statement_is_the_notes():
    assert store.DELETE_SOURCE == (
        "MATCH (s:Source {path:$path}) OPTIONAL MATCH (s)-[]->(c) "
        "WITH s, collect(c) AS cs FOREACH (c IN cs | DETACH DELETE c) DELETE s")


def test_schema_is_note_4_2():
    assert set(store.UNIQUE) == {("Source", "path"), ("Unit", "key"), ("Item", "ref"),
                                 ("KbFile", "path"), ("KbSection", "key")}
    assert set(store.RANGE_ONLY) == {("Ledger", "slug"), ("Backlog", "scope"), ("Unit", "statusToken"),
                                     ("Source", "itemId"), ("Unit", "itemId")}
    assert ("Unit", "unitLabel") not in store.INDEXES  # derived, looked up after the ledger anchor


# --- write boundary ----------------------------------------------------------------------------

def test_nulls_are_omitted_never_written_as_empty():
    meta = store.prepare_meta("review", {"path": "a/docs/reviews/x.md", "docStatus": None,
                                                 "warnings": [], "rowCount": 0})
    assert "docStatus" not in meta
    rows = store.prepare_rows("ledger", [{"key": "p#U1", "statusToken": "unknown", "statusText": None}])
    assert "statusText" not in rows[0]


def test_store_does_not_default_or_patch_warnings_and_rowcount():
    meta = store.prepare_meta("requirement", {"path": "a/docs/requirements/x.md"})
    assert "warnings" not in meta and "rowCount" not in meta


def test_indexed_values_are_normalised_at_the_write_boundary():
    meta = store.prepare_meta("ledger", {"path": "p.md", "slug": " s ", "itemId": "  GAL-F1\t"})
    assert (meta["slug"], meta["itemId"]) == ("s", "GAL-F1")
    assert store.prepare_meta("backlog", {"path": "p.md", "scope": " cpg\n"})["scope"] == "cpg"
    row = store.prepare_rows("ledger", [{"key": "p#U1", "statusToken": " paused ", "itemId": " T1 "}])[0]
    assert (row["statusToken"], row["itemId"]) == ("paused", "T1")


@pytest.mark.parametrize("bad", ["set ", " a", "a\t", "a\n", "a\r", "a\v", "a\f", "x" * 513, ""])
def test_unique_values_are_refused_not_rewritten(bad):
    with pytest.raises(KeyHygieneError):
        store.prepare_meta("review", {"path": bad})
    with pytest.raises(KeyHygieneError):
        store.prepare_rows("ledger", [{"key": bad}])
    with pytest.raises(KeyHygieneError):
        store.prepare_rows("backlog", [{"ref": bad}])


def test_a_crashing_key_never_reaches_the_graph():
    g = FakeGraph()
    with pytest.raises(KeyHygieneError):
        store.replace_source(g, "ledger", {"path": "p.md"}, [{"key": "p.md#U1 "}])
    assert g.calls == []


def test_replace_adds_only_synced_at_and_optional_mtime():
    g = FakeGraph()
    store.replace_source(g, "review", {"path": "p.md", "warnings": [], "rowCount": 0}, synced_at_ms=42, mtime_ns=7)
    _, params = g.calls[0]
    assert params["path"] == "p.md"
    assert params["meta"] == {"path": "p.md", "warnings": [], "rowCount": 0, "syncedAt": 42, "mtimeNs": 7}


def test_children_for_a_childless_kind_are_refused():
    with pytest.raises(store.StoreError):
        store.replace_source(FakeGraph(), "review", {"path": "p.md"}, [{"a": 1}])


def test_written_count_zero_when_the_row_is_collapsed():
    g = FakeGraph({"OPTIONAL MATCH": Result([])})
    assert store.replace_source(g, "ledger", {"path": "p.md"}, []) == 0


def test_item_conflict_names_both_paths_with_the_hint():
    err = RuntimeError("unique constraint violation on node of type Item")
    g = FakeGraph({"OPTIONAL MATCH": err, "MATCH (i:Item)": Result([["GAL-F0042", "a/docs/BACKLOG.md"]])})
    with pytest.raises(store.StoreError) as exc:
        store.replace_source(g, "backlog", {"path": "b/docs/BACKLOG.md"},
                             [{"ref": "GAL-F0042", "ord": 3}])
    msg = str(exc.value)
    assert "GAL-F0042" in msg and "a/docs/BACKLOG.md" in msg and "b/docs/BACKLOG.md" in msg
    assert "wi move may be in progress" in msg


def test_other_failures_say_the_previous_state_was_kept():
    g = FakeGraph({"OPTIONAL MATCH": RuntimeError("unique constraint violation on node of type Unit")})
    with pytest.raises(store.StoreError, match="previous state kept"):
        store.replace_source(g, "ledger", {"path": "p.md"}, [{"key": "p.md#U1"}])


# --- the endpoint guard -----------------------------------------------------------------------

@pytest.mark.parametrize("host,port,env", [
    ("localhost", 6379, {}),
    ("localhost", 6379, {"COORDGRAPH_EPHEMERAL_PORT": "6379"}),  # the shared default is never allowed
    ("localhost", 6390, {"COORDGRAPH_EPHEMERAL_PORT": "6390"}),
    ("127.0.0.1", 45001, {}),  # a free port without the declaration
    ("127.0.0.1", 45001, {"COORDGRAPH_EPHEMERAL_PORT": "45002"}),
    ("10.0.0.5", 45001, {"COORDGRAPH_EPHEMERAL_PORT": "45001"}),
])
def test_guard_refuses(host, port, env):
    with pytest.raises(store.RefusedEndpoint):
        store.require_ephemeral(host, port, env)


def test_guard_allows_the_declared_ephemeral_port():
    store.require_ephemeral("127.0.0.1", 45001, {"COORDGRAPH_EPHEMERAL_PORT": "45001"})


def test_bootstrap_command_refuses_the_default_port_without_connecting(monkeypatch, capsys):
    import socket

    def boom(*a, **k):
        raise AssertionError("a connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.delenv("COORDGRAPH_EPHEMERAL_PORT", raising=False)
    monkeypatch.delenv("COORDGRAPH_FALKORDB_PORT", raising=False)
    monkeypatch.delenv("COORDGRAPH_FALKORDB_HOST", raising=False)
    assert commands_store.bootstrap(type("A", (), {})()) == 1
    assert "refusing" in capsys.readouterr().err


# --- bootstrap ---------------------------------------------------------------------------------

def _schema_graph(existing_idx=(), cons_status=("OPERATIONAL",), constraints_listed=True):
    created_cons: list[tuple[str, str]] = []

    def indexes(q, p):
        return Result([[lbl, [prop]] for lbl, prop in existing_idx])

    def constraints(q, p):
        if not constraints_listed:
            return Result([])
        return Result([[lbl, [prop], cons_status[0]] for lbl, prop in created_cons])

    return FakeGraph({"CALL db.indexes": indexes, "CALL db.constraints": constraints}), created_cons


class RecordingDb(FakeDb):
    def __init__(self, created):
        super().__init__()
        self.created = created

    def execute_command(self, *args):
        self.created.append((args[5], args[8]))
        return super().execute_command(*args)


def test_bootstrap_creates_every_index_before_any_constraint():
    g, created = _schema_graph()
    db = RecordingDb(created)
    order: list[str] = []
    real_query = g.query

    def spy(q, params=None):
        if q.startswith("CREATE INDEX"):
            order.append("index")
        return real_query(q, params)

    g.query = spy
    orig = db.execute_command
    db.execute_command = lambda *a: (order.append("constraint"), orig(*a))[1]
    result = store.bootstrap(db, g, "k")
    assert order == ["index"] * len(store.INDEXES) + ["constraint"] * len(store.UNIQUE)
    assert set(result.created_constraints) == set(store.UNIQUE)
    assert all(a[:4] == ("GRAPH.CONSTRAINT", "CREATE", "k", "UNIQUE") and a[4] == "NODE" for a in db.commands)
    assert g.calls[-1][0].startswith("MERGE (m:SyncState {id:'mirror'})")
    assert g.calls[-1][1] == {"v": store.SCHEMA_VERSION}


def test_bootstrap_is_read_then_create_second_run_changes_nothing():
    g, created = _schema_graph(existing_idx=store.INDEXES)
    created.extend(store.UNIQUE)  # already present
    db = FakeDb()
    result = store.bootstrap(db, g, "k")
    assert not result.changed and db.commands == []
    assert not any(c[0].startswith("CREATE INDEX") for c in g.calls)


def test_bootstrap_fails_loudly_on_a_failed_constraint():
    g, created = _schema_graph(cons_status=("FAILED",))
    with pytest.raises(store.StoreError, match="FAILED"):
        store.bootstrap(RecordingDb(created), g, "k", timeout=1)


def test_bootstrap_does_not_read_an_empty_constraint_listing_as_operational():
    g, created = _schema_graph(constraints_listed=False)
    with pytest.raises(store.StoreError, match="not OPERATIONAL"):
        store.bootstrap(RecordingDb(created), g, "k", timeout=0.2)


def test_bootstrap_waits_while_pending():
    g, created = _schema_graph(cons_status=("PENDING",))
    with pytest.raises(store.StoreError, match="PENDING"):
        store.bootstrap(RecordingDb(created), g, "k", timeout=0.2)


# --- convergence loop --------------------------------------------------------------------------

def test_write_converged_rewrites_when_the_file_moved():
    g = FakeGraph()
    versions = iter(["v1", "v2"])
    state = {"disk": "v1"}

    def parse_current():
        sha = next(versions)
        state["disk"] = "v2"  # the file changes while we write
        return "review", {"path": "p.md", "sha256": sha}, []

    rounds = store.write_converged(g, parse_current, lambda: state["disk"])
    assert rounds == 2 and len(g.calls) == 2


def test_write_converged_gives_up_after_the_bound():
    g = FakeGraph()
    n = iter(range(10))
    with pytest.raises(store.ConvergenceError):
        store.write_converged(g, lambda: ("review", {"path": "p.md", "sha256": f"s{next(n)}"}, []),
                              lambda: "never", rounds=3)
    assert len(g.calls) == 3


def test_offline_modules_never_import_falkordb():
    import subprocess

    code = ("import sys, coordgraph.dryrun, coordgraph.store, coordgraph.commands_store; "
            "sys.exit('falkordb' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
