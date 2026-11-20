"""Live tests of `sync`, `rebuild`, `drift` and `query` through the CLI, on the EPHEMERAL container only.

Covers plan section 5 AC-2, AC-4, AC-6, AC-7 and the S3 leg of plan3 section 6 test 13. Each test
gets its own graph key and its own tmp_path tree; the CLI is reached with the environment of
`tests/fixtures/live.py` (127.0.0.1, a free non-shared port, `COORDGRAPH_EPHEMERAL_PORT`). The shared
`falkordb-dev` is never contacted. Run with `-m live`.
"""
import json
import uuid
from pathlib import Path

import pytest

from coordgraph import store, sync
from coordgraph.cli import main
from coordgraph.identity import LedgerRef, UnitRow, resolve_need
from tests.fixtures.live import SHARED_PORTS, falkordb_container  # noqa: F401  (session fixture)
from tests.test_identity import identity, new

pytestmark = pytest.mark.live

PROJECT = {"key": "GAL", "legacy_paths": {"*/docs/BACKLOG.md": "$1", "docs/BACKLOG.md": "cpg"}}
LEDGER = """# L

> **Status:** active · **Owner:** `teco` · **Tracks:** —

| Unit | Owner | Status | Deliverable |
|---|---|---|---|
| U1 first | coder | **accepted** | x |
| U2 | coder | **queued** | y |
| U3 | coder | **relayed** | "why?" — relayed 2026-11-01 |
| U32 | coder | **queued** | z |
| U32 | coder | **in-flight** | z2 |
"""
BACKLOG = "# B\n\n### K-001 — One\n\n- **Status:** 🔵 proposed\n\n### K-002 — Two\n\n- **Status:** 🔵 proposed\n"
REVIEW = "# R\n\n> **Status:** active · **Owner:** `analyst` · **Tracks:** —\n\n## Pass 1\n\nVerdict: needs changes\n"
REQ = "# Q\n\n> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** —\n\ntext\n"
L = "a/docs/plans/x-coordination.md"
B = "a/docs/BACKLOG.md"
R = "a/docs/reviews/x.md"
Q = "a/docs/requirements/x.md"


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def env(falkordb_container, tmp_path, monkeypatch):  # noqa: F811
    """A tmp repo, a fresh graph key on the ephemeral container, and the CLI pointed at both."""
    c = falkordb_container
    assert c.host == "127.0.0.1" and c.port not in SHARED_PORTS and c.name.startswith("coordgraph-test-")
    put(tmp_path, ".work-items/project.json", json.dumps(PROJECT))
    put(tmp_path, L, LEDGER)
    put(tmp_path, B, BACKLOG)
    put(tmp_path, R, REVIEW)
    put(tmp_path, Q, REQ)
    key = f"t-{uuid.uuid4().hex[:10]}"
    for name, value in (("COORDGRAPH_REPO_ROOT", str(tmp_path)), ("COORDGRAPH_FALKORDB_HOST", c.host),
                        ("COORDGRAPH_FALKORDB_PORT", str(c.port)), ("COORDGRAPH_EPHEMERAL_PORT", str(c.port)),
                        ("COORDGRAPH_KEY", key), ("COORDGRAPH_STATE_DIR", str(tmp_path / ".state"))):
        monkeypatch.setenv(name, value)
    return tmp_path, key, c


def graph_of(env_, key=None):
    root, k, c = env_
    return store.connect(c.host, c.port, key or k, {"COORDGRAPH_EPHEMERAL_PORT": str(c.port)})


def run(capsys, *argv) -> tuple[int, str]:
    capsys.readouterr()
    code = main(list(argv))
    return code, capsys.readouterr().out


def rows(graph, q, params=None):
    return graph.query(q, params or {}).result_set


@pytest.fixture
def synced(env, capsys):
    assert run(capsys, "bootstrap")[0] == 0
    code, out = run(capsys, "sync", "--all")
    assert code == 0, out
    return env


def parser(env_) -> sync.Parser:
    return sync.Parser.for_root(env_[0])


# --- sync -----------------------------------------------------------------------------------------------

def test_sync_of_each_kind_equals_the_parse_and_an_unchanged_resync_writes_nothing(synced, capsys):
    db, graph = graph_of(synced)
    assert sync.parity(graph, parser(synced)).ok
    before = dict(rows(graph, "MATCH (s:Source) RETURN s.path, s.syncedAt"))
    assert len(before) == 4
    code, out = run(capsys, "sync", "--all")
    assert code == 0 and out == ""  # every path unchanged, nothing listed
    assert dict(rows(graph, "MATCH (s:Source) RETURN s.path, s.syncedAt")) == before


def test_sync_of_a_named_path_replaces_only_that_source(synced, capsys):
    root = synced[0]
    _, graph = graph_of(synced)
    before = dict(rows(graph, "MATCH (s:Source) RETURN s.path, s.syncedAt"))
    put(root, B, BACKLOG + "\n### K-003 — Three\n\n- **Status:** 🔵 proposed\n")
    code, out = run(capsys, "sync", B)
    assert code == 0 and out.strip() == f"synced\t{B}"
    after = dict(rows(graph, "MATCH (s:Source) RETURN s.path, s.syncedAt"))
    assert [p for p in after if after[p] != before[p]] == [B]
    assert rows(graph, "MATCH (:Source {path:$p})-[:HAS_ITEM]->(i:Item) RETURN count(i)", {"p": B}) == [[3]]


def test_an_unparseable_edit_keeps_the_previous_subgraph_and_exits_nonzero(synced, capsys):
    root = synced[0]
    _, graph = graph_of(synced)
    put(root, B, BACKLOG + "\n### K-001 — Dup\n\n- **Status:** 🔵 proposed\n")
    code, out = run(capsys, "sync", B)
    assert code == 1 and out.startswith(f"failed\t{B}")
    assert rows(graph, "MATCH (:Source {path:$p})-[:HAS_ITEM]->(i:Item) RETURN count(i)", {"p": B}) == [[2]]


def test_a_nul_byte_in_a_cell_no_longer_leaves_the_source_unmirrored(synced, capsys):
    root = synced[0]
    _, graph = graph_of(synced)
    put(root, L, LEDGER.replace("| y |", "| y\x00z |"))
    code, out = run(capsys, "sync", L)
    assert code == 0 and out.strip() == f"synced\t{L}"
    assert rows(graph, "MATCH (u:Unit {unitKey:'U2'}) RETURN u.deliverable") == [["y�z"]]
    (warnings,) = rows(graph, "MATCH (s:Source {path:$p}) RETURN s.warnings", {"p": L})[0]
    assert any("NUL" in w for w in warnings)


# --- AC-2: rebuild and parity ------------------------------------------------------------------------------

def test_ac2_sync_all_and_rebuild_into_an_empty_key_both_pass_parity_and_a_mutation_is_caught(synced, capsys):
    p = parser(synced)
    _, graph_a = graph_of(synced)
    assert sync.parity(graph_a, p).ok
    scratch = f"t-{uuid.uuid4().hex[:10]}"
    code, out = run(capsys, "rebuild", "--graph", scratch)
    assert code == 0 and out.rstrip().endswith("parity: ok"), out
    _, graph_b = graph_of(synced, scratch)
    assert sync.parity(graph_b, p).ok
    # same shape in both keys
    count = "MATCH (n) RETURN labels(n)[0] AS l, count(n) ORDER BY l"
    assert [r for r in rows(graph_a, count) if r[0] != "SyncState"] == [r for r in rows(graph_b, count)
                                                                       if r[0] != "SyncState"]
    # a one-property mutation in the scratch key is reported by exactly that key
    graph_b.query("MATCH (u:Unit {unitKey:'U2'}) SET u.statusToken = 'paused'")
    bad = sync.parity(graph_b, p)
    assert bad.diffs == [f"{L}: child '{L}#U2' differs"]
    # and `rebuild` repairs it without GRAPH.DELETE
    code, out = run(capsys, "rebuild", "--graph", scratch)
    assert code == 0 and sync.parity(graph_b, p).ok


def test_rebuild_twice_is_idempotent_and_prunes_a_source_that_left_the_disk(synced, capsys):
    root = synced[0]
    _, graph = graph_of(synced)
    count = "MATCH (n) RETURN count(n)"
    assert run(capsys, "rebuild")[0] == 0
    n = rows(graph, count)
    assert run(capsys, "rebuild")[0] == 0 and rows(graph, count) == n
    (root / R).unlink()
    code, out = run(capsys, "rebuild")
    assert code == 0 and f"deleted\t{R}" in out
    assert rows(graph, "MATCH (s:Source {path:$p}) RETURN count(s)", {"p": R}) == [[0]]
    assert sync.parity(graph, parser(synced)).ok


def test_rebuild_with_a_parentless_unit_exits_1_names_the_orphans_and_deletes_them(synced, capsys):
    _, graph = graph_of(synced)
    graph.query("CREATE (:Unit {key:'orphan#U1', statusToken:'queued'}), (:Item {ref:'x:K-9'})")
    code, out = run(capsys, "rebuild")
    assert code == 1 and "ORPHANS Unit: 1 deleted (a bug signal)" in out and "ORPHANS Item: 1 deleted" in out
    assert rows(graph, "MATCH (n) WHERE n:Unit AND n.key = 'orphan#U1' OR n:Item AND n.ref = 'x:K-9' RETURN count(n)") == [[0]]
    code, out = run(capsys, "rebuild")
    assert code == 0 and "ORPHANS" not in out


def test_parity_catches_every_planted_identity_mutant_live(synced):
    p = parser(synced)
    _, graph = graph_of(synced)
    mutants = [
        ("MATCH (s:Source {path:$p}) SET s.warnings = ['x']", L),
        ("MATCH (s:Source {path:$p}) REMOVE s.warnings", R),
        ("MATCH (s:Source {path:$p}) SET s.slug = 'x-graph'", Q),
        ("MATCH (:Source {path:$p})-[:HAS_UNIT]->(u:Unit {unitKey:'U1 first'}) SET u.unitLabel = u.unitKey", L),
        ("MATCH (:Source {path:$p})-[:HAS_UNIT]->(u:Unit {unitKey:'U2'}) DETACH DELETE u", L),
        ("MATCH (:Source {path:$p})-[:HAS_ITEM]->(i:Item {title:'One'}) SET i.ref = 'a:K-9'", B),
    ]
    for cypher, path in mutants:
        assert {o.status for o in sync.sync_all(graph, p, force=True)} == {"synced"}  # restore every source
        g = graph
        assert sync.parity(g, p).ok
        g.query(cypher, {"p": path})
        got = sync.parity(g, p)
        assert not got.ok and all(d.startswith(f"{path}:") for d in got.diffs), (cypher, got.diffs)


def test_parity_catches_a_source_relabelled_and_a_child_relinked_live(synced):
    p = parser(synced)
    _, graph = graph_of(synced)
    graph.query("MATCH (s:Source {path:$p}) REMOVE s:Backlog SET s:Ledger", {"p": B})
    got = sync.parity(graph, p)
    assert got.diffs == [f"{B}: labels ['Ledger', 'Source'], expected ['Backlog', 'Source']"]
    assert {o.status for o in sync.sync_all(graph, p, force=True)} == {"synced"}
    assert sync.parity(graph, p).ok
    graph.query("MATCH (s:Source {path:$p})-[r:HAS_UNIT]->(u:Unit {unitKey:'U2'}) "
                "CREATE (s)-[:HAS_ITEM]->(u) DELETE r", {"p": L})
    got = sync.parity(graph, p)
    assert got.diffs == [f"{L}: child '{L}#U2' is HAS_ITEM ['Unit'], expected HAS_UNIT ['Unit']"]


# --- AC-4: drift and query -----------------------------------------------------------------------------------

def test_ac4_a_hand_edit_without_the_hook_is_flagged_row_by_row_and_a_deleted_file_is_gone(synced, capsys):
    root = synced[0]
    code, out = run(capsys, "drift")
    assert code == 0 and "Drift: none" in out and "(4 sources)" in out and "Hook: active" in out
    put(root, L, LEDGER.replace("| U2 | coder | **queued** |", "| U2 | coder | **accepted** |"))
    (root / R).unlink()
    code, out = run(capsys, "drift")
    assert code == 1
    lines = out.splitlines()
    assert f"  changed\t{L}\t→ file wins; run coordgraph sync" in lines
    assert f"    ⚠ drift\t{L}#U2\tchanged" in lines
    assert f"  gone\t{R}\t→ file wins; run coordgraph sync" in lines
    assert sum("⚠ drift" in line for line in lines) == 1
    code, out = run(capsys, "drift", B)
    assert code == 0 and "Drift: none" in out
    assert run(capsys, "sync", "--all")[0] == 0
    assert "Drift: none" in run(capsys, "drift")[1]


def test_drift_flags_a_stale_parser_version_live(synced, capsys):
    _, graph = graph_of(synced)
    graph.query("MATCH (s:Source {path:$p}) SET s.parserVersion = 0", {"p": R})
    code, out = run(capsys, "drift")
    assert code == 1 and f"stale-parser\t{R}" in out and "parserVersion 0, current" in out
    assert run(capsys, "sync", "--all")[0] == 0 and run(capsys, "drift")[0] == 0


def test_drift_reports_the_disabled_hook_and_the_registry_line(synced, capsys):
    root = synced[0]
    put(root, ".state/disabled", "")
    out = run(capsys, "drift")[1]
    assert "Hook: disabled" in out and out.splitlines()[1].startswith("Registry:")


def test_query_prints_the_header_flags_drift_and_is_read_only(synced, capsys):
    root = synced[0]
    _, graph = graph_of(synced)
    code, out = run(capsys, "query", "MATCH (u:Unit) RETURN count(u) AS n")
    assert code == 0 and "Drift: none" in out and out.splitlines()[-2:] == ["n", "5"]
    put(root, L, LEDGER + "| U9 | coder | **queued** | n |\n")
    out = run(capsys, "query", "MATCH (u:Unit) RETURN count(u) AS n", "--covers", L)[1]
    assert f"  changed\t{L}" in out and f"⚠ drift\t{L}#U9\tadded in file" in out
    assert "Drift: none" in run(capsys, "query", "RETURN 1 AS one", "--covers", B)[1]
    code, _ = run(capsys, "query", "CREATE (:Planted)")
    assert code == 1 and rows(graph, "MATCH (n:Planted) RETURN count(n)") == [[0]]


# --- AC-6: the single-unit lookup rows ------------------------------------------------------------------------

UNIT_73 = """MATCH (l:Ledger {slug: $slug})-[:HAS_UNIT]->(u:Unit)
WHERE u.unitKey = $unit OR u.unitLabel = $unit
RETURN l.path AS path, u.key AS key, u.statusToken AS status, u.statusText AS statusText, u.askedOn AS askedOn
ORDER BY path, u.tableIdx, u.rowIdx"""


def test_ac6_the_note_7_3_lookup_returns_the_row_each_repeat_and_a_paused_relayed_row(synced):
    _, graph = graph_of(synced)
    one = rows(graph, UNIT_73, {"slug": "x", "unit": "U3"})
    assert [(r[1], r[2]) for r in one] == [(f"{L}#U3", "paused")] and one[0][4] == "2026-11-01"
    repeats = rows(graph, UNIT_73, {"slug": "x", "unit": "U32"})
    assert [r[1] for r in repeats] == [f"{L}#U32", f"{L}#U32~2"]
    assert [r[2] for r in repeats] == ["queued", "in-flight"]


# --- AC-7 and plan3 test 13: archived flip, joins -------------------------------------------------------------

def test_ac7_an_archived_flip_keeps_minimal_units_that_no_live_view_selects(synced, capsys):
    root = synced[0]
    _, graph = graph_of(synced)
    live = "MATCH (s:Source:Ledger)-[:HAS_UNIT]->(u:Unit) WHERE s.docStatus <> 'archived' RETURN count(u)"
    assert rows(graph, live) == [[5]]
    put(root, L, LEDGER.replace("**Status:** active", "**Status:** archived"))
    code, out = run(capsys, "sync", L)
    assert code == 0, out
    assert rows(graph, live) == [[0]]
    units = rows(graph, "MATCH (:Source {path:$p})-[:HAS_UNIT]->(u:Unit) RETURN properties(u)", {"p": L})
    assert len(units) == 5
    for (props,) in units:
        assert not {"statusText", "rowHash", "unit", "cites", "owner", "deliverable"} & set(props), props
        assert {"key", "unitKey", "unitLabel", "statusToken"} <= set(props)
    assert sync.parity(graph, parser(synced)).ok
    # and back: a re-activated ledger regains its full units
    put(root, L, LEDGER)
    assert run(capsys, "sync", L)[0] == 0
    assert rows(graph, live) == [[5]]


def test_item_id_lookups_by_candidate_list_use_the_index(synced):
    _, graph = graph_of(synced)
    plan = graph.explain("MATCH (u:Unit) WHERE u.itemId IN $c RETURN u", {"c": ["GAL-I02-K0001", "x"]})
    text = str(plan)
    assert "Index" in text and "All Node Scan" not in text, text


def test_a_ledger_header_item_written_under_an_old_id_joins_after_a_move(synced, capsys):
    """Plan3 test 13: `Source.itemId IN candidates` finds the ledger by its new ID and by its old alias."""
    root = synced[0]
    _, graph = graph_of(synced)
    text = LEDGER.replace("**Tracks:** —", "**Tracks:** —\n").replace(
        "> **Status:** active", "> **Status:** active · **Item:** `GAL-F0002`", 1)
    put(root, L, text.replace("**Tracks:** —\n", "**Tracks:** —"))
    assert run(capsys, "sync", L)[0] == 0
    idn = identity(new("GAL-I02", "I"), new("GAL-F0002", "F"))
    cands = sorted(idn.resolve("GAL-F0002", "a").candidates)
    q = "MATCH (s:Source:Ledger) WHERE s.itemId IN $c RETURN s.path"
    assert rows(graph, q, {"c": cands}) == [[L]]
    assert rows(graph, q, {"c": ["GAL-F0099"]}) == []


class LiveUnits:
    """The slice of graph note 7.3/7.6 `resolve_need` reads, answered from the live mirror."""

    def __init__(self, graph):
        self.graph = graph

    def _rows(self, result):
        return [UnitRow(key=r[0], ledger_path=r[1], slug=r[2], unit_key=r[3], unit_label=r[4], status=r[5],
                        item_id=r[6], status_text=r[7]) for r in result]

    COLS = "u.key, u.ledgerPath, u.slug, u.unitKey, u.unitLabel, u.statusToken, u.itemId, u.statusText"

    def units_with_item(self, cands):
        q = f"MATCH (u:Unit) WHERE u.itemId IN $c RETURN {self.COLS}"
        return self._rows(rows(self.graph, q, {"c": sorted(cands)}))

    def units_matching(self, slug, unit):
        return self._rows(rows(self.graph, "MATCH (l:Ledger {slug:$s})-[:HAS_UNIT]->(u:Unit) "
                               f"WHERE u.unitKey=$u OR u.unitLabel=$u RETURN {self.COLS}", {"s": slug, "u": unit}))

    def units_in_ledger(self, path, unit):
        return self._rows(rows(self.graph, "MATCH (l:Source {path:$p})-[:HAS_UNIT]->(u:Unit) "
                               f"WHERE u.unitKey=$u OR u.unitLabel=$u RETURN {self.COLS}", {"p": path, "u": unit}))

    def ledgers_carrying(self, cands):
        return []


def test_a_task_need_on_an_archived_unit_whose_cell_reads_blocked_is_unresolvable_status_unknown(synced, capsys):
    root = synced[0]
    _, graph = graph_of(synced)
    put(root, L, """# L

> **Status:** archived · **Owner:** `teco` · **Tracks:** —

| Unit | Item | Owner | Status | Deliverable |
|---|---|---|---|---|
| U1 | GAL-F0002-T07 | coder | **blocked (waiting on K-9)** | x |
| U2 | GAL-F0002-T08 | coder | **accepted** | y |
""")
    assert run(capsys, "sync", L)[0] == 0
    (u,) = rows(graph, "MATCH (u:Unit {itemId:'GAL-F0002-T07'}) RETURN u.statusToken, u.statusText")
    assert u == ["unknown", None]
    idn = identity(new("GAL-F0002", "F"), new("GAL-F0002-T07", "T", parent="GAL-F0002"))
    res = resolve_need("GAL-F0002-T07", LedgerRef("a/docs/plans/y-coordination.md", "y"), idn, LiveUnits(graph))
    assert res.state == "unresolvable" and "status unknown" in res.reason
