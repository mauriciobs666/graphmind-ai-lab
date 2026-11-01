"""S17 (P0-14): ``ccg/store/ddl_v1.cypher`` against the live ``ccg-falkordb`` (127.0.0.1:6390).

Spec: ``docs/plans/connected-code-graph-p0-graph.md`` section 1.6 (the table below is its transcription).
Pure tests (parser, static order) run everywhere; the live test is ``-m live`` and skips, with the
reason, when the instance or its credential is missing. Touches only a fresh
``ccg_p0slice__build_ddl<N>`` key, deleted at the end (R-B)."""
import os
import re
import time
from pathlib import Path

import pytest

from ccg.store import DDL_V1, parse_ddl

HOST, PORT = "127.0.0.1", 6390
POLL_TIMEOUT_S = 60

PLAN = Path(__file__).resolve().parents[2] / "docs" / "plans" / "connected-code-graph-p0-graph.md"


def parse_plan_table(text):
    """Section 1.6's table -> (index pairs, constraint pairs). Drift between plan and DDL is red."""
    sec = text.split("### 1.6", 1)[1].split("\n###", 1)[0].split("\n## ", 1)[0]
    rows = [l for l in sec.splitlines() if l.startswith("|")][2:]  # header + separator
    idx, con = set(), set()
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        labels = re.findall(r"`(\w+)`", cells[0])
        for label in labels:
            idx |= {(label, p) for p in re.findall(r"`(\w+)`", cells[1])}
            con |= {(label, p) for p in re.findall(r"`(\w+)`", cells[2])}
    return idx, con


EXPECTED_INDEX_PAIRS, EXPECTED_CONSTRAINT_PAIRS = parse_plan_table(PLAN.read_text())
EXPECTED_CONSTRAINTS = dict(EXPECTED_CONSTRAINT_PAIRS)


def static_problems(stmts):
    """Everything checkable without a database; the live test reuses it so a mutant dies in both."""
    idx = [(s[1], s[2]) for s in stmts if s[0] == "index"]
    con = [(s[1], s[2]) for s in stmts if s[0] == "constraint"]
    problems = []
    if len(idx) != len(set(idx)):
        problems.append("duplicate index statement")
    if set(idx) != EXPECTED_INDEX_PAIRS:
        problems.append(f"index set differs: missing {sorted(EXPECTED_INDEX_PAIRS - set(idx))}, extra {sorted(set(idx) - EXPECTED_INDEX_PAIRS)}")
    if len(con) != len(set(con)) or set(con) != EXPECTED_CONSTRAINT_PAIRS:
        problems.append(f"constraint set differs: missing {sorted(EXPECTED_CONSTRAINT_PAIRS - set(con))}, extra {sorted(set(con) - EXPECTED_CONSTRAINT_PAIRS)}")
    kinds = [s[0] for s in stmts]
    if "constraint" in kinds and "index" in kinds[kinds.index("constraint"):]:
        problems.append("an index statement follows a constraint statement (indexes must all come first)")
    for i, s in enumerate(stmts):
        if s[0] == "constraint" and (s[1], s[2]) not in {(t[1], t[2]) for t in stmts[:i] if t[0] == "index"}:
            problems.append(f"constraint {s[1]}.{s[2]} has no earlier index")
    return problems


# --------------------------------------------------------------------------------------- pure

def test_ddl_file_matches_section_1_6_statically():
    assert static_problems(parse_ddl()) == []
    stmts = parse_ddl()
    assert sum(s[0] == "index" for s in stmts) == 17 and sum(s[0] == "constraint" for s in stmts) == 10


def test_ddl_header_documents_order_and_apply_protocol():
    head = DDL_V1.read_text().split("// ---- PART 1")[0]
    for needle in ("index-before-constraint", "EMPTY", "OPERATIONAL", "ASYNCHRONOUS", "TIMEOUT_MAX"):
        assert needle in head


def test_parser_rejects_unknown_statements():
    with pytest.raises(ValueError):
        parse_ddl("CREATE INDEX FOR (n:Repo) ON (n.repoId, n.x);")


def test_static_check_catches_each_mutation():
    base = parse_ddl()
    drop_idx = [s for s in base if s[:3] != ("index", "Param", "pointId")]
    extra_idx = base[:1] + [("index", "Repo", "bogus", "CREATE INDEX FOR (n:Repo) ON (n.bogus)")] + base[1:]
    drop_con = [s for s in base if s[:3] != ("constraint", "Link", "linkId")]
    swapped = [s for s in base if s[0] == "constraint"][:1] + [s for s in base if s[0] == "index"] + [s for s in base if s[0] == "constraint"][1:]
    for mutant in (drop_idx, extra_idx, drop_con, swapped):
        assert static_problems(mutant)


# --------------------------------------------------------------------------------------- live

def _connect():
    from falkordb import FalkorDB
    pw = (Path(os.environ.get("CCG_CONFIG_DIR", Path.home() / ".config" / "ccg")) / "falkordb-admin.pass").read_text().strip()
    db = FalkorDB(host=HOST, port=PORT, password=pw, socket_timeout=120, socket_connect_timeout=10, retry=None)
    db.connection.ping()
    return db


@pytest.fixture(scope="module")
def db():
    try:
        return _connect()
    except FileNotFoundError as exc:
        pytest.skip(f"ccg-falkordb admin password file not readable ({exc.filename})")
    except Exception as exc:  # noqa: BLE001 -- any connection failure is "unreachable"
        pytest.skip(f"ccg-falkordb unreachable at {HOST}:{PORT} ({type(exc).__name__}); start it with cpg/infra/ccg-falkordb.sh up")


def fresh_key(db):
    n = 1
    while db.connection.exists(f"ccg_p0slice__build_ddl{n}"):
        n += 1
    return f"ccg_p0slice__build_ddl{n}"


def apply_ddl(db, key, stmts, *, between=None, timeout_max=0):
    """The apply protocol: indexes, ``between`` (the load), constraints, bounded poll to OPERATIONAL."""
    assert re.fullmatch(r"ccg_p0slice__build_ddl\d+", key)
    # file order is part of the contract (set differences are left to the live comparison / the engine)
    order = [p for p in static_problems(stmts) if "follows a constraint" in p]
    assert not order, order
    g = db.select_graph(key)
    q = (lambda c, p=None: g.query(c, p, timeout_max)) if timeout_max else (lambda c, p=None: g.query(c, p))
    for s in stmts:
        if s[0] == "index":
            q(s[3])
    if between:
        between(q)
    n = 0
    for s in stmts:
        if s[0] == "constraint":
            # raw command: falkordb-py's create_node_unique_constraint silently creates any missing
            # range index first, which would mask a DDL that dropped one
            db.connection.execute_command("GRAPH.CONSTRAINT", "CREATE", key, "UNIQUE", "NODE", s[1], "PROPERTIES", 1, s[2])
            n += 1
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        cons = [(r[1], r[2], r[4]) for r in q("CALL db.constraints()").result_set]
        assert not [c for c in cons if c[2] == "FAILED"], cons
        if len(cons) == n and all(c[2] == "OPERATIONAL" for c in cons):
            return q, cons
        assert time.monotonic() < deadline, f"constraints not OPERATIONAL in {POLL_TIMEOUT_S}s: {cons}"
        time.sleep(0.2)




def run_against(db, stmts, *, load=True):
    """Apply ``stmts`` to a fresh key, check the section 1.6 contract, always delete the key."""
    key = fresh_key(db)
    g = db.select_graph(key)
    timeout_max = int(db.connection.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")[1])
    try:
        assert not db.connection.exists(key)  # EMPTY key
        loaded = {}

        def between(q):
            if not load:
                return
            for label, prop in EXPECTED_CONSTRAINTS.items():
                q(f"CREATE (:{label} {{{prop}: $v}})", {"v": f"{label}-1"})
                loaded[label] = prop

        q, cons = apply_ddl(db, key, stmts, between=between, timeout_max=timeout_max)
        rows = q("CALL db.indexes()").result_set
        got = []
        for r in rows:  # label, properties, types, options, language, stopwords, entitytype, status
            label, props, types, entity, status = r[0], r[1], r[2], r[6], r[7]
            assert entity == "NODE" and status == "OPERATIONAL", r
            assert all(types[p] == ["RANGE"] for p in props), r
            got += [(label, p) for p in props]
        assert len(got) == len(set(got)), f"duplicate live index entries: {got}"
        assert set(got) == EXPECTED_INDEX_PAIRS, (
            f"missing {sorted(EXPECTED_INDEX_PAIRS - set(got))}, extra {sorted(set(got) - EXPECTED_INDEX_PAIRS)}")
        assert all(len(c[1]) == 1 for c in cons) and {(c[0], c[1][0]) for c in cons} == EXPECTED_CONSTRAINT_PAIRS, (
            f"constraint set differs: {sorted((c[0], c[1][0]) for c in cons)}")
        # the constraint really enforces: a duplicate id is refused
        with pytest.raises(Exception):
            q("CREATE (:Link {linkId: $v})", {"v": "Link-1"})
        return got
    finally:
        assert re.fullmatch(r"ccg_p0slice__build_ddl\d+", key)
        if db.connection.exists(key):
            g.delete()


@pytest.mark.live
def test_ddl_applies_to_empty_key_indexes_match_table_and_constraints_reach_operational(db):
    assert len(run_against(db, parse_ddl())) == 17


def _mutant(name):
    base = parse_ddl()
    cons = [s for s in base if s[0] == "constraint"]
    idx = [s for s in base if s[0] == "index"]
    return {
        "drop_index": [s for s in base if s[:3] != ("index", "Param", "pointId")],
        "drop_unconstrained_index": [s for s in base if s[:3] != ("index", "SkippedFile", "repoId")],
        "extra_index": base[:1] + [("index", "Repo", "bogus", "CREATE INDEX FOR (n:Repo) ON (n.bogus)")] + base[1:],
        "drop_constraint": [s for s in base if s[:3] != ("constraint", "Link", "linkId")],
        "constraint_first": cons[:1] + idx + cons[1:],
    }[name]


#: mutant -> (exception type, message fragment): the failure must be the *intended* one
MUTANT_FAILURES = {
    "drop_index": ("ResponseError", "missing supporting exact-match index"),
    "drop_unconstrained_index": ("AssertionError", "missing [('SkippedFile', 'repoId')], extra []"),
    "extra_index": ("AssertionError", "missing [], extra [('Repo', 'bogus')]"),
    "drop_constraint": ("AssertionError", "constraint set differs"),
    "constraint_first": ("AssertionError", "follows a constraint statement"),
}


@pytest.mark.live
@pytest.mark.parametrize("mutant", sorted(MUTANT_FAILURES))
def test_live_check_fails_on_a_broken_ddl_for_the_intended_reason(db, mutant):
    exc_type, fragment = MUTANT_FAILURES[mutant]
    with pytest.raises(Exception) as info:
        run_against(db, _mutant(mutant))
    assert type(info.value).__name__ == exc_type, repr(info.value)
    assert fragment in str(info.value), str(info.value)


@pytest.mark.live
def test_engine_refuses_a_constraint_without_an_index(db):
    key = fresh_key(db)
    try:
        with pytest.raises(Exception, match="missing supporting exact-match index"):
            db.connection.execute_command("GRAPH.CONSTRAINT", "CREATE", key, "UNIQUE", "NODE", "Unit", "PROPERTIES", 1, "unitId")
    finally:
        if db.connection.exists(key):
            db.select_graph(key).delete()


@pytest.mark.live
def test_failed_constraint_state_fails_the_apply(db):
    """Two Links with one id loaded before the constraints: the Link constraint goes FAILED and apply_ddl raises."""
    key = fresh_key(db)

    def load_duplicates(q):
        q("CREATE (:Link {linkId: 'dup'}), (:Link {linkId: 'dup'})")

    try:
        with pytest.raises(AssertionError, match="FAILED"):
            apply_ddl(db, key, parse_ddl(), between=load_duplicates)
    finally:
        assert re.fullmatch(r"ccg_p0slice__build_ddl\d+", key)
        if db.connection.exists(key):
            db.select_graph(key).delete()


# ------------------------------------------------------------------- pure: TIMEOUT_MAX branch (D16)

class _Res:
    def __init__(self, rows):
        self.result_set = rows


class _RecGraph:
    def __init__(self, calls, n_constraints):
        self.calls, self.n = calls, n_constraints

    def query(self, cypher, params=None, timeout=None):
        self.calls.append((cypher, timeout))
        if cypher == "CALL db.constraints()":
            return _Res([["UNIQUE", l, [p], "NODE", "OPERATIONAL"] for l, p in sorted(EXPECTED_CONSTRAINT_PAIRS)])
        return _Res([])


class _RecDb:
    def __init__(self):
        self.calls, self.raw = [], []
        self.connection = self

    def select_graph(self, key):
        return _RecGraph(self.calls, 10)

    def execute_command(self, *args):
        self.raw.append(args)


@pytest.mark.parametrize("timeout_max", [0, 1000])
def test_every_statement_carries_the_timeout_when_timeout_max_is_nonzero(timeout_max):
    db = _RecDb()
    apply_ddl(db, "ccg_p0slice__build_ddl1", parse_ddl(), between=lambda q: q("CREATE (:Unit {unitId: 'u'})"), timeout_max=timeout_max)
    assert len(db.calls) == 17 + 1 + 1 and len(db.raw) == 10  # indexes, the load query, one poll; constraints are raw commands
    assert {t for _, t in db.calls} == ({1000} if timeout_max else {None})


# ------------------------------------------------------------------- pure: parser negatives (reject, never skip)

@pytest.mark.parametrize("text", [
    "CREATE INDEX FOR (n:Repo) ON (n.repoId); // trailing comment",   # inline comment
    "create index for (n:Repo) on (n.repoId);",                        # lower-case keywords
    "GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE File PROPERTIES 2 fileId path;",  # PROPERTIES 2
    "CREATE INDEX FOR (n:Repo) ON (n.repoId); junk",                   # junk after a statement
    "CREATE INDEX FOR (n:Repo) ON (n.repoId)",                         # final statement without ';'
    "CREATE INDEX FOR (n:Repo) ON (n.repoId);\nCREATE INDEX FOR (n:Unit) ON (n.unitId)",
], ids=["inline-comment", "lowercase", "properties-2", "junk-after", "no-final-semicolon", "second-lacks-semicolon"])
def test_parser_rejects(text):
    with pytest.raises(ValueError):
        parse_ddl(text)


def test_parser_accepts_whole_line_comments_and_blank_lines():
    assert parse_ddl("// c\n\nCREATE INDEX FOR (n:Repo) ON (n.repoId);\n// end\n") == [
        ("index", "Repo", "repoId", "CREATE INDEX FOR (n:Repo) ON (n.repoId)")]


def test_plan_table_parse_is_the_spec_not_a_transcription():
    assert len(EXPECTED_INDEX_PAIRS) == 17 and len(EXPECTED_CONSTRAINT_PAIRS) == 10
    assert ("SkippedFile", "repoId") in EXPECTED_INDEX_PAIRS and ("Cond", "pointId") in EXPECTED_CONSTRAINT_PAIRS
