"""Live tests of the read views (`inflight`, `unit`, `planned`) through the CLI, on the EPHEMERAL container only.

Covers plan section 5 AC-5, AC-6, AC-7, AC-9, FR-6(c), the `planned` parking-lot rule (P2-m1), and the S7a legs
of plan3 section 6 test 13 (`unit <new ID>` / `unit <old ID>` return the ledger after a `wi move`; an archived
flip leaves minimal units that `unit` finds and `inflight` never lists). The tree is built in `tmp_path`, with a
scratch work-item registry of its own (never the repo's), and does not depend on any real backlog.
Run with `-m live`; the shared `falkordb-dev` is never contacted.
"""
import json
import subprocess
import uuid
from pathlib import Path

import pytest

from coordgraph import store
from coordgraph.cli import main
from coordgraph.identity import load_wilib
from tests.fixtures.live import SHARED_PORTS, falkordb_container  # noqa: F401  (session fixture)

pytestmark = pytest.mark.live

LEGACY_PATHS = {"*/docs/BACKLOG.md": "$1", "docs/BACKLOG.md": "cpg", "claude/*/kaizen/plan.md": "$1"}


def ledger(status: str, rows: str, item: str | None = None) -> str:
    head = f"> **Status:** {status} · **Owner:** `teco` · **Tracks:** —" + (f" · **Item:** `{item}`" if item else "")
    cols = "| Unit | Item | Owner | Status | Deliverable |\n|---|---|---|---|---|\n"
    return f"# L\n\n{head}\n\n{cols}{rows}"


LEDGER_X_ROWS = """\
| U1 first | GAL-TASK | coder | **accepted** | shipped x |
| U2 | - | coder | **queued** | later |
| U3 | - | coder | **relayed** | "which port?" — relayed 2026-11-01 |
| U32 | - | coder | **queued** | z |
| U32 | - | coder | **in-flight** | z2 `a/docs/reviews/open.md` |
"""
LEDGER_Y_ROWS = "| S1 | - | coder | **accepted** | done |\n| S2 | - | coder | **accepted** | done |\n"
LEDGER_Z_ROWS = "| R1 | - | coder | **in-flight** | root work |\n"
LEDGER_OLD_ROWS = "| A1 | GAL-TASK2 | coder | **relayed** | \"archived q\" — relayed 2026-10-01 |\n"
REVIEW = ("# R\n\n> **Status:** active · **Owner:** `analyst` · **Tracks:** —\n\n## Pass 1\n\n"
          "Verdict: needs changes\n")
REQ = "# Q\n\n> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** —\n\ntext\n"
BACKLOG_A = ("# B\n\n### K-001 — Alpha\n\n- **Status:** 🔵 proposed\n- **Priority:** high\n\n"
             "### K-002 — Beta\n\n- **Status:** 🟡 in-progress\n\n"
             "### K-003 — Gamma\n\n- **Status:** ⚪ rejected — record\n\n"
             "### K-004 — Delta\n\n- **Status:** ✅ delivered\n")
BACKLOG_CPG = "# B\n\n### C-001 — Cpg thing\n\n- **Status:** 🔵 proposed\n"
PLAN_TECO = ("# P\n\n### K-010 — Agent thing\n\n- **Status:** 🔵 proposed\n\n"
             "### K-011 — Parked\n\n- **Status:** ⚪ deferred\n")

X = "a/docs/plans/x-coordination.md"
Y = "b/docs/plans/y-coordination.md"
Z = "docs/plans/z-coordination.md"
OLD = "d/docs/plans/y-coordination.md"  # archived, and sharing the slug `y` with the active ledger in `b`
REV = "a/docs/reviews/open.md"
REV_DONE = "a/docs/reviews/done.md"
REQ_PATH = "a/docs/requirements/q.md"


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def make_registry(root: Path) -> dict:
    """A scratch registry: feature F under initiative I3, a task under it; returns the ids."""
    load_wilib()
    from wilib import alloc, move, project

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    project.init(root, "GAL", ["a", "b", "c", "cpg", "teco", "repo"])
    pj = root / ".work-items" / "project.json"
    data = json.loads(pj.read_text())
    data["legacy_paths"] = LEGACY_PATHS
    pj.write_text(json.dumps(data))
    i3 = alloc.new(root, "I", title="three")
    i4 = alloc.new(root, "I", title="four")
    old = alloc.new(root, "F", parent=i3, title="feature")
    task = alloc.new(root, "T", parent=old, title="task")
    task2 = alloc.new(root, "T", parent=old, title="archived task")
    new = move.move(root, old, i4)
    new_id = new if isinstance(new, str) else next(iter(new.values())) if isinstance(new, dict) else str(new)
    return {"old": old, "new": new_id, "task": task, "task2": task2, "i3": i3, "i4": i4}


@pytest.fixture
def env(falkordb_container, tmp_path, monkeypatch):  # noqa: F811
    c = falkordb_container
    assert c.host == "127.0.0.1" and c.port not in SHARED_PORTS and c.name.startswith("coordgraph-test-")
    ids = make_registry(tmp_path)
    old = ids["old"]
    put(tmp_path, X, ledger("active", LEDGER_X_ROWS, item=old).replace("GAL-TASK", ids["task"]))
    put(tmp_path, Y, ledger("active", LEDGER_Y_ROWS))
    put(tmp_path, Z, ledger("active", LEDGER_Z_ROWS))
    put(tmp_path, REV, REVIEW)
    put(tmp_path, REV_DONE, REVIEW)  # cited only by an archived ledger below
    put(tmp_path, OLD,
        ledger("archived", LEDGER_OLD_ROWS.replace("GAL-TASK2", ids["task2"])
               + "| A2 | - | coder | **blocked** | `a/docs/reviews/done.md` |\n"))
    put(tmp_path, REQ_PATH, REQ)  # slug q: no ledger
    put(tmp_path, "a/docs/requirements/x.md", REQ)  # slug x: picked up by ledger x
    put(tmp_path, "a/docs/requirements/w.md",  # picked up only through its Item: (old id) = ledger x's candidates
        REQ.replace("**Tracks:** —", f"**Tracks:** — · **Item:** `{ids['new']}`"))
    put(tmp_path, "a/docs/BACKLOG.md", BACKLOG_A)
    put(tmp_path, "docs/BACKLOG.md", BACKLOG_CPG)
    put(tmp_path, "claude/teco/kaizen/plan.md", PLAN_TECO)
    key = f"t-{uuid.uuid4().hex[:10]}"
    for name, value in (("COORDGRAPH_REPO_ROOT", str(tmp_path)), ("COORDGRAPH_FALKORDB_HOST", c.host),
                        ("COORDGRAPH_FALKORDB_PORT", str(c.port)), ("COORDGRAPH_EPHEMERAL_PORT", str(c.port)),
                        ("COORDGRAPH_KEY", key), ("COORDGRAPH_STATE_DIR", str(tmp_path / ".state"))):
        monkeypatch.setenv(name, value)
    return tmp_path, key, c, ids


def graph_of(env_):
    root, key, c, _ = env_
    return store.connect(c.host, c.port, key, {"COORDGRAPH_EPHEMERAL_PORT": str(c.port)})[1]


def run(capsys, *argv) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


@pytest.fixture
def synced(env, capsys):
    assert run(capsys, "bootstrap")[0] == 0
    code, out, err = run(capsys, "sync", "--all")
    assert code == 0, (out, err)
    return env


# --- AC-5 / AC-7 / FR-6(c): inflight ----------------------------------------------------------------------------

def test_ac5_inflight_lists_the_four_kinds_with_component_path_and_the_sync_header(synced, capsys):
    code, out, _ = run(capsys, "inflight")
    assert code == 0
    assert out.startswith("Synced: ") and "Drift: none" in out.split("\n\n")[0]
    assert out.split("\n")[1].startswith("Registry: ")
    units, waiting, verdicts, reqs = (out.split("\n\n", 1)[1].split("\n\n") + ["", "", "", ""])[:4]
    # (a) units of every open coordination, in two components and the root tree (labelled `root`)
    assert f"  a\t{X}" in units and f"  b\t{Y}" in units and f"  root\t{Z}" in units
    assert "    U2\tcoder\tqueued\t-\tlater" in units
    # (b) the paused question with its date
    assert f"  a\t{X}\tU3\tasked 2026-11-01\twhich port?" in waiting
    # (c) the unresolved verdict: present; the one cited only by an archived ledger: absent
    assert f"  a\t{REV}\tpass 1\t" in verdicts and "done.md" not in verdicts
    # (d) requirement not picked up; `x` (slug) and `w` (Item: via identity) are picked up
    assert f"  a\t{REQ_PATH}\tInterviewing" in reqs
    assert "requirements/x.md" not in reqs and "requirements/w.md" not in reqs


def test_ac7_an_all_accepted_active_ledger_stays_and_an_archived_one_never_appears(synced, capsys):
    root = synced[0]
    out = run(capsys, "inflight")[1]
    assert Y in out and "    S2\tcoder\taccepted" in out  # all accepted, still `active`
    assert OLD not in out and "Waiting on the stakeholder (1)" in out  # the archived ledger's paused unit is not one
    put(root, Y, (root / Y).read_text().replace("**Status:** active", "**Status:** archived"))
    assert run(capsys, "sync", Y)[0] == 0
    out = run(capsys, "inflight")[1]
    assert Y not in out
    # the flip kept minimal units: `unit` still finds them, marked archived, with no cell text
    code, out, _ = run(capsys, "unit", "y", "S1")
    assert code == 0 and f"{Y} · docStatus: archived" in out and "status: accepted" in out
    assert "deliverable" not in out and "archived ledger" in out


def test_fr6c_a_review_cited_by_an_accepted_row_leaves_the_view_and_one_cited_by_an_in_flight_row_stays(synced, capsys):
    root = synced[0]
    def verdicts() -> str:
        return run(capsys, "inflight")[1].split('Open "needs changes" verdicts')[1].split("Requirements")[0]

    assert REV in verdicts()  # cited by x's in-flight U32~2 row
    put(root, X, (root / X).read_text().replace("| **in-flight** | z2", "| **accepted** | z2"))
    assert run(capsys, "sync", X)[0] == 0
    assert REV not in verdicts() and "(0)" in verdicts()


def test_inflight_waiting_is_only_the_questions_waiting_on_the_stakeholder(synced, capsys):
    code, out, _ = run(capsys, "inflight", "--waiting")
    assert code == 0 and "U3\tasked 2026-11-01" in out
    assert "Units in open" not in out and "needs changes" not in out and "Requirements" not in out


def test_a_requirement_is_picked_up_through_the_old_id_after_a_move(synced, capsys, env):
    """`w` carries the NEW id, ledger x the OLD one and no ledger the new one: only the registry's candidate
    set joins them."""
    ids = env[3]
    assert ids["old"] != ids["new"] and f"`{ids['old']}`" in (env[0] / X).read_text()
    assert "requirements/w.md" not in run(capsys, "inflight")[1]


def test_a_hand_edit_without_the_hook_is_flagged_on_the_row_and_in_the_header(synced, capsys):
    root = synced[0]
    put(root, X, (root / X).read_text().replace("| U2 | - | coder | **queued** |", "| U2 | - | coder | **accepted** |"))
    out = run(capsys, "inflight")[1]
    assert f"  changed\t{X}\t→ file wins; run coordgraph sync" in out
    flagged = [line for line in out.splitlines() if "⚠ drift" in line and "\t" in line.strip()]
    assert any(line.strip().startswith("U2\t") and line.endswith("\t⚠ drift") for line in flagged)
    assert not any(line.strip().startswith("U1 first") and line.endswith("⚠ drift") for line in out.splitlines())


def test_an_unsynced_graph_key_is_one_clear_line_not_a_traceback(env, capsys):
    code, out, err = run(capsys, "inflight")
    assert code == 1 and out == "" and err.startswith("inflight: ")


# --- AC-6: unit ----------------------------------------------------------------------------------------------

def test_ac6_a_paused_unit_returns_status_agent_deliverable_and_the_question_with_its_date(synced, capsys):
    code, out, _ = run(capsys, "unit", "x", "U3")
    assert code == 0
    assert out.startswith("Synced: ")
    assert f"{X} · docStatus: active\n  U3\t{X}#U3" in out
    assert "status: paused [**relayed**]" in out
    assert "question: which port? (asked 2026-11-01)" in out


def test_ac6_a_repeated_label_returns_every_repeat_and_a_tilde_n_addresses_one(synced, capsys):
    out = run(capsys, "unit", "x", "U32")[1]
    assert f"{X}#U32\n" in out + "\n" and f"{X}#U32~2" in out and "2 rows match" in out
    one = run(capsys, "unit", "x", "U32~2")[1]
    assert f"{X}#U32~2" in one and f"{X}#U32\n" not in one + "\n"
    assert "status: in-flight" in one


def test_unit_matches_by_label_when_no_exact_key_exists_and_lists_every_unit_when_none_is_named(synced, capsys):
    out = run(capsys, "unit", "x", "U1")[1]
    assert f"U1 first\t{X}#U1 first" in out
    everything = run(capsys, "unit", "x")[1]
    for key in ("U1 first", "U2", "U3", "U32", "U32~2"):
        assert f"{X}#{key}" in everything
    assert "5 rows match" in everything


def test_unit_with_no_match_exits_1_and_says_what_was_asked(synced, capsys):
    code, out, _ = run(capsys, "unit", "x", "U99")
    assert code == 1 and out.rstrip().endswith("no unit matches x U99")


# --- plan3 test 13: identity at read time ---------------------------------------------------------------------

def test_unit_by_the_new_id_the_old_id_and_the_task_id_finds_the_ledgers_after_a_move(synced, capsys):
    root, ids = synced[0], synced[3]
    put(root, Y, ledger("active", LEDGER_Y_ROWS, item=ids["new"]))  # y stores the NEW id, x the OLD one
    assert run(capsys, "sync", Y)[0] == 0
    for token in (ids["new"], ids["old"]):
        code, out, _ = run(capsys, "unit", token)
        assert code == 0, (token, out)
        # x stores the OLD id, y the NEW one: both are one item
        assert f"{X} · docStatus" in out and f"{Y} · docStatus" in out
        assert Z not in out and OLD not in out
    code, out, _ = run(capsys, "unit", ids["task"])
    assert code == 0 and f"{X} · docStatus" in out and f"{X}#U1 first" in out and f"{X}#U2" not in out
    # a unit within the ledgers a feature ID found
    out = run(capsys, "unit", ids["old"], "S2")[1]
    assert f"{Y}#S2" in out and f"{X}#" not in out


def test_a_task_id_in_an_archived_ledger_sharing_a_slug_is_attributed_to_its_own_file(synced, capsys):
    code, out, _ = run(capsys, "unit", synced[3]["task2"])
    assert code == 0 and f"{OLD} · docStatus: archived" in out and f"{OLD}#A1" in out
    assert Y not in out


def test_unit_by_an_id_names_why_it_did_not_resolve_and_never_guesses(synced, capsys):
    code, out, _ = run(capsys, "unit", "GAL-F9999")
    assert code == 1 and "GAL-F9999: unknown" in out and "no unit matches GAL-F9999" in out


def test_the_id_joins_use_an_index_not_a_scan(synced):
    graph = graph_of(synced)
    from coordgraph import views

    for q in (views.Q_UNIT_TASK, views.Q_CARRIERS):
        plan = str(graph.explain(q, {"candidates": ["GAL-X-1", "x"]}))
        assert "Index" in plan, plan


def test_registry_line_reports_unavailable_under_a_bogus_wilib_root(synced, capsys, monkeypatch):
    monkeypatch.setenv("COORDGRAPH_WI_ROOT", "/nonexistent")
    import sys

    for m in [m for m in sys.modules if m == "wilib" or m.startswith("wilib.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if "work-items" not in p])
    out = run(capsys, "unit", "x", "U1")[1]
    assert "Registry: unavailable — identities compared as written" in out
    assert "U1 first" in out  # the slug join is the fallback and still works


# --- AC-9: planned ----------------------------------------------------------------------------------------------

def test_ac9_planned_lists_open_items_of_a_component_root_docs_and_an_agent_plan_with_the_header(synced, capsys):
    out = run(capsys, "planned", "a")[1]
    assert out.startswith("Synced: ")
    assert "  a:K-001\t🔵 proposed\thigh\t-\tAlpha\ta/docs/BACKLOG.md" in out
    assert "  a:K-002\t🟡 in-progress" in out and "K-004" not in out  # delivered: not open
    out = run(capsys, "planned", "cpg")[1]
    assert "  cpg:C-001\t🔵 proposed" in out and "docs/BACKLOG.md" in out
    out = run(capsys, "planned", "teco")[1]
    assert "  teco:K-010\t🔵 proposed" in out and "claude/teco/kaizen/plan.md" in out


def test_planned_keeps_rejected_and_deferred_records_out_of_the_planned_group(synced, capsys):
    out = run(capsys, "planned", "a")[1]
    planned_part, parked_part = out.split("\nDeferred / rejected:")
    assert "K-003" not in planned_part and "1 (a:K-003)" in parked_part
    full = run(capsys, "planned", "a", "--all")[1]
    head, tail = full.split("Deferred / rejected (a record, not planned work)")
    assert "K-003" not in head and "a:K-003\t⚪ rejected" in tail


def test_planned_items_never_appear_in_the_inflight_view(synced, capsys):
    out = run(capsys, "inflight")[1]
    assert not any(ref in out for ref in ("K-001", "K-002", "K-010", "C-001", "Alpha"))


def test_planned_for_an_unknown_scope_names_the_scopes_that_exist(synced, capsys):
    code, out, _ = run(capsys, "planned", "nope")
    assert code == 1 and "no backlog with scope 'nope'; known scopes: a, cpg, teco" in out


def test_planned_flags_a_hand_edited_item_row(synced, capsys):
    root = synced[0]
    put(root, "a/docs/BACKLOG.md", BACKLOG_A.replace("Alpha", "Alpha changed"))
    out = run(capsys, "planned", "a")[1]
    assert "\tAlpha\ta/docs/BACKLOG.md\t⚠ drift" in out and "Beta\ta/docs/BACKLOG.md\t⚠" not in out
