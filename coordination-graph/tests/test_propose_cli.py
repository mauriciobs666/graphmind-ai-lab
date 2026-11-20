"""`coordgraph propose <scope>`: the graph reads (note 7.5) feeding the pure `propose`, rendered under the
sync header; read-only (AC-14)."""
import hashlib
import json
from pathlib import Path

import pytest

from coordgraph import commands_propose, commands_store, sync
from coordgraph.cli import main
from coordgraph.config import Config
from memgraph import MemGraph, Result

PROJECT = {"key": "GAL", "legacy_paths": {"*/docs/BACKLOG.md": "$1"}}


def put(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def item(n, status="🔵 proposed", prio="medium", deps="none", touches="none", extra=""):
    return (f"### K-{n:03d} — Item {n}\n\n- **Status:** {status}\n- **Priority:** {prio}\n"
            f"- **Depends on:** {deps}\n- **Touches:** {touches}\n{extra}\n")


class ProposeGraph(MemGraph):
    """`MemGraph` plus the read-only statements `propose` issues, answered from the stored rows."""

    def __init__(self):
        super().__init__()
        self.ro_calls = []

    def _items(self):
        for path, (kind, meta, rows) in sorted(self.sources.items()):
            if kind == "backlog":
                for r in rows:
                    yield meta, path, r

    def ro_query(self, q, params=None):
        params = params or {}
        self.ro_calls.append(q)
        if q == commands_propose.BACKLOG_EXISTS:
            return Result([[len({p for m, p, _r in self._items() if m.get("scope") == params["scope"]})]])
        if q == commands_propose.ITEMS:
            rows = [r for m, p, r in self._items() if m.get("scope") == params["scope"] and r["open"]]
            return Result([[r["ref"], r.get("scope"), r.get("statusMarker"), r["title"], r.get("priority"),
                            r.get("milestone"), r.get("milestoneOrder"), r.get("milestoneStarted"),
                            r.get("dependsOn"), r.get("touches"), p, r["ord"]]
                           for m, p, r in self._items() if m.get("scope") == params["scope"] and r["open"]])
        if q == commands_propose.DEP_ITEMS:
            return Result([[r["ref"], r.get("scope"), r["open"], p] for _m, p, r in self._items()
                           if r["ref"] in params["candidates"]])
        raise AssertionError(f"unexpected read: {q[:60]}")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    put(tmp_path, ".work-items/project.json", json.dumps(PROJECT))
    put(tmp_path, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item(1, prio="high") + item(
        2, deps="K-001", touches="file:a.py") + item(3, deps="K-001", touches="file:a.py") + item(
        4, deps="K-099") + item(5, deps="other:K-007") + item(6, deps="other:K-008") + item(7, prio="low"))
    put(tmp_path, "other/docs/BACKLOG.md", "# B\n\n" + item(7))
    put(tmp_path, "falkor-chat/docs/HISTORY.md", "# H\n\n## 2026-01-01 — K-099 closed: x\n")
    put(tmp_path, "other/docs/HISTORY.md", "# H\n\n## 2026-01-01 — K-008 delivered: x\n")
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(tmp_path))
    graph = ProposeGraph()
    sync.sync_all(graph, sync.Parser.for_root(tmp_path))
    monkeypatch.setattr(commands_store, "_target", lambda args: (Config.from_env(), "k", tmp_path, None, graph))
    return tmp_path, graph


def test_propose_prints_header_then_placements_with_resolved_dependencies(repo, capsys):
    root, graph = repo
    assert main(["propose", "falkor-chat"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Synced:") and "Drift: none" in out.split("Milestone")[0]
    assert "falkor-chat:K-001" in out
    # K-099 is delivered per its history heading -> K-004 is placed, not 'not plannable'
    assert "Not plannable yet" not in out
    # other:K-008 delivered in other's history; other:K-007 open in other's backlog -> held with path
    assert "external blocker other:K-007 in `other/docs/BACKLOG.md`" in out
    assert "falkor-chat:K-005" in out.split("Held:")[1]
    assert "falkor-chat:K-006" in out.split("Milestone proposed-1")[1].split("Held:")[0]


def test_propose_overlap_between_siblings_is_named(repo, capsys):
    main(["propose", "falkor-chat"])
    out = capsys.readouterr().out
    assert "overlap with falkor-chat:K-002 on file a.py" in out


def test_propose_unknown_dependency_is_not_plannable_with_actionable_text(repo, capsys):
    root, _ = repo
    put(root, "falkor-chat/docs/HISTORY.md", "# H\n\nK-099 was mentioned only\n")
    sync.sync_all(repo[1], sync.Parser.for_root(root))
    main(["propose", "falkor-chat"])
    out = capsys.readouterr().out
    assert "Not plannable yet" in out and "unknown dependency K-099" in out


def test_propose_writes_nothing_ac14(repo, capsys):
    root, graph = repo
    def digest():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob("*"))
                if p.is_file() and p.name != ".lock"}  # the registry read's gitignored lock file
    before, writes, state = digest(), list(graph.writes), dict(graph.sync_state)
    main(["propose", "falkor-chat", "--per-milestone", "2"])
    assert digest() == before and graph.writes == writes and graph.sync_state == state
    assert graph.ro_calls and all(q.startswith("MATCH") for q in graph.ro_calls)


def test_per_milestone_flag_reaches_the_planner(repo, capsys):
    main(["propose", "falkor-chat", "--per-milestone", "2"])
    assert "proposed-2" in capsys.readouterr().out


def test_unknown_scope_is_an_error_not_an_empty_proposal(repo, capsys):
    assert main(["propose", "nosuch"]) == 1
    assert "nosuch" in capsys.readouterr().err


def test_default_per_milestone_is_six(repo, capsys):
    main(["propose", "falkor-chat"])
    out = capsys.readouterr().out
    assert "Milestone proposed-1" in out and "proposed-2" not in out


def test_fixed_milestone_order_and_started_marker_are_read_from_the_graph(tmp_path, monkeypatch, capsys):
    put(tmp_path, ".work-items/project.json", json.dumps(PROJECT))
    fixed = "- **Milestone:** M7 — order {o} — approved 2026-11-01 — started 2026-11-02\n"
    put(tmp_path, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item(1, extra=fixed.format(o=2)) + item(
        2, extra=fixed.format(o=1)) + item(3, prio="high"))
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(tmp_path))
    graph = ProposeGraph()
    sync.sync_all(graph, sync.Parser.for_root(tmp_path))
    monkeypatch.setattr(commands_store, "_target", lambda args: (Config.from_env(), "k", tmp_path, None, graph))
    main(["propose", "falkor-chat"])
    out = capsys.readouterr().out
    m7 = out.split("Milestone M7")[1].split("Milestone proposed-1")[0]
    assert m7.index("K-002") < m7.index("K-001")          # order 1 before order 2
    assert "ahead-of-M7" not in out                       # started: no insertion before it
    assert "falkor-chat:K-003" in out.split("Milestone proposed-1")[1]
