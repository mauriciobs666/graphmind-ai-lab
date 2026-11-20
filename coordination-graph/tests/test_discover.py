import glob
import os

import pytest

from coordgraph.discover import (
    Found,
    component_of,
    discover,
    kb_files,
    kind_of,
    slug_of,
)

COVERED = {
    "docs/plans/a-coordination.md": "ledger",
    "falkor-chat/docs/plans/b-coordination.md": "ledger",
    "docs/reviews/r.md": "review",
    "claude/docs/reviews/r2.md": "review",
    "docs/requirements/q.md": "requirement",
    "salesperson/docs/requirements/q.md": "requirement",
    "docs/BACKLOG.md": "backlog",
    "falkor-chat/docs/BACKLOG.md": "backlog",
    "claude/cobb/kaizen/plan.md": "backlog",
}
NOT_COVERED = [
    "docs/plans/a.md",  # a plan, not a ledger
    "docs/plans/a-coordination.txt",
    "docs/reviews/sub/r.md",  # glob `*` does not cross directories
    "docs/archive/reviews/r.md",
    "archive/docs/reviews/r.md",
    "x/archive/docs/plans/y-coordination.md",
    "falkor-chat/docs/archive/plans/x-coordination.md",
    "deprecated/docs/reviews/r.md",
    "claude/cobb/kaizen/history.md",
    "claude/cobb/kaizen/sub/plan.md",
    "kaizen/plan.md",
    "claude/kaizen/plan.md",
    "node_modules/x/docs/reviews/r.md",
    "x/.venv/lib/docs/reviews/r.md",
    ".git/docs/reviews/r.md",
    "x/.hidden/docs/reviews/r.md",
    "docs/reviews/.hidden.md",
    "README.md",
]


def _build(root, paths):
    for rel in paths:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# x\n")


def test_covered_paths_found_with_their_kind_and_nothing_else(tmp_path):
    _build(tmp_path, [*COVERED, *NOT_COVERED])
    assert discover(tmp_path) == sorted(Found(p, k) for p, k in COVERED.items())


@pytest.mark.parametrize("path,kind", sorted(COVERED.items()))
def test_kind_of_covered(path, kind):
    assert kind_of(path) == kind


@pytest.mark.parametrize("path", NOT_COVERED)
def test_kind_of_not_covered(path):
    assert kind_of(path) is None


def test_deprecated_is_excluded_only_at_the_repo_root(tmp_path):
    _build(tmp_path, ["deprecated/docs/reviews/r.md", "x/deprecated/docs/reviews/r.md"])
    assert [f.path for f in discover(tmp_path)] == ["x/deprecated/docs/reviews/r.md"]


def test_agrees_with_python_recursive_glob_on_the_same_tree(tmp_path):
    """The one glob form everywhere: `discover` equals the plan's globs run through `glob`."""
    _build(tmp_path, [*COVERED, *NOT_COVERED])
    patterns = [
        "**/docs/plans/*-coordination.md", "**/docs/reviews/*.md", "**/docs/requirements/*.md",
        "**/docs/BACKLOG.md", "claude/*/kaizen/plan.md",
    ]
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        expected = set()
        for pat in patterns:
            expected |= {
                p for p in glob.glob(pat, recursive=True)
                if "archive" not in p.split("/")[:-1] and not p.startswith("deprecated/")
                and not {"node_modules", ".venv", ".git"} & set(p.split("/"))
            }
    finally:
        os.chdir(old)
    assert {f.path for f in discover(tmp_path)} == expected


def test_a_directory_named_like_a_source_is_not_a_source(tmp_path):
    (tmp_path / "docs" / "reviews" / "dir.md").mkdir(parents=True)
    assert discover(tmp_path) == []


def test_kb_registry_skips_comments_and_blanks(tmp_path):
    reg = tmp_path / "coordination-graph" / "kb-files.txt"
    reg.parent.mkdir(parents=True)
    reg.write_text("# cobb-maintained\n\nclaude/a/kb.md\n  claude/b/kb.md  \n")
    assert kb_files(tmp_path) == ["claude/a/kb.md", "claude/b/kb.md"]
    assert kb_files(tmp_path / "missing") == []


@pytest.mark.parametrize("path,slug", [
    ("docs/plans/connected-code-graph-coordination.md", "connected-code-graph"),
    ("docs/plans/executor2-coordination.md", "executor2"),
    ("docs/plans/x-graph-coordination.md", "x-graph"),
    ("docs/reviews/x-rca.md", "x"),
    ("docs/reviews/x-ml.md", "x"),
    ("docs/reviews/x-impl.md", "x"),
    ("docs/reviews/x-sweep.md", "x-sweep"),  # not a closed-set role
    ("docs/requirements/plain.md", "plain"),
    # a feature whose own name ends in a role word keeps it where that kind carries no role
    ("claude/docs/requirements/coordination-graph.md", "coordination-graph"),
    ("claude/docs/reviews/coordination-graph.md", "coordination-graph"),
    ("docs/BACKLOG.md", "BACKLOG"),
    # outside the coverage globs the whole closed set applies
    ("docs/plans/x-graph.md", "x"),
    ("docs/test-reports/x-report.md", "x"),
])
def test_slug_strips_only_a_role_the_kind_can_carry(path, slug):
    assert slug_of(path) == slug


@pytest.mark.parametrize("path,component", [
    ("docs/BACKLOG.md", "cpg"),
    ("docs/plans/x.md", "root"),
    ("docs/reviews/x.md", "root"),
    ("docs/requirements/x.md", "root"),
    ("falkor-chat/docs/plans/x-coordination.md", "falkor-chat"),
    ("falkor-chat/docs/BACKLOG.md", "falkor-chat"),
    ("claude/cobb/kaizen/plan.md", "claude"),
    ("claude/docs/reviews/x.md", "claude"),
])
def test_component_rule(path, component):
    assert component_of(path) == component
