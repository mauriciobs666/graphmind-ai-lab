"""Scope derivation from `.work-items/project.json` `legacy_paths` (plan3 section 6, test 11)."""
import json
from pathlib import Path

import pytest

from coordgraph.discover import discover
from coordgraph.scope import load_legacy_paths, scope_of

REPO = Path(__file__).resolve().parents[2]


def expected_scope(path: str) -> str:
    """Independent of `scope_of`: the rule project.json's `legacy_paths` states, read by hand."""
    parts = path.split("/")
    if path == "docs/BACKLOG.md":
        return "cpg"
    if parts[0] == "claude" and parts[2:] == ["kaizen", "plan.md"]:
        return parts[1]
    assert parts[1:] == ["docs", "BACKLOG.md"], path
    return parts[0]


def test_every_real_backlog_path_gets_the_scope_project_json_states():
    legacy = load_legacy_paths(REPO)
    assert legacy, "the real repo declares legacy_paths"
    backlogs = [f.path for f in discover(REPO) if f.kind == "backlog"]
    assert len(backlogs) >= 19
    for path in backlogs:
        assert scope_of(path, legacy) == expected_scope(path), path


def test_scope_follows_the_file_not_the_paths_the_repo_has_today(tmp_path):
    (tmp_path / ".work-items").mkdir()
    (tmp_path / ".work-items" / "project.json").write_text(json.dumps(
        {"legacy_paths": {"*/docs/BACKLOG.md": "lab-$1", "docs/BACKLOG.md": "root-home"}}))
    legacy = load_legacy_paths(tmp_path)
    assert scope_of("falkor-chat/docs/BACKLOG.md", legacy) == "lab-falkor-chat"
    assert scope_of("docs/BACKLOG.md", legacy) == "root-home"
    assert scope_of("claude/cobb/kaizen/plan.md", legacy) is None


@pytest.mark.parametrize("content", [None, "not json", "[]", '{"legacy_paths": 3}'])
def test_a_missing_or_unusable_project_file_gives_no_scopes(tmp_path, content):
    if content is not None:
        (tmp_path / ".work-items").mkdir()
        (tmp_path / ".work-items" / "project.json").write_text(content)
    assert load_legacy_paths(tmp_path) == {}


def test_scope_matches_wilib_home_of_for_every_real_backlog_path():
    from coordgraph.lint import load_wilib_rules
    assert load_wilib_rules() is not None
    from wilib import check
    legacy = load_legacy_paths(REPO)
    for f in discover(REPO):
        if f.kind == "backlog":
            assert scope_of(f.path, legacy) == check._home_of(f.path, legacy)
