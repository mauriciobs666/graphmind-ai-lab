"""Hook registration detection: the sync header's `Hook:` state (D-1, S10 acceptance).

The cases live in `tests/fixtures/hook-settings-cases.json` (name, settings, optional settings_local, script,
kill_switch, expected, why): `hookreg` is the authority and the audit checker is aligned to the same table.
"""
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from coordgraph import drift
from coordgraph.config import Config
from coordgraph.hookreg import hook_status

SCRIPT = "coordination-graph/hooks/post-tool-sync.sh"
CASES = json.loads((Path(__file__).parent / "fixtures" / "hook-settings-cases.json").read_text(encoding="utf-8"))


def _fill(value, root):
    if isinstance(value, str):
        return value.replace("{ROOT}", str(root))
    if isinstance(value, list):
        return [_fill(v, root) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, root) for k, v in value.items()}
    return value


def _write(path: Path, content, root):
    path.write_text(content if isinstance(content, str) else json.dumps(_fill(content, root)), encoding="utf-8")


def build(root: Path, case: dict) -> Config:
    """Materialise one table case under `root` and return the config to ask with."""
    (root / ".claude").mkdir(exist_ok=True)
    script = root / SCRIPT
    mode = case.get("script", "executable")
    if mode == "directory":
        script.mkdir(parents=True)
    elif mode != "missing":
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/usr/bin/env bash\n")
        script.chmod(script.stat().st_mode | stat.S_IXUSR if mode == "executable" else 0o644)
    _write(root / ".claude" / "settings.json", case["settings"], root)
    if "settings_local" in case:
        _write(root / ".claude" / "settings.local.json", case["settings_local"], root)
    cfg = Config.from_env({"COORDGRAPH_REPO_ROOT": str(root), "COORDGRAPH_STATE_DIR": str(root / ".state")})
    return replace(cfg, hook_disabled=bool(case.get("kill_switch")))


def test_the_table_is_self_describing_and_unique():
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names)) >= 50
    for c in CASES:
        assert c["expected"] in ("registered", "not registered", "disabled") and c["why"] and "settings" in c


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_hook_status_matches_the_golden_table(tmp_path, case):
    assert hook_status(build(tmp_path, case), tmp_path) == case["expected"], case["why"]


def test_no_settings_files_at_all_is_not_registered(tmp_path):
    cfg = Config.from_env({"COORDGRAPH_REPO_ROOT": str(tmp_path), "COORDGRAPH_STATE_DIR": str(tmp_path / ".state")})
    assert hook_status(replace(cfg, hook_disabled=False), tmp_path) == "not registered"


def test_the_disabled_state_file_overrides_registration(tmp_path):
    case = next(c for c in CASES if c["name"] == "documented-block-in-settings-json")
    cfg = build(tmp_path, case)
    assert hook_status(cfg, tmp_path) == "registered"
    (tmp_path / ".state").mkdir()
    (tmp_path / ".state" / "disabled").touch()
    assert hook_status(cfg, tmp_path) == "disabled"


def test_header_prints_the_state_verbatim():
    rep = drift.DriftReport(0, None, None, [])
    for state in ("registered", "not registered", "disabled"):
        assert drift.render_header(rep, state).splitlines()[0].endswith(f" · Hook: {state}")


def test_a_symlinked_script_counts_because_it_runs(tmp_path):
    case = next(c for c in CASES if c["name"] == "documented-block-in-settings-json")
    cfg = build(tmp_path, case)
    real = tmp_path / "real.sh"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    (tmp_path / SCRIPT).unlink()
    os.symlink(real, tmp_path / SCRIPT)
    assert hook_status(cfg, tmp_path) == "registered"
