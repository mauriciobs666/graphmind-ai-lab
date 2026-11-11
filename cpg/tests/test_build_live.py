"""P1-44 live — `build_unit` over the real sandbox runner (opt-in: `pytest -m live`; needs Docker and the Joern image).

Throwaway mechanics only, as `tests/test_joern_run.py`'s live tests: the runner's audit records go to a temp `CCG_AUDIT_DIR`,
the work area is a temp `.ccg-work`, the key area a temp directory, and the repository a one-commit git repository in `tmp_path`.
"""
from __future__ import annotations

import os
import subprocess

import pytest

from ccg import atrest, bundle, build, ids, sandbox_audit
from ccg.units import Unit

pytestmark = pytest.mark.live

SRC = "def add(a, b):\n    return a + b\n\n\ndef twice(x):\n    return add(x, x)\n"


def one_commit_repo(path):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    path.mkdir()
    (path / "pyproject.toml").write_text("[project]\nname = 'live'\n")
    (path / "m.py").write_text(SRC)
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True, text=True, env=env).stdout.strip()
    run("init", "-q", "-b", "main")
    run("add", "-A")
    run("commit", "-q", "-m", "c1")
    return run("rev-parse", "HEAD")


def test_a_real_python_unit_builds_a_valid_1_0_bundle_through_the_sandbox_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("CCG_AUDIT_DIR", str(tmp_path / "audit"))
    area = tmp_path / ".ccg-work"
    area.mkdir(mode=0o700)
    repo = tmp_path / "repo"
    sha = one_commit_repo(repo)
    u = Unit(ids.unit_id("live", ".", "python"), "live", ".", "python")
    ctx = build.BuildContext(units=(u,), root=area, keys_root=tmp_path / "keys")
    res = build.build_unit(u, repo, sha, "live-sys", ctx)
    assert (res.ok, res.fail_reason) == (True, None)
    records = bundle.loads(atrest.read_bundle("live-sys", u.unitId, sha, area))
    assert (records[0]["bundleSchemaVersion"], "bindKeyCheck" in records[0]) == ("1.0", False)
    assert sandbox_audit.check_sandbox_ids(records[0], tmp_path / "audit") == []       # the header's ids name real, clean runner records
    assert records[0]["sandboxAuditIds"]
    assert {r["name"] for r in records if r["type"] == "symbol" and r["kind"] == "function"} >= {"add", "twice"}
