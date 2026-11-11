"""P1-15 — `extractors/indexers/run.py`: the indexer runner and its stager.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-15; contract `docs/plans/connected-code-graph-egress-spec3.md`
§7.4 and `-egress-spec4.md` §1. The container step runs against a stand-in runner (a script that records its argv and
writes what the real runner writes: a `.scip` and one `sandbox` audit record); the real runner's refusals are
exercised by the `live` tests at the end (a refusal launches no container).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
import yaml

from extractors.indexers import run as ir
from extractors.joern.run import RunnerError, StageError
from test_joern_run import make_repo, units_of

CPG = Path(__file__).resolve().parent.parent
RUNNER_SH = CPG / "sandbox" / "offline-run.sh"

STUB = textwrap.dedent('''\
    #!/usr/bin/env python3
    import json, os, pathlib, signal, subprocess, sys, time, uuid
    here = pathlib.Path(__file__).resolve().parent
    a = sys.argv[1:]
    i = a.index("--")
    opt = {a[j]: a[j + 1] for j in range(0, i, 2)}
    mode = (here / "mode").read_text().strip() if (here / "mode").exists() else "ok"
    with open(here / "argv.log", "a") as f:
        f.write(json.dumps({"argv": a, "image": os.environ.get("CCG_TOOLCHAIN_IMAGE")}) + "\\n")
    (here / "pid").write_text(str(os.getpid()))
    aid = str(uuid.uuid4())
    out = pathlib.Path(opt["--output"])

    def record(rc, dur):
        d = pathlib.Path(os.environ["CCG_AUDIT_DIR"])
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "sandbox.jsonl", "a") as f:
            f.write(json.dumps({"auditId": aid, "frontend": opt["--frontend"], "exitCode": rc, "durationMs": dur,
                                "mounts": [{"target": "/src", "source": opt["--source"]}]}) + "\\n")

    if mode == "ok":
        (out / (aid + ".scip")).write_bytes(b"scip")
        record(0, 5)
    elif mode == "flood":
        for _ in range(30):
            sys.stdout.buffer.write(b"x" * 1048576)
            sys.stderr.buffer.write(b"e" * 1048576)
        sys.stdout.flush(); sys.stderr.flush()
        (out / (aid + ".scip")).write_bytes(b"scip")
        record(0, 5)
    elif mode == "leaky":
        child = subprocess.Popen(["sleep", "60"])                  # inherits the pipes: it outlives the runner holding them
        (here / "child").write_text(str(child.pid))
        (out / (aid + ".scip")).write_bytes(b"scip")
        record(0, 5)
    elif mode == "badname":
        (out / "not-a-runner-id.scip").write_bytes(b"scip")
        record(0, 5)
    elif mode == "leaver":
        child = subprocess.Popen(["sleep", "60"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (here / "child").write_text(str(child.pid))
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        time.sleep(60)
    elif mode == "fail":
        (out / (aid + ".scip")).write_bytes(b"partial")
        record(3, 5)
        sys.exit(3)
    elif mode.startswith("t124:"):
        (out / (aid + ".scip")).write_bytes(b"partial")
        if mode != "t124:none":
            record(124, int(mode.split(":")[1]))
        sys.exit(124)
    elif mode == "term":
        def term(*_):
            (here / "mark").write_text("terminated")
            sys.exit(143)
        signal.signal(signal.SIGTERM, term)
        time.sleep(60)
    elif mode == "wedge":
        child = subprocess.Popen(["sleep", "60"])
        (here / "child").write_text(str(child.pid))
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
''')


@pytest.fixture()
def stub(tmp_path, monkeypatch):
    d = tmp_path / "stubdir"
    d.mkdir()
    p = d / "stub-runner"
    p.write_text(STUB)
    p.chmod(0o755)
    monkeypatch.setenv("CCG_AUDIT_DIR", str(tmp_path / "audit"))
    return p


def set_mode(stub: Path, mode: str) -> None:
    (stub.parent / "mode").write_text(mode)


def runner_calls(stub: Path) -> list[dict]:
    log = stub.parent / "argv.log"
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


@pytest.fixture()
def area(tmp_path):
    a = tmp_path / ".ccg-work"
    a.mkdir(mode=0o755)
    return a


FAST = ir.Timeouts(wall_s=2, outer_s=30, kill_grace_s=1)
IMAGE = "ccg-toolchain-indexer:0123456789ab"


def unit_repo(tmp_path, lang):
    if lang == "python":
        repo, sha = make_repo(tmp_path / "repo", {"svc/pyproject.toml": "", "svc/m.py": "m = 1\n"})
        return repo, sha, units_of(repo)[("svc", "python")]
    repo, sha = make_repo(tmp_path / "repo", {"web/package.json": "{}", "web/a.ts": "export const a = 1;\n"})
    return repo, sha, units_of(repo)[("web", "javascript")]


def provisioned(tmp_path, lang) -> ir.Provisioned:
    root = tmp_path / "operator"
    root.mkdir()
    if lang == "python":
        (root / "env.json").write_text("[]")
        (root / "venv").mkdir()
        (root / "venv" / "marker").write_text("v")
        return ir.Provisioned(root, env_json="env.json")
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "node_modules" / "left-pad" / "index.js").write_text("1")
    (root / "node_modules" / ".bin").mkdir()
    os.symlink("../left-pad/index.js", root / "node_modules" / ".bin" / "lp")
    (root / "package-lock.json").write_text("{}")
    return ir.Provisioned(root, lockfile="package-lock.json")


def go(tmp_path, stub, area, lang="python", prov="yes", timeouts=FAST, **kw):
    repo, sha, unit = unit_repo(tmp_path, lang)
    p = provisioned(tmp_path, lang) if prov == "yes" else prov
    res = ir.run_indexer(repo, unit, [unit], sha, area / "w", p, image=IMAGE, area=area, runner=stub, timeouts=timeouts, **kw)
    return res, sha


# --- the §7.4 grammar --------------------------------------------------------------------------------------------


def test_typescript_argv():
    assert ir.tool_argv("scip-typescript") == ["index", "--cwd", "/src"]


def test_python_argv_is_the_grammar_of_spec3_and_spec4():
    assert ir.tool_argv("scip-python", env_json="env.json", project_name="falkorchat", project_version="abc123") == [
        "index", "--cwd", "/src", "--environment", "/src/env.json", "--project-name", "falkorchat",
        "--project-version", "abc123"]
    assert ir.tool_argv("scip-python", env_json="deps/e.json", project_name="p", project_version="v")[4] == "/src/deps/e.json"


@pytest.mark.parametrize("kw", [
    {"env_json": "env.json"}, {"project_name": "p"}, {"project_version": "v"},
])
def test_typescript_takes_nothing_but_cwd(kw):
    with pytest.raises(ValueError):
        ir.tool_argv("scip-typescript", **kw)


@pytest.mark.parametrize("kw", [
    {"env_json": "env.json", "project_name": "p"},                                     # no project version
    {"env_json": "env.json", "project_version": "v"},                                  # no project name
    {"project_name": "p", "project_version": "v"},                                     # no env.json
    {"env_json": "../env.json", "project_name": "p", "project_version": "v"},
    {"env_json": "a b.json", "project_name": "p", "project_version": "v"},
    {"env_json": "a//b.json", "project_name": "p", "project_version": "v"},
    {"env_json": "env.json", "project_name": "-p", "project_version": "v"},
    {"env_json": "env.json", "project_name": "p" * 65, "project_version": "v"},
    {"env_json": "env.json", "project_name": "p", "project_version": "v 1"},
    {"env_json": "env.json", "project_name": "p", "project_version": "v=1"},
    {"env_json": "env.json", "project_name": "p", "project_version": "v\n"},        # `$` would accept a trailing newline
    {"env_json": "env.json", "project_name": "p\n", "project_version": "v"},
    {"env_json": "env.json\n", "project_name": "p", "project_version": "v"},
])
def test_python_argv_outside_the_grammar_is_refused_client_side(kw):
    with pytest.raises(ValueError):
        ir.tool_argv("scip-python", **kw)


@pytest.mark.parametrize("fe", ["joern", "joern-parse", "pysrc2cpg", "jssrc2cpg", "scip", ""])
def test_no_frontend_but_the_two_indexers_ever_reaches_the_runner(fe):
    with pytest.raises(ValueError, match="never joern"):
        ir.runner_argv(Path("/r"), fe, Path("/s"), Path("/o"), "repo", ["index"])
    with pytest.raises(ValueError, match="never joern"):
        ir.tool_argv(fe)


def test_the_runner_argv_is_container_offline_with_the_indexer_frontend():
    a = ir.runner_argv(Path("/r"), "scip-python", Path("/s"), Path("/o"), "repo", ["index", "--cwd", "/src"])
    assert a == ["/r", "--backend", "container", "--mode", "offline", "--frontend", "scip-python", "--source", "/s",
                 "--output", "/o", "--codebase", "repo", "--", "index", "--cwd", "/src"]


def test_the_wall_clock_constant_is_the_runners_literal():
    text = RUNNER_SH.read_text()
    m = re.search(r"^SBX_INDEXER_WALL_S=(\d+)\s*$", text, re.M)
    assert m and int(m.group(1)) == ir.WALL_S


def test_the_outer_bound_exceeds_the_runners_cap_by_more_than_the_margin():
    assert ir.OUTER_S > ir.WALL_S + ir.MARGIN_S == 1860


def test_the_real_runner_refuses_an_outer_bound_at_or_below_the_sum(tmp_path, area):
    repo, sha, unit = unit_repo(tmp_path, "javascript")
    for outer in (ir.WALL_S, ir.WALL_S + ir.MARGIN_S):
        with pytest.raises(ValueError, match="outer bound"):
            ir.run_indexer(repo, unit, [unit], sha, area / "w", None, image=IMAGE, area=area,
                           timeouts=ir.Timeouts(outer_s=outer))
    with pytest.raises(ValueError, match="outer bound"):
        ir.run_indexer(repo, unit, [unit], sha, area / "w", None, image=IMAGE, area=area,
                       timeouts=ir.Timeouts(wall_s=5, outer_s=ir.OUTER_S))


def test_the_degradation_reason_is_an_existing_registry_code():
    rows = yaml.safe_load((CPG / "ccg" / "registry" / "reasons.yaml").read_text())
    assert ir.DEGRADED_REASON in {r["code"] for r in rows}


# --- the stager's rejections -------------------------------------------------------------------------------------


def tree(tmp_path) -> Path:
    t = tmp_path / "t"
    (t / "pkg" / "sub").mkdir(parents=True)
    (t / "pkg" / "f.txt").write_text("x")
    return t


def test_a_plain_tree_with_an_inside_relative_link_passes(tmp_path):
    t = tree(tmp_path)
    os.symlink("../f.txt", t / "pkg" / "sub" / "ok")
    os.symlink("pkg", t / "dirlink")
    os.symlink("nowhere", t / "dangling-inside")
    ir.validate_tree(t)


@pytest.mark.parametrize("target", ["/", "/etc/passwd", "../..", "../../x", "sub/../../../out"])
def test_a_symlink_that_is_absolute_or_leaves_the_directory_is_rejected(tmp_path, target):
    t = tree(tmp_path)
    os.symlink(target, t / "pkg" / "evil")
    with pytest.raises(ir.StageRejected, match="symlink"):
        ir.validate_tree(t)


def test_an_absolute_link_to_a_path_inside_the_directory_is_still_rejected(tmp_path):
    t = tree(tmp_path)
    os.symlink(str(t / "pkg"), t / "abs")
    with pytest.raises(ir.StageRejected, match="absolute"):
        ir.validate_tree(t)


def test_a_chain_of_links_that_ends_outside_is_rejected(tmp_path):
    t = tree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink("../../../outside", t / "pkg" / "sub" / "hop")           # t/pkg/sub/../../../outside: leaves t
    with pytest.raises(ir.StageRejected, match="leaving"):
        ir.validate_tree(t)
    (t / "pkg" / "sub" / "hop").unlink()
    os.symlink("sub2", t / "pkg" / "a")                                  # inside ...
    (t / "pkg" / "sub2").mkdir()
    os.symlink("../../../outside", t / "pkg" / "sub2" / "b")             # ... then out
    with pytest.raises(ir.StageRejected, match="leaving"):
        ir.validate_tree(t)


def test_a_link_that_is_inside_lexically_but_leaves_through_a_directory_link_is_rejected(tmp_path):
    t = tmp_path / "t"
    (t / "q" / "r").mkdir(parents=True)
    os.symlink("../..", t / "q" / "r" / "d")                     # d -> t/q: inside
    os.symlink("d/../../x", t / "q" / "r" / "l")                 # t/q/r/d/../../x is t/q/x lexically, parent-of-t/x for real
    with pytest.raises(ir.StageRejected, match="leaving"):
        ir.validate_tree(t)


def test_a_hardlink_is_rejected(tmp_path):
    t = tree(tmp_path)
    os.link(t / "pkg" / "f.txt", t / "pkg" / "g.txt")
    with pytest.raises(ir.StageRejected, match="hardlink"):
        ir.validate_tree(t)


def test_a_fifo_is_rejected_without_being_opened(tmp_path):
    t = tree(tmp_path)
    os.mkfifo(t / "pkg" / "pipe")
    with pytest.raises(ir.StageRejected, match="FIFO"):
        ir.validate_tree(t)


def test_a_socket_is_rejected(tmp_path):
    import socket
    t = tree(tmp_path)
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(t / "pkg" / "sock"))
    try:
        with pytest.raises(ir.StageRejected):
            ir.validate_tree(t)
    finally:
        s.close()


def test_the_root_itself_must_be_a_plain_directory(tmp_path):
    t = tree(tmp_path)
    os.symlink(t, tmp_path / "rootlink")
    with pytest.raises(ir.StageRejected):
        ir.validate_tree(tmp_path / "rootlink")
    with pytest.raises(ir.StageRejected):
        ir.validate_tree(t / "pkg" / "f.txt")


def test_provisioned_hardlinks_are_rejected_before_the_copy_hides_them(tmp_path, stub, area):
    repo, sha, unit = unit_repo(tmp_path, "javascript")
    prov = provisioned(tmp_path, "javascript")
    os.link(prov.root / "package-lock.json", prov.root / "node_modules" / "lock2")
    with pytest.raises(ir.StageRejected, match="hardlink"):
        ir.run_indexer(repo, unit, [unit], sha, area / "w", prov, image=IMAGE, area=area, runner=stub, timeouts=FAST)
    assert runner_calls(stub) == []


def test_the_staged_directory_is_validated_again_after_the_overlay(tmp_path, stub, area, monkeypatch):
    real = ir._overlay

    def plant(prov, stage):
        real(prov, stage)
        os.mkfifo(stage / "late-pipe")

    monkeypatch.setattr(ir, "_overlay", plant)
    with pytest.raises(ir.StageRejected, match="FIFO"):
        go(tmp_path, stub, area, "javascript")
    assert runner_calls(stub) == []


def test_a_provisioned_entry_colliding_with_the_units_own_file_is_rejected(tmp_path, stub, area):
    repo, sha = make_repo(tmp_path / "repo", {"web/package.json": "{}", "web/a.ts": "1\n", "web/node_modules/x": "own\n"})
    unit = units_of(repo)[("web", "javascript")]
    with pytest.raises(ir.StageRejected, match="collides"):
        ir.run_indexer(repo, unit, [unit], sha, area / "w", provisioned(tmp_path, "javascript"), image=IMAGE, area=area,
                       runner=stub, timeouts=FAST)
    assert runner_calls(stub) == []


# --- a run -------------------------------------------------------------------------------------------------------


def test_a_typescript_run_stages_dependencies_in_its_own_directory_and_calls_only_the_indexer_frontend(tmp_path, stub, area):
    res, sha = go(tmp_path, stub, area, "javascript")
    assert isinstance(res, ir.IndexerRun) and res.frontend == "scip-typescript"
    (call,) = runner_calls(stub)
    a = call["argv"]
    assert a[a.index("--frontend") + 1] == "scip-typescript" and a[a.index("--mode") + 1] == "offline"
    assert a[a.index("--backend") + 1] == "container" and "joern" not in a and "joern-parse" not in a
    assert a[a.index("--") + 1:] == ["index", "--cwd", "/src"]
    assert call["image"] == IMAGE
    stage = Path(a[a.index("--source") + 1])
    assert stage == area / "w" / "indexer" / "stage" == res.stage
    assert not (area / "w" / "stage").exists()                                    # the Joern stage is a different directory
    assert (stage / "a.ts").is_file() and (stage / "node_modules" / "left-pad" / "index.js").is_file()
    assert os.readlink(stage / "node_modules" / ".bin" / "lp") == "../left-pad/index.js"
    assert res.scip.parent == res.out_dir == area / "w" / "indexer" / "out" and res.scip.read_bytes() == b"scip"
    assert res.audit_id == res.scip.name[:-5]
    assert res.lockfile_sha256 == hashlib.sha256(b"{}").hexdigest() and res.env_json_sha256 is None


def test_a_python_run_passes_the_full_python_grammar_with_the_commit_as_version(tmp_path, stub, area):
    res, sha = go(tmp_path, stub, area, "python", project_name="falkorchat")
    (call,) = runner_calls(stub)
    a = call["argv"]
    assert a[a.index("--frontend") + 1] == "scip-python"
    assert a[a.index("--") + 1:] == ["index", "--cwd", "/src", "--environment", "/src/env.json", "--project-name",
                                      "falkorchat", "--project-version", sha]
    assert (res.stage / "env.json").is_file() and (res.stage / "venv" / "marker").is_file()
    assert res.env_json_sha256 == hashlib.sha256(b"[]").hexdigest() and res.lockfile_sha256 is None


def test_a_default_python_project_name_is_derived_and_valid(tmp_path, stub, area):
    go(tmp_path, stub, area, "python")
    a = runner_calls(stub)[0]["argv"]
    assert ir.NAME_RE.match(a[a.index("--project-name") + 1])


def test_a_tool_failure_leaves_no_artifact(tmp_path, stub, area):
    set_mode(stub, "fail")
    with pytest.raises(RunnerError, match="exited 3"):
        go(tmp_path, stub, area, "javascript")


@pytest.mark.parametrize("lang,mk", [("javascript", lambda t: None), ("python", lambda t: ir.Provisioned(Path(t / "nope"), "env.json"))])
def test_absent_dependencies_degrade_under_the_existing_code_and_run_nothing(tmp_path, stub, area, lang, mk):
    res, _ = go(tmp_path, stub, area, lang, prov=mk(tmp_path))
    assert isinstance(res, ir.Degraded) and res.reason == "headers-unavailable" and "Joern runs alone" in res.detail
    assert runner_calls(stub) == [] and not (area / "w").exists()


def test_a_provisioned_directory_missing_what_the_tool_needs_degrades(tmp_path, stub, area):
    py = unit_repo(tmp_path / "p", "python")
    js = unit_repo(tmp_path / "j", "javascript")
    empty = tmp_path / "empty"
    empty.mkdir()

    def attempt(repo_unit, prov):
        repo, sha, unit = repo_unit
        return ir.run_indexer(repo, unit, [unit], sha, area / "w", prov, image=IMAGE, area=area, runner=stub, timeouts=FAST)

    assert isinstance(attempt(js, ir.Provisioned(empty)), ir.Degraded)
    assert isinstance(attempt(py, ir.Provisioned(empty, env_json="env.json")), ir.Degraded)
    (empty / "env.json").mkdir()                                                 # a directory is not a file
    assert isinstance(attempt(py, ir.Provisioned(empty, env_json="env.json")), ir.Degraded)
    assert runner_calls(stub) == []


def test_a_language_without_an_indexer_is_a_stage_error(tmp_path, stub, area):
    repo, sha, unit = unit_repo(tmp_path, "python")
    go_unit = unit._replace(language="go")
    with pytest.raises(StageError, match="no SCIP indexer"):
        ir.run_indexer(repo, go_unit, [go_unit], sha, area / "w", None, image=IMAGE, area=area, runner=stub, timeouts=FAST)


@pytest.mark.parametrize("bad", ["", "a b", "x;y", "-img", "a" * 300])
def test_an_image_that_is_not_a_plain_reference_is_refused(tmp_path, stub, area, bad):
    repo, sha, unit = unit_repo(tmp_path, "javascript")
    with pytest.raises(ValueError, match="image"):
        ir.run_indexer(repo, unit, [unit], sha, area / "w", None, image=bad, area=area, runner=stub, timeouts=FAST)


# --- exit 124 and the outer bound --------------------------------------------------------------------------------


@pytest.mark.parametrize("mode,watchdog,ms", [("t124:2500", True, 2500), ("t124:2000", True, 2000), ("t124:1999", False, 1999),
                                              ("t124:30", False, 30), ("t124:none", None, None)])
def test_exit_124_is_a_timeout_failure_judged_by_duration_against_the_cap(tmp_path, stub, area, mode, watchdog, ms):
    set_mode(stub, mode)
    with pytest.raises(ir.RunnerTimeout) as e:
        go(tmp_path, stub, area, "javascript")                                   # wall_s = 2 -> cap 2000 ms
    assert e.value.watchdog is watchdog and e.value.duration_ms == ms
    assert isinstance(e.value, RunnerError) and "no artifact is used" in str(e.value)


def test_the_outer_bound_sends_term_so_the_runners_trap_can_run(tmp_path, stub, area):
    set_mode(stub, "term")
    with pytest.raises(ir.RunnerTimeout, match="outer bound") as e:
        go(tmp_path, stub, area, "javascript", timeouts=ir.Timeouts(wall_s=2, outer_s=1.5, kill_grace_s=5))
    assert e.value.watchdog is None and (stub.parent / "mark").read_text() == "terminated"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return "Z" not in Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False


def test_a_wedged_runner_is_ended_by_kill_with_its_children(tmp_path, stub, area):
    set_mode(stub, "wedge")
    t0 = time.monotonic()
    with pytest.raises(ir.RunnerTimeout, match="outer bound"):
        go(tmp_path, stub, area, "javascript", timeouts=ir.Timeouts(wall_s=2, outer_s=1, kill_grace_s=1))
    assert time.monotonic() - t0 < 8
    time.sleep(0.3)
    child = int((stub.parent / "child").read_text())
    assert not alive(int((stub.parent / "pid").read_text())) and not alive(child)


def test_a_normal_exit_leaves_no_thread_or_process_behind(tmp_path, stub, area):
    before = threading.active_count()
    go(tmp_path, stub, area, "javascript")
    assert threading.active_count() == before
    assert not alive(int((stub.parent / "pid").read_text()))


# --- the real runner: refusals launch no container ---------------------------------------------------------------

BAD_ARGV = [
    ("scip-typescript", ["index", "--cwd", "/src", "--install"], "forbidden-flag"),
    ("scip-typescript", ["npm", "ci"], "forbidden-flag"),
    ("scip-typescript", ["index", "--cwd", "/src", "--output", "/out/x.scip"], "forbidden-flag"),
    ("scip-typescript", ["index", "--cwd", "/src", "--environment", "/src/env.json"], "forbidden-flag"),
    ("scip-typescript", ["index", "--cwd=/src"], "forbidden-flag"),
    ("scip-typescript", ["index", "--cwd", "/src/.."], "forbidden-flag"),
    ("scip-typescript", ["index", "--cwd", "/etc"], "forbidden-flag"),
    ("scip-typescript", ["index", "--cwd", "/src", "--pnpm-workspaces"], "forbidden-flag"),
    ("scip-python", ["pip", "install", "x"], "forbidden-flag"),
    ("scip-python", ["index", "--cwd", "/src", "--environment", "/src/env.json", "--project-name", "p", "--project-version",
                     "v", "--infer-tsconfig"], "forbidden-flag"),
    ("scip-python", ["index", "--cwd", "/src", "--environment", "/src/env.json", "--project-name", "p"], "usage"),
    ("scip-python", ["index", "--cwd", "/src", "--project-name", "p", "--project-version", "v"], "usage"),
    ("scip-python", ["index", "--environment", "/src/env.json", "--project-name", "p", "--project-version", "v"], "usage"),
]


@pytest.mark.live
@pytest.mark.parametrize("fe,argv,reason", BAD_ARGV)
def test_the_real_runner_refuses_an_argv_outside_the_grammar(tmp_path, fe, argv, reason):
    src, out, audit = tmp_path / "src", tmp_path / "out", tmp_path / "audit"
    src.mkdir()
    out.mkdir()
    (src / "env.json").write_text("[]")
    # a grammar-conforming argv must be accepted by run.py's own builder; these are not, so build the runner argv directly
    full = [str(ir.RUNNER), "--backend", "container", "--mode", "offline", "--frontend", fe, "--source", str(src),
            "--output", str(out), "--codebase", "t", "--", *argv]
    env = {"PATH": os.environ["PATH"], "CCG_TOOLCHAIN_IMAGE": ir.indexer_image_tag(), "CCG_AUDIT_DIR": str(audit)}
    before = subprocess.run(["docker", "ps", "-aq", "--filter", "label=ccg.sandbox"], capture_output=True, text=True).stdout
    p = subprocess.run(full, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 64, p.stderr
    (rec,) = [json.loads(l) for l in (audit / "sandbox.jsonl").read_text().splitlines()]
    assert rec["refused"] is True and rec["refusalReason"] == reason
    assert subprocess.run(["docker", "ps", "-aq", "--filter", "label=ccg.sandbox"], capture_output=True, text=True).stdout == before


@pytest.mark.live
def test_the_real_runner_indexes_a_toy_typescript_project_and_the_parser_reads_it(tmp_path, monkeypatch):
    """A positive control through the throwaway container (a toy project, not a corpus): the audit record is found,
    its durationMs is read, the `.scip` parses, and no sandbox container is left behind."""
    from extractors.indexers import scip
    monkeypatch.setenv("CCG_AUDIT_DIR", str(tmp_path / "audit"))
    repo, sha = make_repo(tmp_path / "repo", {
        "web/package.json": '{"name":"toy","version":"1.0.0"}',
        "web/tsconfig.json": '{"compilerOptions":{"target":"es2020","module":"commonjs","strict":true},"include":["*.ts"]}',
        "web/a.ts": "export function helper(x: number): number { return x + 1; }\n",
        "web/b.ts": "import { helper } from './a';\nexport function run(): number { return helper(2); }\n"})
    unit = units_of(repo)[("web", "javascript")]
    op = tmp_path / "op"
    (op / "node_modules").mkdir(parents=True)
    area = tmp_path / ".ccg-work"
    area.mkdir()
    res = ir.run_indexer(repo, unit, [unit], sha, area / "w", ir.Provisioned(op), image=ir.indexer_image_tag(), area=area)
    assert isinstance(res, ir.IndexerRun) and res.duration_ms is not None
    ref = scip.parse_scip(res.scip).refs[("b.ts", 2, 39)]
    assert [(t.file, t.chain, t.callable) for t in ref] == [("a.ts", ("helper",), True)]
    (rec,) = [json.loads(l) for l in (tmp_path / "audit" / "sandbox.jsonl").read_text().splitlines()]
    assert rec["auditId"] == res.audit_id and rec["exitCode"] == 0 and rec["egressMode"] == "offline"
    assert rec["toolArgv"][:3] == ["index", "--cwd", "/src"]
    assert subprocess.run(["docker", "ps", "-aq", "--filter", "label=ccg.sandbox.auditId=" + res.audit_id],
                          capture_output=True, text=True).stdout == ""


# --- the review fixes: bounded output, group sweep, no orphan, grammar first, id check, caps, load_provisioned ------------


def test_a_flooding_runner_costs_the_parent_a_bounded_buffer(tmp_path, stub, area):
    import tracemalloc
    set_mode(stub, "flood")
    tracemalloc.start()
    res, _ = go(tmp_path, stub, area, "javascript")
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert isinstance(res, ir.IndexerRun) and peak < 8 * 1024 * 1024
    log = (area / "w" / "indexer" / "indexer.log").read_bytes()
    assert len(log) <= 4 * ir.OUTPUT_KEEP_BYTES + 1024 and b"bytes dropped" in log
    assert log.count(b"x") <= 2 * ir.OUTPUT_KEEP_BYTES + 100 and log.count(b"e") <= 2 * ir.OUTPUT_KEEP_BYTES + 100


def test_capped_reader_keeps_head_and_tail_and_counts_the_rest():
    import io
    for total, keep, dropped in ((3, 4, False), (4, 4, False), (8, 4, False), (9, 4, True), (1000, 4, True)):
        data = bytes(range(256)) * 4
        raw = data[:total]
        c = ir._Capped(io.BufferedReader(io.BytesIO(raw)), keep)
        c.run()
        out = c.result()
        assert c.total == total
        if not dropped:
            assert out == raw
        else:
            assert out.startswith(raw[:keep]) and out.endswith(raw[-keep:]) and f"{total - 2 * keep} bytes dropped".encode() in out


def test_the_process_group_is_swept_even_when_the_runner_leaves_on_term(tmp_path, stub, area):
    set_mode(stub, "leaver")
    with pytest.raises(ir.RunnerTimeout):
        go(tmp_path, stub, area, "javascript", timeouts=ir.Timeouts(wall_s=2, outer_s=1, kill_grace_s=5))
    time.sleep(0.3)
    assert not alive(int((stub.parent / "child").read_text()))


def test_an_exception_in_the_parent_does_not_orphan_the_runner(tmp_path, stub, area, monkeypatch):
    base = subprocess.Popen

    class Boom(base):
        raised = False

        def wait(self, timeout=None):
            if not Boom.raised and str(self.args[0]) == str(stub):
                Boom.raised = True
                time.sleep(0.5)                                  # let the stub start and spawn its child
                raise KeyboardInterrupt
            return super().wait(timeout)

    set_mode(stub, "wedge")
    monkeypatch.setattr(ir.subprocess, "Popen", Boom)
    with pytest.raises(KeyboardInterrupt):
        go(tmp_path, stub, area, "javascript", timeouts=ir.Timeouts(wall_s=2, outer_s=30, kill_grace_s=1))
    time.sleep(0.3)
    assert not alive(int((stub.parent / "pid").read_text())) and not alive(int((stub.parent / "child").read_text()))


@pytest.mark.parametrize("env_json,lockfile", [("../env.json", None), ("/etc/passwd", None), ("a b.json", None), ("a//b", None),
                                               ("env.json", "../../etc/passwd"), ("env.json", "/etc/shadow"), ("env.json", "")])
def test_provisioned_names_are_checked_against_the_grammar_before_anything_is_read(tmp_path, stub, area, env_json, lockfile):
    prov = provisioned(tmp_path, "python")
    with pytest.raises(ValueError, match="grammar"):
        go(tmp_path, stub, area, "python", prov=ir.Provisioned(prov.root, env_json=env_json, lockfile=lockfile))
    assert runner_calls(stub) == [] and not (area / "w").exists()


def test_a_scip_not_named_by_a_runner_audit_id_is_refused(tmp_path, stub, area):
    set_mode(stub, "badname")
    with pytest.raises(RunnerError, match="audit id"):
        go(tmp_path, stub, area, "javascript")


def test_the_provisioned_copy_is_bounded_in_entries_and_bytes(tmp_path, stub, area, monkeypatch):
    repo, sha, unit = unit_repo(tmp_path, "javascript")
    prov = provisioned(tmp_path, "javascript")

    def attempt():
        return ir.run_indexer(repo, unit, [unit], sha, area / "w", prov, image=IMAGE, area=area, runner=stub, timeouts=FAST)

    monkeypatch.setattr(ir, "MAX_PROVISIONED_ENTRIES", 3)
    with pytest.raises(ir.StageRejected, match="entries"):
        attempt()
    monkeypatch.setattr(ir, "MAX_PROVISIONED_ENTRIES", 10_000)
    monkeypatch.setattr(ir, "MAX_PROVISIONED_BYTES", 1)
    with pytest.raises(ir.StageRejected, match="bytes"):
        attempt()
    assert runner_calls(stub) == []


def test_validate_tree_caps_at_the_boundary(tmp_path):
    t = tmp_path / "t"
    t.mkdir()
    (t / "a").write_text("12345")
    (t / "b").write_text("123")
    ir.validate_tree(t, max_entries=2, max_bytes=8)
    with pytest.raises(ir.StageRejected, match="entries"):
        ir.validate_tree(t, max_entries=1)
    with pytest.raises(ir.StageRejected, match="bytes"):
        ir.validate_tree(t, max_bytes=7)


@pytest.fixture()
def provdir(tmp_path, monkeypatch):
    base = tmp_path / "provision"
    monkeypatch.setenv("CCG_PROVISION_DIR", str(base))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return base


def test_an_absent_provision_directory_is_not_provisioned_and_nothing_is_created(provdir, tmp_path):
    assert ir.load_provisioned("sys1", "u1") is None
    assert not provdir.exists() and not (tmp_path / "home").exists()


def test_a_provision_directory_loads_with_defaults(provdir):
    (provdir / "sys1" / "u1").mkdir(parents=True)
    assert ir.load_provisioned("sys1", "u1") == ir.Provisioned(provdir / "sys1" / "u1", env_json="env.json", lockfile=None)


def test_provision_json_names_the_env_file_and_the_lockfile(provdir):
    d = provdir / "sys1" / "u1"
    d.mkdir(parents=True)
    (d / "provision.json").write_text('{"envJson": "deps/e.json", "lockfile": "package-lock.json"}')
    assert ir.load_provisioned("sys1", "u1") == ir.Provisioned(d, env_json="deps/e.json", lockfile="package-lock.json")


@pytest.mark.parametrize("text", ["not json", "[]", '{"envJson": 1}', '{"other": "x"}', '{"envJson": "../x"}', '{"lockfile": "/etc/passwd"}'])
def test_a_bad_provision_json_is_refused(provdir, text):
    d = provdir / "sys1" / "u1"
    d.mkdir(parents=True)
    (d / "provision.json").write_text(text)
    with pytest.raises(ValueError):
        ir.load_provisioned("sys1", "u1")


def test_an_oversized_or_symlinked_provision_json_is_refused(provdir, tmp_path):
    d = provdir / "sys1" / "u1"
    d.mkdir(parents=True)
    (d / "provision.json").write_text(" " * 70000)
    with pytest.raises(ValueError, match="64 KiB"):
        ir.load_provisioned("sys1", "u1")
    (d / "provision.json").unlink()
    (tmp_path / "real.json").write_text("{}")
    os.symlink(tmp_path / "real.json", d / "provision.json")
    with pytest.raises(ValueError, match="symlink"):
        ir.load_provisioned("sys1", "u1")


@pytest.mark.parametrize("system,unit", [("..", "u"), ("s", ".."), ("s/x", "u"), ("s", "u/../v"), ("", "u"), ("s", ""), (".s", "u"), ("s", "u\n")])
def test_system_and_unit_id_must_be_single_safe_components(provdir, system, unit):
    with pytest.raises(ValueError, match="single safe"):
        ir.load_provisioned(system, unit)


def test_a_symlinked_unit_directory_is_not_provisioned(provdir, tmp_path):
    (provdir / "sys1").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", provdir / "sys1" / "u1")
    assert ir.load_provisioned("sys1", "u1") is None


def test_the_default_location_and_the_override_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CCG_PROVISION_DIR", raising=False)
    assert ir.provision_base() == tmp_path / "home" / ".local" / "share" / "ccg" / "provision"
    monkeypatch.setenv("CCG_PROVISION_DIR", "relative/dir")
    with pytest.raises(ValueError, match="absolute"):
        ir.provision_base()


@pytest.mark.parametrize("sub", ["", "provision"])
def test_a_location_under_the_key_directory_is_refused_before_any_filesystem_access(tmp_path, monkeypatch, sub):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CCG_PROVISION_DIR", str(home / ".config" / "ccg" / sub) if sub else str(home / ".config"))
    if sub == "":
        monkeypatch.setenv("CCG_PROVISION_DIR", str(home / ".config"))
        (home / ".config").mkdir(parents=True)
        with pytest.raises(ValueError, match="never read"):
            ir.load_provisioned("ccg", "u1")                    # <home>/.config/ccg/u1
        return
    with pytest.raises(ValueError, match="never read"):
        ir.load_provisioned("sys1", "u1")                       # nothing under home exists: a stat would answer None
    assert not home.exists()


# --- the second review: a leaky descendant, the key-directory guard by every route, the shipped defaults ------------------


def test_a_descendant_holding_the_pipes_after_a_clean_exit_cannot_hang_the_call(tmp_path, stub, area):
    set_mode(stub, "leaky")
    t0 = time.monotonic()
    res, _ = go(tmp_path, stub, area, "javascript")
    assert isinstance(res, ir.IndexerRun) and time.monotonic() - t0 < 3
    time.sleep(0.3)
    assert not alive(int((stub.parent / "child").read_text()))


def test_a_reader_that_is_still_alive_is_never_closed_under(tmp_path, monkeypatch):
    """A descendant outside the group (not killed by the sweep) keeps the pipe: the call returns after the bounded join
    and does not close the pipe a live reader is blocked on."""
    script = tmp_path / "r"
    script.write_text("#!/bin/sh\nsetsid sleep 3 &\nexit 0\n")
    script.chmod(0o755)
    orig = ir._Capped.join
    monkeypatch.setattr(ir._Capped, "join", lambda self, timeout=None: orig(self, 0.3))
    t0 = time.monotonic()
    c = ir._run([str(script)], {"PATH": os.environ["PATH"]}, ir.Timeouts(outer_s=10, kill_grace_s=1))
    assert c.rc == 0 and time.monotonic() - t0 < 2.5


@pytest.fixture()
def keyhome(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".config" / "ccg").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CCG_PROVISION_DIR", raising=False)
    return home


def test_a_dotdot_in_the_provision_base_is_refused(keyhome, monkeypatch):
    monkeypatch.setenv("CCG_PROVISION_DIR", str(keyhome / "other" / ".." / ".config" / "ccg"))
    with pytest.raises(ValueError, match="'..' component"):
        ir.load_provisioned("sys1", "u1")


def test_a_symlinked_provision_base_into_the_key_directory_is_refused(keyhome, tmp_path, monkeypatch):
    os.symlink(keyhome / ".config" / "ccg", tmp_path / "link")
    monkeypatch.setenv("CCG_PROVISION_DIR", str(tmp_path / "link"))
    (keyhome / ".config" / "ccg" / "sys1" / "u1").mkdir(parents=True)
    with pytest.raises(ValueError, match="never read"):
        ir.load_provisioned("sys1", "u1")


def test_a_symlinked_system_directory_into_the_key_directory_is_refused(keyhome, tmp_path, monkeypatch):
    base = tmp_path / "provision"
    base.mkdir()
    os.symlink(keyhome / ".config" / "ccg", base / "sys1")
    (keyhome / ".config" / "ccg" / "u1").mkdir()
    monkeypatch.setenv("CCG_PROVISION_DIR", str(base))
    with pytest.raises(ValueError, match="never read"):
        ir.load_provisioned("sys1", "u1")


def test_a_symlinked_home_does_not_hide_the_key_directory(tmp_path, monkeypatch):
    real = tmp_path / "real"
    (real / ".config" / "ccg").mkdir(parents=True)
    os.symlink(real, tmp_path / "homelink")
    monkeypatch.setenv("HOME", str(tmp_path / "homelink"))
    monkeypatch.setenv("CCG_PROVISION_DIR", str(real / ".config" / "ccg"))        # named through the real path
    with pytest.raises(ValueError, match="never read"):
        ir.load_provisioned("sys1", "u1")


def test_a_fine_base_beside_the_key_directory_is_not_refused(keyhome, tmp_path, monkeypatch):
    monkeypatch.setenv("CCG_PROVISION_DIR", str(keyhome / ".config" / "ccg-other"))
    assert ir.load_provisioned("sys1", "u1") is None


def test_provision_json_is_not_copied_into_the_staged_directory(tmp_path, stub, area):
    prov = provisioned(tmp_path, "javascript")
    (prov.root / "provision.json").write_text("{}")
    res, _ = go(tmp_path, stub, area, "javascript", prov=prov)
    assert isinstance(res, ir.IndexerRun) and not (res.stage / "provision.json").exists() and (res.stage / "node_modules").is_dir()


def test_the_shipped_provisioned_tree_bounds_are_wired_and_refuse_just_over_the_byte_bound(tmp_path, stub, area, monkeypatch):
    assert (ir.MAX_PROVISIONED_ENTRIES, ir.MAX_PROVISIONED_BYTES) == (500_000, 8 * 1024 ** 3)
    seen = []
    real = ir.validate_tree
    monkeypatch.setattr(ir, "validate_tree", lambda root, **kw: (seen.append(kw), real(root, **kw))[1])
    prov = provisioned(tmp_path, "javascript")
    go(tmp_path, stub, area, "javascript", prov=prov)
    assert {"max_entries": 500_000, "max_bytes": 8 * 1024 ** 3} in seen
    with open(prov.root / "node_modules" / "big", "wb") as f:
        f.truncate(8 * 1024 ** 3 + 1)                                              # sparse
    with pytest.raises(ir.StageRejected, match="bytes"):
        ir.validate_tree(prov.root, max_entries=ir.MAX_PROVISIONED_ENTRIES, max_bytes=ir.MAX_PROVISIONED_BYTES)
