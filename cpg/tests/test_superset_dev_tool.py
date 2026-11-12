"""Offline proof of ``cpg/eval/superset_dev.py`` (P1-39c): one planted input per refusal of ``tool.refuseList``, each exiting non-zero before any walk.

The record, the pinned files and the tool files live in a throwaway git repository built per test (never the real repository), so a changed byte,
an uncommitted line or a moved HEAD is planted for real. The live pipeline (build, load, digest queries) is injected; its live proof is
``tests/test_engine_superset_dev.py``.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CPG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPG / "eval"))
sd = importlib.import_module("superset_dev")

ENGINE = {"cpg/ccg/engine/answer_build.py": b"ab\n", "cpg/ccg/engine/bfs.py": b"bfs\n", "cpg/ccg/engine/flow.py": b"flow\n"}
BUILDER = {"cpg/ccg/build.py": b"build\n", "cpg/extractors/joern/extract.sc": b"sc\n"}
PLANS = {"docs/plans/connected-code-graph-p1-ml.md": b"ml\n", "docs/plans/connected-code-graph-answer-spec2.md": b"spec2\n"}
REGISTRY = {"cpg/ccg/registry/flow-availability.yaml": b"reg\n"}
BFS_TEST = {"cpg/tests/test_engine_bfs_rows.py": b"rows\n"}
TOOLS = {"cpg/eval/superset_dev.py": b"tool\n", "cpg/tests/test_engine_superset_dev.py": b"live\n", "cpg/tests/test_superset_dev_tool.py": b"offline\n"}
EXTRA = {"cpg/extractors/common/errors.py": b"errors\n", "cpg/packs/fastapi/pack.yaml": b"pack\n", "cpg/packs/infra-signatures/negatives.yaml": b"neg\n"}
FILE_GROUPS = {"engine": ENGINE, "builder": BUILDER, "plan": PLANS, "registry": REGISTRY, "bfsrows": BFS_TEST}
UNITS = ["U1", "U2"]
EVAL_SHA_PLACEHOLDER = "0" * 40
ENV_KEYS = {"head", "evalCommit", "toolSha256", "python", "images", "loadAverageBefore", "loadAverageAfter", "pinnedFilesVerified", "extraPinsVerified", "digestAlgorithm",
            "digestDescription", "stageClaim", "system"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dev_rows():
    """7 items under 4 anchors; impacted files .py/.ts/.tsx so every language cell exists."""
    rows = []
    for a, (qn, n) in enumerate([("a0", 2), ("a1", 2), ("a2", 2), ("a3", 1)]):
        for k in range(n):
            ext = (".py", ".ts", ".tsx")[(a + k) % 3]
            rows.append({"itemId": f"item{a}{k}", "anchorId": f"anchor{a}", "impactedId": f"imp{a}{k}", "counting": False,
                         "anchor": {"repoId": "R", "path": f"src/a{a}.ts", "qname": f"src/a{a}.ts::{qn}", "type": "symbol"},
                         "impacted": {"repoId": "R", "path": f"src/i{a}{k}{ext}", "qname": "x", "type": "symbol"},
                         "chain": [{"from": "x", "kind": "callee-arg", "to": "y"}] + ([{"from": "y", "kind": "return-use", "to": "z"}] if k else [])})
    return rows


class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid", "PATH": "/usr/bin:/bin", "HOME": str(root)}
        root.mkdir()
        self.git("init", "-q", "-b", "main")

    def git(self, *a):
        return subprocess.run(["git", "-C", str(self.root), *a], check=True, capture_output=True, text=True, env=self.env).stdout.strip()

    def write(self, rel: str, data: bytes):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def commit(self, msg="c"):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg, "--allow-empty")
        return self.git("rev-parse", "HEAD")

    def lines(self):
        return (self.root / sd.RECORD_PATH).read_bytes().split(b"\n")[:-1]

    def append(self, obj: dict):
        with open(self.root / sd.RECORD_PATH, "ab") as fh:
            fh.write(json.dumps(obj, sort_keys=True).encode() + b"\n")


def record_line(files: dict[str, bytes], eval_sha: str, dev_bytes: bytes) -> dict:
    return {"id": sd.RECORD_ID, "pins": {
        "evalCommit": eval_sha,
        "dev": {"cpg/eval/dev/ac5-flow.jsonl": sha(dev_bytes), "items": 7, "anchorClusters": 4},
        "engine": {"cpg/ccg/engine/answer_build.py": sha(ENGINE["cpg/ccg/engine/answer_build.py"]), "bfs.py": sha(ENGINE["cpg/ccg/engine/bfs.py"]),
                   "flow.py": sha(ENGINE["cpg/ccg/engine/flow.py"])},
        "flagRegistry": {**{k: sha(v) for k, v in REGISTRY.items()}, "note": "x"},
        "plans": {k: sha(v) for k, v in PLANS.items()},
        "corpusAndBuilder": {"builderFiles": {k: sha(v) for k, v in BUILDER.items()}}},
        "retLimit": {"pins": {k: sha(v) for k, v in BFS_TEST.items()}}}


@pytest.fixture(autouse=True)
def _restore_env():
    saved = dict(__import__("os").environ)
    yield
    __import__("os").environ.clear()
    __import__("os").environ.update(saved)


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "EXTRA_PIN_FILES", tuple(EXTRA))
    r = Repo(tmp_path / "repo")
    r.write("README", b"x\n")
    eval_sha = r.commit("base")
    dev_bytes = b"".join(json.dumps(d, sort_keys=True).encode() + b"\n" for d in dev_rows())
    for group in FILE_GROUPS.values():
        for k, v in group.items():
            r.write(k, v)
    for k, v in {**TOOLS, **EXTRA}.items():
        r.write(k, v)
    r.write("cpg/eval/dev/ac5-flow.jsonl", dev_bytes)
    r.write(sd.EVAL_COMMIT_PATH, eval_sha.encode() + b"\n")
    r.write(sd.RECORD_PATH, b"")
    r.append({"id": "ccg-p1-flow-python-v1", "x": 1})
    r.append(record_line({}, eval_sha, dev_bytes))
    r.commit("record")
    r.eval_sha, r.dev_bytes = eval_sha, dev_bytes
    r.tool_line = {"id": sd.TOOL_ID, "sha256": {k: sha(v) for k, v in TOOLS.items()}, "extraPins": {k: sha(v) for k, v in EXTRA.items()}}
    r.append(r.tool_line)
    r.commit("tool line")
    return r


class FakeReader:
    def __init__(self, eval_sha, *, missing=(), status="fresh", unit_commit=None, files=1, flow=False, absent_units=()):
        self.files, self.flow, self.absent_units = files, flow, set(absent_units)
        self.eval_sha, self.missing, self.status, self.unit_commit = eval_sha, set(missing), status, unit_commit or eval_sha
        self.dev = dev_rows()

    def read(self, name, params=None):
        if name in ("symbols_of_file", "file_by_path"):
            assert set(params) == {"repo", "path"} and params["repo"] == "R"
        if name == "symbols_of_file":
            out = []
            for d in self.dev:
                for sid, side in ((d["anchorId"], "anchor"), (d["impactedId"], "impacted")):
                    if d[side]["path"] == params["path"] and sid not in self.missing:
                        out.append((sid, "q", 1, 2, None, 0))
            return out
        if name == "file_by_path":
            return [("f", "U1")] * self.files
        if name == "units":
            return [(u, "R", ".", "javascript", self.status, self.unit_commit, "t", "failed-at", "why", self.flow) for u in params["ids"] if u not in self.absent_units]
        if name == "snapshot":
            return [("snap1", 0, "h", "1.1", "1.1")]
        raise AssertionError(name)


class FakeGraph:
    def __init__(self, eval_sha, **kw):
        self.reader = FakeReader(eval_sha, **kw)
        self.x = 1

    def ro(self, text, params):
        if text.startswith("MATCH (n:Repo)"):
            return [({"repoId": "R", "x": self.x},)]
        if text == "MATCH (n) RETURN count(n)":
            return [(1,)]
        if text == "MATCH ()-[r]->() RETURN count(r)":
            return [(0,)]
        return []


class Spy:
    """The deps: a docker that publishes the throwaway port, a pipeline, a graph, images, and a walk; each call is counted."""

    def __init__(self, fx, graph=None):
        self.fx, self.pipeline_calls, self.walk_calls = fx, 0, 0
        self.failed: dict[str, str] = {}                                          # unit -> the build's failReason (that unit's build is not ok)
        self.hidden: dict[str, set[str]] = {}                                    # candidate table -> impacted ids its walk does not list
        self.graph = graph or FakeGraph(fx.eval_sha)
        self.images = {"toolchainImageId": "sha256:tc", "falkordbImageDigest": "sha256:fdb [x]"}
        self.dev = dev_rows()

    def build_and_load(self, ctx):
        self.pipeline_calls += 1
        return {"snapshotId": "snap1", "builds": {u: {"ok": u not in self.failed, "failReason": self.failed.get(u)} for u in UNITS}}

    def walk(self, reader, anchor, *, column, table, **kw):
        self.walk_calls += 1
        ids = [d["impactedId"] for d in self.dev if d["anchorId"] == anchor[1] and d["impactedId"] not in self.hidden.get(table, ())]
        items = [SimpleNamespace(kind="symbol", id=i, distance=1, possible_hops=1, length=1, chain=[]) for i in ids]
        return SimpleNamespace(items=lambda: items, bound=2)

    def deps(self):
        return sd.Deps(docker_ps=lambda: "127.0.0.1:16400->6379/tcp", build_and_load=self.build_and_load, open_graph=lambda ctx: self.graph,
                       images=lambda ctx: self.images, walk=self.walk, tables=lambda n: n, impact="impact", loadavg=lambda: (0.1, 0.1, 0.1))


def env_for(tmp_path):
    cfg = tmp_path / "scratch-cfg"
    cfg.mkdir(exist_ok=True)
    return {"CCG_FALKORDB_PORT": "16400", "CCG_CONFIG_DIR": str(cfg)}


def run(fx, spy, phase, tmp_path, env=None):
    out, err = io.StringIO(), io.StringIO()
    code = sd.main([phase], deps=spy.deps(), env=env or env_for(tmp_path), root=fx.root, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def snapshot_line(phase_a_output: dict) -> dict:
    return {"id": sd.SNAPSHOT_ID, **{k: phase_a_output[k] for k in sd.SNAPSHOT_KEYS}}


def commit_snapshot(fx, spy, tmp_path, **override):
    code, out, err = run(fx, spy, "A", tmp_path)
    assert code == 0, err
    line = {**snapshot_line(json.loads(out)), **override}
    fx.append(line)
    fx.commit("snapshot line")
    return line


# ------------------------------------------------------------------------------------------------------------ the happy path (so a refusal is not vacuous)

def test_phase_a_prints_the_snapshot_facts_runs_no_walk_and_writes_no_result(fx, tmp_path):
    spy = Spy(fx)
    code, out, err = run(fx, spy, "A", tmp_path)
    assert code == 0, err
    got = json.loads(out)
    assert (got["snapshotCommit"], got["unitCommits"], spy.walk_calls, spy.pipeline_calls) == (fx.eval_sha, {u: fx.eval_sha for u in UNITS}, 0, 1)
    assert set(sd.SNAPSHOT_KEYS) <= set(got) and got["digestAlgorithm"] == sd.DIGEST_ALGORITHM and got["environment"]["stageClaim"] == sd.STAGE_CLAIM
    assert got["unitStatus"]["U1"]["buildStatus"] == "fresh"
    assert not (fx.root / "cpg/eval/results").exists()


def test_a_built_unit_with_no_row_in_the_graph_is_reported_with_null_graph_fields(fx, tmp_path):
    spy = Spy(fx, FakeGraph(fx.eval_sha, absent_units={"U2"}))                      # U2 hosts no dev node; the graph has no Unit row for it
    code, out, err = run(fx, spy, "A", tmp_path)
    assert code == 0, err
    got = json.loads(out)
    assert got["unitStatus"]["U2"] == {"buildStatus": None, "builtCommit": None, "failReason": None, "ok": True, "buildFailReason": None}
    assert got["unitCommits"] == {"U1": fx.eval_sha, "U2": None}


def test_phase_b_without_injected_walk_dependencies_runs_the_engines_walk_with_the_impact_column_and_the_candidate_tables(fx, tmp_path, monkeypatch):
    from ccg.engine import bfs, flow
    spy = Spy(fx)
    commit_snapshot(fx, spy, tmp_path)
    seen = []

    def engine_walk(reader, anchor, **kw):
        seen.append((anchor[0], kw["column"], kw["table"]))
        return spy.walk(reader, anchor, **kw)
    monkeypatch.setattr(bfs, "walk", engine_walk)
    deps = spy.deps()
    deps.walk = deps.tables = deps.impact = None
    out, err = io.StringIO(), io.StringIO()
    assert sd.main(["B"], deps=deps, env=env_for(tmp_path), root=fx.root, out=out, err=err) == 0, err.getvalue()
    assert {(a, c) for a, c, _ in seen} == {("Symbol", bfs.IMPACT)} and [t for _, _, t in seen][:2] == [flow.candidate_table("as-built")] * 2
    assert {tuple(t) for _, _, t in seen} == {tuple(flow.candidate_table(n)) for n in sd.CANDIDATES}


def test_phase_a_prints_every_snapshot_key_with_the_value_its_source_gives(fx, tmp_path):
    spy = Spy(fx)
    spy.failed = {"U2": "extract:failed"}
    code, out, err = run(fx, spy, "A", tmp_path)
    assert code == 0, err
    got = json.loads(out)
    assert (got["snapshotId"], got["toolchainImageId"], got["falkordbImageDigest"]) == ("snap1", "sha256:tc", "sha256:fdb [x]")
    assert got["graphCounts"] == {"nodes": 1, "edges": 0} and got["digestAlgorithm"] == sd.DIGEST_ALGORITHM and len(got["graphDigest"]) == 64
    assert got["unitStatus"] == {"U1": {"buildStatus": "fresh", "builtCommit": fx.eval_sha, "failReason": "why", "ok": True, "buildFailReason": None},
                                 "U2": {"buildStatus": "fresh", "builtCommit": fx.eval_sha, "failReason": "why", "ok": False, "buildFailReason": "extract:failed"}}
    assert set(got) == {"phase", "snapshotCommit", "snapshotId", "unitCommits", "unitStatus", "graphDigest", "digestAlgorithm", "graphCounts", "toolchainImageId",
                        "falkordbImageDigest", "environment", "next"} and got["phase"] == "A"
    assert set(got["environment"]) == ENV_KEYS and got["environment"]["system"] == sd.SYSTEM and got["environment"]["digestDescription"] == sd.DIGEST_DESCRIPTION
    assert (got["environment"]["pinnedFilesVerified"], got["environment"]["extraPinsVerified"]) == (len(sd.pinned_files(json.loads(fx.lines()[1]))), len(EXTRA))
    assert got["environment"]["images"] == spy.images and got["environment"]["loadAverageBefore"] == [0.1, 0.1, 0.1] == got["environment"]["loadAverageAfter"]
    assert got["environment"]["head"] == fx.git("rev-parse", "HEAD") and got["environment"]["evalCommit"] == fx.eval_sha and got["environment"]["python"]
    assert sd.SNAPSHOT_ID in got["next"] and sd.RECORD_PATH in got["next"]


def test_phase_b_measures_after_a_matching_snapshot_line_and_writes_only_under_results(fx, tmp_path):
    spy = Spy(fx)
    commit_snapshot(fx, spy, tmp_path)
    spy.pipeline_calls = 0
    code, out, err = run(fx, spy, "B", tmp_path)
    assert code == 0, err
    res = json.loads((fx.root / sd.RESULT_PATH).read_text())
    assert (spy.pipeline_calls, spy.walk_calls) == (0, 4 * 4 * 2)                  # phase B never builds; 4 tables x 4 anchors x 2 runs
    assert res["candidates"]["as-built"]["pooled"]["listed"] == 7 and res["environment"]["head"] == fx.git("rev-parse", "HEAD")
    assert sorted(p.name for p in (fx.root / "cpg/eval/results").iterdir()) == ["ccg-p1-flow-fallback-v1.json"]


def test_both_phases_point_credentials_and_audit_at_the_throwaways_own_config_dir(fx, tmp_path):
    import os
    env = env_for(tmp_path)
    assert run(fx, Spy(fx), "A", tmp_path, env)[0] == 0
    assert os.environ["CCG_CREDENTIALS_DIR"] == str(Path(env["CCG_CONFIG_DIR"]) / "credentials")
    assert os.environ["CCG_AUDIT_DIR"] == str(Path(env["CCG_CONFIG_DIR"]) / "audit")


def test_the_result_file_carries_the_unlisted_ids_of_each_walk_and_its_fixed_fields(fx, tmp_path):
    spy = Spy(fx)
    spy.hidden = {"V-flow": {"imp21"}, "as-built": {"imp00", "imp21"}}
    commit_snapshot(fx, spy, tmp_path)
    code, out, err = run(fx, spy, "B", tmp_path)
    assert code == 0, err
    res = json.loads((fx.root / sd.RESULT_PATH).read_text())
    assert res["unlistedPerWalk"] == {"as-built": ["item00", "item21"], "V-flow": ["item21"], "V-both": [], "V-union": []}
    assert res["candidates"]["V-flow"]["listedIds"] == [i for i in sorted(d["itemId"] for d in dev_rows()) if i != "item21"]
    assert (res["recordId"], res["phase"], res["depthBound"], res["items"], res["anchorClusters"]) == (sd.RECORD_ID, "B", 2, 7, 4)
    assert "held-out" in res["inSample"] and "(end, callee)" in res["secondLevelCalleeOnlyDefinition"]
    assert json.loads(out) == {"result": str(fx.root / sd.RESULT_PATH), "rule": res["rule"]}
    assert set(res) == {"recordId", "phase", "depthBound", "items", "anchorClusters", "unlistedPerWalk", "secondLevelCalleeOnlyDefinition", "inSample", "candidates",
                        "setUnionLowerBound", "asBuiltComparison", "rule", "environment"}
    text = (fx.root / sd.RESULT_PATH).read_text()
    assert text == json.dumps(res, indent=1, sort_keys=True) + "\n" and set(res["environment"]) == ENV_KEYS
    env = res["environment"]
    assert (env["digestAlgorithm"], env["system"], env["toolSha256"]) == (sd.DIGEST_ALGORITHM, sd.SYSTEM, fx.tool_line["sha256"]) and env["images"] == spy.images
    assert (env["pinnedFilesVerified"], env["extraPinsVerified"]) == (len(sd.pinned_files(json.loads(fx.lines()[1]))), len(EXTRA))


def test_phase_b_never_overwrites_a_result(fx, tmp_path):
    spy = Spy(fx)
    commit_snapshot(fx, spy, tmp_path)
    assert run(fx, spy, "B", tmp_path)[0] == 0
    code, _, err = run(fx, spy, "B", tmp_path)
    assert code == 2 and "overwrite" in err


# ------------------------------------------------------------------------------------------------------------ refusals common to both phases

def _flip(fx, rel):
    p = fx.root / rel
    data = bytearray(p.read_bytes())
    data[0] ^= 1
    p.write_bytes(bytes(data))


PINNED = [(g, k) for g, d in FILE_GROUPS.items() for k in d] + [("dev", "cpg/eval/dev/ac5-flow.jsonl")]


@pytest.mark.parametrize("phase", ["A", "B"])
@pytest.mark.parametrize("group,rel", PINNED, ids=[f"{g}:{k.rsplit('/', 1)[-1]}" for g, k in PINNED])
def test_one_changed_byte_in_any_pinned_file_refuses_before_any_build_or_walk(fx, tmp_path, phase, group, rel):
    spy = Spy(fx)
    if phase == "B":
        commit_snapshot(fx, spy, tmp_path)
    spy.pipeline_calls = spy.walk_calls = 0
    _flip(fx, rel)
    code, _, err = run(fx, spy, phase, tmp_path)
    assert code == 2 and "pinned file" in err and (spy.pipeline_calls, spy.walk_calls) == (0, 0)


@pytest.mark.parametrize("phase", ["A", "B"])
def test_a_one_byte_change_of_eval_commit_refuses_it_equals_the_pin(fx, tmp_path, phase):
    spy = Spy(fx)
    if phase == "B":
        commit_snapshot(fx, spy, tmp_path)
    spy.pipeline_calls = spy.walk_calls = 0
    p = fx.root / sd.EVAL_COMMIT_PATH
    p.write_text(p.read_text().replace(fx.eval_sha[0], "f" if fx.eval_sha[0] != "f" else "e", 1))
    code, _, err = run(fx, spy, phase, tmp_path)
    assert code == 2 and "EVAL_COMMIT" in err and (spy.pipeline_calls, spy.walk_calls) == (0, 0)


def test_a_mismatched_tool_sha_refuses(fx, tmp_path):
    (fx.root / "cpg/eval/superset_dev.py").write_bytes(b"tool changed\n")
    spy = Spy(fx)
    code, _, err = run(fx, spy, "A", tmp_path)
    assert code == 2 and "differs from the" in err and spy.pipeline_calls == 0


def test_the_live_test_file_is_pinned_by_the_tool_line_too(fx, tmp_path):
    (fx.root / "cpg/tests/test_engine_superset_dev.py").write_bytes(b"live changed\n")
    code, _, err = run(fx, Spy(fx), "A", tmp_path)
    assert code == 2 and "test_engine_superset_dev.py differs" in err


def test_the_offline_test_file_is_pinned_by_the_tool_line_too(fx, tmp_path):
    (fx.root / "cpg/tests/test_superset_dev_tool.py").write_bytes(b"offline changed\n")
    code, _, err = run(fx, Spy(fx), "A", tmp_path)
    assert code == 2 and "test_superset_dev_tool.py differs" in err


@pytest.mark.parametrize("phase", ["A", "B"])
@pytest.mark.parametrize("rel", list(EXTRA), ids=[k.rsplit("/", 1)[-1] for k in EXTRA])
def test_one_changed_byte_in_any_extra_pinned_file_refuses_before_any_build_or_walk(fx, tmp_path, phase, rel):
    """The files the build imports or reads that the record does not pin: carried by the ``-tool`` line's ``extraPins``."""
    spy = Spy(fx)
    if phase == "B":
        commit_snapshot(fx, spy, tmp_path)
    spy.pipeline_calls = spy.walk_calls = 0
    _flip(fx, rel)
    code, _, err = run(fx, spy, phase, tmp_path)
    assert code == 2 and f"{rel} differs from the {sd.TOOL_ID} line" in err and (spy.pipeline_calls, spy.walk_calls) == (0, 0)


def test_a_missing_extra_pinned_file_refuses(fx, tmp_path):
    (fx.root / "cpg/packs/fastapi/pack.yaml").unlink()
    code, _, err = run(fx, Spy(fx), "A", tmp_path)
    assert code == 2 and "pack.yaml differs" in err


def test_the_tool_line_must_carry_exactly_the_tools_extra_pins(fx):
    good = fx.tool_line
    drop = {**good, "extraPins": {k: v for k, v in list(good["extraPins"].items())[1:]}}
    more = {**good, "extraPins": {**good["extraPins"], "cpg/extractors/other.py": "0" * 64}}
    no_key = {k: v for k, v in good.items() if k != "extraPins"}
    bad_sha = {**good, "extraPins": {**good["extraPins"], "cpg/packs/fastapi/pack.yaml": "xyz"}}
    for bad in (drop, more, no_key, {**good, "extraPins": []}, bad_sha):
        with pytest.raises(sd.Refusal, match="must pin exactly|not a sha256"):
            sd.check_tool_line(fx.root, [json.dumps(bad).encode()])


def test_an_uncommitted_tool_line_does_not_count(fx, tmp_path):
    fx.git("reset", "-q", "--hard", "HEAD~1")                                      # drop the committed -tool line ...
    fx.append(fx.tool_line)                                                         # ... and put it back uncommitted
    spy = Spy(fx)
    code, _, err = run(fx, spy, "A", tmp_path)
    assert code == 2 and sd.TOOL_ID in err and spy.pipeline_calls == 0


def test_a_missing_tool_line_refuses(fx, tmp_path):
    fx.git("reset", "-q", "--hard", "HEAD~1")
    code, _, err = run(fx, Spy(fx), "A", tmp_path)
    assert code == 2 and sd.TOOL_ID in err


@pytest.mark.parametrize("port,ps,cfg", [
    ("6379", "127.0.0.1:6379->6379/tcp", None), ("6390", "127.0.0.1:6390->6379/tcp", None),
    ("16401", "127.0.0.1:16400->6379/tcp", None),                                  # the port is not one a throwaway-labelled container publishes
    ("1640", "127.0.0.1:16400->6379/tcp", None),                                   # a prefix of a published port is not that port
    ("", "127.0.0.1:16400->6379/tcp", None), ("16400", "127.0.0.1:16400->6379/tcp", "REAL"), ("16400", "127.0.0.1:16400->6379/tcp", "UNSET")],
    ids=["6379", "6390", "not-throwaway", "port-prefix", "unset", "real-config-dir", "no-config-dir"])
def test_a_non_throwaway_port_or_credentials_dir_refuses(fx, tmp_path, port, ps, cfg):
    spy = Spy(fx)
    env = env_for(tmp_path)
    env["CCG_FALKORDB_PORT"] = port
    if cfg == "REAL":
        env["CCG_CONFIG_DIR"] = str(Path.home() / ".config" / "ccg")
    if cfg == "UNSET":
        del env["CCG_CONFIG_DIR"]
    deps = spy.deps()
    deps.docker_ps = lambda: ps
    out, err = io.StringIO(), io.StringIO()
    code = sd.main(["A"], deps=deps, env=env, root=fx.root, out=out, err=err)
    assert code == 2 and spy.pipeline_calls == 0 and spy.walk_calls == 0


def test_an_edited_earlier_line_is_not_append_only(fx, tmp_path):
    lines = fx.lines()
    lines[0] = lines[0].replace(b'"x": 1', b'"x": 2')
    (fx.root / sd.RECORD_PATH).write_bytes(b"\n".join(lines) + b"\n")
    fx.commit("rewrite line 1")
    code, _, err = run(fx, Spy(fx), "A", tmp_path)
    assert code == 2 and "append-only" in err


def test_pins_are_read_from_the_committed_blob_not_the_working_file(fx, tmp_path):
    lines = fx.lines()
    rec = json.loads(lines[1])
    rec["pins"]["engine"]["bfs.py"] = "0" * 64                                      # a working-tree edit that would move the bar
    lines[1] = json.dumps(rec, sort_keys=True).encode()
    (fx.root / sd.RECORD_PATH).write_bytes(b"\n".join(lines) + b"\n")
    code, _, err = run(fx, Spy(fx), "A", tmp_path)
    assert code == 0, err


def test_the_head_is_captured_once_and_blobs_are_read_at_that_sha_not_the_symbol(fx, tmp_path):
    early = sd.capture_head(fx.root)
    fx.git("reset", "-q", "--hard", "HEAD~1")                                       # HEAD now lacks the -tool line; the captured sha still has it
    fx.git("branch", "-f", "pinned", early)
    spy = Spy(fx)
    ctx = sd.preflight(fx.root, early, "A", env_for(tmp_path), spy.deps())
    assert ctx.head == early
    with pytest.raises(sd.Refusal, match=sd.TOOL_ID):
        sd.preflight(fx.root, sd.capture_head(fx.root), "A", env_for(tmp_path), spy.deps())


# ------------------------------------------------------------------------------------------------------------ phase B refusals

def test_phase_b_without_a_committed_snapshot_line_refuses(fx, tmp_path):
    code, _, err = run(fx, Spy(fx), "B", tmp_path)
    assert code == 2 and sd.SNAPSHOT_ID in err


def test_an_uncommitted_snapshot_line_does_not_count(fx, tmp_path):
    spy = Spy(fx)
    code, out, _ = run(fx, spy, "A", tmp_path)
    fx.append(snapshot_line(json.loads(out)))                                       # appended, never committed
    spy.walk_calls = 0
    code, _, err = run(fx, spy, "B", tmp_path)
    assert code == 2 and sd.SNAPSHOT_ID in err and spy.walk_calls == 0


def test_a_changed_graph_digest_between_phases_refuses(fx, tmp_path):
    spy = Spy(fx)
    commit_snapshot(fx, spy, tmp_path)
    spy.graph.x = 2                                                                  # the live graph changed after phase A
    code, _, err = run(fx, spy, "B", tmp_path)
    assert code == 2 and "digest" in err and spy.walk_calls == 0


@pytest.mark.parametrize("override,word", [
    ({"unitCommits": {"U1": "a" * 40, "U2": "a" * 40}}, "unit commits"),
    ({"unitStatus": {"U1": {"buildStatus": "stale-kept"}, "U2": {"buildStatus": "fresh"}}}, "build status"),
    ({"snapshotId": "other"}, "Snapshot"),
    ({"snapshotCommit": "b" * 40}, "commit"),
    ({"digestAlgorithm": "other"}, "algorithm"),
    ({"falkordbImageDigest": "sha256:other"}, "image")], ids=["unit-commits", "unit-status", "snapshot-id", "snapshot-commit", "algorithm", "image"])
def test_every_field_of_the_snapshot_line_is_compared_before_any_walk(fx, tmp_path, override, word):
    spy = Spy(fx)
    commit_snapshot(fx, spy, tmp_path, **override)
    code, _, err = run(fx, spy, "B", tmp_path)
    assert code == 2 and word in err and spy.walk_calls == 0


# ------------------------------------------------------------------------------------------------------------ the dev ids (both phases) and G3

@pytest.mark.parametrize("missing", ["anchor1", "imp21"])
@pytest.mark.parametrize("phase", ["A", "B"])
def test_a_dev_id_absent_from_the_graph_is_an_error_not_an_unlisted_item(fx, tmp_path, phase, missing):
    spy = Spy(fx)
    if phase == "B":
        commit_snapshot(fx, spy, tmp_path)
    spy.graph = FakeGraph(fx.eval_sha, missing={missing})
    spy.walk_calls = 0
    code, _, err = run(fx, spy, phase, tmp_path)
    assert code == 2 and "absent from the loaded graph" in err and spy.walk_calls == 0


def test_phase_a_refuses_when_a_unit_hosting_a_dev_node_did_not_build_fresh(fx, tmp_path):
    spy = Spy(fx, FakeGraph(fx.eval_sha, status="stale-kept"))
    code, out, err = run(fx, spy, "A", tmp_path)
    assert code == 2 and "did not build fresh" in err and out == ""


def test_phase_a_refuses_when_the_host_unit_was_built_at_another_commit(fx, tmp_path):
    spy = Spy(fx, FakeGraph(fx.eval_sha, unit_commit="c" * 40))
    code, _, err = run(fx, spy, "A", tmp_path)
    assert code == 2 and "did not build fresh at EVAL_COMMIT" in err


# ------------------------------------------------------------------------------------------------------------ G1

STAGE_OK = '''def stage_unit(repo):
    """never git archive"""
    argv = _git(repo, "cat-file", "--batch")
    return argv
'''


def test_the_real_stage_unit_reads_the_commit_tree_through_cat_file():
    assert "cat-file --batch" in sd.assert_stage_reads_the_commit()


@pytest.mark.parametrize("src", [STAGE_OK.replace('"cat-file", "--batch"', '"archive", "-o", "x"'), STAGE_OK + '    _git(repo, "archive")\n'.replace("\n    ", "\n    "),
                                 'def stage_unit(repo):\n    return _git(repo, "ls-files")\n'])
def test_a_stage_that_is_not_the_commit_tree_through_cat_file_is_refused(src):
    with pytest.raises(sd.Refusal):
        sd.assert_stage_reads_the_commit(src)


def test_the_docstring_mention_of_git_archive_is_not_a_use():
    assert sd.assert_stage_reads_the_commit(STAGE_OK)


# ------------------------------------------------------------------------------------------------------------ the digest (G2)

def _store():
    nodes = {"Repo": [{"repoId": "R"}], "Unit": [{"unitId": "U1", "flowAvailable": False}, {"unitId": "U2"}], "SkippedFile": [{"unitId": "U1", "path": "a"}]}
    edges = {("Unit", "IN_REPO", "Repo"): [("U1", "R", {}), ("U2", "R", {"k": 1})], ("Unit", "SKIPS", "SkippedFile"): [("U1", "U1|a", {})]}
    return nodes, edges


def test_digest_is_order_independent_and_sees_a_node_an_edge_and_a_property():
    nodes, edges = _store()
    base = sd.digest_of(nodes, edges)
    rev = sd.digest_of({k: list(reversed(v)) for k, v in reversed(list(nodes.items()))}, {k: list(reversed(v)) for k, v in edges.items()})
    assert rev == base and len(base) == 64
    n2 = {**nodes, "Unit": nodes["Unit"] + [{"unitId": "U3"}]}
    e2 = {**edges, ("Unit", "IN_REPO", "Repo"): edges[("Unit", "IN_REPO", "Repo")][:1]}
    n3 = {**nodes, "Unit": [{"unitId": "U1", "flowAvailable": True}, {"unitId": "U2"}]}
    e3 = {**edges, ("Unit", "IN_REPO", "Repo"): [("U1", "R", {}), ("U2", "R", {"k": 2})]}
    assert len({base, sd.digest_of(n2, edges), sd.digest_of(nodes, e2), sd.digest_of(n3, edges), sd.digest_of(nodes, e3)}) == 5


class PagedGraph:
    """A fake of the store: answers exactly the query shapes of ``dump_graph`` from in-memory rows."""

    def __init__(self, nodes, edges, id_prop):
        self.nodes, self.edges, self.id_prop, self.sizes = nodes, edges, id_prop, []

    def ro(self, text, params):
        import re
        if text == "MATCH (n) RETURN count(n)":
            return [(sum(len(v) for v in self.nodes.values()),)]
        if text == "MATCH ()-[r]->() RETURN count(r)":
            return [(sum(len(v) for v in self.edges.values()),)]
        if m := re.match(r"MATCH \(n:(\w+)\) WHERE n\.(\w+) > \$c RETURN properties\(n\) ORDER BY n\.\w+ LIMIT (\d+)", text):
            label, idp, lim = m[1], m[2], int(m[3])
            rows = sorted((r for r in self.nodes.get(label, []) if r[idp] > params["c"]), key=lambda r: r[idp])[:lim]
            return [(r,) for r in rows]
        if m := re.match(r"MATCH \(n:SkippedFile\) RETURN properties\(n\) ORDER BY n\.unitId, n\.path SKIP (\d+) LIMIT (\d+)", text):
            rows = sorted(self.nodes.get("SkippedFile", []), key=lambda r: (r["unitId"], r["path"]))
            return [(r,) for r in rows[int(m[1]):int(m[1]) + int(m[2])]]
        if m := re.match(r"MATCH \(a:(\w+)\)-\[r:(\w+)\]->\(b:(\w+)\) WHERE a\.(\w+) IN \$ids RETURN a\.(\w+), (.+), properties\(r\)$", text):
            from ccg.store.loader import ID_PROP as REAL
            idp = {**REAL, **self.id_prop}
            assert m[4] == m[5] == idp[m[1]]
            assert m[6] == ("b.unitId, b.path" if m[3] == "SkippedFile" else f"b.{idp[m[3]]}")        # the destination key columns the digest joins with '|'
            self.sizes.append(len(params["ids"]))
            out = []
            for sid, dst, props in self.edges.get((m[1], m[2], m[3]), []):
                if sid in params["ids"]:
                    out.append((sid, *dst.split("|"), props))
            return out
        raise AssertionError(text)


ID_PROP = {"Repo": "repoId", "Unit": "unitId"}
GROUPS = [("Unit", "IN_REPO", "Repo"), ("Unit", "SKIPS", "SkippedFile")]


@pytest.mark.parametrize("page", [1, 2, 3, 4000])
def test_dump_graph_pages_reassemble_the_whole_graph(page):
    nodes, edges = _store()
    got_n, got_e = sd.dump_graph(PagedGraph(nodes, edges, ID_PROP), ID_PROP, GROUPS, page=page)
    assert sd.digest_of(got_n, got_e) == sd.digest_of(nodes, edges)


def test_an_edge_page_that_fills_the_server_cap_is_split_then_refused(monkeypatch):
    nodes = {"Repo": [{"repoId": "R"}], "Unit": [{"unitId": f"U{i}"} for i in range(4)]}
    edges = {("Unit", "IN_REPO", "Repo"): [(f"U{i}", "R", {}) for i in range(4)]}
    monkeypatch.setattr(sd, "RESULTSET_SIZE", 3)
    g = PagedGraph(nodes, edges, ID_PROP)
    got_n, got_e = sd.dump_graph(g, ID_PROP, GROUPS[:1])
    assert sd.digest_of(got_n, got_e) == sd.digest_of(nodes, edges) and min(g.sizes) < 4         # the 4-id page was split to stay under the cap
    monkeypatch.setattr(sd, "RESULTSET_SIZE", 1)
    with pytest.raises(sd.Refusal, match="truncated"):
        sd.dump_graph(PagedGraph(nodes, edges, ID_PROP), ID_PROP, GROUPS[:1])


# ------------------------------------------------------------------------------------------------------------ the statistic

def test_wilson_matches_the_records_figures():
    assert sd.wilson(28, 28)["lower"] == pytest.approx(0.8794, abs=5e-5)
    assert sd.wilson(4, 4)["lower"] == pytest.approx(0.5101, abs=5e-5)
    assert sd.wilson(10, 28)["lower"] < 10 / 28 < sd.wilson(10, 28)["upper"]


def _item(kind, ident, chain=()):
    return SimpleNamespace(kind=kind, id=ident, distance=1, possible_hops=0, length=1, chain=list(chain))


def _dev(impacted="I"):
    return sd.DevItem("it", "A", impacted, "R", "a", "R", "i.py", "q", "callee-arg", "python")


def test_an_item_is_listed_only_for_kind_symbol_and_the_impacted_id():
    d = _dev()
    assert sd.listed([_item("symbol", "I")], d)
    assert not sd.listed([_item("config", "I")], d)                                  # a different (kind, node) pair
    assert not sd.listed([_item("symbol", "J")], d)
    assert not sd.listed([], d)


def _hop(t, dir_="->", state=("end", "callee")):
    """A synthetic hop for the predicate cases only (edge type, direction, arrival state); the definition itself is pinned over a real walk below."""
    return {"node": ("Symbol", "x"), "state": state, "edge": {"type": t, "dir": dir_, "status": "possible"}}


def test_second_level_callee_only_needs_two_callee_state_hops_in_the_best_chain():
    d = _dev()
    one = [_hop("CALLS")]
    two = [_hop("CALLS"), _hop("HAS_CALL", state=("end", "")), _hop("CALLS_NAME")]
    wrong_dir = [_hop("CALLS", "<-"), _hop("CALLS", "<-")]
    wrong_state = [_hop("CALLS", state=("path", "a0")), _hop("CALLS", state=("end", ""))]       # an argument-flow hop is no callee hop
    assert sd.second_level_callee_only([_item("symbol", "I", one)], d) is None
    assert sd.second_level_callee_only([_item("symbol", "I", wrong_dir)], d) is None
    assert sd.second_level_callee_only([_item("symbol", "I", wrong_state)], d) is None
    assert sd.second_level_callee_only([_item("symbol", "I", two)], d)["itemId"] == "it"
    assert sd.second_level_callee_only([_item("symbol", "J", two)], d) is None


def _reach_with_f3():
    """``reach_graph`` plus an argument flow ``F2 -> F3`` (``pf`` of ``F2`` flows into a call of ``F3``), walked from the changed param ``px``."""
    from test_engine_flow import reach_graph
    g = reach_graph()
    g.sym("F3", unit="ut"), g.param("pf3", "F3", index=0, name="o")
    g.site("cf2", "F2", unit="ut"), g.calls("cf2", "F3"), g.data("CallSite", "cf2", "Param", "pf", "p0", "a0")
    return g


def _real_items(g, table):
    from engine_walk_helpers import MemoryReader
    from ccg.engine import bfs, flow
    return {i.id: i for i in bfs.walk(MemoryReader(g), ("Param", "px"), column=bfs.IMPACT, table=flow.candidate_table(table)).items()}


def test_an_argument_flow_chain_is_not_a_second_level_callee_but_a_callee_chain_is_over_a_real_walk():
    """Probed on a real walk (B1): the argument rows (``CALLS ->`` into state ``path``) twice reach ``F3``; only V-both's ``HAS_CALL ->`` / ``CALLS ->`` into
    state ``("end", "callee")`` is a call-depth callee, so V-union's best chain to ``F3`` (the argument one) has none and V-both's has two."""
    g = _reach_with_f3()
    both, union = _real_items(g, "V-both"), _real_items(g, "V-union")
    assert (sd.callee_hops(both["F3"].chain), sd.callee_hops(union["F3"].chain)) == (2, 0)
    assert (sd.callee_hops(both["F2"].chain), sd.callee_hops(union["F2"].chain)) == (1, 1)
    d3, d2 = _dev("F3"), _dev("F2")
    assert sd.second_level_callee_only(list(both.values()), d3)["itemId"] == "it"
    assert sd.second_level_callee_only(list(union.values()), d3) is None
    assert sd.second_level_callee_only(list(both.values()), d2) is None and sd.second_level_callee_only(list(union.values()), d2) is None


def _cands(spec):
    return {n: {"pooled": {"listed": s, "n": 28}, "burdenSum": b, "burdenMax": m} for n, (s, b, m) in spec.items()}


def test_the_rule_takes_max_s_then_burden_sum_then_max_burden_then_the_fixed_order():
    c = _cands({"as-built": (10, 5, 3), "V-flow": (28, 40, 20), "V-both": (28, 30, 20), "V-union": (28, 30, 10)})
    assert sd.rule_pick(c)["pick"] == "V-union"                                      # tie on sum 30: lower max burden
    c["V-union"]["burdenMax"] = 20
    assert sd.rule_pick(c)["pick"] == "V-both"                                       # still tied: V-both precedes V-union in the fixed list
    c["V-flow"].update(burdenSum=30)
    assert sd.rule_pick(c)["pick"] == "V-both"                                       # V-both precedes V-flow
    c["as-built"]["pooled"]["listed"] = 28
    assert sd.rule_pick(c)["pick"] == "as-built"                                     # lowest burden among the max-s walks
    assert sd.rule_pick(_cands({n: (27, 1, 1) for n in sd.CANDIDATES}))["safeClaim"] is False


# ------------------------------------------------------------------------------------------------------------ measure over the real engine (in-memory graph)

def _real_measure(dev, graph):
    from engine_walk_helpers import MemoryReader
    from ccg.engine import bfs, flow
    return sd.measure(MemoryReader(graph), dev, walk=bfs.walk, tables=flow.candidate_table, impact=bfs.IMPACT)


def test_measure_over_the_real_walk_lists_callers_and_reports_the_unlisted_by_id():
    from test_engine_flow import caller_chain
    dev = [sd.DevItem("g", "F", "G", "R", "a", "R", "i.py", "q", "callee-arg", "python"), sd.DevItem("h", "F", "H", "R", "a", "R", "i.ts", "q", "callee-arg,return-use", "javascript"),
           sd.DevItem("z", "F", "ZZ", "R", "a", "R", "i.ts", "q", "return-use", "javascript")]
    res = _real_measure(dev, caller_chain(flow=False))
    asb = res["candidates"]["as-built"]
    assert (asb["listedIds"], asb["unlistedIds"], asb["burdenSum"]) == (["g", "h"], ["z"], 2)
    assert asb["byLanguage"] == {"javascript": {"listed": 1, "n": 2}, "python": {"listed": 1, "n": 1}}
    assert asb["bySignature"]["return-use"] == {"listed": 0, "n": 1}
    assert asb["clusterLevel"]["k"] == 0 and asb["byAnchorCluster"]["F"]["allListed"] is False
    assert res["rule"]["pick"] == "as-built" and res["setUnionLowerBound"]["k"] == 2
    assert res["asBuiltComparison"]["pooled"]["measured"] == [2, 3] and res["asBuiltComparison"]["pooled"]["equal"] is False


def _multi_graph():
    """Two anchors, python and javascript units, everything flag-false. ``A1`` (python) is called by ``C1`` and ``C1b``; ``C1`` also calls ``Q1`` and its call result
    feeds a call in ``D1``. ``A2`` (javascript) is called by ``C2``, which also calls ``Q2``. By the shipped rows a caller is listed; V-flow adds ``D1`` (the data use),
    V-both adds ``Q1`` / ``Q2`` (the callees of a caller), V-union both."""
    from engine_walk_helpers import G
    g = G(unit="upy", flow=False, lang="python")
    g.unit_node("ujs", flow=False, lang="javascript")
    g.sym("A1", path="a1.py"), g.sym("A2", path="a2.ts", unit="ujs")
    g.sym("C1"), g.site("cC1", "C1", callee="a1"), g.calls("cC1", "A1")
    g.sym("C1b"), g.site("cC1b", "C1b", callee="a1"), g.calls("cC1b", "A1")                       # a walk item that is no dev item
    g.sym("Q1"), g.site("cQ1", "C1", callee="q1", line=6), g.calls("cQ1", "Q1")
    g.sym("D1"), g.site("cD1", "D1", callee="d1", line=7), g.data("CallSite", "cD1", "CallSite", "cC1", "", "")
    g.sym("C2", unit="ujs"), g.site("cC2", "C2", unit="ujs", callee="a2"), g.calls("cC2", "A2")
    g.sym("Q2", unit="ujs"), g.site("cQ2", "C2", unit="ujs", callee="q2", line=6), g.calls("cQ2", "Q2")
    return g


def _multi_dev():
    def d(item, anchor, imp, lang, sig):
        return sd.DevItem(item, anchor, imp, "R", "a", "R", "i." + ("py" if lang == "python" else "ts"), f"q-{anchor}", sig, lang)
    return [d("c1", "A1", "C1", "python", "return-use"), d("c1b", "A1", "C1b", "python", "return-use"), d("q1", "A1", "Q1", "python", "callee-arg"), d("d1", "A1", "D1", "javascript", "callee-arg"),
            d("z1", "A1", "ZZ", "javascript", "return-use"),                                         # no such node: unlisted by every walk
            d("c2", "A2", "C2", "javascript", "return-use"), d("q2", "A2", "Q2", "javascript", "callee-arg")]


def test_measure_over_a_real_multi_anchor_two_language_walk_has_every_figure_the_hand_count_gives():
    """Expected values written by hand from ``_multi_graph`` (not read from the tool): per walk, the items of A1 + A2 (burden counts every item, dev or not, ``C1b`` is both)."""
    res = _real_measure(_multi_dev(), _multi_graph())
    c = res["candidates"]
    assert set(c) == set(sd.CANDIDATES)
    everything = {"c1", "c1b", "q1", "d1", "z1", "c2", "q2"}
    listed = {"as-built": ["c1", "c1b", "c2"], "V-flow": ["c1", "c1b", "c2", "d1"], "V-both": ["c1", "c1b", "c2", "q1", "q2"], "V-union": ["c1", "c1b", "c2", "d1", "q1", "q2"]}
    burden = {"as-built": {"A1": 2, "A2": 1}, "V-flow": {"A1": 3, "A2": 1}, "V-both": {"A1": 3, "A2": 2}, "V-union": {"A1": 4, "A2": 2}}
    for n in sd.CANDIDATES:
        assert c[n]["listedIds"] == listed[n], n
        assert c[n]["unlistedIds"] == sorted(everything - set(listed[n])), n
        assert c[n]["burdenPerAnchor"] == burden[n], n
        assert (c[n]["burdenSum"], c[n]["burdenMax"]) == (sum(burden[n].values()), max(burden[n].values())), n
        assert c[n]["pooled"]["listed"] == len(listed[n]) and c[n]["pooled"]["n"] == 7 and c[n]["pooled"]["k"] == len(listed[n]) and c[n]["pooled"]["s"] == len(listed[n]) / 7
    assert [c[n]["burdenSum"] for n in sd.CANDIDATES] == [3, 4, 5, 6] and [c[n]["burdenMax"] for n in sd.CANDIDATES] == [2, 3, 3, 4]
    # languages (python: c1, c1b, q1; javascript: d1, z1, c2, q2) and signatures (return-use: c1, c1b, z1, c2; callee-arg: q1, d1, q2)
    assert c["as-built"]["byLanguage"] == {"javascript": {"listed": 1, "n": 4}, "python": {"listed": 2, "n": 3}}
    assert c["V-flow"]["byLanguage"] == {"javascript": {"listed": 2, "n": 4}, "python": {"listed": 2, "n": 3}}
    assert c["V-both"]["byLanguage"] == {"javascript": {"listed": 2, "n": 4}, "python": {"listed": 3, "n": 3}}
    assert c["V-union"]["byLanguage"] == {"javascript": {"listed": 3, "n": 4}, "python": {"listed": 3, "n": 3}}
    assert c["V-flow"]["bySignature"] == {"callee-arg": {"listed": 1, "n": 3}, "return-use": {"listed": 3, "n": 4}}
    assert c["V-union"]["bySignature"] == {"callee-arg": {"listed": 3, "n": 3}, "return-use": {"listed": 3, "n": 4}}
    # clusters: A1 holds c1 c1b q1 d1 z1, A2 holds c2 q2; the Wilson interval is over the 2 clusters, not the 7 items
    assert c["V-union"]["byAnchorCluster"] == {"A1": {"listed": 4, "n": 5, "allListed": False, "qname": "q-A1"}, "A2": {"listed": 2, "n": 2, "allListed": True, "qname": "q-A2"}}
    for n, full in (("as-built", 0), ("V-flow", 0), ("V-both", 1), ("V-union", 1)):
        cl = c[n]["clusterLevel"]
        assert (cl["k"], cl["n"]) == (full, 2) and cl["lower"] == pytest.approx(sd.wilson(full, 2)["lower"]) and cl["upper"] == pytest.approx(sd.wilson(full, 2)["upper"]), n
    assert c["V-union"]["pooled"]["lower"] == pytest.approx(sd.wilson(6, 7)["lower"])
    # the set-union lower bound is V-flow union V-both (6 items), not their intersection (c1, c1b, c2)
    assert res["setUnionLowerBound"]["listedIds"] == ["c1", "c1b", "c2", "d1", "q1", "q2"]
    assert (res["setUnionLowerBound"]["k"], res["setUnionLowerBound"]["n"]) == (6, 7) and res["setUnionLowerBound"]["lower"] == pytest.approx(sd.wilson(6, 7)["lower"])
    # as-built comparison: the python cell and the javascript cell differ (2/3 against 1/4), each against its own prediction
    assert res["asBuiltComparison"] == {"pooled": {"predicted": [10, 28], "measured": [3, 7], "equal": False}, "python": {"predicted": [6, 9], "measured": [2, 3], "equal": False},
                                        "javascript": {"predicted": [4, 19], "measured": [1, 4], "equal": False}}
    assert res["rule"] == {"pick": "V-union", "maxListed": 6, "safeClaim": False, "among": ["V-union"]}
    assert set(res) == {"candidates", "setUnionLowerBound", "asBuiltComparison", "rule"}


def test_the_as_built_comparison_says_equal_only_when_every_cell_equals_the_prediction():
    """Over a graph whose as-built walk lists exactly the record's prediction (10 of 28, python 6/9, javascript 4/19): a stub walk, since only the comparison is under test."""
    dev = [sd.DevItem(f"p{i}", f"A{i}", f"I{i}", "R", "a", "R", "i.py", "q", "return-use", "python") for i in range(9)] + \
          [sd.DevItem(f"j{i}", f"B{i}", f"J{i}", "R", "a", "R", "i.ts", "q", "return-use", "javascript") for i in range(19)]
    want = {f"I{i}" for i in range(6)} | {f"J{i}" for i in range(4)}

    def walk(reader, anchor, *, column, table):
        ids = {"as-built": want}.get(table, {d.impacted_id for d in dev})
        return SimpleNamespace(items=lambda: [_item("symbol", i) for i in sorted(ids) if (anchor[1][0], i[0]) in (("A", "I"), ("B", "J"))], bound=2)
    cmp_ = sd.measure(None, dev, walk=walk, tables=lambda n: n, impact="i")["asBuiltComparison"]
    assert {k: v["equal"] for k, v in cmp_.items()} == {"pooled": True, "python": True, "javascript": True}
    assert cmp_["python"]["measured"] == [6, 9] and cmp_["javascript"]["measured"] == [4, 19]


def test_the_emitted_keys_of_every_candidate_block_are_exactly_the_contracts():
    res = _real_measure(_multi_dev(), _multi_graph())
    base = {"listedIds", "unlistedIds", "burdenPerAnchor", "burdenSum", "burdenMax", "pooled", "byLanguage", "bySignature", "byAnchorCluster", "clusterLevel"}
    for n, block in res["candidates"].items():
        assert set(block) == base | ({"secondLevelCalleeOnly"} if n in ("V-both", "V-union") else set()), n
        assert set(block["pooled"]) == {"listed", "n", "k", "s", "lower", "upper"} and set(block["clusterLevel"]) == {"k", "n", "s", "lower", "upper"}
    assert set(res["setUnionLowerBound"]) == {"listedIds", "k", "n", "s", "lower", "upper"}


def test_second_level_callee_only_items_are_emitted_for_v_both_and_v_union_only_and_over_a_real_walk():
    """The walk is anchored at the param ``px`` of ``reach_graph`` + ``F2 -> F3`` (the id of the dev anchor maps to it); V-both reaches ``F3`` through two callee hops,
    V-union and V-flow through the argument rows (no callee hop), so only V-both emits a hit."""
    from engine_walk_helpers import MemoryReader
    from ccg.engine import bfs, flow
    g = _reach_with_f3()
    dev = [sd.DevItem("zz", "PX", "F3", "R", "a", "R", "i.py", "q", "callee-arg", "python"), sd.DevItem("f2", "PX", "F2", "R", "a", "R", "i.py", "q", "callee-arg", "python"),
           sd.DevItem("aa", "PY", "F3", "R", "a", "R", "i.py", "q", "callee-arg", "python")]            # two anchors (both walk from px): the hits come in anchor order zz, aa
    def walk(reader, anchor, **kw):
        return bfs.walk(reader, ("Param", "px"), **kw)
    res = sd.measure(MemoryReader(g), dev, walk=walk, tables=flow.candidate_table, impact=bfs.IMPACT)
    c = res["candidates"]
    assert "secondLevelCalleeOnly" not in c["as-built"] and "secondLevelCalleeOnly" not in c["V-flow"]
    assert [h["itemId"] for h in c["V-both"]["secondLevelCalleeOnly"]] == ["aa", "zz"] and c["V-union"]["secondLevelCalleeOnly"] == []      # sorted by item id
    chain = c["V-both"]["secondLevelCalleeOnly"][0]["chain"]
    assert [tuple(s["state"]) for s in chain if s["edge"] and s["edge"]["type"] == "CALLS"] == [("end", "callee")] * 2
    assert c["V-flow"]["listedIds"] == ["aa", "f2", "zz"] and c["V-both"]["listedIds"] == ["aa", "f2", "zz"]


def test_a_walk_that_is_not_byte_equal_twice_voids_the_run():
    calls = {"n": 0}

    def flaky(reader, anchor, *, column, table):
        calls["n"] += 1
        return SimpleNamespace(items=lambda: [_item("symbol", f"x{calls['n']}")], bound=2)
    with pytest.raises(sd.VoidRun, match="byte-equal"):
        sd.measure(None, [_dev()], walk=flaky, tables=lambda n: n, impact="i")


@pytest.mark.parametrize("field", ["kind", "id", "distance", "possible_hops", "length", "chain"])
def test_every_field_of_every_item_is_part_of_the_byte_equality_of_two_runs(field):
    calls = {"n": 0}
    other = {"kind": "config", "id": "y", "distance": 2, "possible_hops": 3, "length": 4, "chain": [{"state": ["end", ""]}]}

    def flaky(reader, anchor, *, column, table):
        calls["n"] += 1
        it = _item("symbol", "x")
        if calls["n"] % 2 == 0:
            setattr(it, field, other[field])
        return SimpleNamespace(items=lambda: [it], bound=2)
    with pytest.raises(sd.VoidRun, match="byte-equal"):
        sd.measure(None, [_dev()], walk=flaky, tables=lambda n: n, impact="i")


def test_a_walk_that_returns_the_same_items_in_the_same_order_twice_is_accepted_with_two_items():
    def walk(reader, anchor, *, column, table):
        return SimpleNamespace(items=lambda: [_item("symbol", "a"), _item("symbol", "b")], bound=2)
    assert sd.measure(None, [_dev("b")], walk=walk, tables=lambda n: n, impact="i")["candidates"]["as-built"]["burdenSum"] == 2


def test_a_walk_with_another_depth_bound_voids_the_run():
    def deep(reader, anchor, *, column, table):
        return SimpleNamespace(items=lambda: [], bound=3)
    with pytest.raises(sd.VoidRun, match="depth bound"):
        sd.measure(None, [_dev()], walk=deep, tables=lambda n: n, impact="i")


def test_a_v_union_that_misses_a_listed_item_of_v_flow_or_v_both_voids_the_run():
    def walk(reader, anchor, *, column, table):
        listed_here = [] if table == "V-union" else [_item("symbol", "I")]
        return SimpleNamespace(items=lambda: listed_here, bound=2)
    with pytest.raises(sd.VoidRun, match="superset"):
        sd.measure(None, [_dev()], walk=walk, tables=lambda n: n, impact="i")


def test_the_walk_is_called_with_the_symbol_anchor_the_impact_column_and_the_candidate_table_only():
    seen = []

    def walk(reader, anchor, **kw):
        seen.append((anchor, tuple(sorted(kw)), kw["column"], kw["table"]))
        return SimpleNamespace(items=lambda: [], bound=2)
    sd.measure(None, [_dev()], walk=walk, tables=lambda n: f"T-{n}", impact="impact")
    assert {s[1] for s in seen} == {("column", "table")} and {s[0] for s in seen} == {("Symbol", "A")} and {s[3] for s in seen} == {f"T-{n}" for n in sd.CANDIDATES}


# ------------------------------------------------------------------------------------------------------------ the dev file and the real pins

def test_an_unclassifiable_impacted_extension_refuses(fx):
    rows = dev_rows()
    rows[0]["impacted"]["path"] = "x.rs"
    b = b"".join(json.dumps(d, sort_keys=True).encode() + b"\n" for d in rows)
    (fx.root / "cpg/eval/dev/ac5-flow.jsonl").write_bytes(b)
    with pytest.raises(sd.Refusal, match="extension"):
        sd.load_dev(fx.root, json.loads(fx.lines()[1]))


def test_the_dev_item_and_cluster_counts_must_equal_the_record(fx):
    rec = json.loads(fx.lines()[1])
    rec["pins"]["dev"]["items"] = 28
    with pytest.raises(sd.Refusal, match="count"):
        sd.load_dev(fx.root, rec)


def test_a_record_that_pins_no_engine_file_cannot_pass():
    rec = {"pins": {"dev": {"a/b": "0" * 64}, "engine": {}, "flagRegistry": {"c/d": "0" * 64}, "plans": {"e/f": "0" * 64},
                    "corpusAndBuilder": {"builderFiles": {"g/h": "0" * 64}}}, "retLimit": {"pins": {"cpg/tests/test_engine_bfs_rows.py": "0" * 64}}}
    with pytest.raises(sd.Refusal, match="pins.engine"):
        sd.pinned_files(rec)


def test_the_committed_real_record_yields_its_pinned_set_and_the_real_dev_file_loads():
    """Against the real repository (read-only): the pin extraction finds the record's groups (92 entries, 91 distinct paths: the flag registry is pinned twice) and the dev file has its 28 items / 4 clusters."""
    head = sd.capture_head(sd.REPO)
    lines = sd.committed_record_lines(sd.REPO, head)
    rec = sd._line_of(lines, sd.RECORD_ID)
    pins = sd.pinned_files(rec)
    assert len(pins) == 91 and "cpg/ccg/engine/bfs.py" in pins and "cpg/eval/dev/ac5-flow.jsonl" in pins
    dev = sd.load_dev(sd.REPO, rec)
    assert (len(dev), len({d.anchor_id for d in dev})) == (28, 4)
    assert sum(d.language == "python" for d in dev) == 9 and sum(d.language == "javascript" for d in dev) == 19


def test_the_graph_digest_covers_the_snapshot_node_the_loader_writes_last():
    """Found live: ``Snapshot`` is not in ``ID_PROP`` (the loader's own table) yet is a node of the graph, so a dump over ``ID_PROP`` alone left it out of the digest."""
    nodes = {"Snapshot": [{"snapshotId": "s1"}], "Repo": [{"repoId": "R"}]}
    g = PagedGraph(nodes, {}, {})
    digest, counts = sd.graph_digest(g)
    assert counts["nodes"] == 2
    nodes2 = {"Snapshot": [{"snapshotId": "s2"}], "Repo": [{"repoId": "R"}]}
    assert sd.graph_digest(PagedGraph(nodes2, {}, {}))[0] != digest


# ------------------------------------------------------------------------------------------------------------ gaps found by mutation

def test_a_host_unit_with_flow_available_true_refuses_the_premise_is_all_false(fx, tmp_path):
    spy = Spy(fx, FakeGraph(fx.eval_sha, flow=True))
    code, _, err = run(fx, spy, "A", tmp_path)
    assert code == 2 and "flowAvailable" in err


def test_the_digest_refuses_when_an_independent_node_or_edge_count_differs_from_the_dump():
    nodes = {"Snapshot": [{"snapshotId": "s1"}], "Repo": [{"repoId": "R"}], "Unit": [{"unitId": "U1"}]}
    edges = {("Unit", "IN_REPO", "Repo"): [("U1", "R", {})]}
    assert sd.graph_digest(PagedGraph(nodes, edges, {}))[1] == {"nodes": 3, "edges": 1}
    with pytest.raises(sd.Refusal, match="nodes"):
        sd.graph_digest(PagedGraph({**nodes, "Mystery": [{"k": 1}]}, edges, {}))                 # a label the dump does not enumerate
    with pytest.raises(sd.Refusal, match="edges"):
        sd.graph_digest(PagedGraph(nodes, {**edges, ("Unit", "ODD", "Repo"): [("U1", "R", {})]}, {}))   # an edge group the dump does not enumerate


def test_an_unknown_unit_kind_marks_that_unit_failed_never_aborts_the_build():
    from ccg.build import UnknownUnitKind
    unit = SimpleNamespace(unitId="u")

    def real(u, repo, commit, system, ctx):
        raise UnknownUnitKind("x")
    got = sd.unknown_kind_as_failed(lambda u, *a: SimpleNamespace(ok=True))(unit, "r", "c", "s", None)
    assert got.ok is True
    res = sd.unknown_kind_as_failed(real)(unit, "r", "c" * 40, "s", None)
    assert (res.ok, res.fail_reason, res.unit, res.commit) == (False, "extract:failed", unit, "c" * 40)


def test_a_bind_key_mismatch_is_a_refusal_because_refresh_would_demote_that_unit():
    sd.check_bind_keys(SimpleNamespace(bind_key_mismatch=[]))
    with pytest.raises(sd.Refusal, match="bind-key"):
        sd.check_bind_keys(SimpleNamespace(bind_key_mismatch=["unit U1: x"]))


def test_the_scratch_dirs_also_turn_on_the_forbid_real_credentials_guard(fx, tmp_path):
    import os
    os.environ.pop("CCG_FORBID_REAL_CREDENTIALS", None)
    assert run(fx, Spy(fx), "A", tmp_path)[0] == 0
    assert os.environ["CCG_FORBID_REAL_CREDENTIALS"] == "1"


def test_a_record_with_no_committed_version_is_not_append_only_checkable(fx):
    with pytest.raises(sd.Refusal, match="no committed version"):
        sd.check_append_only(fx.root, fx.eval_sha)                                           # the base commit predates the record file


def test_the_digest_is_the_documented_algorithm_over_canonical_json_and_bytewise_sorted_lines():
    """Written out by hand from ``DIGEST_DESCRIPTION``: header line, then node and edge lines sorted bytewise, ``\\n``-joined; properties as compact sorted-key ASCII JSON."""
    nodes = {"Repo": [{"repoId": "R", "name": "caf\u00e9", "a": 1}], "Unit": [{"unitId": "U1"}]}
    edges = {("Unit", "IN_REPO", "Repo"): [("U1", "R", {"z": 1, "b": 2})]}
    lines = ['N|Repo|{"a":1,"name":"caf\\u00e9","repoId":"R"}', 'N|Unit|{"unitId":"U1"}', 'E|Unit|IN_REPO|Repo|U1|R|{"b":2,"z":1}']
    want = hashlib.sha256(("ccg-flowfb-graph-digest-v1\n" + "\n".join(sorted(lines, key=lambda x: x.encode()))).encode()).hexdigest()
    assert sd.digest_of(nodes, edges) == want


def test_wilson_of_an_empty_cell_has_no_figures():
    assert sd.wilson(0, 0) == {"k": 0, "n": 0, "s": None, "lower": None, "upper": None}


def test_a_dev_file_with_blank_lines_loads_the_same_items(fx):
    rec = json.loads(fx.lines()[1])
    (fx.root / "cpg/eval/dev/ac5-flow.jsonl").write_bytes(b"\n" + fx.dev_bytes.replace(b"\n", b"\n\n  \n"))
    assert len(sd.load_dev(fx.root, rec)) == 7


def test_a_dev_item_is_immutable():
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        _dev().anchor_id = "other"


def test_a_stage_with_exactly_the_two_constants_is_accepted():
    assert sd.assert_stage_reads_the_commit('def stage_unit(repo):\n    return _git(repo, "cat-file", "--batch")\n')


@pytest.mark.parametrize("group,gone", [("pins.dev", lambda r: r["pins"]["dev"].update({"cpg/eval/dev/ac5-flow.jsonl": "x"})), ("pins.engine", lambda r: r["pins"]["engine"].clear()),
                                        ("pins.flagRegistry", lambda r: r["pins"]["flagRegistry"].clear()), ("pins.plans", lambda r: r["pins"]["plans"].clear()),
                                        ("pins.corpusAndBuilder.builderFiles", lambda r: r["pins"]["corpusAndBuilder"]["builderFiles"].clear()),
                                        ("retLimit.pins", lambda r: r["retLimit"]["pins"].clear())])
def test_each_group_of_pins_must_hold_at_least_one_file_and_the_refusal_names_it(fx, group, gone):
    rec = json.loads(fx.lines()[1])
    gone(rec)
    if group == "pins.dev":
        rec["pins"]["dev"] = {"items": 7}
    with pytest.raises(sd.Refusal, match=f"under {group.replace('.', chr(92) + '.')}$"):
        sd.pinned_files(rec)


def test_a_committed_record_without_the_records_own_line_refuses(fx, tmp_path):
    other = Repo(tmp_path / "other")
    other.write("README", b"x\n")
    other.write(sd.RECORD_PATH, json.dumps({"id": "something-else"}).encode() + b"\n")
    head = other.commit("only another line")
    with pytest.raises(sd.Refusal, match=f"no {sd.RECORD_ID} line"):
        sd.preflight(other.root, head, "A", env_for(tmp_path), Spy(fx).deps())


def test_a_phase_other_than_a_or_b_refuses_in_preflight(fx, tmp_path):
    with pytest.raises(sd.Refusal, match="A or B"):
        sd.preflight(fx.root, sd.capture_head(fx.root), "C", env_for(tmp_path), Spy(fx).deps())


def test_git_runs_with_a_clean_config_no_prompt_a_c_locale_and_a_timeout(fx, monkeypatch):
    seen = {}
    real = subprocess.run

    def spy(argv, **kw):
        seen.update(kw)
        return real(argv, **kw)
    monkeypatch.setattr(sd.subprocess, "run", spy)
    sd.git(fx.root, "rev-parse", "HEAD")
    env = seen["env"]
    assert (env["GIT_CONFIG_GLOBAL"], env["GIT_CONFIG_SYSTEM"], env["GIT_TERMINAL_PROMPT"], env["LC_ALL"]) == ("/dev/null", "/dev/null", "0", "C")
    assert seen["timeout"] == 120 and seen["stdin"] == subprocess.DEVNULL and seen["capture_output"] is True


def test_an_edge_page_over_more_than_500_source_ids_is_chunked_without_loss_or_duplication():
    nodes = {"Repo": [{"repoId": "R"}], "Unit": [{"unitId": f"U{i:04d}"} for i in range(1201)]}
    edges = {("Unit", "IN_REPO", "Repo"): [(f"U{i:04d}", "R", {"i": i}) for i in range(1201)]}
    g = PagedGraph(nodes, edges, ID_PROP)
    got_n, got_e = sd.dump_graph(g, ID_PROP, GROUPS[:1])
    assert sd.digest_of(got_n, got_e) == sd.digest_of(nodes, edges) and sorted(g.sizes) == [201, 500, 500]


def test_a_page_that_fills_the_cap_is_split_down_to_single_ids_and_only_a_single_full_id_refuses(monkeypatch):
    nodes = {"Repo": [{"repoId": "R"}], "Unit": [{"unitId": f"U{i}"} for i in range(4)]}
    edges = {("Unit", "IN_REPO", "Repo"): [(f"U{i}", "R", {}) for i in range(4)]}
    monkeypatch.setattr(sd, "RESULTSET_SIZE", 2)                                      # a chunk of 2 ids still fills it and must be split again, not refused
    got_n, got_e = sd.dump_graph(PagedGraph(nodes, edges, ID_PROP), ID_PROP, GROUPS[:1])
    assert sd.digest_of(got_n, got_e) == sd.digest_of(nodes, edges)


def test_the_as_built_cells_of_an_absent_language_are_zero_of_zero():
    dev = [sd.DevItem("j", "A", "I", "R", "a", "R", "i.ts", "q", "return-use", "javascript")]

    def walk(reader, anchor, *, column, table):
        return SimpleNamespace(items=lambda: [_item("symbol", "I")], bound=2)
    cmp_ = sd.measure(None, dev, walk=walk, tables=lambda n: n, impact="i")["asBuiltComparison"]
    assert cmp_["python"]["measured"] == [0, 0] and cmp_["javascript"]["measured"] == [1, 1]


def test_main_reads_argv_and_the_environment_by_default_and_rejects_a_bad_command_line(fx, tmp_path, monkeypatch):
    for bad in ([], ["C"], ["A", "B"]):
        out, err = io.StringIO(), io.StringIO()
        assert sd.main(bad, deps=Spy(fx).deps(), env=env_for(tmp_path), root=fx.root, out=out, err=err) == 2 and "usage" in err.getvalue() and out.getvalue() == ""
    monkeypatch.setattr(sys, "argv", ["superset_dev.py", "A"])
    for k, v in env_for(tmp_path).items():
        monkeypatch.setenv(k, v)
    out, err = io.StringIO(), io.StringIO()
    assert sd.main(deps=Spy(fx).deps(), root=fx.root, out=out, err=err) == 0 and json.loads(out.getvalue())["phase"] == "A"


def test_a_void_run_exits_three_and_a_refusal_exits_two(fx, tmp_path):
    spy = Spy(fx)
    commit_snapshot(fx, spy, tmp_path)
    flaky = {"n": 0}

    def walk(reader, anchor, *, column, table):
        flaky["n"] += 1
        return SimpleNamespace(items=lambda: [_item("symbol", f"x{flaky['n']}")], bound=2)
    spy.walk = walk
    code, out, err = run(fx, spy, "B", tmp_path)
    assert (code, out) == (3, "") and err.startswith("VOID: ") and not (fx.root / "cpg/eval/results").exists()


def test_the_script_entry_point_refuses_a_missing_phase_with_usage():
    done = subprocess.run([sys.executable, str(CPG / "eval/superset_dev.py")], capture_output=True, text=True, cwd=str(CPG), timeout=60)
    assert done.returncode == 2 and "usage: superset_dev.py A|B" in done.stderr


def test_a_pinned_file_that_is_missing_refuses(fx, tmp_path):
    (fx.root / "cpg/ccg/engine/bfs.py").unlink()
    code, _, err = run(fx, Spy(fx), "A", tmp_path)
    assert code == 2 and "pinned file" in err


def test_an_eval_commit_that_is_not_an_object_of_the_repository_refuses_even_when_it_equals_the_pin(fx):
    rec = json.loads(fx.lines()[1])
    rec["pins"]["evalCommit"] = "f" * 40
    (fx.root / sd.EVAL_COMMIT_PATH).write_text("f" * 40 + "\n")
    with pytest.raises(sd.Refusal, match="git cat-file failed"):
        sd.check_eval_commit(fx.root, rec)
    (fx.root / sd.EVAL_COMMIT_PATH).write_text(fx.eval_sha + "\n")
    assert sd.check_eval_commit(fx.root, json.loads(fx.lines()[1])) == fx.eval_sha


def test_the_tool_line_must_pin_exactly_the_tool_files(fx):
    one = {"id": sd.TOOL_ID, "sha256": {"cpg/eval/superset_dev.py": sha(TOOLS["cpg/eval/superset_dev.py"])}}
    extra = {**fx.tool_line, "sha256": {**fx.tool_line["sha256"], "cpg/eval/other.py": "0" * 64}}
    for bad in (one, extra, {"id": sd.TOOL_ID}):
        with pytest.raises(sd.Refusal, match="must pin exactly"):
            sd.check_tool_line(fx.root, [json.dumps(bad).encode()])
    assert sd.check_tool_line(fx.root, [json.dumps(fx.tool_line).encode()])["id"] == sd.TOOL_ID


def test_the_snapshot_line_must_carry_every_key():
    full = {"id": sd.SNAPSHOT_ID, **{k: "x" for k in sd.SNAPSHOT_KEYS}}
    assert sd.check_snapshot_line([json.dumps(full).encode()])
    for k in sd.SNAPSHOT_KEYS:
        partial = {kk: vv for kk, vv in full.items() if kk != k}
        with pytest.raises(sd.Refusal, match=k):
            sd.check_snapshot_line([json.dumps(partial).encode()])


def test_two_committed_lines_with_one_id_refuse():
    line = json.dumps({"id": sd.TOOL_ID}).encode()
    with pytest.raises(sd.Refusal, match="2 lines"):
        sd._line_of([line, line], sd.TOOL_ID)


def test_append_only_is_judged_at_the_captured_sha_not_at_a_later_head(fx):
    early = sd.capture_head(fx.root)
    lines = fx.lines()
    lines[0] = lines[0].replace(b'"x": 1', b'"x": 2')
    (fx.root / sd.RECORD_PATH).write_bytes(b"\n".join(lines) + b"\n")
    fx.commit("a later rewrite of line 1")
    sd.check_append_only(fx.root, early)                                              # fine as of the captured sha
    with pytest.raises(sd.Refusal, match="append-only"):
        sd.check_append_only(fx.root, sd.capture_head(fx.root))


def test_the_bfs_rows_test_pin_is_required():
    rec = {"pins": {"dev": {"a/b": "0" * 64}, "engine": {"x.py": "0" * 64}, "flagRegistry": {"c/d": "0" * 64}, "plans": {"e/f": "0" * 64},
                    "corpusAndBuilder": {"builderFiles": {"g/h": "0" * 64}}}, "retLimit": {"pins": {"cpg/tests/other.py": "0" * 64}}}
    with pytest.raises(sd.Refusal, match="test_engine_bfs_rows"):
        sd.pinned_files(rec)


@pytest.mark.parametrize("files", [0, 2])
def test_a_dev_file_that_is_not_exactly_one_file_node_refuses(fx, tmp_path, files):
    spy = Spy(fx, FakeGraph(fx.eval_sha, files=files))
    code, _, err = run(fx, spy, "A", tmp_path)
    assert code == 2 and "not exactly one File node" in err


def test_the_digest_sees_an_edge_type():
    nodes = {"Unit": [{"unitId": "U1"}], "Repo": [{"repoId": "R"}]}
    a = sd.digest_of(nodes, {("Unit", "IN_REPO", "Repo"): [("U1", "R", {})]})
    b = sd.digest_of(nodes, {("Unit", "HAS_OTHER", "Repo"): [("U1", "R", {})]})
    assert a != b


@pytest.mark.parametrize("page", [1, 2])
def test_skipped_file_nodes_are_paged_by_skip_and_none_is_dropped(page):
    nodes = {"Repo": [{"repoId": "R"}], "SkippedFile": [{"unitId": "U1", "path": p} for p in ("a", "b", "c")]}
    got_n, got_e = sd.dump_graph(PagedGraph(nodes, {}, ID_PROP), {"Repo": "repoId"}, [], page=page)
    assert len(got_n["SkippedFile"]) == 3 and sd.digest_of(got_n, got_e) == sd.digest_of(nodes, {})


def test_the_rule_ranks_burden_sum_before_the_per_anchor_maximum():
    c = _cands({"as-built": (10, 1, 1), "V-flow": (28, 31, 5), "V-both": (28, 30, 20), "V-union": (27, 1, 1)})
    assert sd.rule_pick(c)["pick"] == "V-both"                                       # the lower sum wins although V-flow's maximum is lower


def test_the_signature_is_the_hop_kind_sequence_of_the_chain(fx):
    dev = {d.item_id: d for d in sd.load_dev(fx.root, json.loads(fx.lines()[1]))}
    assert dev["item00"].signature == "callee-arg" and dev["item01"].signature == "callee-arg,return-use"


def test_the_as_built_prediction_is_the_records_own_cut_by_impacted_file():
    assert sd.AS_BUILT_PREDICTION == {"pooled": (10, 28), "python": (6, 9), "javascript": (4, 19)}


def test_the_records_as_built_prediction_follows_from_the_real_dev_hop_kinds():
    """Pre-data derivation (a pure return-use chain, every hop a return-use, is the only kind the as-built walk ends at): the real dev file gives 10/28, python 6/9, TS/JS 4/19."""
    rec = sd._line_of(sd.committed_record_lines(sd.REPO, sd.capture_head(sd.REPO)), sd.RECORD_ID)
    dev = sd.load_dev(sd.REPO, rec)
    pure = [d for d in dev if set(d.signature.split(",")) == {"return-use"}]
    cut = lambda lang: (sum(d.language == lang for d in pure), sum(d.language == lang for d in dev))
    assert (len(pure), len(dev)) == sd.AS_BUILT_PREDICTION["pooled"]
    assert cut("python") == sd.AS_BUILT_PREDICTION["python"] and cut("javascript") == sd.AS_BUILT_PREDICTION["javascript"]


# ------------------------------------------------------------------------------------------------------------ the builder inputs the record does not pin (M2)

SWEEP = """
import importlib, json, sys
from pathlib import Path
for m in ("ccg.refresh", "ccg.build", "ccg.assemble", "ccg.assemble.rows", "ccg.linker", "ccg.packrt", "ccg.rulepack", "ccg.manifest", "ccg.units", "ccg.sandbox_audit",
          "ccg.atrest", "ccg.sysname", "ccg.store.acl", "ccg.store.loader", "ccg.engine.bfs", "ccg.engine.flow", "ccg.engine.reader", "ccg.engine.config", "extractors.joern.run"):
    importlib.import_module(m)
repo = Path(sys.argv[1]).resolve()
out = set()
for mod in list(sys.modules.values()):
    f = getattr(mod, "__file__", None)
    if f:
        try:
            rel = Path(f).resolve().relative_to(repo).as_posix()
        except ValueError:
            continue
        if ".venv" not in rel and "site-packages" not in rel:
            out.add(rel)
print(json.dumps(sorted(out)))
"""


def test_every_cpg_file_the_build_path_imports_or_reads_is_pinned_by_the_record_or_the_tool_line():
    """An unfiltered ``sys.modules`` sweep in a fresh interpreter of the build path (what ``build_and_load`` and ``LiveGraph`` import) plus the packs ``DEFAULT_PACKS`` read:
    each file is in the record's pins, in ``TOOL_FILES`` or in ``EXTRA_PIN_FILES``; the extra files exist and the record pins none of them."""
    import os
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(CPG), "HOME": str(CPG)}
    got = subprocess.run([sys.executable, "-c", SWEEP, str(sd.REPO)], capture_output=True, text=True, env=env, cwd=str(CPG), timeout=60, check=True).stdout
    imported = set(json.loads(got))
    from ccg import refresh
    packs = {p.relative_to(sd.REPO).as_posix() for n in refresh.DEFAULT_PACKS for p in (CPG / "packs" / n).glob("*.yaml")}
    pinned = set(sd.pinned_files(sd._line_of(sd.committed_record_lines(sd.REPO, sd.capture_head(sd.REPO)), sd.RECORD_ID)))
    assert imported - pinned - set(sd.TOOL_FILES) - set(sd.EXTRA_PIN_FILES) == set()
    assert packs <= set(sd.EXTRA_PIN_FILES) and len(packs) == 9
    assert set(sd.EXTRA_PIN_FILES) <= imported | packs, "an extra pin the build neither imports nor reads"
    assert not set(sd.EXTRA_PIN_FILES) & pinned and all((sd.REPO / f).is_file() for f in sd.EXTRA_PIN_FILES)
    assert len(sd.EXTRA_PIN_FILES) == 23 and len(set(sd.EXTRA_PIN_FILES)) == 23 and all((sd.REPO / f).is_file() for f in sd.TOOL_FILES) and len(sd.TOOL_FILES) == 3


def test_the_docstring_states_the_tool_line_shape():
    doc = sd.__doc__
    assert "extraPins" in doc and all(f in doc for f in ("sha256",)) and sd.TOOL_ID in doc


def test_the_tools_literals_are_the_records_and_the_servers():
    """Constants the record or the server fixes: a tool that drifts on one of them measures something else (the tests above read them through ``sd``, so only a literal pins them)."""
    assert (sd.RECORD_ID, sd.TOOL_ID, sd.SNAPSHOT_ID) == ("ccg-p1-flow-fallback-v1", "ccg-p1-flow-fallback-v1-tool", "ccg-p1-flow-fallback-v1-snapshot")
    assert (sd.RECORD_PATH, sd.EVAL_COMMIT_PATH, sd.RESULT_PATH) == ("cpg/eval/preregistration.jsonl", "cpg/eval/EVAL_COMMIT", "cpg/eval/results/ccg-p1-flow-fallback-v1.json")
    assert (sd.SYSTEM, sd.HOST, sd.FORBIDDEN_PORTS, sd.DEPTH_BOUND) == ("flowfb", "127.0.0.1", (6379, 6390), 2)
    assert sd.CANDIDATES == ("as-built", "V-flow", "V-both", "V-union") and sd.TIE_ORDER == ("as-built", "V-both", "V-flow", "V-union")
    assert sd.DIGEST_ALGORITHM == "ccg-flowfb-graph-digest-v1" and (sd.PAGE, sd.RESULTSET_SIZE) == (4000, 10000)
    assert sd.LANGUAGE_OF_EXT == {".py": "python", ".ts": "javascript", ".tsx": "javascript", ".js": "javascript"}
    assert sd.HEX64 == frozenset("0123456789abcdef")
