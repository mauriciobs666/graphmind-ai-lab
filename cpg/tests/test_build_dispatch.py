"""P1-44 — build dispatch by unit kind (`ccg/build.py`).

Plan: `docs/plans/connected-code-graph-p1-linking.md` § P1-44 behaviours (1)-(8). Repositories are real git repositories built in
`tmp_path`; the work area and the key area are throwaway 0700 directories, so no real bundle or key is touched. Joern is never
run: `joern_run.extract_unit` is replaced (recording) and the real normalizer runs over hand-built raw facts. Expected values are
written by hand; an extractor's output is only ever compared with the same extractor's own contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import sys
from collections import Counter
from pathlib import Path

import pytest

from ccg import atrest, bundle, ids, build
from ccg.manifest import Manifest, Policy, Repo
from ccg.units import Unit, discover_units
from extractors.common import writer_rules as wr
from extractors.common.errors import SourceRefused
from extractors.joern import run as joern_run
from extractors.joern.normalize import assemble_bundle, blame
from extractors.joern.normalize.core import RuntimeRefused
from extractors.markdown.extract import MAX_FILE_BYTES, MarkdownError
from extractors.markdown.globs import load_globs
from normalize_assembly_helpers import COMMIT_DATE, KINDS, SRC, full_facts

CPG = Path(__file__).resolve().parent.parent
AUDIT = "0f8fad5b-d9cb-469f-a165-70867728950e"
BUILT_AT = "2026-11-11T00:00:00Z"
SYSTEM = "sys1"
SECRET = "S3CR3T-planted-9f2c"
EVAL_GLOBS = load_globs(CPG / "eval" / "eval-manifest.yaml")
FX_AGENTS = CPG / "fixtures" / "system1" / "fx-svc" / ".claude" / "agents"
URL = "https://ntfy.sh/x7k2-q9w4-m1p8-z3v6"           # a capability URL: its entry carries bindParts


# --- helpers -----------------------------------------------------------------------------------------------------


def git(path: Path, *args: str, env=None) -> str:
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def make_repo(path: Path, files: dict[str, str | bytes]) -> tuple[Path, str]:
    """A git repository at ``path`` with ``files`` in one commit (fixed dates; bytes allowed); returns (path, commit id)."""
    env = {**os.environ, "GIT_AUTHOR_DATE": COMMIT_DATE, "GIT_COMMITTER_DATE": COMMIT_DATE, "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    path.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    git(path, "init", "-q", "-b", "main", env=env)
    git(path, "add", "-A", env=env)
    git(path, "commit", "-q", "-m", "c1", env=env)
    return path, git(path, "rev-parse", "HEAD")


def unit(lang: str, root: str, repo: str = "r") -> Unit:
    return Unit(ids.unit_id(repo, root, lang), repo, root, lang)


@pytest.fixture
def ctx_for(tmp_path):
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    keys = tmp_path / "keys"

    def make(units=(), **kw):
        kw.setdefault("audit_ids", (AUDIT,))
        kw.setdefault("built_at", BUILT_AT)
        kw.setdefault("globs", EVAL_GLOBS)               # the eval manifest's list: the system manifest's block is another unit's
        return build.BuildContext(units=tuple(units), root=work, keys_root=keys, **kw)

    make.work, make.keys = work, keys
    return make


def read(ctx, result: build.BundleResult) -> list[dict]:
    assert result.ok, result
    return bundle.loads(atrest.read_bundle(SYSTEM, result.unit.unitId, result.commit, ctx.root))


def of(records, t):
    return [r for r in records if r["type"] == t]


class Recorder:
    """Replaces a ``build`` module attribute; records every call and delegates to the original unless ``result`` is given."""

    def __init__(self, monkeypatch, name, result=None, raises=None):
        self.calls, self.orig, self.result, self.raises = [], getattr(build, name), result, raises
        monkeypatch.setattr(build, name, self)

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        if self.raises is not None:
            raise self.raises
        return self.orig(*a, **kw) if self.result is None else self.result


def joern_stub(monkeypatch, tmp_path, *, raises=None, files=None):
    """Replaces ``joern_run.extract_unit``: records its call and returns hand-built raw facts for a python unit (or raises)."""
    calls = []

    def stub(repo, u, units, commit, work, **kw):
        calls.append((u, commit))
        if raises is not None:
            raise raises
        facts = tmp_path / f"facts-{u.unitId}.jsonl"
        facts.write_text("".join(json.dumps(r) + "\n" for k in KINDS for r in full_facts()[k]), encoding="utf-8")
        return joern_run.RawFacts(u, commit, facts, (AUDIT,), Path(repo), tmp_path, None, tmp_path / "x", 1)

    monkeypatch.setattr(build.joern_run, "extract_unit", stub)
    return calls


@pytest.fixture
def py_repo(tmp_path):
    path, sha = make_repo(tmp_path / "pyrepo", {"r.py": SRC})
    return path, sha


# --- (1) one extractor per kind, actually invoked (behaviour 5) ----------------------------------------------------------


def test_a_python_unit_runs_the_joern_extractor_and_the_normalizer_and_writes_a_1_0_bundle(monkeypatch, tmp_path, py_repo, ctx_for):
    calls = joern_stub(monkeypatch, tmp_path)
    path, sha = py_repo
    u = unit("python", ".")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert calls == [(u, sha)]
    recs = read(ctx, res)
    assert (recs[0]["bundleSchemaVersion"], "bindKeyCheck" in recs[0], res.bundle_schema_version) == ("1.0", False, "1.0")
    assert recs[0]["tools"]["frontend"] == "pysrc2cpg" and of(recs, "symbol")


def test_a_javascript_unit_runs_the_joern_extractor(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "jsrepo", {"a.ts": "export const x = 1;\n"})
    calls = joern_stub(monkeypatch, tmp_path, raises=joern_run.RunnerError("x"))
    u = unit("javascript", ".")
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert calls == [(u, sha)] and (res.ok, res.fail_reason) == (False, "extract:failed")


def test_a_shell_unit_runs_the_shell_extractor_over_the_files_of_the_commit(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "sh", {"bin/run.sh": "#!/bin/sh\necho hi\n"})
    (path / "bin" / "run.sh").write_text("#!/bin/sh\necho CHANGED\n")           # the working tree is never read
    spy = Recorder(monkeypatch, "extract_shell")
    u = unit("shell", "bin/run.sh")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    ((a, kw),) = spy.calls
    assert kw["repo_id"] == "r" and kw["unit_id"] == u.unitId and a[1] == ["bin/run.sh"] and callable(kw["history"])
    recs = read(ctx, res)
    assert [r["path"] for r in of(recs, "file")] == ["bin/run.sh"] and recs[0]["tools"]["frontend"] == "tree-sitter-bash"
    assert "CHANGED" not in json.dumps(recs)


def test_a_config_unit_runs_the_config_extractor_and_writes_its_file_record(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "cfg", {"app.json": '{"name": "x"}\n'})
    spy = Recorder(monkeypatch, "extract_config")
    u = unit("config", "app.json")
    ctx = ctx_for([u])
    recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
    assert any(kw.get("key") for _, kw in spy.calls) and [a[1] for a, _ in spy.calls][-1] == {"app.json": b'{"name": "x"}\n'}
    (f,) = of(recs, "file")
    assert (f["path"], f["lang"], f["kind"], f["role"]) == ("app.json", "config", "config", "primary")
    assert f["contentHash"] == hashlib.sha256(b'{"name": "x"}\n').hexdigest() and of(recs, "configEntry")


def test_a_markdown_unit_runs_the_markdown_extractor_with_history_and_the_vocabulary(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "md", {".claude/agents/a.md": "Call the `get_item` tool.\n"})
    spy = Recorder(monkeypatch, "extract_markdown")
    u = unit("markdown", ".claude/agents/a.md")
    ctx = ctx_for([u], known_tools=frozenset({"get_item"}))
    recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
    ((a, kw),) = spy.calls
    assert a[0] == ".claude/agents/a.md" and kw["bundle_path"] is None and kw["known_tools"] == frozenset({"get_item"})
    assert kw["history"] == ("2026-01-02T03:04:05Z", sha)                   # blame over the whole file at the built commit
    assert [r["calleeName"] for r in of(recs, "callsite")] == ["get_item"]


def test_an_empty_markdown_file_takes_its_history_from_the_last_commit_that_touched_it(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "md", {"claude/e.md": ""})
    u = unit("markdown", "claude/e.md")
    ctx = ctx_for([u])
    (sym,) = of(read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx)), "symbol")
    assert (sym["lastChangedAt"], sym["lastChangedCommit"]) == ("2026-01-02T03:04:05Z", sha)


def test_an_unknown_kind_is_a_typed_error_and_reaches_no_extractor(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "x", {"a.go": "package a\n"})
    spies = [Recorder(monkeypatch, n) for n in ("extract_shell", "extract_config", "extract_markdown")]
    with pytest.raises(build.UnknownUnitKind):
        build.build_unit(unit("go", "a.go"), path, sha, SYSTEM, ctx_for())
    assert all(s.calls == [] for s in spies)


def test_a_host_unit_without_audit_ids_is_refused_before_any_work(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "x", {"a.json": "{}\n"})
    with pytest.raises(build.MissingAuditIds):
        build.build_unit(unit("config", "a.json"), path, sha, SYSTEM, ctx_for(audit_ids=()))


# --- (3) the failure mapping, value-free ---------------------------------------------------------------------------------


class NewExtractError(joern_run.ExtractError):
    """A subclass that did not exist when the mapping was written: the base class covers it."""


MAPPING = [
    (joern_run.RunnerTimeout("t"), "extract:timeout"),
    (joern_run.ExtractError("e"), "extract:failed"),
    (NewExtractError("n"), "extract:failed"),
    (joern_run.StageError("s"), "extract:failed"),
    (joern_run.RunnerError("r"), "extract:failed"),
    (joern_run.EmptyCpgError("c"), "extract:failed"),
    (SourceRefused("u"), "extract:failed"),
    (RuntimeRefused("v"), "extract:failed"),
    (assemble_bundle.BuildError("b"), "extract:failed"),
    (blame.HistoryError("h"), "extract:failed"),
    (bundle.BundleError("x"), "bundle:invalid"),
    (subprocess.CalledProcessError(1, ["git"], stderr=b"x"), "extract:failed"),
    (subprocess.TimeoutExpired(["git"], 1), "extract:failed"),
    (atrest.AtRestError("a"), "extract:failed"),
    (ZeroDivisionError("z"), "extract:failed"),
]


@pytest.mark.parametrize("exc, code", MAPPING, ids=[type(e).__name__ for e, _ in MAPPING])
def test_each_planted_exception_fails_its_unit_with_exactly_its_code(monkeypatch, tmp_path, py_repo, ctx_for, exc, code):
    joern_stub(monkeypatch, tmp_path, raises=exc)
    path, sha = py_repo
    u = unit("python", ".")
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert (res.ok, res.fail_reason, res.path) == (False, code, None)
    assert code in build.FAIL_CODES
    assert [p for p in ctx_for.work.rglob("*.bundle.jsonl")] == []


def test_a_shallow_clone_fails_with_its_code_before_any_extractor_runs(monkeypatch, tmp_path, ctx_for):
    src, _ = make_repo(tmp_path / "full", {"a.json": "{}\n"})
    (src / "b.json").write_text("{}\n")
    git(src, "add", "-A")
    git(src, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "c2")
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{src}", str(shallow)], check=True, capture_output=True)
    spy = Recorder(monkeypatch, "extract_config")
    calls = joern_stub(monkeypatch, tmp_path)
    for u in (unit("config", "a.json"), unit("python", ".")):
        res = build.build_unit(u, shallow, git(shallow, "rev-parse", "HEAD"), SYSTEM, ctx_for([u]))
        assert (res.ok, res.fail_reason) == (False, "build:shallow-clone")
    assert spy.calls == [] and calls == []


def test_a_bundle_the_validator_refuses_is_bundle_invalid(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "x", {"s.sh": "echo hi\n"})
    Recorder(monkeypatch, "extract_shell", result=[{"type": "callsite", "id": "not-hex"}])
    u = unit("shell", "s.sh")
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert (res.ok, res.fail_reason) == (False, "bundle:invalid")


def test_the_failure_is_value_free_a_message_with_a_secret_never_reaches_the_result(monkeypatch, tmp_path, py_repo, ctx_for):
    joern_stub(monkeypatch, tmp_path, raises=ValueError(f"token={SECRET}"))
    path, sha = py_repo
    u = unit("python", ".")
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert res.fail_reason == "extract:failed" and SECRET not in repr(res) and res.report == {}


def test_the_fail_codes_are_members_of_the_graph_constant_when_it_exists():
    staleness = pytest.importorskip("ccg.staleness")                  # P1-18 creates it; until then this cannot run
    assert build.FAIL_CODES <= set(staleness.FAIL_REASONS)


LATIN1_PY = "# caf\xe9 \xff\nx = 1\n".encode("latin-1")


def test_a_latin1_python_file_fails_its_unit_alone_with_no_source_byte_in_the_result(monkeypatch, tmp_path, ctx_for):
    bad, bsha = make_repo(tmp_path / "bad", {"r.py": SRC})
    (bad / "r.py").write_bytes(LATIN1_PY)                                   # the staged source is what the normalizer reads
    git(bad, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-am", "latin1")
    bsha = git(bad, "rev-parse", "HEAD")
    good, gsha = make_repo(tmp_path / "good", {"r.py": SRC})
    joern_stub(monkeypatch, tmp_path)
    ub, ug = unit("python", ".", "bad"), unit("python", ".", "good")
    stub_stage = {"bad": bad, "good": good}

    real = build.joern_run.extract_unit

    def with_stage(repo, u, units, commit, work, **kw):                      # point the staged tree at the repo's working tree
        raw = real(repo, u, units, commit, work, **kw)
        return joern_run.RawFacts(raw.unit, raw.commit, raw.facts, raw.sandbox_audit_ids, stub_stage[u.repoId], raw.cpg_dir,
                                  None, raw.content_check, 1)

    monkeypatch.setattr(build.joern_run, "extract_unit", with_stage)
    ctx = ctx_for([ub, ug])
    report = build.build_units([ub, ug], {"bad": bad, "good": good}, {"bad": bsha, "good": gsha}, SYSTEM, ctx)
    by = {r.unit.repoId: r for r in report.results}
    assert (by["bad"].ok, by["bad"].fail_reason) == (False, "extract:failed") and by["good"].ok
    shown = repr(by["bad"]) + json.dumps(dict(by["bad"].report))
    for leak in ("\xe9", "\xff", "caf", "xe9", "xff"):
        assert leak not in shown.lower()


# --- (6) markdown hand-offs ----------------------------------------------------------------------------------------------


def test_an_oversize_markdown_file_is_skipped_and_counted_while_its_unit_builds(tmp_path, ctx_for):
    big = "a" * (MAX_FILE_BYTES + 1)
    path, sha = make_repo(tmp_path / "md", {"claude/big.md": big, "claude/ok.md": "plain\n"})
    u = unit("markdown", "claude")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert res.ok and res.report["skippedDocs"] == 1
    assert [r["path"] for r in of(read(ctx, res), "file")] == ["claude/ok.md"]


def test_a_markdown_error_from_any_file_is_the_skip_not_a_unit_failure(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "md", {"claude/a.md": "x\n"})
    Recorder(monkeypatch, "extract_markdown", raises=MarkdownError("big"))
    u = unit("markdown", "claude/a.md")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert res.ok and res.report["skippedDocs"] == 1 and of(read(ctx, res), "file") == []


# --- (7) config call convention ------------------------------------------------------------------------------------------

COMPOSE = "services:\n  api:\n    image: x\n"
WEB = "services:\n  web:\n    environment:\n      API_HOST: api\n"


def config_repo(tmp_path):
    return make_repo(tmp_path / "cfg", {"compose.yaml": COMPOSE, "compose.override.yaml": WEB, "app.json": '{"a": 1}\n'})


def config_units():
    return [unit("config", p) for p in ("app.json", "compose.override.yaml", "compose.yaml")]


def test_extract_config_derives_the_declared_services_once_per_repository_over_all_its_compose_files(monkeypatch, tmp_path, ctx_for):
    path, sha = config_repo(tmp_path)
    units = config_units()
    spy = Recorder(monkeypatch, "extract_config")
    ctx = ctx_for(units)
    report = build.build_units(units, {"r": path}, {"r": sha}, SYSTEM, ctx)
    assert all(r.ok for r in report.results)
    repo_wide = [a[1] for a, kw in spy.calls if not kw.get("extra_declared_services")]
    assert len(repo_wide) == 1 and sorted(repo_wide[0]) == ["compose.override.yaml", "compose.yaml"]
    per_unit = [(sorted(a[1]), set(kw["extra_declared_services"])) for a, kw in spy.calls if kw.get("extra_declared_services")]
    assert sorted(f for f, _ in per_unit) == [["app.json"], ["compose.override.yaml"], ["compose.yaml"]]
    assert all(d == {"api", "web"} for _, d in per_unit)


def test_a_service_declared_in_another_file_of_the_repository_counts_for_w2(tmp_path, ctx_for):
    path, sha = config_repo(tmp_path)
    u = unit("config", "compose.override.yaml")
    ctx = ctx_for(config_units())
    entries = {r["keyPath"]: r.get("bindValue") for r in of(read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx)), "configEntry")}
    assert entries["services.web.environment.API_HOST"] == "api"
    alone = ctx_for([u])
    alone_path, alone_sha = make_repo(tmp_path / "cfg2", {"compose.override.yaml": WEB})
    entries = {r["keyPath"]: r.get("bindValue") for r in of(read(alone, build.build_unit(u, alone_path, alone_sha, SYSTEM, alone)), "configEntry")}
    assert entries["services.web.environment.API_HOST"] is None


def test_each_repository_gets_its_own_declared_set(monkeypatch, tmp_path, ctx_for):
    a, asha = make_repo(tmp_path / "a", {"compose.yaml": COMPOSE})
    b, bsha = make_repo(tmp_path / "b", {"compose.yaml": "services:\n  other:\n    image: x\n"})
    units = [unit("config", "compose.yaml", "a"), unit("config", "compose.yaml", "b")]
    spy = Recorder(monkeypatch, "extract_config")
    build.build_units(units, {"a": a, "b": b}, {"a": asha, "b": bsha}, SYSTEM, ctx_for())
    sets = {a_[0]: set(kw["extra_declared_services"]) for a_, kw in spy.calls if kw.get("extra_declared_services")}
    assert sets == {"a": {"api"}, "b": {"other"}}


def test_a_config_file_that_yields_no_records_still_gets_its_file_record_with_the_file_role(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"examples/broken.json": "{not json"})
    u = unit("config", "examples/broken.json")
    ctx = ctx_for([u])
    (f,) = of(read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx)), "file")
    assert (f["path"], f["role"]) == ("examples/broken.json", "example")


# --- (8) the header is built from the merged record set ------------------------------------------------------------------


def test_a_config_unit_with_a_url_entry_is_1_1_with_the_systems_key_check(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"a.json": json.dumps({"url": URL})})
    u = unit("config", "a.json")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    h = read(ctx, res)[0]                                                    # `loads` validates V1a-V1d
    assert (h["bundleSchemaVersion"], res.bundle_schema_version) == ("1.1", "1.1")
    assert h["bindKeyCheck"] == atrest.bind_key_check(atrest.hmac_key(SYSTEM, ctx.keys_root))


def test_a_config_unit_without_a_url_entry_is_1_0_with_no_key_check(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"a.json": json.dumps({"name": "x"})})
    u = unit("config", "a.json")
    ctx = ctx_for([u])
    h = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))[0]
    assert (h["bundleSchemaVersion"], "bindKeyCheck" in h) == ("1.0", False)


def test_a_shell_and_a_markdown_unit_are_1_0(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"s.sh": "echo hi\n", "claude/a.md": "hi\n"})
    for u in (unit("shell", "s.sh"), unit("markdown", "claude/a.md")):
        ctx = ctx_for([u])
        h = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))[0]
        assert (h["bundleSchemaVersion"], "bindKeyCheck" in h) == ("1.0", False)


def test_a_multi_file_config_unit_whose_bind_parts_come_from_two_files_gets_one_header(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"cfg/a.json": json.dumps({"url": URL}), "cfg/b.json": json.dumps({"url": URL + "-2"}),
                                           "cfg/c.json": json.dumps({"n": 1})})
    u = unit("config", "cfg")
    ctx = ctx_for([u])
    recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
    assert [r["type"] for r in recs].count("header") == 1
    with_parts = {r["path"] for r in of(recs, "configEntry") if "bindParts" in r}
    assert with_parts == {"cfg/a.json", "cfg/b.json"}
    assert (recs[0]["bundleSchemaVersion"], recs[0]["bindKeyCheck"]) == ("1.1", atrest.bind_key_check(atrest.hmac_key(SYSTEM, ctx.keys_root)))
    assert sorted(r["path"] for r in of(recs, "file")) == ["cfg/a.json", "cfg/b.json", "cfg/c.json"]


def test_the_header_version_follows_the_merged_set_not_one_files_header_fields(tmp_path, ctx_for):
    """c.json alone has no parts (its own header fields are `{}`); the merged set has some, so the unit is 1.1."""
    path, sha = make_repo(tmp_path / "c", {"cfg/a.json": json.dumps({"n": 1}), "cfg/z.json": json.dumps({"url": URL})})
    u = unit("config", "cfg")
    ctx = ctx_for([u])
    assert read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))[0]["bundleSchemaVersion"] == "1.1"


def test_the_key_is_read_through_atrest_only(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"a.json": json.dumps({"url": URL})})
    seen = []
    real = atrest.hmac_key
    monkeypatch.setattr(atrest, "hmac_key", lambda system, keys_root=None: (seen.append(system), real(system, keys_root))[1])
    u = unit("config", "a.json")
    build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert seen and set(seen) == {SYSTEM}


# --- (1), (c) code first, known_tools once, then the markdown units -------------------------------------------------------


def synthetic_mcp_bundle(system_repo: str, tools: list[str]) -> list[dict]:
    """A schema-valid python bundle whose functions are `@mcp.tool()` handlers under a `mcp.server.fastmcp` import."""
    u = unit("python", ".", system_repo)
    path = "mcp/server.py"
    fid = ids.file_id(system_repo, path)
    recs = [{"type": "file", "id": fid, "repoId": system_repo, "unitId": u.unitId, "path": path, "lang": "python", "kind": "code",
             "role": "primary", "contentHash": "0" * 64, "lineCount": 10 * len(tools) + 2},
            {"type": "import", "id": ids.import_id(fid, 0), "fileId": fid, "line": 1, "ordinal": 0, "module": "mcp.server.fastmcp",
             "names": [{"name": "FastMCP"}], "dynamic": False}]
    for n, tool in enumerate(tools):
        qname = f"{path}::{tool}"
        sid = ids.symbol_id("python", system_repo, qname)
        recs.append({"type": "symbol", "id": sid, "lang": "python", "repoId": system_repo, "qname": qname, "fileId": fid, "path": path,
                     "kind": "function", "name": tool, "nameNorm": tool, "lineStart": 10 * n + 3, "lineEnd": 10 * n + 8,
                     "fingerprint": "1" * 64, "complexity": 1,
                     "lastChangedAt": "2026-01-02T03:04:05Z", "lastChangedCommit": "1" * 40})
        recs.append({"type": "decorator", "id": ids.decorator_id(sid, 0), "targetId": sid, "ordinal": 0, "form": "decorator",
                     "name": "mcp.tool", "line": 10 * n + 2, "args": []})
    counts = Counter(r["type"] for r in recs)
    header = {"type": "header", "bundleSchemaVersion": "1.0", "unit": {"repoId": system_repo, "rootPath": ".", "lang": "python"},
              "unitId": u.unitId, "builtCommit": "1" * 40, "builtTree": "2" * 40, "builtAt": BUILT_AT,
              "tools": {"joern": "4.0.579", "frontend": "pysrc2cpg", "frontendVersion": "4.0.579", "normalizer": "n",
                        "normalizerVersion": "0", "extractScript": "extract.sc", "extractScriptHash": "0" * 64},
              "sandboxAuditIds": [AUDIT], "egressMode": "offline", "flowProjected": True, "fieldDepthCap": 3, "vendoredDirs": [],
              "history": "available", "counts": {t: counts.get(t, 0) for t in bundle.RECORD_TYPES[1:]}}
    return [header, *recs]


def test_known_tools_are_the_fastmcp_packs_provided_tool_names_over_code_bundles():
    got = build.known_tools_of([synthetic_mcp_bundle("fx-svc", ["get_item", "list_orders", "reindex"])])
    assert got == frozenset({"get_item", "list_orders", "reindex"})
    assert build.known_tools_of([]) == frozenset()


def fx_svc_repo(tmp_path):
    files = {f".claude/agents/{n}.md": (FX_AGENTS / f"{n}.md").read_text() for n in ("agent", "tools", "exec")}
    return make_repo(tmp_path / "fx-svc", files | {"pyproject.toml": "[project]\nname='x'\n", "mcp/__init__.py": ""})


def stub_code_build(monkeypatch, tools):
    """Replace only the Joern step of a code unit with a synthetic, valid bundle written through ``atrest.write_bundle``."""
    def fake(ctx, u, repo, sha, system, report):
        recs = synthetic_mcp_bundle(u.repoId, tools)
        recs[0].update(builtCommit=sha)
        return atrest.write_bundle(system, u.unitId, sha, recs, root=ctx.root), "1.0"

    monkeypatch.setattr(build, "_build_code", fake)


def test_the_fx_svc_prompt_files_yield_the_ac6_pm_call_sites_end_to_end(monkeypatch, tmp_path, ctx_for):
    path, sha = fx_svc_repo(tmp_path)
    stub_code_build(monkeypatch, ["get_item", "list_orders", "reindex"])
    manifest = Manifest(system=SYSTEM, repos=(Repo(repo_id="fx-svc", path=".", policy=Policy(egress="local-only")),))
    units = discover_units(path, manifest)
    ctx = ctx_for()
    rep = build.build_units(units, {"fx-svc": path}, {"fx-svc": sha}, SYSTEM, ctx)
    assert all(r.ok for r in rep.results), [(r.unit.language, r.unit.rootPath, r.fail_reason) for r in rep.results]
    sites = {}
    for r in rep.results:
        if r.unit.language == "markdown":
            for cs in of(read(ctx, r), "callsite"):
                sites[r.unit.rootPath] = (cs["calleeName"], cs["callKind"], cs["status"], cs["reason"])
    assert sites == {
        ".claude/agents/agent.md": ("get_item", "mention", "possible", "prompt-mention"),
        ".claude/agents/tools.md": ("reindex", "mention", "possible", "prompt-mention"),
        ".claude/agents/exec.md": ("mcp__fx-svc-exec__list_orders", "mention", "possible", "prompt-mention"),
    }
    assert rep.known_tools == frozenset({"get_item", "list_orders", "reindex"})


def test_without_the_code_bundles_the_vocabulary_is_empty_and_bare_names_are_not_mentions(monkeypatch, tmp_path, ctx_for):
    path, sha = fx_svc_repo(tmp_path)
    u = unit("markdown", ".claude/agents/agent.md", "fx-svc")
    ctx = ctx_for([u])
    assert of(read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx)), "callsite") == []


def test_code_units_are_built_before_shell_config_and_markdown_and_known_tools_is_computed_once(monkeypatch, tmp_path, ctx_for):
    path, sha = fx_svc_repo(tmp_path)
    order = []
    stub_code_build(monkeypatch, ["get_item"])
    real_unit, real_tools = build.build_unit, build.known_tools_of
    seen = []

    def spy_unit(u, repo, commit, system, ctx=None):
        order.append(u.language)
        if u.language == "markdown":
            seen.append(ctx.known_tools)
        return real_unit(u, repo, commit, system, ctx)

    tools = Recorder(monkeypatch, "known_tools_of")
    units = [unit("markdown", ".claude/agents/agent.md", "fx-svc"), unit("markdown", ".claude/agents/tools.md", "fx-svc"),
             unit("config", "pyproject.toml", "fx-svc"), unit("python", ".", "fx-svc")]
    build.build_units(units, {"fx-svc": path}, {"fx-svc": sha}, SYSTEM, ctx_for(), build=spy_unit)
    assert order == ["python", "config", "markdown", "markdown"]
    assert len(tools.calls) == 1 and seen == [frozenset({"get_item"})] * 2
    assert len(tools.calls[0][0][0]) == 1                                    # the one code bundle that was built


def test_a_failed_code_unit_leaves_its_bundle_out_of_the_vocabulary_and_does_not_stop_the_rest(monkeypatch, tmp_path, ctx_for):
    path, sha = fx_svc_repo(tmp_path)
    joern_stub(monkeypatch, tmp_path, raises=joern_run.RunnerTimeout("t"))
    units = [unit("python", ".", "fx-svc"), unit("markdown", ".claude/agents/agent.md", "fx-svc")]
    rep = build.build_units(units, {"fx-svc": path}, {"fx-svc": sha}, SYSTEM, ctx_for())
    assert [(r.unit.language, r.ok, r.fail_reason) for r in rep.results] == [
        ("python", False, "extract:timeout"), ("markdown", True, None)]
    assert rep.known_tools == frozenset()


def test_build_units_sums_the_skipped_docs_into_the_report(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "md", {"claude/big.md": "a" * (MAX_FILE_BYTES + 1), "claude/ok.md": "x\n"})
    units = [unit("markdown", "claude/big.md"), unit("markdown", "claude/ok.md")]
    rep = build.build_units(units, {"r": path}, {"r": sha}, SYSTEM, ctx_for())
    assert all(r.ok for r in rep.results) and rep.skipped_docs == 1


# --- (a) dot directories ---------------------------------------------------------------------------------------------------


def test_dot_directories_are_discovered_and_built_not_skipped(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "dot", {".claude/agents/exec.md": "run `mcp__s__t`\n", ".claude/settings.json": '{"a": 1}\n',
                                              ".github/run.sh": "echo hi\n"})
    manifest = Manifest(system=SYSTEM, repos=(Repo(repo_id="r", path=".", policy=Policy(egress="local-only")),))
    units = discover_units(path, manifest)
    assert {u.rootPath for u in units} == {".claude/agents/exec.md", ".claude/settings.json", ".github/run.sh"}
    ctx = ctx_for()
    rep = build.build_units(units, {"r": path}, {"r": sha}, SYSTEM, ctx)
    built = {f["path"] for r in rep.results for f in of(read(ctx, r), "file")}
    assert built == {".claude/agents/exec.md", ".claude/settings.json", ".github/run.sh"}


def test_a_directory_unit_includes_the_files_below_its_dot_directories(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "dot", {"p/.claude/agents/a.md": "x\n", "p/node_modules/b.md": "y\n", "p/docs/c.md": "z\n"})
    u = unit("markdown", "p")
    ctx = ctx_for([u])
    paths = {f["path"] for f in of(read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx)), "file")}
    assert paths == {"p/.claude/agents/a.md", "p/docs/c.md"}                # a vendored directory is skipped, a dot directory is not


# --- (h) bundles through atrest only --------------------------------------------------------------------------------------


def test_bundles_are_private_files_in_the_work_area(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "x", {"a.json": "{}\n"})
    u = unit("config", "a.json")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert res.path == atrest.bundle_path(SYSTEM, u.unitId, sha, ctx.root) and (res.path.stat().st_mode & 0o777) == 0o600
    assert not any(p.suffix == ".tmp" for p in res.path.parent.iterdir())


def test_the_module_writes_no_file_and_reads_no_key_except_through_atrest():
    src = (CPG / "ccg" / "build.py").read_text()
    for banned in ("open(", "write_text(", "token_bytes", ".hmac\"", "write_private"):
        assert banned not in src
    assert src.count("write_bytes(") == 1                                     # the shell extractor's throwaway stage, nothing else


# === fix batch: the analyst's review of P1-44 (`docs/reviews/connected-code-graph-p1-44.md`) ======================================


def _tree_bytes(entries: list[tuple[str, str, bytes]]) -> bytes:
    return b"".join(f"{mode} ".encode() + name.encode("utf-8", "surrogateescape") + b"\0" + oid for mode, name, oid in entries)


def hostile_repo(path: Path, files: dict[tuple[str, ...], bytes]) -> tuple[Path, str]:
    """A one-commit repository whose tree holds entries a normal ``git add`` never makes (``..``, empty, ``.git``, long names):
    trees are written with ``hash-object --literally`` and keyed by *segment tuples*, so a segment is one tree-entry name."""
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")

    def hash_object(kind: str, data: bytes) -> bytes:
        out = subprocess.run(["git", "-C", str(path), "hash-object", "-w", "--literally", "-t", kind, "--stdin"], input=data,
                             check=True, capture_output=True).stdout.decode().strip()
        return bytes.fromhex(out)

    def write(prefix: tuple[str, ...]) -> bytes:
        names: dict[str, bytes] = {}
        for segs, data in files.items():
            if segs[: len(prefix)] != prefix or len(segs) <= len(prefix):
                continue
            head = segs[len(prefix)]
            if len(segs) == len(prefix) + 1:
                names[head] = hash_object("blob", data)
            elif head not in names:
                names[head] = write(prefix + (head,))
        rows = [("40000" if _is_dir(files, prefix, n) else "100644", n, o) for n, o in names.items()]
        rows.sort(key=lambda r: r[1].encode("utf-8", "surrogateescape") + (b"/" if r[0] == "40000" else b""))     # git's tree order
        return hash_object("tree", _tree_bytes(rows))

    root = write(())
    env = {**os.environ, "GIT_AUTHOR_DATE": COMMIT_DATE, "GIT_COMMITTER_DATE": COMMIT_DATE, "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    sha = git(path, "commit-tree", root.hex(), "-m", "hostile", env=env)
    git(path, "update-ref", "refs/heads/main", sha)
    return path, sha


def _is_dir(files, prefix, name) -> bool:
    return any(len(s) > len(prefix) + 1 and s[: len(prefix) + 1] == prefix + (name,) for s in files)


@pytest.fixture
def tmp_base(tmp_path, monkeypatch):
    """The temp directory every stage writes under: nested, so a write one or two levels up is observable; returns the sandbox."""
    base = tmp_path / "tb" / "x" / "y"
    base.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(base))
    return tmp_path / "tb"


def files_under(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() or p.is_symlink())


HOSTILE = {
    "dotdot": ("a", "..", "..", "evil.sh"),
    "dotdot-one": ("a", "..", "evil.sh"),
    "dot": ("a", ".", "evil.sh"),
    "empty-segment": ("a", "", "evil.sh"),
    "git-dir-case": ("sub", ".GIT", "evil.sh"),
    "backslash": ("a\\..\\evil.sh",),
    "newline": ("a\nb.sh",),
    "very-long": ("a" * 300 + ".sh",),
    "very-long-path": tuple(["d" * 200] * 25 + ["x.sh"]),
}


def _hostile_cases():
    for label, segs in sorted(HOSTILE.items()):
        roots = ["."] + ([segs[0]] if len(segs) > 1 else [])               # the unit that selects the hostile name
        for lang in ("shell", "config", "markdown"):
            for root in roots:
                yield label, lang, root


@pytest.mark.parametrize("label, lang, root", list(_hostile_cases()))
def test_a_hostile_tree_path_inside_a_unit_fails_that_unit_value_free_and_writes_no_byte_outside_the_stage(tmp_path, ctx_for, tmp_base,
                                                                                                          label, lang, root):
    segs = HOSTILE[label]
    ext = {"shell": ".sh", "config": ".json", "markdown": ".md"}[lang]
    segs = segs[:-1] + (segs[-1].replace(".sh", "") + ext,) if segs[-1].endswith(".sh") else segs
    path, sha = hostile_repo(tmp_path / "h", {segs: b"echo PWNED\n", ("ok" + ext,): b"echo ok\n"})
    u = unit(lang, root)
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert (res.ok, res.fail_reason, res.path) == (False, "extract:failed", None)
    assert "PWNED" not in repr(res) and files_under(tmp_base) == []
    assert [p for p in ctx_for.work.rglob("*.bundle.jsonl")] == []
    assert [p for p in tmp_base.rglob("ccg-shell-*")] == []


UNRELATED = ["unrelated\x01.txt", ".GIT/config", "x" * 300, "back\\slash.txt", "a\nb.txt", "d/../e.txt"]


@pytest.mark.parametrize("odd", UNRELATED)
def test_an_odd_name_outside_every_unit_fails_no_unit_of_the_repository(tmp_path, ctx_for, tmp_base, odd):
    segs = tuple(odd.split("/")) if odd.startswith((".GIT", "d/")) else (odd,)
    path, sha = hostile_repo(tmp_path / "h", {segs: b"x\n", ("ok.json",): b"{}\n", ("ok.sh",): b"echo ok\n", ("ok.md",): b"hi\n"})
    units = [unit("config", "ok.json"), unit("shell", "ok.sh"), unit("markdown", "ok.md"), unit("config", "."), unit("shell", ".")]
    rep = build.build_units(units, {"r": path}, {"r": sha}, SYSTEM, ctx_for(units))
    assert [(r.unit.language, r.unit.rootPath, r.ok) for r in rep.results] == [
        (u.language, u.rootPath, True) for u in sorted(units, key=lambda u: (build._ORDER[u.language], u.repoId, u.rootPath))]
    assert files_under(tmp_base) == []


def test_a_hostile_name_inside_a_git_directory_is_skipped_like_any_vendored_file(tmp_path, ctx_for, tmp_base):
    path, sha = hostile_repo(tmp_path / "h", {(".git", "hooks", "evil.sh"): b"echo PWNED\n", ("ok.sh",): b"echo ok\n"})
    u = unit("shell", ".")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert [f["path"] for f in of(read(ctx, res), "file")] == ["ok.sh"] and files_under(tmp_base) == []


def test_a_hostile_unit_root_is_refused_by_the_file_selection_itself(tmp_path):
    path, sha = make_repo(tmp_path / "h", {"ok.sh": "echo\n"})
    tree = build._tree(build.BuildContext(), path, sha)
    for root in ("../ok.sh", "a/../ok.sh", "/abs", "a\\b"):
        with pytest.raises(SourceRefused):
            build._unit_files(path, tree, unit("shell", root))
    assert build._unit_files(path, tree, unit("shell", "ok.sh")) == ["ok.sh"]


def test_a_hostile_unit_root_fails_its_unit(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "h", {"ok.sh": "echo\n"})
    u = unit("shell", "../ok.sh")
    assert build.build_unit(u, path, sha, SYSTEM, ctx_for([u])).fail_reason == "extract:failed"


@pytest.mark.parametrize("name, ok", [("a" * 255, True), ("a" * 256, False), ("\u00e9" * 127, True), ("\u00e9" * 128, False),
                                      ("a\x1fb", False), ("a\x20b", True), ("a\x7fb", False), ("a\x7eb", True),
                                      ("a\x00b", False), ("a/b", True)])
def test_the_tree_path_rule_at_its_boundaries(name, ok):
    if ok:
        assert build._check_tree_path(name) == name
    else:
        with pytest.raises(SourceRefused):
            build._check_tree_path(name)


def test_a_path_of_exactly_4096_bytes_is_accepted_and_one_more_refused():
    segs = ["d" * 255] * 15 + ["e" * 200, "f" * 55]                          # 4080 bytes of names + 16 slashes
    name = "/".join(segs)
    assert len(name.encode()) == 4096 and build._check_tree_path(name) == name
    with pytest.raises(SourceRefused):
        build._check_tree_path(name + "f")


def test_an_absolute_tree_path_writes_nothing_at_its_target(tmp_path, ctx_for, tmp_base):
    target = tmp_path / "abs-target"
    segs = ("",) + tuple(target.parts[1:]) + ("evil.sh",)
    path, sha = hostile_repo(tmp_path / "h", {segs: b"echo PWNED\n"})
    u = unit("shell", ".")
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert (res.ok, res.fail_reason) == (False, "extract:failed") and not target.exists() and files_under(tmp_base) == []


def test_the_dotdot_probe_wrote_one_level_up_before_the_fix_so_the_test_has_teeth(tmp_path, ctx_for, tmp_base, monkeypatch):
    """Positive control: with the tree validation switched off the same tree writes ``evil.sh`` above the stage directory."""
    monkeypatch.setattr(build, "_check_tree_path", lambda p: p)
    monkeypatch.setattr(build, "_assert_inside", lambda dest, root: None)
    path, sha = hostile_repo(tmp_path / "h", {("a", "..", "..", "evil.sh"): b"echo PWNED\n"})
    u = unit("shell", ".")
    build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert files_under(tmp_base) == ["x/y/evil.sh"] or "evil.sh" in " ".join(files_under(tmp_base))


def test_a_stage_destination_that_resolves_outside_the_stage_is_refused_before_the_write(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (tmp_path / "out").mkdir()
    (stage / "link").symlink_to(tmp_path / "out")
    with pytest.raises(SourceRefused):
        build._assert_inside(stage / "link" / "x.sh", stage)
    build._assert_inside(stage / "sub" / "x.sh", stage)


def test_a_hostile_path_in_one_repository_leaves_the_other_repositorys_units_building(tmp_path, ctx_for, tmp_base):
    bad, bsha = hostile_repo(tmp_path / "bad", {("a", "..", "..", "evil.sh"): b"x\n"})
    good, gsha = make_repo(tmp_path / "good", {"s.sh": "echo hi\n"})
    ub, ug = unit("shell", ".", "bad"), unit("shell", "s.sh", "good")
    rep = build.build_units([ub, ug], {"bad": bad, "good": good}, {"bad": bsha, "good": gsha}, SYSTEM, ctx_for())
    assert {r.unit.repoId: (r.ok, r.fail_reason) for r in rep.results} == {"bad": (False, "extract:failed"), "good": (True, None)}


def test_awkward_but_harmless_names_build_a_leading_dash_a_case_collision_and_a_name_with_spaces(tmp_path, ctx_for, tmp_base):
    path, sha = hostile_repo(tmp_path / "h", {("-rf.sh",): b"echo a\n", ("A.sh",): b"echo b\n", ("a.sh",): b"echo c\n",
                                              ("with space.sh",): b"echo d\n"})
    u = unit("shell", ".")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert sorted(f["path"] for f in of(read(ctx, res), "file")) == ["-rf.sh", "A.sh", "a.sh", "with space.sh"]
    assert files_under(tmp_base) == []


# --- M1: the commit is read, never the working tree, for config and markdown too ---------------------------------------


def test_a_config_unit_reads_the_commit_not_the_working_tree(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"app.json": '{"name": "committed"}\n'})
    (path / "app.json").write_text('{"name": "worktree"}\n')
    u = unit("config", "app.json")
    ctx = ctx_for([u])
    recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
    (f,) = of(recs, "file")
    assert f["contentHash"] == hashlib.sha256(b'{"name": "committed"}\n').hexdigest()
    assert "worktree" not in json.dumps(recs)


def test_a_compose_file_of_a_config_unit_is_read_from_the_commit_too(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"compose.yaml": COMPOSE, "compose.override.yaml": WEB})
    (path / "compose.yaml").write_text("services:\n  other:\n    image: x\n")        # the work tree no longer declares `api`
    units = [unit("config", "compose.override.yaml"), unit("config", "compose.yaml")]
    ctx = ctx_for(units)
    entries = {r["keyPath"]: r.get("bindValue") for r in of(read(ctx, build.build_unit(units[0], path, sha, SYSTEM, ctx)), "configEntry")}
    assert entries["services.web.environment.API_HOST"] == "api"


def test_a_markdown_unit_reads_the_commit_not_the_working_tree(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "m", {"claude/a.md": "committed text\n"})
    (path / "claude" / "a.md").write_text("worktree text, much longer than before\n")
    u = unit("markdown", "claude/a.md")
    ctx = ctx_for([u])
    recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
    (f,) = of(recs, "file")
    assert f["contentHash"] == hashlib.sha256(b"committed text\n").hexdigest() and "worktree" not in json.dumps(recs)


# --- M2: symlinks and gitlinks are excluded, never followed ---------------------------------------------------------------


def special_repo(tmp_path):
    path, _ = make_repo(tmp_path / "sp", {"real.sh": "echo ok\n", "real.json": "{}\n", "real.md": "x\n", "target.txt": "T\n"})
    outside = tmp_path / "secret.json"
    outside.write_text('{"leak": "SYMLINK-TARGET-CONTENT"}\n')
    for name, tgt in (("link.sh", "real.sh"), ("link.json", str(outside)), ("link.md", "real.md"), ("noext", "real.sh")):
        os.symlink(tgt, path / name)
    git(path, "add", "-A")
    git(path, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},sub.sh")
    git(path, "update-index", "--add", "--cacheinfo", f"160000,{'2' * 40},mod.json")
    git(path, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "special")
    return path, git(path, "rev-parse", "HEAD")


@pytest.mark.parametrize("lang, expected", [("shell", ["real.sh"]), ("config", ["real.json"]), ("markdown", ["real.md"])])
def test_symlink_and_gitlink_rows_are_excluded_from_a_directory_unit(tmp_path, ctx_for, lang, expected):
    path, sha = special_repo(tmp_path)
    u = unit(lang, ".")
    ctx = ctx_for([u])
    recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
    assert [f["path"] for f in of(recs, "file")] == expected
    assert "SYMLINK-TARGET-CONTENT" not in json.dumps(recs)


@pytest.mark.parametrize("lang, root", [("shell", "link.sh"), ("config", "link.json"), ("markdown", "link.md"), ("shell", "sub.sh"),
                                        ("config", "mod.json")])
def test_a_unit_rooted_at_a_symlink_or_a_gitlink_builds_an_empty_bundle(tmp_path, ctx_for, lang, root):
    path, sha = special_repo(tmp_path)
    u = unit(lang, root)
    ctx = ctx_for([u])
    recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
    assert of(recs, "file") == [] and "SYMLINK-TARGET-CONTENT" not in json.dumps(recs)


def test_the_regular_modes_are_exactly_the_two_blob_modes():
    assert tuple(build._REGULAR) == ("100644", "100755")


# --- M3: one undecodable Compose file fails its own unit only ---------------------------------------------------------------


def test_a_non_utf8_compose_file_fails_its_own_unit_and_every_other_config_unit_builds(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"compose.yaml": b"\xff\xfe\x00bad", "compose.override.yaml": WEB, "app.json": '{"a": 1}\n',
                                           ".mcp.json": '{"mcpServers": {"s": {"command": "x"}}}\n'})
    units = [unit("config", p) for p in ("app.json", ".mcp.json", "compose.override.yaml", "compose.yaml")]
    rep = build.build_units(units, {"r": path}, {"r": sha}, SYSTEM, ctx_for())
    by = {r.unit.rootPath: (r.ok, r.fail_reason) for r in rep.results}
    assert by == {"compose.yaml": (False, "extract:failed"), "app.json": (True, None), ".mcp.json": (True, None),
                  "compose.override.yaml": (True, None)}


def test_services_declared_only_in_an_undecodable_compose_file_are_unknown_so_w2_binds_nothing_for_them(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"compose.yaml": b"\xff\xfe\x00bad", "compose.override.yaml": WEB})
    u = unit("config", "compose.override.yaml")
    ctx = ctx_for([u, unit("config", "compose.yaml")])
    entries = {r["keyPath"]: r.get("bindValue") for r in of(read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx)), "configEntry")}
    assert entries["services.web.environment.API_HOST"] is None


# --- M4: the runner's own exit 124 is a timeout when its duration says the cap fired -----------------------------------------


def test_build_unit_maps_the_runners_own_124_to_extract_timeout(monkeypatch, tmp_path, py_repo, ctx_for):
    joern_stub(monkeypatch, tmp_path, raises=joern_run.RunnerTimeout("cap"))          # what extract_unit raises for 124 at the cap
    path, sha = py_repo
    u = unit("python", ".")
    assert build.build_unit(u, path, sha, SYSTEM, ctx_for([u])).fail_reason == "extract:timeout"


# --- M5: an oversize markdown file is skipped before any blame -------------------------------------------------------------


def test_an_oversize_markdown_file_is_never_blamed_so_a_failing_blame_cannot_fail_the_unit(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "md", {"c/big.md": "a" * (MAX_FILE_BYTES + 1), "c/ok.md": "plain\n"})
    blamed = []
    real = blame.blame

    def spy(repo, commit, rel):
        blamed.append(rel)
        if rel == "c/big.md":
            raise blame.HistoryError("planted")
        return real(repo, commit, rel)

    monkeypatch.setattr(blame, "blame", spy)
    u = unit("markdown", "c")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert res.ok and res.report["skippedDocs"] == 1 and blamed == ["c/ok.md"]
    assert [f["path"] for f in of(read(ctx, res), "file")] == ["c/ok.md"]


# --- M6: size and count bounds on what is read from the tree --------------------------------------------------------------------


def blob_reads(monkeypatch):
    reads = []
    real = build._blob
    monkeypatch.setattr(build, "_blob", lambda repo, oid: (reads.append(oid), real(repo, oid))[1])
    return reads


def test_an_oversize_shell_blob_fails_the_unit_before_it_is_fetched_or_written(monkeypatch, tmp_path, ctx_for, tmp_base):
    monkeypatch.setattr(build, "MAX_SHELL_BLOB_BYTES", 16)
    path, sha = make_repo(tmp_path / "s", {"big.sh": "echo " + "a" * 40 + "\n", "ok.sh": "echo\n"})
    reads = blob_reads(monkeypatch)
    big, ok = unit("shell", "big.sh"), unit("shell", "ok.sh")
    results = {u.rootPath: build.build_unit(u, path, sha, SYSTEM, ctx_for([u])) for u in (big, ok)}
    assert (results["big.sh"].ok, results["big.sh"].fail_reason) == (False, "extract:failed") and results["ok.sh"].ok
    assert len(reads) == 1 and files_under(tmp_base) == []                  # only ok.sh's blob was ever fetched


def test_an_oversize_markdown_blob_is_skipped_and_counted_without_being_fetched(monkeypatch, tmp_path, ctx_for):
    monkeypatch.setattr(build, "MAX_MARKDOWN_BLOB_BYTES", 16)
    path, sha = make_repo(tmp_path / "m", {"c/big.md": "a" * 40 + "\n", "c/ok.md": "x\n"})
    reads = blob_reads(monkeypatch)
    u = unit("markdown", "c")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert res.ok and res.report["skippedDocs"] == 1 and len(reads) == 1
    assert [f["path"] for f in of(read(ctx, res), "file")] == ["c/ok.md"]


def test_an_oversize_config_blob_is_skipped_and_counted_without_being_fetched(monkeypatch, tmp_path, ctx_for):
    monkeypatch.setattr(build, "MAX_CONFIG_BLOB_BYTES", 16)
    path, sha = make_repo(tmp_path / "c", {"c/big.json": '{"a": "' + "x" * 40 + '"}\n', "c/ok.json": "{}\n"})
    reads = blob_reads(monkeypatch)
    u = unit("config", "c")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert res.ok and res.report["skippedFiles"] == 1 and len(reads) == 1
    assert [f["path"] for f in of(read(ctx, res), "file")] == ["c/ok.json"]


def test_an_oversize_compose_blob_is_not_fetched_for_the_declared_set(monkeypatch, tmp_path, ctx_for):
    monkeypatch.setattr(build, "MAX_CONFIG_BLOB_BYTES", 64)
    path, sha = make_repo(tmp_path / "c", {"compose.yaml": COMPOSE + "# " + "x" * 200 + "\n", "compose.override.yaml": WEB})
    reads = blob_reads(monkeypatch)
    u = unit("config", "compose.override.yaml")
    ctx = ctx_for([u, unit("config", "compose.yaml")])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert res.ok and git(path, "rev-parse", f"{sha}:compose.yaml") not in reads and reads


@pytest.mark.parametrize("cap, value", [("MAX_UNIT_FILES", 2), ("MAX_UNIT_BYTES", 20)])
def test_a_unit_over_the_file_count_or_total_byte_cap_fails_before_any_read(monkeypatch, tmp_path, ctx_for, cap, value):
    monkeypatch.setattr(build, cap, value)
    path, sha = make_repo(tmp_path / "c", {f"d/f{i}.json": '{"k": "0123456789"}\n' for i in range(3)})
    reads = blob_reads(monkeypatch)
    u = unit("config", "d")
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert (res.ok, res.fail_reason) == (False, "extract:failed") and reads == []


def test_a_unit_at_the_caps_still_builds(monkeypatch, tmp_path, ctx_for):
    monkeypatch.setattr(build, "MAX_UNIT_FILES", 3)
    path, sha = make_repo(tmp_path / "c", {f"d/f{i}.json": "{}\n" for i in range(3)})
    u = unit("config", "d")
    assert build.build_unit(u, path, sha, SYSTEM, ctx_for([u])).ok


def test_an_extensionless_blob_over_the_shell_cap_is_not_fetched_to_look_for_a_shebang(monkeypatch, tmp_path, ctx_for):
    monkeypatch.setattr(build, "MAX_SHELL_BLOB_BYTES", 16)
    path, sha = make_repo(tmp_path / "c", {"d/big": "#!/bin/sh\n" + "x" * 40 + "\n", "d/s": "#!/bin/sh\n"})
    reads = blob_reads(monkeypatch)
    u = unit("shell", "d")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    assert res.ok and [f["path"] for f in of(read(ctx, res), "file")] == ["d/s"]
    assert len(reads) == 2                                                  # d/s: once to classify, once to extract


# --- minors ------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lang, root, files", [("shell", "s.sh", {"s.sh": "echo\n"}), ("config", "a.json", {"a.json": "{}\n"}),
                                               ("markdown", "m.md", {"m.md": "x\n"}), ("shell", "d", {"d/a.sh": "echo\n"})])
def test_built_tree_is_the_git_tree_id_of_the_unit_root(tmp_path, ctx_for, lang, root, files):
    path, sha = make_repo(tmp_path / "t", files)
    u = unit(lang, root)
    ctx = ctx_for([u])
    h = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))[0]
    assert h["builtTree"] == git(path, "rev-parse", f"{sha}:{root}") and h["builtTree"] != sha
    whole = unit(lang, ".")
    ctx = ctx_for([whole])
    assert read(ctx, build.build_unit(whole, path, sha, SYSTEM, ctx))[0]["builtTree"] == git(path, "rev-parse", f"{sha}^{{tree}}")


MODULES = {"shell": ["extractors/common/errors.py", "extractors/common/writer_rules.py", "extractors/shell/extract.py"],
           "config": ["extractors/common/errors.py", "extractors/common/writer_rules.py", "extractors/config/extract.py",
                      "extractors/config/formats.py", "extractors/config/launch.py"],
           "markdown": ["ccg/globs.py", "extractors/common/writer_rules.py", "extractors/markdown/extract.py", "extractors/markdown/globs.py"]}


@pytest.mark.parametrize("kind", sorted(MODULES))
def test_the_extractor_hash_covers_every_module_of_the_extractor(monkeypatch, kind):
    assert sorted(build._EXTRACTOR_MODULES[kind]) == MODULES[kind]
    want = hashlib.sha256()
    for rel in MODULES[kind]:
        want.update(rel.encode() + b"\0" + hashlib.sha256((CPG / rel).read_bytes()).digest())
    assert build._extractor_hash(kind) == want.hexdigest()
    real = Path.read_bytes
    for rel in MODULES[kind]:                                               # a one-byte change in any one module changes the hash
        monkeypatch.setattr(Path, "read_bytes", lambda self, rel=rel: real(self) + (b"x" if str(self).endswith(rel) else b""))
        assert build._extractor_hash(kind) != want.hexdigest(), rel
        monkeypatch.undo()


# `from ccg import <name>` shared helpers that are NOT hashed (ids, the schema validator, the key store, the value hash): a known gap,
# listed so a *new* ccg module an extractor starts importing fails here instead of passing silently.
UNHASHED_CCG_HELPERS = {"ids", "bundle", "atrest", "valuehash", "units"}


def test_the_imported_modules_of_each_extractor_are_all_in_its_hash():
    """Every module an extractor module imports from ``extractors`` (absolute or relative) or from a ``ccg`` submodule
    (``from ccg.<mod> import``) is hashed; ``from ccg import <name>`` must name one of the listed unhashed helpers. A new import
    that is neither listed nor an exempt helper fails here. (The matcher moved to ``ccg/globs.py`` once without this noticing.)"""
    import ast
    for kind, mods in build._EXTRACTOR_MODULES.items():
        listed = {m[:-3] for m in mods}                                        # "extractors/markdown/globs", "ccg/globs", ...
        for rel in mods:
            for node in ast.walk(ast.parse((CPG / rel).read_text())):
                if not isinstance(node, ast.ImportFrom):
                    continue
                here = Path(rel).parent
                if node.level == 1:
                    targets = [str(here / node.module)] if node.module else [str(here / a.name) for a in node.names]
                elif node.module == "ccg":
                    assert {a.name for a in node.names} <= UNHASHED_CCG_HELPERS, (kind, rel, [a.name for a in node.names])
                    continue
                elif (node.module or "").startswith(("extractors.", "ccg.")):
                    parts = node.module.split(".")
                    targets = ["/".join(parts)] if len(parts) > 2 or parts[0] == "ccg" else ["/".join(parts) + "/" + a.name for a in node.names]
                else:
                    continue
                assert set(targets) <= listed | {f"ccg/{h}" for h in UNHASHED_CCG_HELPERS}, (kind, rel, targets)


def test_header_tool_versions_are_derived_not_constants(tmp_path, ctx_for):
    import importlib.metadata as md
    path, sha = make_repo(tmp_path / "t", {"s.sh": "echo\n", "a.json": "{}\n", "m.md": "x\n"})
    for lang, root in (("shell", "s.sh"), ("config", "a.json"), ("markdown", "m.md")):
        u = unit(lang, root)
        ctx = ctx_for([u])
        t = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))[0]["tools"]
        assert t["normalizerVersion"] == build._extractor_hash(lang)[:12] and t["extractScriptHash"] == build._extractor_hash(lang)
        assert t["frontendVersion"] == (md.version("tree-sitter-bash") if lang == "shell" else "none")


def test_vendored_dirs_are_exactly_the_directories_a_unit_listing_skips(tmp_path, ctx_for):
    from ccg import units as units_mod
    skip = sorted(units_mod.SKIP_DIRS)
    files = {f"p/{d}/x.md": "hidden\n" for d in skip} | {"p/keep/y.md": "shown\n"}
    path, sha = make_repo(tmp_path / "v", files)
    u = unit("markdown", "p")
    ctx = ctx_for([u])
    recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
    assert recs[0]["vendoredDirs"] == skip and [f["path"] for f in of(recs, "file")] == ["p/keep/y.md"]


def test_the_language_classifier_agrees_with_discovery_on_one_tree(tmp_path):
    files = {"a.sh": "echo\n", "b.bash": "echo\n", "run": "#!/usr/bin/env bash\necho\n", "py": "#!/usr/bin/python\n", "plain": "text\n",
             "m.md": "x\n", "M.MD": "x\n", "c.json": "{}\n", "c.yaml": "a: 1\n", "c.toml": "a = 1\n", "Dockerfile": "FROM x\n",
             "Dockerfile.dev": "FROM x\n", ".env": "A=1\n", ".env.local": "A=1\n", "x.txt": "t\n", ".mcp.json": "{}\n",
             "d/.hidden.sh": "echo\n", "d/e": "#!/bin/zsh\n", "empty": ""}
    path, sha = make_repo(tmp_path / "l", files)
    manifest = Manifest(system=SYSTEM, repos=(Repo(repo_id="r", path=".", policy=Policy(egress="local-only")),))
    found = {u.rootPath: u.language for u in discover_units(path, manifest)}
    ctx = build.BuildContext()
    tree = build._tree(ctx, path, sha)
    got = {p: lang for p, e in tree.items() if (lang := build._language_of(path, p, e))}
    expected = {"a.sh": "shell", "b.bash": "shell", "run": "shell", "d/e": "shell", "d/.hidden.sh": "shell", "m.md": "markdown",
                "c.json": "config", "c.yaml": "config", "c.toml": "config", "Dockerfile": "config", "Dockerfile.dev": "config",
                ".env": "config", ".env.local": "config", ".mcp.json": "config"}                  # by hand: M.MD is `.md` too
    expected["M.MD"] = "markdown"
    assert got == found == expected


def test_the_declared_service_cache_does_not_go_stale_across_commits(tmp_path, ctx_for):
    path, sha1 = make_repo(tmp_path / "c", {"compose.yaml": COMPOSE, "compose.override.yaml": WEB})
    (path / "compose.yaml").write_text("services:\n  other:\n    image: x\n")
    git(path, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-am", "drop api")
    sha2 = git(path, "rev-parse", "HEAD")
    u = unit("config", "compose.override.yaml")
    ctx = ctx_for([u, unit("config", "compose.yaml")])

    def bind(sha):
        recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
        return {r["keyPath"]: r.get("bindValue") for r in of(recs, "configEntry")}["services.web.environment.API_HOST"]

    assert (bind(sha1), bind(sha2)) == ("api", None)
    assert bind(sha1) == "api"                                              # and back: both commits stay cached side by side


@pytest.mark.parametrize("bad", ["HEAD", "main", "abc1234", "--help", "", "A" * 40, "1" * 39, "1" * 41, None, 7])
def test_a_commit_that_is_not_a_full_object_id_is_a_typed_error_before_any_work(monkeypatch, tmp_path, ctx_for, bad):
    path, sha = make_repo(tmp_path / "c", {"a.json": "{}\n"})
    calls = []
    monkeypatch.setattr(blame, "git", lambda *a, **k: calls.append(a))
    for lang in ("config", "python"):
        with pytest.raises(build.InvalidCommit):
            build.build_unit(unit(lang, "."), path, bad, SYSTEM, ctx_for())
    assert calls == []


def test_the_commit_prefix_of_a_real_commit_is_refused_not_resolved(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"a.json": "{}\n"})
    with pytest.raises(build.InvalidCommit):
        build.build_unit(unit("config", "a.json"), path, sha[:12], SYSTEM, ctx_for())


def test_the_report_keeps_only_integer_counts_a_planted_string_or_dict_never_reaches_the_result(monkeypatch, tmp_path, py_repo, ctx_for):
    def fake(ctx, u, repo, sha, system, report):
        report.update({"files": 3, "leak": f"x={SECRET}", "nested": {"k": SECRET}, "flag": True, "list": [SECRET], "none": None})
        raise ValueError(SECRET)

    monkeypatch.setattr(build, "_build_code", fake)
    path, sha = py_repo
    u = unit("python", ".")
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert dict(res.report) == {"files": 3} and SECRET not in repr(res)
    assert build._counts({"a": 1, "b": "s", "c": True, "d": 2.5}) == {"a": 1}


def test_the_declared_service_cache_is_keyed_by_the_set_of_compose_files_too(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"compose.yaml": COMPOSE, "compose.override.yaml": WEB})
    u = unit("config", "compose.override.yaml")
    ctx = ctx_for([u])                                                       # only the override is a unit: `api` is unknown

    def bind():
        recs = read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx))
        return {r["keyPath"]: r.get("bindValue") for r in of(recs, "configEntry")}["services.web.environment.API_HOST"]

    assert bind() is None
    ctx.units = (u, unit("config", "compose.yaml"))
    assert bind() == "api"


@pytest.mark.parametrize("lang, name, cap_attr, cap", [("markdown", "c/x.md", "MAX_MARKDOWN_BLOB_BYTES", 8),
                                                       ("shell", "c/x.sh", "MAX_SHELL_BLOB_BYTES", 8),
                                                       ("config", "c/x.json", "MAX_CONFIG_BLOB_BYTES", 8)])
def test_a_blob_exactly_at_its_cap_is_read_and_one_byte_over_is_not(monkeypatch, tmp_path, ctx_for, lang, name, cap_attr, cap):
    monkeypatch.setattr(build, cap_attr, cap)
    pad = {"markdown": "a", "shell": "\n", "config": " "}[lang]
    path, sha = make_repo(tmp_path / "c", {name: pad * cap, "c/y." + name.rsplit(".", 1)[1]: pad * (cap + 1)})
    reads = blob_reads(monkeypatch)
    u = unit(lang, "c")
    ctx = ctx_for([u])
    res = build.build_unit(u, path, sha, SYSTEM, ctx)
    if lang == "shell":                                                      # shell refuses a unit holding a file over the cap
        assert not res.ok
        one = unit("shell", name)
        assert build.build_unit(one, path, sha, SYSTEM, ctx_for([one])).ok
        return
    assert res.ok and [f["path"] for f in of(read(ctx, res), "file")] == [name] and len(reads) == 1


def test_the_build_context_hands_its_wall_cap_to_the_joern_run(monkeypatch, tmp_path, py_repo, ctx_for):
    seen = []

    def stub(repo, u, units, commit, work, **kw):
        seen.append(kw)
        raise joern_run.RunnerError("stop")

    monkeypatch.setattr(build.joern_run, "extract_unit", stub)
    path, sha = py_repo
    u = unit("python", ".")
    build.build_unit(u, path, sha, SYSTEM, ctx_for([u], wall_cap=7.0, timeout=11.0))
    assert (seen[0]["wall_cap"], seen[0]["timeout"]) == (7.0, 11.0)
    assert build.BuildContext().wall_cap == joern_run.INDEXER_WALL_CAP_S


def test_the_stage_write_is_guarded_by_the_destination_check_even_when_the_name_check_is_off(monkeypatch, tmp_path, ctx_for, tmp_base):
    monkeypatch.setattr(build, "_check_tree_path", lambda p: p)           # defence in depth: the second guard alone stops the write
    path, sha = hostile_repo(tmp_path / "h", {("a", "..", "..", "evil.sh"): b"echo PWNED\n"})
    u = unit("shell", ".")
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u]))
    assert (res.ok, res.fail_reason) == (False, "extract:failed") and files_under(tmp_base) == []


def test_a_compose_blob_exactly_at_the_config_cap_is_fetched_for_the_declared_set_and_one_over_is_not(monkeypatch, tmp_path, ctx_for):
    pad_to = lambda text, n: text + "#" * (n - len(text) - 1) + "\n"
    cap = 80
    monkeypatch.setattr(build, "MAX_CONFIG_BLOB_BYTES", cap)
    for size, fetched in ((cap, True), (cap + 1, False)):
        path, sha = make_repo(tmp_path / f"c{size}", {"compose.yaml": pad_to(COMPOSE, size), "compose.override.yaml": WEB})
        assert len(((path / "compose.yaml").read_bytes())) == size
        u = unit("config", "compose.override.yaml")
        ctx = ctx_for([u, unit("config", "compose.yaml")])
        entries = {r["keyPath"]: r.get("bindValue") for r in of(read(ctx, build.build_unit(u, path, sha, SYSTEM, ctx)), "configEntry")}
        assert entries["services.web.environment.API_HOST"] == ("api" if fetched else None)


def test_a_unit_whose_total_bytes_equal_the_cap_builds_and_one_byte_more_fails(monkeypatch, tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "c", {"d/a.json": "{}\n", "d/b.json": "{}\n"})                 # 6 bytes
    u = unit("config", "d")
    monkeypatch.setattr(build, "MAX_UNIT_BYTES", 6)
    assert build.build_unit(u, path, sha, SYSTEM, ctx_for([u])).ok
    monkeypatch.setattr(build, "MAX_UNIT_BYTES", 5)
    assert build.build_unit(u, path, sha, SYSTEM, ctx_for([u])).fail_reason == "extract:failed"


# --- the glob source: the manifest `ctx.manifest` names, never a built-in list ------------------------------------------------------


def test_a_manifest_without_the_prompt_bearing_block_fails_every_markdown_unit_and_no_other(tmp_path, ctx_for):
    manifest = tmp_path / "no-block.yaml"
    manifest.write_text("schemaVersion: 1\nsystem: s\negress: {schemaVersion: 1, local: [], outbound: [], hosted: []}\nrepos: []\n")
    path, sha = make_repo(tmp_path / "r", {"claude/a.md": "plain\n", "a.json": "{}\n", "s.sh": "echo\n"})
    units = [unit("markdown", "claude/a.md"), unit("config", "a.json"), unit("shell", "s.sh")]
    ctx = ctx_for(units, globs=None, manifest=manifest)
    rep = build.build_units(units, {"r": path}, {"r": sha}, SYSTEM, ctx)
    assert {r.unit.language: (r.ok, r.fail_reason) for r in rep.results} == {
        "markdown": (False, "extract:failed"), "config": (True, None), "shell": (True, None)}
    md = next(r for r in rep.results if r.unit.language == "markdown")
    assert md.path is None and "plain" not in repr(md) and dict(md.report) == {}
    assert not any(p.name.startswith(md.unit.unitId) for p in ctx.root.rglob("*.bundle.jsonl"))


def test_an_absent_manifest_fails_the_markdown_unit_the_same_way(tmp_path, ctx_for):
    path, sha = make_repo(tmp_path / "r", {"claude/a.md": "plain\n"})
    u = unit("markdown", "claude/a.md")
    res = build.build_unit(u, path, sha, SYSTEM, ctx_for([u], globs=None, manifest=tmp_path / "nope.yaml"))
    assert (res.ok, res.fail_reason, res.path) == (False, "extract:failed", None)


def test_the_fixture_systems_own_manifest_classifies_the_fx_svc_prompt_files_with_no_injected_globs(tmp_path, ctx_for):
    """``globs=None``: the block is read from ``fixtures/system1/system.yaml`` (the manifest that matches the fx-svc fixture repo)."""
    fixture_manifest = CPG / "fixtures" / "system1" / "system.yaml"
    units = [unit("markdown", ".claude/agents/agent.md", "fx-svc"), unit("markdown", "README.md", "fx-svc")]
    path2, sha2 = make_repo(tmp_path / "fx2", {".claude/agents/agent.md": "run `mcp__s__t`\n", "README.md": "run `mcp__s__t`\n"})
    ctx = ctx_for(units, globs=None, manifest=fixture_manifest)
    rep = build.build_units(units, {"fx-svc": path2}, {"fx-svc": sha2}, SYSTEM, ctx)
    assert all(r.ok for r in rep.results)
    mentioned = {r.unit.rootPath: bool(of(read(ctx, r), "callsite")) for r in rep.results}      # only a prompt-md file has call sites
    assert mentioned == {".claude/agents/agent.md": True, "README.md": False}


def test_the_real_system_manifest_is_the_default_glob_source(tmp_path, ctx_for):
    """No ``globs`` and no ``manifest`` argument: ``BuildContext`` reads ``ccg/system.yaml`` (production's source)."""
    assert build.BuildContext().manifest == CPG / "ccg" / "system.yaml"
    body = "run `mcp__s__t`\n"
    path, sha = make_repo(tmp_path / "r", {"claude/agents/a.md": body, "claude/README.md": body, "notes/b.md": body})
    units = [unit("markdown", p) for p in ("claude/agents/a.md", "claude/README.md", "notes/b.md")]
    ctx = ctx_for(units, globs=None)
    rep = build.build_units(units, {"r": path}, {"r": sha}, SYSTEM, ctx)
    mentioned = {r.unit.rootPath: bool(of(read(ctx, r), "callsite")) for r in rep.results}
    assert mentioned == {"claude/agents/a.md": True, "claude/README.md": False, "notes/b.md": False}
