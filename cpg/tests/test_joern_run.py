"""P1-04 — the Joern extractor's runner (`extractors/joern/run.py`).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-04 behaviours (2), (4), (5). Staging runs against real
git repositories built in `tmp_path` with the real `ccg.units.discover_units`, so the unit roots the exclusion rule
sees are the ones production sees. The container steps run against a stand-in runner (a script that records its argv
and writes the files the real runner's tools write); the real runner is exercised by the `live` tests at the end.

The expected stage listings are written out by hand, never derived from `descendant_roots`.
"""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ccg import atrest, ids
from ccg import units as un
from ccg.atrest import AtRestError
from ccg.manifest import Manifest, Policy, Repo
from ccg.units import Unit
from extractors.joern import run

CPG = Path(__file__).resolve().parent.parent
REPO = CPG.parent
FORBIDDEN = CPG / "sandbox" / "forbidden-flags.txt"


# --- fixtures ---------------------------------------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    env = {"PATH": os.environ["PATH"], "HOME": str(repo), "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                           *args], capture_output=True, check=True, text=True, env=env).stdout.strip()


def make_repo(root: Path, files: dict[str, str]) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "c")
    return root, git(root, "rev-parse", "HEAD")


def manifest_of(path: str = ".") -> Manifest:
    return Manifest(system="s", repos=(Repo(repo_id="r", path=path, policy=Policy(egress="local-only")),), egress={})


def units_of(repo: Path) -> dict[tuple[str, str], Unit]:
    return {(u.rootPath, u.language): u for u in un.discover_units(repo, manifest_of(), per_file=False)}


def stage_py(repo: Path, commit: str, units, root: str, dest: Path, lang: str = "python") -> set[str]:
    u = units[(root, lang)]
    run.stage_unit(repo, commit, u, run.descendant_roots(u, units.values()), dest)
    return {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}


#: every overlap the plan's behaviour (5) names, in one tree. Python units, by discovery: `.` (fallback, forced by
#: `x.py` beside nested manifests), `svc` and `svc/inner` (manifest over manifest), `fb` (fallback, forced by `fb/m`),
#: `fb/m` (manifest) and `fb/b` (a fallback under the fallback `fb`). `svc/ui` is a javascript unit inside a python one.
OVERLAPS = {
    "x.py": "x = 1\n",
    "svc/pyproject.toml": "[project]\n",
    "svc/s.py": "s = 1\n",
    "svc/inner/pyproject.toml": "[project]\n",
    "svc/inner/i.py": "i = 1\n",
    "svc/innerx/z.py": "z = 1\n",
    "svc/ui/package.json": "{}\n",
    "svc/ui/u.js": "export const u = 1;\n",
    "fb/a.py": "a = 1\n",
    "fb/m/pyproject.toml": "[project]\n",
    "fb/m/m.py": "m = 1\n",
    "fb/b/b.py": "b = 1\n",
    "web/package.json": "{}\n",
    "web/app.js": "export const app = 1;\n",
}
PY_STAGES = {
    ".": {"x.py", "web/package.json", "web/app.js"},        # the javascript units are no python descendant: they stay
    "svc": {"pyproject.toml", "s.py", "innerx/z.py", "ui/package.json", "ui/u.js"},
    "svc/inner": {"pyproject.toml", "i.py"},
    "fb": {"a.py"},
    "fb/m": {"pyproject.toml", "m.py"},
    "fb/b": {"b.py"},
}


@pytest.fixture()
def overlaps(tmp_path):
    repo, sha = make_repo(tmp_path / "repo", OVERLAPS)
    return repo, sha, units_of(repo)


# --- (5) staging excludes descendant same-language unit roots ----------------------------------------------------


def test_the_overlap_tree_discovers_the_units_the_tests_assume(overlaps):
    _, _, units = overlaps
    assert sorted(r for r, l in units if l == "python") == [".", "fb", "fb/b", "fb/m", "svc", "svc/inner"]
    assert sorted(r for r, l in units if l == "javascript") == ["svc/ui", "web"]


@pytest.mark.parametrize("root", sorted(PY_STAGES))
def test_each_python_unit_stage_is_exactly_its_own_files(overlaps, tmp_path, root):
    repo, sha, units = overlaps
    got = stage_py(repo, sha, units, root, tmp_path / "stage")
    assert got == PY_STAGES[root]


def test_manifest_over_manifest_drops_the_nested_unit(overlaps, tmp_path):
    repo, sha, units = overlaps
    got = stage_py(repo, sha, units, "svc", tmp_path / "s")
    assert "inner/i.py" not in got and "s.py" in got


def test_fallback_over_manifest_forced_drops_the_manifest_units(overlaps, tmp_path):
    repo, sha, units = overlaps
    got = stage_py(repo, sha, units, ".", tmp_path / "s")
    assert not any(p.startswith(("svc/", "fb/")) for p in got) and "x.py" in got


def test_fallback_over_fallback_drops_the_inner_fallback(overlaps, tmp_path):
    repo, sha, units = overlaps
    assert run.descendant_roots(units[("fb", "python")], units.values()) == ["fb/b", "fb/m"]
    got = stage_py(repo, sha, units, "fb", tmp_path / "s")
    assert "b/b.py" not in got and "a.py" in got


def test_every_python_file_is_staged_in_exactly_one_unit(overlaps, tmp_path):
    repo, sha, units = overlaps
    seen: dict[str, list[str]] = {}
    for root in [r for r, l in units if l == "python"]:
        for rel in stage_py(repo, sha, units, root, tmp_path / ("s" + root.replace("/", "_"))):
            if rel.endswith(".py"):
                seen.setdefault(rel if root == "." else f"{root}/{rel}", []).append(root)
    tracked = {f for f in git(repo, "ls-files").split("\n") if f.endswith(".py")}
    assert set(seen) == tracked
    assert all(len(v) == 1 for v in seen.values()), seen


def test_a_sibling_directory_sharing_a_prefix_is_not_excluded(overlaps, tmp_path):
    # `svc/inner` is a unit root, `svc/innerx` is not: the exclusion is on whole path segments.
    repo, sha, units = overlaps
    assert "innerx/z.py" in stage_py(repo, sha, units, "svc", tmp_path / "s")


def test_another_language_unit_below_is_not_an_exclusion(overlaps):
    _, _, units = overlaps
    assert run.descendant_roots(units[("svc", "python")], units.values()) == ["svc/inner"]
    assert run.descendant_roots(units[(".", "python")], units.values()) == ["fb", "fb/b", "fb/m", "svc", "svc/inner"]


def test_a_unit_of_another_repository_is_not_a_descendant():
    a = Unit("1", "ra", "pkg", "python")
    assert run.descendant_roots(a, [Unit("2", "rb", "pkg/inner", "python"), Unit("3", "ra", "pkg/inner", "python")]) == ["pkg/inner"]


def test_a_unit_is_not_its_own_descendant():
    a = Unit("1", "r", "pkg", "python")
    assert run.descendant_roots(a, [a]) == []


def test_javascript_stage_excludes_a_nested_javascript_unit_only(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {
        "app/package.json": "{}", "app/a.ts": "export const a = 1;\n", "app/pkg/package.json": "{}",
        "app/pkg/p.ts": "export const p = 1;\n", "app/py/pyproject.toml": "", "app/py/m.py": "m = 1\n"})
    units = units_of(repo)
    got = stage_py(repo, sha, units, "app", tmp_path / "s", "javascript")
    assert got == {"package.json", "a.ts", "py/pyproject.toml", "py/m.py"}


def test_the_stage_is_the_committed_tree_not_the_working_tree(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"svc/pyproject.toml": "", "svc/a.py": "a = 1\n"})
    (repo / "svc" / "a.py").write_text("a = 2  # edited after the commit\n")
    (repo / "svc" / "untracked.py").write_text("u = 1\n")
    units = units_of(repo)
    stage_py(repo, sha, units, "svc", tmp_path / "s")
    assert (tmp_path / "s" / "a.py").read_text() == "a = 1\n"
    assert not (tmp_path / "s" / "untracked.py").exists()


def test_the_stage_is_at_the_named_commit_not_head(tmp_path):
    repo, first = make_repo(tmp_path / "r", {"svc/pyproject.toml": "", "svc/a.py": "a = 1\n"})
    (repo / "svc" / "a.py").write_text("a = 2\n")
    (repo / "svc" / "new.py").write_text("n = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "second")
    units = units_of(repo)
    assert stage_py(repo, first, units, "svc", tmp_path / "old") == {"pyproject.toml", "a.py"}
    assert (tmp_path / "old" / "a.py").read_text() == "a = 1\n"
    assert "new.py" in stage_py(repo, "HEAD", units, "svc", tmp_path / "new")


def test_the_stage_carries_no_git_history(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"svc/pyproject.toml": "", "svc/a.py": "a = 1\n"})
    stage_py(repo, sha, units_of(repo), "svc", tmp_path / "s")
    assert not (tmp_path / "s" / ".git").exists()


def test_the_repo_root_unit_stages_the_whole_tree_with_no_strip(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a.py": "a = 1\n", "d/data.txt": "x\n", "e/b.py": "b = 1\n"})
    units = units_of(repo)
    assert sorted(r for r, l in units if l == "python") == [".", "e"]       # `e/b.py` is a fallback unit of its own
    assert stage_py(repo, sha, units, ".", tmp_path / "s") == {"a.py", "d/data.txt"}


def test_a_unit_whose_only_sources_are_in_a_descendant_unit_stages_nothing_and_fails(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a/pyproject.toml": "", "a/b/pyproject.toml": "", "a/b/y.py": "y = 1\n"})
    units = units_of(repo)
    with pytest.raises(run.StageError, match="no non-empty source file"):
        stage_py(repo, sha, units, "a", tmp_path / "s")


def test_empty_source_files_do_not_count_as_sources(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a/pyproject.toml": "", "a/__init__.py": ""})
    with pytest.raises(run.StageError, match="no non-empty source file"):
        stage_py(repo, sha, units_of(repo), "a", tmp_path / "s")


def test_a_unit_with_one_non_empty_source_among_empty_ones_stages(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a/pyproject.toml": "", "a/__init__.py": "", "a/m.py": "m = 1\n"})
    u = units_of(repo)[("a", "python")]
    assert run.stage_unit(repo, sha, u, [], tmp_path / "s") == 1


def test_an_exclusion_outside_the_unit_is_refused(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a/pyproject.toml": "", "a/m.py": "m = 1\n"})
    u = units_of(repo)[("a", "python")]
    for bad in ("b", "a", "ab/c", "."):
        with pytest.raises(run.StageError, match="not below the unit root"):
            run.stage_unit(repo, sha, u, [bad], tmp_path / "s")


def test_a_language_without_a_joern_extractor_is_refused(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a/run.sh": "echo\n"})
    with pytest.raises(run.StageError, match="no Joern extractor"):
        run.stage_unit(repo, sha, Unit("1", "r", "a", "shell"), [], tmp_path / "s")


@pytest.mark.parametrize("commit", ["deadbeef" * 5, "--output=/tmp/x", "no-such-ref"])
def test_a_commit_that_is_not_one_is_refused(tmp_path, commit):
    repo, _ = make_repo(tmp_path / "r", {"a/m.py": "m = 1\n"})
    with pytest.raises(run.StageError, match="not a commit"):
        run.stage_unit(repo, commit, Unit("1", "r", "a", "python"), [], tmp_path / "s")


#: The reviewer's two-attribute tree (`docs/reviews/connected-code-graph-p1-04.md` M1): `git archive` honours both
#: attributes, so it would drop `svc/tests/t.py` and rewrite `$Format:%H$` in `svc/v.py` with a commit hash.
ATTRIBUTES = {
    ".gitattributes": "svc/tests/t.py export-ignore\nsvc/v.py export-subst\n",
    "svc/pyproject.toml": "[project]\n",
    "svc/m.py": "m = 1\n",
    "svc/tests/t.py": "t = 1\n",
    "svc/v.py": "V = '$Format:%H$'\n",
}


def test_export_attributes_do_not_change_the_stage(tmp_path):
    repo, sha = make_repo(tmp_path / "r", ATTRIBUTES)
    units = units_of(repo)
    # the attributes really bite `git archive` in this git, so the assertions below cannot pass by accident
    arch = subprocess.run(["git", "-C", str(repo), "archive", sha, "--", "svc"], capture_output=True, check=True).stdout
    assert b"svc/tests/t.py" not in arch and b"$Format:%H$" not in arch
    got = stage_py(repo, sha, units, "svc", tmp_path / "s")
    assert got == {"pyproject.toml", "m.py", "tests/t.py", "v.py"}
    assert (tmp_path / "s" / "v.py").read_text() == "V = '$Format:%H$'\n"


def test_the_stage_is_the_trees_blobs_byte_for_byte(tmp_path):
    repo, sha = make_repo(tmp_path / "r", ATTRIBUTES | {"svc/bin.py": "x = b'\\xff'\n", "svc/run.sh": "#!/bin/sh\n"})
    os.chmod(repo / "svc" / "run.sh", 0o755)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "x")
    sha = git(repo, "rev-parse", "HEAD")
    stage_py(repo, sha, units_of(repo), "svc", tmp_path / "s")
    for line in git(repo, "ls-tree", "-r", sha, "--", "svc").split("\n"):
        meta, path = line.split("\t")
        _mode, _typ, oid = meta.split(" ")
        staged = tmp_path / "s" / path[len("svc/"):]
        assert git(tmp_path / "s", "hash-object", "--no-filters", str(staged)) == oid, path
    assert os.access(tmp_path / "s" / "run.sh", os.X_OK) and not os.access(tmp_path / "s" / "m.py", os.X_OK)


def test_a_root_with_glob_characters_names_that_directory_only(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a*/m.py": "m = 1\n", "ab/n.py": "n = 1\n", "a1/o.py": "o = 1\n"})
    dest = tmp_path / "s"
    assert run.stage_unit(repo, sha, Unit("1", "r", "a*", "python"), [], dest) == 1
    assert {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()} == {"m.py"}
    repo2, sha2 = make_repo(tmp_path / "r2", {"a[1]/m.py": "m = 1\n", "a1/n.py": "n = 1\n"})
    assert run.stage_unit(repo2, sha2, Unit("1", "r", "a[1]", "python"), [], tmp_path / "s2") == 1
    assert {p.name for p in (tmp_path / "s2").rglob("*.py")} == {"m.py"}


def test_a_sibling_directory_with_the_roots_name_as_prefix_is_not_staged(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a/m.py": "m = 1\n", "ab/n.py": "n = 1\n"})
    got = stage_py(repo, sha, {("a", "python"): Unit("1", "r", "a", "python")}, "a", tmp_path / "s")
    assert got == {"m.py"}


@pytest.mark.parametrize("files", [
    {"a/a.pyi": "x: int\n"},
    {"a/a.mts": "export const a = 1;\n", "a/b.cts": "export const b = 1;\n"},
    {"a/empty.py": ""},
], ids=["only-pyi", "only-mts-cts", "only-empty-py"])
def test_files_joern_ignores_or_that_are_empty_do_not_count_as_sources(tmp_path, files):
    lang = "javascript" if any(k.endswith((".mts", ".cts")) for k in files) else "python"
    repo, sha = make_repo(tmp_path / "r", files)
    with pytest.raises(run.StageError, match="no non-empty source file"):
        run.stage_unit(repo, sha, Unit("1", "r", "a", lang), [], tmp_path / "s")


def test_a_symlink_named_like_a_source_does_not_count_as_one(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a/real.txt": "data\n"})
    os.symlink("real.txt", repo / "a" / "link.py")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "link")
    with pytest.raises(run.StageError, match="no non-empty source file"):
        run.stage_unit(repo, git(repo, "rev-parse", "HEAD"), Unit("1", "r", "a", "python"), [], tmp_path / "s")
    assert (tmp_path / "s" / "link.py").is_symlink()          # it was staged (an in-unit link is allowed), just not counted


def test_a_submodule_entry_is_skipped_not_staged(tmp_path):
    repo, _ = make_repo(tmp_path / "r", {"a/pyproject.toml": "", "a/m.py": "m = 1\n"})
    git(repo, "update-index", "--add", "--cacheinfo", "160000," + "1" * 40 + ",a/sub")      # a gitlink, as a submodule is
    git(repo, "commit", "-q", "-m", "gitlink")
    sha = git(repo, "rev-parse", "HEAD")
    assert "commit a/sub" in git(repo, "ls-tree", "-r", sha).replace("\t", " ").replace(" " + "1" * 40, "")
    assert stage_py(repo, sha, units_of(repo), "a", tmp_path / "s") == {"pyproject.toml", "m.py"}


def test_a_symlink_leaving_the_stage_fails_the_stage(tmp_path):
    repo, sha = make_repo(tmp_path / "r", {"a/pyproject.toml": "", "a/m.py": "m = 1\n"})
    os.symlink("/etc", repo / "a" / "etc-link")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "link")
    with pytest.raises(run.StageError):
        run.stage_unit(repo, git(repo, "rev-parse", "HEAD"), units_of(repo)[("a", "python")], [], tmp_path / "s")


# --- (2) the argv of every stage ---------------------------------------------------------------------------------


def _forbidden() -> list[tuple[str, str, str]]:
    out = []
    for line in FORBIDDEN.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            fe, flag, where, _reason = line.split("\t", 3)
            out.append((fe, flag, where))
    return out


def forbidden_hits(argv: list[str], frontends: set[str], entries=None) -> list[str]:
    """Egress spec3 §7.5 (2): equals the flag, starts with `flag=`, or contains it as a substring, case-insensitively,
    for the frontend (or `*`) in offline mode.

    Bound: `forbidden-flags.txt` has no entry for `joern-parse`, `pysrc2cpg` or `jssrc2cpg`, so for the parse stage this
    check is empty by construction and cannot fail on the real list. The parse stage's real pin is
    `test_the_exact_argv_of_each_stage`; the Joern script stages are pinned by this check too (`joern --dep` is real).
    `entries` substitutes a list, to show the check fires for a frontend once an entry exists."""
    hits = []
    for el in argv:
        low = el.lower()
        for fe, flag, _where in (_forbidden() if entries is None else entries):
            if fe in frontends | {"*"} and flag.lower() in low:
                hits.append(el)
    return hits


R = Path("/r/offline-run.sh")
SRC, OUT = Path("/w/stage"), Path("/w/cpg")
JOERN_SCRIPT_GRAMMAR = re.compile(r"--script /scripts/(?P<name>[A-Za-z0-9._-]+\.sc)( --param [A-Za-z][A-Za-z0-9_]*=\S+)*")


def all_stage_argvs(lang: str) -> dict[str, list[str]]:
    d = {"parse": run.parse_argv(R, SRC, OUT, "r", lang),
         "check": run.content_check_argv(R, SRC, OUT, "r", "c.cpg.bin"),
         "extract": run.extract_argv(R, SRC, OUT, "r", "c.cpg.bin", lang)}
    if lang == "javascript":
        d["astgen"] = run.astgen_argv(R, SRC, OUT, "r")
    return d


def test_the_exact_argv_of_each_stage():
    assert run.parse_argv(R, SRC, OUT, "repo1", "python") == [
        "/r/offline-run.sh", "--backend", "container", "--mode", "offline", "--frontend", "joern-parse",
        "--source", "/w/stage", "--output", "/w/cpg", "--codebase", "repo1", "--", "/src", "--language", "pythonsrc"]
    assert run.parse_argv(R, SRC, OUT, "repo1", "javascript")[-3:] == ["/src", "--language", "jssrc"]
    ex = run.extract_argv(R, SRC, OUT, "repo1", "c.cpg.bin", "javascript")
    assert ex[ex.index("--frontend") + 1] == "joern"
    assert ex[ex.index("--script-file") + 1] == str(run.EXTRACT_SC)
    assert ex[ex.index("--") + 1:] == ["--script", "/scripts/extract.sc", "--param", "cpgFile=/out/c.cpg.bin",
                                      "--param", "out=/out/@AUDIT_ID@.facts.jsonl", "--param", "lang=javascript"]
    ag = run.astgen_argv(R, SRC, OUT, "repo1")
    assert ag[ag.index("--") + 1:] == ["--script", "/scripts/astgen.sc", "--param", "cpgFile=unused",
                                       "--param", "outDir=/out/@AUDIT_ID@.astgen"]
    ck = run.content_check_argv(R, SRC, OUT, "repo1", "c.cpg.bin")
    assert ck[ck.index("--script-file") + 1] == str(run.CONTENT_CHECK_SC)
    assert ck[ck.index("--") + 1:] == ["--script", "/scripts/cpg-content-check.sc", "--param", "cpgFiles=/out/c.cpg.bin"]


@pytest.mark.parametrize("lang", ["python", "javascript"])
def test_every_stage_is_offline_on_the_container_backend_and_carries_no_forbidden_flag(lang):
    for name, argv in all_stage_argvs(lang).items():
        assert argv[argv.index("--mode") + 1] == "offline", name
        assert argv[argv.index("--backend") + 1] == "container", name
        assert argv[0] == str(R) and "controlled" not in argv, name
        fe = argv[argv.index("--frontend") + 1]
        derived = {"joern-parse": {"pysrc2cpg", "jssrc2cpg"}}.get(fe, set())
        assert forbidden_hits(argv, {fe} | derived) == [], name
        assert argv.count("--output") == 1, name                    # the runner's own option, once
        tool = argv[argv.index("--") + 1:]
        assert "--output" not in tool and "-o" not in tool, name


@pytest.mark.parametrize("lang", ["python", "javascript"])
def test_every_joern_script_argv_matches_the_runners_script_grammar(lang):
    for name, argv in all_stage_argvs(lang).items():
        if argv[argv.index("--frontend") + 1] != "joern":
            continue
        tool = " ".join(argv[argv.index("--") + 1:])
        m = JOERN_SCRIPT_GRAMMAR.fullmatch(tool)
        assert m, (name, tool)
        assert m["name"] == Path(argv[argv.index("--script-file") + 1]).name, name


def test_the_forbidden_flag_check_fires_on_a_planted_flag():
    """Positive control: the `joern --dep` entry (and the parse frontends' own) catches a planted flag."""
    assert _forbidden(), "forbidden-flags.txt parsed to nothing"
    argv = run.extract_argv(R, SRC, OUT, "r", "c.cpg.bin", "python") + ["--dep"]
    assert forbidden_hits(argv, {"joern"}) == ["--dep"]
    parse = run.parse_argv(R, SRC, OUT, "r", "javascript")
    assert forbidden_hits(parse, {"joern-parse", "jssrc2cpg"}) == []
    # control for the parse stage: under a planted entry for the derived frontend the same check fires
    planted = [("jssrc2cpg", "--fetch-dependencies", "offline")]
    assert forbidden_hits(parse + ["--fetch-dependencies"], {"joern-parse", "jssrc2cpg"}, planted) == ["--fetch-dependencies"]
    assert forbidden_hits(parse, {"joern-parse", "jssrc2cpg"}, planted) == []


def test_every_scripts_name_is_a_file_that_exists_and_has_no_directive():
    for sc in (run.EXTRACT_SC, run.ASTGEN_SC, run.CONTENT_CHECK_SC):
        assert sc.is_file(), sc
        for line in sc.read_text().replace("\r", "\n").split("\n"):
            assert not line.lstrip().startswith("//>"), (sc, line)


def test_a_repository_that_is_a_subdirectory_of_its_git_work_tree_stages_its_own_unit(tmp_path):
    """The manifest's repo path may be a subdirectory of the work tree: discovery is relative to it, so staging is too.
    `other/svc` and a top-level `svc` must not be staged in place of `mono/svc`."""
    top, _ = make_repo(tmp_path / "top", {"mono/svc/pyproject.toml": "", "mono/svc/m.py": "m = 1\n",
                                          "other/svc/o.py": "o = 1\n", "svc/t.py": "t = 1\n"})
    mono = top / "mono"
    units = {(u.rootPath, u.language): u for u in un.discover_units(mono, manifest_of(), per_file=False)}
    assert ("svc", "python") in units
    got = stage_py(mono, git(top, "rev-parse", "HEAD"), units, "svc", tmp_path / "s")
    assert got == {"pyproject.toml", "m.py"}


# --- the stand-in runner and extract_unit -----------------------------------------------------------------------

STUB = textwrap.dedent('''\
    #!/usr/bin/env python3
    import json, os, pathlib, signal, subprocess, sys, time, uuid
    a = sys.argv[1:]
    i = a.index("--")
    opt = {a[j]: a[j + 1] for j in range(0, i, 2)}
    tool = a[i + 1:]
    mode = os.environ.get("STUB_MODE", "ok")
    stage = os.environ.get("STUB_FAIL_STAGE", "")
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write(json.dumps(a) + "\\n")
    aid = str(uuid.uuid4())
    out = pathlib.Path(opt["--output"])
    fe = opt["--frontend"]
    params = {}
    for k in range(len(tool) - 1):
        if tool[k] == "--param":
            n, _, v = tool[k + 1].partition("=")
            params[n] = v
    script = tool[1].rsplit("/", 1)[-1] if fe == "joern" else ""
    kind = {"joern-parse": "parse", "cpg-content-check.sc": "check", "astgen.sc": "astgen", "extract.sc": "extract"}[
        "joern-parse" if fe == "joern-parse" else script]
    if kind == stage:
        if mode == "hang":
            def term(*_):
                pathlib.Path(os.environ["STUB_MARK"]).write_text("terminated")
                sys.exit(143)
            signal.signal(signal.SIGTERM, term)
            time.sleep(60)
        sys.exit(3)
    if kind == "parse":
        (out / (aid + ".cpg.bin")).write_bytes(b"cpg")
        if os.environ.get("STUB_ADVANCE"):      # a branch moves while the runner works
            subprocess.run(["git", "-C", os.environ["STUB_ADVANCE"], "-c", "user.name=t", "-c", "user.email=t@t",
                            "commit", "--allow-empty", "-q", "-m", "moved"], check=True)
    elif kind == "check":
        name = params["cpgFiles"]
        if mode == "empty":
            print("METHODS " + name + "=\\nCONTENT " + name + "=EMPTY"); sys.exit(1)
        if mode == "check-crash":
            print("docker: cannot connect"); sys.exit(1)
        if mode == "check-silent":
            print("METHOD_COUNT " + name + "=4"); sys.exit(0)
        print("CONTENT " + name + "=ok")
    elif kind == "astgen":
        (out / (aid + ".astgen")).mkdir()
        (out / (aid + ".astgen") / "x.json").write_text("{}")
    elif kind == "extract":
        path = out / params["out"].replace("@AUDIT_ID@", aid).rsplit("/", 1)[-1]
        path.write_text('{"k":"m"}\\n{"k":"stats","methods":1}\\n')
''')


@pytest.fixture()
def stub(tmp_path, monkeypatch):
    p = tmp_path / "stub-runner"
    p.write_text(STUB)
    p.chmod(0o755)
    log = tmp_path / "stub.log"
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setenv("STUB_MARK", str(tmp_path / "mark"))
    monkeypatch.delenv("STUB_MODE", raising=False)
    monkeypatch.delenv("STUB_FAIL_STAGE", raising=False)
    monkeypatch.delenv("STUB_ADVANCE", raising=False)
    return p, log


def calls(log: Path) -> list[list[str]]:
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def kinds(log: Path) -> list[str]:
    out = []
    for a in calls(log):
        fe = a[a.index("--frontend") + 1]
        tool = a[a.index("--") + 1:]
        out.append("parse" if fe == "joern-parse" else tool[1].rsplit("/", 1)[-1])
    return out


@pytest.fixture()
def area(tmp_path):
    a = tmp_path / ".ccg-work"
    a.mkdir(mode=0o755)
    return a


def small_repo(tmp_path, lang="python"):
    if lang == "python":
        repo, sha = make_repo(tmp_path / "repo", {"svc/pyproject.toml": "", "svc/m.py": "m = 1\n"})
        return repo, sha, units_of(repo)[("svc", "python")], list(units_of(repo).values())
    repo, sha = make_repo(tmp_path / "repo", {"web/package.json": "{}", "web/a.ts": "export const a = 1;\n"})
    u = units_of(repo)
    return repo, sha, u[("web", "javascript")], list(u.values())


def test_a_python_unit_runs_stage_parse_check_extract_and_returns_two_audit_ids(tmp_path, stub, area):
    runner, log = stub
    repo, sha, unit, units = small_repo(tmp_path)
    rf = run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    assert kinds(log) == ["parse", "cpg-content-check.sc", "extract.sc"]
    assert len(rf.sandbox_audit_ids) == 2 and rf.astgen_dir is None
    assert rf.facts.name == rf.sandbox_audit_ids[1] + ".facts.jsonl"
    assert (rf.cpg_dir / (rf.sandbox_audit_ids[0] + ".cpg.bin")).is_file()
    assert rf.commit == sha and rf.source_files == 1
    assert (rf.stage / "m.py").is_file()


def test_a_javascript_unit_adds_astgen_and_its_id_sits_between_parse_and_extract(tmp_path, stub, area):
    runner, log = stub
    repo, sha, unit, units = small_repo(tmp_path, "javascript")
    rf = run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    assert kinds(log) == ["parse", "cpg-content-check.sc", "astgen.sc", "extract.sc"]
    parse_id, astgen_id, facts_id = rf.sandbox_audit_ids
    assert rf.astgen_dir.name == astgen_id + ".astgen" and rf.facts.name == facts_id + ".facts.jsonl"
    assert (rf.cpg_dir / (parse_id + ".cpg.bin")).is_file()


def test_the_content_check_is_not_an_input_of_the_bundle(tmp_path, stub, area):
    runner, log = stub
    repo, sha, unit, units = small_repo(tmp_path)
    rf = run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    assert len(set(rf.sandbox_audit_ids)) == 2          # parse and extract only, three runner calls were made
    assert len(calls(log)) == 3


def test_the_runner_calls_extract_unit_makes_obey_the_argv_rules(tmp_path, stub, area):
    runner, log = stub
    repo, sha, unit, units = small_repo(tmp_path, "javascript")
    run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    for a in calls(log):
        assert a[a.index("--mode") + 1] == "offline" and a[a.index("--backend") + 1] == "container"
        fe = a[a.index("--frontend") + 1]
        assert forbidden_hits(a, {fe, "jssrc2cpg"}) == []
        assert a[a.index("--codebase") + 1] == unit.repoId
    last = calls(log)[-1]
    assert last[last.index("--source") + 1] == str(area / "w" / "stage")


def test_an_empty_cpg_is_refused_and_extract_never_runs(tmp_path, stub, area, monkeypatch):
    runner, log = stub
    monkeypatch.setenv("STUB_MODE", "empty")
    repo, sha, unit, units = small_repo(tmp_path)
    with pytest.raises(run.EmptyCpgError, match="no non-external METHOD"):
        run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    assert kinds(log) == ["parse", "cpg-content-check.sc"]


def test_a_content_check_that_dies_without_its_verdict_is_a_runner_error_not_an_empty_cpg(tmp_path, stub, area, monkeypatch):
    runner, log = stub
    monkeypatch.setenv("STUB_MODE", "check-crash")
    repo, sha, unit, units = small_repo(tmp_path)
    with pytest.raises(run.RunnerError, match="without a CONTENT=EMPTY verdict") as e:
        run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    assert not isinstance(e.value, run.EmptyCpgError)
    assert kinds(log) == ["parse", "cpg-content-check.sc"]


def test_exit_zero_without_a_content_ok_line_does_not_certify_the_parse(tmp_path, stub, area, monkeypatch):
    runner, log = stub
    monkeypatch.setenv("STUB_MODE", "check-silent")
    repo, sha, unit, units = small_repo(tmp_path)
    with pytest.raises(run.RunnerError, match="no CONTENT=ok line"):
        run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    assert "extract.sc" not in kinds(log)


@pytest.mark.parametrize("stage", ["parse", "astgen", "extract.sc"])
def test_a_failing_stage_stops_the_run_with_a_runner_error(tmp_path, stub, area, monkeypatch, stage):
    runner, log = stub
    monkeypatch.setenv("STUB_FAIL_STAGE", {"parse": "parse", "astgen": "astgen", "extract.sc": "extract"}[stage])
    repo, sha, unit, units = small_repo(tmp_path, "javascript")
    with pytest.raises(run.RunnerError, match="exited 3"):
        run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    order = ["parse", "cpg-content-check.sc", "astgen.sc", "extract.sc"]
    assert kinds(log)[-1] == {"parse": "parse", "astgen": "astgen.sc", "extract.sc": "extract.sc"}[stage]
    assert len(kinds(log)) <= len(order)


def test_a_hung_stage_is_terminated_with_sigterm_so_the_runner_can_clean_up(tmp_path, stub, area, monkeypatch):
    runner, log = stub
    monkeypatch.setenv("STUB_MODE", "hang")
    monkeypatch.setenv("STUB_FAIL_STAGE", "parse")
    repo, sha, unit, units = small_repo(tmp_path)
    with pytest.raises(run.RunnerError, match="timed out"):
        run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner, timeout=1.5)
    assert (tmp_path / "mark").read_text() == "terminated"


def test_an_unstageable_unit_never_reaches_the_runner(tmp_path, stub, area):
    runner, log = stub
    repo, sha = make_repo(tmp_path / "r2", {"a/pyproject.toml": "", "a/__init__.py": ""})
    u = units_of(repo)[("a", "python")]
    with pytest.raises(run.StageError):
        run.extract_unit(repo, u, [u], sha, area / "w", area=area, runner=runner)
    assert calls(log) == []


def test_a_rerun_starts_from_an_empty_stage_and_cpg_dir(tmp_path, stub, area):
    runner, log = stub
    repo, sha, unit, units = small_repo(tmp_path)
    first = run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    (first.stage / "leftover.py").write_text("x = 1\n")
    second = run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=runner)
    assert not (second.stage / "leftover.py").exists()
    assert not first.facts.exists() and second.facts.exists()


def test_stage_overlap_exclusion_is_applied_by_extract_unit(tmp_path, stub, area):
    runner, log = stub
    repo, sha = make_repo(tmp_path / "r3", {"a/pyproject.toml": "", "a/m.py": "m = 1\n", "a/b/pyproject.toml": "", "a/b/y.py": "y = 1\n"})
    units = units_of(repo)
    rf = run.extract_unit(repo, units[("a", "python")], list(units.values()), sha, area / "w", area=area, runner=runner)
    assert (rf.stage / "m.py").is_file() and not (rf.stage / "b").exists()


def test_the_commit_is_resolved_once_so_a_moving_ref_cannot_split_stage_and_label(tmp_path, stub, area, monkeypatch):
    runner, log = stub
    repo, sha, unit, units = small_repo(tmp_path)
    real_stage, staged = run.stage_unit, []

    def stage_then_move(repo_, commit, *a, **kw):
        staged.append(commit)
        n = real_stage(repo_, commit, *a, **kw)
        git(repo_, "commit", "--allow-empty", "-q", "-m", "moved")    # the ref moves after staging, before labelling
        return n

    monkeypatch.setattr(run, "stage_unit", stage_then_move)
    rf = run.extract_unit(repo, unit, units, "HEAD", area / "w", area=area, runner=runner)
    assert git(repo, "rev-parse", "HEAD") != sha                      # it did move
    assert staged == [sha]                                            # staged by the resolved sha, not the ref
    assert rf.commit == sha                                           # and labelled with the one that was staged


def test_a_missing_runner_is_the_modules_own_error(tmp_path, area):
    repo, sha, unit, units = small_repo(tmp_path)
    with pytest.raises(run.RunnerError, match="no-such-runner"):
        run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=tmp_path / "no-such-runner")


STUBBORN = textwrap.dedent('''\
    #!/usr/bin/env python3
    import os, pathlib, signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
    time.sleep(120)
''')


def test_a_runner_that_ignores_sigterm_is_sigkilled_after_the_grace(tmp_path):
    """The kill guarantees no runner process outlives the call. It does NOT remove the runner's container: a killed
    runner never reaches its cleanup and `run.py` starts no container itself, so that leak is documented in
    `run._run`, not tested away. (The SIGTERM path, where the runner's trap does remove it, is the live-verified one.)"""
    script, pidfile = tmp_path / "stubborn", tmp_path / "pid"
    script.write_text(STUBBORN)
    script.chmod(0o755)
    with pytest.raises(run.RunnerError, match="timed out"):
        run._run([str(script), str(pidfile)], timeout=1.5, kill_grace=1.0)
    pid = int(pidfile.read_text())
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    if alive:
        os.kill(pid, 9)
    assert not alive


# --- the work directory is private -------------------------------------------------------------------------------


def test_work_directories_are_created_0700_under_umask_0(tmp_path, stub, area):
    runner, _ = stub
    repo, sha, unit, units = small_repo(tmp_path)
    old = os.umask(0)
    try:
        rf = run.extract_unit(repo, unit, units, sha, area / "p" / "w", area=area, runner=runner)
    finally:
        os.umask(old)
    for d in (area / "p", area / "p" / "w", rf.cpg_dir):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, d


def test_a_work_directory_outside_the_area_is_refused_before_anything_runs(tmp_path, stub, area):
    runner, log = stub
    repo, sha, unit, units = small_repo(tmp_path)
    with pytest.raises(AtRestError):
        run.extract_unit(repo, unit, units, sha, tmp_path / "elsewhere", area=area, runner=runner)
    assert calls(log) == []


def test_a_symlinked_work_directory_is_refused(tmp_path, stub, area):
    runner, log = stub
    repo, sha, unit, units = small_repo(tmp_path)
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", area / "link")
    with pytest.raises(AtRestError):
        run.extract_unit(repo, unit, units, sha, area / "link" / "w", area=area, runner=runner)
    assert calls(log) == []


# --- the command line --------------------------------------------------------------------------------------------


def test_the_command_line_refuses_a_root_that_is_no_unit(capsys):
    assert run.main(["--root", "no/such/unit", "--language", "python", "--commit", "HEAD", "--work", "/nowhere"]) == 2
    assert "0 units match" in capsys.readouterr().err


MANIFEST = "schemaVersion: 1\nsystem: s\negress: {schemaVersion: 1, local: [], outbound: [], hosted: []}\n" \
           "repos:\n  - repoId: r\n    path: .\n    policy: {egress: local-only}\n"


def test_the_help_and_docstring_say_units_come_from_the_index_not_the_commit(capsys):
    with pytest.raises(SystemExit) as e:
        run.main(["--help"])
    assert e.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "from the git index, not from --commit" in out and "P1-20" in out
    assert "git index, not from ``--commit``" in " ".join(run.main.__doc__.split()) and "P1-20" in run.main.__doc__


def test_the_command_line_takes_its_units_from_the_index_as_the_caveat_says(tmp_path, monkeypatch):
    """Pins the caveat: a unit that exists only in the index is in the `units` the CLI hands `extract_unit`.
    When P1-20 gives `discover_units(commit=...)` this test flips (the unit must be absent) and the caveat goes."""
    repo, sha = make_repo(tmp_path / "r", {"a/pyproject.toml": "", "a/m.py": "m = 1\n"})
    (repo / "a" / "b").mkdir()
    (repo / "a" / "b" / "pyproject.toml").write_text("")
    (repo / "a" / "b" / "y.py").write_text("y = 1\n")
    git(repo, "add", "-A")                                            # staged, never committed
    (tmp_path / "m.yaml").write_text(MANIFEST)
    seen = {}

    def fake(repo_, unit, units, commit, work, **kw):
        seen["roots"] = sorted(u.rootPath for u in units)
        seen["commit"] = commit
        return run.RawFacts(unit, commit, Path("/f"), ("i",), Path("/s"), Path("/c"), None, Path("/k"), 1)

    monkeypatch.setattr(run, "extract_unit", fake)
    rc = run.main(["--repo-root", str(repo), "--manifest", str(tmp_path / "m.yaml"), "--root", "a", "--language", "python",
                   "--commit", sha, "--work", str(tmp_path / ".ccg-work" / "w"), "--area", str(tmp_path / ".ccg-work")])
    assert rc == 0 and seen["commit"] == sha
    assert git(repo, "ls-tree", "-r", "--name-only", sha, "--", "a/b") == ""      # not in the commit ...
    assert seen["roots"] == ["a", "a/b"]                                          # ... yet in the units


def test_the_command_line_reports_a_missing_runner_cleanly(tmp_path, monkeypatch, capsys):
    repo, sha = make_repo(tmp_path / "r", {"a/pyproject.toml": "", "a/m.py": "m = 1\n"})
    (tmp_path / "m.yaml").write_text(MANIFEST)
    (tmp_path / ".ccg-work").mkdir()
    real = run.extract_unit
    monkeypatch.setattr(run, "extract_unit", lambda *a, **kw: real(*a, **kw, runner=tmp_path / "no-such-runner"))
    rc = run.main(["--repo-root", str(repo), "--manifest", str(tmp_path / "m.yaml"), "--root", "a", "--language", "python",
                   "--commit", sha, "--work", str(tmp_path / ".ccg-work" / "w"), "--area", str(tmp_path / ".ccg-work")])
    err = capsys.readouterr().err
    assert rc == 1 and "extract: FAIL:" in err and "no-such-runner" in err


# --- live: the real runner --------------------------------------------------------------------------------------

SHIM = textwrap.dedent('''\
    #!/usr/bin/env python3
    # Pass-through to the real runner, except that the parse stage runs jimple2cpg (a bytecode frontend) on the
    # python source: an exit-0 parse with no methods (egress spec3 §9 H13), the R11 case, produced by a real tool.
    import os, sys
    real = sys.argv[1]
    a = sys.argv[2:]
    if a[a.index("--frontend") + 1] == "joern-parse":
        a[a.index("--frontend") + 1] = "jimple2cpg"
        i = a.index("--")
        a = a[: i + 1] + ["/src"]
    os.execv(real, [real] + a)
''')


@pytest.mark.live
@pytest.mark.parametrize("lang", ["python", "javascript"])
def test_the_real_runner_extracts_a_small_unit(tmp_path, monkeypatch, lang):
    monkeypatch.setenv("CCG_AUDIT_DIR", str(tmp_path / "audit"))
    area = tmp_path / ".ccg-work"
    area.mkdir()
    repo, sha, unit, units = small_repo(tmp_path, lang)
    rf = run.extract_unit(repo, unit, units, sha, area / "w", area=area)
    lines = rf.facts.read_text().splitlines()
    assert json.loads(lines[-1])["k"] == "stats" and json.loads(lines[-1])["methods"] >= 1
    recs = [json.loads(l) for l in (tmp_path / "audit" / "sandbox.jsonl").read_text().splitlines()]
    by_id = {r["auditId"]: r for r in recs}
    for aid in rf.sandbox_audit_ids:
        assert by_id[aid]["exitCode"] == 0 and by_id[aid]["refused"] is False
    assert all(r["egressMode"] == "offline" for r in recs)


@pytest.mark.live
def test_a_real_exit_zero_empty_cpg_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("CCG_AUDIT_DIR", str(tmp_path / "audit"))
    shim = tmp_path / "shim"
    shim.write_text(SHIM)
    shim.chmod(0o755)
    # the shim takes the real runner as its first argument: wrap it in a one-line script the module can call alone
    wrapper = tmp_path / "wrapper"
    wrapper.write_text(f'#!/bin/sh\nexec {shim} {run.RUNNER} "$@"\n')
    wrapper.chmod(0o755)
    area = tmp_path / ".ccg-work"
    area.mkdir()
    repo, sha, unit, units = small_repo(tmp_path)
    with pytest.raises(run.EmptyCpgError):
        run.extract_unit(repo, unit, units, sha, area / "w", area=area, runner=wrapper)
