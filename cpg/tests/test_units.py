"""P1-03 — unit discovery over a manifest's repositories.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-03 behaviours (2)-(8). Trees are built in `tmp_path`;
the real monorepo is read through `cpg/ccg/system.yaml` and compared to a committed golden list.

`discover_units(repo_root, manifest, per_file=True)` returns code units (a build manifest or the directory-per-language
fallback) and, unless `per_file=False`, one unit per shell, config or Markdown file. The golden pins the code units: a
per-file golden would fail on every new document in the monorepo.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ccg import ids, manifest as mf
from ccg import units as un
from ccg.manifest import Manifest, Policy, Repo
from ccg.units import Unit

CPG = Path(__file__).resolve().parent.parent
REPO = CPG.parent
GOLDEN = CPG / "tests" / "golden" / "system-units.json"
EVAL_MANIFEST = CPG / "eval" / "eval-manifest.yaml"


def tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def manifest_of(*repos: tuple, excl_paths=(), excl_files=()) -> Manifest:
    rs = tuple(
        Repo(repo_id=rid, path=path, policy=Policy(egress="local-only"),
             exclusion_paths=tuple(excl_paths), content_check_files=tuple(excl_files))
        for rid, path in repos
    )
    return Manifest(system="s", repos=rs, egress={})


def code(root: Path, files: dict[str, str], *, repo=("r", "."), **kw) -> list[tuple[str, str]]:
    tree(root, files)
    got = un.discover_units(root, manifest_of(repo, **kw), per_file=False)
    return [(u.rootPath, u.language) for u in got]


# --- (2)-(5) manifests and the fallback -------------------------------------------------------------------------


def test_one_pyproject_gives_one_python_unit_with_the_right_root(tmp_path):
    got = un.discover_units(tree(tmp_path, {"svc/pyproject.toml": "", "svc/a.py": "x=1"}), manifest_of(("r", ".")),
                            per_file=False)
    assert got == [Unit(ids.unit_id("r", "svc", "python"), "r", "svc", "python")]


def test_unit_id_is_the_ids_module_hash_pinned_by_a_literal(tmp_path):
    (u,) = un.discover_units(tree(tmp_path, {"svc/pyproject.toml": ""}), manifest_of(("r", ".")), per_file=False)
    assert u.unitId == ids.unit_id("r", "svc", "python")
    assert ids.unit_id("r", "svc", "python") == ids.tuple_id(["unit", "r", "svc", "python"])
    assert ids.tuple_id(["unit", "r", "svc", "python"]) == "8347765fbb2a68dff2b4815fb37399e4"  # frozen once, here


def test_requirements_txt_is_a_python_manifest(tmp_path):
    assert code(tmp_path, {"job/requirements.txt": "", "job/run.py": ""}) == [("job", "python")]


@pytest.mark.parametrize("manifest_file", ["package.json", "tsconfig.json"])
def test_a_package_json_or_tsconfig_gives_a_javascript_unit(tmp_path, manifest_file):
    # No source file beside it: the unit exists because of the manifest, not through the fallback.
    assert code(tmp_path, {f"salesperson/{manifest_file}": "{}"}) == [("salesperson", "javascript")]


def test_package_json_and_tsconfig_in_one_directory_are_one_unit(tmp_path):
    assert code(tmp_path, {"app/package.json": "{}", "app/tsconfig.json": "{}"}) == [("app", "javascript")]


def test_a_manifest_at_the_repo_root_gives_root_dot(tmp_path):
    assert code(tmp_path, {"pyproject.toml": "", "x.py": ""}) == [(".", "python")]


def test_a_directory_with_python_files_and_no_manifest_falls_back_to_a_directory_unit(tmp_path):
    assert code(tmp_path, {"tools/a.py": "", "tools/sub/b.py": ""}) == [("tools", "python")]


def test_fallback_root_is_the_top_level_directory_even_for_deep_files(tmp_path):
    assert code(tmp_path, {"src/pkg/deep/a.py": ""}) == [("src", "python")]


def test_fallback_root_is_dot_for_a_file_at_the_repo_root(tmp_path):
    assert code(tmp_path, {"a.py": ""}) == [(".", "python")]


def test_fallback_treats_ts_and_js_as_one_javascript_unit_and_pyi_as_python(tmp_path):
    assert code(tmp_path, {"web/a.ts": "", "web/b.mjs": "", "stubs/c.pyi": ""}) == [
        ("stubs", "python"), ("web", "javascript"),
    ]


@pytest.mark.parametrize("ext", [".py", ".pyi", ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"])
def test_every_source_extension_of_python_and_javascript_reaches_the_fallback(tmp_path, ext):
    lang = "python" if ext.startswith(".py") else "javascript"
    assert code(tmp_path, {f"d/a{ext}": ""}) == [("d", lang)]


def test_a_manifest_directory_does_not_cover_a_sibling_that_shares_its_name_prefix(tmp_path):
    assert code(tmp_path, {"app/pyproject.toml": "", "app2/a.py": ""}) == [("app", "python"), ("app2", "python")]


def test_a_file_under_a_manifest_unit_of_its_language_does_not_create_a_fallback_unit(tmp_path):
    assert code(tmp_path, {"pyproject.toml": "", "pkg/a.py": "", "pkg/deep/b.py": ""}) == [(".", "python")]


def test_a_file_of_another_language_under_a_manifest_unit_still_falls_back(tmp_path):
    assert code(tmp_path, {"pkg/pyproject.toml": "", "pkg/a.py": "", "pkg/web.js": ""}) == [
        ("pkg", "javascript"), ("pkg", "python"),
    ]


def test_languages_outside_p1_have_no_code_unit(tmp_path):
    assert code(tmp_path, {"lib/A.java": "", "lib/pom.xml": "", "x/main.go": "", "y/a.c": ""}) == []


def test_nested_manifests_give_two_units_not_one(tmp_path):
    assert code(tmp_path, {"pyproject.toml": "", "common/pyproject.toml": "", "common/a.py": "", "b.py": ""}) == [
        (".", "python"), ("common", "python"),
    ]


def test_the_same_directory_with_two_ecosystems_gives_one_unit_per_language(tmp_path):
    assert code(tmp_path, {"pyproject.toml": "", "package.json": "{}"}) == [(".", "javascript"), (".", "python")]


def test_vendored_and_cache_directories_never_yield_units(tmp_path):
    files = {
        "web/node_modules/dep/package.json": "{}", "web/vendor/x.py": "", "a/third_party/y.py": "",
        "a/_vendor/z.py": "", "b/.venv/lib/pyproject.toml": "", "b/__pycache__/c.py": "", "t/vendored/q.js": "",
        "p/x.egg-info/a.py": "", "p/third-party/b.py": "",
    }
    assert code(tmp_path, files) == []


def test_two_repos_get_their_own_units_with_their_own_repo_ids(tmp_path):
    tree(tmp_path, {"one/pyproject.toml": "", "two/pyproject.toml": "", "two/sub/pyproject.toml": ""})
    got = un.discover_units(tmp_path, manifest_of(("repo-one", "one"), ("repo-two", "two")), per_file=False)
    assert [(u.repoId, u.rootPath, u.language) for u in got] == [
        ("repo-one", ".", "python"), ("repo-two", ".", "python"), ("repo-two", "sub", "python"),
    ]
    assert got[0].unitId == ids.unit_id("repo-one", ".", "python")


# --- (6) exclusions -----------------------------------------------------------------------------------------------


def test_a_directory_exclusion_removes_units_below_it(tmp_path):
    files = {"cpg/pyproject.toml": "", "cpg/a.py": "", "app/pyproject.toml": ""}
    assert code(tmp_path, files, excl_paths=["cpg/"]) == [("app", "python")]


def test_a_directory_exclusion_does_not_match_a_sibling_that_shares_the_prefix(tmp_path):
    assert code(tmp_path, {"cpg2/pyproject.toml": "", "cpg/pyproject.toml": ""}, excl_paths=["cpg/"]) == [
        ("cpg2", "python")
    ]


def test_a_directory_exclusion_is_rooted_at_the_repo_root_not_a_substring_anywhere(tmp_path):
    assert code(tmp_path, {"app/cpg/pyproject.toml": ""}, excl_paths=["cpg/"]) == [("app/cpg", "python")]


def test_an_excluded_file_creates_no_fallback_unit(tmp_path):
    assert code(tmp_path, {"deprecated/old.py": "", "keep/new.py": ""}, excl_paths=["deprecated/"]) == [
        ("keep", "python")
    ]


@pytest.mark.parametrize(
    "pattern, excluded, kept",
    [
        ("docs/**/connected-code-graph*", "docs/plans/connected-code-graph-x.md", "docs/plans/other.md"),
        ("docs/**/connected-code-graph*", "docs/a/b/connected-code-graph.md", "docs/connected-x.md"),
        ("docs/**/connected-code-graph*", "docs/connected-code-graph-p1.md", "docs2/connected-code-graph.md"),
        ("docs/BACKLOG.md", "docs/BACKLOG.md", "docs/sub/BACKLOG.md"),
        ("*.md", "a.md", "dir/a.md"),
        ("a?.md", "ab.md", "abc.md"),
        ("*.md", "a.md", "a.md.sh"),  # the glob must cover the whole path, not a prefix of it
    ],
)
def test_exclusion_globs(tmp_path, pattern, excluded, kept):
    tree(tmp_path, {excluded: "", kept: ""})
    got = {u.rootPath for u in un.discover_units(tmp_path, manifest_of(("r", "."), excl_paths=[pattern]))}
    assert kept in got and excluded not in got


def test_content_check_files_are_excluded_exactly(tmp_path):
    tree(tmp_path, {"a/x.md": "", "a/y.md": ""})
    got = {u.rootPath for u in un.discover_units(tmp_path, manifest_of(("r", "."), excl_files=["a/x.md"]))}
    assert got == {"a/y.md"}


def test_exclusions_are_relative_to_the_repo_path_not_the_root_given(tmp_path):
    tree(tmp_path, {"one/cpg/a.py": "", "one/keep/a.py": ""})
    got = un.discover_units(tmp_path, manifest_of(("r", "one"), excl_paths=["cpg/"]), per_file=False)
    assert [u.rootPath for u in got] == ["keep"]


# --- per-file units -------------------------------------------------------------------------------------------------


def test_shell_config_and_markdown_files_are_per_file_units(tmp_path):
    tree(tmp_path, {
        "scripts/run.sh": "", "scripts/lib.bash": "", "bin/tool": "#!/usr/bin/env bash\necho\n",
        "bin/zsh-tool": "#!/bin/zsh\n", "bin/sp": "#! /bin/sh -e\n", "bin/fish": "#!/usr/bin/fish\n",
        "bin/env-s": "#!/usr/bin/env -S bash -e\n", "bin/py-tool": "#!/usr/bin/env python3\n", "bin/plain": "text\n",
        "README.md": "", "docs/a.md": "",
        "config/a.yaml": "", "config/b.yml": "", "c.json": "{}", "d.toml": "", ".env": "", ".env.local": "",
        "Dockerfile": "", "deploy/Dockerfile.dev": "", ".mcp.json": "{}", "compose.yaml": "",
        "data/x.csv": "", "img/a.jpg": "",
    })
    got = {(u.rootPath, u.language) for u in un.discover_units(tmp_path, manifest_of(("r", ".")))}
    assert got == {
        ("scripts/run.sh", "shell"), ("scripts/lib.bash", "shell"), ("bin/tool", "shell"), ("bin/zsh-tool", "shell"),
        ("bin/sp", "shell"), ("bin/env-s", "shell"), ("README.md", "markdown"), ("docs/a.md", "markdown"),
        ("config/a.yaml", "config"), ("config/b.yml", "config"), ("c.json", "config"), ("d.toml", "config"),
        (".env", "config"), (".env.local", "config"), ("Dockerfile", "config"), ("deploy/Dockerfile.dev", "config"),
        (".mcp.json", "config"), ("compose.yaml", "config"),
    }


def test_per_file_units_can_be_switched_off(tmp_path):
    tree(tmp_path, {"a.sh": "", "pyproject.toml": ""})
    both = un.discover_units(tmp_path, manifest_of(("r", ".")))
    code_only = un.discover_units(tmp_path, manifest_of(("r", ".")), per_file=False)
    assert [(u.rootPath, u.language) for u in both] == [(".", "python"), ("a.sh", "shell"), ("pyproject.toml", "config")]
    assert [(u.rootPath, u.language) for u in code_only] == [(".", "python")]


def test_a_per_file_unit_id_is_the_ids_hash_of_its_file_path(tmp_path):
    tree(tmp_path, {"x/a.sh": ""})
    (u,) = un.discover_units(tmp_path, manifest_of(("r", ".")))
    assert u.unitId == ids.unit_id("r", "x/a.sh", "shell")


# --- (7) stability ---------------------------------------------------------------------------------------------


def test_units_are_sorted_and_identical_across_runs_and_creation_order(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    names = ["z/pyproject.toml", "a/pyproject.toml", "m/package.json", "x.sh", "doc.md", "k/q.py"]
    tree(a, {n: "" for n in names})
    tree(b, {n: "" for n in reversed(names)})
    ua = un.discover_units(a, manifest_of(("r", ".")))
    assert ua == un.discover_units(a, manifest_of(("r", ".")))
    assert ua == un.discover_units(b, manifest_of(("r", ".")))
    assert ua == sorted(ua, key=lambda u: (u.repoId, u.rootPath, u.language))


# --- git-listed trees ---------------------------------------------------------------------------------------------


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_in_a_git_tree_only_tracked_files_count_so_scratch_files_never_change_the_units(tmp_path):
    tree(tmp_path, {"app/pyproject.toml": "", "app/a.py": ""})
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    tree(tmp_path, {"scratch/b.py": "", "note.sh": ""})  # untracked
    got = un.discover_units(tmp_path, manifest_of(("r", ".")))
    assert [(u.rootPath, u.language) for u in got] == [
        ("app", "python"), ("app/pyproject.toml", "config"),
    ]  # no `scratch` python unit and no `note.sh` shell unit


def test_in_a_git_tree_vendored_directories_are_skipped_too(tmp_path):
    tree(tmp_path, {"web/node_modules/dep/package.json": "{}", "a/vendor/x.py": "", "ok/pyproject.toml": "",
                     "p/x.egg-info/a.py": ""})
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    got = un.discover_units(tmp_path, manifest_of(("r", ".")), per_file=False)
    assert [(u.rootPath, u.language) for u in got] == [("ok", "python")]


def test_an_empty_git_tree_falls_back_to_walking_the_directory(tmp_path):
    _git(tmp_path, "init", "-q")
    tree(tmp_path, {"app/a.py": ""})  # nothing added: the index is empty
    got = un.discover_units(tmp_path, manifest_of(("r", ".")), per_file=False)
    assert [(u.rootPath, u.language) for u in got] == [("app", "python")]


# --- (8) the real repository and the fixture system --------------------------------------------------------------


def _real() -> list[Unit]:
    return un.discover_units(REPO, mf.load_manifest(CPG / "ccg" / "system.yaml"), per_file=False)


def test_the_real_repo_yields_the_units_the_slice_used_among_its_list():
    got = {(u.rootPath, u.language) for u in _real()}
    assert ("falkor-chat/server", "python") in got
    assert ("salesperson", "javascript") in got


def test_the_real_repo_contains_every_unit_of_the_eval_manifests_census_list():
    want = {(u["rootPath"], u["lang"]) for u in yaml.safe_load(EVAL_MANIFEST.read_text())["units"]}
    assert len(want) == 6
    assert want <= {(u.rootPath, u.language) for u in _real()}


def test_no_unit_is_discovered_under_cpg_or_deprecated_for_the_proving_system():
    roots = {u.rootPath for u in un.discover_units(REPO, mf.load_manifest(CPG / "ccg" / "system.yaml"))}
    assert roots and not any(r == "cpg" or r.startswith(("cpg/", "deprecated")) or r == "deprecated" for r in roots)
    assert not any("connected-code-graph" in r for r in roots) and "docs/BACKLOG.md" not in roots


def test_the_real_repos_per_file_units_include_a_script_and_a_prompt_markdown():
    got = {(u.rootPath, u.language) for u in un.discover_units(REPO, mf.load_manifest(CPG / "ccg" / "system.yaml"))}
    assert ("AGENTS.md", "markdown") in got and ("opencode/AGENTS.md", "markdown") in got
    assert ("falkor-chat/scripts/start_agent_team.sh", "shell") in got
    assert ("cpg/README.md", "markdown") not in got


def test_discover_units_on_the_monorepo_equals_the_committed_golden_list():
    golden = json.loads(GOLDEN.read_text())
    assert golden["system"] == "graphmind-ai-lab" and len(golden["units"]) > 5
    got = [{"unitId": u.unitId, "repoId": u.repoId, "rootPath": u.rootPath, "language": u.language} for u in _real()]
    assert got == golden["units"], "regenerate with: .venv/bin/python -m ccg.units --golden (cwd cpg/)"


def test_the_golden_list_unit_ids_are_the_ids_hash_of_their_tuple():
    for u in json.loads(GOLDEN.read_text())["units"]:
        assert u["unitId"] == ids.tuple_id(["unit", u["repoId"], u["rootPath"], u["language"]])


def test_the_fixture_system_yields_its_units():
    system = mf.load_manifest(CPG / "fixtures" / "system1" / "system.yaml")
    got = {(u.repoId, u.rootPath, u.language) for u in
           un.discover_units(CPG / "fixtures" / "system1", system, per_file=False)}
    assert got == {
        ("fx-svc", ".", "python"), ("fx-svc", "common", "python"), ("fx-svc", "legacy", "python"),
        ("fx-svc", "tools", "python"), ("fx-web", "app", "javascript"), ("fx-web", "scripts", "javascript"),
        ("fx-job", ".", "python"),  # fx-web/scripts holds JS outside `app`: a fallback unit
    }


# --- fallback roots: order independence, coverage, no overlap with manifest units (review F1a, F1b) ----------------

# One symbolic tree: `R` a root-level name, `A` and `B` directory names. Every subset of these files is built under
# three name assignments with different sort orders; the units must be the same shape each time.
_POOL = [
    "{R}.py", "{A}/x.py", "{A}/{B}/y.py", "{A}/{B}/pyproject.toml", "{A}/w.js", "{A}/{B}/z.js", "{B}/q.py",
    "{A}/skip.py",
]
_NAMES = [("a", "b", "c"), ("z", "y", "x"), ("m", "a", "z")]
_EXCL = ["{A}/skip.py"]


def _shape(tmp: Path, subset, names) -> tuple[set, set, set]:
    """(units as symbolic (root, lang), manifest roots, files) for one subset under one name assignment."""
    r, a, b = names
    sym = {r: "R", a: "A", b: "B"}

    def back(path: str) -> str:
        return "/".join(
            sym[seg] if seg in sym else sym[seg[:-3]] + ".py" if seg[:-3] in sym and seg.endswith(".py") else seg
            for seg in path.split("/")
        )

    files = [f.format(R=r, A=a, B=b) for f in subset]
    root = tree(tmp, {f: "" for f in files})
    excl = [e.format(R=r, A=a, B=b) for e in _EXCL]
    got = un.discover_units(root, manifest_of(("r", "."), excl_paths=excl), per_file=False)
    return (
        {(back(u.rootPath), u.language) for u in got},
        {(back(posixdir(f)), "python") for f in files if f.endswith("pyproject.toml")},
        {back(f) for f in files if f not in excl},
    )


def posixdir(f: str) -> str:
    return f.rsplit("/", 1)[0] if "/" in f else "."


def test_fallback_units_do_not_depend_on_name_order_cover_every_code_file_and_overlap_only_when_forced(tmp_path):
    n = 0
    for k in range(len(_POOL) + 1):
        for subset in itertools.combinations(_POOL, k):
            shapes = []
            for i, names in enumerate(_NAMES):
                shapes.append(_shape(tmp_path / f"c{n}_{i}", subset, names))
            n += 1
            assert shapes[0][0] == shapes[1][0] == shapes[2][0], (subset, shapes)
            units, manifest_roots, files = shapes[0]
            r_set = {u for u in units}
            for f in files:  # coverage probe: every code file has an enclosing unit of its language
                lang = "python" if f.endswith((".py",)) or f.endswith("pyproject.toml") else "javascript"
                if not (f.endswith(".py") or f.endswith(".js")):
                    continue
                assert any(l == lang and (d == "." or f.startswith(d + "/")) for d, l in r_set), (subset, f, units)
            for d, lang in units:  # a fallback unit never encloses a manifest unit unless a file sits directly in it
                if (d, lang) in manifest_roots:
                    continue
                inside = [m for m, l in manifest_roots if l == lang and m != d and (d == "." or m.startswith(d + "/"))]
                direct = [f for f in files if f.endswith(".py" if lang == "python" else ".js") and posixdir(f) == d]
                assert not inside or direct, (subset, d, inside, units)
    assert n == 2 ** len(_POOL)


def test_a_root_level_file_sorting_before_or_after_a_directory_gives_the_same_units(tmp_path):
    first = code(tmp_path / "1", {"a.py": "", "b/x.py": ""})
    second = code(tmp_path / "2", {"b.py": "", "a/x.py": ""})
    assert first == [(".", "python"), ("b", "python")]
    assert second == [(".", "python"), ("a", "python")]


def test_a_fallback_unit_contains_no_manifest_units_files_in_a_nested_layout(tmp_path):
    got = code(tmp_path, {"fc/server/pyproject.toml": "", "fc/server/s.py": "", "fc/scripts/a.py": "", "fc/lib/b.py": ""})
    assert got == [("fc/lib", "python"), ("fc/scripts", "python"), ("fc/server", "python")]


def test_a_manifest_directory_sharing_a_name_prefix_with_an_ancestor_is_not_below_it(tmp_path):
    got = code(tmp_path, {"ab/pyproject.toml": "", "a/x/y.py": ""})
    assert got == [("a", "python"), ("ab", "python")]


def test_two_unowned_files_in_one_directory_with_a_nested_manifest_force_the_overlap_and_the_nested_unit_wins(tmp_path):
    got = code(tmp_path, {"fc/x.py": "", "fc/server/pyproject.toml": "", "fc/server/s.py": ""})
    assert got == [("fc", "python"), ("fc/server", "python")]  # forced: x.py can only root at `fc`


def test_in_the_real_repo_no_fallback_unit_encloses_a_manifest_unit():
    units = _real()
    py_manifest = {u.rootPath for u in units if (REPO / u.rootPath / "pyproject.toml").exists()
                   or (REPO / u.rootPath / "requirements.txt").exists()}
    js_manifest = {u.rootPath for u in units if (REPO / u.rootPath / "package.json").exists()}
    assert "falkor-chat/server" in py_manifest and "salesperson" in js_manifest
    for u in units:
        manifest = py_manifest if u.language == "python" else js_manifest
        if u.rootPath in manifest:
            continue
        assert not [m for m in manifest if m.startswith(u.rootPath + "/")], u


# --- exclusion semantics agree with the census matcher (review F2) ---------------------------------------------


def _export_kit():
    spec = importlib.util.spec_from_file_location("export_kit", CPG / "eval" / "tools" / "export_kit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("export_kit", mod)
    spec.loader.exec_module(mod)
    return mod


_PLANTED = [
    "docs/a/connected-code-graph-d/x.md", "docs/connected-code-graph/y.md", "docs/connected-code-graph-p1.md",
    "docs/x/y/connected-code-graph/z/w.md", "docs/plans/other.md", "docs2/connected-code-graph/a.md",
    "x/docs/connected-code-graph-a.md", "connected-code-graph/a.md", "cpg/a.py", "cpg2/a.py", "app/cpg/a.py",
    "deprecated/a.py", "deprecated2/a.py", "docs/BACKLOG.md", "docs/sub/BACKLOG.md", "claude/cobb/kaizen/plan.md",
    "claude/cobb/kaizen/plan.md.sh", "claude/cobb/kaizen/other.md",
]


def test_the_manifest_exclusions_agree_with_export_kit_over_every_tracked_path_and_planted_axes():
    ek = _export_kit()
    repo = mf.load_manifest(CPG / "ccg" / "system.yaml").repos[0]
    excluded = un._excluder(repo)
    content = set(repo.content_check_files)
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True).stdout
    paths = [p for p in tracked.decode().split("\0") if p] + _PLANTED
    assert len(paths) > 1000
    diffs = [p for p in paths if excluded(p) != (ek.path_excluded(p) or p in content)]
    assert diffs == []
    # Positive controls: the axes are really exercised in both directions.
    assert excluded("docs/a/connected-code-graph-d/x.md") and not excluded("docs/plans/other.md")
    assert excluded("claude/cobb/kaizen/plan.md") and not excluded("claude/cobb/kaizen/other.md")


def test_a_glob_exclusion_also_excludes_files_below_a_matching_directory(tmp_path):
    tree(tmp_path, {"docs/g/connected-code-graph-d/x.md": "", "docs/g/ok.md": ""})
    got = {u.rootPath for u in un.discover_units(tmp_path, manifest_of(("r", "."),
                                                                      excl_paths=["docs/**/connected-code-graph*"]))}
    assert got == {"docs/g/ok.md"}


@pytest.mark.parametrize("entry", ["dep", "dep/"])
def test_a_directory_named_without_a_trailing_slash_excludes_what_is_below_it(tmp_path, entry):
    got = code(tmp_path, {"dep/x.py": "", "dep/sub/pyproject.toml": "", "depot/y.py": ""}, excl_paths=[entry])
    assert got == [("depot", "python")]
