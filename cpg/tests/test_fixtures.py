"""S39: the fixture system's own tests (TP-001..006, 013, 014, 016, 018, the AC-25 column of TP-008, TP-006 live).

Plan: ``docs/test-plans/connected-code-graph-fixtures.md`` v1.3, section 9. Run with ``cpg/`` as the working
directory (``.venv/bin/python -m pytest -q tests/test_fixtures.py``); the ``live`` item needs Docker, the
``ccg-toolchain`` image and the S08 runner (``-m live``). Every check carries a planted failing input (a mutant
manifest, repo copy or answer builder) that must turn it red: a check that cannot fail is not a check.

Named-symbol rule (ruling, S39 only): D15's unit floor (`cpg-content-check.sc`, S08, unchanged) passes any unit whose
files emit the synthetic `<module>` METHOD, so a comments-only unit reads `ok`. This file adds a named-symbol rule and
per-unit floors (`UNIT_FLOORS`) for the fixtures. **S11, S25 and S28 must apply a named-symbol rule themselves**; they
cannot rely on the shared check's verdict alone.

Conventions fixed by this file's schema (``ccg/schemas/fixture-manifest-v1.schema.json``):
F-6 (tree = base tree + forward ops), F-8 (TP-018 keys on (literal, enclosing locator)), F-4 (skew key form).
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import jsonschema
import pytest
import yaml

CPG = Path(__file__).resolve().parents[1]
DOCS = CPG.parent / "docs"
PLAN = DOCS / "test-plans" / "connected-code-graph-fixtures.md"
SCHEMA_PATH = CPG / "ccg" / "schemas" / "fixture-manifest-v1.schema.json"
REGISTRY_PATH = CPG / "ccg" / "registry" / "reasons.yaml"
MATERIALIZE = CPG / "fixtures" / "materialize.py"


def _load_materialize(name="ccg_fixture_materialize", path=MATERIALIZE):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_materialize()
REPOS = M.REPOS
SCHEMA = json.loads(SCHEMA_PATH.read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


# ======================================================================================== shared fixtures


@pytest.fixture(scope="session")
def manifests():
    return M.load_all()


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """The four repositories materialized once for the session: (out dir, SHA map)."""
    out = tmp_path_factory.mktemp("fixtures-session")
    return out, M.materialize(out)


def registry_codes() -> dict:
    rows = yaml.safe_load(REGISTRY_PATH.read_text())
    return {r["code"]: r["status"] for r in rows}


def all_docs(manifests):
    """Every manifest document by name, cross.yaml included."""
    docs = dict(manifests)
    docs["cross"] = M._fix_keys(yaml.safe_load(M.CROSS_PATH.read_text()))
    return docs


# ======================================================================================== TP-001 determinism


def _shas_equal_or_raise(a: dict, b: dict):
    assert a == b, "materializer is not deterministic: " + json.dumps(
        {r: {k: (a[r][k], b[r][k]) for k in a[r] if a[r][k] != b[r][k]} for r in a}, indent=1)[:600]


def _run_twice(mod, tmp_path, sleep=0.0):
    first = mod.materialize(tmp_path / "a")
    time.sleep(sleep)
    second = mod.materialize(tmp_path / "b")
    return first, second


def test_tp001_two_runs_give_the_same_shas_for_every_commit_branch_and_tag(tmp_path):
    first, second = _run_twice(M, tmp_path, sleep=1.1)
    _shas_equal_or_raise(first, second)
    assert set(first) == set(REPOS)


def test_tp001_shas_do_not_depend_on_umask_timezone_home_or_user_git_config(tmp_path):
    """Run the CLI in a hostile ambient environment; the SHA map must equal a plain in-process run."""
    plain = M.materialize(tmp_path / "plain")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[user]\n name = Someone Else\n email = else@example.org\n"
                                     "[core]\n autocrlf = true\n[commit]\n gpgsign = true\n[i18n]\n commitEncoding = ISO-8859-1\n")
    env = {**os.environ, "HOME": str(home), "TZ": "Pacific/Kiritimati", "GIT_AUTHOR_NAME": "Mallory",
           "GIT_COMMITTER_DATE": "2030-01-01T00:00:00", "GIT_AUTHOR_DATE": "2030-01-01T00:00:00",
           "XDG_CONFIG_HOME": str(home)}
    code = ("import os,sys,json; os.umask(0o077); sys.path.insert(0, %r); import materialize as m; "
            "print(json.dumps(m.materialize(%r), sort_keys=True))" % (str(MATERIALIZE.parent), str(tmp_path / "hostile")))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    hostile = json.loads(r.stdout)
    _shas_equal_or_raise(json.loads(json.dumps(plain)), hostile)


def test_tp001_planted_mutant_committer_date_unset_is_detected(tmp_path):
    """The plan's planted failing input: a materializer that leaves the committer date to the clock."""
    mut = _load_materialize("ccg_fixture_materialize_mut")
    real = mut._git_env

    def no_committer_date(date):
        env = real(date)
        env.pop("GIT_COMMITTER_DATE", None)
        return env

    mut._git_env = no_committer_date
    first, second = _run_twice(mut, tmp_path, sleep=1.1)
    with pytest.raises(AssertionError, match="not deterministic"):
        _shas_equal_or_raise(first, second)


# F-7: pinned SHAs (the materializer is also deterministic against a recorded value, not only run against run).
# These change when a fixture tree, a manifest history block or the materializer's fixed constants change on purpose:
# regenerate with `cd cpg && .venv/bin/python fixtures/materialize.py --json` and update this table in the same commit.
PINNED_SHAS = {
    'fx-job': {
        "commits": {'H0': '7eb574a91d1abfc3e38930e139be2ef38ef8b42d', 'H1': '1595bdc063bacd92728767a049aa5ed5421ddec3', 'H2': '40f23388e6ab3b01ab5b1191857fb6059aff0d6d', 'H3': 'bf1fff8f2fdfc1aaeba960bfae771d6206dff154', 'H4': 'efbaa1946d52d5f7b4b3618bf73e7930f009db39'},
        "branches": {'ov-job': '6a29ed25ee6808f7d4bfb063a6cb6589f389612f'},
    },
    'fx-lib': {
        "commits": {'H0': '436df2c4bfdfda2f7253e9708b83d47b27998b17', 'H1': '0042d7183f93e2c0023129f3a50a21fd54f3f599', 'H3': '7f407188f9743be5c975d45faed8b5238410e175'},
        "branches": {},
    },
    'fx-svc': {
        "commits": {'H0': 'e9eaec94de55c6e3f6f48bde9943dc7541bc8c71', 'H1': 'f07306e5e04f4be2d82529ee5e8d26e60be7e9b0', 'H2': '72543cc6fd3b76a34b097777d29003481babacbe', 'H3': '0c826e7677a69ad75abb4c936fdfd64ae8b6f802', 'H4': '3547b6ffbedb6826ff59c2922f0f4749bda60787'},
        "branches": {'ov-svc': '3a6312f2df7b95fd9adb4b0ff7ddacb9bcbe72a4', 'tf-mid': '4ccf96f30043835fccb038f33b7241221ece991f', 'tf-sink': '4cc6371be31f2c67ca3c597c45254920653fc1de', 'tf-src': '9007d1e83d5bcfd1e7be9a0a9b8caa6272e9cdc8'},
    },
    'fx-web': {
        "commits": {'H0': '83926b1ce6a603de66e276a666bccdf5714d53ad', 'H1': '26ecd3c6e93ab35ad0744f905a89115688850b3e'},
        "branches": {},
    },
}


def sha_pin_violations(shas):
    bad = []
    for repo, got in shas.items():
        want = PINNED_SHAS[repo]
        if got["commits"] != want["commits"]:
            bad.append((repo, "commits", {k: v for k, v in got["commits"].items() if want["commits"].get(k) != v}))
        if {k: v for k, v in got["branches"].items() if k != "main"} != want["branches"]:
            bad.append((repo, "branches"))
    return bad


def test_f7_materialized_shas_equal_the_pinned_shas(built):
    assert sha_pin_violations(built[1]) == []
    assert sum(len(v["commits"]) + len(v["branches"]) for v in PINNED_SHAS.values()) == 15 + 5


def test_f7_file_modes_are_only_0644_and_0755_and_scripts_are_executable_as_in_the_tree(built):
    out, _ = built
    for repo in REPOS:
        ls = git(out / repo, "ls-files", "-s").splitlines()
        modes = {ln.split("\t")[1]: ln.split()[0] for ln in ls}
        assert set(modes.values()) <= {"100644", "100755"}, repo
        for path, mode in modes.items():
            src = M.SYSTEM / repo / path
            if src.exists():
                want = "100755" if os.access(src, os.X_OK) and src.stat().st_mode & 0o111 else "100644"
                assert mode == want, (repo, path, mode)


def test_tp001_history_shape_matches_the_plan_section_8(built, manifests):
    out, shas = built
    want = {"fx-svc": ["H0", "H1", "H2", "H3", "H4"], "fx-web": ["H0", "H1"], "fx-lib": ["H0", "H1", "H3"],
            "fx-job": ["H0", "H1", "H2", "H3", "H4"]}
    for repo, tags in want.items():
        assert list(shas[repo]["tags"]) == tags, repo
        assert shas[repo]["head"] == shas[repo]["commits"][tags[-1]]
    assert sorted(b for b in shas["fx-svc"]["branches"] if b != "main") == ["ov-svc", "tf-mid", "tf-sink", "tf-src"]
    assert sorted(b for b in shas["fx-job"]["branches"] if b != "main") == ["ov-job"]
    assert [b for b in shas["fx-web"]["branches"] if b != "main"] == [] and [b for b in shas["fx-lib"]["branches"] if b != "main"] == []
    for repo in REPOS:
        log = git(out / repo, "log", "--format=%an|%ae|%cn|%ce|%ad|%cd", "--date=iso-strict", "--all").splitlines()
        assert all(l.startswith("Fixture Author|fixture@example.invalid|Fixture Author|fixture@example.invalid|") for l in log)
        assert all(l.endswith("+00:00") for l in log)


def test_tp001_branches_are_cut_from_their_base_commit_and_differ_from_it_by_their_ops_only(built, manifests):
    out, shas = built
    n = 0
    for repo in REPOS:
        for b in manifests[repo]["history"].get("branches", []):
            parent = git(out / repo, "rev-parse", f"{b['name']}^")
            assert parent == shas[repo]["commits"][b["base"]], (repo, b["name"])
            changed = set(git(out / repo, "diff", "--name-only", b["base"], b["name"]).splitlines())
            assert changed == {o["path"] for o in b["ops"]}, (repo, b["name"], changed)
            n += 1
    assert n == 5, "ov-svc, tf-mid, tf-sink, tf-src, ov-job"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_tp001_heads_realize_the_checked_in_trees_where_the_plan_says_so(built):
    """fx-web/fx-lib/fx-job HEAD is the checked-in tree (F-6: only H0/early commits differ); fx-svc HEAD adds H2-H4."""
    out, _ = built
    for repo in ("fx-web", "fx-lib", "fx-job"):
        r = subprocess.run(["diff", "-rq", "-x", ".git", "-x", "__pycache__", str(out / repo), str(M.SYSTEM / repo)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout
    assert (out / "fx-svc" / "legacy" / "broken.py").is_file(), "AB-1: broken.py is added at H2 and never reverted"
    assert not (out / "fx-job" / "job" / "broken.py").exists(), "fx-job's failure is reverted at H3"


def test_f6_history_ops_are_strict(tmp_path):
    """A missed or ambiguous anchor is an error, never a silent no-op."""
    (tmp_path / "f.txt").write_text("a\nb\na\n")
    with pytest.raises(M.MaterializeError, match="2 times"):
        M.apply_ops(tmp_path, [{"op": "replace_text", "path": "f.txt", "from": "a\n", "to": "c\n"}], "t")
    with pytest.raises(M.MaterializeError, match="0 times"):
        M.apply_ops(tmp_path, [{"op": "replace_text", "path": "f.txt", "from": "zzz", "to": "c"}], "t")
    with pytest.raises(M.MaterializeError, match="already exists"):
        M.apply_ops(tmp_path, [{"op": "add_file", "path": "f.txt", "content": "x"}], "t")
    with pytest.raises(M.MaterializeError, match="no such file"):
        M.apply_ops(tmp_path, [{"op": "delete_file", "path": "g.txt"}], "t")


def test_materialize_refuses_to_clobber_a_directory_it_did_not_create(tmp_path):
    (tmp_path / "fx-svc").mkdir()
    (tmp_path / "fx-svc" / "precious").write_text("x")
    with pytest.raises(M.MaterializeError, match="exists"):
        M.materialize(tmp_path)
    with pytest.raises(M.MaterializeError, match="not a materialized fixture"):
        M.materialize(tmp_path, force=True)
    assert (tmp_path / "fx-svc" / "precious").read_text() == "x"


# ======================================================================================== TP-003 manifests validate


def errors(doc):
    return sorted(VALIDATOR.iter_errors(doc), key=lambda e: list(map(str, e.path)))


def row_by_id(doc, rid):
    return next(r for r in doc["rows"] if r["id"] == rid)


@pytest.mark.parametrize("name", [*REPOS, "cross"])
def test_tp003_manifest_validates_against_the_schema(manifests, name):
    doc = all_docs(manifests)[name]
    errs = errors(doc)
    assert not errs, [f"{list(e.path)}: {e.message[:120]}" for e in errs[:5]]


def test_tp003_the_schema_is_a_valid_draft_2020_12_schema():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def reason_codes(reason):
    """Registry codes inside a reason value: 'a', 'a / b' or {edge, link}."""
    if isinstance(reason, dict):
        return [reason["edge"], reason["link"]]
    return [c.strip() for c in reason.split("/")]


def reasons_not_in_registry(doc):
    reg = registry_codes()
    bad = []
    for r in doc["rows"]:
        if r.get("reason") is None:
            continue
        for c in reason_codes(r["reason"]):
            if c != "none" and c not in reg:
                bad.append((r["id"], c))
    return bad


@pytest.mark.parametrize("name", [*REPOS, "cross"])
def test_tp003_every_reason_code_is_in_the_registry_and_agrees_with_the_status(manifests, name):
    doc = all_docs(manifests)[name]
    assert reasons_not_in_registry(doc) == []
    reg = registry_codes()
    for r in doc["rows"]:
        st = r.get("status", r.get("st"))
        if st is None or r.get("reason") is None:
            continue
        codes = [c for c in reason_codes(r["reason"]) if c != "none"]
        if st == "possible":
            assert codes, f"{r['id']}: a possible row needs its reason"
        # the edge/link pair may mix statuses (a confirmed link over a possible edge); a single code must agree
        if isinstance(r["reason"], str) and " / " not in r["reason"] and codes:
            assert reg[codes[0]] in (st, "none", "possible" if st == "possible" else st), (r["id"], codes, st)


def test_tp003_planted_failing_inputs_are_rejected(manifests):
    base = copy.deepcopy(manifests["fx-svc"])
    # (1) a `possible` row with no reason
    d = copy.deepcopy(base)
    row = next(r for r in d["rows"] if r.get("status") == "possible")
    del row["reason"]
    assert errors(d), "possible with no reason must fail the schema"
    # (2) a row with reason `foo-bar`: the schema accepts the shape, the registry check rejects it
    d = copy.deepcopy(base)
    row = next(r for r in d["rows"] if r.get("status") == "possible" and isinstance(r["reason"], str) and "/" not in r["reason"])
    row["reason"] = "foo-bar"
    assert not errors(d) and reasons_not_in_registry(d) == [(row["id"], "foo-bar")]


def test_f6_one_history_convention_the_schema_rejects_reverse_constructs(manifests):
    """F-6: tree state is base tree + forward ops. A reverse construct or a second tree-state key is rejected."""
    for name in ("fx-job", "fx-lib"):
        d = copy.deepcopy(manifests[name])
        d["pre_h3_edits"] = [{"path": "x", "old": "a", "new": "b"}]            # the old fx-lib form
        assert errors(d), f"{name}: top-level pre_h3_edits must be rejected"
        d = copy.deepcopy(manifests[name])
        d["history"]["commits"][0]["treeState"] = "HEAD"                     # a per-commit tree-state flag
        assert errors(d), f"{name}: a commit-level treeState must be rejected"
        d = copy.deepcopy(manifests[name])
        d["history"]["commits"][0]["ops"].append({"op": "reverse_text", "path": "x", "from": "a", "to": "b"})
        assert errors(d), f"{name}: an unknown (reverse) op must be rejected"
        d = copy.deepcopy(manifests[name])
        del d["history"]["baseTree"]
        assert errors(d), f"{name}: baseTree is required"
        d = copy.deepcopy(manifests[name])
        d["history"]["commits"].pop(0)
        assert errors(d), f"{name}: the first commit must be H0"
    d = copy.deepcopy(manifests["fx-web"])
    d["history"] = {"H0": {"date": "2026-01-05", "content": "x"}, "H1": {"date": "2026-02-10", "content": "y"}}
    assert errors(d), "the dict-of-commits form of the builders' first drafts must be rejected"


def test_f6_every_manifest_expresses_h0_as_base_tree_plus_h0_ops(manifests):
    """The three repos whose checked-in tree is not H0 carry non-empty H0 ops; the others carry none."""
    h0 = {r: manifests[r]["history"]["commits"][0]["ops"] for r in REPOS}
    assert h0["fx-svc"] == [] and h0["fx-web"] == []
    assert h0["fx-job"] and h0["fx-lib"]
    for r in REPOS:
        assert manifests[r]["history"]["commits"][0]["id"] == "H0"


def test_f6_the_materialized_h0_and_h1_trees_are_the_plan_trees(built):
    """Plan s8: H0 = the initial tree (out of window), H1 keeps the pre-H3 signatures, H4 holds the comment-only edits."""
    out, _ = built

    def show(repo, ref, path):
        return git(out / repo, "show", f"{ref}:{path}")

    # fx-job: tree = HEAD (H4); H0..H3 lack the H4 comment line
    for ref in ("H0", "H1", "H3"):
        assert "Reviewed for the quarterly audit" not in show("fx-job", ref, "job/run_report.py"), ref
    assert "# Reviewed for the quarterly audit." in show("fx-job", "H4", "job/run_report.py")
    assert git(out / "fx-job", "diff", "--stat", "H0", "H1") == ""
    # fx-lib: tree = HEAD (H3); H0 and H1 have quote(int,String) and no 3-argument overload
    pr = "pricing/src/main/java/com/acme/fxlib/Pricing.java"
    for ref in ("H0", "H1"):
        t = show("fx-lib", ref, pr)
        assert "quote(int qty, String sku) {" in t and "String sku, Currency currency) {" not in t.split("quote(int qty, String sku, Currency currency, Locale")[0], ref
    assert "quote(int qty, String sku, Currency currency) {" in show("fx-lib", "H3", pr)
    assert show("fx-lib", "H0", "quote-cli/src/main/java/com/acme/fxlib/cli/QuoteCli.java").count('Pricing.quote(1, "SKU-1")') == 1
    # fx-svc: tree = H0; price gains region at H3; H1 still has price(item, qty)
    assert "def price(item, qty):" in show("fx-svc", "H1", "core/flow.py")
    assert "def price(item, qty, *, region=None):" in show("fx-svc", "H3", "core/flow.py")
    assert "# Pricing flow, reviewed." in show("fx-svc", "H4", "core/flow.py")


def test_f4_skew_keys_have_one_form_in_every_manifest(manifests):
    """LNK-34/35/41 (and 33): the provider's full key `python:<module>.<name>(<params>)` wherever the row is held."""
    form = re.compile(r"^python:fx_svc_common\.(fmt|split)\.\w+\([^()]*\)$")
    seen = {}
    for repo in REPOS:
        for row in manifests[repo]["rows"]:
            if row["id"] in ("LNK-33", "LNK-34", "LNK-35", "LNK-41"):
                key = row.get("keyNorm", row.get("key"))
                assert key and form.match(key), (repo, row["id"], key)
                seen.setdefault(row["id"], set()).add(key)
    assert set(seen) == {"LNK-33", "LNK-34", "LNK-35", "LNK-41"}
    assert all(len(v) == 1 for v in seen.values()), seen


SKEW_ROWS = ("LNK-33", "LNK-34", "LNK-35", "LNK-41")


def provider_skew_key(svc_dir: Path, rel_path: str, name: str) -> str:
    """`python:<module>.<name>(<params>)` derived from the provider source with ast (defaults dropped, `*` marks keyword-only)."""
    tree = ast.parse((svc_dir / rel_path).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    a = fn.args
    params = [x.arg for x in [*a.posonlyargs, *a.args]]
    if a.vararg:
        params.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        params.append("*")
    params += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        params.append("**" + a.kwarg.arg)
    module = ".".join(Path(rel_path).with_suffix("").parts[1:])                 # common/fx_svc_common/fmt.py -> fx_svc_common.fmt
    return f"python:{module}.{name}({','.join(params)})"


def skew_key_violations(manifests, svc_dir):
    bad = []
    for rid, held in M.holders(manifests).items():
        if rid not in SKEW_ROWS:
            continue
        loc = next(l for r in held.values() for f, l in M.row_locators(r) if f in ("P", "provider"))
        want = provider_skew_key(svc_dir, loc["path"], loc["qname"].split("::")[-1])
        for repo, row in held.items():
            key = row.get("keyNorm", row.get("key"))
            if key != want:
                bad.append((rid, repo, key, want))
    return bad


def test_f4_skew_keys_are_derived_from_the_provider_source(manifests, built):
    svc = built[0] / "fx-svc"
    assert skew_key_violations(manifests, svc) == []
    assert provider_skew_key(svc, "common/fx_svc_common/fmt.py", "round_money") == "python:fx_svc_common.fmt.round_money(amount,*,mode)"
    assert provider_skew_key(svc, "common/fx_svc_common/split.py", "split_money") == "python:fx_svc_common.split.split_money(amount,parts,*,remainder)"


def test_f4_probe_every_axis_of_every_skew_key_turns_red(manifests, built):
    """Each key x {module, name, each parameter, the `*` marker (removed, moved)}, applied consistently to every holder."""
    svc = built[0] / "fx-svc"
    n, survivors = 0, []
    for rid in SKEW_ROWS:
        key = next(r.get("keyNorm", r.get("key")) for r in M.holders(manifests)[rid].values())
        m = re.match(r"^python:(fx_svc_common)\.(\w+)\.(\w+)\((.*)\)$", key)
        _, mod, name, params = m.groups()
        plist = params.split(",")
        variants = {"module": key.replace(f".{mod}.", ".zz." if mod != "zz" else ".yy."),
                    "name": key.replace(f".{name}(", f".{name}x(")}
        for i, prm in enumerate(plist):
            if prm != "*":
                variants[f"param:{prm}"] = "python:fx_svc_common.%s.%s(%s)" % (mod, name, ",".join(x + "x" if j == i else x for j, x in enumerate(plist)))
        if "*" in plist:
            variants["star-removed"] = key.replace("*,", "")
            others = [x for x in plist if x != "*"]
            variants["star-moved"] = "python:fx_svc_common.%s.%s(%s)" % (mod, name, ",".join(["*", *others]))
        for axis, bad_key in variants.items():
            bad = copy.deepcopy(manifests)
            for r in M.holders(bad)[rid].values():
                r["keyNorm" if "keyNorm" in r else "key"] = bad_key
            n += 1
            if not skew_key_violations(bad, svc):
                survivors.append((rid, axis))
    assert survivors == []
    assert n >= 20
    print(f"F-4 probe: {n} mutants, 0 survivors")


def test_f4_planted_short_skew_key_is_rejected_by_the_schema(manifests):
    d = copy.deepcopy(manifests["fx-job"])
    row_by_id(d, "LNK-34")["key"] = "round_money"                         # the pre-S39 fx-job form
    assert errors(d)
    d = copy.deepcopy(manifests["fx-job"])
    row_by_id(d, "LNK-41")["key"] = "fmt_date"
    assert errors(d)


def test_f8_a_planted_literal_must_name_its_enclosing_locator(manifests):
    d = copy.deepcopy(manifests["fx-svc"])
    row = next(r for r in d["rows"] if "literal" in r)
    for k in ("consumer", "provider", "locator", "C", "P"):
        row.pop(k, None)
    assert errors(d)


# ======================================================================================== cross.yaml


def test_cross_yaml_equals_the_reconciliation_of_the_four_manifests(manifests):
    """Drift test: cross.yaml is exactly what `materialize.py --emit-cross` produces."""
    assert M.CROSS_PATH.read_text() == M.emit_cross(manifests)


def test_cross_yaml_holds_every_multi_held_row_once_and_the_three_cross_unit_rows(manifests):
    held = M.holders(manifests)
    multi = {rid for rid, h in held.items() if len(h) > 1}
    cross = all_docs(manifests)["cross"]
    ids = [r["id"] for r in cross["rows"]]
    assert len(ids) == len(set(ids)), "deduped by id"
    assert set(ids) == multi | {"AB-1", "LNK-42", "LNK-43"}
    assert len(multi) == 40 and len(ids) == 43
    for r in cross["rows"]:
        assert set(r["holders"]) == set(held[r["id"]]), r["id"]


def test_cross_yaml_agrees_with_every_holder_on_status_reason_gate_and_key(manifests):
    cross = all_docs(manifests)["cross"]
    held = M.holders(manifests)
    for c in cross["rows"]:
        for repo, row in held[c["id"]].items():
            v = M.row_view(row)
            assert v["gate"] == c["gate"], (c["id"], repo)
            if v["status"] is not None:
                assert v["status"] == c.get("status"), (c["id"], repo)
            if v["reason"] is not None:
                assert v["reason"] == c.get("reason"), (c["id"], repo)
            if v["key"] is not None:
                assert v["key"] == c.get("keyNorm"), (c["id"], repo)


def test_cross_reconciliation_rejects_holders_that_disagree(manifests):
    """Planted failing input: the F-4 keys, as they stood before S39, make the reconciliation fail loudly."""
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-job"], "LNK-08")["key"] = "GET /api/items/{}"           # F-4's other disagreement
    with pytest.raises(M.ReconcileError, match="LNK-08.*key"):
        M.reconcile(bad)
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-job"], "LNK-34")["gate"] = "P1"
    with pytest.raises(M.ReconcileError, match="gate"):
        M.reconcile(bad)
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-job"], "LNK-34")["st"] = "confirmed"
    with pytest.raises(M.ReconcileError, match="status"):
        M.reconcile(bad)


# ======================================================================================== TP-002 locators resolve
# A locator is {repo, path, qname | keyPath | objectKey}. It resolves when the file exists at a pinned ref and the
# symbol is declared in it (spec3 s3.5, s1.9). Resolvers are per language; each returns (start_line, end_line) of
# the declaration or None. They read text, never import fixture code.

JAVA_KW = {"if", "for", "while", "switch", "catch", "return", "new", "throw", "else", "do", "try", "synchronized"}


def _brace_end(lines, start):
    """1-based line where the brace block opened on/after *start* closes (or start if there is none)."""
    depth, opened = 0, False
    for i in range(start - 1, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return i + 1
        if not opened and lines[i].rstrip().endswith(";"):
            return i + 1
    return start


def resolve_python(text, qname):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    found = {}

    def visit(body, prefix):
        for node in body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            for n in names:
                found.setdefault(prefix + n, (node.lineno, node.end_lineno))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                visit(node.body, prefix)

    visit(tree.body, "")
    return found.get(qname)


def resolve_ts(text, qname):
    lines = text.splitlines()
    parts = qname.split(".")
    head = re.escape(parts[0])
    decl = re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?"
                      rf"(?:function\*?\s+{head}\b|(?:const|let|var)\s+{head}\b|class\s+{head}\b|interface\s+{head}\b|"
                      rf"type\s+{head}\b|enum\s+{head}\b)")
    start = next((i + 1 for i, ln in enumerate(lines) if decl.match(ln)), None)
    if start is None:
        return None
    end = _brace_end(lines, start)
    if len(parts) == 1:
        return (start, end)
    body = "\n".join(lines[start - 1:end])
    for seg in parts[1:]:
        if not re.search(rf"(?<![\w$.]){re.escape(seg)}\s*[(:=<?!]", body) and not re.search(rf"\b{re.escape(seg)}\b", body):
            return None
    return (start, end)


def _split_params(params):
    out, depth, cur = [], 0, ""
    for ch in params:
        depth += ch in "<[("
        depth -= ch in ">])"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def java_param_types(params):
    types = []
    for p in _split_params(params):
        p = re.sub(r"@\w+(\([^)]*\))?\s*", "", p).replace("final ", "").strip()
        m = re.match(r"^(.*\S)\s+\w+$", p)
        types.append(re.sub(r"\s+", "", m.group(1) if m else p))
    return types


def java_methods(text):
    """[(class, name, [param types], start, end)] plus nothing for fields (see resolve_java)."""
    lines = text.splitlines()
    classes = [(m.start(), m.group(1)) for m in re.finditer(r"\b(?:class|interface|enum|record)\s+(\w+)", text)]
    out = []
    hdr = re.compile(r"(?m)^[ \t]*(?:@\w+(?:\([^)]*\))?[ \t]*\n[ \t]*)*(?:(?:public|protected|private|static|final|abstract|"
                     r"synchronized|native|default)\s+)*(?:<[^>]+>\s+)?(?:[\w.<>\[\]?,]+\s+)?(\w+)\s*\(([^)]*)\)\s*"
                     r"(?:throws\s+[\w.,\s]+)?\{")
    for m in hdr.finditer(text):
        name = m.group(1)
        if name in JAVA_KW:
            continue
        owners = [c for pos, c in classes if pos < m.start()]
        if not owners:
            continue
        start = text.count("\n", 0, m.start(1)) + 1
        out.append((owners[-1], name, java_param_types(m.group(2)), start, _brace_end(lines, start)))
    return out


def resolve_java(text, qname):
    m = re.match(r"^([\w.]+?)(?:\((.*)\))?$", qname)
    if not m:
        return None
    dotted, sig = m.group(1), m.group(2)
    cls, _, member = dotted.rpartition(".")
    cls = cls.split(".")[-1]
    if not cls:
        return None
    if sig is None and not re.search(rf"\b{re.escape(member)}\b\s*[=;(]", text):
        return None
    cands = [(c, n, ps, s, e) for c, n, ps, s, e in java_methods(text) if c == cls and n == member]
    if sig is not None:
        want = [re.sub(r"\s+", "", x) for x in _split_params(sig)]
        cands = [c for c in cands if c[2] == want]
        return (cands[0][3], cands[0][4]) if cands else None
    if cands:
        return (cands[0][3], cands[0][4])
    m2 = re.search(rf"^.*\b{re.escape(member)}\b\s*[=;].*$", text, re.M)       # a field
    return (text.count("\n", 0, m2.start()) + 1,) * 2 if m2 else None


def resolve_keypath(text, path, key_path):
    data = json.loads(text) if path.endswith(".json") else yaml.safe_load(text)

    def walk(node, parts):
        if not parts:
            return True
        if isinstance(node, dict):
            for i in range(1, len(parts) + 1):
                k = ".".join(parts[:i])
                if k in node and walk(node[k], parts[i:]):
                    return True
            return False
        if isinstance(node, list):
            if len(parts) == 1:
                return any((isinstance(x, str) and (x == parts[0] or x.startswith(parts[0] + "="))) or
                           (isinstance(x, dict) and parts[0] in (x.get("name"), *x.keys())) for x in node)
            return any(walk(x, parts) for x in node if isinstance(x, (dict, list)))
        return False

    return (1, 1) if walk(data, key_path.split(".")) else None


def resolve_text(path, text, loc):
    """Resolve one locator against a file's text; (start, end) or None."""
    if loc.get("keyPath"):
        return resolve_keypath(text, path, loc["keyPath"])
    if loc.get("objectKey"):
        m = re.match(r"^sql:table:(\w+)$", loc["objectKey"])
        hit = re.search(rf"(?i)create\s+table\s+(?:if\s+not\s+exists\s+)?{re.escape(m.group(1))}\b", text) if m else None
        return (text.count("\n", 0, hit.start()) + 1,) * 2 if hit else None
    qname = loc.get("qname")
    if not qname:
        return (1, max(1, text.count("\n") + 1))
    if qname.startswith(path + "::"):
        qname = qname[len(path) + 2:]
    if qname.startswith("<"):                                   # <document> / <script>: the file itself
        return (1, max(1, text.count("\n") + 1))
    ext = os.path.splitext(path)[1]
    if ext in (".py", ".pyi"):
        return resolve_python(text, qname)
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cts"):
        return resolve_ts(text, qname)
    if ext == ".java":
        return resolve_java(text, qname)
    if ext == ".sh":
        return (1, 1) if re.search(rf"\b{re.escape(qname)}\b", text) else None
    return None


class Tree:
    """File access to a materialized repository at a ref (HEAD = working tree, else `git show`)."""

    def __init__(self, repo_dir: Path):
        self.dir = repo_dir
        self._cache = {}

    def read(self, ref, path):
        key = (ref, path)
        if key not in self._cache:
            if ref == "HEAD":
                f = self.dir / path
                self._cache[key] = f.read_text() if f.is_file() else None
            else:
                r = subprocess.run(["git", "-C", str(self.dir), "show", f"{ref}:{path}"], capture_output=True, text=True)
                self._cache[key] = r.stdout if r.returncode == 0 else None
        return self._cache[key]


def refs_of(shas, repo):
    return ["HEAD", *shas[repo]["tags"], *[b for b in shas[repo]["branches"] if b != "main"]]


def resolve_locator(trees, shas, loc):
    """Return ('ok', ref, span) | ('nofile', ...) | ('nosymbol', ...) trying HEAD then every tag and branch."""
    tree = trees[loc["repo"]]
    seen_file = False
    for ref in refs_of(shas, loc["repo"]):
        text = tree.read(ref, loc["path"])
        if text is None:
            continue
        seen_file = True
        span = resolve_text(loc["path"], text, loc)
        if span:
            return "ok", ref, span
    return ("nosymbol" if seen_file else "nofile"), None, None


def all_locators(docs):
    """[(doc, row id, field, loc)] over the four repo manifests (the cross rows are strings and checked separately)."""
    out = []
    for name in REPOS:
        for row in docs[name]["rows"]:
            for field, loc in M.row_locators(row):
                out.append((name, row["id"], field, loc))
    for row in docs["cross"]["rows"]:
        for field in ("provider", "consumer", "refs"):
            for s in row.get(field, []):
                out.append(("cross", row["id"], field, M.parse_locator(s)))
    return out


def unresolved(docs, trees, shas, locs=None):
    bad = []
    for name, rid, field, loc in (locs if locs is not None else all_locators(docs)):
        status, ref, span = resolve_locator(trees, shas, loc)
        if status != "ok":
            bad.append((name, rid, field, M.locator_string(loc), status))
            continue
        if loc.get("line") is not None and not (span[0] <= loc["line"] <= span[1]):
            bad.append((name, rid, field, M.locator_string(loc), f"line {loc['line']} outside {span}"))
    return bad


@pytest.fixture(scope="session")
def trees(built):
    out, _ = built
    return {r: Tree(out / r) for r in REPOS}


def test_tp002_every_locator_resolves_at_a_pinned_commit(manifests, built, trees):
    docs = all_docs(manifests)
    locs = all_locators(docs)
    assert len(locs) > 300, "the walk must find the manifests' locators"
    assert unresolved(docs, trees, built[1], locs) == []


def test_tp002_locator_forms_are_the_spec3_forms(manifests):
    """qname starts path + '::' (structured rows); Java carries a section 4.3 signature on methods."""
    for name in REPOS:
        for row in manifests[name]["rows"]:
            for field, loc in M.row_locators(row):
                if loc["path"].endswith(".java") and loc["qname"]:
                    member = loc["qname"].split(".")[-1]
                    is_field = re.match(r"^[A-Z0-9_]+$", member)
                    assert "(" in loc["qname"] or is_field, f"{name} {row['id']}: Java locator without a signature: {loc['qname']}"
    for name in REPOS:
        for row in manifests[name]["rows"]:
            for k, v in row.items():
                for d in (v if isinstance(v, list) else [v]):
                    if isinstance(d, dict) and "qname" in d:
                        assert d["qname"].startswith(d["path"] + "::"), (name, row["id"], d)


def test_tp002_contract_points_hash(manifests):
    """Every {kind, keyNorm} hashes (ids.contract_point_id); a key the id function refuses is a defect."""
    from ccg import ids

    n = 0
    for rid, held in M.holders(manifests).items():
        for repo, row in held.items():
            key = row.get("keyNorm", row.get("key"))
            kind = row.get("kind")
            if key and kind in {"http", "mcp_tool", "topic", "rpc", "library_api", "data_entity", "config_key", "env_var", "launch", "script_tool"}:
                cp = ids.contract_point_id(kind, key)
                assert re.fullmatch(r"[0-9a-f]{16,64}", cp) or isinstance(cp, str)
                n += 1
    assert n >= 40


def test_tp002_planted_failing_inputs(manifests, built, trees):
    """(1) svc:api/health.py::health rewritten to api/healthz.py::health; (2) a Java locator with no signature."""
    docs = all_docs(manifests)
    shas = built[1]
    bad = copy.deepcopy(docs)
    row = row_by_id(bad["fx-svc"], "LNK-01")
    row["provider"][0]["path"] = "api/healthz.py"
    row["provider"][0]["qname"] = "api/healthz.py::health"
    got = unresolved(bad, trees, shas)
    assert any(g[1] == "LNK-01" and g[4] == "nofile" for g in got)
    bad = copy.deepcopy(docs)
    row_by_id(bad["fx-svc"], "LNK-01")["provider"][0]["qname"] = "api/health.py::no_such_symbol"
    assert any(g[1] == "LNK-01" and g[4] == "nosymbol" for g in unresolved(bad, trees, shas))
    # a Java locator with the wrong signature does not resolve; a bare name on a method is a contract defect
    bad = copy.deepcopy(docs)
    row = row_by_id(bad["fx-lib"], "LNK-25")
    row["P"] = row["P"].replace("(int,String,Currency)", "(int,String)")
    row["P_line"] = 25
    got = unresolved({**bad}, trees, shas)
    # (int,String) exists at H0, so it resolves there; the declared line 25 is then outside that span or elsewhere
    row["P"] = row["P"].replace("(int,String)", "(int,Currency)")
    assert any(g[1] == "LNK-25" for g in unresolved(bad, trees, shas)), "wrong signature must not resolve"
    sig_less = "fx-lib:pricing/src/main/java/com/acme/fxlib/Pricing.java::Pricing.quote"
    assert not _has_signature(M.parse_locator(sig_less))
    assert _has_signature(M.parse_locator(sig_less + "(int,String)"))
    # a line outside the symbol's span
    bad = copy.deepcopy(docs)
    row_by_id(bad["fx-lib"], "LNK-25")["P_line"] = 1
    assert any(g[1] == "LNK-25" and "outside" in g[4] for g in unresolved(bad, trees, shas))


def _has_signature(loc):
    return bool(loc["qname"]) and loc["qname"].endswith(")")


def test_tp002_resolvers_distinguish_overloads_and_nested_members():
    java = ("package p;\npublic class A {\n  public static int f(int a, String b) { return 1; }\n"
            "  public static int f(int a, String b, java.util.List<String> c) { return 2; }\n  static final int K = 3;\n}\n")
    assert resolve_java(java, "A.f(int,String)") == (3, 3)
    assert resolve_java(java, "A.f(int,String,java.util.List<String>)") == (4, 4)
    assert resolve_java(java, "A.f(String)") is None
    assert resolve_java(java, "A.K") is not None and resolve_java(java, "A.Z") is None
    py = "class C:\n    def m(self):\n        pass\n\nX = 1\n"
    assert resolve_python(py, "C.m") == (2, 3) and resolve_python(py, "X") == (5, 5) and resolve_python(py, "C.n") is None
    assert resolve_keypath("a:\n  b.c:\n    d: 1\n", "x.yaml", "a.b.c.d") and not resolve_keypath("a: 1\n", "x.yaml", "a.b")


# ======================================================================================== TP-004 minimums census
PLAN_TEXT = PLAN.read_text()
NEGATIVES = {f"NC-{i}" for i in range(1, 8)} | {"TW-1", "TW-2", "TW-3", "OV-3", "FT-2"}
GATES = ["P1", "P2", "P3", "P4", "P5"]
TEN_KINDS = ["http", "mcp_tool", "script_tool", "launch", "rpc", "topic", "library_api", "data_entity", "config_key", "env_var"]


def merged_attr(held, attr):
    return next((r[attr] for r in held.values() if r.get(attr) is not None), None)


def census(manifests):
    """Recompute section 6's figures from the manifests alone."""
    held = M.holders(manifests)
    fig = {"rows": len(held), "prefix": {}, "gate": {}, "shape": {}, "cells": set(), "ac6": {}}
    for rid, h in held.items():
        pre = rid.split("-")[0]
        pre = "B1" if pre in ("B1F", "B1S") else "DFCD" if pre in ("DF", "CD") else pre
        fig["prefix"][pre] = fig["prefix"].get(pre, 0) + 1
        gate = merged_attr(h, "gate")
        fig["gate"][gate] = fig["gate"].get(gate, 0) + 1
        shape = merged_attr(h, "shape") or merged_attr(h, "tag")
        if shape:
            fig["shape"].setdefault(shape, []).append(gate)
        kind, status = merged_attr(h, "kind"), merged_attr(h, "status") or merged_attr(h, "st")
        base = rid.split("-")[0]
        if base == "B1S":
            kind, status = "data_entity", "confirmed"
        if base in ("LNK", "AC6", "KN", "TW", "B1S") and kind in TEN_KINDS and status in ("confirmed", "possible"):
            fig["cells"].add((kind, status))
        if base == "AC6":
            grp = rid.rsplit("-", 1)[0]
            g = fig["ac6"].setdefault(grp, {"n": 0, "shapes": set(), "confirmed": 0, "by_gate": {}})
            g["n"] += 1
            g["shapes"].add(merged_attr(h, "item") or merged_attr(h, "shape_syntax"))
            g["confirmed"] += status == "confirmed"
            bg = g["by_gate"].setdefault(gate, {"n": 0, "shapes": set()})      # the floor reads the group's OWN gate
            bg["n"] += 1
            bg["shapes"].add(merged_attr(h, "item") or merged_attr(h, "shape_syntax"))
    fig["positives"] = len([r for r in held if r not in NEGATIVES])
    fig["df"] = sorted(merged_attr(held[r], "dist") or merged_attr(held[r], "distance") for r in held if r.split("-")[0] == "DF")
    return fig


def plan_figures():
    prefix = {}
    for line in PLAN_TEXT.split("## 6.")[1].split("## 7.")[0].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for i in range(0, len(cells) - 1, 2):
            m = re.search(r"\(`([A-Z0-9]+)`(?:, `([A-Z0-9]+)`)?\)", cells[i])
            if m and cells[i + 1].isdigit():
                key = "B1" if m.group(1).startswith("B1") else "DFCD" if m.group(1) in ("DF", "CD") else m.group(1)
                prefix[key] = int(cells[i + 1])
    gates = {g: int(n) for g, n in re.findall(r"(P[1-5]) (\d+)", re.search(r"By earliest gate: ([^\n]*?)\(= 133\)", PLAN_TEXT).group(1))}
    shapes = {}
    for line in PLAN_TEXT.split("**Difficulty shapes, instances by earliest gate**")[1].split("Skew is below 3")[0].splitlines():
        m = re.match(r"\|\s*([A-Za-z][A-Za-z \-]*?)(?: \(cross-repository\))?\s*\|\s*(\*{0,2}\d\*{0,2})\s*\|\s*(\d)\s*\|\s*(\d)\s*\|", line)
        if m:
            shapes[m.group(1).strip()] = tuple(int(x.strip("*")) for x in m.groups()[1:])
    return {"prefix": prefix, "gate": gates, "shapes": shapes}


AC6_GROUP_GATE = {"AC6-URL": "P1", "AC6-DYN": "P1", "AC6-EX": "P1", "AC6-PM": "P1", "AC6-TOP": "P2", "AC6-QL": "P3"}


def census_violations(manifests):
    fig, plan = census(manifests), plan_figures()
    bad = []
    for rid, h in M.holders(manifests).items():                    # holders must agree on the figures the census reads
        for attr in ("shape", "tag", "gate"):
            vals = {r[attr] for r in h.values() if r.get(attr) is not None}
            if len(vals) > 1:
                bad.append(f"{rid}: holders disagree on {attr}: {sorted(vals)}")
    if fig["rows"] != 133:
        bad.append(f"rows {fig['rows']} != 133")
    if fig["prefix"] != plan["prefix"]:
        bad.append(f"prefix counts {fig['prefix']} != plan {plan['prefix']}")
    if fig["gate"] != plan["gate"]:
        bad.append(f"gate counts {fig['gate']} != plan {plan['gate']}")
    if fig["positives"] != 121:
        bad.append(f"positives {fig['positives']} != 121")
    for shape, (p1, p2, p3) in plan["shapes"].items():
        key = shape if shape in fig["shape"] else next((k for k in fig["shape"] if k == shape), None)
        got = fig["shape"].get(shape, [])
        have = tuple(sum(1 for g in got if GATES.index(g) <= i) for i in range(3))
        if have != (p1, p2, p3):
            bad.append(f"shape {shape}: by P1/P2/P3 {have} != plan {(p1, p2, p3)}")
        if shape == "version skew":
            if have[0] != 0 and p1 == 0:
                bad.append("skew reads non-zero before P2")
        elif have[0] < 3:
            bad.append(f"shape {shape}: floor 3 at P1 not met ({have[0]})")
        del key
    for grp, gate in AC6_GROUP_GATE.items():     # per gate, not per prefix: a row of the group at a later gate does not count
        g = fig["ac6"].get(grp)
        bg = (g or {}).get("by_gate", {}).get(gate, {"n": 0, "shapes": set()})
        if bg["n"] < 3 or len(bg["shapes"]) < 3:
            bad.append(f"{grp}: floor 3 instances / 3 shapes at {gate} not met: {(bg['n'], len(bg['shapes']))}")
        elif g["confirmed"]:
            bad.append(f"{grp}: an AC-6 row is confirmed")
    cells = {(k, s) for k in TEN_KINDS for s in ("confirmed", "possible")}
    if fig["cells"] != cells:
        bad.append(f"kind x status cells missing: {sorted(cells - fig['cells'])}")
    if len(fig["df"]) < 3 or 1 not in fig["df"] or 2 not in fig["df"]:
        bad.append(f"AC-25 data-flow distances {fig['df']}")
    return bad


def test_tp004_census_equals_section_6_and_meets_every_floor(manifests):
    plan = plan_figures()
    assert sum(plan["prefix"].values()) == 133 and len(plan["shapes"]) == 10 and sum(plan["gate"].values()) == 133
    assert census_violations(manifests) == []


def test_tp004_ac6_floor_is_read_per_gate_not_per_prefix(manifests):
    """AC6-EX-3 is gated P3, so the P1 example-config floor needs its own third row (AC6-EX-4). Planted failing inputs:
    the third P1 row deleted (the group still holds EX-1..3 by prefix), and a P1 row moved to P3."""
    assert census_violations(manifests) == []
    bad = copy.deepcopy(manifests)
    bad["fx-svc"]["rows"] = [r for r in bad["fx-svc"]["rows"] if r["id"] != "AC6-EX-4"]
    v = census_violations(bad)
    assert any(x.startswith("AC6-EX: floor 3 instances / 3 shapes at P1 not met: (2,") for x in v), v
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "AC6-EX-2")["gate"] = "P3"
    row_by_id(bad["fx-web"], "AC6-EX-2")["gate"] = "P3"
    assert any(x.startswith("AC6-EX: floor 3") for x in census_violations(bad))


def test_tp004_planted_failing_inputs(manifests):
    """One AC6-PM row deleted, and one shape tag changed: each must turn the census red."""
    bad = copy.deepcopy(manifests)
    bad["fx-svc"]["rows"] = [r for r in bad["fx-svc"]["rows"] if r["id"] != "AC6-PM-2"]
    v = census_violations(bad)
    assert any("AC6-PM" in x or "rows 132" in x for x in v), v
    bad = copy.deepcopy(manifests)
    row = next(r for r in bad["fx-svc"]["rows"] if r.get("shape") == "client wrapper")
    row["shape"] = "template literal"
    v = census_violations(bad)
    assert any("client wrapper" in x for x in v) and any("template literal" in x for x in v), v
    bad = copy.deepcopy(manifests)                                  # the same edit in only one of two holders
    next(r for r in bad["fx-web"]["rows"] if r.get("shape") == "client wrapper")["shape"] = "template literal"
    assert any("holders disagree on shape" in x for x in census_violations(bad))
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-job"], "LNK-26")["gate"] = "P1"                     # also a census failure: gate counts
    assert any("disagree on gate" in x for x in census_violations(bad))
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "LNK-26")["gate"] = "P1"
    row_by_id(bad["fx-job"], "LNK-26")["gate"] = "P1"
    assert any("gate counts" in x for x in census_violations(bad))
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-job"], "LNK-26")["reason"] = "key-match"           # (status/reason are not census figures)
    assert census_violations(bad) == []
    bad = copy.deepcopy(manifests)
    bad["fx-svc"]["rows"] = [r for r in bad["fx-svc"]["rows"] if r["id"] != "LNK-26"]
    bad["fx-job"]["rows"] = [r for r in bad["fx-job"]["rows"] if r["id"] != "LNK-26"]
    bad["fx-lib"]["rows"] = [r for r in bad["fx-lib"]["rows"] if r["id"] != "LNK-26"]
    assert any("rows 132" in x for x in census_violations(bad))


# ======================================================================================== plan <-> manifest row tie (F-3)
# Every row of plan section 5 carries its status, reason and gate (and, for KN rows, its key; for DF/CD rows, its
# distance, directness and group). Each holder's copy must equal the plan: the count-based census cannot see a
# reason swapped for another registry-valid one.

_PLAN_ST = {"c": "confirmed", "p": "possible", "confirmed": "confirmed", "possible": "possible", "none": None}


def _clean(cell):
    cell = cell.replace("**", "").replace("`", "").strip()
    return re.sub(r"\s*\([^)]*\)\s*$", "", cell).strip()        # `name-only-provider (versionSkew)` -> the code


def _canon_reason(r):
    if r is None:
        return None
    if isinstance(r, dict):
        r = f"{r['edge']} / {r['link']}"
    r = re.sub(r"\s*/\s*", " / ", r.strip())
    return None if r == "none" else r


def plan_row_table():
    """{row id: {gate, status, reason, keyNorm?, distance?, direct?, group?, planted}} parsed from plan section 5."""
    text = PLAN_TEXT.split("## 5. Planted items")[1].split("## 6.")[0]
    out, hdr = {}, None
    for ln in text.splitlines():
        if not ln.startswith("|"):
            hdr = None
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells[0] == "ID":
            hdr = [c.lower() for c in cells]
            continue
        if hdr is None or set(cells[0]) <= set("-: "):
            continue
        if len(cells) > len(hdr):                              # a `|` inside a code span (AC6-PM-2): align from the right
            cells = cells[:2] + [" | ".join(cells[2:2 + len(cells) - len(hdr) + 1])] + cells[len(cells) - (len(hdr) - 3):]
        row = {"planted": " ".join(cells)}
        for h, c in zip(hdr, cells):
            if h == "gate":
                row["gate"] = _clean(c)
            elif h == "st":
                row["status"] = _PLAN_ST.get(_clean(c).split()[0] if _clean(c) else "", "?")
            elif h.startswith("reason"):
                row["reason"] = _canon_reason(_clean(c)) if _clean(c) else None
            elif h == "expected `keynorm`":
                row["keyNorm"] = _clean(c)
            elif h.startswith("dist"):
                m = re.match(r"(\d)", _clean(c))
                row["distance"], row["direct"] = int(m.group(1)), "direct" in c and "indirect" not in c
            elif h == "group":
                row["group"] = norm_group(_clean(c))
        out[cells[0]] = row
    return out


def _key_in_plan_row(key, planted):
    """The key appears in the plan row, either verbatim or as a plan key whose `{}` the literal key fills (F-4's literal keys)."""
    if key in planted:
        return True
    for k in re.findall(r"`((?:GET|POST|PUT|DELETE) /[^`]*)`", planted):
        if "{}" in k and re.fullmatch(re.escape(k).replace(re.escape("{}"), r"[^/]+"), key):
            return True
    return False


def plan_row_violations(manifests, plan=None):
    plan = plan or plan_row_table()
    bad = []
    for rid, held in M.holders(manifests).items():
        p = plan.get(rid)
        if p is None:
            bad.append((rid, "row has no plan entry"))
            continue
        for repo, row in held.items():
            v = M.row_view(row)
            for field, have in (("gate", v["gate"]), ("status", None if v["status"] == "none" else v["status"]), ("reason", _canon_reason(v["reason"]))):
                if field in p and p[field] != "?" and (p[field] != have):
                    bad.append((rid, repo, field, p[field], have))
            if "keyNorm" in p and p["keyNorm"] != v["key"]:
                bad.append((rid, repo, "keyNorm", p["keyNorm"], v["key"]))
            if "distance" in p:
                d = row.get("distance", row.get("dist"))
                if (p["distance"], p["direct"]) != (d, row.get("direct")):
                    bad.append((rid, repo, "distance/direct", (p["distance"], p["direct"]), (d, row.get("direct"))))
                if p.get("group") != norm_group(row["group"]):
                    bad.append((rid, repo, "group", p.get("group"), row["group"]))
            if v["key"] and re.match(r"^(GET|POST|PUT|DELETE) /", v["key"]) and rid.startswith("LNK") and not _key_in_plan_row(v["key"], p["planted"]):
                bad.append((rid, repo, "key not in the plan row", v["key"]))
    return bad


def _mutations(row, plan_row):
    """(field, setter) pairs that corrupt one plan-tied field of *row* (a manifest row dict, in place)."""
    muts = []
    if "gate" in plan_row:
        muts.append(("gate", lambda r: r.__setitem__("gate", "P5" if r["gate"] != "P5" else "P1")))
    if plan_row.get("status") not in (None, "?") and ("status" in row or "st" in row):
        k = "status" if "status" in row else "st"
        muts.append(("status", lambda r, k=k: r.__setitem__(k, "possible" if r[k] == "confirmed" else "confirmed")))
    if plan_row.get("reason") is not None and "reason" in row:
        muts.append(("reason", lambda r: r.__setitem__("reason", "port-shared" if _canon_reason(r["reason"]) != "port-shared" else "name-match")))
    verb_key = row["id"].startswith("LNK") and re.match(r"^(GET|POST|PUT|DELETE) /", str(row.get("keyNorm", row.get("key", ""))))
    if "keyNorm" in plan_row or verb_key:
        k = "keyNorm" if "keyNorm" in row else "key"
        if k in row:
            muts.append(("key", lambda r, k=k: r.__setitem__(k, r[k] + "/zz")))
    if "distance" in plan_row:
        k = "distance" if "distance" in row else "dist"
        muts.append(("distance", lambda r, k=k: r.__setitem__(k, r[k] + 1)))
        muts.append(("group", lambda r: r.__setitem__("group", "zzz" + r["group"])))
    return muts


def plan_tie_probe(manifests):
    """Mutate every (holder row x plan-tied field); return (count run, survivors)."""
    plan = plan_row_table()
    n, survivors = 0, []
    for repo in REPOS:
        for i, row in enumerate(manifests[repo]["rows"]):
            p = plan.get(row["id"])
            for field, setter in _mutations(row, p):
                bad = copy.deepcopy(manifests)
                setter(bad[repo]["rows"][i])
                n += 1
                if not plan_row_violations(bad, plan):
                    survivors.append((repo, row["id"], field))
    return n, survivors


def test_f3_every_manifest_row_equals_its_plan_row_on_status_reason_gate_and_key(manifests):
    plan = plan_row_table()
    assert len(plan) == 133 and set(plan) == set(M.holders(manifests))
    assert plan_row_violations(manifests, plan) == []
    assert plan["AC6-PM-2"]["gate"] == "P1" and plan["SQC-PY"]["gate"] == "P4" and plan["KN-4"]["keyNorm"] == "GET /api/files/a%2Fb"


def test_f3_probe_every_row_x_field_mutation_turns_the_tie_red(manifests):
    n, survivors = plan_tie_probe(manifests)
    assert survivors == [], survivors[:10]
    assert n >= 400, f"the probe must enumerate rows x fields; ran {n}"
    print(f"F-3 probe: {n} mutants, 0 survivors")


# ======================================================================================== TP-013 / TP-014 / TP-016


def stratum_violations(docs):
    bad = []
    for name, doc in docs.items():
        for r in doc["rows"]:
            if r.get("stratum") != "fixture-dev":
                bad.append((name, r["id"], r.get("stratum")))
        if "held" in json.dumps(doc.get("manifest", {})).lower():
            bad.append((name, "manifest header names a held-out set"))
    return bad


def test_tp013_every_row_is_fixture_dev_and_in_no_held_out_split(manifests):
    docs = all_docs(manifests)
    assert stratum_violations(docs) == []
    assert not any("held-out" in json.dumps(r).lower() for d in docs.values() for r in d["rows"])
    ids = {r["id"] for d in docs.values() for r in d["rows"]}
    # no split/held-out file under cpg/eval may list a fixture row id or repo (F7)
    hits = []
    for p in (CPG / "eval").rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".yaml", ".yml", ".tsv", ".csv"):
            t = p.read_text(errors="ignore")
            if re.search(r"\bfx-(svc|web|lib|job)\b", t):
                hits.append(str(p.relative_to(CPG)))
    assert hits == []
    assert len(ids) == 133


def test_tp013_planted_failing_inputs(manifests):
    docs = copy.deepcopy(all_docs(manifests))
    row_by_id(docs["fx-svc"], "LNK-01")["stratum"] = "held-out"
    assert stratum_violations(docs)
    docs = copy.deepcopy(all_docs(manifests))
    del row_by_id(docs["fx-web"], "LNK-01")["stratum"]
    assert stratum_violations(docs)


MIN_GATE_BY_KIND = {"rpc": "P2", "topic": "P2", "library_api": "P2", "data_entity": "P3", "config_key": "P3", "env_var": "P3"}
PREFIX_GATE = {"SQC-PY": "P4", "SQC-JAVA": "P2", "TF": "P5", "OV": "P3", "AB-3": "P3", "AB-4": "P3"}


def gate_violations(manifests):
    """TP-014: no row asserted before its dependency's gate; fx-lib rows only from P2; P3 rows need no Java.

    Every holder's own gate is checked (a copy that disagrees with its twin is itself a violation)."""
    bad = []
    for rid, h in M.holders(manifests).items():
        base = rid.split("-")[0]
        java = any(loc["path"].endswith(".java") for r in h.values() for _, loc in M.row_locators(r))
        kind = merged_attr(h, "kind")
        for repo, row in h.items():
            gate = row["gate"]
            if repo == "fx-lib" and gate != "P2":
                bad.append((rid, "fx-lib row not at P2", gate))
            if java and gate == "P1":
                bad.append((rid, "Java row at P1", gate))
            if gate == "P3" and java:
                bad.append((rid, "P3 row needs Java", gate))
            if base == "LNK" and kind in MIN_GATE_BY_KIND and GATES.index(gate) < GATES.index(MIN_GATE_BY_KIND[kind]):
                bad.append((rid, f"{kind} link before {MIN_GATE_BY_KIND[kind]}", gate))
            want = PREFIX_GATE.get(rid) or PREFIX_GATE.get(base)
            if want and gate != want:
                bad.append((rid, f"must gate at {want}", gate))
    return bad


def test_tp014_gate_hygiene(manifests):
    assert gate_violations(manifests) == []
    lib = {r["id"] for r in manifests["fx-lib"]["rows"]}
    assert {"LNK-23", "LNK-24", "LNK-25", "LNK-31", "AC6-TOP-3", "SQC-JAVA"} <= lib


def test_tp014_planted_failing_inputs(manifests):
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-job"], "LNK-26")["gate"] = "P1"                 # LNK-26 moved to gate P1
    assert any(v[0] == "LNK-26" for v in gate_violations(bad))
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-lib"], "LNK-25")["gate"] = "P1"                 # a Java row at P1
    v = gate_violations(bad)
    assert any(x[0] == "LNK-25" and "Java" in x[1] or "fx-lib" in x[1] for x in v)
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-job"], "OV-1")["gate"] = "P1"
    assert any(x[0] == "OV-1" for x in gate_violations(bad))


def window_commits(shas_out, start, end):
    ids = set()
    for repo in REPOS:
        out = git(shas_out / repo, "log", "main", f"--since={start}T00:00:00+0000", f"--until={end}T00:00:00+0000",
                  "--format=%s")
        ids |= {ln.split(":")[0] for ln in out.splitlines()}
    return ids


def test_tp016_the_rca_window_holds_exactly_h1_h2_h3(built, manifests):
    out, _ = built
    assert window_commits(out, "2026-02-01", "2026-03-01") == {"H1", "H2", "H3"}
    dates = {c["id"]: c["date"] for r in REPOS for c in manifests[r]["history"]["commits"]}
    assert dates["H0"] < "2026-02-01" and dates["H4"] >= "2026-03-01"
    for r in REPOS:
        ds = [c["date"] for c in manifests[r]["history"]["commits"]]
        assert ds == sorted(ds) and len(set(ds)) == len(ds), f"{r}: history is linear and strictly dated"


def test_tp016_planted_failing_input_boundary_moved_to_2026_02_12(built):
    out, _ = built
    assert window_commits(out, "2026-02-12", "2026-03-01") != {"H1", "H2", "H3"}
    assert window_commits(out, "2026-02-01", "2026-02-18") != {"H1", "H2", "H3"}


# ======================================================================================== TP-005 keyNorm rows (F-13)
import test_fixture_keynorm_mutants as KNM  # noqa: E402


def kn_rows(manifests):
    out = {}
    for rid, h in M.holders(manifests).items():
        if rid.startswith("KN-"):
            lit = merged_attr(h, "literal")
            out[rid] = (lit, merged_attr(h, "keyNorm") or merged_attr(h, "key"))
    return out


def test_tp005_the_manifest_kn_rows_are_the_rows_the_mutant_harness_uses(manifests):
    rows = kn_rows(manifests)
    assert set(rows) == set(KNM.KN) == {"KN-1", "KN-2", "KN-3", "KN-4", "KN-5"}
    for rid, (lit, key) in rows.items():
        assert (lit, key) == KNM.KN[rid], rid


def test_tp005_correct_order_normalizes_every_manifest_literal_to_its_key(manifests):
    for rid, (lit, key) in kn_rows(manifests).items():
        assert KNM.correct(lit) == key, rid


def test_tp005_the_two_planted_mutants_fail_exactly_the_planned_rows(manifests):
    rows = kn_rows(manifests)
    fails = lambda fn: {rid for rid, (lit, key) in rows.items() if fn(lit) != key}   # noqa: E731
    assert fails(KNM.resolve_before_decode) == {"KN-1", "KN-2"}
    assert fails(KNM.mask_before_decode) == {"KN-5"}


def test_tp005_a_manifest_key_edited_to_the_wrong_order_is_caught(manifests):
    bad = copy.deepcopy(manifests)
    row = next(r for r in bad["fx-svc"]["rows"] if r["id"] == "KN-4")
    row["keyNorm"] = "GET /api/files/a/b"
    lit, key = kn_rows(bad)["KN-4"]
    assert KNM.correct(lit) != key


# ======================================================================================== AC-25 column (TP-008)
# S33's answer schema is unwritten (OI-1): the four properties are asserted on a neutral answer shape,
# list[group{repo, component, distance, items[{name, locator, status, direct, distance}]}], built by the
# reference builder below. Each assertion id is counted once per planted item; the total must be 60 (plan s3).


def _loc(v):
    if isinstance(v, str):
        return M.parse_locator(v)
    q = v.get("qname")
    q = q[len(v["path"]) + 2:] if q and q.startswith(v["path"] + "::") else q
    return {"repo": v["repoId"], "path": v["path"], "qname": q}


def component_of(loc):
    parts = loc["path"].split("/")
    return parts[2] if loc["repo"] == "fx-web" and parts[:2] == ["app", "src"] else parts[0]


def ac25_items(manifests):
    """Planted AC-25 items from the manifests: forward {DF,CD}-n and reverse RV-n with their dependencies."""
    held = M.holders(manifests)
    fwd, rev = {}, {}
    for rid, h in held.items():
        base = rid.split("-")[0]
        row = next(iter(h.values()))
        if base in ("DF", "CD"):
            loc = _loc(row["locator"])
            fwd[rid] = {"id": rid, "locator": loc, "name": loc["qname"].split("::")[-1], "distance": row.get("distance", row.get("dist")),
                        "direct": row["direct"], "status": row.get("status", row.get("st")), "group": row["group"],
                        "changed": row.get("changed") and _loc(row["changed"])}
        elif base == "RV":
            deps = []
            for r in h.values():
                for d in r.get("dependencies", r.get("deps", [])):
                    loc = _loc(d.get("locator", d.get("loc")))
                    deps.append({"locator": loc, "name": loc["qname"].split("::")[-1], "distance": d.get("distance", d.get("dist")),
                                 "status": d.get("status", d.get("st")), "group": d["group"]})
                subject = _loc(r["subject"])
            seen = []
            for d in deps:                                             # RV-3 is held by two manifests: dedupe
                if (d["locator"]["path"], d["name"]) not in [(x["locator"]["path"], x["name"]) for x in seen]:
                    seen.append(d)
            rev[rid] = {"id": rid, "subject": subject, "deps": seen}
    for pre in ("DF", "CD"):                  # CF-DF / CF-CD (plan s5.5): fx-web's DF-3 does not repeat the changed function
        cf = next(v["changed"] for k, v in fwd.items() if k.startswith(pre) and v["changed"])
        for k, v in fwd.items():
            if k.startswith(pre) and not v["changed"]:
                v["changed"] = cf
    return fwd, rev


def norm_group(g):
    return tuple(x.strip() for x in g.replace(" / ", "/").split("/"))


def derived_group(loc, distance):
    return (loc["repo"], component_of(loc), str(distance))


def answer_item(it):
    loc = it["locator"]
    return {"name": it["name"], "locator": loc, "status": it["status"], "distance": it["distance"],
            "direct": it.get("direct", it["distance"] == 1)}


def build_answer(items, *, group_key, item_key, flat=False, distance_shift=0, direct_all=False, distance_only_shift=0):
    """items: list of {locator, name, status, distance}. Returns the ordered list of groups."""
    shifted = []
    for it in items:
        it2 = dict(it)
        it2["distance"] = it["distance"] + distance_shift + distance_only_shift
        # distance_only_shift moves the distance but keeps `direct` true to the real distance (isolates the two conjuncts)
        it2["direct"] = True if direct_all else (it["distance"] == 1 if distance_only_shift else it2["distance"] == 1)
        shifted.append(it2)
    if flat:
        return [{"repo": "*", "component": "*", "distance": 0, "items": [answer_item(i) for i in sorted(shifted, key=item_key)]}]
    groups = {}
    for it in shifted:
        groups.setdefault((it["locator"]["repo"], component_of(it["locator"]), it["distance"]), []).append(it)
    return [{"repo": k[0], "component": k[1], "distance": k[2], "items": [answer_item(i) for i in sorted(v, key=item_key)]}
            for k, v in sorted(groups.items(), key=group_key)]


def _bytes(s):
    return s.encode()


CORRECT = dict(group_key=lambda kv: (_bytes(kv[0][0]), _bytes(kv[0][1]), kv[0][2]),
               item_key=lambda i: (i["status"] != "confirmed", _bytes(i["name"])))
MUTANTS = {
    "a: sort by (repository, component, distance, name) with no status key":
        dict(group_key=CORRECT["group_key"], item_key=lambda i: _bytes(i["name"])),
    "b: sort by distance first": dict(group_key=lambda kv: (kv[0][2], _bytes(kv[0][0]), _bytes(kv[0][1])), item_key=CORRECT["item_key"]),
    "c: distance off by one": dict(**CORRECT, distance_shift=1),
    "d: groups flattened": dict(**CORRECT, flat=True),
    "e: drops the last dependency of every reverse answer": dict(**CORRECT, drop_last=True),
    "f: right distances, direct=True on every item": dict(**CORRECT, direct_all=True),
    "g: wrong distance, direct kept true to the real distance": dict(**CORRECT, distance_only_shift=1),
}


def check_answer(answer, planted, kind):
    """Assert A25-1..3 (per planted item) against *answer*; returns [(assertion id, item id, ok)]."""
    out = []
    order = [(g["repo"], g["component"], g["distance"]) for g in answer]
    for pid, it in planted:
        gi = next((i for i, g in enumerate(answer) for x in g["items"] if x["locator"] == it["locator"] and x["name"] == it["name"]), None)
        found = None if gi is None else next(x for x in answer[gi]["items"] if x["locator"] == it["locator"] and x["name"] == it["name"])
        if found is None and kind == "reverse":
            continue                  # a dependency the answer omits is A25-4's failure alone, not A25-1..3's
        # A25-1: direct iff distance 1, else indirect, with the expected distance
        out.append(("A25-1", pid, found is not None and found["distance"] == it["distance"] and found["direct"] == (it["distance"] == 1)))
        # A25-2: the item sits in its named group; groups ordered by repository, component (bytes), distance
        want = derived_group(it["locator"], it["distance"])
        in_group = gi is not None and (answer[gi]["repo"], answer[gi]["component"], str(answer[gi]["distance"])) == want
        sorted_ok = order == sorted(order, key=lambda g: (_bytes(g[0]), _bytes(g[1]), g[2]))
        out.append(("A25-2", pid, in_group and sorted_ok))
        # A25-3: inside its group every confirmed item precedes every possible one
        ok3 = False
        if gi is not None:
            sts = [x["status"] for x in answer[gi]["items"]]
            ok3 = sts == sorted(sts, key=lambda s: s != "confirmed")
        out.append(("A25-3", pid, ok3))
    return out


def run_ac25(manifests, builder):
    fwd, rev = ac25_items(manifests)
    results = []
    for chg in sorted({str(v["changed"]["qname"]) for v in fwd.values()}):
        planted = [(i, v) for i, v in fwd.items() if v["changed"]["qname"] == chg]
        results += check_answer(build_answer([v for _, v in planted], **{k: x for k, x in builder.items() if k != "drop_last"}), planted, "forward")
    for rid, rv in rev.items():
        planted = [(f"{rid}/{d['name']}", d) for d in rv["deps"]]
        bk = {k: v for k, v in builder.items() if k != "drop_last"}
        ans = build_answer([d for _, d in (planted[:-1] if builder.get("drop_last") else planted)], **bk)
        results += check_answer(ans, planted, "reverse")
        listed = {(x["locator"]["path"], x["name"]) for g in ans for x in g["items"]}
        results.append(("A25-4", rid, listed == {(d["locator"]["path"], d["name"]) for d in rv["deps"]}))
    return results


def test_ac25_the_assertion_column_is_60_distinct_assertions(manifests):
    fwd, rev = ac25_items(manifests)
    assert len(fwd) == 8 and len(rev) == 3
    assert sum(len(r["deps"]) for r in rev.values()) == 11
    res = run_ac25(manifests, CORRECT)
    assert len(res) == 8 * 3 + 11 * 3 + 3 == 60
    assert {a for a, _, _ in res} == {"A25-1", "A25-2", "A25-3", "A25-4"}
    failed = [r for r in res if not r[2]]
    assert failed == []


def test_ac25_manifest_rows_are_self_consistent(manifests):
    """direct iff distance 1; the stated group equals (repository, component, distance) derived from the locator."""
    fwd, rev = ac25_items(manifests)
    for it in fwd.values():
        assert it["direct"] == (it["distance"] == 1), it["id"]
        assert norm_group(it["group"]) == derived_group(it["locator"], it["distance"]), it["id"]
    for rv in rev.values():
        for d in rv["deps"]:
            assert norm_group(d["group"]) == derived_group(d["locator"], d["distance"]), (rv["id"], d["name"])


def test_ac25_expected_answer_order_matches_the_plan(manifests):
    """The plan's 'Expected answer order' lines and RV tables, parsed from the plan text, equal the manifests'."""
    fwd, rev = ac25_items(manifests)
    text = PLAN_TEXT
    for chg, line in re.findall(r"- forward `CF-(DF|CD)`: ([^\n]*)", text):
        planned = [(g[0], g[1], g[2], [x.strip() for x in g[3].split(",")])
                   for g in re.findall(r"\((fx-[a-z]+), (\w+), (\d)\) \[([^\]]+)\]", line)]
        ids = [i for g in planned for i in g[3]]
        assert ids and all(i.startswith(chg) for i in ids)
        answer = build_answer([fwd[i] for i in ids], **CORRECT)
        got = [(g["repo"], g["component"], str(g["distance"]), [i for i in ids if fwd[i]["name"] in {x["name"] for x in g["items"]}]) for g in answer]
        assert [(a, b, c) for a, b, c, _ in got] == [(a, b, c) for a, b, c, _ in planned]
        assert [sorted(d) for *_, d in got] == [sorted(d) for *_, d in planned]
        flat = [x["name"] for g in answer for x in g["items"]]
        assert flat == [fwd[i]["name"] for i in ids], f"within-group order of the {chg} answer"
    # RV rows: planted dependency set per RV row equals the plan's table
    for rid, rv in rev.items():
        row = re.search(rf"\| {rid} \|[^\n]*", text).group(0)
        plan_deps = re.findall(r"`(?:svc|web):([^`]+)::([^`]+)` \((c|p)", row.split("|")[3])       # the dependencies cell
        have = [(d["locator"]["path"], d["name"], d["status"][0]) for d in rv["deps"]]
        assert sorted(have) == sorted(plan_deps), rid
        answer = build_answer(rv["deps"], **CORRECT)
        assert [(g["repo"], g["component"], str(g["distance"])) for g in answer] == [
            norm_group(g) for g in dict.fromkeys(d["group"] for d in rv["deps"])], f"{rid}: the manifest lists dependencies in answer order"


def test_ac25_the_a25_3_pairs_really_sort_possible_first_under_name_path_and_line_order(manifests, built, trees):
    """A25-3 is only non-vacuous if the possible item sorts BEFORE the confirmed one under every key (plan s5.5)."""
    pairs = [("accept_quote", "create_order"), ("publish_alert", "publish_created"), ("find_hook", "is_member"), ("find_hook", "tier_of")]
    fwd, rev = ac25_items(manifests)
    every = [*fwd.values(), *[d for r in rev.values() for d in r["deps"]]]
    by_name = {}
    for it in every:
        by_name.setdefault(it["name"], it)
    shas = built[1]
    for p, c in pairs:
        pl, cl = by_name[p]["locator"], by_name[c]["locator"]
        assert by_name[p]["status"] == "possible" and by_name[c]["status"] == "confirmed", (p, c)
        assert p.encode() < c.encode(), f"name order: {p} < {c}"
        if pl["path"] == cl["path"]:
            lp = resolve_locator(trees, shas, pl)[2][0]
            lc = resolve_locator(trees, shas, cl)[2][0]
            assert lp < lc, f"line order: {p} (line {lp}) defined before {c} (line {lc})"
        else:
            assert pl["path"] < cl["path"], f"path order: {pl['path']} < {cl['path']}"


@pytest.mark.parametrize("name", list(MUTANTS))
def test_ac25_planted_failing_answer_builders_turn_the_column_red(manifests, name):
    res = run_ac25(manifests, MUTANTS[name])
    failed = {(a, i) for a, i, ok in res if not ok}
    kinds = {a for a, _ in failed}
    fwd, rev = ac25_items(manifests)
    every = {**{i: v for i, v in fwd.items()},
             **{f"{rid}/{d['name']}": d for rid, r in rev.items() for d in r["deps"]}}
    if name.startswith("a:"):
        assert kinds == {"A25-3"}, failed                                         # only A25-3, in exactly the four pair groups
        assert {i for _, i in failed} == {"DF-2", "DF-4", "CD-3", "CD-4", "RV-1/is_member", "RV-1/find_hook",
                                          "RV-2/tier_of", "RV-2/find_hook"}
    elif name.startswith("b:"):
        assert "A25-2" in kinds, failed                                          # api/2 before core/1
    elif name.startswith("c:"):
        assert "A25-1" in kinds, failed
    elif name.startswith("d:"):
        assert "A25-2" in kinds, failed
    elif name.startswith("e:"):
        assert failed == {("A25-4", "RV-1"), ("A25-4", "RV-2"), ("A25-4", "RV-3")}, failed
    elif name.startswith("f:"):
        assert kinds == {"A25-1"} and {i for _, i in failed} == {k for k, v in every.items() if v["distance"] != 1}, failed
        assert {"DF-2", "DF-3", "DF-4", "CD-3", "CD-4"} <= {i for _, i in failed}
    else:                                                                         # g: every distance is wrong, directness is not
        assert kinds <= {"A25-1", "A25-2"}, failed                                # a wrong distance also puts the item in the wrong group
        assert {i for a, i in failed if a == "A25-1"} == set(every), failed       # ... but A25-1 must itself fail on every item
        assert len(every) == 19


def test_ac25_a_manifest_with_the_wrong_group_or_direct_flag_is_caught(manifests):
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "DF-2")["group"] = "fx-svc / core / 2"
    fwd, _ = ac25_items(bad)
    assert norm_group(fwd["DF-2"]["group"]) != derived_group(fwd["DF-2"]["locator"], fwd["DF-2"]["distance"])
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "DF-1")["direct"] = False
    assert ac25_items(bad)[0]["DF-1"]["direct"] != (ac25_items(bad)[0]["DF-1"]["distance"] == 1)


# ======================================================================================== TP-018 planter completeness
# F-8: the check keys on (literal, enclosing locator). A literal that is allowed *somewhere* (a global literal set)
# would let a second `fetch("/api/health")` in a new function pass; here the pair must be explained by one row that
# names that enclosing code element. Limit (stated in the report): a repeat inside an already-explained element is
# the same pair.

ROUTE_PY = re.compile(r"^(https?://[^/\s]+)?/[A-Za-z_{][\w{}.%:\-/]*$")
ROUTE_JS = re.compile(r"""(["'`])((?:https?://[\w.:\-${}]+)?/[A-Za-z0-9_{}$.%:\-/]*)\1""")
TOOL_RE = re.compile(r"""(?:callTool|call_tool|mcp-call)\(?\s*(?:\{\s*name:\s*)?["']([A-Za-z_][\w.\-]*)["']""")
TOPIC_RE = re.compile(r"""(?:\.send|subscribe|\.publish|\.produce)\(\s*["']([A-Za-z_][\w.\-]*)["']""")
SQL_START = re.compile(r"(?i)^\s*(select|insert|update|delete|with|create)\b")
TABLE_RE = re.compile(r"(?i)\b(?:from|into|update|join)\s+([a-z_][a-z0-9_]*)\b")
ENV_RE = re.compile(r"""(?:environ(?:\.get)?\[?\(?|getenv\(|process\.env\.|import\.meta\.env\.)\s*["']?([A-Z][A-Z0-9_]+)""")


def py_enclosing(tree):
    """[(start, end, qname)] of every function/class/module-level assignment in a Python module."""
    spans = []

    def visit(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([n.lineno, *[d.lineno for d in n.decorator_list]])
                spans.append((start, n.end_lineno, prefix + n.name, False))
                visit(n.body, prefix + n.name + ".")
            elif isinstance(n, (ast.Assign, ast.AnnAssign)) and not prefix:
                t = n.targets[0] if isinstance(n, ast.Assign) else n.target
                if isinstance(t, ast.Name):
                    spans.append((n.lineno, n.end_lineno, t.id, True))
            elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                visit(n.body, prefix)

    visit(tree.body, "")
    return spans


def ts_enclosing(text):
    lines = text.splitlines()
    decl = re.compile(r"^(\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?\s+(\w+)|(?:const|let|var)\s+(\w+)|class\s+(\w+))")
    spans = []
    for i, ln in enumerate(lines):
        m = decl.match(ln)
        if m:
            name = next(g for g in m.groups()[1:] if g)
            is_const = not m.group(1) and bool(re.match(r"\s*(?:export\s+)?(?:const|let|var)\b", ln)) and "=>" not in ln and "function" not in ln
            spans.append((i + 1, _brace_end(lines, i + 1), name, is_const))
    return spans


def planter_sites(repo_dir: Path):
    """[(path, line, kind, literal, [enclosing candidates], at_module_level)] for every route/tool/topic/table/env literal.

    A module-level constant (`API_PREFIX = "/api"`) is explained by its file's rows; a function body is not."""
    sites = []
    for f in sorted(repo_dir.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts or ".git" in f.parts or f.suffix not in (".py", ".ts", ".tsx"):
            continue
        rel = str(f.relative_to(repo_dir))
        text = f.read_text()
        try:
            tree = ast.parse(text) if f.suffix == ".py" else None
        except SyntaxError:
            tree = None
        spans = py_enclosing(tree) if tree else ts_enclosing(text) if f.suffix != ".py" else []
        found = []
        if tree:
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    if ROUTE_PY.match(n.value):
                        found.append((n.lineno, "route", n.value))
                    if SQL_START.match(n.value):
                        found += [(n.lineno, "table", m) for m in TABLE_RE.findall(n.value)]
        else:
            found += [(text.count("\n", 0, m.start()) + 1, "route", m.group(2)) for m in ROUTE_JS.finditer(text) if len(m.group(2)) > 1]
        for rx, kind in ((TOOL_RE, "tool"), (TOPIC_RE, "topic"), (ENV_RE, "env")):
            found += [(text.count("\n", 0, m.start()) + 1, kind, m.group(1)) for m in rx.finditer(text)]
        for line, kind, lit in found:
            inside = [(q, c) for s, e, q, c in spans if s <= line <= e]
            sites.append((rel, line, kind, lit, [q for q, _ in inside] or ["<module>"], not inside or inside[-1][1]))
    return sites


def _static_core(lit):
    lit = re.sub(r"^https?://[^/]+", "", lit)
    lit = re.sub(r"\$\{[^}]*\}|\{[^}]*\}", "{}", lit).split("?")[0]
    return lit.rstrip("/") or "/"


def row_vocabulary(row):
    """Text a row uses to explain its literals: keys, true targets, `literal`, and its descriptive fields."""
    parts = []
    for k, v in row.items():
        if k in ("id", "serves", "stratum", "gate"):
            continue
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            parts.append(str(v))
        elif isinstance(v, (list, dict)):
            parts.append(json.dumps(v, default=str))
    return re.sub(r"\$\{[^}]*\}|\{[^}]*\}", "{}", " ".join(parts))


def unexplained_sites(repo, repo_dir, manifests):
    """Planter sites with no row that names the enclosing element AND carries the literal."""
    rows = []                       # (locators, vocabulary) for every row of every manifest
    for held in M.holders(manifests).values():
        for r in held.values():
            rows.append(([loc for _, loc in M.row_locators(r)], row_vocabulary(r)))
    bad = []
    for path, line, kind, lit, cands, const in planter_sites(repo_dir):
        core = _static_core(lit) if kind == "route" else lit
        if kind == "route" and len(core.strip("/{}")) < 2:
            continue
        ok = False
        for locs, vocab in rows:
            if core not in vocab and (kind != "route" or core.lstrip("/") not in vocab):
                continue
            for loc in locs:
                if loc["repo"] != repo or loc["path"] != path:
                    continue
                q = (loc.get("qname") or "").split("(")[0].split("::")[-1]
                if not q or const or q in cands or q.split(".")[0] in cands:   # aux/const locators and module-level constants: file-level
                    ok = True
                    break
            if ok:
                break
        if not ok:
            bad.append((repo, path, line, kind, lit, cands))
    return bad


# Findings the run reports instead of hiding: a literal with no row naming its enclosing element. Keyed on
# (repo, path, literal, enclosing element). The list must match the sweep exactly (a stale entry fails too).
PLANTER_FINDINGS: dict = {}      # S39-D1 (the `/api` mount prefix) was closed by `aux: fx-svc:api/app.py` on LNK-04


def finding_key(b):
    repo, path, line, kind, lit, cands = b
    return (repo, path, lit, cands[0])


def planter_findings(manifests, root):
    return [b for repo in REPOS for b in unexplained_sites(repo, root / repo, manifests)]


def test_tp018_every_planter_site_is_explained_by_a_row_naming_its_enclosing_element(manifests, built):
    out, _ = built
    got = {finding_key(b) for b in planter_findings(manifests, out)}
    assert got == set(PLANTER_FINDINGS), got ^ set(PLANTER_FINDINGS)
    assert sum(len(planter_sites(out / r)) for r in REPOS) >= 50, "the sweep must find the planted literals"


def _copy_tree(out, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(out, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    return dst


def test_tp018_planted_failing_inputs_a_second_health_call_in_a_new_function(manifests, built, tmp_path):
    """The plan's input, `fetch("/api/health")` added to a fixture file, in the three trees. A global-literal
    allowlist (the builders' checkers, review F-8) passes all of them because "/api/health" is planted elsewhere."""
    out, _ = built
    base = {finding_key(b) for b in planter_findings(manifests, out)}
    cases = {
        "fx-web": ("app/src/ui/Health.tsx", '\nexport function extraProbe() { return fetch("/api/health"); }\n'),
        "fx-job": ("job/run_report.py", '\n\ndef extra_probe():\n    return requests.get(BASE + "/api/health")\n'),
        "fx-svc": ("clients/jobs.py", '\n\ndef extra_probe():\n    return requests.get("/api/health")\n'),
    }
    for repo, (rel, addition) in cases.items():
        root = _copy_tree(out, tmp_path / repo)
        f = root / repo / rel
        f.write_text(f.read_text() + addition)
        got = {finding_key(b) for b in planter_findings(manifests, root)}
        new = got - base
        assert new and all(k[0] == repo and k[1] == rel for k in new), (repo, new)
        assert any(k[2] == "/api/health" and k[3] in ("extra_probe", "extraProbe") for k in new), (repo, new)
    # a second tool call and a new SQL table in new elements
    root = _copy_tree(out, tmp_path / "t")
    f = root / "fx-web" / "app/src/mcp/client.ts"
    f.write_text(f.read_text() + '\nexport function extraTool(c: any) { return c.callTool({ name: "get_item" }); }\n')
    f = root / "fx-job" / "job/run_report.py"
    f.write_text(f.read_text() + '\n\ndef extra_table(cur):\n    return cur.execute("SELECT 1 FROM audit_log")\n')
    new = {finding_key(b) for b in planter_findings(manifests, root)} - base
    assert ("fx-web", "app/src/mcp/client.ts", "get_item", "extraTool") in new
    assert ("fx-job", "job/run_report.py", "audit_log", "extra_table") in new


def test_tp018_a_literal_moved_into_a_new_element_is_a_finding_but_the_same_element_is_not(manifests, built, tmp_path):
    """The pair, not the literal: the same literal in the element a row names stays explained (documented limit)."""
    out, _ = built
    base = {finding_key(b) for b in planter_findings(manifests, out)}
    root = _copy_tree(out, tmp_path)
    f = root / "fx-web" / "app/src/ui/Health.tsx"
    text = f.read_text()
    assert "fetch(" in text
    f.write_text(text.replace('fetch("/api/health")', 'fetch("/api/health").then(() => fetch("/api/health"))'))
    assert {finding_key(b) for b in planter_findings(manifests, root)} == base


def test_tp018_the_builders_global_literal_check_would_have_passed_the_planted_input(manifests, built):
    """Why F-8 matters, demonstrated: the allowlist form accepts "/api/health" anywhere once any row has it."""
    all_literals = {_static_core(lit) for repo in REPOS for *_, kind, lit, _c, _k in
                    [(s[0], s[1], s[2], s[3], s[4], s[5]) for s in planter_sites(built[0] / repo)] if kind == "route"}
    assert "/api/health" in all_literals


# ======================================================================================== AB-1, LNK-42, LNK-43 (builder gate)
# The S39 builder gate for the cross-unit build-state rows: svc-legacy is unbuildable from H2 to HEAD with a previous
# refresh at H1; svc-tools was never built; the links into both are `possible / target-not-mapped`.


def presence_by_commit(manifest, rel, base_has):
    """Does *rel* exist after each commit? Computed from the manifest's forward ops alone (F-6)."""
    present, out = base_has, {}
    for c in manifest["history"]["commits"]:
        for op in c["ops"]:
            if op["path"] == rel and op["op"] == "add_file":
                present = True
            elif op["path"] == rel and op["op"] == "delete_file":
                present = False
        out[c["id"]] = present
    return out


def unit_gate_violations(docs, svc_dir: Path):
    bad = []
    svc = docs["fx-svc"]
    cross = {r["id"]: r for r in docs["cross"]["rows"]}
    units = {u["unitId"]: u for u in svc["units"]}
    for uid, root in (("svc-legacy", "legacy"), ("svc-tools", "tools")):
        if uid not in units or units[uid]["rootPath"] != root + "/":
            bad.append(f"{uid}: not a unit rooted at {root}/")
        if not (svc_dir / root / "pyproject.toml").is_file():
            bad.append(f"{uid}: {root}/pyproject.toml missing (a unit needs its own project file)")
        if not (svc_dir / root / "app.py").is_file():
            bad.append(f"{uid}: provider file {root}/app.py missing")
    app_routes = {"legacy": "/legacy/fee", "tools": "/tools/run"}
    for root, route in app_routes.items():
        f = svc_dir / root / "app.py"
        if f.is_file() and route not in f.read_text():
            bad.append(f"{root}/app.py does not provide {route}")
    orders = (svc_dir / "api" / "orders.py").read_text() if (svc_dir / "api" / "orders.py").is_file() else ""
    for fn, route in (("legacy_fee", "/legacy/fee"), ("run_tool", "/tools/run")):
        if f"def {fn}" not in orders or route not in orders:
            bad.append(f"api/orders.py::{fn} does not fetch {route}")
    # build state: broken.py is added at H2 and never removed; it is a syntax error; H1 is an allow-empty build point
    pres = presence_by_commit(svc, "legacy/broken.py", False)
    if pres != {"H0": False, "H1": False, "H2": True, "H3": True, "H4": True}:
        bad.append(f"legacy/broken.py presence by commit is {pres}")
    h1 = next(c for c in svc["history"]["commits"] if c["id"] == "H1")
    if h1["ops"] or not h1.get("allowEmpty"):
        bad.append("H1 must be an allow-empty build point (the 'previous refresh')")
    bf = svc.get("buildFailure", [])
    if [(b["unitId"], b["from"], b["until"]) for b in bf] != [("svc-legacy", "H2", "HEAD")]:
        bad.append(f"buildFailure is {bf}")
    broken = next((o["content"] for c in svc["history"]["commits"] for o in c["ops"] if o["path"] == "legacy/broken.py"), "")
    try:
        ast.parse(broken)
        bad.append("legacy/broken.py parses: it is not a failing build")
    except SyntaxError:
        pass
    if "never built" not in str(units.get("svc-tools", {}).get("steadyState", "")):
        bad.append("svc-tools must be 'never built' in the steady state")
    if "unbuildable" not in str(units.get("svc-legacy", {}).get("steadyState", "")):
        bad.append("svc-legacy must be 'unbuildable' in the steady state")
    # rows
    for rid, key in (("LNK-42", "GET /legacy/fee"), ("LNK-43", "GET /tools/run")):
        r = cross.get(rid)
        if not r or (r.get("status"), r.get("reason"), r.get("gate"), r.get("keyNorm")) != ("possible", "target-not-mapped", "P1", key):
            bad.append(f"{rid} row is {r}")
    ab = cross.get("AB-1")
    if not ab or ab.get("scenario") != "SC-FAIL-UNIT" or ab.get("unit", {}).get("rootPath") != "legacy/" or ab.get("gate") != "P1":
        bad.append(f"AB-1 row is {ab}")
    return bad


def test_ab1_lnk42_lnk43_the_unit_build_state_gate(manifests, built):
    docs = all_docs(manifests)
    assert unit_gate_violations(docs, built[0] / "fx-svc") == []
    out = built[0] / "fx-svc"
    assert git(out, "cat-file", "-e", "H1:legacy/app.py") == "" and subprocess.run(
        ["git", "-C", str(out), "cat-file", "-e", "H1:legacy/broken.py"], capture_output=True).returncode != 0
    for tag in ("H2", "H3", "H4"):
        assert git(out, "cat-file", "-t", f"{tag}:legacy/broken.py") == "blob", tag
    assert git(out, "diff", "--stat", "H0", "H1") == "", "H1 is a no-change build point"


def test_ab1_lnk42_lnk43_planted_failing_inputs(manifests, built, tmp_path):
    docs = all_docs(manifests)
    out = built[0] / "fx-svc"
    # (1) the failure reverted at H3 (the fx-job shape applied to legacy): presence is no longer H2..HEAD
    bad = copy.deepcopy(docs)
    bad["fx-svc"]["history"]["commits"][3]["ops"].append({"op": "delete_file", "path": "legacy/broken.py"})
    assert any("presence" in v for v in unit_gate_violations(bad, out))
    # (2) a broken.py that parses is not a failure
    bad = copy.deepcopy(docs)
    for c in bad["fx-svc"]["history"]["commits"]:
        for o in c["ops"]:
            if o["path"] == "legacy/broken.py":
                o["content"] = "def fine():\n    pass\n"
    assert any("parses" in v for v in unit_gate_violations(bad, out))
    # (3) LNK-42 asserted confirmed, LNK-43 reason changed, AB-1 scenario dropped
    for rid, field, val in (("LNK-42", "status", "confirmed"), ("LNK-43", "reason", "key-match"), ("AB-1", "scenario", "SC-STEADY")):
        bad = copy.deepcopy(docs)
        next(r for r in bad["cross"]["rows"] if r["id"] == rid)[field] = val
        assert any(rid in v for v in unit_gate_violations(bad, out)), rid
    # (4) svc-tools given a build failure; (5) the unit's project file removed from the tree
    bad = copy.deepcopy(docs)
    bad["fx-svc"]["buildFailure"].append({"unitId": "svc-tools", "from": "H2", "until": "HEAD"})
    assert any("buildFailure" in v for v in unit_gate_violations(bad, out))
    root = _copy_tree(built[0], tmp_path)
    (root / "fx-svc" / "legacy" / "pyproject.toml").unlink()
    assert any("pyproject" in v for v in unit_gate_violations(docs, root / "fx-svc"))
    orders = root / "fx-svc" / "api" / "orders.py"
    orders.write_text(orders.read_text().replace("/tools/run", "/tools/other"))
    assert any("run_tool" in v for v in unit_gate_violations(docs, root / "fx-svc"))


# ======================================================================================== branches (F-2)
# Plan 5.8 / 5.10: five un-merged change sets, each cut from HEAD (= H4). The expectations below are the plan's text
# turned into checks on the materialized branch (what changed, in which element, how), so an edit that keeps the paths
# but changes the substance fails; the pinned SHAs (F-7) then fix the exact bytes.

BRANCH_PLAN = {
    # repo: {branch: (changed files, {file: changed top-level functions}, rule)}
    "fx-job": {"ov-job": ({"job/run_report.py", "job/trigger_server.py"},
                          {"job/run_report.py": {"write_report"}, "job/trigger_server.py": {"status"}}, "ov-job")},
    "fx-svc": {"ov-svc": ({"db/schema.sql"}, {}, "ov-svc"),
               "tf-mid": ({"api/ops.py"}, {"api/ops.py": {"normalize"}}, "tf-mid"),
               "tf-sink": ({"api/ops.py"}, {"api/ops.py": {"exec_cmd"}}, "tf-sink"),
               "tf-src": ({"api/ops.py"}, {"api/ops.py": {"run_cmd"}}, "tf-src")},
}


def _funcs(text):
    tree = ast.parse(text)
    return {n.name: (ast.dump(n), n.lineno, n.end_lineno) for n in tree.body if isinstance(n, ast.FunctionDef)}


def _insert_columns(text):
    m = re.search(r"INSERT INTO order_report \(([^)]*)\)", text)
    return [c.strip() for c in m.group(1).split(",")] if m else None


def _return_keys(text, fn):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == fn)
    ret = next(n for n in ast.walk(node) if isinstance(n, ast.Return))
    return {k.value for k in ret.value.keys} if isinstance(ret.value, ast.Dict) else None


def _table_columns(text, table="orders"):
    m = re.search(rf"(?is)create\s+table\s+{table}\s*\((.*?)\n\)\s*;", text)
    return [ln.strip().split()[0] for ln in m.group(1).splitlines() if ln.strip() and not ln.strip().upper().startswith(("PRIMARY", "UNIQUE", "FOREIGN"))] if m else None


def branch_violations(manifests, repo, repo_dir):
    """Plan-derived checks of every branch of *repo* in the materialized repository *repo_dir*."""
    bad = []
    for b in manifests[repo]["history"].get("branches", []):
        name = b["name"]
        if b["base"] != "H4":
            bad.append((repo, name, "plan: off HEAD (H4), manifest base is " + b["base"]))
        files, funcs, _ = BRANCH_PLAN[repo][name]
        parent = git(repo_dir, "rev-parse", f"{name}^")
        if parent != git(repo_dir, "rev-parse", "H4"):
            bad.append((repo, name, "branch is not cut from H4"))
        changed = set(git(repo_dir, "diff", "--name-only", "H4", name).splitlines())
        if changed != files:
            bad.append((repo, name, f"changed files {sorted(changed)} != plan {sorted(files)}"))
            continue
        before = {f: git(repo_dir, "show", f"H4:{f}") for f in files}
        after = {f: git(repo_dir, "show", f"{name}:{f}") for f in files}
        for f in files:
            if f.endswith(".py"):
                try:
                    fb, fa = _funcs(before[f]), _funcs(after[f])
                except SyntaxError as exc:
                    bad.append((repo, name, f"{f} does not parse: {exc.msg}"))
                    continue
                diff = {n for n in set(fb) | set(fa) if fb.get(n, (None,))[0] != fa.get(n, (None,))[0]}
                if diff != funcs[f]:
                    bad.append((repo, name, f"{f}: changed functions {sorted(diff)} != plan {sorted(funcs[f])}"))
        if name == "ov-job":
            cb, ca = _insert_columns(before["job/run_report.py"]), _insert_columns(after["job/run_report.py"])
            if not (cb and ca and set(cb) < set(ca)):
                bad.append((repo, name, f"write_report must write more columns to order_report: {cb} -> {ca}"))
            kb, ka = _return_keys(before["job/trigger_server.py"], "status"), _return_keys(after["job/trigger_server.py"], "status")
            if not (kb and ka and kb != ka):
                bad.append((repo, name, f"status response shape must change: {kb} -> {ka}"))
        elif name == "ov-svc":
            cb, ca = _table_columns(before["db/schema.sql"]), _table_columns(after["db/schema.sql"])
            if not (cb and ca and "total" in cb and "total" not in ca and len(set(ca) - set(cb)) == 1 and len(ca) == len(cb)):
                bad.append((repo, name, f"orders.total must be removed and one column added: {cb} -> {ca}"))
        elif name == "tf-mid":
            fb, fa = _funcs(before["api/ops.py"]), _funcs(after["api/ops.py"])
            if not fa["exec_cmd"][1] > fb["exec_cmd"][1]:
                bad.append((repo, name, "tf-mid must add a line above the sink (exec_cmd must move down)"))
        elif name == "tf-sink":                       # the sink is the last element: an in-place edit moves nothing
            fb, fa = _funcs(before["api/ops.py"]), _funcs(after["api/ops.py"])
            if {n: v[1] for n, v in fb.items()} != {n: v[1] for n, v in fa.items()}:
                bad.append((repo, name, "tf-sink is a single in-place edit: no function may move"))
    return bad


def test_f2_branches_realize_the_plan_change_sets(built, manifests):
    out, _ = built
    for repo in ("fx-job", "fx-svc"):
        assert branch_violations(manifests, repo, out / repo) == [], repo
    for cue in ("`write_report`", "`status`", "`orders.total`", "adds a line above the sink", "`tf-mid`", "`tf-sink`", "`tf-src`"):
        assert cue in PLAN_TEXT.replace("job:job/run_report.py::", "`").replace("**adds a line above the sink**", "adds a line above the sink") or cue.strip("`") in PLAN_TEXT, cue


def _branch_mutants(manifests):
    """[(label, repo, mutated manifest)]: per branch {base}, per op {path, from, to x3}."""
    out = []
    for repo in ("fx-job", "fx-svc"):
        for bi, b in enumerate(manifests[repo]["history"]["branches"]):
            other_path = {"fx-job": "job/paths.py", "fx-svc": "api/health.py"}[repo]

            def with_(edit, label, bi=bi, repo=repo, b=b):
                m = copy.deepcopy(manifests[repo])
                edit(m["history"]["branches"][bi])
                out.append((label, repo, m))

            with_(lambda x: x.__setitem__("base", "H3"), f"{b['name']}:base")
            for oi, op in enumerate(b["ops"]):
                tag = f"{b['name']}:op{oi}"
                with_(lambda x, oi=oi, p=other_path: x["ops"][oi].__setitem__("path", p), tag + ":path")
                with_(lambda x, oi=oi: x["ops"][oi].__setitem__("from", x["ops"][oi]["from"][:-1]), tag + ":from")
                with_(lambda x, oi=oi: x["ops"][oi].__setitem__("to", x["ops"][oi]["to"] + ")"), tag + ":to-syntax")
                with_(lambda x, oi=oi: x["ops"][oi].__setitem__("to", x["ops"][oi]["from"]), tag + ":to-noop")
                with_(lambda x, oi=oi: x["ops"][oi].__setitem__("to", re.sub(r"([A-Za-z])(\W*)$", lambda m_: ("b" if m_.group(1).lower() != "b" else "c").upper() * m_.group(1).isupper() + ("b" if m_.group(1).lower() != "b" else "c") * (not m_.group(1).isupper()) + m_.group(2), x["ops"][oi]["to"].rstrip("\n"), count=1) + ("\n" if x["ops"][oi]["to"].endswith("\n") else "")), tag + ":to-token")
    return out


def run_branch_probe(manifests, tmp_path):
    results = []
    for label, repo, mutated in _branch_mutants(manifests):
        assert mutated != manifests[repo], f"{label}: an equivalent mutant (no change)"
        exp = tmp_path / "exp"
        exp.mkdir(exist_ok=True)
        (exp / f"{repo}.yaml").write_text(yaml.safe_dump({**manifests[repo], "history": mutated["history"] if "history" in mutated else mutated}, sort_keys=False, width=10000))
        out = tmp_path / "m" / label.replace(":", "_")
        out.mkdir(parents=True)
        try:
            info = M.materialize_repo(repo, out, expected_dir=exp)
        except M.MaterializeError:
            results.append((label, "anchor", True))
            continue
        try:
            sem = branch_violations({repo: mutated}, repo, out / repo)
        except (SyntaxError, StopIteration, AttributeError, subprocess.CalledProcessError) as exc:
            sem = [(repo, label, f"unreadable after-state: {type(exc).__name__}")]
        pin = sha_pin_violations({repo: info})
        results.append((label, "semantic" if sem else "pin-only" if pin else "SURVIVED", bool(sem or pin)))
    return results


def test_f2_probe_every_branch_op_field_mutation_is_caught(manifests, tmp_path):
    """5 branches: base x1 and, per op (6 ops), path, from and three `to` variants. Every mutant must be red."""
    mutants = _branch_mutants(manifests)
    assert len(mutants) == 5 + 6 * 5, len(mutants)
    res = run_branch_probe({r: manifests[r] for r in manifests}, tmp_path)
    survivors = [r for r in res if not r[2]]
    assert survivors == [], survivors
    how = {}
    for _, kind, _ in res:
        how[kind] = how.get(kind, 0) + 1
    print(f"F-2 probe: {len(res)} mutants, 0 survivors; detected by {how}")
    assert how.get("semantic", 0) + how.get("anchor", 0) >= 20, how


# ======================================================================================== TP-006 content check, live
RUNNER = CPG / "sandbox" / "offline-run.sh"
CHECK_SC = CPG / "sandbox" / "scripts" / "cpg-content-check.sc"
LANG_FLAG = {"python": "pythonsrc", "javascript": "jssrc", "java": "javasrc"}


def runner(args, audit_dir, timeout=900):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "CCG_AUDIT_DIR": str(audit_dir)}
    return subprocess.run([str(RUNNER), "--backend", "container", "--mode", "offline", *args], env=env,
                          capture_output=True, text=True, timeout=timeout)


def last_audit_id(audit_dir):
    return json.loads((audit_dir / "sandbox.jsonl").read_text().splitlines()[-1])["auditId"]


def parse_unit(codebase, src, lang, outdir, audit_dir):
    r = runner(["--frontend", "joern-parse", "--codebase", codebase, "--source", str(src), "--output", str(outdir),
                "--", "/src", "--language", LANG_FLAG[lang]], audit_dir)
    return r, last_audit_id(audit_dir)


def content_check(cpg_ids, outdir, audit_dir, src):
    files = ";".join(f"/out/{i}.cpg.bin" for i in cpg_ids)
    r = runner(["--frontend", "joern", "--script-file", str(CHECK_SC), "--source", str(src), "--output", str(outdir),
                "--", "--script", "/scripts/cpg-content-check.sc", "--param", f"cpgFiles={files}"], audit_dir)
    verdict, counts, per_file, methods = {}, {}, {}, {}
    for ln in r.stdout.splitlines():
        m = re.match(r"^METHODS /out/([0-9a-f-]+)\.cpg\.bin=(.*)$", ln)
        if m:
            methods[m.group(1)] = set(filter(None, m.group(2).split(",")))
        m = re.match(r"^CONTENT /out/([0-9a-f-]+)\.cpg\.bin=(ok|EMPTY)$", ln)
        if m:
            verdict[m.group(1)] = m.group(2)
        m = re.match(r"^METHOD_COUNT /out/([0-9a-f-]+)\.cpg\.bin=(\d+)$", ln)
        if m:
            counts[m.group(1)] = int(m.group(2))
        m = re.match(r"^FILE_METHODS /out/([0-9a-f-]+)\.cpg\.bin\t(.*)=(\d+)$", ln)
        if m:
            per_file.setdefault(m.group(1), {})[m.group(2)] = int(m.group(3))
    return r, verdict, counts, per_file, methods


def is_callable_symbol(text, path, qname):
    """True when the locator names a function, method or class (not a constant, a field or a path)."""
    q = qname.split("(")[0]
    ext = os.path.splitext(path)[1]
    if ext == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return False
        want = q.split(".")
        scope = tree.body
        node = None
        for part in want:
            node = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))
                         and (getattr(n, "name", None) == part or any(isinstance(t, ast.Name) and t.id == part for t in getattr(n, "targets", [getattr(n, "target", None)])))), None)
            if node is None:
                return False
            scope = getattr(node, "body", [])
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    if ext in (".ts", ".tsx"):
        head = re.escape(q.split(".")[0])
        m = re.search(rf"(?m)^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(function\*?\s+{head}\b|class\s+{head}\b|(?:const|let|var)\s+{head}\b[^\n]*)", text)
        if not m:
            return False
        decl = m.group(0)
        return bool(re.search(r"\bfunction\b|\bclass\b|=>|=\s*(?:async\s*)?\(", decl))
    if ext == ".java":
        return "(" in qname
    return False


def expected_symbol_names(manifests, unit, root_dir: Path):
    """Last qname segments of the callable symbols (function/method/class) the manifest places in *unit*."""
    nested = [u["rootPath"] for u in manifests[unit["repoId"]]["units"] if u is not unit and u["rootPath"] != "."
              and (unit["rootPath"] == "." or u["rootPath"].startswith(unit["rootPath"]))]
    root = "" if unit["rootPath"] == "." else unit["rootPath"]
    ext = {"python": (".py",), "javascript": (".ts", ".tsx"), "java": (".java",)}[unit["lang"]]
    names = {}
    for _, _, _, loc in all_locators({**manifests, "cross": {"rows": []}}):
        if loc["repo"] != unit["repoId"] or not loc.get("qname") or not loc["path"].startswith(root) or any(loc["path"].startswith(n) for n in nested):
            continue
        if not loc["path"].endswith(ext):
            continue
        q = loc["qname"].split("::")[-1] if "::" in loc["qname"] else loc["qname"]
        if q.startswith("<"):
            continue
        f = root_dir / loc["repo"] / loc["path"]
        if f.is_file() and is_callable_symbol(f.read_text(), loc["path"], q):
            names[(loc["path"], q.split("(")[0])] = q.split("(")[0].split(".")[-1]
    return names          # {(path, dotted qname): last segment}: Rules.fee and PremiumRules.fee are two entries


# Recorded per-unit floors, from the live run of 2026-11-04 (callable symbols only): (expected callable symbols,
# of which present in the parsed CPG). A drop is a red, not an allowance. S39 owns this rule only: S25/S28 (and S11)
# must apply a named-symbol rule themselves, because `cpg-content-check.sc` (S08, unchanged) passes any unit whose
# files each emit the synthetic `<module>` METHOD, including a comments-only unit (planted test below).
# svc is (65, 64): `db/settings.py::Settings` is a class without methods; classes are TYPE_DECLs, not METHODs.
UNIT_FLOORS = {"job": (15, 15), "lib-cli": (3, 3), "lib-pricing": (5, 5), "svc": (65, 64), "svc-common": (4, 4),
               "svc-legacy": (1, 1), "svc-tools": (1, 1), "web": (33, 33)}


@pytest.fixture(scope="module")
def runner_ready():
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.fail("blocked: Docker is not available (the live item needs the S08 container backend)")
    tag = subprocess.run([str(CPG / "infra" / "toolchain" / "build.sh"), "--print-tag"], capture_output=True, text=True).stdout.strip()
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
        pytest.fail(f"blocked: toolchain image {tag!r} is not built (cpg/infra/toolchain/build.sh)")
    assert CHECK_SC.is_file(), "S08's follow-up round must have delivered cpg-content-check.sc (P3-7)"


@pytest.mark.live
def test_tp006_every_unit_parses_in_the_sandbox_and_passes_the_d15_content_check(manifests, built, runner_ready, tmp_path, request):
    """D15's floor plus the S39-only named-symbol rule and recorded per-unit floors (see the module docstring)."""
    out = built[0]
    audit, cpgs = tmp_path / "audit", tmp_path / "cpgs"
    cpgs.mkdir()
    ids, unit_of = [], {}
    for repo in REPOS:
        for u in manifests[repo]["units"]:
            r, aid = parse_unit(u["unitId"], out / repo / u["rootPath"], u["lang"], cpgs, audit)
            assert r.returncode == 0, f"{u['unitId']}: joern-parse exit {r.returncode}: {r.stderr[-400:]}"
            assert "UnknownHostException" not in r.stdout + r.stderr
            assert (cpgs / f"{aid}.cpg.bin").stat().st_size > 20000, f"{u['unitId']}: CPG suspiciously small"
            ids.append(aid)
            unit_of[aid] = u
    r, verdict, counts, per_file, methods = content_check(ids, cpgs, audit, out / "fx-job")
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-400:]
    assert set(verdict) == set(ids)
    assert all(v == "ok" for v in verdict.values()), {unit_of[i]["unitId"]: v for i, v in verdict.items()}
    assert all(counts[i] >= 1 for i in ids)
    report = {}
    for i in ids:
        u = unit_of[i]
        want = expected_symbol_names(manifests, u, out)
        got = {n for n in methods[i] if not n.startswith("<")}
        found = {k for k, v in want.items() if v in got}
        report[u["unitId"]] = {"methods": counts[i], "named": len(got), "expected_callable": len(want), "expected_found": len(found),
                               "expected_missing": sorted(f"{p}::{q}" for (p, q) in set(want) - found),
                               "files_with_zero_methods": sorted(f for f, n in per_file.get(i, {}).items() if n == 0 and f != "N/A")}
        assert got, f"{u['unitId']}: only synthetic methods (a <module> per file): the D15 floor passes but nothing was parsed"
        assert set(report[u["unitId"]]["files_with_zero_methods"]) <= {"<unknown>"}, (u["unitId"], report[u["unitId"]]["files_with_zero_methods"])
    if os.environ.get("CCG_FIXTURES_REPORT"):
        Path(os.environ["CCG_FIXTURES_REPORT"]).write_text(json.dumps(report, indent=1, sort_keys=True))
    assert set(report) == set(UNIT_FLOORS), "every unit has a recorded floor"
    for unit, rep in report.items():
        want_n, want_found = UNIT_FLOORS[unit]
        assert rep["expected_callable"] == want_n, (unit, rep)
        assert rep["expected_found"] >= want_found, (unit, rep)


@pytest.mark.live
def test_tp006_planted_failing_inputs_a_comments_only_unit_and_a_unit_with_no_source_fail_the_floor(runner_ready, tmp_path):
    comments, nosrc = tmp_path / "comments", tmp_path / "nosrc"
    comments.mkdir()
    nosrc.mkdir()
    (comments / "only_comments.py").write_text("# nothing but a comment\n# and another\n")
    (nosrc / "README.md").write_text("no source here\n")
    audit, cpgs = tmp_path / "audit", tmp_path / "cpgs"
    cpgs.mkdir()
    good = tmp_path / "good"
    good.mkdir()
    (good / "m.py").write_text("def f(x):\n    return x + 1\n")
    ids = {}
    for name, src in (("comments", comments), ("good", good)):
        r, aid = parse_unit(name, src, "python", cpgs, audit)
        assert r.returncode == 0, (name, r.stderr[-300:])
        ids[name] = aid
    r, verdict, counts, _, methods = content_check([ids["comments"], ids["good"]], cpgs, audit, good)
    # OBSERVED (plan TP-006's branch "if the frontend still emits a module-level METHOD"): pysrc2cpg emits a synthetic
    # `<module>` METHOD per file, so D15's unit-level floor passes a comments-only unit. The named-symbol rule catches it.
    assert verdict[ids["comments"]] == "ok" and methods[ids["comments"]] == {"<module>"}
    assert not {n for n in methods[ids["comments"]] if not n.startswith("<")}, "the named-symbol rule must fail the comments-only unit"
    assert {n for n in methods[ids["good"]] if not n.startswith("<")} == {"f"}
    assert verdict[ids["good"]] == "ok"
    # a unit with no source files is the case the floor does catch: EMPTY and a non-zero exit
    r, aid = parse_unit("nosrc", nosrc, "python", cpgs, audit)
    assert r.returncode == 0 and (cpgs / f"{aid}.cpg.bin").is_file(), "the exit-0 empty parse is the hazard (egress s9 H12)"
    r, v, c, _, _ = content_check([aid, ids["good"]], cpgs, audit, good)
    assert v == {aid: "EMPTY", ids["good"]: "ok"} and c[aid] == 0
    assert r.returncode != 0, "the check exits non-zero when any CPG is EMPTY"

# ======================================================================================== P1 form of the flow-dependent rows (plan 5.5)
# The manifests hold the flag-true form in `status`/`reason`/`dependencies`; `p1Form` is the form P1-41 asserts while
# the flow flag is false. The expected values are read from the plan's "P1 form" table, not from a constant here.


def p1_form_plan_rows():
    text = PLAN_TEXT.split("**P1 form (flow flag false")[1].split("Consequence for A25-3")[0]
    out = {}
    for ln in text.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if ln.startswith("|") and re.match(r"^(DF|CD|RV)-\d", cells[0]):
            for rid in re.findall(r"(?:DF|CD|RV)-\d", cells[0]):
                out[rid] = cells[1]
    return out


def _qn(loc):
    """The plan's `short:qname` spelling of a locator: a dict (full dialect) or an already-prefixed string (compact)."""
    if isinstance(loc, dict):
        return f"{loc['repoId'].removeprefix('fx-')}:{loc['qname']}"
    return loc


def _leaf_forms(form):
    """[(status, reason, locator, distance)] a `p1Form` states (full and compact dialects)."""
    out = []
    if "status" in form or "st" in form:
        out.append((form.get("status") or form["st"], form["reason"], _qn(form.get("locator") or form["loc"]),
                    form.get("distance", form.get("dist"))))
    for d in form.get("dependencies", []) + form.get("deps", []):
        out.append((d.get("status") or d["st"], d["reason"], _qn(d.get("locator") or d["loc"]), d.get("distance", d.get("dist"))))
    if "caller" in form:
        c = form["caller"]
        out.append((c["status"], c["reason"], _qn(c["locator"]), c["distance"]))
    return out


def p1_form_violations(manifests):
    plan, bad, seen = p1_form_plan_rows(), [], {}
    for repo in REPOS:
        for r in manifests[repo]["rows"]:
            if "p1Form" in r:
                seen.setdefault(r["id"], []).append(r["p1Form"])
    for rid, forms in seen.items():
        if any(sorted(_leaf_forms(f)) != sorted(_leaf_forms(forms[0])) for f in forms):   # fx-web writes the compact dialect
            bad.append(f"{rid}: holders disagree on p1Form")
        cell = plan.get(rid)
        if cell is None:
            bad.append(f"{rid}: p1Form with no row in the plan's P1-form table")
            continue
        f = forms[0]
        if f.get("asserted") is False and "not asserted" not in cell:
            bad.append(f"{rid}: manifest says not asserted, plan says {cell!r}")
        for st, reason, loc, dist in _leaf_forms(f):
            if f"`{st}` / `{reason}`" not in cell:
                bad.append(f"{rid}: p1Form {st}/{reason} not in plan cell {cell!r}")
            if f"`{loc}`" not in cell:
                bad.append(f"{rid}: p1Form locator {loc} not in plan cell {cell!r}")
            if f"`{loc}` (d{dist})" not in cell:
                bad.append(f"{rid}: p1Form distance {dist} for {loc} not in plan cell {cell!r}")
    for rid in set(plan) - set(seen) - {"DF-4", "CD-4"}:
        bad.append(f"{rid}: in the plan's P1-form table, no p1Form in any manifest")
    return bad


def test_p1_form_equals_the_plan_table_and_covers_every_flow_dependent_row(manifests):
    assert set(p1_form_plan_rows()) >= {"DF-1", "DF-2", "DF-3", "CD-1", "CD-2", "CD-3", "RV-1", "RV-2", "RV-3"}
    assert p1_form_violations(manifests) == []
    held = M.holders(manifests)
    for rid in ("DF-1", "DF-2", "CD-1", "CD-2", "CD-3"):             # the flag-true form stays in place for the re-run
        assert next(iter(held[rid].values()))["status"] == "confirmed"
        assert next(iter(held[rid].values()))["reason"] == "intra-flow"


def test_p1_form_planted_failing_inputs(manifests):
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "DF-2")["p1Form"]["status"] = "confirmed"            # the form index 5 forbids under a false flag
    assert any(x.startswith("DF-2") for x in p1_form_violations(bad))
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-web"], "RV-3")["p1Form"]["deps"][0]["st"] = "confirmed"     # one holder of RV-3 only
    v = p1_form_violations(bad)
    assert any("RV-3" in x and "disagree" in x for x in v), v
    bad = copy.deepcopy(manifests)
    del row_by_id(bad["fx-svc"], "RV-1")["p1Form"]
    assert any(x.startswith("RV-1") for x in p1_form_violations(bad))
    # DF-3 is no longer exempt: its P1 form is possible/flow-unavailable:python (spec2 5, B-flow item C)
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-web"], "DF-3")["p1Form"]["st"] = "confirmed"
    assert any(x.startswith("DF-3") for x in p1_form_violations(bad))
    bad = copy.deepcopy(manifests)
    del row_by_id(bad["fx-web"], "DF-3")["p1Form"]
    assert any(x.startswith("DF-3") and "no p1Form" in x for x in p1_form_violations(bad))
    # N3: the locator and the distance are compared, not only status and reason
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-web"], "DF-3")["p1Form"]["loc"] = "web:app/src/ui/Reports.tsx::Reports"
    assert any(x.startswith("DF-3") and "locator" in x for x in p1_form_violations(bad))
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "DF-2")["p1Form"]["locator"]["qname"] = "api/orders.py::accept_quote"
    assert any(x.startswith("DF-2") and "locator" in x for x in p1_form_violations(bad))
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "RV-1")["p1Form"]["dependencies"][0]["distance"] = 3
    assert any(x.startswith("RV-1") and "distance" in x for x in p1_form_violations(bad))
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "CD-2")["p1Form"]["caller"]["distance"] = 2
    assert any(x.startswith("CD-2") and "distance" in x for x in p1_form_violations(bad))


# ======================================================================================== AC6-PM rows vs the Markdown extractor
# A prompt-mention row whose consumer is a `.md` file must be (a) in a file the manifest globs classify `prompt-md` and
# (b) give exactly one `possible`/`prompt-mention` call site naming the row's tool (the last `::` segment of its
# trueTarget) on the row's line (the one line of the file that names the tool; rows carry no line field).

def _load_md_extractor():
    sys.path.insert(0, str(CPG))
    try:
        from extractors.markdown.extract import extract_markdown
        from extractors.markdown.globs import load_globs
    finally:
        sys.path.remove(str(CPG))
    return extract_markdown, load_globs


def pm_md_violations(manifests, tree=None, globs_manifest=None, tools=None):
    extract_markdown, load_globs = _load_md_extractor()
    globs = load_globs(globs_manifest) if globs_manifest else load_globs()
    tree = Path(tree) if tree else M.SYSTEM
    bad = []
    for repo, doc in manifests.items():
        for r in doc.get("rows", []):
            if not r["id"].startswith("AC6-PM-") or not r["consumer"][0]["path"].endswith(".md"):
                continue
            path, tool = r["consumer"][0]["path"], r["trueTarget"][0]["qname"].split("::")[-1]
            if globs.classify(path) != "prompt-md":
                bad.append(f"{r['id']}: {path} classifies {globs.classify(path)}, not prompt-md")
                continue
            data = (tree / repo / path).read_bytes()
            want_line = [i for i, ln in enumerate(data.decode().splitlines(), 1) if tool in ln]
            ex = extract_markdown(path, data, repo_id=repo, unit_id="svc", globs=globs,
                                  known_tools=frozenset({tool} if tools is None else tools))
            sites = [x for x in ex.records if x["type"] == "callsite"]
            got = [(x["calleeName"], x["line"], x["status"], x["reason"]) for x in sites]
            if got != [(tool, ln, "possible", "prompt-mention") for ln in want_line] or len(want_line) != 1:
                bad.append(f"{r['id']}: call sites {got}, expected one {tool} possible/prompt-mention at line {want_line}")
    return bad


def test_ac6_pm_markdown_rows_classify_prompt_md_and_yield_their_mention(manifests):
    assert pm_md_violations(manifests) == []
    assert sum(1 for r in manifests["fx-svc"]["rows"] if r["id"].startswith("AC6-PM-") and r["consumer"][0]["path"].endswith(".md")) == 2


def test_ac6_pm_planted_failures_turn_it_red(manifests, tmp_path):
    # the defect P1-12 found: the row re-pathed to the pre-move `prompts/agent.md` (a `doc`); the file is copied there
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "AC6-PM-1")["consumer"][0]["path"] = "prompts/agent.md"
    tree = tmp_path / "tree"
    shutil.copytree(M.SYSTEM / "fx-svc", tree / "fx-svc")
    (tree / "fx-svc" / "prompts").mkdir(exist_ok=True)
    shutil.copy(tree / "fx-svc" / ".claude/agents/agent.md", tree / "fx-svc" / "prompts" / "agent.md")
    assert any(x.startswith("AC6-PM-1") and "classifies doc" in x for x in pm_md_violations(bad, tree=tree))
    # a row path swapped to the other row's file: the tool is not on any line there
    bad = copy.deepcopy(manifests)
    row_by_id(bad["fx-svc"], "AC6-PM-1")["consumer"][0]["path"] = ".claude/agents/tools.md"
    assert any(x.startswith("AC6-PM-1") for x in pm_md_violations(bad))
    # an emptied known_tools set: no call site at all
    assert len(pm_md_violations(manifests, tools=())) == 2
    # a manifest whose globs lack `.claude/agents/**/*.md`: the files classify `doc`
    m = yaml.safe_load((CPG / "eval" / "eval-manifest.yaml").read_text())
    m["promptBearingMarkdown"]["include"] = [g for g in m["promptBearingMarkdown"]["include"] if not g.startswith(".claude/agents")]
    alt = tmp_path / "eval-manifest.yaml"
    alt.write_text(yaml.safe_dump(m))
    assert len(pm_md_violations(manifests, globs_manifest=alt)) == 2
