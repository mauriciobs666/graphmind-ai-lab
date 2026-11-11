"""P1-44a: rule S6a (``-bundle-spec6.md`` §1) against REAL host bundles.

The architect's check: S6a (every ``file.lang`` equals ``header.unit.lang`` when ``sandboxAuditIds`` is empty) is stricter than set
membership, so it must hold for what the committed extractors really emit. The fixture repositories (``cpg/fixtures/system1``: shell
scripts, json/yaml/toml/Dockerfile/Compose/pom.xml/requirements/.env config files, Markdown prompt files) are committed to a throwaway
git repository and every host unit is built through the real ``ccg.build`` path (read only: ``build.py`` is not edited); the written
bundle then has its ``sandboxAuditIds`` emptied and must validate.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ccg import atrest, build, bundle, ids
from ccg.manifest import Manifest, Policy, Repo
from ccg.units import Unit, discover_units
from extractors.markdown.globs import load_globs

CPG = Path(__file__).resolve().parent.parent
FIXTURES = CPG / "fixtures" / "system1"
AUDIT = "0f8fad5b-d9cb-469f-a165-70867728950e"
SYSTEM = "sys1"


def _git(path: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


@pytest.fixture(scope="module")
def real_host_bundles(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("s6a")
    repo = tmp / "repo"
    for name in ("fx-svc", "fx-job", "fx-lib", "fx-web"):
        shutil.copytree(FIXTURES / name, repo / name)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c1")
    sha = _git(repo, "rev-parse", "HEAD")
    work, keys = tmp / "work", tmp / "keys"
    work.mkdir(mode=0o700)
    manifest = Manifest(system=SYSTEM, repos=(Repo(repo_id="r", path=".", policy=Policy(egress="local-only")),))
    units = [u for u in discover_units(repo, manifest) if u.language in bundle.HOST_LANGS]              # one file per unit
    units += [Unit(ids.unit_id("r", root, lang), "r", root, lang)                                       # and whole directories
              for root in (".", "fx-svc", "fx-job") for lang in bundle.HOST_LANGS]
    ctx = build.BuildContext(units=tuple(units), root=work, keys_root=keys, audit_ids=(AUDIT,), built_at="2026-11-11T00:00:00Z",
                             globs=load_globs(CPG / "eval" / "eval-manifest.yaml"))
    out = []
    for u in units:
        res = build.build_unit(u, repo, sha, SYSTEM, ctx)
        assert res.ok, (u.language, u.rootPath, res.fail_reason)
        out.append(bundle.loads(atrest.read_bundle(SYSTEM, u.unitId, sha, work)))
    return out


def test_the_fixture_yields_host_units_of_all_three_languages_and_a_multi_file_one(real_host_bundles):
    assert max(sum(1 for r in b if r["type"] == "file") for b in real_host_bundles if b[0]["unit"]["lang"] == "config") > 1
    assert {b[0]["unit"]["lang"] for b in real_host_bundles} == set(bundle.HOST_LANGS)
    assert sum(1 for b in real_host_bundles for r in b if r["type"] == "file") > 10


def test_every_real_host_bundle_passes_s6a_with_an_empty_list(real_host_bundles):
    """Real config units hold json/yaml/toml/Dockerfile/Compose/pom.xml files; their ``file.lang`` is still ``config``."""
    for recs in real_host_bundles:
        assert all(r["lang"] == recs[0]["unit"]["lang"] for r in recs if r["type"] == "file"), recs[0]["unit"]
        emptied = [{**recs[0], "sandboxAuditIds": []}, *recs[1:]]
        bundle.validate(emptied)
        bundle.validate(recs)                                   # and a host bundle that names an id stays valid


def test_a_real_host_bundle_with_one_foreign_file_label_is_refused(real_host_bundles):
    for recs in real_host_bundles:
        i = next((n for n, r in enumerate(recs) if r["type"] == "file"), None)
        if i is None:
            continue
        bad = [{**recs[0], "sandboxAuditIds": []}, *recs[1:]]
        bad[i] = {**bad[i], "lang": "python"}
        with pytest.raises(bundle.BundleError, match="S6a"):
            bundle.validate(bad)
