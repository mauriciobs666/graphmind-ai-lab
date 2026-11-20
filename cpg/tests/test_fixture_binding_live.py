"""P1-53a, live tier: the Link-level rows of defect D3 over the real fixture build (Joern in the sandbox), `-m live`.

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-53 (53a). The fixtures are materialized, every unit is built by `ccg.build.build_units`
(the sandbox runner, no store), and `run_packs` (the default packs, the TypeScript MCP-client pack of P1-52 among them), `Binder` and `link`
run over the stored bundles, as `ccg.refresh`'s link stage does. The rows LNK-45, 16, 19, 12, 02 and 06 are asserted by
`test_fixture_binding_offline.link_violations`: the expectations are the manifest rows' and the `BOUND_BY` locators the manifest's `aux` rows.

Against the unmodified fixtures it is red on exactly: LNK-45 has no `BOUND_BY`, no `port-shared` link for LNK-16, and LNK-02, 06 and 12 are
`confirmed`/`key-match` (no 8200 fact), LNK-12 linking to both `reindex` providers; the `launchUnit` for `svc` (absent) is the offline tier's
assertion. It goes green when P1-53b edits the fixtures.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ccg import atrest, build, bundle, units as U
from ccg.linker import Binder, link
from ccg.manifest import load_manifest
from ccg.packrt import run_packs
from ccg.refresh import DEFAULT_PACKS
from ccg.rulepack import load_pack
from ccg.staleness import local_head
from test_fixture_binding_offline import CPG, M, SYSTEM_YAML, World, link_violations, locator_str

SYSTEM = "fixture-binding-live"


@pytest.fixture(scope="module")
def runner_ready():
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.fail("blocked: Docker is not available (the live tier needs the S08 container backend)")
    tag = subprocess.run([str(CPG / "infra" / "toolchain" / "build.sh"), "--print-tag"], capture_output=True, text=True).stdout.strip()
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
        pytest.fail(f"blocked: toolchain image {tag!r} is not built (cpg/infra/toolchain/build.sh)")


@pytest.fixture(scope="module")
def manifests():
    return M.load_all()


@pytest.fixture(scope="module")
def fixture_build(runner_ready, tmp_path_factory):
    """The materialized fixtures, every unit built through the sandbox, the packs, the Binder and the linker: a ``World``."""
    root = tmp_path_factory.mktemp("fx-binding-live")
    M.materialize(root)
    mf = load_manifest(SYSTEM_YAML)
    repos = {r.repo_id: root / r.path for r in mf.repos}
    heads = {k: local_head(v) for k, v in repos.items()}
    units = U.discover_units(root, mf, commit=heads)
    area, keys, audit = root / ".ccg-work", root / "keys", root / "audit"
    for d in (area, keys, audit):
        d.mkdir(mode=0o700)
    old = os.environ.get("CCG_AUDIT_DIR")
    os.environ["CCG_AUDIT_DIR"] = str(audit)
    try:
        ctx = build.BuildContext(units=tuple(units), root=area, keys_root=keys, manifest=SYSTEM_YAML)
        report = build.build_units(units, repos, heads, SYSTEM, ctx)
    finally:
        if old is None:
            os.environ.pop("CCG_AUDIT_DIR", None)
        else:
            os.environ["CCG_AUDIT_DIR"] = old
    failed = [(r.unit.unitId, r.fail_reason) for r in report.results if not r.ok]
    assert failed == [], failed
    bundles = [bundle.loads(atrest.read_bundle(SYSTEM, u.unitId, heads[u.repoId], area)) for u in units]
    packs = [load_pack(CPG / "packs" / n / "pack.yaml") for n in DEFAULT_PACKS]
    facts = run_packs(bundles, packs)
    binder = Binder(facts, bundles, key=atrest.hmac_key(SYSTEM, keys), repos={r.repo_id: r.path for r in mf.repos}, packs=packs, units=units)
    links = link(facts, bundles, packs=packs, bindings=binder)
    records, sym_of, loc_of = {}, {}, {}
    for b in bundles:
        repo = b[0]["unit"]["repoId"]
        for r in b:
            if isinstance(r.get("id"), str):
                records[r["id"]] = (r, r.get("repoId", repo))
            if r.get("type") == "symbol":
                loc = locator_str(repo, r["path"], r["qname"].split("::", 1)[-1])
                sym_of[loc], loc_of[r["id"]] = r["id"], loc
    return World(root, units, links.links, binder, facts, records, sym_of, loc_of, {}, {}, {}, {})


@pytest.mark.live
def test_the_fixture_build_links_the_binding_rows_as_the_manifests_say(fixture_build, manifests):
    """LNK-45 (`possible`/`prompt-mention` with its two-entry chain), LNK-16 (`port-shared`), LNK-19 (unbound, both `reindex` providers), and
    LNK-12, 02, 06 (`binding-confirmed`, each to exactly its `svc` provider)."""
    assert link_violations(fixture_build, manifests) == []


@pytest.mark.live
def test_a_real_extraction_of_jobs_py_yields_the_environ_read_shape_the_binder_table_expects(fixture_build):
    """P1-54 M1: the census over the static golden proves the table matches stored records; this pairs it with a live extraction of the committed
    ``fx-svc/clients/jobs.py``, so a change in the extractor's naming (not in the golden) reddens here instead of silently killing behaviour (5)."""
    from ccg.linker.bind import _ENV_READ_TARGETS
    reads = [r for r, _ in fixture_build.records.values()
             if r.get("type") == "callsite" and "clients/jobs.py" in fixture_build.loc_of.get(r["symbolId"], "") and "environ" in (r.get("externalTarget") or "")]
    assert reads, "no environ callsite extracted from fx-svc/clients/jobs.py"
    assert all(r["externalTarget"] in _ENV_READ_TARGETS for r in reads), [r["externalTarget"] for r in reads]
