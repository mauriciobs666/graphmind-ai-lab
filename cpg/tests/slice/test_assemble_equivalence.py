"""P1-18 equivalence: ``ccg.assemble.rows.build_rows`` over the stored ``s14a`` slice bundles reproduces the slice's graph.

Oracle: ``cpg/spikes/vertical-slice/assemble.py`` + ``load.py:build_rows`` run live over the same bundles (never a golden copied from
the assembler's own output), and the figures recorded in ``docs/plans/connected-code-graph-p0-graph.md`` §1.1 (38,121 nodes, 90,867
edges, written out below). The assembler is fed the P1 pack runtime and linker's output for the ``http`` kind only (the slice's
recognizer knows no other), so the comparison holds for the whole graph; the full-pack figures are asserted for internal consistency.

**Stated deltas** (``-p1-linking.md`` P1-18 (1)): the 3 ``Pack`` nodes of the packs run; ``MOUNTED_BY`` edges (the slice wrote none): one
per distinct (provider handler, mount call site) pair of the oracle's own provider mount chains; ``Unit`` gains ``flowAvailable`` and
``bundleSchemaVersion``; ``Snapshot`` gains props. Nothing else.

D9 (``docs/plans/connected-code-graph-p1.md``): the bundles are gitignored; an absent artifact *skips* this file (unrun, not green), and
their sha256 is pinned (shared with ``test_packrt_conformance.py``). The sandbox audit check runs against the real
``cpg/.audit/sandbox.jsonl`` when it holds every id the two bundles name, else against records planted for exactly those ids.
"""
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packrt_helpers import CPG, S14A  # noqa: E402
from test_packrt_conformance import SHA256  # noqa: E402

from ccg import ids  # noqa: E402
from ccg.assemble import rows as R  # noqa: E402
from ccg.linker import link  # noqa: E402
from ccg.packrt import PackFacts, run_packs  # noqa: E402
from ccg.rulepack import load_pack  # noqa: E402

pytestmark = [pytest.mark.slice,
              pytest.mark.skipif(not S14A.is_dir() or not list(S14A.glob("*.bundle.jsonl")),
                                 reason="stored s14a slice bundles absent (cpg/.ccg-work is gitignored): UNRUN, not green")]

# -p0-graph.md §1.1, written out (not read from the oracle or the assembler)
RECORDED_NODES, RECORDED_EDGES = 38_121, 90_867
PACKS = ("fastapi", "fetch", "fastmcp")
REAL_AUDIT = CPG / ".audit"


def _load(name):
    spec = importlib.util.spec_from_file_location("slice_" + name, CPG / "spikes/vertical-slice" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["slice_" + name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bundles():
    for name, sha in SHA256.items():
        assert hashlib.sha256((S14A / name).read_bytes()).hexdigest() == sha, f"{name}: not the artifact the oracle was pinned to"
    return [[json.loads(line) for line in (S14A / name).read_text().splitlines()] for name in sorted(SHA256)]


@pytest.fixture(scope="module")
def oracle(bundles):
    facts = _load("assemble").assemble(bundles)
    return facts, _load("load").build_rows(bundles, facts)


@pytest.fixture(scope="module")
def packs():
    return [load_pack(CPG / "packs" / n / "pack.yaml") for n in PACKS]


@pytest.fixture(scope="module")
def audit_dir(bundles, tmp_path_factory):
    from ccg.sandbox_audit import check_sandbox_ids
    if all(check_sandbox_ids(b[0], REAL_AUDIT) == [] for b in bundles):
        return REAL_AUDIT
    d = tmp_path_factory.mktemp("audit")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from test_assemble_rows import sbx
    (d / "sandbox.jsonl").write_text("".join(json.dumps(sbx(i)) + "\n" for b in bundles for i in b[0]["sandboxAuditIds"]))
    return d


@pytest.fixture(scope="module")
def p1(bundles, packs):
    facts = run_packs(bundles, packs)
    return facts, link(facts, bundles, packs=packs)


def http_only(facts, links):
    f = PackFacts([x for x in facts.providers if x["kind"] == "http"], [x for x in facts.consumers if x["kind"] == "http"],
                  facts.mounts, facts.lifted, facts.launch_facts, facts.marks, facts.diagnostics)
    return f, links._replace(links=[x for x in links.links if x.kind == "http"])


@pytest.fixture(scope="module")
def mine(bundles, packs, p1, audit_dir):
    f, ls = http_only(*p1)
    return R.build_rows(bundles, ls, f, system="p0slice", packs=packs, audit_dir=audit_dir)


def test_the_oracle_is_the_recorded_slice(oracle):
    _, rows = oracle
    assert rows.node_count() + 1 == RECORDED_NODES and rows.edge_count() == RECORDED_EDGES


def oracle_mounted_pairs(facts) -> int:
    """Distinct (provider handler, mount call site) over the oracle's own provider mount chains."""
    return len({(p["handlerId"], s["siteId"]) for p in facts["providers"] for m in p["mounts"] for s in m["sites"]})


def test_only_the_stated_deltas_separate_the_two_row_sets(oracle, mine):
    facts, o = oracle
    mounted = oracle_mounted_pairs(facts)
    assert mounted == 99                                                    # the slice's recorded provider mount chains
    assert mine.node_count() + 1 == RECORDED_NODES + len(PACKS)
    assert mine.edge_count() == RECORDED_EDGES + mounted
    assert {k: len(v) for k, v in mine.nodes.items() if k != "Pack"} == {k: len(v) for k, v in o.nodes.items()}
    got = {k: len(v) for k, v in mine.edges.items() if k[1] != "MOUNTED_BY"}
    assert got == {k: len(v) for k, v in o.edges.items()}
    assert len(mine.edges[("Symbol", "MOUNTED_BY", "CallSite")]) == mounted
    assert mine.dropped == {}                                               # the oracle drops nothing either (asserted below)
    assert o.dropped == {}


def test_node_and_edge_identities_are_equal_group_by_group(oracle, mine):
    _, o = oracle
    for label, rs in o.nodes.items():
        idp = R.ID_PROP.get(label)
        key = (lambda r: (r["unitId"], r["path"])) if label == "SkippedFile" else (lambda r: r[idp])
        assert sorted(map(key, rs)) == sorted(map(key, mine.nodes[label])), label
    for group, rs in o.edges.items():
        assert Counter((r["s"], r["t"]) for r in rs) == Counter((r["s"], r["t"]) for r in mine.edges[group]), group


def test_node_and_edge_properties_equal_the_oracles_except_the_stated_additions(oracle, mine):
    _, o = oracle
    added = {"Unit": {"flowAvailable", "bundleSchemaVersion"}}
    for label, rs in o.nodes.items():
        if label == "SkippedFile":
            continue
        idp = R.ID_PROP[label]
        mp = {r[idp]: r for r in mine.nodes[label]}
        for r in rs:
            m = {k: v for k, v in mp[r[idp]].items() if k not in added.get(label, ())}
            assert m == r, (label, r[idp])
    for group, rs in o.edges.items():
        if group[1] in ("PROVIDES", "CONSUMES"):          # the facts' own status/reason vs the slice's constants: compared below
            continue
        mp = {}
        for r in mine.edges[group]:
            mp.setdefault((r["s"], r["t"]), []).append(r.get("p"))
        for r in rs:
            assert r.get("p") in mp[(r["s"], r["t"])], group


def test_provides_and_consumes_props_are_the_packs_own_not_the_slices_constants(oracle, mine):
    _, o = oracle
    assert {(r["s"], r["t"]) for r in o.edges[("Symbol", "PROVIDES", "ContractPoint")]} == \
           {(r["s"], r["t"]) for r in mine.edges[("Symbol", "PROVIDES", "ContractPoint")]}
    prov = Counter((r["p"]["status"], r["p"]["reason"]) for r in mine.edges[("Symbol", "PROVIDES", "ContractPoint")])
    assert prov == Counter({("confirmed", "key-match"): 57, ("possible", "partial-key"): 4})     # 61 providers, 4 with `{?}`
    want = Counter((r["p"]["status"], r["p"]["reason"]) for r in o.edges[("Symbol", "PROVIDES", "ContractPoint")])
    # the slice's 59 / 2, less the two providers the pack runtime weakens (P1-16's stated delta: test_packrt_conformance.py §the 61)
    assert want == Counter({("confirmed", "key-match"): 59, ("possible", "partial-key"): 2})
    assert prov == Counter({("confirmed", "key-match"): 59 - 2, ("possible", "partial-key"): 2 + 2})
    cons = {(r["s"], r["t"]): r["p"] for r in mine.edges[("CallSite", "CONSUMES", "ContractPoint")]}
    assert Counter(p["reason"] for p in cons.values()) == Counter({"lifted-role": 11})


def test_d8_default_omission_over_the_slice_and_the_slice_has_a_non_default_to_omit(mine):
    data = [r["p"] for g in mine.edges if g[1] == "DATA" for r in mine.edges[g]]
    assert len(data) == 16_424
    assert not [p for p in data if p.get("status") == "confirmed" or p.get("reason") == "intra-flow"]
    assert Counter((p.get("status"), p.get("reason")) for p in data if "status" in p or "reason" in p) == \
           Counter({("possible", "field-truncated"): 4})
    assert all(not r.get("p") for r in mine.edges[("CallSite", "CALLS", "Symbol")])


def test_mounted_by_ids_are_the_handlers_and_mount_call_sites_of_the_oracles_chains(oracle, mine):
    facts, _ = oracle
    want = {(p["handlerId"], s["siteId"]) for p in facts["providers"] for m in p["mounts"] for s in m["sites"]}
    got = {(r["s"], r["t"]) for r in mine.edges[("Symbol", "MOUNTED_BY", "CallSite")]}
    assert got == want
    line = {r["callSiteId"]: r["line"] for r in mine.nodes["CallSite"]}
    by_sym = {}
    for r in mine.edges[("Symbol", "MOUNTED_BY", "CallSite")]:
        by_sym.setdefault(r["s"], []).append((r["p"]["ord"], r["p"]["mountPath"], line[r["t"]]))
    for ps in by_sym.values():
        assert sorted(o for o, _, _ in ps) == list(range(len(ps)))
        seq = [(path, ln) for _, path, ln in sorted(ps)]
        assert seq == sorted(seq)                                           # ord follows (mountPath, line), ascending
    assert any(len(ps) > 1 and len({path for _, path, _ in ps}) < len(ps) for ps in by_sym.values())   # a tie on the path, broken by line
    assert {path for ps in by_sym.values() for _, path, _ in ps} == {"/", "/shop/api"}


def test_the_slice_snapshot_is_the_stamped_row(mine, packs):
    s = mine.snapshot
    assert s["graphModelVersion"] == "1.1" and s["systemId"] == "p0slice" and s["bundleSchemaVersion"] == "1.0"
    assert s["snapshotId"] == ids.tuple_id(["snapshot", "p0slice", s["manifestHash"]])
    assert [p["packId"] for p in mine.nodes["Pack"]] == sorted(PACKS)
    assert {u["flowAvailable"] for u in mine.nodes["Unit"]} == {False}
    assert {u["bundleSchemaVersion"] for u in mine.nodes["Unit"]} == {"1.0"}


def test_the_full_pack_set_assembles_consistently(bundles, packs, p1, audit_dir):
    """All three packs: the mcp_tool facts and links add nodes/edges the slice never had; nothing may be silently dropped."""
    facts, links = p1
    rows = R.build_rows(bundles, links, facts, system="p0slice", packs=packs, audit_dir=audit_dir)
    assert set(rows.dropped) <= {"provider-key-none", "consumer-key-none"}, rows.dropped
    assert len(rows.nodes["Link"]) == len(links.links) == 180
    assert len(rows.edges[("Link", "TO", "Symbol")]) == len([x for x in links.links if x.provider])
    assert len(rows.edges[("Symbol", "MOUNTED_BY", "CallSite")]) == 99 + 28          # 28 FastMCP tools under the unit-level /mcp mount
    assert {p["mountPath"] for r in rows.edges[("Symbol", "MOUNTED_BY", "CallSite")] for p in [r["p"]]} == {"/", "/shop/api", "/mcp"}


def test_two_builds_of_the_slice_give_identical_rows(bundles, packs, p1, audit_dir):
    facts, links = p1
    a = R.build_rows(bundles, links, facts, system="p0slice", packs=packs, audit_dir=audit_dir)
    b = R.build_rows(list(reversed(bundles)), links, facts, system="p0slice", packs=list(reversed(packs)), audit_dir=audit_dir)
    assert R.canonical_rows(a) == R.canonical_rows(b) and a.snapshot == b.snapshot


def test_a_bundle_whose_audit_ids_have_no_record_is_refused_over_the_slice(bundles, packs, p1, tmp_path):
    facts, links = p1
    (tmp_path / "sandbox.jsonl").write_text("")
    with pytest.raises(R.BundleRefused) as e:
        R.build_rows(bundles, links, facts, system="p0slice", packs=packs, audit_dir=tmp_path)
    assert e.value.code == "bundle:audit-missing"
