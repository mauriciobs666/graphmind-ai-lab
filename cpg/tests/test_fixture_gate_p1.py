"""P1-41 dispatch B: the fixture gate, part 1 — the 87 P1 rows asserted on the materialized fixture system, one test per row id.

Plan: `docs/plans/connected-code-graph-p1-gates.md` P1-41; fixtures plan §5-§7 (`docs/test-plans/connected-code-graph-fixtures.md`); `docs/test-plans/connected-code-graph-p1.md` §6.
Run (cwd `cpg/`; the live tier needs Docker, the toolchain image and a THROWAWAY FalkorDB, never 6379, 6390 or another run's instance):

    CCG_FALKORDB_PORT=<throwaway port> CCG_CONFIG_DIR=<scratch dir with falkordb-admin.pass> CCG_REQUIRE_LIVE=1 \\
        .venv/bin/python -m pytest tests/test_fixture_gate_p1.py -m "live or not live" -q -p no:cacheprovider

Instruments (each P1 row is asserted by the one the plan names; the Link-level rows by two, whose spread the report states):
- **world**: the fixtures materialized, every unit built through the sandbox (all units built), `run_packs` (the nine default packs), `Binder`, `link`,
  `build_rows` (the `test_fixture_binding_live` plumbing, plus the bundles for `build_rows`): the pack facts, the Binder outcomes (LNK-21, LNK-22), the
  assembled edges.
- **store**: the SC-STEADY scenario (`svc-legacy` and `svc-tools` not built: the real build of the materialized HEAD builds `svc-legacy` fresh, so the failure
  is injected through ``refresh(build=...)``) refreshed into a throwaway store; Links, edges, skipped files, units and the walk rows (`Client.ask`) are read from it.
- **scenario SC-FAIL-UNIT** (AB-1): fx-svc at H1 refreshed, then at H4 with `svc-legacy` failing: `stale-kept`, the previous commit, the reason.
Rows the plan names an existing test for are asserted by calling its helper (LNK-45/16/19/12/02/06/AC6-EX-4 chains: `test_fixture_binding_offline.link_violations`;
AC6-PM-3/EX-1: `p56_link_violations`).
Every verdict is written to ``$CCG_GATE_P1_OUT`` (JSON) when set. Expectations come from the manifests only (``fixture_gate_p1_helpers``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from loader_helpers import admin_password, skip_or_fail, throwaway_port

import fixture_gate_p1_helpers as H
import test_fixture_binding_offline as O
from ccg import atrest, build, bundle, sysname, units as U
from ccg import refresh as RF
from ccg.assemble.rows import build_rows
from ccg.linker import Binder, link
from ccg.manifest import load_manifest
from ccg.packrt import run_packs
from ccg.refresh import DEFAULT_PACKS
from ccg.rulepack import load_pack
from ccg.staleness import local_head
from ccg.store import acl
from ccg.store import loader as L
from test_fixture_binding_live import runner_ready  # noqa: F401  (fixture)

HOST = "127.0.0.1"
SYSTEM_YAML = O.SYSTEM_YAML
LEDGER: dict[str, dict] = {}
_CONFIG_DIR_AT_IMPORT = os.environ.get("CCG_CONFIG_DIR")      # the conftest autouse fixture repoints it per test; a lazily built fixture needs the caller's value

MFS = H.manifests()
P1 = H.p1_rows(MFS)
ROW_IDS = list(P1)

# ---- which instrument asserts which row ----------------------------------------------------------------------------------------------------
LINK_IDS = [r for r in ROW_IDS if r.startswith(("LNK", "KN", "AC6-EX", "AC6-PM", "TW-2")) and r not in ("LNK-21", "LNK-22", "LNK-30", "LNK-39", "LNK-40", "LNK-42", "LNK-43")]
BINDER_IDS = ["LNK-21", "LNK-22"]
FANOUT_IDS = list(H.FANOUT_ROWS)
CALLNAME_IDS = ["LNK-30", "LNK-39", "LNK-40", "TW-3", "MR-PY", "MR-TS"]
LIB_IDS = ["AC6-DYN-1", "AC6-DYN-2", "AC6-DYN-3", "TW-1"]
FT_IDS = ["FT-1", "FT-2"]
NC_IDS = ["NC-1", "NC-2", "NC-3", "NC-5", "NC-6", "NC-7"]
SK_IDS = [r for r in ROW_IDS if r.startswith("SK-")]
SCEN_IDS = ["AB-1", "AB-2", "LNK-42", "LNK-43"]
WALK_IDS = ["DF-1", "DF-2", "DF-3", "DF-4", "CD-1", "CD-2", "CD-3", "CD-4", "RV-1", "RV-2", "RV-3"]
MEANS = {**{r: "link (world + store)" for r in LINK_IDS}, **{r: "Binder outcome (world)" for r in BINDER_IDS}, **{r: "fan-out link + CONSUMES fact (world + store)" for r in FANOUT_IDS},
         **{r: "call edges (store)" for r in CALLNAME_IDS}, **{r: "library_api consumer fact (world)" for r in LIB_IDS}, **{r: "DATA edge (store)" for r in FT_IDS},
         **{r: "negative control (store)" for r in NC_IDS}, **{r: "skippedFiles + File (store)" for r in SK_IDS},
         **{r: "scenario build (store)" for r in SCEN_IDS}, **{r: "Client.ask (store)" for r in WALK_IDS}}


# ---- offline: the row set and the mapping ---------------------------------------------------------------------------------------------------
def test_the_p1_row_set_is_the_87_distinct_ids_of_the_manifests_and_every_row_has_one_means():
    assert len(H.M.holders(MFS)) == 134
    assert len(P1) == 87, len(P1)
    raw, raw_p1 = H.raw_counts(MFS)
    assert (raw, raw_p1) == (175, 111), (raw, raw_p1)         # the second definition: holder copies counted separately
    assert not any(r.startswith("B1F") or r.startswith("B1S") for r in P1)        # the B1 rows gate P3
    assert sorted(MEANS) == sorted(ROW_IDS), (sorted(set(ROW_IDS) - set(MEANS)), sorted(set(MEANS) - set(ROW_IDS)))


def test_a_mutated_manifest_status_turns_the_link_checker_red_and_the_unmutated_one_green():
    """The driver's own mutants on synthetic records (the live mutation run is in the report): a status, a provider id and a BOUND_BY ordinal."""
    form = H.expected_form(P1["LNK-45"])
    c, p = H.consumers_of(MFS, "LNK-45"), H.providers_of(MFS, "LNK-45")
    ok = [H.LinkRec(c[0], p[0], "mcp_tool", form["status"], form["link"], (0, 1))]
    assert H.link_row_violations("LNK-45", form, c, p, ok) == []
    assert H.link_row_violations("LNK-45", form, c, p, [H.LinkRec(c[0], p[0], "mcp_tool", "confirmed", form["link"], (0, 1))])
    assert H.link_row_violations("LNK-45", form, c, p, [H.LinkRec(c[0], "svc:mcp/server.py::get_item", "mcp_tool", form["status"], form["link"], (0, 1))])
    assert H.link_row_violations("LNK-45", form, c, p, [H.LinkRec(c[0], p[0], "mcp_tool", form["status"], form["link"], (0, 2))])
    assert H.link_row_violations("LNK-45", form, c, p, [])


# ---- instruments ----------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gate_build(runner_ready, tmp_path_factory):
    """World: the plumbing of `test_fixture_binding_live.fixture_build`, keeping the bundles, packs and audit directory `build_rows` needs."""
    root = tmp_path_factory.mktemp("fx-gate-p1-world")
    O.M.materialize(root)
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
        report = build.build_units(units, repos, heads, "fixture-gate-world", ctx)
    finally:
        if old is None:
            os.environ.pop("CCG_AUDIT_DIR", None)
        else:
            os.environ["CCG_AUDIT_DIR"] = old
    failed = [(r.unit.unitId, r.fail_reason) for r in report.results if not r.ok]
    assert failed == [], failed
    bundles = [bundle.loads(atrest.read_bundle("fixture-gate-world", u.unitId, heads[u.repoId], area)) for u in units]
    packs = [load_pack(O.CPG / "packs" / n / "pack.yaml") for n in DEFAULT_PACKS]
    facts = run_packs(bundles, packs)
    binder = Binder(facts, bundles, key=atrest.hmac_key("fixture-gate-world", keys), repos={r.repo_id: r.path for r in mf.repos}, packs=packs, units=units)
    links = link(facts, bundles, packs=packs, bindings=binder)
    records, sym_of, loc_of = {}, {}, {}
    for b in bundles:
        repo = b[0]["unit"]["repoId"]
        for r in b:
            if isinstance(r.get("id"), str):
                records[r["id"]] = (r, r.get("repoId", repo))
            if r.get("type") == "symbol":
                lc = O.locator_str(repo, r["path"], r["qname"].split("::", 1)[-1])
                sym_of[lc], loc_of[r["id"]] = r["id"], lc
    w = O.World(root, units, links.links, binder, facts, records, sym_of, loc_of, {}, {}, {}, {})
    rows = build_rows(bundles, links, facts, system="fixture-gate-world", packs=packs, audit_dir=audit)
    return SimpleNamespace(w=w, rows=rows, bundles=bundles, units=units, recs=world_recs(w))


def world_recs(w) -> list:
    out = []
    for l in w.links:
        out.append(H.LinkRec(w.loc_of.get(l.consumer, "?"), w.loc_of.get(l.provider, f"unit:{l.to_unit}"), l.kind, l.status, l.reason,
                             tuple(sorted(e["ord"] for e in l.evidence if e["edge"] == "BOUND_BY"))))
    return out


SCEN_UNITS_FAILING = ("legacy", "tools")


def _failing_build(real, failing):
    def build_one(unit, repo, commit, system, ctx=None):
        if unit.language == "python" and unit.rootPath.rstrip("/") in failing:
            return build.BundleResult(unit, commit, False, fail_reason=build.EXTRACT_FAILED)
        return real(unit, repo, commit, system, ctx)
    return build_one


@pytest.fixture(scope="module")
def live_env(tmp_path_factory):
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance: cpg/infra/ccg-falkordb.sh --throwaway up, then export CCG_FALKORDB_PORT and CCG_CONFIG_DIR (a scratch dir)")
    try:
        with pytest.MonkeyPatch.context() as cm:
            if _CONFIG_DIR_AT_IMPORT:
                cm.setenv("CCG_CONFIG_DIR", _CONFIG_DIR_AT_IMPORT)
            pw = admin_password()
    except FileNotFoundError as exc:
        skip_or_fail(f"throwaway admin password not readable ({exc})")
    cred = tmp_path_factory.mktemp("gate-cred")
    cred.chmod(0o700)
    mp = pytest.MonkeyPatch()
    mp.setenv("CCG_CREDENTIALS_DIR", str(cred))
    mp.setenv("CCG_FALKORDB_PORT", str(port))
    admin = L.connect(HOST, port, username=None, password=pw)
    env = SimpleNamespace(port=port, admin=admin, cred=cred, systems=[], tmp=tmp_path_factory.mktemp("gate-live"), mp=mp)
    try:
        yield env
    finally:
        for s in env.systems:
            for k in list(admin.connection.scan_iter(match=f"ccg_{s}*")):
                admin.connection.delete(k)
            for u in (sysname.acl_user(r, s) for r in sysname.ROLES):
                admin.connection.execute_command("ACL", "DELUSER", u)
        mp.undo()


def _manifest_for(env, name: str) -> Path:
    p = env.tmp / f"{name}.yaml"
    p.write_text(SYSTEM_YAML.read_text().replace("system: fixture-system1", f"system: {name}"))
    return p


def _refresh(env, name, root, work, *, scope=None, failing=()):
    with pytest.MonkeyPatch.context() as mp:      # a lazily built scenario runs inside a test, where the autouse conftest fixture has repointed the credentials dir
        mp.setenv("CCG_CREDENTIALS_DIR", str(env.cred))
        if name not in env.systems:
            acl.create_users(env.admin, name)
            env.systems.append(name)
        cr = acl.read_credentials(name)
        bdb = L.connect(HOST, env.port, username=sysname.acl_user("build", name), password=cr["build"])
        vdb = L.connect(HOST, env.port, username=sysname.acl_user("verdict", name), password=cr["verdict"])
        audit = env.tmp / "audit"
        audit.mkdir(exist_ok=True, mode=0o700)
        mp.setenv("CCG_AUDIT_DIR", str(audit))
        for d in (work, env.tmp / "keys"):
            d.mkdir(exist_ok=True, mode=0o700)
        return RF.refresh(name, scope=scope, manifest=_manifest_for(env, name), repo_root=root, work_root=work, keys_root=env.tmp / "keys", audit_dir=audit,
                          target=RF.Target(L.LiveStore(bdb, name), bdb, vdb), build=_failing_build(build.build_unit, failing) if failing else None)


class Store:
    def __init__(self, env, system):
        self.env, self.system = env, system
        self.g = env.admin.select_graph(sysname.live_key(system))

    def q(self, cypher, **p):
        return self.g.ro_query(cypher, p).result_set

    def ask(self, spec):
        from clients.common import Client
        with pytest.MonkeyPatch.context() as mp:           # the autouse conftest fixture points the credentials dir at a per-test directory
            mp.setenv("CCG_CREDENTIALS_DIR", str(self.env.cred))
            return Client(self.system, host=HOST, port=self.env.port).ask(spec)

    def link_recs(self) -> list:
        rows = self.q("MATCH (o:Symbol)-[:HAS_CALL]->(c:CallSite)-[v:VIA]->(l:Link) OPTIONAL MATCH (l)-[:TO]->(t:Symbol) OPTIONAL MATCH (l)-[:TO_UNIT]->(tu:Unit) "
                      "OPTIONAL MATCH (l)-[b:BOUND_BY]->(:ConfigEntry) RETURN o.repoId, o.qname, t.repoId, t.qname, tu.rootPath, l.linkId, l.kind, l.status, l.reason, "
                      "v.consumerKey, collect(distinct b.ord)")
        seen, out = set(), []
        for r in rows:
            prov = H.loc(r[2], r[3].split("::", 1)[0], r[3].split("::", 1)[1]) if r[3] else f"unit:{r[4]}"
            rec = H.LinkRec(H.loc(r[0], r[1].split("::", 1)[0], r[1].split("::", 1)[1]), prov, r[6], r[7], r[8], tuple(sorted(r[10])), r[9])
            if (r[5], rec) not in seen:
                seen.add((r[5], rec))
                out.append(rec)
        return out


@pytest.fixture(scope="module")
def steady(live_env):
    root = live_env.tmp / "root-steady"
    O.M.materialize(root)
    rep = _refresh(live_env, "fxp1-steady", root, live_env.tmp / "work-steady", failing=SCEN_UNITS_FAILING)
    s = Store(live_env, "fxp1-steady")
    s.report, s.recs = rep, s.link_recs()
    return s


@pytest.fixture(scope="module")
def ab1(live_env):
    """SC-FAIL-UNIT: fx-svc built at H1, then at H4 with `svc-legacy` failing."""
    root = live_env.tmp / "root-ab1"
    O.M.materialize(root)
    svc = root / "fx-svc"
    h4 = subprocess.run(["git", "-C", str(svc), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    h1 = subprocess.run(["git", "-C", str(svc), "log", "--format=%H", "--grep", "build point: legacy builds"], capture_output=True, text=True, check=True).stdout.split()[0]
    subprocess.run(["git", "-C", str(svc), "checkout", "-q", "--detach", h1], check=True, env={**os.environ, **O._GIT_ENV})
    work = live_env.tmp / "work-ab1"
    first = _refresh(live_env, "fxp1-ab1", root, work, scope=["fx-svc"])
    subprocess.run(["git", "-C", str(svc), "checkout", "-q", "--detach", h4], check=True, env={**os.environ, **O._GIT_ENV})
    second = _refresh(live_env, "fxp1-ab1", root, work, scope=["fx-svc"], failing=("legacy",))
    s = Store(live_env, "fxp1-ab1")
    s.h1, s.h4, s.first, s.second = h1, h4, first, second
    return s


# ---- the per-row verdict --------------------------------------------------------------------------------------------------------------------
def _record(rid, viol, instruments):
    LEDGER[rid] = {"id": rid, "means": MEANS[rid], "instruments": instruments, "verdict": "fail" if viol else "pass", "violations": viol}
    assert not viol, "\n".join(viol)


@pytest.mark.live
@pytest.mark.parametrize("rid", LINK_IDS)
def test_link_level_row(rid, gate_build, steady):
    form = H.expected_form(P1[rid])
    cons, prov = H.consumers_of(MFS, rid), H.providers_of(MFS, rid)
    key = next(iter(P1[rid].values())).get("keyNorm") if rid.startswith("KN") else None
    sup = rid.startswith("AC6-PM")        # a prompt mention names a tool: the true target is among the Links; a same-named second provider is the Q-NC5 shape
    viol = [f"[world] {v}" for v in H.link_row_violations(rid, form, cons, prov, gate_build.recs, superset_ok=sup)]
    viol += [f"[store] {v}" for v in H.link_row_violations(rid, form, cons, prov, steady.recs, expect_key=key, superset_ok=sup)]
    if not form["linked"] and form["key"]:        # the consumer key survives as a CONSUMES edge of the consumer's call site (partial-key)
        n = steady.q("MATCH (s:Symbol {qname:$q, repoId:$r})-[:HAS_CALL]->(:CallSite)-[e:CONSUMES]->(p:ContractPoint {keyNorm:$k}) "
                     "WHERE e.status='possible' AND e.reason='partial-key' RETURN count(*)", q=cons[0].split(":", 1)[1], r="fx-" + cons[0].split(":", 1)[0], k=form["key"])[0][0]
        if n < 1:
            viol.append(f"[store] {rid}: no CONSUMES possible/partial-key edge with key {form['key']!r} from {cons[0]}")
    if rid in ("LNK-45", "LNK-16", "LNK-19", "LNK-12", "LNK-02", "LNK-06", "AC6-EX-4"):       # the existing pin's chain assertions, over the same build
        viol += [f"[pin link_violations] {v}" for v in O.link_violations(gate_build.w, MFS) if v.startswith(rid)]
    if rid in ("AC6-PM-3", "AC6-EX-1"):
        viol += [f"[pin p56_link_violations] {v}" for v in O.p56_link_violations(gate_build.w, MFS) if v.startswith(rid)]
    _record(rid, viol, ["world", "store", "pin"])


@pytest.mark.live
@pytest.mark.parametrize("rid", BINDER_IDS)
def test_binder_outcome_row(rid, gate_build):
    """LNK-21: server `fx-svc` binds `.mcp.json` entry -> `scripts/Dockerfile` CMD -> unit svc. LNK-22: a URL naming Compose service `api` port 8000 binds
    to unit svc through the service's published port and its build-context mount (the Binder outcome; no stored Link at P1)."""
    w = gate_build.w
    svc = next(u.unitId for u in gate_build.units if (u.repoId, u.rootPath, u.language) == ("fx-svc", ".", "python"))
    viol = []
    anchor = w.sym_of.get("svc:.claude/agents/exec.md::<document>")
    if rid == "LNK-21":
        b = w.binder.bind({"siteId": "gate-lnk21", "kind": "mcp_tool", "anchorId": anchor, "key": "list_orders", "keyForm": "literal", "line": 1,
                           "status": "confirmed", "reason": "key-match", "hops": [], "server": "fx-svc"})
        if b is None or b.units != (svc,):
            viol.append(f"LNK-21: server fx-svc binds to {None if b is None else list(b.units)}, expected [{svc}]")
        else:
            recs = [(i, *w.records[e.entry_id]) for i, e in enumerate(b.entries)]
            viol += [f"LNK-21 {v}" for v in O.check_chain(w, recs, [("fx-svc", ".mcp.json", "mcpServers.fx-svc"), O.DOCKERFILE_ENTRY], "chain")]
    else:
        w.facts.lifted.append({"packId": "fetch", "lifting": "fetch-url", "siteId": "gate-lnk22", "symbolId": anchor, "path": "x", "line": 1, "urlPath": "a0",
                               "alternatives": ["http://api:8000/api/health"], "resolved": True, "leaf": True, "hops": [], "verbs": None,
                               "partialAlternatives": False, "weakenedBy": None, "status": "confirmed", "reason": "lifted-role"})
        try:
            b = w.binder.bind({"siteId": "gate-lnk22", "kind": "http", "anchorId": anchor, "key": "GET /api/health", "keyForm": "literal", "line": 1,
                               "status": "confirmed", "reason": "key-match", "hops": []})
        finally:
            w.facts.lifted.pop()
        if b is None or b.units != (svc,):
            viol.append(f"LNK-22: http://api:8000 binds to {None if b is None else list(b.units)}, expected [{svc}]")
        else:
            kinds = [(w.records[e.entry_id][0].get("kind"), w.records[e.entry_id][0].get("service"), w.records[e.entry_id][0].get("path")) for e in b.entries]
            if ("port", "api", "deploy/compose.yaml") not in kinds or ("mount", "api", "deploy/compose.yaml") not in kinds:
                viol.append(f"LNK-22: chain {kinds} lacks the api port 8000 and build-context mount of deploy/compose.yaml")
    _record(rid, viol, ["world"])


@pytest.mark.live
@pytest.mark.parametrize("rid", FANOUT_IDS)
def test_keyless_url_row(rid, gate_build, steady):
    """AC6-URL-n (P1 form): the keyless consumer's CONSUMES fact is `possible`/`runtime-built` (the edge reason; asserted on the pack's CONSUMES fact, the
    row `build_rows` consumes, never on the lifted fact), `build_rows` drops it for its missing key (`consumer-key-none`, so no stored CONSUMES edge exists),
    and the Links are `possible`/`unknown-key` fan-outs, in both instruments."""
    form = H.expected_form(P1[rid])
    cons = H.consumers_of(MFS, rid)
    viol = [f"[world] {v}" for v in H.fanout_violations(rid, form, cons, gate_build.recs)]
    viol += [f"[store] {v}" for v in H.fanout_violations(rid, form, cons, steady.recs)]
    w = gate_build.w
    sym_ids = {i for i, lc in w.loc_of.items() if any(H.is_under(lc, c) for c in cons)}
    facts = [f for f in w.facts.consumers if f["anchorId"] in sym_ids and f["kind"] == "http"]
    if not facts:
        lifted = [(f.get("status"), f.get("reason"), f.get("alternatives")) for f in w.facts.lifted if f["symbolId"] in sym_ids]
        keyless = sorted((w.loc_of.get(f["anchorId"], "?"), f["siteId"][:8], f["status"], f["reason"]) for f in w.facts.consumers if f["key"] is None and f["kind"] == "http")
        viol.append(f"{rid}: no http CONSUMES fact for {cons} (symbols {sorted(w.loc_of[i] for i in sym_ids)}; lifted facts {lifted}; every keyless http CONSUMES fact: {keyless})")
    for f in facts:
        if (f["key"], f["status"], f["reason"]) != (None, form["status"], form["edge"]):
            viol.append(f"{rid}: CONSUMES fact is key={f['key']!r} {f['status']}/{f['reason']}, expected keyless {form['status']}/{form['edge']}")
    sites = {f["siteId"] for f in facts}
    edges = [e for e in gate_build.rows.edges.get(("CallSite", "CONSUMES", "ContractPoint"), []) if e["s"] in sites]
    if edges:
        viol.append(f"{rid}: build_rows holds {len(edges)} CONSUMES edge(s) from the keyless site (expected none: dropped as consumer-key-none)")
    _record(rid, viol, ["world", "store"])


def _calls(store, loc_):
    repo, rest = "fx-" + loc_.split(":", 1)[0], loc_.split(":", 1)[1]
    return store.q("MATCH (s:Symbol {qname:$q, repoId:$r})-[:HAS_CALL]->(c:CallSite) OPTIONAL MATCH (c)-[k:CALLS]->(t:Symbol) "
                   "OPTIONAL MATCH (c)-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(u:Symbol) RETURN c.line, c.calleeName, collect(distinct t.qname), collect(distinct u.qname)",
                   q=rest, r=repo)


@pytest.mark.live
@pytest.mark.parametrize("rid", CALLNAME_IDS)
def test_call_edge_row(rid, steady):
    """LNK-30/39/40: the site of the consumer has no resolved `CALLS` edge and a `CALLS_NAME` candidate set that holds every planted candidate (the engine reads
    that as `possible`/`name-match`). TW-3: the site is resolved (`CALLS`) to the true target of its LNK twin. MR-PY/MR-TS: the precondition (fixtures TP-015) is a
    resolved `CALLS` edge to the frontend's wrong callee Y; without it the row is vacuous, reported BLOCKED."""
    viol = []
    row = next(iter(P1[rid].values()))
    cons = H.consumers_of(MFS, rid)[0]
    calls = _calls(steady, cons)
    if rid in ("LNK-30", "LNK-39", "LNK-40"):
        cands = H.locs_of(MFS, rid, ("candidates",))
        names = {c.rsplit(".", 1)[-1] for c in cands}
        sites = [c for c in calls if c[1] in names]
        if not sites:
            viol.append(f"{rid}: no call site named {sorted(names)} in {cons}")
        for line, name, resolved, cand in sites:
            if resolved:
                viol.append(f"{rid}: site {name}@{line} is resolved to {resolved}, expected unresolved (name-match)")
            miss = [c for c in cands if c.split(":", 1)[1] not in cand]
            if miss:
                viol.append(f"{rid}: CALLS_NAME candidate set {cand} lacks {miss}")
    elif rid == "TW-3":
        true = H.locs_of(MFS, "LNK-39", ("trueTarget",)) or [H.loc(t["repoId"], t["path"], t["qname"].split("::", 1)[-1]) for t in next(iter(P1["LNK-39"].values()))["trueTarget"]]
        sites = [c for c in calls if c[1] == "run"]
        if not sites or sites[0][2] != [true[0].split(":", 1)[1]]:
            viol.append(f"TW-3: site run resolves to {[s[2] for s in sites]}, expected [{true[0].split(':', 1)[1]}] (confirmed/resolved-call)")
    else:
        y = H.loc(row["resolvedCallee"]["repoId"], row["resolvedCallee"]["path"], row["resolvedCallee"]["qname"].split("::", 1)[-1]) if "resolvedCallee" in row else None
        if y is None:
            y = H.locs_of(MFS, rid, ("Y",))[0]
        name = (row.get("callsite") or row.get("call_site"))["name"]
        sites = [c for c in calls if c[1] == name]
        if not sites or y.split(":", 1)[1] not in sites[0][2]:
            viol.append(f"{rid}: BLOCKED (fixtures TP-015): the frontend leaves no resolved CALLS edge from {cons} site {name} to {y} "
                        f"(resolved {[s[2] for s in sites]}, CALLS_NAME candidates {[s[3] for s in sites]}), so the possible/resolution-unverified form cannot be derived")
    _record(rid, viol, ["store"])


@pytest.mark.live
@pytest.mark.parametrize("rid", LIB_IDS)
def test_library_api_row(rid, gate_build):
    """AC6-DYN-1..3 (`possible`/`dynamic-import`) and TW-1 (`confirmed`/`key-match`): a `library_api` CONSUMES fact at the consumer, in the pack output."""
    form = H.expected_form(P1[rid])
    w = gate_build.w
    cons = H.consumers_of(MFS, rid)
    ids = {i for i, lc in w.loc_of.items() if any(H.is_under(lc, c) for c in cons)}
    facts = [f for f in w.facts.consumers if f["anchorId"] in ids and f["kind"] == "library_api"]
    viol = []
    if not facts:
        viol.append(f"{rid}: no library_api CONSUMES fact for {cons} (packs run: {sorted(DEFAULT_PACKS)}; no pack in cpg/packs emits kind library_api)")
    for f in facts:
        if (f["status"], f["reason"]) != (form["status"], form["edge"]):
            viol.append(f"{rid}: library_api fact is {f['status']}/{f['reason']}, expected {form['status']}/{form['edge']}")
    _record(rid, viol, ["world"])


@pytest.mark.live
@pytest.mark.parametrize("rid", FT_IDS)
def test_field_depth_row(rid, steady):
    """FT-1 (`recv.a.b.c.d`, over the cap): the DATA edge into `ret` is `possible`/`field-truncated`. FT-2 (exactly the cap): the edge carries no downgrade."""
    form = H.expected_form(P1[rid])
    cons = H.consumers_of(MFS, rid)[0]
    rows = steady.q("MATCH (s:Symbol {qname:$q, repoId:'fx-svc'})-[:HAS_RETURN]->(:Ret)-[d:DATA]-(:Param) RETURN d.status, d.reason", q=cons.split(":", 1)[1])
    viol = []
    if not rows:
        viol.append(f"{rid}: no DATA edge from a parameter into the return of {cons}")
    for st, rs in rows:
        if rid == "FT-1" and (st, rs) != (form["status"], form["edge"]):
            viol.append(f"{rid}: DATA edge is {st}/{rs}, expected {form['status']}/{form['edge']}")
        if rid == "FT-2" and (st not in (None, "confirmed") or rs not in (None, form["edge"])):
            viol.append(f"{rid}: DATA edge is {st}/{rs}, expected confirmed (no downgrade)")
    _record(rid, viol, ["store"])


@pytest.mark.live
@pytest.mark.parametrize("rid", NC_IDS)
def test_negative_control_row(rid, gate_build, steady):
    row = next(iter(P1[rid].values()))
    viol = []
    if rid == "NC-1":
        n = steady.q("MATCH (c:CallSite {path:'docs/README.md'}) OPTIONAL MATCH (c)-[e:CONSUMES]->() OPTIONAL MATCH (c)-[v:VIA]->() RETURN count(e)+count(v)")[0][0]
        if n:
            viol.append(f"NC-1: {n} CONSUMES/VIA edge(s) from call sites of docs/README.md (a documentation snippet is not a caller)")
    elif rid == "NC-2":
        text = (gate_build.w.root / "fx-job" / "job" / "run_report.py").read_text().splitlines()
        lines = [i + 1 for i, t in enumerate(text) if t.lstrip().startswith("#") and "POST /api/v1/orders" in t]
        if not lines:
            viol.append("NC-2: the comment is gone from job/run_report.py (the control is vacuous)")
        n = steady.q("MATCH (c:CallSite {path:'job/run_report.py'})-[e:CONSUMES]->() WHERE c.line IN $ls RETURN count(e)", ls=lines)[0][0]
        if n:
            viol.append(f"NC-2: {n} CONSUMES edge(s) from the comment line(s) {lines}")
    elif rid == "NC-6":
        n = steady.q("MATCH (s:Symbol {qname:'api/items.py::build_router.read_item'})-[:HAS_CALL]->(c:CallSite)-[e:CONSUMES]->() RETURN count(e)")[0][0]
        if n:
            viol.append(f"NC-6: {n} CONSUMES edge(s) from read_item (its log string is not a call)")
    elif rid in ("NC-3", "NC-7"):
        cons = H.consumers_of(MFS, rid)[0].split(":", 1)[1]
        edge = steady.q("MATCH (s:Symbol {qname:$q})-[:HAS_CALL]->(:CallSite)-[e:CONSUMES]->(p:ContractPoint) RETURN p.keyNorm", q=cons)
        if not edge:
            viol.append(f"{rid}: no CONSUMES edge from {cons} (expected one)")
        prov = "svc:api/items.py::build_router.read_item"
        got = [r for r in H.recs_from(steady.recs, [H.consumers_of(MFS, rid)[0]]) if r.provider == prov]
        bad = [r for r in got if rid == "NC-3" or r.status == "confirmed"]
        if bad:
            viol.append(f"{rid}: Link to read_item {[(r.status, r.reason) for r in bad]} (wrong verb / missing prefix must not link{'' if rid == 'NC-3' else ' confirmed'})")
    else:  # NC-5
        c12, c19, c20 = (H.consumers_of(MFS, r) for r in ("LNK-12", "LNK-19", "LNK-20"))
        p12 = sorted({r.provider for r in H.recs_from(steady.recs, c12)})
        if p12 != sorted(H.providers_of(MFS, "LNK-12")):
            viol.append(f"NC-5: LNK-12 providers {p12}, expected exactly {H.providers_of(MFS, 'LNK-12')}")
        for r, c in (("LNK-19", c19), ("LNK-20", c20)):
            if sorted({x.provider for x in H.recs_from(steady.recs, c)}) != sorted(H.providers_of(MFS, r)):
                viol.append(f"NC-5: {r} does not list every provider of the kind ({H.providers_of(MFS, r)})")
    _record(rid, viol, ["store"])


@pytest.mark.live
@pytest.mark.parametrize("rid", SK_IDS)
def test_omitted_file_row(rid, steady):
    """SK-n in the OFF form (the runner always passes `override=False`; `jssrc2cpg` has no ignore override, so the ON column of SK-5 is not executable)."""
    exp = H.sk_expectation(P1[rid])
    skipped: dict = {}
    for repo, path, reason in steady.q("MATCH (f:SkippedFile) RETURN f.repoId, f.path, f.reason"):
        skipped.setdefault((repo, path), []).append(reason)
    files = {(r[0], r[1]) for r in steady.q("MATCH (f:File) RETURN f.repoId, f.path")}
    roots = {r[0] for r in steady.q("MATCH (u:Unit {repoId:$r}) RETURN u.rootPath", r=exp[0])}
    _record(rid, H.sk_violations(rid, exp, skipped, files, roots), ["store"])


@pytest.mark.live
@pytest.mark.parametrize("rid", ["LNK-42", "LNK-43"])
def test_target_not_mapped_row(rid, steady):
    """LNK-42/43 (SC-STEADY): the consumer's Link into the unmapped unit is `possible`/`target-not-mapped`, and a contract-point walk from the key lists the consumer."""
    form = H.expected_form(P1[rid])
    row = next(iter(P1[rid].values()))
    cons = H.consumers_of(MFS, rid)
    unit_root = row["provider"][0]["path"].split("/")[0]
    got = H.recs_from(steady.recs, cons)
    viol = []
    if not got:
        cp = steady.q("MATCH (s:Symbol {qname:$q})-[:HAS_CALL]->(c:CallSite)-[e:CONSUMES]->(p:ContractPoint) OPTIONAL MATCH (p)<-[:PROVIDES]-(x) "
                      "RETURN e.status, e.reason, p.keyNorm, count(x)", q=cons[0].split(":", 1)[1])
        viol.append(f"{rid}: no Link from {cons} (consumer CONSUMES edge, status/reason/key/providers: {cp}; units: "
                    f"{steady.q('MATCH (u:Unit {repoId:$r}) WHERE u.rootPath=$p RETURN u.buildStatus', r='fx-svc', p=unit_root)})")
    for r in got:
        if (r.status, r.reason) != (form["status"], form["link"]) or r.provider != f"unit:{unit_root}":
            viol.append(f"{rid}: Link is {r.status}/{r.reason} to {r.provider}, expected {form['status']}/{form['link']} to unit:{unit_root}")
    try:
        doc = steady.ask({"useCase": "impact", "anchor": {"kind": "contract", "key": row["keyNorm"]}})
        listed = {(i["repoId"], i["subject"]["qname"]) for i in doc["items"]}
        want = H.consumers_of(MFS, rid)[0].split(":", 1)[1]
        if ("fx-svc", want) not in listed:
            viol.append(f"{rid}: the walk from {row['keyNorm']} lists {sorted(q for _, q in listed)}, not the consumer {want}")
    except Exception as exc:  # noqa: BLE001 - a client error is the finding
        viol.append(f"{rid}: ask from contract {row['keyNorm']} failed: {type(exc).__name__}: {str(exc)[:160]}")
    _record(rid, viol, ["store"])


@pytest.mark.live
@pytest.mark.parametrize("rid", ["AB-1", "AB-2"])
def test_partial_failure_row(rid, request):
    viol = []
    if rid == "AB-2":
        steady = request.getfixturevalue("steady")
        u = steady.q("MATCH (u:Unit {repoId:'fx-svc', rootPath:'tools', language:'python'}) RETURN u.buildStatus, u.failReason")
        if not u or u[0][0] != "not-mapped":
            viol.append(f"AB-2: unit svc-tools is {u}, expected not-mapped")
        others = steady.q("MATCH (u:Unit) WHERE u.rootPath <> 'legacy' AND u.rootPath <> 'tools' AND u.buildStatus <> 'fresh' RETURN count(u)")[0][0]
        if others:
            viol.append(f"AB-2: {others} other unit(s) not fresh in SC-STEADY")
    else:
        ab1 = request.getfixturevalue("ab1")
        rows = ab1.q("MATCH (u:Unit {repoId:'fx-svc', rootPath:'legacy', language:'python'}) RETURN u.buildStatus, u.builtCommit, u.failReason, u.failedAt")
        if not rows or rows[0][0] != "stale-kept" or rows[0][1] != ab1.h1 or not rows[0][2] or not rows[0][3]:
            viol.append(f"AB-1: svc-legacy is {rows}, expected stale-kept at the previous commit {ab1.h1} with a failReason and failedAt")
        other = ab1.q("MATCH (u:Unit {repoId:'fx-svc'}) WHERE u.rootPath <> 'legacy' AND u.language = 'python' RETURN u.rootPath, u.buildStatus, u.builtCommit")
        if not other or any(r[1] != "fresh" or r[2] != ab1.h4 for r in other):
            viol.append(f"AB-1: the other fx-svc units are {other}, expected fresh at H4 {ab1.h4}")
        # the answer crossing svc-legacy: the impact of its provider symbol (the contract walk from GET /legacy/fee visits only the consumer's unit)
        doc = ab1.ask({"useCase": "impact", "anchor": {"kind": "symbol", "repo": "fx-svc", "qname": "legacy/app.py::fee"}})
        ent = [u for u in doc["staleness"]["units"] if u["rootPath"] == "legacy"]
        if not ent or ent[0]["buildStatus"] != "stale-kept" or ent[0].get("builtCommit") != ab1.h1 or not ent[0].get("reason") or not ent[0].get("failedAt"):
            viol.append(f"AB-1: the answer's staleness.units for legacy is {ent} (all: {doc['staleness']['units']}), expected stale-kept, builtCommit {ab1.h1}, reason, failedAt")
    _record(rid, viol, ["store"])


@pytest.fixture(scope="module")
def answers(steady):
    out = {}
    for name, spec in H.walk_specs(MFS).items():
        repo, qname = spec.anchor
        try:
            out[name] = steady.ask({"useCase": spec.use_case, "anchor": {"kind": "symbol", "repo": repo, "qname": qname}})
        except Exception as exc:  # noqa: BLE001
            out[name] = exc
    return out


@pytest.mark.live
@pytest.mark.parametrize("rid", WALK_IDS)
def test_walk_row(rid, answers):
    specs = H.walk_specs(MFS)
    spec = next(s for s in specs.values() if rid == s.owner or any(v[3] == rid for v in s.items.values()) or rid in s.absent.values())
    doc = answers[spec.name]
    viol = [f"{rid}: ask {spec.name} failed: {doc!r}"[:300]] if isinstance(doc, Exception) else H.walk_violations(spec, doc, rid)
    _record(rid, viol, ["store"])


@pytest.mark.live
def test_closed_world_every_link_matches_some_row_of_some_manifest(gate_build, steady):
    """A link is unplanted only if it matches no row of any manifest at any gate. Definition B (the verdict): the Link's consumer is a consumer-ish locator
    (`consumer`, `C`, `locator`) of some row; definition A (the spread): the consumer is any locator of any row. Rows gated P2+ are planted, never red."""
    anyl, cons = H.manifest_locators(MFS)
    report = {}
    for name, recs in (("world", gate_build.recs), ("store", steady.recs)):
        a = H.unplanted(recs, anyl)
        b = H.unplanted(recs, cons)
        report[name] = {"links": len(recs), "unplanted_A_links": len(a), "unplanted_B_links": len(b), "unplanted_B_consumers": sorted({r.consumer for r in b})}
    LEDGER["_closed_world"] = report
    assert all(r["unplanted_B_links"] == 0 for r in report.values()), json.dumps(report, indent=1)


def test_zz_write_the_ledger():
    out = os.environ.get("CCG_GATE_P1_OUT")
    if out:
        Path(out).write_text(json.dumps(LEDGER, indent=1, sort_keys=True, default=str))
