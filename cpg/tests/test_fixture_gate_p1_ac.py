"""P1-42: the fixture gate, part 2 — AC-level assertions on the materialized fixture system (AC-6, 9, 16, 17, 18, 21, 25).

Plan: `docs/plans/connected-code-graph-p1.md` P1-42; `docs/test-plans/connected-code-graph-p1.md` §7 (P1T-17..25); report: `docs/test-reports/connected-code-graph-p1-report.md`.
Run (cwd `cpg/`; live tier: Docker, the toolchain image and a THROWAWAY FalkorDB, never 6379, 6390 or another run's instance):

    CCG_FALKORDB_PORT=<throwaway port> CCG_CONFIG_DIR=<scratch dir with falkordb-admin.pass> CCG_REQUIRE_LIVE=1 \\
        .venv/bin/python -m pytest tests/test_fixture_gate_p1_ac.py -m "live or not live" -q -p no:cacheprovider

It reuses the dispatch C instruments (`test_fixture_gate_p1`: the SC-STEADY store, the refresh plumbing) and its manifest-derived expectations (`fixture_gate_p1_helpers`). Every verdict is
written to ``$CCG_GATE_P1_AC_OUT`` (JSON) when set. Expectations are the requirement text or manifest rows, never read from the code under test.
"""
from __future__ import annotations

import copy
import io
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

import fixture_gate_p1_ac_helpers as AC
import fixture_gate_p1_helpers as H
from ccg import answer, sysname
from ccg.engine.answer_build import build_answer
from ccg.front import Front
from ccg.store import acl
from ccg.store import loader as L
from clients import common
from clients.cli import main as cli
from clients.common import Client
from clients.mcp import server as mcp_server
from test_fixture_gate_p1 import (HOST, MFS, O, P1, Store, _refresh, live_env, runner_ready, steady)  # noqa: F401  (fixtures)

LEDGER: dict[str, dict] = {}
REPOS = ("fx-svc", "fx-web", "fx-job", "fx-lib")


def _rec(key: str, verdict: str, **info):
    LEDGER[key] = {"verdict": verdict, **info}


# ---- client plumbing --------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def creds(steady, monkeypatch):
    """The environment the MCP server and the CLI are configured by (the conftest autouse fixture repoints the credentials directory per test)."""
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(steady.env.cred))
    monkeypatch.setenv("CCG_SYSTEM", steady.system)
    monkeypatch.setenv("CCG_FALKORDB_HOST", HOST)
    monkeypatch.setenv("CCG_FALKORDB_PORT", str(steady.env.port))
    return steady


def make_client(store, *, plain: bool = False) -> Client:
    """A client of the store; ``plain`` injects the plain-language front over the fixture system's repositories (the default front knows only the default manifest's system)."""
    c = Client(store.system, host=HOST, port=store.env.port)
    if plain:
        c._front = Front(REPOS, lambda anchor, direction: build_answer(c._reader(), anchor, question=direction))
    return c


def cli_ask(factory, request=None, question=None, system=None) -> bytes:
    out, err = io.StringIO(), io.StringIO()
    argv = (["--system", system] if system else []) + ["ask"] + (["--request", json.dumps(request)] if request is not None else ["--question", question])
    code = cli.main(argv, client_factory=factory, stdout=out, stderr=err)
    assert (code, err.getvalue()) == (0, ""), (code, err.getvalue()[:300], out.getvalue()[:300])
    return out.getvalue().encode()


def mcp_ask(factory, request=None, question=None, system=None) -> bytes:
    import asyncio
    srv = mcp_server.build_server(factory)
    args = {**({"request": request} if request is not None else {"question": question}), **({"system": system} if system else {})}
    res = asyncio.run(srv.call_tool("ask", args))
    (content,) = res.content
    assert not res.isError, content.text[:300]
    return content.text.encode()


def sym(repo, qname):
    return {"useCase": "impact", "anchor": {"kind": "symbol", "repo": repo, "qname": qname}}


def spec_request(spec) -> dict:
    return {"useCase": spec.use_case, "anchor": {"kind": "symbol", "repo": spec.anchor[0], "qname": spec.anchor[1]}}


SPECS = H.walk_specs(MFS)
#: the structured requests the AC-16/18/25 legs run: the six walk asks of P1-41, two contract-point asks and a code-location ask
REQUESTS = {**{name: spec_request(s) for name, s in SPECS.items()},
            "contract:GET /api/search": {"useCase": "impact", "anchor": {"kind": "contract", "key": "GET /api/search"}},
            "contract:list_orders": {"useCase": "impact", "anchor": {"kind": "contract", "key": "list_orders"}},
            "location:core/flow.py:17": {"useCase": "impact", "anchor": {"kind": "location", "repo": "fx-svc", "path": "core/flow.py", "line": 17}}}


@pytest.fixture(scope="module")
def docs(steady):
    out = {}
    for name, req in REQUESTS.items():
        try:
            out[name] = steady.ask(req)
        except Exception as exc:  # noqa: BLE001 - a client error is the finding
            out[name] = exc
    return out


def _doc(docs, name):
    d = docs[name]
    assert not isinstance(d, Exception), f"ask {name} failed: {d!r}"[:300]
    return d


# ---- AC-6 -------------------------------------------------------------------------------------------------------------------------------
def _contract_key(rid: str) -> str:
    row = next(iter(P1[rid].values()))
    if row.get("keyNorm") or row.get("trueKeyNorm"):
        return row.get("keyNorm") or row.get("trueKeyNorm")
    if row.get("kind") == "mcp_tool" and row.get("trueTarget"):                    # a prompt mention: the tool it names is its provider's name
        return row["trueTarget"][0]["qname"].split("::")[-1]
    return "GET /api/search"          # AC6-URL-3 has no true target: any http key lists the keyless fan-out


def _shape(rid: str) -> str:
    row = next(iter(P1[rid].values()))
    return row.get("shape_syntax") or row.get("item") or rid


def test_ac6_the_p1_rows_are_exactly_the_three_kinds_and_dynamic_import_is_p2():
    assert sorted(r for r in P1 if r.startswith("AC6")) == sorted(sum(AC.AC6_KINDS.values(), []))
    allrows = H.M.holders(MFS)
    for rid in ("AC6-DYN-1", "AC6-DYN-2", "AC6-DYN-3"):
        assert H.merged_attr(allrows[rid], "gate") == "P2", rid
    for kind, floor in AC.AC6_FLOOR.items():
        assert len(AC.AC6_KINDS[kind]) == floor
        assert len({_shape(r) for r in AC.AC6_KINDS[kind]}) == floor, f"{kind}: the instances are not in distinct shapes"


@pytest.mark.live
@pytest.mark.parametrize("kind", list(AC.AC6_KINDS))
def test_ac6_per_kind_every_instance_is_possible_with_its_reason_and_none_confirmed(kind, steady):
    """P1T-25 at the answer level: an impact ask from the contract point of each planted instance lists the planted consumer as `possible` with the manifest's link reason."""
    viol, seen = [], []
    for rid in AC.AC6_KINDS[kind]:
        form = H.expected_form(P1[rid])
        doc = steady.ask({"useCase": "impact", "anchor": {"kind": "contract", "key": _contract_key(rid)}})
        cons = H.consumers_of(MFS, rid)[0]
        repo, rest = "fx-" + cons.split(":", 1)[0], cons.split(":", 1)[1]
        item = AC.find_item(doc, repo, rest)
        viol += AC.ac6_item_violations(rid, form["link"], item)
        if item:
            seen.append((rid, item["status"], item["reason"], item["distance"]))
        conf = [i["subject"].get("qname") for i in doc["items"] if i["status"] == "confirmed" and i["repoId"] == repo and i["subject"].get("qname") == rest]
        if conf:
            viol.append(f"{rid}: the planted consumer is also listed confirmed: {conf}")
    _rec(f"AC-6/{kind}", "fail" if viol else "pass", instances=len(AC.AC6_KINDS[kind]), floor=AC.AC6_FLOOR[kind], seen=seen, violations=viol)
    assert not viol, "\n".join(viol)
    assert len(seen) >= AC.AC6_FLOOR[kind]


@pytest.mark.live
def test_ac6_dynamic_import_is_not_met_at_p1_observed(steady):
    """The honest half of AC-6: no `library_api` contract point or consumer exists in the store, so dynamic import (AC6-DYN-1..3, TW-1; gate P2) is not met at P1. Recorded, not a red."""
    n_cp = steady.q("MATCH (p:ContractPoint {kind:'library_api'}) RETURN count(p)")[0][0]
    n_link = steady.q("MATCH (l:Link {kind:'library_api'}) RETURN count(l)")[0][0]
    loader_calls = steady.q("MATCH (s:Symbol {qname:'plugins/loader.py::load_plugin'})-[:HAS_CALL]->(c:CallSite)-[e:CONSUMES]->() RETURN count(e)")[0][0]
    _rec("AC-6/dynamic-import", "not met at P1", library_api_contract_points=n_cp, library_api_links=n_link, load_plugin_consumes_edges=loader_calls)
    assert (n_cp, n_link, loader_calls) == (0, 0, 0), (n_cp, n_link, loader_calls)


# ---- AC-17 ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.live
def test_ac17_a_code_location_and_a_contract_point_resolve_without_naming_component_language_or_graph(steady, docs):
    viol = []
    loc = _doc(docs, "location:core/flow.py:17")
    (a,) = loc["interpretation"]["request"]["anchors"]
    if (a["type"], a["qname"], a["repoId"]) != ("symbol", "core/flow.py::apply_discount", "fx-svc"):
        viol.append(f"location core/flow.py:17 resolved to {a}")
    web = steady.q("MATCH (s:Symbol {repoId:'fx-web'}) WHERE s.path ENDS WITH 'OrderTotal.tsx' AND s.lineStart IS NOT NULL RETURN s.path, s.lineStart, s.qname ORDER BY s.lineStart LIMIT 1")
    path, line, qn = web[0]
    r = steady.ask({"useCase": "impact", "anchor": {"kind": "location", "repo": "fx-web", "path": path, "line": line}})
    got = r["interpretation"]["request"]["anchors"][0]
    if got["repoId"] != "fx-web" or got["type"] != "symbol":
        viol.append(f"location fx-web {path}:{line} resolved to {got}")
    cp = _doc(docs, "contract:GET /api/search")
    (c,) = cp["interpretation"]["request"]["anchors"]
    if c["type"] != "contract" or c["keyNorm"] != "GET /api/search":
        viol.append(f"contract GET /api/search resolved to {c}")
    if not cp["items"]:
        viol.append("the contract-point ask lists nothing")
    _rec("AC-17/resolve", "fail" if viol else "pass", location=a["qname"], location_web=got.get("qname"), contract=c["keyNorm"], violations=viol)
    assert not viol, "\n".join(viol)


@pytest.mark.live
def test_ac17_plain_language_names_only_the_anchor(steady, creds):
    """The plain-language front (deterministic parsers, no model call) resolves `impact <file>:<line>`, `impact <file>::<symbol>` and a contract question without a repository, a language or a graph."""
    c = make_client(steady, plain=True)
    ref = steady.ask(sym("fx-svc", "core/flow.py::apply_discount"))
    viol, seen = [], {}
    for q in ("impact core/flow.py:17", "impact core/flow.py::apply_discount"):
        try:
            d = c.ask(question=q)
            seen[q] = d["interpretation"]["status"]
            if answer.facts_bytes(d) != answer.facts_bytes(ref):
                viol.append(f"{q!r}: facts differ from the structured symbol request")
        except Exception as exc:  # noqa: BLE001
            viol.append(f"{q!r}: {type(exc).__name__}: {str(exc)[:160]}")
    for q in ("impact GET /api/search",):
        try:
            d = c.ask(question=q)
            seen[q] = d["interpretation"]["status"]
            if d["interpretation"]["status"] != "resolved" or not d["items"]:
                viol.append(f"{q!r}: status {d['interpretation']['status']}, {len(d['items'])} items")
        except Exception as exc:  # noqa: BLE001
            viol.append(f"{q!r}: {type(exc).__name__}: {str(exc)[:160]}")
    _rec("AC-17/plain-language", "fail" if viol else "pass", seen=seen, violations=viol)
    assert not viol, "\n".join(viol)


@pytest.mark.live
def test_ac17_a_dts_a_pyi_and_a_vendored_file_answer_not_found_never_a_symbol(steady):
    viol, seen = [], {}
    for repo, path in (("fx-web", "app/src/types/api.d.ts"), ("fx-job", "job/stubs/fx_svc_common_1_4.pyi"), ("fx-web", "app/vendor/pkg/x.test.ts")):
        try:
            steady.ask({"useCase": "impact", "anchor": {"kind": "location", "repo": repo, "path": path, "line": 1}})
            viol.append(f"{repo}:{path}: answered (expected 'not found')")
        except common.AnchorNotResolved as exc:
            seen[path] = (exc.detail.get("status"), exc.detail.get("state"), len(exc.detail.get("candidates", [])))
            if exc.detail.get("status") != "unresolved" or exc.detail.get("candidates"):
                viol.append(f"{repo}:{path}: {exc.detail}")
        except Exception as exc:  # noqa: BLE001
            viol.append(f"{repo}:{path}: {type(exc).__name__}: {str(exc)[:160]}")
    _rec("AC-17/negatives", "fail" if viol else "pass", seen=seen, violations=viol)
    assert not viol, "\n".join(viol)


@pytest.mark.live
def test_ac17_the_other_groups_are_not_built_at_p1_observed(steady, creds):
    """AC-17 names four groups (file+line, branch, queue name, stack trace); P1 closes the code location and the contract point only (plan §12). Recorded: what the other two ask shapes answer."""
    c = make_client(steady, plain=True)
    seen = {}
    for label, q in (("stack trace", "impact Traceback (most recent call last):\n  File \"core/flow.py\", line 17, in apply_discount"), ("branch", "impact branch feature/x")):
        try:
            d = c.ask(question=q)
            seen[label] = f"answered: {d['interpretation']['status']}"
        except common.ClientError as exc:
            seen[label] = f"{type(exc).__name__}: {str(exc)[:100]}"
    _rec("AC-17/other-groups", "not built at P1", seen=seen)


# ---- AC-18 ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.live
def test_ac18_every_item_of_every_answer_shows_its_chain_with_file_and_line(docs):
    viol, counts, n_items = [], {}, 0
    for name in REQUESTS:
        d = _doc(docs, name)
        viol += AC.ac18_violations(d, name, counts)
        n_items += len(d["items"]) + sum(len(g["members"]) for g in d["groups"])
    _rec("AC-18", "fail" if viol else "pass", answers=len(REQUESTS), items=n_items, hop_kinds=counts, violations=viol[:20])
    assert n_items > 0 and not viol, "\n".join(viol[:20])


# ---- AC-25 ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.live
def test_ac25_shape_direct_indirect_distance_grouping_and_order_on_every_answer(docs):
    viol = []
    for name in REQUESTS:
        viol += AC.ac25_shape_violations(_doc(docs, name), name)
    _rec("AC-25/shape", "fail" if viol else "pass", answers=len(REQUESTS), violations=viol[:20])
    assert not viol, "\n".join(viol[:20])


@pytest.mark.live
def test_ac25_data_flow_effect_is_listed_with_distance_and_the_reverse_question_lists_dependencies_the_same_way(steady, docs):
    """The data-flow effect (a changed return value used two calls away: DF-1, DF-2, DF-4 at their P1 form) and the reverse question for the same function (`apply_discount`: its callees, listed
    in the same shape, distance 1 `confirmed` and the flow-continuation callees `possible`/`flow-unavailable:python` at 2)."""
    viol = []
    df = _doc(docs, "CF-DF")
    for rid in ("DF-1", "DF-2", "DF-4"):
        viol += H.walk_violations(SPECS["CF-DF"], df, rid)
    rev = steady.ask({"useCase": "depends-on", "anchor": {"kind": "symbol", "repo": "fx-svc", "qname": "core/flow.py::apply_discount"}})
    viol += AC.ac25_shape_violations(rev, "reverse apply_discount")
    viol += AC.ac18_violations(rev, "reverse apply_discount")
    got = {i["subject"]["qname"].split("::")[-1]: (i["distance"], i["status"], i.get("reason")) for i in rev["items"]}
    if "base_price" not in got or got["base_price"][:2] != (1, "confirmed"):
        viol.append(f"reverse apply_discount: base_price is {got.get('base_price')}, expected distance 1 confirmed")
    if not rev["items"] or rev["interpretation"]["request"]["direction"] != "depends-on":
        viol.append("the reverse ask did not run as depends-on")
    _rec("AC-25/data-flow-and-reverse", "fail" if viol else "pass", reverse_items=got, violations=viol)
    assert not viol, "\n".join(viol)


@pytest.mark.live
def test_ac25_control_dependency_half_is_not_closed_observed(docs):
    """The control half: P1 lists the callers of the function holding the changed condition (CD-2: `finalize` d1 confirmed, `notify` d2 possible/flow-unavailable) and states the flow notice;
    the guarded-code items (CD-1, CD-3, CD-4) are NOT listed. Reported not closed (stakeholder 2026-11-19; V-both)."""
    cd = _doc(docs, "CF-CD")
    viol = H.walk_violations(SPECS["CF-CD"], cd, "CD-2")
    listed = {i["subject"].get("qname", "").split("::")[-1] for i in cd["items"]}
    guarded = {"audit", "notify_slow", "slow_path"}                    # names are informational only; the manifest rows below decide
    for rid in ("CD-1", "CD-3", "CD-4"):
        viol += H.walk_violations(SPECS["CF-CD"], cd, rid)           # the P1 form says "not listed"
    notices = sorted(l for n in cd.get("notices", []) if n["code"] == "flow-not-enumerated" for l in n["langs"])
    if notices != ["python"]:
        viol.append(f"CF-CD notices {notices}, expected the python flow notice")
    _rec("AC-25/control-half", "not closed (V-both lists no guarded point)" if not viol else "fail", listed=sorted(listed), notices=notices, violations=viol)
    assert not viol, "\n".join(viol)


# ---- AC-9 -------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.live
def test_ac9_every_answer_states_the_refresh_and_the_decision_state(docs):
    viol, seen = [], set()
    for name in REQUESTS:
        d = _doc(docs, name)
        st = d["stamp"]
        seen.add((st["snapshotId"], st["verdictLogOffset"], st["verdictLogHash"]))
        if not (st.get("snapshotId") and st.get("verdictLogHash") and st.get("verdictLogOffset") == 0):
            viol.append(f"{name}: stamp {st}")
        units = d["staleness"]["units"]
        if not units or any(not (u.get("builtCommit") and u.get("builtAt") and u.get("buildStatus")) for u in units):
            viol.append(f"{name}: staleness.units lacks builtCommit/builtAt/buildStatus: {[u for u in units if not (u.get('builtCommit') and u.get('builtAt'))][:2]}")
    if len(seen) != 1:
        viol.append(f"the stamps differ across answers of one snapshot: {sorted(seen)}")
    _rec("AC-9/stamp", "fail" if viol else "pass", stamp=sorted(seen), verdictLogOffset="0 (the decision-state half is the empty log until P4)", violations=viol)
    assert not viol, "\n".join(viol)


@pytest.fixture(scope="module")
def ac9(live_env):
    """fx-svc at H1 refreshed; the same request asked twice; a rebuild from the same commits; then H4 (a changed commit)."""
    root = live_env.tmp / "root-ac9"
    O.M.materialize(root)
    svc = root / "fx-svc"
    h4 = subprocess.run(["git", "-C", str(svc), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    h1 = subprocess.run(["git", "-C", str(svc), "log", "--format=%H", "--grep", "build point: legacy builds"], capture_output=True, text=True, check=True).stdout.split()[0]
    git = {**os.environ, **O._GIT_ENV}
    subprocess.run(["git", "-C", str(svc), "checkout", "-q", "--detach", h1], check=True, env=git)
    work = live_env.tmp / "work-ac9"
    store = Store(live_env, "fxp1-ac9")
    req = sym("fx-svc", "core/flow.py::apply_discount")
    out = SimpleNamespace(h1=h1, h4=h4, steps={})
    for step, commit in (("first", None), ("rebuild", None), ("changed", h4)):
        if commit:
            subprocess.run(["git", "-C", str(svc), "checkout", "-q", "--detach", commit], check=True, env=git)
        rep = _refresh(live_env, "fxp1-ac9", root, work, scope=["fx-svc"])
        a, b = store.ask(req), store.ask(req)
        out.steps[step] = SimpleNamespace(report=rep, a=a, b=b, units=[u.status for u in rep.units])
    return out


@pytest.mark.live
def test_ac9_stamp_and_facts_are_equal_across_repeats_and_a_rebuild_and_change_with_a_changed_commit(ac9):
    s = ac9.steps
    viol = []
    if answer.facts_bytes(s["first"].a) != answer.facts_bytes(s["first"].b) or s["first"].a["stamp"] != s["first"].b["stamp"]:
        viol.append("two asks of one request on one snapshot differ")
    if s["rebuild"].a["stamp"] != s["first"].a["stamp"]:
        viol.append(f"stamp after a rebuild from the same commits differs: {s['first'].a['stamp']} vs {s['rebuild'].a['stamp']}")
    if answer.facts_bytes(s["rebuild"].a) != answer.facts_bytes(s["first"].a):
        viol.append("facts_bytes after a rebuild from the same commits differ")
    if s["changed"].a["stamp"] == s["first"].a["stamp"]:
        viol.append("stamp did not change after a rebuild from a changed commit (H1 -> H4)")
    commits = {step: sorted({u["builtCommit"] for u in s[step].a["staleness"]["units"] if u["repoId"] == "fx-svc"}) for step in s}
    _rec("AC-9/repeat-rebuild-change", "fail" if viol else "pass", snapshotIds={k: v.a["stamp"]["snapshotId"] for k, v in s.items()}, builtCommits=commits,
         rebuild_unit_statuses=sorted(set(s["rebuild"].units)), violations=viol)
    assert not viol, "\n".join(viol)


# ---- AC-16 ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.live
@pytest.mark.parametrize("name", list(REQUESTS))
def test_ac16_mcp_cli_and_engine_return_identical_facts_and_echo_the_request(name, steady, creds, docs):
    req = REQUESTS[name]
    factory = lambda system: Client(system or steady.system, host=HOST, port=steady.env.port)        # noqa: E731
    engine = _doc(docs, name)
    legs = {"engine": answer.facts_bytes(engine),
            "cli": answer.facts_bytes(answer.loads(cli_ask(factory, request=req))),
            "mcp": answer.facts_bytes(answer.loads(mcp_ask(factory, request=req)))}
    viol = [f"{name}: {leg} facts differ from the engine's" for leg, b in legs.items() if b != legs["engine"]]
    if not (engine["items"] or engine["groups"]):
        viol.append(f"{name}: an empty answer proves nothing")
    echo = engine["interpretation"]["request"]
    if echo["useCase"] != "impact" or echo["direction"] != req["useCase"]:
        viol.append(f"{name}: the echoed request is {echo}, the structured one {req}")
    _rec(f"AC-16/legs/{name}", "fail" if viol else "pass", facts_bytes_len=len(legs["engine"]), violations=viol)
    assert not viol, "\n".join(viol)


@pytest.mark.live
def test_ac16_plain_language_pin_replay_equals_the_structured_request(steady, creds, docs):
    """The question the parsers resolve (no model call) through the engine, the CLI and MCP: the same facts as the structured request, and the request it echoes, replayed as a pin, gives the same facts."""
    ref = _doc(docs, "CF-DF3")
    q = "impact core/flow.py::apply_discount"
    factory = lambda system: make_client(steady, plain=True)        # noqa: E731
    legs = {"engine": answer.facts_bytes(make_client(steady, plain=True).ask(question=q)),
            "cli": answer.facts_bytes(answer.loads(cli_ask(factory, question=q))),
            "mcp": answer.facts_bytes(answer.loads(mcp_ask(factory, question=q)))}
    viol = [f"plain-language {leg} facts differ from the structured request's" for leg, b in legs.items() if b != answer.facts_bytes(ref)]
    plain = make_client(steady, plain=True).ask(question=q)
    (a,) = plain["interpretation"]["request"]["anchors"]
    pin = {"useCase": plain["interpretation"]["request"]["useCase"], "anchor": {"kind": "symbol", "repo": a["repoId"], "qname": a["qname"]}}
    if answer.facts_bytes(steady.ask(pin)) != answer.facts_bytes(ref):
        viol.append("the echoed request replayed as a structured pin gives different facts")
    rv = make_client(steady, plain=True).ask(question="depends-on core/flow.py::base_price")
    (ra,) = rv["interpretation"]["request"]["anchors"]
    rv_pin = {"useCase": rv["interpretation"]["request"]["direction"], "anchor": {"kind": "symbol", "repo": ra["repoId"], "qname": ra["qname"]}}
    if answer.facts_bytes(steady.ask(rv_pin)) != answer.facts_bytes(rv) or answer.facts_bytes(steady.ask(rv_pin)) != answer.facts_bytes(_doc(docs, "RV-1")):
        viol.append("the depends-on echo (useCase 'impact', direction 'depends-on') replayed through its direction does not give the structured RV-1 facts")
    if "provenance" in plain["interpretation"]:
        viol.append("a parser-resolved question carries model provenance (a model call happened)")
    _rec("AC-16/plain-language", "fail" if viol else "pass", violations=viol)
    assert not viol, "\n".join(viol)


def _flow_false(doc, run, lang) -> bool:
    if not any(n["code"] == "flow-not-enumerated" and lang in n["langs"] for n in doc.get("notices", [])):
        return False
    rows = run("units", {"ids": [u["unitId"] for u in doc["staleness"]["units"]]})
    return any(r[3] == lang and r[9] is False for r in rows)


class Underivable(AssertionError):
    """The rows hold something the replay cannot match to the recipe's reads: the raw leg FAILS (never skips)."""


def replay_chains(doc: dict, raw) -> dict:
    """The raw leg beyond P1-25's call column: every consecutive hop pair of every item's chain is re-read through ``raw(cypher_text_with_params)`` from the document's OWN ``recipe.queries``
    (verbatim, the ``ccg_raw_<s>`` role) and must be returned as a row (id and, where the read carries one, the edge status); the Link hops' ``lifted`` and ``bound`` annotations are re-read by
    ``lifted_of``/``bound_of_link``; and the first level of a symbol-anchored impact ask equals the rows of ``calls_in`` plus ``name_in`` for the anchor. Returns counters."""
    queries = {q["name"]: q["cypher"] for q in doc["recipe"]["queries"]}
    cache: dict = {}

    def run(name, params):
        if name not in queries:
            raise Underivable(f"the recipe names no read {name!r} (it has {sorted(queries)})")
        key = (name, json.dumps(params, sort_keys=True))
        if key not in cache:
            cache[key] = raw(AC.cypher_with_params(queries[name], params))
        return cache[key]

    n_hops = n_link_notes = n_flow = 0
    reads_used: set = set()
    items = [*doc["items"], *(m for g in doc["groups"] for m in g["members"])]
    for it in items:
        chain = it["chain"]
        for i in range(1, len(chain)):
            prev, hop = chain[i - 1], chain[i]
            key = (hop["kind"], hop.get("edgeType"), hop.get("dir"))
            if key not in AC.HOP_READS:
                raise Underivable(f"hop {key} of item {it['subject'].get('qname')} has no recipe read the driver can replay")
            name, col, stat = AC.HOP_READS[key]
            rows = run(name, {"ids": [prev["id"]]})
            match = [r for r in rows if r[0] == prev["id"] and r[col] == hop["id"]]
            if not match:
                raise Underivable(f"hop {key} {prev['id'][:8]} -> {hop['id'][:8]} of {it['subject'].get('qname')} is not returned by {name} ({len(rows)} rows)")
            if stat is not None and hop.get("status") and all(r[stat] != hop["status"] for r in match):
                # the one engine-side rule the rows do not hold: a function-state continuation through a unit whose flow flag is false is `possible`/`flow-unavailable:<lang>`
                # (answer spec 2 §5); it reproduces from the rows only as: the stored edge is `confirmed`, the answer carries the flow notice, and a unit of that language has `flowAvailable` false
                lang = (hop.get("reason") or "").partition("flow-unavailable:")[2]
                if not (lang and hop["status"] == "possible" and all(r[stat] == "confirmed" for r in match) and _flow_false(doc, run, lang)):
                    raise Underivable(f"hop {key} {hop['id'][:8]}: {name} returns status {[r[stat] for r in match]}, the answer says {hop['status']!r}/{hop.get('reason')!r}")
                n_flow += 1
            reads_used.add(name)
            n_hops += 1
        for h in chain:
            if h["kind"] == "Link" and h.get("lifted"):
                sites = sorted({l["siteId"] for l in h["lifted"]})
                rows = run("lifted_of", {"pairs": [[h["id"], s] for s in sites]})
                if sorted((r[1], r[2]) for r in rows) != sorted((l["siteId"], l["ord"]) for l in h["lifted"]):
                    raise Underivable(f"Link {h['id'][:8]}: lifted_of returns {sorted((r[1], r[2]) for r in rows)}, the answer holds {sorted((l['siteId'], l['ord']) for l in h['lifted'])}")
                n_link_notes += 1
                reads_used.add("lifted_of")
            if h["kind"] == "Link" and h.get("bound"):
                rows = run("bound_of_link", {"ids": [h["id"]]})
                stored = {(r[1], r[3]) for r in rows}
                shown = {(b["entryId"], b["ord"]) for b in h["bound"]}
                if not shown <= stored:
                    raise Underivable(f"Link {h['id'][:8]}: bound entries {sorted(shown - stored)} are not returned by bound_of_link")
                reads_used.add("bound_of_link")
    first = None
    req = doc["interpretation"]["request"]
    (anchor,) = req["anchors"]
    if req["direction"] == "impact" and anchor["type"] == "symbol":
        rows = run("calls_in", {"ids": [anchor["symbolId"]]}) + run("name_in", {"ids": [anchor["symbolId"]]})
        raw_sites = {r[1] for r in rows}
        answer_sites = {it["chain"][1]["id"] for it in items if len(it["chain"]) > 1 and it["chain"][0]["id"] == anchor["symbolId"] and it["chain"][1]["kind"] == "CallSite"
                        and it["chain"][1].get("edgeType") in ("CALLS", "CALLS_NAME")}
        if raw_sites != answer_sites:
            raise Underivable(f"first level: calls_in+name_in return {len(raw_sites)} call sites, the answer's first hops are {len(answer_sites)}: only raw {sorted(raw_sites - answer_sites)[:3]}, only answer {sorted(answer_sites - raw_sites)[:3]}")
        first = len(raw_sites)
    return {"items": len(items), "hops_replayed": n_hops, "link_notes": n_link_notes, "flow_downgrades_reproduced": n_flow, "reads_used": sorted(reads_used), "first_level_sites": first}


@pytest.mark.live
@pytest.mark.parametrize("name", list(REQUESTS))
def test_ac16_the_raw_recipe_reproduces_the_chains_of_every_item(name, steady, creds, docs):
    """P1-42's addition to P1-25's raw leg: the shapes beyond the flag-true call column (flag-false continuation, Link/BOUND_BY/lifted hops, contract-point and depends-on asks)."""
    c = make_client(steady)

    def raw(text):
        return c.raw_query(text)["rows"]
    summary = replay_chains(_doc(docs, name), raw)
    _rec(f"AC-16/raw/{name}", "pass", **summary)


@pytest.mark.live
def test_ac16_mutants_a_recipe_missing_or_altered_read_turns_the_raw_leg_red(steady, creds, docs):
    """P1T-22's planted input: a published recipe missing one read (C3), or one whose text drops the edge status, is refused by the replay (never skipped)."""
    c = make_client(steady)
    raw = lambda text: c.raw_query(text)["rows"]        # noqa: E731
    base = _doc(docs, "CF-DF3")
    replay_chains(base, raw)                              # the unmutated recipe is green
    killed = {}
    for label in ("to_in", "via_in", "calls_in", "lifted_of", "owner_of:CallSite", "units"):
        d = copy.deepcopy(base)
        d["recipe"]["queries"] = [q for q in d["recipe"]["queries"] if q["name"] != label]
        with pytest.raises(Underivable):
            replay_chains(d, raw)
        killed[f"drop {label}"] = True
    d = copy.deepcopy(base)
    for q in d["recipe"]["queries"]:
        if q["name"] == "calls_in":
            q["cypher"] = q["cypher"].replace("MATCH (c:CallSite)-[e:CALLS]->", "MATCH (c:CallSite)-[e:CALLS_NAME]->")      # a different edge
    with pytest.raises(Underivable):
        replay_chains(d, raw)
    killed["calls_in reads another edge"] = True
    _rec("AC-16/raw-mutants", "pass", killed=killed)


# ---- AC-21 ------------------------------------------------------------------------------------------------------------------------------
def _refused(call) -> tuple[bool, str]:
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the database's refusal arrives as whatever the driver raises
        return True, f"{type(exc).__name__}: {str(exc)[:120]}"
    return False, "accepted"


def _state(store):
    snap = store.q("MATCH (s:Snapshot) RETURN s.snapshotId, s.verdictLogOffset, s.verdictLogHash")
    return snap, store.q("MATCH (n) RETURN count(n)")[0][0], store.q("MATCH ()-[e]->() RETURN count(e)")[0][0]


def ac21_probe(env, system, store) -> dict:
    """Every write shape through the three raw surfaces: refused and the view unchanged. Returns {label: (refused, text)}."""
    cr = acl.read_credentials(system)
    live = sysname.live_key(system)
    raw_db = L.connect(HOST, env.port, username=sysname.acl_user("raw", system), password=cr["raw"])
    out = {}
    writes = {"CREATE": "CREATE (:Ac21Probe {x: 1})", "MERGE": "MERGE (:Ac21Probe {x: 2})", "SET": "MATCH (u:Unit) SET u.buildStatus = 'x'",
              "DELETE": "MATCH (n:Symbol) DETACH DELETE n", "REMOVE": "MATCH (u:Unit) REMOVE u.buildStatus", "INDEX": "CREATE INDEX FOR (n:Ac21Probe) ON (n.x)"}
    for label, q in writes.items():
        out[f"GRAPH.QUERY {label}"] = _refused(lambda q=q: raw_db.execute_command("GRAPH.QUERY", live, q))
        out[f"GRAPH.PROFILE {label}"] = _refused(lambda q=q: raw_db.execute_command("GRAPH.PROFILE", live, q))
    out["GRAPH.QUERY read"] = _refused(lambda: raw_db.execute_command("GRAPH.QUERY", live, "MATCH (n) RETURN count(n)"))
    out["GRAPH.PROFILE read"] = _refused(lambda: raw_db.execute_command("GRAPH.PROFILE", live, "MATCH (n) RETURN count(n)"))
    out["GRAPH.DELETE"] = _refused(lambda: raw_db.execute_command("GRAPH.DELETE", live))
    out["DEL key"] = _refused(lambda: raw_db.execute_command("DEL", live))
    out["GRAPH.CONFIG SET"] = _refused(lambda: raw_db.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", "1"))
    out["ACL SETUSER"] = _refused(lambda: raw_db.execute_command("ACL", "SETUSER", sysname.acl_user("raw", system), "+@all"))
    out["FLUSHALL"] = _refused(lambda: raw_db.execute_command("FLUSHALL"))
    c = Client(system, host=HOST, port=env.port)
    for label, q in writes.items():
        out[f"raw_query {label}"] = _refused(lambda q=q: c.raw_query(q))
    return out


@pytest.fixture(scope="module")
def ac21(steady):
    env = steady.env
    mp = pytest.MonkeyPatch()
    mp.setenv("CCG_CREDENTIALS_DIR", str(env.cred))
    try:
        before = _state(steady)
        probe = ac21_probe(env, steady.system, steady)
        after = _state(steady)
    finally:
        mp.undo()
    return SimpleNamespace(before=before, after=after, probe=probe)


@pytest.mark.live
def test_ac21_every_write_through_raw_query_and_a_direct_raw_credential_is_refused_and_the_view_is_unchanged(ac21):
    viol = [f"{k}: {v[1]}" for k, v in ac21.probe.items() if not v[0]]
    if ac21.before != ac21.after:
        viol.append(f"the view changed: {ac21.before} -> {ac21.after}")
    _rec("AC-21/refusals", "fail" if viol else "pass", attempts=len(ac21.probe), refused=sum(1 for v in ac21.probe.values() if v[0]), stamp_before=ac21.before[0], stamp_after=ac21.after[0],
         nodes=ac21.after[1], edges=ac21.after[2], texts={k: v[1] for k, v in list(ac21.probe.items())[:3]}, violations=viol)
    assert not viol, "\n".join(viol)


@pytest.mark.live
def test_ac21_surfaces_mcp_and_cli_refuse_a_write_and_the_positive_controls_hold(steady, creds):
    """The raw_query tool (MCP) and `raw` (CLI) refuse a write with a typed error; a raw read succeeds (the credential works, it is not merely locked out); the same CREATE under the build credential
    succeeds on a scratch build key (the statement is valid and the throwaway permits writes: the refusal above is the raw role's, not a typo)."""
    import asyncio
    viol = []
    factory = lambda system: Client(system or steady.system, host=HOST, port=steady.env.port)        # noqa: E731
    srv = mcp_server.build_server(factory)
    res = asyncio.run(srv.call_tool("raw_query", {"cypher": "CREATE (:Ac21Probe)"}))
    if not res.isError:
        viol.append("MCP raw_query accepted a write")
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["--system", steady.system, "raw", "--cypher", "CREATE (:Ac21Probe)"], client_factory=factory, stdout=out, stderr=err)
    if code == 0:
        viol.append("CLI raw accepted a write")
    read = Client(steady.system, host=HOST, port=steady.env.port).raw_query("MATCH (n:Symbol) RETURN count(n)")
    if not read["rows"] or not read["rows"][0][0]:
        viol.append(f"raw read control returned {read['rows']}")
    cr = acl.read_credentials(steady.system)
    bdb = L.connect(HOST, steady.env.port, username=sysname.acl_user("build", steady.system), password=cr["build"])
    scratch = sysname.build_key(steady.system, "ac21ctl")
    try:
        bdb.execute_command("GRAPH.QUERY", scratch, "CREATE (:Ac21Probe {x: 1})")
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok, why = False, f"{type(exc).__name__}: {str(exc)[:120]}"
        viol.append(f"positive control: the build credential's CREATE on a scratch build key failed: {why}")
    finally:
        try:
            steady.env.admin.connection.delete(scratch)
        except Exception:  # noqa: BLE001
            pass
    _rec("AC-21/surfaces-and-controls", "fail" if viol else "pass", mcp_is_error=bool(res.isError), cli_exit=code, cli_stderr=err.getvalue()[:160], raw_read_rows=read["rows"][:1], build_write_control=ok, violations=viol)
    assert not viol, "\n".join(viol)


@pytest.mark.live
def test_ac21_mutant_a_raw_role_granted_graph_query_turns_the_refusal_probe_red(live_env, tmp_path_factory):
    """The probe's own mutant: on a scratch system, grant the raw user `+graph.query` and the probe reports the write as accepted (so the green verdict above is not vacuous)."""
    system = "fxp1-ac21m"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CCG_CREDENTIALS_DIR", str(live_env.cred))
        acl.create_users(live_env.admin, system)
        live_env.systems.append(system)
        cr = acl.read_credentials(system)
        live = sysname.live_key(system)
        raw_db = L.connect(HOST, live_env.port, username=sysname.acl_user("raw", system), password=cr["raw"])
        before = _refused(lambda: raw_db.execute_command("GRAPH.QUERY", live, "CREATE (:Ac21Probe)"))
        live_env.admin.connection.execute_command("ACL", "SETUSER", sysname.acl_user("raw", system), "+graph.query")
        after = _refused(lambda: raw_db.execute_command("GRAPH.QUERY", live, "CREATE (:Ac21Probe)"))
    _rec("AC-21/mutant", "pass" if (before[0], after[0]) == (True, False) else "fail", before=before, after=after)
    assert (before[0], after[0]) == (True, False), (before, after)


# ---- offline: the checkers' own mutants ---------------------------------------------------------------------------------------------------
def _item(**kw):
    base = {"subject": {"type": "symbol", "symbolId": "s1", "qname": "a.py::f", "path": "a.py", "lineStart": 3, "lineEnd": 9}, "repoId": "r", "unitId": "u", "status": "possible", "reason": "unknown-key",
            "directness": "direct", "distance": 1, "chain": [{"kind": "Symbol", "id": "s0", "path": "b.py", "line": 2, "edgeType": None},
                                                             {"kind": "CallSite", "id": "c1", "path": "a.py", "line": 5, "edgeType": "CALLS", "dir": "<-"},
                                                             {"kind": "Symbol", "id": "s1", "path": "a.py", "line": 3, "edgeType": "HAS_CALL", "dir": "<-"}]}
    base.update(kw)
    return base


def _doc_of(*items):
    return {"interpretation": {"request": {"anchors": [{"type": "symbol", "symbolId": "s0"}]}}, "items": list(items), "groups": [], "depthBound": {"bound": 2},
            "staleness": {"units": [{"unitId": "u", "rootPath": "."}]}}


def test_ac6_checker_mutants():
    assert AC.ac6_item_violations("X", "unknown-key", _item()) == []
    assert AC.ac6_item_violations("X", "unknown-key", _item(status="confirmed"))
    assert AC.ac6_item_violations("X", "unknown-key", _item(reason="flow-unavailable:python"))
    assert AC.ac6_item_violations("X", "prompt-mention", _item())
    assert AC.ac6_item_violations("X", "unknown-key", None)


def test_ac18_checker_mutants():
    good = _doc_of(_item())
    assert AC.ac18_violations(good, "g") == []
    for mutate in (lambda it: it["chain"][1].pop("line"), lambda it: it["chain"][1].update(path=""), lambda it: it["chain"].pop(), lambda it: it["chain"][0].update(id="zz"),
                   lambda it: it["chain"][1].pop("edgeType"), lambda it: it.update(chain=[])):
        d = copy.deepcopy(good)
        mutate(d["items"][0])
        assert AC.ac18_violations(d, "m"), mutate
    link = _item()
    link["chain"].insert(2, {"kind": "Link", "id": "l1", "edgeType": "TO", "dir": "<-", "lifted": [{"ord": 0, "siteId": "c1", "path": "a.py"}]})
    assert AC.ac18_violations(_doc_of(link), "l")        # a lifted entry without a line


def test_ac25_checker_mutants():
    ok = _doc_of(_item(), _item(distance=2, directness="indirect", status="possible"))
    assert AC.ac25_shape_violations(ok, "g") == []
    for mutate in (lambda d: d["items"][0].update(directness="indirect"), lambda d: d["items"][1].update(directness="direct"), lambda d: d["items"][1].update(distance=3),
                   lambda d: d["items"][0].update(directness=None), lambda d: d["items"].reverse(), lambda d: d["items"][0].update(reason=None)):
        d = copy.deepcopy(ok)
        mutate(d)
        assert AC.ac25_shape_violations(d, "m"), mutate
    d = copy.deepcopy(ok)
    d["items"][1].update(distance=1, directness="direct", status="confirmed")        # confirmed after possible at the same distance
    assert AC.ac25_shape_violations(d, "order")


def test_the_replay_refuses_what_it_cannot_derive_and_a_wrong_hop():
    doc = {"interpretation": {"request": {"direction": "impact", "anchors": [{"type": "symbol", "symbolId": "s0"}]}}, "groups": [],
           "recipe": {"queries": [{"name": "calls_in", "cypher": "X"}, {"name": "owner_of:CallSite", "cypher": "Y"}, {"name": "name_in", "cypher": "Z"}]},
           "items": [_item()]}
    rows = {"X": [["s0", "c1", 0, 0, "a.py", 5, "f", "confirmed", "resolved-call"]], "Y": [["c1", "s1"]], "Z": []}

    def raw(text):
        return rows[text.rsplit(" ", 1)[-1]]
    out = replay_chains(doc, raw)
    assert out["hops_replayed"] == 2 and out["first_level_sites"] == 1
    bad = copy.deepcopy(doc)
    bad["items"][0]["chain"][1]["id"] = "other"
    with pytest.raises(Underivable):
        replay_chains(bad, raw)
    wrong_status = copy.deepcopy(doc)
    wrong_status["items"][0]["chain"][1]["status"] = "possible"
    with pytest.raises(Underivable, match="status"):
        replay_chains(wrong_status, raw)
    unknown = copy.deepcopy(doc)
    unknown["items"][0]["chain"][1]["edgeType"] = "WEIRD"
    with pytest.raises(Underivable, match="no recipe read"):
        replay_chains(unknown, raw)


def test_zz_write_the_ledger():
    out = os.environ.get("CCG_GATE_P1_AC_OUT")
    if out:
        from pathlib import Path
        Path(out).write_text(json.dumps(LEDGER, indent=1, sort_keys=True, default=str))
