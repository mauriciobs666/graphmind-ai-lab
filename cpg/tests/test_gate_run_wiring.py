"""P1-32b run wiring (``gate_run.py``): the engine-ask adapter and the ``run`` verb. Offline and synthetic: a fake ``ask``, a fake refresh,
throwaway git repos and an in-memory snapshot. Nothing here is held-out data and the run verb is never pointed at the real lists."""
import json
import types

import pytest

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import (FakeClient, Env, GATE, OBS, many_world, resolution_of, spec_key)  # noqa: F401

import gate_run as R
from clients.common import AnchorNotResolved, StoreError


# ------------------------------------------------------------------------------------------------ the ask adapter
def test_contract_request_is_issued_in_the_cli_shape():
    w = H.World()
    c = FakeClient(w.answer())
    ans = R.make_ask(c)(G.build_request(w.rec, w.cp))
    assert c.calls == [{"useCase": "impact", "anchor": {"kind": "contract", "key": "GET /a", "contractKind": "http"}}]
    assert ans == w.answer()                                  # the full answer JSON, untouched


def test_symbol_and_config_requests_are_issued_in_the_cli_shape():
    c = FakeClient({})
    ask = R.make_ask(c)
    s = H.item(1, H.sym("svc/a.py::f"), H.sym("svc/b.py::g"))
    k = H.item(2, H.cfg("compose.yaml", "services.s"), H.sym("svc/b.py::g"))
    ask(G.build_request(s, s["anchorId"]))
    ask(G.build_request(k, k["anchorId"]))
    assert c.calls == [{"useCase": "impact", "anchor": {"kind": "location", "repo": "r", "path": "svc/a.py", "symbol": "svc/a.py::f"}},
                       {"useCase": "impact", "anchor": {"kind": "config", "repo": "r", "path": "compose.yaml", "keyPath": "services.s"}}]


def test_the_cli_request_passes_the_clients_own_parser():
    from clients import common
    for rec in (H.item(1, H.sym("a.py::f"), H.sym("b.py::g")), H.item(2, H.cfg("c.yaml", "k"), H.sym("b.py::g")), H.World().rec):
        cli = R.cli_request(G.build_request(rec, rec["anchorId"]))
        assert common.parse_request(cli)[0] == "impact"
        need, may = common._ANCHOR_KEYS[cli["anchor"]["kind"]]
        keys = set(cli["anchor"]) - {"kind"}
        assert need <= keys and not keys - need - may


def test_unit_anchor_is_asked_through_run_asks_from_its_probe_contract():
    c1, c2 = H.contract("mcp_tool", "alpha"), H.contract("mcp_tool", "beta")
    unit = {"type": "unit", "repoId": "r", "rootPath": "svc", "lang": "python"}
    rec = H.item(1, unit, H.cfg("compose.yaml", "services.s"), serves=[c1, c2])
    other = H.item(2, c1, H.sym("a.py::f"))                   # alpha has one item, beta none: the probe is beta
    snap = H.DictSnapshot()
    snap.add_node("ContractPoint", G._mk.contract_id(c2))
    c = FakeClient({})
    G.run_asks([rec, other], snap, R.make_ask(c), lambda: 0, probes={rec["anchorId"]: G._mk.contract_id(c2)})
    assert c.calls == [{"useCase": "impact", "anchor": {"kind": "contract", "key": "beta", "contractKind": "mcp_tool"}}]


def test_an_anchor_absent_from_the_graph_that_does_not_resolve_becomes_a_not_resolved_answer_that_scores_as_a_miss_not_a_void():
    w = H.World()
    exc = AnchorNotResolved("x", status="ambiguous", detail={"status": "ambiguous", "candidates": [{}, {}]})
    ask = R.make_ask(FakeClient(exc=exc), stamp=H.stamp(), in_graph=lambda req: False)
    run = G.run_asks([w.rec], w.snap, ask, lambda: 0)
    ans = run["answers"][w.cp]["answer"]
    assert ans["interpretation"]["status"] == "ambiguous" and ans["synthesized"] == "anchor-ambiguous"
    res = G.score_gate([w.rec], run, w.snap, ())
    assert res["items"][w.rec["itemId"]]["outcome"] == "miss" and res["items"][w.rec["itemId"]]["reason"] == "not-resolved"
    assert res["stops"] == {}


def test_a_store_failure_is_not_swallowed_it_is_the_engines_error_to_record():
    w = H.World()
    with pytest.raises(R.EngineAskError) as ei:
        R.make_ask(FakeClient(exc=StoreError("down")), stamp=H.stamp())(G.build_request(w.rec, w.cp))
    assert isinstance(ei.value.__cause__, StoreError) and "down" not in str(ei.value)


def test_a_connection_error_from_the_engine_is_not_wrapped_it_stays_transient():
    w = H.World()
    with pytest.raises(ConnectionError):
        R.make_ask(FakeClient(exc=ConnectionError("c")), stamp=H.stamp())(G.build_request(w.rec, w.cp))


def test_an_anchor_in_the_graph_that_does_not_resolve_is_a_harness_error_never_a_synthesized_document():
    w = H.World()
    exc = AnchorNotResolved("x", status="unresolved", detail={"status": "unresolved", "candidates": []})
    for kw in ({"in_graph": lambda req: True}, {}):                    # the default is the strict reading
        with pytest.raises(G.HarnessError):
            R.make_ask(FakeClient(exc=exc), stamp=H.stamp(), **kw)(G.build_request(w.rec, w.cp))


def test_a_config_key_path_is_asked_in_the_form_the_build_stores():
    long_seg = "x" * 30
    k = H.item(2, H.cfg("compose.yaml", f"services.{long_seg}.env"), H.sym("svc/b.py::g"))
    cli = R.cli_request(G.build_request(k, k["anchorId"]))
    assert cli["anchor"]["keyPath"] == "services.[#\u2026].env"           # the loader's MASK ``[#…]``, not ``answer.mask_identifier``'s ``[#]``


# ------------------------------------------------------------------------------------------------ the run verb
def test_run_writes_the_run_json_that_score_reads_and_returns_counts_only(tmp_path):
    e = Env(tmp_path)
    out = e.go()
    assert out == {"records": 1, "asked": 1, "label": "first-contact"}
    run = json.loads(e.run_files()[0].read_text())
    res = G.score_gate([e.w.rec], run, e.w.snap, ())
    assert res["items"][e.w.rec["itemId"]]["outcome"] == "hit"
    assert [ev[0] for ev in e.events] == ["refresh", "resolve", "ask"]


def test_run_resolves_every_anchor_before_the_marker_and_asks_only_after_it(tmp_path):
    e = Env(tmp_path)
    many_world(e, 3)
    e.go()
    assert len(e.resolved) == 3
    assert [ev for ev in e.events if ev[0] == "ask"] == [("ask", True)] * 3
    assert all(ev == ("resolve",) or ev[0] != "resolve" for ev in e.events) and [ev[0] for ev in e.events][:4] == ["refresh", "resolve", "resolve", "resolve"]


def test_run_builds_the_refresh_target_at_eval_commit_under_the_custodian_root(tmp_path):
    e = Env(tmp_path)
    e.go()
    (_, built), = [ev for ev in e.events if ev[0] == "refresh"]
    assert built.startswith(str(e.cust.resolve())) or built.startswith(str(e.cust))
    assert H.git(e.repo, "-C", built, "rev-parse", "HEAD").strip() == e.eval_commit


def test_run_takes_the_first_run_marker_and_a_second_run_is_refused_before_any_build(tmp_path):
    e = Env(tmp_path)
    e.go()
    assert e.markers()
    e.events.clear()
    with pytest.raises(G.Refusal, match="already run"):
        e.go()
    assert e.events == []


def test_the_marker_file_is_named_by_the_full_canonical_list_key(tmp_path):
    e = Env(tmp_path)
    e.go()
    (m,) = e.markers()
    assert m.name == f"gate-run-{G.list_key([e.w.rec])}.marker" and len(G.list_key([e.w.rec])) == 64


@pytest.mark.parametrize("variant", ["trailing-blank", "key-order", "duplicate", "swap-order"])
def test_the_same_records_in_other_bytes_are_the_same_list_and_the_rerun_is_refused(tmp_path, variant):
    e = Env(tmp_path)
    second = H.item(2, H.contract("http", "GET /b"), H.sym("svc/other.py::g"))
    e.snap.add_node("ContractPoint", second["anchorId"])
    e.set_records([e.w.rec, second])
    e.go()
    lists = [e.lists]
    if variant == "trailing-blank":
        e.lists.write_text(e.lists.read_text() + "\n\n")
    elif variant == "key-order":
        e.lists.write_text("".join(json.dumps(json.loads(l), sort_keys=False, separators=(",", ":")) + "\n" for l in reversed(e.lists.read_text().splitlines())))
    elif variant == "duplicate":
        e.lists.write_text(e.lists.read_text() + json.dumps(e.w.rec) + "\n")
    else:
        a, b = e.cust / "a.jsonl", e.cust / "b.jsonl"
        a.write_text(json.dumps(e.w.rec) + "\n")
        b.write_text(json.dumps(second) + "\n")
        lists = [b, a]
    e.events.clear()
    with pytest.raises(G.Refusal, match="already run"):
        e.go(lists=lists)
    assert e.events == []


def test_run_refuses_a_dirty_cpg_tree_before_any_build_or_ask(tmp_path):
    e = Env(tmp_path)
    (e.repo / "cpg/ccg/a.py").write_text("A = 5\n")
    with pytest.raises(G.Refusal, match="first-contact"):
        e.go()
    assert e.events == [] and not e.markers() and not e.run_files()


def test_run_refuses_when_a_prereg_field_differs(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="first-contact"):
        e.go(observed={**OBS, "defaultDepthBound": 3})
    assert e.events == []


def test_run_refuses_without_the_committed_prereg_record(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="preregistration"):
        e.go(gate_id="ac5-gate-nope")
    assert e.events == []


def test_run_refuses_a_custodian_root_outside_the_allowed_area(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="custodian"):
        e.go(custodian=tmp_path / "elsewhere")
    assert e.events == [] and not (tmp_path / "elsewhere").exists()


def test_run_refuses_held_out_lists_read_from_outside_the_custodian_area(tmp_path):
    e = Env(tmp_path)
    stray = tmp_path / "stray.jsonl"
    stray.write_text(json.dumps(e.w.rec) + "\n")
    with pytest.raises(G.Refusal, match="custodian"):
        e.go(lists=[stray])
    assert e.events == []


def test_run_refuses_a_non_empty_cache_and_still_asks_nothing(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="cache is not empty"):
        e.go(cache_len=lambda: 2)
    assert ("ask", True) not in e.events and not e.markers()


def test_a_new_list_is_not_blocked_by_an_earlier_run_and_keeps_the_earlier_run_file(tmp_path):
    e = Env(tmp_path)
    e.go()
    first = e.run_files()[0].read_text()
    second = H.item(2, H.contract("http", "GET /b"), H.sym("svc/other.py::g"))
    e.snap.add_node("ContractPoint", second["anchorId"])
    e.set_records([second])
    e.go()
    assert len(e.markers()) == 2 and len(e.run_files()) == 2
    assert first in [f.read_text() for f in e.run_files()]


def test_run_refuses_a_run_when_stored_gold_does_not_renormalize(tmp_path):
    e = Env(tmp_path)
    bad = {**e.w.rec, "anchorId": "f" * 32}
    e.lists.write_text(json.dumps(bad) + "\n")
    with pytest.raises(G.Refusal, match="renormalize"):
        e.go()
    assert e.events == []


# --- M5/M2: the preflight block, the label, the probe map, the flow stratum
def test_run_json_carries_the_preflight_block_the_label_the_probes_and_the_withheld_counts(tmp_path):
    e = Env(tmp_path)
    e.go()
    run = json.loads(e.run_files()[0].read_text())
    ctx = G.require_run_context(run)
    assert ctx["label"] == "first-contact" and ctx["recordCommit"] and set(ctx["harnessPin"]["sha256"]) == set(G.HARNESS_FILES) - {
        r for r in G.HARNESS_FILES if not (e.repo / r).is_file()}
    assert run["withheld"] == {} and run["flowFlag"] is False
    assert e.marker()["pins"]["recordCommit"] == ctx["recordCommit"]


def _edit_harness_after_the_record(e):
    f = e.repo / G.HARNESS_FILES[0]
    f.write_text(f.read_text() + "# adapted\n")
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "adapt the harness")


def test_run_refuses_an_after_adaptation_label_unless_explicitly_allowed_and_then_records_it(tmp_path):
    e = Env(tmp_path)
    _edit_harness_after_the_record(e)
    with pytest.raises(G.Refusal, match="after-adaptation"):
        e.go()
    assert e.events == [] and not e.markers()
    out = e.go(allow_after_adaptation=True)
    assert out["label"] == "after-adaptation"
    run = json.loads(e.run_files()[0].read_text())
    assert run["preflight"]["label"] == "after-adaptation" and run["preflight"]["afterAdaptationAllowed"] is True
    assert e.marker()["label"] == "after-adaptation"


def test_run_refuses_when_the_record_commit_is_not_found(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal):
        e.go(record_commit_sha="0" * 40)
    assert e.events == [] and not e.markers()


def test_run_refuses_before_the_build_when_the_probe_map_does_not_exist_and_says_so_with_counts_only(tmp_path):
    e = Env(tmp_path)
    e.probes.unlink()
    with pytest.raises(G.Refusal, match="probe map") as exc:
        e.go()
    assert e.events == [] and not e.markers()
    assert str(e.cust) not in str(exc.value) and "custodian unit" in str(exc.value)


def test_a_probe_map_path_outside_custody_that_does_not_exist_is_a_custody_refusal_not_a_missing_file(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="must be under the custodian area"):
        e.go(probe_map=tmp_path / "nowhere.json")


def test_a_refusal_that_already_carries_the_prefix_is_printed_with_one_prefix(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    monkeypatch.setenv("HOME", str(e.home))
    monkeypatch.setattr(R.G, "read_probe_map", lambda *a, **k: (_ for _ in ()).throw(G.Refusal("REFUSED: the probe map is odd")))
    assert R.main(e.argv(), deps=e.deps()) == 2 and capsys.readouterr().err.count("REFUSED") == 1


def test_run_refuses_a_probe_map_outside_the_custodian_area(tmp_path):
    e = Env(tmp_path)
    stray = tmp_path / "probes.json"
    stray.write_text("{}")
    with pytest.raises(G.Refusal, match="custodian"):
        e.go(probe_map=stray)
    assert e.events == []


def test_a_unit_item_is_asked_and_resolved_from_its_s47_probe_and_the_map_is_kept_in_run_json(tmp_path):
    e = Env(tmp_path)
    c1, c2 = H.contract("mcp_tool", "alpha"), H.contract("mcp_tool", "beta")
    unit = {"type": "unit", "repoId": "r", "rootPath": "svc", "lang": "python"}
    rec = H.item(1, unit, H.cfg("compose.yaml", "services.s"), serves=[c1, c2], stratum="scripts")
    probe = G._mk.contract_id(c1)                                     # S47 picks alpha: NOT the record-local rule's beta (its own item count is 0)
    e.set_probes({rec["anchorId"]: probe})
    e.set_records([rec, H.item(2, c2, H.sym("a.py::f"), stratum="mcp")])
    e.snap.add_node("ContractPoint", probe)
    e.snap.add_node("ContractPoint", G._mk.contract_id(c2))
    e.go()
    run = json.loads(e.run_files()[0].read_text())
    assert run["probes"] == {rec["anchorId"]: probe} and probe in run["answers"]
    assert e.asked.count(probe) == 1 and spec_key({"kind": "contract", "key": "alpha", "contractKind": "mcp_tool"}) in [spec_key(s) for s in e.resolved]


def test_a_unit_item_without_a_probe_entry_is_refused_before_the_marker(tmp_path):
    e = Env(tmp_path)
    unit = {"type": "unit", "repoId": "r", "rootPath": "svc", "lang": "python"}
    rec = H.item(1, unit, H.cfg("compose.yaml", "services.s"), serves=[H.contract("mcp_tool", "alpha")], stratum="scripts")
    e.set_records([e.w.rec])                                          # the table cannot be built for it: write the list by hand
    e.lists.write_text(json.dumps(rec) + "\n")
    with pytest.raises(G.HarnessError, match="probe"):
        e.go()
    assert not e.markers() and not e.asked


def test_flow_items_are_withheld_before_the_resolve_dry_run_and_before_asking_while_the_flag_is_false(tmp_path):
    e = Env(tmp_path)
    flow = H.item(2, H.sym("svc/f.py::fl"), H.sym("svc/g.py::gl"), stratum="flow")
    e.set_records([e.w.rec, flow])
    e.snap.add_symbol(flow["anchorId"], "svc/f.py::fl", 1, 9)          # in the graph: it WOULD be resolved and asked if it were not withheld
    e.go()
    assert len(e.resolved) == 1 and e.asked == [e.w.cp]
    run = json.loads(e.run_files()[0].read_text())
    assert run["withheld"] == {"flow": 1} and run["flowFlag"] is False and set(run["answers"]) == {e.w.cp}
    assert flow["itemId"] not in json.dumps(run) and flow["anchorId"] not in json.dumps(run)


def test_wiring_items_are_withheld_before_resolve_and_asking_and_counted_whatever_the_flow_flag(tmp_path):
    """CO-5: the stakeholder withheld wiring; it is neither resolved nor asked nor scored, and a true flow flag does not gate it."""
    e = Env(tmp_path)
    wiring = H.item(2, H.sym("svc/f.py::wi"), H.cfg("compose.yaml", "services.s"), stratum="wiring")
    e.set_records([e.w.rec, wiring])
    e.snap.add_symbol(wiring["anchorId"], "svc/f.py::wi", 1, 9)        # in the graph: it WOULD be resolved and asked if it were not withheld
    e.go()
    assert len(e.resolved) == 1 and e.asked == [e.w.cp]
    run = json.loads(e.run_files()[0].read_text())
    assert run["withheld"] == {"wiring": 1} and set(run["answers"]) == {e.w.cp}
    assert wiring["itemId"] not in json.dumps(run) and wiring["anchorId"] not in json.dumps(run)


def test_a_true_flow_flag_gates_flow_but_still_withholds_wiring():
    assert G.gated_strata(True) == ("http", "mcp", "scripts", "flow") and "wiring" not in G.gated_strata(True)


# --- ruling (a): the pre-marker resolve dry-run
def _planted(tmp_path, kind):
    e = Env(tmp_path)
    (spec,) = e.table
    cand = {"type": "contract", "cpId": e.w.cp}
    e.overrides[spec] = {"two": {"status": "ambiguous", "candidates": [cand, {**cand, "cpId": "e" * 32}]},
                         "resolved-two": {"status": "resolved", "candidates": [cand, {**cand, "cpId": "e" * 32}]},
                         "none": {"status": "unresolved", "candidates": []},
                         "other": {"status": "resolved", "candidates": [{**cand, "cpId": "e" * 32}]}}[kind]
    return e


@pytest.mark.parametrize("kind,status", [("two", "ambiguous"), ("resolved-two", "ambiguous"), ("none", "none"), ("other", "id-mismatch")])
def test_a_resolve_that_is_not_exactly_one_candidate_with_the_asked_id_refuses_before_the_marker_with_counts_only(tmp_path, kind, status):
    e = _planted(tmp_path, kind)
    with pytest.raises(R.ResolveRefusal) as exc:
        e.go()
    assert exc.value.counts[status] == 1 and exc.value.counts["total"] == 1
    for secret in (e.w.cp, "e" * 32, "GET /a", e.w.rec["itemId"]):
        assert secret not in str(exc.value)
    assert not e.markers() and not e.asked and not e.run_files()


def test_a_refused_resolve_does_not_spend_the_list(tmp_path):
    e = _planted(tmp_path, "none")
    with pytest.raises(R.ResolveRefusal):
        e.go()
    e.overrides.clear()
    assert e.go()["asked"] == 1


def test_the_cli_exits_with_a_distinct_code_and_counts_only_when_the_resolve_dry_run_refuses(tmp_path, monkeypatch, capsys):
    e = _planted(tmp_path, "two")
    monkeypatch.setenv("HOME", str(e.home))
    rc = R.main(e.argv("--probe-map", str(e.probes)), deps=e.deps())
    cap = capsys.readouterr()
    assert rc == R.EXIT_RESOLVE == 5 and cap.out == ""
    assert "ambiguous=1" in cap.err and cap.err.count("REFUSED") == 1
    for secret in (e.w.cp, "e" * 32, "GET /a", "SENTINEL", e.w.rec["itemId"]):
        assert secret not in cap.err


def test_a_contract_anchor_absent_from_the_graph_is_neither_resolved_nor_asked_and_keeps_its_miss(tmp_path):
    e = Env(tmp_path)
    e.snap.nodes.discard(("ContractPoint", e.w.cp))
    e.go()
    assert e.resolved == [] and e.asked == []
    run = json.loads(e.run_files()[0].read_text())
    assert G.score_gate([e.w.rec], run, e.snap, ())["items"][e.w.rec["itemId"]]["reason"] == "anchor-not-in-graph"


def test_an_engine_snapshot_id_that_differs_from_the_snapshot_reader_refuses_before_the_marker(tmp_path):
    e = Env(tmp_path)
    e.engine_sid = "b" * 32
    with pytest.raises(G.Refusal, match="snapshot"):
        e.go()
    assert not e.markers() and not e.asked


# --- B1: the build
def test_a_leftover_build_directory_does_not_block_the_retry(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        e.go(refresh_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))   # the build fails after the clone: nothing is spent
    assert not e.markers() and list(e.cust.glob("gate-build-*"))
    assert e.go()["asked"] == 1                                                    # the retry cleans the leftover and builds again
    assert [ev[0] for ev in e.events if ev[0] == "refresh"] == ["refresh"]


def test_only_a_gate_build_directory_of_this_list_under_the_custodian_root_is_ever_cleaned(tmp_path):
    e = Env(tmp_path)
    keep = e.cust / "gate-build-other"
    keep.mkdir()
    (keep / "f").write_text("x")
    mine = e.cust / f"gate-build-{G.list_key([e.w.rec])}"
    mine.mkdir()
    (mine / "stale").write_text("x")
    e.go()
    assert (keep / "f").exists() and not (mine / "stale").exists()


def test_the_default_deps_build_a_real_refresh_target_without_any_injection(tmp_path, monkeypatch):
    seen = {}
    refresh_mod = types.SimpleNamespace(
        Target=lambda store, build_conn, verdict_conn: types.SimpleNamespace(store=store, build_conn=build_conn, verdict_conn=verdict_conn),
        refresh=lambda system, **kw: seen.update(system=system, **kw))
    conns = []

    def connect(host, port, *, username, password, **kw):
        conns.append((host, port, username))
        return types.SimpleNamespace(user=username)
    a = types.SimpleNamespace(host="127.0.0.1", port=16421, graph=None, system="gt1")
    deps = R._default_deps(a, connect=connect, credential=lambda system, role: f"pw-{role}", refresh_module=refresh_mod)
    dest = tmp_path / "clone"
    (dest / "cpg/ccg").mkdir(parents=True)
    (dest / "cpg/ccg/system.yaml").write_text("x")
    deps["refresh_fn"]("gt1", repo_root=dest, manifest=tmp_path / "head-manifest.yaml")
    assert seen["system"] == "gt1" and seen["repo_root"] == dest and seen["manifest"] == tmp_path / "head-manifest.yaml"      # the caller's (the working tree's), never the clone's (D2)
    t = seen["target"]
    assert t.build_conn.user == "ccg_build_gt1" and t.verdict_conn.user == "ccg_verdict_gt1" and t.store.db is t.build_conn
    assert {c[:2] for c in conns} == {("127.0.0.1", 16421)}


@pytest.mark.parametrize("port", [None, 6379, 6390, 16390, 16400])
def test_the_default_deps_refuse_a_missing_or_forbidden_store_port_before_any_connection(port):
    a = types.SimpleNamespace(host="127.0.0.1", port=port, graph=None, system="gt1")
    with pytest.raises(G.Refusal):
        R._default_deps(a, connect=lambda *x, **k: pytest.fail("connected"))


def test_the_default_deps_refuse_a_non_loopback_host():
    a = types.SimpleNamespace(host="10.0.0.5", port=16421, graph=None, system="gt1")
    with pytest.raises(G.Refusal):
        R._default_deps(a, connect=lambda *x, **k: pytest.fail("connected"))


def test_the_default_deps_name_the_snapshot_graph_through_the_gate_helper(tmp_path):
    a = types.SimpleNamespace(host="127.0.0.1", port=16421, graph="ccg_other", system="gt1")
    with pytest.raises(G.Refusal, match="disagree"):
        R._default_deps(a, connect=lambda *x, **k: pytest.fail("connected"))


# --- the CLI
def test_cli_run_prints_counts_only_and_leaks_no_item_content(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    monkeypatch.setenv("HOME", str(e.home))
    rc = R.main(e.argv(), deps=e.deps())
    cap = capsys.readouterr()
    assert rc == 0 and json.loads(cap.out) == {"records": 1, "asked": 1, "label": "first-contact"}
    for secret in ("SENTINEL", e.w.rec["itemId"], e.w.rec["anchorId"], e.w.cp, e.w.sid, "GET /a"):
        assert secret not in cap.out and secret not in cap.err


def test_cli_run_refusal_exits_2_with_no_content_and_one_REFUSED_prefix(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    monkeypatch.setenv("HOME", str(e.home))
    custodian_arg = ["--lists", str(e.lists), "--custodian", str(tmp_path / "elsewhere")]
    rc = R.main(["run", "--gate-id", GATE, *custodian_arg, "--repo", str(e.repo), "--dev-probe", str(e.dev_probe)], deps=e.deps())
    cap = capsys.readouterr()
    assert rc == 2 and cap.out == "" and cap.err.count("REFUSED") == 1 and "SENTINEL" not in cap.err + cap.out


def test_cli_run_with_a_probe_map_outside_custody_has_one_REFUSED_prefix(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    monkeypatch.setenv("HOME", str(e.home))
    stray = tmp_path / "p.json"
    stray.write_text("{}")
    rc = R.main(e.argv("--probe-map", str(stray)), deps=e.deps())
    assert rc == 2 and capsys.readouterr().err.count("REFUSED") == 1


def test_run_writes_nothing_outside_the_custodian_root_and_the_repo_stays_clean(tmp_path):
    e = Env(tmp_path)
    before = set(p for p in tmp_path.rglob("*") if ".git" not in p.parts and not str(p).startswith(str(e.home)))
    e.go()
    after = set(p for p in tmp_path.rglob("*") if ".git" not in p.parts and not str(p).startswith(str(e.home)))
    assert after == before
    assert G._porcelain(e.repo) == []
