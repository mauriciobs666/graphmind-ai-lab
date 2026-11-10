"""P1-16 conformance: the `fastapi`, `fetch` and `fastmcp` packs, run by `ccg.packrt`, over the stored `s14a` slice bundles.

Oracle: `cpg/spikes/vertical-slice/assemble.py`, run live over the same bundles (never a golden copied from the runtime's own
output). Contract: rule-pack spec §2.3, §3, §4, §12.1 (the P1 pack-runtime row), O-3, O-5, O-8, O-13; plan
`docs/plans/connected-code-graph-p1-linking.md` § P1-16 behaviours (6), (7), (9).

D9 (`docs/plans/connected-code-graph-p1.md`): the bundles are gitignored (`cpg/.ccg-work/`). A missing artifact *skips* this
file, which the report must count as unrun, never green; the bundles' sha256 is pinned so a regenerated artifact fails
instead of silently becoming a new oracle.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packrt_helpers import CPG, PACKS, S14A, launch  # noqa: E402

from ccg.packrt import run_packs  # noqa: E402
from ccg.rulepack import load_pack  # noqa: E402

pytestmark = [pytest.mark.slice,
              pytest.mark.skipif(not S14A.is_dir() or not list(S14A.glob("*.bundle.jsonl")),
                                 reason="stored s14a slice bundles absent (cpg/.ccg-work is gitignored): UNRUN, not green")]

SHA256 = {
    "0fb899742d4d3687495c30357ca16253.bundle.jsonl": "a0cae513fb6795605a5818942942869fbb1c575de7cdbd4a264ff82833ba1c86",
    "c2341bca0404a6a3eadab1d72ba5f4db.bundle.jsonl": "f2ef6f5add18691d94cecfcb05a742e03d3e06f00cf3fedea462d79886e13e65",
}
SERVER_UNIT = "0fb899742d4d3687495c30357ca16253"


@pytest.fixture(scope="module")
def bundles():
    for name, sha in SHA256.items():
        assert hashlib.sha256((S14A / name).read_bytes()).hexdigest() == sha, f"{name}: not the artifact the oracle was pinned to"
    # the sha256 above pins the bytes, so schema validation (`bundle.read_bundle`, ~45 s here) would re-prove a pinned artifact
    return [[json.loads(line) for line in (S14A / name).read_text().splitlines()] for name in sorted(SHA256)]


@pytest.fixture(scope="module")
def oracle(bundles):
    path = CPG / "spikes/vertical-slice/assemble.py"
    spec = importlib.util.spec_from_file_location("slice_assemble", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.assemble(bundles)


@pytest.fixture(scope="module")
def packs():
    return [load_pack(PACKS / n / "pack.yaml") for n in ("fastapi", "fetch", "fastmcp")]


@pytest.fixture(scope="module")
def out(bundles, packs):
    return run_packs(bundles, packs)


def test_the_slice_oracle_is_the_one_the_plan_states(oracle):
    assert (len(oracle["providers"]), len(oracle["consumers"])) == (61, 11)


# ---- fastapi: 61 providers, hops, and the two handlers of spec 2.3

def http_providers(out):
    return {(p["anchorId"], p["key"]): p for p in out.providers if p["kind"] == "http"}


def test_the_fastapi_pack_gives_the_slices_61_providers_with_the_same_keys_and_mount_hops(out, oracle):
    got, want = http_providers(out), {(p["handlerId"], p["key"]): p for p in oracle["providers"]}
    assert len(got) == 61 and set(got) == set(want)
    for k, w in want.items():
        g = got[k]
        assert (g["qname"], g["path"], g["line"], g["decorator"]) == (w["qname"], w["path"], w["line"], w["decorator"])


def differing(out, oracle):
    got = http_providers(out)
    return {k: (w, got[k]) for k, w in ((( p["handlerId"], p["key"]), p) for p in oracle["providers"])
            if (w["status"], w["reason"], w["mounts"]) != (got[k]["status"], got[k]["reason"], got[k]["evidence"]["mounts"])}


def test_the_only_differences_are_the_two_variable_router_handlers_of_spec_2_3(out, oracle):
    diff = differing(out, oracle)
    names = sorted((w["qname"].split("::")[0].rsplit("/", 1)[-1], w["decorator"], w["status"], g["status"], g["reason"])
                   for w, g in diff.values())
    assert names == [("test_app.py", "inner.get", "confirmed", "possible", "partial-key"),
                     ("test_input_bound_parity.py", "router.post", "confirmed", "possible", "partial-key")]
    for w, g in diff.values():
        assert [m["how"] for m in w["mounts"]] == ["app"] and [m["how"] for m in g["evidence"]["mounts"]] == ["unmounted-router"]


def test_the_status_split_of_the_61_is_57_confirmed_and_4_possible(out):
    from collections import Counter
    assert Counter((p["status"], p["reason"]) for p in http_providers(out).values()) == {
        ("confirmed", "key-match"): 57, ("possible", "partial-key"): 4}      # the slice's 59 / 2, less the two above


def test_the_one_unresolved_provider_alternative_is_dropped_and_listed(out, oracle):
    assert [(d["anchorId"], d["line"]) for d in out.diagnostics] == [(d["handlerId"], d["line"]) for d in oracle["droppedProviderAlternatives"]]
    assert len(out.diagnostics) == 1


# ---- fetch: 11 consumers, same hops

def http_consumers(out):
    return [c for c in out.consumers if c["kind"] == "http"]


def test_the_fetch_pack_gives_the_slices_11_consumers_with_the_same_keys_and_hops(out, oracle):
    got = {c["siteId"]: c for c in http_consumers(out)}
    want = {c["siteId"]: c for c in oracle["consumers"]}
    assert len(got) == 11 and set(got) == set(want)
    for sid, w in want.items():
        g = got[sid]
        assert [g["key"]] == w["keys"] and g["evidence"]["hops"] == w["liftedVia"]
        assert (g["anchorId"], g["path"], g["line"]) == (w["symbolId"], w["path"], w["line"])
        assert (g["status"], g["reason"]) == ("confirmed", "lifted-role") and not w["partialAlternatives"]


def test_every_fetch_slot_that_resolves_has_no_lifted_child_on_s14_so_leaf_only_emission_equals_the_slice(out, oracle):
    """O-13: the pack emits resolved leaves, the slice every resolved slot; equal here, so the lifted list has no resolved non-leaf."""
    assert [i for i in out.lifted if i["resolved"] and not i["leaf"]] == []
    assert [i for i in out.lifted if i["leaf"] and not i["resolved"]] == [] and oracle["unresolvedConsumers"] == []
    assert sum(1 for i in out.lifted if i["resolved"] and i["leaf"]) == 11


# ---- fastmcp: tools, tool calls, the mount path, the paper launch records

def test_the_tool_rule_gives_the_25_mcp_py_tools_and_lists_the_three_test_file_look_alikes_of_o_8(out):
    tools = [p for p in out.providers if p["kind"] == "mcp_tool"]
    assert all((p["edge"], p["status"], p["reason"]) == ("PROVIDES", "confirmed", "key-match") for p in tools)
    mcp_py = [p for p in tools if p["path"].endswith("falkorchat/mcp.py")]
    assert len(mcp_py) == 25 and "send_message" in {p["key"] for p in mcp_py}
    look_alikes = sorted((p["path"].rsplit("/", 1)[-1], p["decorator"], p["key"]) for p in tools if p not in mcp_py)
    assert look_alikes == [("test_input_bound_parity.py", "server.tool", "probe_tool"),
                           ("test_mcp_client.py", "stub_server.tool", "add"),
                           ("test_mcp_client.py", "stub_server.tool", "greet")]
    assert all("/tests/" in p["path"] for p in tools if p not in mcp_py)


def test_the_tool_call_rule_gives_consumes_mcp_tool_with_the_unresolved_names_runtime_built(out):
    calls = [c for c in out.consumers if c["kind"] == "mcp_tool"]
    assert len(calls) == 96
    assert {(c["status"], c["reason"]) for c in calls} == {("confirmed", "key-match"), ("possible", "runtime-built")}
    assert all(c["key"] is None for c in calls if c["status"] == "possible")
    assert all(c["key"] for c in calls if c["status"] == "confirmed")


def test_the_server_unit_has_the_mcp_mount_path_under_the_mount_mcp_condition(out):
    (m,) = [f for f in out.launch_facts if f["fact"] == "mountPath"]
    assert (m["unitId"], m["path"], m["pathMatch"], m["line"]) == (SERVER_UNIT, "/mcp", "prefix", 543)
    assert [c["code"] for c in m["conditions"]] == ["mount_mcp"]


def test_over_the_paper_records_of_spec_4_3_the_server_unit_gets_port_8200_and_not_6379(bundles, packs):
    start = "falkor-chat/scripts/start_agent_team.sh"
    paper = [
        launch("L1", "falkor-chat/Dockerfile", "command", command="uvicorn",
               args=["falkorchat.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/app/config/logging.json"]),
        launch("L2", "falkor-chat/scripts/start_server.sh", "command", command="<redacted>", args=["falkorchat.app:app", "<redacted>", "<redacted>"]),
        launch("L3", start, "env-default", envVar="AGENT_TEAM_PORT", default="8200", origin="env-default"),
        launch("L4", start, "command", command="<redacted>", args=["falkorchat.app:app", "<redacted>", "<redacted>"]),
        launch("L5", start, "env-default", envVar="FALKORDB_PORT", default="6379", origin="env-default"),
    ]
    out = run_packs(bundles + [paper], packs)
    got = sorted((f["unitId"], f["port"], f["origin"]) for f in out.launch_facts if f["fact"] == "port")
    assert got == [(SERVER_UNIT, 8000, "explicit"), (SERVER_UNIT, 8200, "env-default")]
    assert {f["launchId"] for f in out.launch_facts if f["fact"] == "launchUnit"} == {"L1", "L2", "L4"}


def test_every_item_carries_a_registry_status_and_reason(out):
    from ccg.rulepack import reason_row
    for f in out.providers + out.consumers + out.mounts + out.lifted:
        row = reason_row(f["reason"])
        assert row is not None and row["status"] == f["status"], (f["packId"], f.get("key"), f["reason"])
