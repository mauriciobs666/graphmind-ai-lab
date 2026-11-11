"""P1-17 conformance: ``ccg.linker.link`` with ``Binder`` over the stored ``s14a`` slice bundles.

Oracle: ``cpg/spikes/vertical-slice/assemble.py`` run live over the same bundles (never a golden copied from the runtime's own output).
Contract: plan ``docs/plans/connected-code-graph-p1-linking.md`` P1-17 (Done / Gate: "conformance with the slice's 11 links and chains"),
the spec §4.4 walkthrough (8000 and 8200 are two deployments of one unit; two units on one port are ``port-shared``) and the AC-4
``cypher`` chain of ``-p1-graph.md`` §2.

D9 (``docs/plans/connected-code-graph-p1.md``): the bundles are gitignored (``cpg/.ccg-work/``); a missing artifact *skips* this file, which
a report must count as unrun, never green; the bundles' sha256 is pinned (shared with ``test_packrt_conformance.py``).

**Real and paper records.** The ``.mcp.json`` and ``cypher-mcp/Dockerfile`` records are the real config extractor's output over **pinned
copies** of those two repository files (``tests/data/linker_conformance/``), so an edit to the repository's own ``.mcp.json`` cannot redden
this file; the ``cypher`` server's ``command`` is what P1-11b emits today (``command = cypher-mcp/docker-run.sh``, ``args = []``), unpatched.
**Paper records** (hand-made, because no extractor emits them: the shell extractor writes no ``launchFact``): the launch facts L1-L5 of
rule-pack spec §4.3 and the script's own launch fact ``LS``. A row that depends on a paper record carries ``_on_paper_launch_facts`` in its
name and has a companion on the real output (``_on_real_extractor_output``), which for the 8200 client is a strict ``xfail`` naming the
missing extractor, so the gap stays visible in a report.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linker_bind_helpers import KEY, REPOS  # noqa: E402
from linker_helpers import mention, prov, sym_in  # noqa: E402
from packrt_helpers import CPG, S14A, call, config, file_, imp, launch, pack, sym, vs  # noqa: E402
from test_packrt_conformance import SERVER_UNIT, SHA256  # noqa: E402

from ccg.linker import Binder, link  # noqa: E402
from ccg.packrt import PackFacts, run_packs  # noqa: E402

pytestmark = [pytest.mark.slice,
              pytest.mark.skipif(not S14A.is_dir() or not list(S14A.glob("*.bundle.jsonl")),
                                 reason="stored s14a slice bundles absent (cpg/.ccg-work is gitignored): UNRUN, not green")]

REPO = "graphmind-ai-lab"
ROOT_REPOS = {REPO: "."}
SEND = "mcp__falkor-chat-agent-team__send_message"


def header(unit: str, root: str, lang: str, **extra) -> dict:
    return {"type": "header", "unitId": unit, "unit": {"repoId": REPO, "rootPath": root, "lang": lang}, **extra}


def frec(unit: str, path: str, role: str = "primary") -> dict:
    return {"type": "file", "id": f"file-{path}", "repoId": REPO, "unitId": unit, "path": path, "role": role}


@pytest.fixture(scope="module")
def slice_bundles():
    for name, sha in SHA256.items():
        assert hashlib.sha256((S14A / name).read_bytes()).hexdigest() == sha, f"{name}: not the artifact the oracle was pinned to"
    return [[json.loads(line) for line in (S14A / name).read_text().splitlines()] for name in sorted(SHA256)]


@pytest.fixture(scope="module")
def oracle(slice_bundles):
    spec = importlib.util.spec_from_file_location("slice_assemble", CPG / "spikes/vertical-slice/assemble.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.assemble(slice_bundles)


@pytest.fixture(scope="module")
def packs():
    return [pack(n) for n in ("fastapi", "fetch", "fastmcp")]


PINNED = Path(__file__).resolve().parents[1] / "data" / "linker_conformance"


def real_config_records():
    from extractors.config import extract as X
    return X.extract_config(REPO, {".mcp.json": (PINNED / ".mcp.json").read_bytes(),
                                   "cypher-mcp/Dockerfile": (PINNED / "cypher-mcp/Dockerfile").read_bytes()}, key=KEY)


def extra_bundles(paper_launch: bool = True):
    """The units the slice lacks: the real ``.mcp.json`` entries (and, with ``paper_launch``, the paper launch facts L1-L5 and ``LS``), a
    client snippet, the ``cypher-mcp`` unit."""
    res = real_config_records()
    cfg = [dict(r) for r in res.records]
    script = "cypher-mcp/docker-run.sh"
    real = next(r for r in cfg if r["type"] == "launchFact" and r.get("service") == "cypher")
    assert (real["command"], real["args"]) == (script, []), "P1-11b: the allow-listed exec of the project-dir script is the command"
    config_bundle = [header("cfg", ".mcp.json", "config", **res.header_fields), frec("cfg", ".mcp.json"),
                     frec("cfg", "cypher-mcp/Dockerfile"), frec("cfg", script), *cfg]
    if paper_launch:
        config_bundle.append(launch("LS", script, "command", command="docker", args=["run", "--rm", "-i", "cypher-mcp"]))
    paper = [header("paper", "falkor-chat/scripts", "config"), frec("paper", "falkor-chat/Dockerfile"),
             launch("L1", "falkor-chat/Dockerfile", "command", command="uvicorn",
                    args=["falkorchat.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/app/config/logging.json"]),
             launch("L2", "falkor-chat/scripts/start_server.sh", "command", command="<redacted>", args=["falkorchat.app:app", "<redacted>", "<redacted>"]),
             launch("L3", "falkor-chat/scripts/start_agent_team.sh", "env-default", envVar="AGENT_TEAM_PORT", default="8200", origin="env-default"),
             launch("L4", "falkor-chat/scripts/start_agent_team.sh", "command", command="<redacted>", args=["falkorchat.app:app", "<redacted>", "<redacted>"]),
             launch("L5", "falkor-chat/scripts/start_agent_team.sh", "env-default", envVar="FALKORDB_PORT", default="6379", origin="env-default")]
    fid = "f-snippet"
    snippet = [header("snippet", "snippet", "python"), {**file_(fid, "snippet/c.py", unit="snippet"), "repoId": REPO},
               sym("C", "c", fid=fid, path="snippet/c.py"), imp(fid, "mcp", "ClientSession"),
               call("uc", "C", "streamablehttp_client", resolution="external", external="mcp/client/streamable_http.py:<module>.streamablehttp_client"),
               vs("uc", "a0", "http://localhost:8200/mcp"),
               call("ct", "C", "call_tool", resolution="external", external="mcp/client/session.py:<module>.ClientSession.call_tool"),
               vs("ct", "a0", "send_message"),
               sym("D", "prompt", fid=fid, path="snippet/c.py"),
               mention("m-agent", "D", SEND), mention("m-cypher", "D", "mcp__cypher__query")]
    cypher = [header("cypher", "cypher-mcp", "python"), *sym_in("PQ", "cypher", path="cypher-mcp/server.py")]
    return (config_bundle, paper, snippet, cypher) if paper_launch else (config_bundle, snippet, cypher)


@pytest.fixture(scope="module")
def world(slice_bundles, packs):
    """Real config records plus the paper launch facts."""
    bundles = slice_bundles + list(extra_bundles())
    facts = run_packs(bundles, packs)
    facts.providers.append(prov("PQ", "mcp_tool", "query"))
    return facts, bundles


@pytest.fixture(scope="module")
def real_world(slice_bundles, packs):
    """Only what the extractors emit today: the real config records, no paper launch fact."""
    bundles = slice_bundles + list(extra_bundles(paper_launch=False))
    facts = run_packs(bundles, packs)
    facts.providers.append(prov("PQ", "mcp_tool", "query"))
    return facts, bundles


def link_all(facts, bundles, packs, repos=ROOT_REPOS, verify_ids=False, **kw):
    binder = Binder(facts, bundles, key=KEY, repos=repos, packs=packs, verify_ids=verify_ids)
    return link(facts, bundles, packs=packs, bindings=binder, **kw)


@pytest.fixture(scope="module")
def result(world, packs):
    return link_all(*world, packs)


def bound_by(l):
    return [(e["siteId"], e["ord"], e["entryId"]) for e in l.evidence if e["edge"] == "BOUND_BY"]


# ---- the slice's 11 http links, same ids, statuses, matchers and chains --------------------------------------------
def test_the_slices_11_links_and_their_lifted_chains_are_reproduced(result, oracle):
    got = {l.link_id: l for l in result.links if l.kind == "http"}
    assert len(oracle["links"]) == 11 and set(got) == {o["id"] for o in oracle["links"]}
    for o in oracle["links"]:
        l = got[o["id"]]
        assert (l.key_norm, l.status, l.reason, l.matcher) == (o["keyNorm"], o["status"], o["reason"], o["matcher"])
        assert (l.provider, l.consumer) == (o["provider"]["anchorId"], o["consumer"]["anchorId"])
        assert [(v.site_id, v.line, v.consumer_key, v.status, v.reason) for v in l.sites] == \
               [(s["siteId"], s["line"], s["consumerKey"], s["status"], s["reason"]) for s in o["sites"]]
        hops = [{k: v for k, v in e.items() if k != "edge"} for e in l.evidence if e["edge"] == "LIFTED_VIA"]
        assert hops == [h for s in o["sites"] for h in s["liftedVia"]]


def test_the_slices_11_links_have_no_binding_because_their_urls_are_relative(result):
    assert not any(e["edge"] == "BOUND_BY" for l in result.links if l.kind == "http" for e in l.evidence)


# ---- 8000 and 8200: two deployments of falkor-chat/server are one provider ------------------------------------------
def tool_links(result, consumer):
    return [l for l in result.links if l.consumer == consumer and l.kind == "mcp_tool"]


def test_a_client_on_8200_binds_the_server_unit_by_port_and_mount_path_binding_confirmed_on_paper_launch_facts(result):
    (l,) = [l for l in tool_links(result, "C") if l.key_norm == "send_message" and l.provider is not None]
    assert (l.status, l.reason, l.matcher) == ("confirmed", "binding-confirmed", "exact")
    assert l.evidence and bound_by(l) == [("ct", 0, "L3"), ("ct", 1, "L4")]
    first = [e for e in l.evidence if e["edge"] == "BOUND_BY"][0]
    assert (first["port"], first["portOrigin"]) == (8200, "env-default")


def test_the_unit_has_ports_8000_and_8200_and_a_url_on_either_is_one_provider_binding_confirmed_on_paper_launch_facts(world, packs):
    facts, bundles = world
    assert {(f["port"], f["origin"]) for f in facts.launch_facts if f["fact"] == "port" and f["unitId"] == SERVER_UNIT} == \
           {(8000, "explicit"), (8200, "env-default")}
    for port, evidence_ids in ((8000, [("ct", 0, "L1")]), (8200, [("ct", 0, "L3"), ("ct", 1, "L4")])):
        patched = [[dict(r, **({"values": [{"const": f"http://localhost:{port}/mcp"}]} if r.get("type") == "valueSet" and r["callSiteId"] == "uc" else {}))
                    for r in b] for b in bundles]
        got = link_all(facts, patched, packs)
        (l,) = [l for l in tool_links(got, "C") if l.provider is not None and l.key_norm == "send_message"]
        assert (l.status, l.reason) == ("confirmed", "binding-confirmed"), port
        assert bound_by(l) == evidence_ids, port


def test_the_pack_default_port_8000_from_the_paper_command_is_explicit_here_and_never_pack_default_on_paper_launch_facts(world):
    facts, _ = world
    assert not [f for f in facts.launch_facts if f["fact"] == "port" and f["origin"] == "pack-default"]


def test_two_units_on_one_port_give_one_possible_port_shared_link_per_unit_on_paper_launch_facts(world, packs):
    facts, bundles = world
    second = [header("second", "second", "python"), *sym_in("PS", "second", path="second/s.py")]
    more = PackFacts(facts.providers + [prov("PS", "mcp_tool", "send_message")], facts.consumers, facts.mounts, facts.lifted,
                     facts.launch_facts + [{"fact": "port", "packId": "fastmcp", "status": "confirmed", "reason": None, "unitId": "second",
                                           "port": 8200, "origin": "explicit", "protocol": "http", "launchIds": ["L4"]}],
                     facts.marks, facts.diagnostics)
    got = link_all(more, bundles + [second], packs)
    mine = sorted((l.provider is not None and l.status, l.reason) for l in tool_links(got, "C") if l.key_norm == "send_message" and l.provider)
    assert [(s, r) for s, r in mine] == [("possible", "port-shared")] * 2


# ---- the wiring: the real .mcp.json entry binds the mention of its server -------------------------------------------
def test_the_real_mcp_json_url_entry_binds_the_mention_through_the_real_bind_parts_on_paper_launch_facts(result, world):
    _, bundles = world
    c1 = next(r["id"] for r in bundles[2] if r.get("keyPath") == "mcpServers.falkor-chat-agent-team.url")
    assert "bindParts" in next(r for r in bundles[2] if r.get("id") == c1)
    (l,) = [l for l in tool_links(result, "D") if l.key_norm == "send_message" and l.provider is not None]
    assert (l.status, l.reason) == ("possible", "prompt-mention") and bound_by(l) == [("m-agent", 0, c1), ("m-agent", 1, "L3"), ("m-agent", 2, "L4")]


# ---- the 8200 client on what the extractors emit today ---------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="no extractor emits falkor-chat's launch facts (L3 `AGENT_TEAM_PORT` default, L4 uvicorn): the shell "
                                       "extractor writes no launchFact, so 8200 has no route to the server unit; the paper row above is the contract")
def test_a_client_on_8200_binds_the_server_unit_binding_confirmed_on_real_extractor_output(real_world, packs):
    got = link_all(*real_world, packs, verify_ids=True)
    (l,) = [l for l in tool_links(got, "C") if l.key_norm == "send_message" and l.provider is not None]
    assert (l.status, l.reason) == ("confirmed", "binding-confirmed")


def test_the_8200_client_is_unbound_key_match_on_real_extractor_output_today(real_world, packs):
    got = link_all(*real_world, packs)
    (l,) = [l for l in tool_links(got, "C") if l.key_norm == "send_message" and l.provider is not None]
    assert (l.status, l.reason, bound_by(l)) == ("confirmed", "key-match", [])


# ---- AC-4: the cypher chain, .mcp.json server -> command -> script -> container -> unit ----------------------------
def cfg_ids(bundles):
    cfg = bundles[2]
    return (next(r["id"] for r in cfg if r.get("service") == "cypher"),
            next(r["id"] for r in cfg if r["type"] == "launchFact" and r["path"] == "cypher-mcp/Dockerfile"))


def test_the_ac_4_cypher_chain_runs_from_the_mcp_json_server_to_the_cypher_mcp_unit_on_real_extractor_output(real_world, packs):
    """The real ``.mcp.json`` record (command = the script, no args) and the real Dockerfile ``CMD``: the script hop has no entry (P1 chain shape)."""
    facts, bundles = real_world
    l0, ld = cfg_ids(bundles)
    got = link_all(facts, bundles, packs, verify_ids=True)         # the real extractor's ids are the ones its fields derive
    (l,) = [l for l in tool_links(got, "D") if l.key_norm == "query"]
    assert (l.provider, l.status, l.reason) == ("PQ", "possible", "prompt-mention")
    assert bound_by(l) == [("m-cypher", 0, l0), ("m-cypher", 1, ld)]


def test_the_ac_4_cypher_chain_with_the_scripts_own_launch_fact_is_written_in_order_on_paper_launch_facts(result, world):
    l0, ld = cfg_ids(world[1])
    (l,) = [l for l in tool_links(result, "D") if l.key_norm == "query"]
    assert (l.provider, l.status, l.reason) == ("PQ", "possible", "prompt-mention")
    assert bound_by(l) == [("m-cypher", 0, l0), ("m-cypher", 1, "LS"), ("m-cypher", 2, ld)]


def test_the_chain_names_only_launch_entries_and_every_edge_carries_its_site(result):
    for l in result.links:
        for e in (e for e in l.evidence if e["edge"] == "BOUND_BY"):
            assert e["siteId"] in {v.site_id for v in l.sites} and isinstance(e["ord"], int)
    ords = {}
    for l in result.links:
        for e in (e for e in l.evidence if e["edge"] == "BOUND_BY"):
            ords.setdefault((l.link_id, e["siteId"]), []).append(e["ord"])
    assert ords and all(v == list(range(len(v))) for v in ords.values())


def test_a_synthetic_redacted_bash_c_command_leaves_the_cypher_chain_unbound_a_known_limit(world, packs):
    """Synthetic: what the argv gate leaves for any non-allow-listed ``bash -c`` string. The real record no longer has this shape (P1-11b)."""
    facts, bundles = world
    redacted = [[dict(r, command="bash", args=["-c", "<redacted>"]) if r.get("service") == "cypher" else r for r in b] for b in bundles]
    got = link_all(facts, redacted, packs)
    (l,) = [l for l in tool_links(got, "D") if l.key_norm == "query"]
    assert bound_by(l) == []
