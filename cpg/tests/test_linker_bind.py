"""P1-17 dispatch 2: the binding provider ``ccg.linker.bind.Binder`` (URL, env/Compose; behaviours (4), (5), (13), trust rules a-e).

Plan: ``docs/plans/connected-code-graph-p1-linking.md`` P1-17; chain shape ``-p1-graph.md`` §2; parts ``-bundle-spec5.md`` §5.4, §7.
Records are hand-made (readable ids), shaped as the extractors and the pack runtime emit them. Launch binding (6): ``test_linker_bind_launch.py``.
"""
import ast
import builtins
import os
import re
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccg import ids, valuehash
from ccg.linker import Binder, Unmapped, bind as bind_mod, link
from linker_bind_helpers import (KEY, OTHER_KEY, REPOS, config_entry, fastmcp, http_consumer_world, launch_fact, launch_unit, make_binder, mount_fact,
                                 pack_facts, parts_of, port_fact, run_link, unit_bundle, url_lifted)
from linker_helpers import cons, hid, mention, prov, sym_in
from packrt_helpers import call, data, vs


def one(result):
    assert len(result.links) == 1, result.links
    return result.links[0]


def bound_by(link):
    return [(e["siteId"], e["ord"], e["entryId"]) for e in link.evidence if e["edge"] == "BOUND_BY"]


def consumer(site="s1", kind="http", anchor="C", key="GET /api/x", server=None):
    return {"siteId": site, "kind": kind, "anchorId": anchor, "key": key, "keyForm": "literal", "line": 1, "status": "confirmed",
            "reason": "key-match", "hops": [], "server": server}


PORTS_8200 = [launch_fact("L3", "scripts/start.sh", "env-default", envVar="AGENT_TEAM_PORT", default="8200", origin="env-default"),
              launch_fact("L4", "scripts/start.sh", "command", command="uvicorn", args=["app:app"])]


def server_world(url: str, *, extra_units=(), ports=((8200, "env-default", ["L3", "L4"]),), key="GET /api/x"):
    """unit ``svc`` (provider of ``key``, ports per ``ports``) and a web consumer that fetches ``url``."""
    svc = unit_bundle("svc", records=[*sym_in("P", "svc"), *PORTS_8200])
    csym, cfact, lifted = http_consumer_world(url, key=key)
    web = unit_bundle("web", repo="fx-web", records=csym)
    facts = pack_facts(providers=[prov("P", "http", key)], consumers=[cfact], lifted=[lifted],
                       launch=[port_fact("svc", p, o, ids_) for p, o, ids_ in ports])
    return facts, [svc, web, *extra_units]


# --- (4) URL binding: the port, the unit, one provider for several deployments ------------------------
def test_a_loopback_url_selects_the_unit_that_listens_on_its_port_and_the_link_is_binding_confirmed_with_the_chain():
    facts, bundles = server_world("http://localhost:8200/api/x")
    got, _ = run_link(facts, bundles)
    link = one(got)
    assert (link.status, link.reason, link.provider) == ("confirmed", "binding-confirmed", "P")
    assert bound_by(link) == [("s1", 0, "L3"), ("s1", 1, "L4")]                       # ord 0 nearest the consumer, siteId on every edge
    first = [e for e in link.evidence if e["edge"] == "BOUND_BY"][0]
    assert (first["port"], first["portOrigin"]) == (8200, "env-default") and "port" not in link.evidence[1]


def test_the_port_evidence_sits_on_the_command_launch_for_an_explicit_flag_and_carries_the_pack_for_a_pack_default():
    facts, bundles = server_world("http://localhost:8000/api/x", ports=((8000, "pack-default", ["L4"]),))
    ev = [e for e in one(run_link(facts, bundles)[0]).evidence if e["edge"] == "BOUND_BY"]
    assert [(e["entryId"], e["port"], e["portOrigin"], e["packId"]) for e in ev] == [("L4", 8000, "pack-default", "fastmcp")]
    facts, bundles = server_world("http://localhost:8000/api/x", ports=((8000, "explicit", ["L4"]),))
    (e,) = [e for e in one(run_link(facts, bundles)[0]).evidence if e["edge"] == "BOUND_BY"]
    assert (e["port"], e["portOrigin"]) == (8000, "explicit") and "packId" not in e


def test_8000_and_8200_serving_one_unit_are_one_provider_and_both_urls_bind_it():
    ports = ((8000, "explicit", ["L4"]), (8200, "env-default", ["L3", "L4"]))
    for url in ("http://localhost:8000/api/x", "http://localhost:8200/api/x"):
        facts, bundles = server_world(url, ports=ports)
        link = one(run_link(facts, bundles)[0])
        assert (link.status, link.reason) == ("confirmed", "binding-confirmed"), url


def test_two_different_units_on_one_port_are_possible_port_shared_for_each_unit():
    facts, bundles = server_world("http://localhost:8200/api/x")
    job = unit_bundle("job", repo="fx-job", records=[*sym_in("PJ", "job"), launch_fact("JL", "scripts/dev.sh", "command", command="uvicorn")])
    facts.providers.append(prov("PJ", "http", "GET /api/x"))
    facts.launch_facts.append(port_fact("job", 8200, "explicit", ["JL"]))
    got, _ = run_link(facts, [*bundles, job])
    assert sorted((l.provider, l.status, l.reason) for l in got.links) == [("P", "possible", "port-shared"), ("PJ", "possible", "port-shared")]


def test_a_port_no_unit_serves_binds_nothing_and_the_key_match_stays_unbound():
    facts, bundles = server_world("http://localhost:9999/api/x")
    got, b = run_link(facts, bundles)
    assert b.bind(consumer()) is None
    assert (one(got).status, one(got).reason) == ("confirmed", "key-match")


def test_an_absent_port_is_the_scheme_default_decided_in_the_linker():
    facts, bundles = server_world("https://localhost/api/x", ports=((443, "explicit", ["L4"]),))
    assert one(run_link(facts, bundles)[0]).reason == "binding-confirmed"
    facts, bundles = server_world("http://localhost/api/x", ports=((443, "explicit", ["L4"]),))
    assert one(run_link(facts, bundles)[0]).reason == "key-match"


def test_a_binding_restricts_the_providers_to_its_units_so_a_key_match_in_another_unit_makes_no_link():
    facts, bundles = server_world("http://localhost:8200/api/x")
    other = unit_bundle("other", repo="fx-job", records=sym_in("PO", "other"))
    facts.providers.append(prov("PO", "http", "GET /api/x"))
    got, _ = run_link(facts, [*bundles, other])
    assert [l.provider for l in got.links] == ["P"]


def test_a_url_that_is_not_http_like_or_has_an_unresolved_piece_binds_nothing():
    for url in ("/api/x", "\x00/api/x", "http://\x00:8200/api/x"):
        facts, bundles = server_world(url)
        assert make_binder(facts, bundles).bind(consumer()) is None, url


# --- path: mount paths of the packs, prefix-only (O-5) ---------------------------------------------------
def mcp_world(url: str, *, mounts=("/mcp",)):
    svc = unit_bundle("svc", records=[*sym_in("P", "svc"), *PORTS_8200])
    web = unit_bundle("web", repo="fx-web", records=[*sym_in("C", "web", path="web/c.py"),
                                                   call("uc", "C", "streamablehttp_client"), vs("uc", "a0", url)])
    facts = pack_facts(providers=[prov("P", "mcp_tool", "send_message")], consumers=[cons("C", "mcp_tool", "send_message", "s1")],
                       launch=[port_fact("svc", 8200, "env-default", ["L3", "L4"])] + [mount_fact("svc", m) for m in mounts])
    return facts, [svc, web]


@pytest.mark.parametrize("path,binds", [("/mcp", True), ("/mcp/", True), ("/mcp/never-registered/deep", True), ("/MCP", False),
                                        ("/api", False), ("", False), ("/mcpx", False)])
def test_a_consumer_url_path_must_carry_the_mount_path_as_a_prefix_case_sensitively(path, binds):
    facts, bundles = mcp_world("http://localhost:8200" + path)
    got = make_binder(facts, bundles).bind(consumer(kind="mcp_tool", key="send_message"))
    assert (got is not None) == binds


def test_a_unit_without_a_mount_path_of_a_pack_serving_the_kind_is_not_restricted_by_path():
    facts, bundles = mcp_world("http://localhost:8200/anything", mounts=())
    assert make_binder(facts, bundles).bind(consumer(kind="mcp_tool", key="send_message")).units == ("svc",)


def test_an_exact_mount_needs_the_whole_path_and_a_mount_of_another_kind_does_not_restrict_http():
    facts, bundles = mcp_world("http://localhost:8200/mcp/more")
    facts.launch_facts[:] = [port_fact("svc", 8200, "env-default", ["L3", "L4"]), mount_fact("svc", "/mcp", "exact")]
    assert make_binder(facts, bundles).bind(consumer(kind="mcp_tool", key="send_message")) is None
    facts, bundles = server_world("http://localhost:8200/api/x")
    facts.launch_facts.append(mount_fact("svc", "/mcp"))                                 # an mcp_tool mount: no say over http
    assert make_binder(facts, bundles).bind(consumer()).units == ("svc",)


def test_the_tool_link_through_a_file_scope_callsite_url_is_binding_confirmed():
    facts, bundles = mcp_world("http://localhost:8200/mcp")
    link = one(run_link(facts, bundles)[0])
    assert (link.status, link.reason, link.matcher) == ("confirmed", "binding-confirmed", "exact")
    assert bound_by(link) == [("s1", 0, "L3"), ("s1", 1, "L4")]


def test_the_callsite_url_of_another_file_does_not_bind_a_consumer_under_file_scope():
    facts, bundles = mcp_world("http://localhost:8200/mcp")
    bundles[1] += sym_in("C2", "web", path="web/other.py")
    facts.consumers.append(cons("C2", "mcp_tool", "send_message", "s2"))
    got, b = run_link(facts, bundles)
    assert b.bind(consumer(site="s2", anchor="C2", kind="mcp_tool", key="send_message")) is None


def test_a_mention_that_names_its_server_takes_no_callsite_url_source_of_its_file():
    facts, bundles = mcp_world("http://localhost:8200/mcp")
    got = make_binder(facts, bundles).bind(consumer(kind="mcp_tool", key="send_message", server="other"))
    assert got is None
    assert make_binder(facts, bundles).bind(consumer(kind="mcp_tool", key="send_message")).units == ("svc",)


# --- (13) bindParts: the key epoch, BR-1..BR-3 ------------------------------------------------------------
def mention_world(entry: dict, check, *, repo="fx-svc"):
    svc = unit_bundle("svc", records=[*sym_in("P", "svc"), *PORTS_8200])
    cfgu = unit_bundle("mcpcfg", repo=repo, root=".mcp.json", lang="config", check=check, files=[(".mcp.json", "primary")], records=[entry])
    prompts = unit_bundle("prompts", repo=repo, records=[*sym_in("D", "prompts", path="p/d.md"),
                                                      mention("m1", "D", "mcp__falkor-chat-agent-team__send_message")])         # in the config's own repository
    facts = pack_facts(providers=[prov("P", "mcp_tool", "send_message")], launch=[port_fact("svc", 8200, "env-default", ["L3", "L4"])])
    return facts, [svc, cfgu, prompts]


C1 = "mcpServers.falkor-chat-agent-team.url"


def url_entry(bind=None, parts=None, **kw):
    return config_entry("C1", ".mcp.json", C1, fmt="mcp-json", bind=bind, parts=parts, **kw)


def test_br_1_a_usable_bundle_binds_by_its_own_parts_and_no_mismatch_line():
    facts, bundles = mention_world(url_entry(parts=parts_of("http://localhost:8200/mcp")), KEY)
    got, b = run_link(facts, bundles)
    link = one(got)
    assert (link.status, link.reason) == ("possible", "prompt-mention") and bound_by(link) == [("m1", 0, "C1"), ("m1", 1, "L3"), ("m1", 2, "L4")]
    assert got.bind_key_mismatch == []


def test_parts_win_over_a_bind_value_when_the_epoch_is_usable():
    facts, bundles = mention_world(url_entry(bind="http://localhost:9999/mcp", parts=parts_of("http://localhost:8200/mcp")), KEY)
    assert bound_by(one(run_link(facts, bundles)[0]))[0] == ("m1", 0, "C1")


def test_br_2_an_unusable_bundle_binds_by_parts_derived_from_its_bind_value_and_reports_the_unit():
    stale = parts_of("http://localhost:9999/mcp", key=OTHER_KEY)           # the old epoch's parts name 9999: never compared
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp", parts=stale), OTHER_KEY)
    got, _ = run_link(facts, bundles)
    assert bound_by(one(got))[0] == ("m1", 0, "C1") and got.bind_key_mismatch == ["bind-key-mismatch mcpcfg"]


def test_br_3_an_unusable_bundle_with_only_parts_stays_unbound_and_reports_the_unit():
    facts, bundles = mention_world(url_entry(parts=parts_of("http://localhost:8200/mcp", key=OTHER_KEY)), OTHER_KEY)
    got, _ = run_link(facts, bundles)
    assert bound_by(one(got)) == [] and got.bind_key_mismatch == ["bind-key-mismatch mcpcfg"]


def test_a_bundle_of_the_first_minor_with_no_parts_binds_by_its_bind_value_without_a_mismatch_line():
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)
    got, _ = run_link(facts, bundles)
    assert bound_by(one(got))[0] == ("m1", 0, "C1") and got.bind_key_mismatch == []


def test_planted_parts_for_the_wrong_port_under_the_usable_epoch_bind_nothing():
    facts, bundles = mention_world(url_entry(parts=parts_of("http://localhost:9999/mcp")), KEY)
    assert bound_by(one(run_link(facts, bundles)[0])) == []


def test_the_mismatch_line_carries_no_hash_value_and_names_each_unit_once():
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp", parts=parts_of("http://localhost:8200/mcp", key=OTHER_KEY)),
                                   OTHER_KEY)
    lines = make_binder(facts, bundles).key_mismatches()
    assert lines == ["bind-key-mismatch mcpcfg"]
    assert not re.search(r"[0-9a-f]{16}", lines[0])


# --- (4) hostHmac: an external host that equals a declared service's hash ---------------------------------
def compose_world(*, ports=("8300",), declared=("job", "api"), url="http://job:8300/job/status", consumer_key=None):
    compose = [launch_fact("M-api", "deploy/compose.yaml", "mount", service="api", origin="compose", mountPath=".."),
               launch_fact("M-job", "deploy/compose.yaml", "mount", service="job", origin="compose", mountPath="../../fx-job")]
    compose += [launch_fact(f"P-job{p}", "deploy/compose.yaml", "port", service="job", origin="compose", port=int(p), protocol="tcp", line=9,
                            ordinal=i) for i, p in enumerate(ports)]
    svc = unit_bundle("svc", files=[("deploy/compose.yaml", "primary")], records=[*sym_in("P", "svc"), *compose])
    job = unit_bundle("job", repo="fx-job", records=sym_in("PJ", "job"))
    if consumer_key is None:
        consumer_key = "GET /" + url.split("//")[-1].partition("/")[2]
    csym, cfact, lifted = http_consumer_world(url, key=consumer_key)
    web = unit_bundle("web", repo="fx-web", records=csym)
    facts = pack_facts(providers=[prov("PJ", "http", "GET /job/status"), prov("P", "http", "GET /api/x")], consumers=[cfact], lifted=[lifted])
    return facts, [svc, job, web]


def test_an_external_host_binds_when_its_hash_equals_a_declared_service_and_proceeds_as_that_service():
    facts, bundles = compose_world()
    ext = parts_of("http://job:8300/api", declared=())                              # the writer saw no declared `job`: external
    assert ext["hostKind"] == "external" and "host" not in ext
    ent = config_entry("CE", ".mcp.json", "mcpServers.srv.url", repo="fx-web", fmt="mcp-json", parts=ext)
    bundles += [unit_bundle("cfgu", repo="fx-web", root=".mcp.json", lang="config", check=KEY, files=[(".mcp.json", "primary")], records=[ent]),
                unit_bundle("prompts", repo="fx-web", records=[*sym_in("D", "prompts", path="p/d.md"), mention("m1", "D", "mcp__srv__reindex")])]
    facts.providers.append(prov("PJ", "mcp_tool", "reindex"))
    facts.lifted.clear()
    facts.consumers.clear()
    link = one(run_link(facts, bundles)[0])
    assert (link.provider, link.reason) == ("PJ", "prompt-mention") and bound_by(link) == [("m1", 0, "CE"), ("m1", 1, "P-job8300"), ("m1", 2, "M-job")]


def test_an_external_host_that_hashes_to_no_declared_service_binds_nothing():
    facts, bundles = compose_world()
    ext = parts_of("http://stranger:8300/api", declared=())
    ent = config_entry("CE", ".mcp.json", "mcpServers.srv.url", repo="fx-web", fmt="mcp-json", parts=ext)
    bundles += [unit_bundle("cfgu", repo="fx-web", root=".mcp.json", lang="config", check=KEY, records=[ent])]
    assert make_binder(facts, bundles).bind(consumer(anchor="D", kind="mcp_tool", key="reindex", server="srv")) is None


def test_a_dotted_declared_name_is_never_compared_by_hash_bl_16():
    compose = [launch_fact("M-n", "compose.yaml", "mount", service="ntfy.sh", origin="compose", mountPath=".")]
    svc = unit_bundle("svc", records=[*sym_in("P", "svc"), *compose])
    csym, cfact, lifted = http_consumer_world("https://ntfy.sh/x")
    facts = pack_facts(consumers=[cfact], lifted=[lifted])
    assert make_binder(facts, [svc, unit_bundle("web", repo="fx-web", records=csym)]).bind(consumer()) is None


def test_an_ip_literal_host_and_an_unmatched_external_one_bind_nothing():
    for url in ("http://10.0.0.5:8200/api/x", "https://api.github.com/api/x", "http://unknownhost:8200/api/x"):
        facts, bundles = server_world(url)
        assert make_binder(facts, bundles).bind(consumer()) is None, url


# --- (d) link-local and metadata hosts are never bindable, whatever wrote the bundle -------------------
META = ["169.254.169.254", "169.254.0.1", "[fe80::1]", "[fd00:ec2::254]", "100.100.100.200", "168.63.129.16", "192.0.0.192",
        "[2002:a9fe:a9fe::1]", "[fec0::1]", "[100::1]"]


@pytest.mark.parametrize("host", META)
def test_a_metadata_address_planted_in_bind_parts_binds_nothing_even_as_a_loopback_kind(host):
    for kind in ("ip", "loopback"):
        planted = {"scheme": "http", "port": 8200, "hostKind": kind, "host": host,
                   "hostHmac": valuehash.entry_hmac(KEY, valuehash.BIND_HOST_TAG, host), "pathHmacs": []}
        facts, bundles = mention_world(url_entry(parts=planted), KEY)
        assert bound_by(one(run_link(facts, bundles)[0])) == [], (host, kind)


@pytest.mark.parametrize("host", META)
def test_a_metadata_address_planted_in_a_bind_value_binds_nothing(host):
    facts, bundles = mention_world(url_entry(bind=f"http://{host}:8200/mcp"), None)
    assert bound_by(one(run_link(facts, bundles)[0])) == []


def test_the_neighbours_of_the_metadata_set_are_not_forbidden_and_loopback_127_binds():
    for ok in ("169.253.255.255", "169.255.0.0", "[2001::1]", "100.100.100.201", "168.63.129.17", "192.0.0.191", "example.com", "127.0.0.1"):
        assert not bind_mod.forbidden_host(ok), ok
    facts, bundles = mention_world(url_entry(bind="http://127.0.0.1:8200/mcp"), None)
    assert bound_by(one(run_link(facts, bundles)[0]))[0] == ("m1", 0, "C1")


def test_the_forbidden_decision_is_the_writers_one_definition_read_at_call_time(monkeypatch):
    assert bind_mod.forbidden_host("[::ffff:169.254.169.254]") and not bind_mod.forbidden_host(None)
    assert not hasattr(bind_mod, "FORBIDDEN_V4") and not hasattr(bind_mod, "FORBIDDEN_V6")           # no second set to drift
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)
    monkeypatch.setattr(valuehash, "forbidden_target", lambda value: "localhost" in value)          # the shared rule, not a copy of it
    assert bound_by(one(run_link(facts, bundles)[0])) == []


@pytest.mark.parametrize("host", ["0xa9fea9fe", "2852039166", "169.254.43518", "0251.0376.0251.0376", "[::a9fe:a9fe]",
                                  "[64:ff9b::a9fe:a9fe]", "[::ffff:a9fe:a9fe]"])
def test_sec_m3_the_inet_aton_and_embedded_forms_the_writer_forbids_are_never_bindable_in_any_host_kind(host):
    for kind in ("service", "loopback", "external"):
        planted = {"scheme": "http", "port": 8200, "hostKind": kind, "host": host,
                   "hostHmac": valuehash.entry_hmac(KEY, valuehash.BIND_HOST_TAG, host), "pathHmacs": []}
        facts, bundles = mention_world(url_entry(parts=planted), KEY)
        assert bound_by(one(run_link(facts, bundles)[0])) == [], (host, kind)


# --- (5) env / Compose: the unit index is system-wide, the context may leave its repo ---------------------
def test_a_service_host_url_selects_the_unit_of_its_build_context():
    facts, bundles = compose_world(url="http://api:8000/api/x")
    got, _ = run_link(facts, bundles)
    link = one(got)
    assert (link.provider, link.reason) == ("P", "binding-confirmed") and bound_by(link) == [("s1", 0, "M-api")]


def test_the_first_real_case_a_build_context_outside_its_own_repo_reaches_the_sibling_repos_unit():
    facts, bundles = compose_world()
    got, _ = run_link(facts, bundles)
    link = one(got)
    assert (link.provider, link.status, link.reason) == ("PJ", "confirmed", "binding-confirmed")
    assert bound_by(link) == [("s1", 0, "P-job8300"), ("s1", 1, "M-job")] and link.evidence[0]["port"] == 8300


@pytest.mark.parametrize("context,unit", [("..", "svc"), (".", None), ("../../fx-job", "job"), ("../../fx-svc/../fx-job", "job"),
                                          ("../../fx-job/", "job"), ("../../fx-svc", "svc"), ("../../fx-job/sub", None)])
def test_a_mount_path_is_joined_to_its_compose_files_directory_and_normalised_lexically(context, unit):
    facts, bundles = compose_world(url="http://api:8000/x")
    for r in bundles[0]:
        if r.get("id") == "M-api":
            r["mountPath"] = context
    got = make_binder(facts, bundles).bind(consumer())
    assert (None if got is None else got.units and got.units[0]) == unit


@pytest.mark.parametrize("context", ["../../../../etc", "/var/run", "../../../fx-svc", "../../../../../../../etc/x/../.."])
def test_a_path_that_leaves_every_manifest_repo_or_is_absolute_binds_nothing_and_is_target_not_mapped(context):
    facts, bundles = compose_world(url="http://api:8000/api/x")
    for r in bundles[0]:
        if r.get("id") == "M-api":
            r["mountPath"] = context
    got, b = run_link(facts, bundles)
    binding = b.bind(consumer())
    assert binding.units == () and binding.unmapped.unit_id is None and len(binding.unmapped.ref) == 32
    link = one(got)
    assert (link.status, link.reason, link.provider, link.to_unit) == ("possible", "target-not-mapped", None, None)
    assert link.link_id == ids.link_id("http", "GET /api/x", "unmapped:" + binding.unmapped.ref, "C")
    assert "etc" not in binding.unmapped.ref and "var" not in binding.unmapped.ref


def test_two_unmapped_paths_get_two_stable_distinct_value_free_refs():
    refs = set()
    for context in ("/var/run", "/etc"):
        facts, bundles = compose_world(url="http://api:8000/api/x")
        for r in bundles[0]:
            if r.get("id") == "M-api":
                r["mountPath"] = context
        refs.add(make_binder(facts, bundles).bind(consumer()).unmapped.ref)
        assert refs <= {ids.tuple_id(["unmapped", "/var/run"]), ids.tuple_id(["unmapped", "/etc"])}
    assert len(refs) == 2


def test_a_context_into_a_unit_that_was_never_built_is_target_not_mapped_with_that_units_id():
    facts, bundles = compose_world()
    bundles.pop(1)                                                                       # fx-job has no bundle
    never = SimpleNamespace(unitId=hid("job-unit"), repoId="fx-job", rootPath=".")
    got, b = run_link(facts, bundles, units=[never])
    assert b.bind(consumer(key="GET /job/status")).unmapped == Unmapped(hid("job-unit"), hid("job-unit"))
    link = one(got)
    assert (link.status, link.reason, link.to_unit, link.provider) == ("possible", "target-not-mapped", hid("job-unit"), None)


def test_a_context_inside_a_repo_with_no_unit_rooted_there_binds_nothing():
    facts, bundles = compose_world()
    bundles.pop(1)
    assert make_binder(facts, bundles).bind(consumer(key="GET /job/status")) is None


def test_the_service_port_must_be_one_the_service_publishes_or_exposes():
    facts, bundles = compose_world(url="http://job:9999/job/status")
    assert make_binder(facts, bundles).bind(consumer(key="GET /job/status")) is None


def test_a_default_url_read_from_the_environment_binds_through_the_compose_port_fact_lnk_15():
    facts, bundles = compose_world(url="")
    facts.lifted[:] = [url_lifted("s1", "C", "\x00/job/trigger")]
    web = unit_bundle("web", repo="fx-web", records=[*sym_in("C", "web", path="clients/jobs.py"), call("env1", "C", "os.environ.get"),
                                                    vs("env1", "a0", "FX_JOB_URL"), vs("env1", "a1", "http://localhost:8300"),
                                                    data("s1", "env1", "a0", "r")])
    bundles[2] = web
    facts.providers[0]["key"] = "POST /job/trigger"
    facts.consumers[0]["key"] = "POST /job/trigger"
    got, _ = run_link(facts, bundles)
    link = one(got)
    assert (link.provider, link.status, link.reason) == ("PJ", "confirmed", "binding-confirmed")
    assert bound_by(link) == [("s1", 0, "P-job8300"), ("s1", 1, "M-job")]


def test_a_url_the_compose_file_sets_for_the_environment_variable_binds_the_reader_lnk_13():
    facts, bundles = compose_world(url="", consumer_key="GET /api/x")
    facts.lifted[:] = [url_lifted("s1", "C", "\x00/api/x")]
    env = config_entry("E1", "deploy/compose.yaml", "services.web.environment.FX_PEER_BASE", bind="http://api:8000")
    bundles[0] += [env]
    bundles[2] = unit_bundle("web", repo="fx-web", records=[*sym_in("C", "web", path="clients/p.py"), call("env1", "C", "os.getenv"),
                                                          vs("env1", "a0", "FX_PEER_BASE"), data("s1", "env1", "a0", "r")])
    got, _ = run_link(facts, bundles)
    link = one(got)
    assert (link.provider, link.reason) == ("P", "binding-confirmed") and bound_by(link) == [("s1", 0, "E1"), ("s1", 1, "M-api")]


def test_an_environment_read_of_another_variable_does_not_pick_up_a_compose_value():
    facts, bundles = compose_world(url="")
    facts.lifted[:] = [url_lifted("s1", "C", "\x00/api/x")]
    bundles[0] += [config_entry("E1", "deploy/compose.yaml", "services.web.environment.FX_PEER_BASE", bind="http://api:8000")]
    bundles[2] = unit_bundle("web", repo="fx-web", records=[*sym_in("C", "web"), call("env1", "C", "os.getenv"), vs("env1", "a0", "OTHER_VAR"),
                                                          data("s1", "env1", "a0", "r")])
    assert make_binder(facts, bundles).bind(consumer()) is None


# --- (e) the published-vs-container residual, reported and not weakened -----------------------------------
def test_a_loopback_match_through_an_a_colon_b_compose_pair_is_counted_as_the_residual_and_stays_confirmed():
    facts, bundles = compose_world(ports=("8200", "8000"), url="http://localhost:8000/job/status")
    got, b = run_link(facts, bundles)
    link = one(got)
    assert (link.status, link.reason) == ("confirmed", "binding-confirmed")
    assert got.counts["loopback_port_residual"] == 1


def test_a_service_host_url_has_no_residual_and_a_pack_port_fact_match_has_none_either():
    facts, bundles = compose_world(ports=("8200", "8000"), url="http://job:8000/job/status")
    got, _ = run_link(facts, bundles)
    assert got.counts["loopback_port_residual"] == 0
    facts, bundles = server_world("http://localhost:8200/api/x")
    assert run_link(facts, bundles)[0].counts["loopback_port_residual"] == 0


def test_an_m3_a_lone_port_fact_a_lone_expose_and_a_long_syntax_pair_on_two_lines_are_all_counted_as_the_residual():
    """Rule (e): the facts carry no published/container distinction, so a loopback match through ANY Compose port fact is counted."""
    facts, bundles = compose_world(ports=("8300",), url="http://localhost:8300/job/status")             # `ports: "8300"` or `expose: 8300`
    assert run_link(facts, bundles)[0].counts["loopback_port_residual"] == 1
    facts, bundles = compose_world(ports=("8000", "8300"), url="http://localhost:8000/job/status")      # long syntax: published, target, on two lines
    for r in bundles[0]:
        if r.get("id") == "P-job8300":
            r["line"] = 11
    got, _ = run_link(facts, bundles)
    assert one(got).reason == "binding-confirmed" and got.counts["loopback_port_residual"] == 1


def test_the_residual_counts_sites_not_facts_two_sites_on_one_compose_port_are_two():
    facts, bundles = compose_world(ports=("8300",), url="http://localhost:8300/job/status")
    csym, cfact, lifted = http_consumer_world("http://localhost:8300/job/status")
    facts.consumers.append(cons("C", "http", "GET /job/status", "s2"))
    facts.lifted.append(url_lifted("s2", "C", "http://localhost:8300/job/status"))
    assert run_link(facts, bundles)[0].counts["loopback_port_residual"] == 2


# --- (c) a service is (origin, repoId, path, service) -------------------------------------------------
def test_a_mcp_json_server_and_a_compose_service_of_one_name_stay_apart():
    facts, bundles = compose_world(url="http://api:8000/api/x")
    bundles[1] = unit_bundle("job", repo="fx-job", records=[*sym_in("PJ", "job"),
                                                          launch_fact("MCPL", ".mcp.json", "command", service="api", origin="explicit")])
    facts.launch_facts.append(launch_unit("MCPL", "job"))
    bundles.append(unit_bundle("prompts", repo="fx-job", records=sym_in("D", "prompts", path="p/d.md")))
    b = make_binder(facts, bundles)
    assert b.bind(consumer()).units == ("svc",)                                                  # the URL host names the compose service only
    mention_c = consumer(anchor="D", kind="mcp_tool", key="t", server="api")
    assert b.bind(mention_c | {"siteId": "m1"}).units == ("job",)                                # the mention names the .mcp.json server only


def test_one_service_name_in_two_compose_files_is_two_services_with_their_own_chains():
    facts, bundles = compose_world(url="http://api:8000/api/x")
    other = unit_bundle("job", repo="fx-job", files=[("compose.yaml", "primary")], records=[
        *sym_in("PJ2", "job"), launch_fact("M-api2", "compose.yaml", "mount", service="api", origin="compose", mountPath=".")])
    bundles[1] = other
    got = make_binder(facts, bundles).bind(consumer())
    assert got.units == ("job", "svc") and sorted(e.entry_id for e in got.entries) == ["M-api", "M-api2"]


def test_the_port_of_one_compose_file_never_borrows_the_build_context_of_another_with_the_same_service_name():
    facts, bundles = compose_world(url="http://localhost:7000/x")
    bundles[0] += [launch_fact("P-a", "deploy/compose.yaml", "port", service="api", origin="compose", port=7000, protocol="tcp")]
    other = unit_bundle("job", repo="fx-job", files=[("compose.yaml", "primary")], records=[
        launch_fact("M-api2", "compose.yaml", "mount", service="api", origin="compose", mountPath=".")])
    bundles[1] += other[1:]
    got = make_binder(facts, bundles).bind(consumer())
    assert got.units == ("svc",) and [e.entry_id for e in got.entries] == ["P-a", "M-api"]


# --- (7) the example role ------------------------------------------------------------------------------
def example_world(*, primary_too=False):
    from extractors.config import extract as X
    real = X.extract_config("fx-svc", {".mcp.example.json": b'{"mcpServers": {"falkor-chat-agent-team": {"url": "http://localhost:8200/mcp"}}}'}, key=KEY)
    entries = [dict(r, id="EX") for r in real.records if r["type"] == "configEntry" and r["keyPath"] == C1]       # the real extractor's record for the real example name
    assert len(entries) == 1 and entries[0]["format"] == "mcp-json"
    svc = unit_bundle("svc", records=[*sym_in("P", "svc"), *PORTS_8200])
    cfg = unit_bundle("cfgu", root=".mcp.example.json", lang="config", files=[(".mcp.example.json", "example")], records=entries)
    prompts = unit_bundle("prompts", repo="fx-svc", records=[*sym_in("D", "prompts", path="p/d.md"),
                                                          mention("m1", "D", "mcp__falkor-chat-agent-team__send_message")])
    bundles = [svc, cfg, prompts]
    if primary_too:
        bundles.append(unit_bundle("cfg2", root=".mcp.json", lang="config", files=[(".mcp.json", "primary")], records=[
            config_entry("PR", ".mcp.json", C1, fmt="mcp-json", bind="http://localhost:8200/mcp")]))
    facts = pack_facts(providers=[prov("P", "mcp_tool", "send_message")], launch=[port_fact("svc", 8200, "env-default", ["L3", "L4"])])
    return facts, bundles


def test_a_chain_through_an_example_file_is_possible_example_config():
    facts, bundles = example_world()
    link = one(run_link(facts, bundles)[0])
    assert (link.status, link.reason) == ("possible", "example-config") and bound_by(link)[0] == ("m1", 0, "EX")


def test_an_entry_flagged_example_in_a_primary_file_weakens_too_and_a_primary_definition_wins_over_an_example_one():
    facts, bundles = example_world(primary_too=True)
    b = make_binder(facts, bundles)
    got = b.bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="send_message", server="falkor-chat-agent-team"))
    assert got.weaken is None and got.entries[0].entry_id == "PR" and "EX" not in [e.entry_id for e in got.entries]
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp", example=True), None)
    assert one(run_link(facts, bundles)[0]).reason == "example-config"


# --- (a) lexical only: nothing is opened, resolved or dialled ---------------------------------------------
def test_binding_never_touches_the_file_system_or_the_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the linker touched the file system or the network")
    facts, bundles = compose_world(url="http://api:8000/api/x")
    for r in bundles[0]:
        if r.get("id") == "M-api":
            r["mountPath"] = "../../../../etc"
    packs = [fastmcp()]                                 # loading a pack reads its file: before the patch
    with monkeypatch.context() as m:                    # patched only around the call: pytest's own reporting needs the file system
        for mod, names in ((builtins, ["open"]), (os, ["stat", "lstat", "listdir", "scandir", "readlink"]),
                           (os.path, ["exists", "isfile", "isdir", "realpath", "islink"]),
                           (socket, ["socket", "create_connection", "getaddrinfo", "gethostbyname"]),
                           (Path, ["stat", "exists", "resolve", "open", "read_text", "iterdir"])):
            for n in names:
                m.setattr(mod, n, boom, raising=False)
        got, _ = run_link(facts, bundles, packs=packs)
    assert one(got).reason == "target-not-mapped"


def test_the_binder_module_imports_no_file_system_or_network_module():
    src = (Path(bind_mod.__file__)).read_text()
    imported = {a.name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"os", "pathlib", "socket", "subprocess", "shutil", "urllib", "http", "glob", "tempfile", "io", "requests"}
    assert "open(" not in src and "os.path" not in src


# --- the lexical confinement, argument by argument (trust rule b) -----------------------------------------
def ctx_world(context: str, *, repos, units=(("svc", "fx-svc", "."),), compose_repo="fx-svc", compose_path="deploy/compose.yaml"):
    compose = [launch_fact("M-api", compose_path, "mount", service="api", origin="compose", mountPath=context)]
    bundles = [unit_bundle(u, repo=r, root=root, files=[(compose_path, "primary")] if i == 0 else (),
                           records=[*sym_in("P" + u, u), *(compose if i == 0 else [])]) for i, (u, r, root) in enumerate(units)]
    bundles[0][0]["unit"]["repoId"] = compose_repo
    csym, cfact, lifted = http_consumer_world("http://api:8000/api/x")
    facts = pack_facts(providers=[prov("Psvc", "http", "GET /api/x")], consumers=[cfact], lifted=[lifted])
    bundles.append(unit_bundle("web", repo="fx-web", records=csym))
    return make_binder(facts, bundles, repos=repos)


@pytest.mark.parametrize("context", ["/var/run", "../../x", "../..", "../../../etc", "deploy/../../.."])
def test_a_root_repository_never_follows_an_absolute_context_or_one_that_climbs_out_of_it(context):
    b = ctx_world(context, repos={"fx-svc": "."})
    got = b.bind(consumer())
    assert got.units == () and got.unmapped is not None and got.unmapped.unit_id is None


@pytest.mark.parametrize("context,unit", [("..", "svc"), ("../deploy/..", "svc"), (".", None)])
def test_a_root_repository_resolves_a_context_inside_it(context, unit):
    got = ctx_world(context, repos={"fx-svc": "."}).bind(consumer())
    assert (None if got is None else got.units[0]) == unit


def test_a_sibling_directory_that_only_shares_a_repository_name_prefix_is_not_inside_that_repository():
    got = ctx_world("../../fx-job-evil", repos={"fx-svc": "fx-svc", "fx-job": "fx-job"}).bind(consumer())
    assert got.units == () and got.unmapped is not None


def test_a_context_inside_nested_manifest_repositories_belongs_to_the_deepest_one():
    repos = {"outer": ".", "inner": "fx-job"}
    for order in (repos, dict(reversed(list(repos.items())))):
        b = ctx_world("../../fx-job", repos=order | {"fx-svc": "fx-svc"}, units=(("svc", "fx-svc", "."), ("job", "inner", ".")))
        assert b.bind(consumer()).units == ("job",)


def test_a_context_into_a_subdirectory_of_a_repository_is_matched_against_the_units_root_path_there():
    b = ctx_world("../../fx-job/sub", repos={"fx-svc": "fx-svc", "fx-job": "fx-job"}, units=(("svc", "fx-svc", "."), ("sub", "fx-job", "sub")))
    assert b.bind(consumer()).units == ("sub",)


def test_two_spellings_of_one_outside_path_give_one_ref_and_the_ref_is_a_hash_of_the_normalised_path():
    refs = {ctx_world(c, repos={"fx-svc": "fx-svc"}).bind(consumer()).unmapped.ref for c in ("../../../../etc", "../../../../etc/.", "../../../x/../../etc")}
    assert refs == {ids.tuple_id(["unmapped", "../../etc"])}


def test_when_one_service_maps_and_another_does_not_only_the_mapped_unit_is_bound():
    b = ctx_world("..", repos={"fx-svc": "fx-svc"})
    b._launch["M-api2"] = type(b._launch["M-api"])("M-api2", "fx-svc", "deploy/compose.yaml", "svc", False,
                                                  launch_fact("M-api2", "deploy/compose.yaml", "mount", service="api", origin="compose",
                                                              mountPath="../../../../etc", line=2))
    got = b.bind(consumer())
    assert got.units == ("svc",) and got.unmapped is None


def test_the_origin_of_a_launch_fact_keeps_a_mcp_json_command_out_of_a_compose_service_and_the_reverse():
    facts, bundles = compose_world(url="http://api:8000/api/x")
    bundles[0] = unit_bundle("svc", files=[("deploy/compose.yaml", "primary")], records=[
        *sym_in("P", "svc"), launch_fact("C-api", "deploy/compose.yaml", "command", service="api", origin="compose", command="uvicorn")])
    bundles[1] = unit_bundle("job", repo="fx-job", records=[*sym_in("PJ", "job"),
                                                          launch_fact("MCPL", ".mcp.json", "command", service="api", origin="explicit"),
                                                          launch_fact("CC", "deploy/compose.yaml", "command", service="srv", origin="compose")])
    facts.launch_facts.extend([launch_unit("C-api", "svc"), launch_unit("MCPL", "job"), launch_unit("CC", "job")])
    b = make_binder(facts, bundles)
    assert b.bind(consumer()).units == ("svc",)                                                   # the compose service, not the mcp server
    assert b.bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="t", server="srv")) is None            # a compose command is not an mcp server


def test_a_mount_path_of_the_root_matches_every_path_and_a_two_segment_mount_needs_both_segments():
    facts, bundles = mcp_world("http://localhost:8200/whatever/at/all", mounts=("/",))
    assert make_binder(facts, bundles).bind(consumer(kind="mcp_tool", key="send_message")).units == ("svc",)
    for path, binds in (("/api/v1", True), ("/api/v1/x/y", True), ("/api", False), ("/api/v2", False), ("/v1/api", False)):
        facts, bundles = mcp_world("http://localhost:8200" + path, mounts=("/api/v1/",))
        assert (make_binder(facts, bundles).bind(consumer(kind="mcp_tool", key="send_message")) is not None) == binds, path


def test_bind_parts_with_no_key_check_in_the_header_are_not_usable_and_the_unit_is_reported():
    facts, bundles = mention_world(url_entry(bind="http://localhost:9999/mcp", parts=parts_of("http://localhost:8200/mcp")), None)
    got, _ = run_link(facts, bundles)
    assert got.bind_key_mismatch == ["bind-key-mismatch mcpcfg"] and bound_by(one(got)) == []      # derived from bindValue: 9999 is not served


def test_a_config_entry_outside_the_packs_key_path_clause_binds_no_mention():
    facts, bundles = mention_world(config_entry("C1", ".mcp.json", "mcpServers.falkor-chat-agent-team.other", fmt="mcp-json",
                                                bind="http://localhost:8200/mcp"), None)
    assert bound_by(one(run_link(facts, bundles)[0])) == []


def test_a_service_defined_in_a_primary_and_an_example_compose_file_binds_the_primary_ones_unit_only_and_not_weakened():
    facts, bundles = compose_world(url="http://api:8000/api/x")
    bundles[1] = unit_bundle("job", repo="fx-job", files=[("compose.example.yaml", "example")], records=[
        *sym_in("PJ2", "job"), launch_fact("M-ex", "compose.example.yaml", "mount", service="api", origin="compose", mountPath=".")])
    got = make_binder(facts, bundles).bind(consumer())
    assert got.units == ("svc",) and got.weaken is None and [e.entry_id for e in got.entries] == ["M-api"]


def test_a_mention_is_bound_by_the_entry_of_its_own_server_and_not_by_another_servers():
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)
    other = config_entry("C2", ".mcp.json", "mcpServers.other.url", fmt="mcp-json", bind="http://localhost:8300/mcp")
    bundles[1].append(other)
    bundles.append(unit_bundle("other", repo="fx-job", records=[*sym_in("PO", "other")]))
    facts.providers.append(prov("PO", "mcp_tool", "send_message"))
    facts.launch_facts.append(port_fact("other", 8300, "explicit", ["L4"]))
    b = make_binder(facts, bundles)
    assert b.bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="send_message", server="other")).units == ("other",)
    assert b.bind(consumer(anchor="D", site="m2", kind="mcp_tool", key="send_message", server="falkor-chat-agent-team")).units == ("svc",)


def test_the_unmapped_ref_chosen_for_several_outside_paths_is_the_smallest_ref_not_the_first_entry():
    refs = {c: ids.tuple_id(["unmapped", c]) for c in ("/var/run", "/etc")}
    low, high = sorted(refs, key=refs.get)
    facts, bundles = compose_world(url="http://api:8000/api/x")
    for r in bundles[0]:
        if r.get("id") == "M-api":
            r["mountPath"] = high
    bundles[0].append(launch_fact("M-zzz", "deploy/compose.yaml", "mount", service="api", origin="compose", mountPath=low, line=2))
    got = make_binder(facts, bundles).bind(consumer())
    assert got.unmapped.ref == refs[low] and [e.entry_id for e in got.entries] == ["M-zzz"]


def test_one_compose_path_and_service_name_in_two_repositories_are_two_services_whose_facts_never_mix():
    facts, bundles = compose_world(url="http://localhost:7000/api/x")
    bundles[0] = unit_bundle("svc", files=[("compose.yaml", "primary")], records=[
        *sym_in("P", "svc"), launch_fact("M-a", "compose.yaml", "mount", service="api", origin="compose", mountPath="."),
        launch_fact("P-a", "compose.yaml", "port", service="api", origin="compose", port=7000, protocol="tcp")])
    bundles[1] = unit_bundle("job", repo="fx-job", files=[("compose.yaml", "primary")], records=[
        *sym_in("PJ", "job"), launch_fact("M-b", "compose.yaml", "mount", service="api", origin="compose", mountPath=".")])
    got = make_binder(facts, bundles).bind(consumer())
    assert got.units == ("svc",) and [e.entry_id for e in got.entries] == ["P-a", "M-a"]


def test_two_compose_files_with_one_port_fact_each_still_count_one_site_once():
    facts, bundles = compose_world(url="http://localhost:7000/api/x")
    bundles[0] = unit_bundle("svc", files=[("deploy/compose.yaml", "primary")], records=[
        *sym_in("P", "svc"), launch_fact("M-a", "deploy/compose.yaml", "mount", service="api", origin="compose", mountPath=".."),
        launch_fact("P-a", "deploy/compose.yaml", "port", service="api", origin="compose", port=7000, protocol="tcp", line=9)])
    bundles[1] = unit_bundle("job", repo="fx-job", files=[("compose.yaml", "primary")], records=[
        *sym_in("PJ", "job"), launch_fact("M-b", "compose.yaml", "mount", service="api", origin="compose", mountPath="."),
        launch_fact("P-b", "compose.yaml", "port", service="api", origin="compose", port=7001, protocol="tcp", line=9)])
    got, _ = run_link(facts, bundles)
    assert got.counts["loopback_port_residual"] == 1


# --- SEC M2: the launch facts are indexed once, so a site costs its matches, not services x facts --------
def test_sec_m2_3000_compose_services_and_20_loopback_sites_bind_in_seconds_not_a_minute():
    import time
    n, sites = 3000, 20
    compose = []
    for i in range(n):
        compose += [launch_fact(f"M{i}", "deploy/compose.yaml", "mount", service=f"s{i}", origin="compose", mountPath="..", line=3 * i),
                    launch_fact(f"P{i}", "deploy/compose.yaml", "port", service=f"s{i}", origin="compose", port=10000 + i % 5000, protocol="tcp",
                                line=3 * i + 1),
                    launch_fact(f"K{i}", "deploy/compose.yaml", "command", service=f"s{i}", origin="compose", command="uvicorn", args=["app:app"],
                                line=3 * i + 2)]
    svc = unit_bundle("svc", files=[("deploy/compose.yaml", "primary")], records=[*sym_in("P", "svc"), *compose])
    web = unit_bundle("web", repo="fx-web", records=sym_in("C", "web", path="web/c.py"))
    facts = pack_facts(providers=[prov("P", "http", "GET /api/x")],
                       consumers=[cons("C", "http", "GET /api/x", f"s{k}") for k in range(sites)],
                       lifted=[url_lifted(f"s{k}", "C", f"http://localhost:{10000 + k * 100}/api/x") for k in range(sites)])
    t = time.perf_counter()
    got, _ = run_link(facts, [svc, web])
    took = time.perf_counter() - t
    assert one(got).reason == "binding-confirmed" and got.counts["loopback_port_residual"] == sites and len(one(got).sites) == sites and took < 5, took


# --- SEC m1: an absolute compose-file path never bypasses trust rule (b) -----------------------------------
def test_sec_m1_an_absolute_compose_file_path_in_a_root_repository_binds_no_unit_rooted_at_that_absolute_directory():
    b = ctx_world("x", repos={"fx-svc": "."}, units=(("svc", "fx-svc", "/etc/x"),), compose_path="/etc/compose.yml")
    got = b.bind(consumer())
    assert got.units == () and got.unmapped is not None and got.unmapped.unit_id is None


def test_sec_m1_each_lexical_guard_refuses_an_absolute_path_by_itself():
    b = ctx_world("x", repos={"fx-svc": "."}, compose_path="/etc/compose.yml")
    assert b._locate("/etc/x") is None and b._locate("etc/x") == ("fx-svc", "etc/x")
    rec = b._launch["M-api"]
    assert b._system_dir(rec) is None


# --- SEC m2: a later bundle reusing a record id never shadows an earlier one ----------------------------------
def test_sec_m2_a_record_id_reused_by_another_bundle_drops_both_records_and_is_counted():
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)
    assert bound_by(one(run_link(facts, bundles)[0]))[0] == ("m1", 0, "C1")                              # control: the legitimate entry binds
    evil = unit_bundle("evil", repo="fx-job", root="x.json", lang="config", files=[("x.json", "primary")], records=[
        config_entry("C1", "x.json", C1, fmt="mcp-json", bind="http://localhost:9999/mcp")])
    got, b = run_link(facts, [*bundles, evil])
    assert bound_by(one(got)) == [] and got.counts["binding_duplicate_ids"] == 1                          # neither wins; the forged one binds nothing


def test_sec_m2_the_same_record_listed_twice_is_not_a_clash_and_a_launch_id_clash_is_one_too():
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)
    got, _ = run_link(facts, [*bundles, bundles[1]])
    assert bound_by(one(got))[0] == ("m1", 0, "C1") and got.counts["binding_duplicate_ids"] == 0
    facts, bundles = server_world("http://localhost:8200/api/x")
    dup = unit_bundle("dup", repo="fx-job", records=[launch_fact("L3", "scripts/other.sh", "env-default", envVar="X_PORT", default="1", origin="env-default")])
    got, _ = run_link(facts, [*bundles, dup])
    assert got.counts["binding_duplicate_ids"] == 1


# --- SEC m4: a malformed record or bindParts never raises; the entry fails closed -----------------------------
@pytest.mark.parametrize("parts", [{}, {"scheme": "http", "port": 8200, "hostKind": "loopback", "host": "localhost"},
                                   {"scheme": "http", "port": 8200, "hostKind": "loopback", "host": "localhost", "hostHmac": "x", "pathHmacs": "abc"},
                                   {"scheme": "http", "port": 8200, "hostKind": "loopback", "host": ["localhost"], "hostHmac": "x", "pathHmacs": []},
                                   {"scheme": "http", "port": "8200", "hostKind": "loopback", "host": "localhost", "hostHmac": "x", "pathHmacs": []},
                                   {"scheme": 7, "port": 8200, "hostKind": "loopback", "host": "localhost", "hostHmac": "x", "pathHmacs": []},
                                   {"scheme": "http", "port": 8200, "hostKind": "nonsense", "host": "localhost", "hostHmac": "x", "pathHmacs": []},
                                   {"scheme": "http", "port": 8200, "hostKind": "loopback", "host": "localhost", "hostHmac": "x", "pathHmacs": [1]},
                                   "http://localhost:8200", ["a"], 5])
def test_sec_m4_malformed_bind_parts_raise_nothing_and_bind_nothing(parts):
    facts, bundles = mention_world(url_entry(parts=parts), KEY)
    assert bound_by(one(run_link(facts, bundles)[0])) == []


def test_sec_m4_one_malformed_record_does_not_stop_the_others_from_binding():
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)
    bundles[1] += [{"type": "configEntry", "id": "Cbad", "repoId": "fx-svc", "path": ".mcp.json", "line": 1, "format": "mcp-json"},
                   {"type": "launchFact", "id": "Lbad", "path": ".mcp.json", "kind": "command", "origin": "explicit", "service": "x",
                    "command": 5, "args": "not-a-list"},
                   {"type": "launchFact", "id": "Lbad2", "path": "deploy/compose.yaml", "kind": "port", "origin": "compose", "service": "api", "port": "x"},
                   {"type": "launchFact", "id": 9, "path": "p", "kind": "mount", "origin": "compose", "service": "api", "mountPath": 5}]
    got, _ = run_link(facts, bundles)
    assert bound_by(one(got))[0] == ("m1", 0, "C1")


# --- SEC m5 (the part that needs no contract decision): only an mcp-json entry is a server's url --------------------
def test_sec_m5_a_server_url_key_in_a_compose_or_other_format_file_binds_no_mention():
    for fmt in ("compose", "yaml", "toml"):
        facts, bundles = mention_world(config_entry("C1", ".mcp.json", C1, fmt=fmt, bind="http://localhost:8200/mcp"), None)
        assert bound_by(one(run_link(facts, bundles)[0])) == [], fmt


# --- AN M2: a port-shared link carries only its own unit's chain ------------------------------------------------------
def test_an_m2_each_port_shared_link_holds_only_the_launch_entries_of_its_own_unit():
    facts, bundles = server_world("http://localhost:8200/api/x")
    job = unit_bundle("job", repo="fx-job", records=[*sym_in("PJ", "job"), launch_fact("JL", "scripts/dev.sh", "command", command="uvicorn")])
    facts.providers.append(prov("PJ", "http", "GET /api/x"))
    facts.launch_facts.append(port_fact("job", 8200, "explicit", ["JL"]))
    got, _ = run_link(facts, [*bundles, job])
    by_provider = {l.provider: bound_by(l) for l in got.links}
    assert by_provider == {"P": [("s1", 0, "L3"), ("s1", 1, "L4")], "PJ": [("s1", 0, "JL")]}


def test_an_m2_a_mapped_link_carries_no_entry_of_an_unmapped_chain():
    facts, bundles = compose_world(url="http://api:8000/api/x")
    bundles[0].append(launch_fact("M-out", "deploy/compose.yaml", "mount", service="api", origin="compose", mountPath="../../../../etc", line=4))
    got, _ = run_link(facts, bundles)
    link = one(got)
    assert (link.provider, link.reason) == ("P", "binding-confirmed") and bound_by(link) == [("s1", 0, "M-api")]


# --- AN N1: decisions with no failing test --------------------------------------------------------------------------
def example_compose_world(url: str):
    ex = unit_bundle("svc", files=[("deploy/compose.example.yaml", "example")], records=[
        *sym_in("P", "svc"),
        launch_fact("M-x", "deploy/compose.example.yaml", "mount", service="exsvc", origin="compose", mountPath=".."),
        launch_fact("P-x", "deploy/compose.example.yaml", "port", service="exsvc", origin="compose", port=9100, protocol="tcp"),
        config_entry("E-x", "deploy/compose.example.yaml", "services.exsvc.image", fmt="compose", bind=None)])      # declares it by an entry too
    csym, cfact, lifted = http_consumer_world(url)
    web = unit_bundle("web", repo="fx-web", records=csym)
    facts = pack_facts(providers=[prov("P", "http", "GET /api/x")], consumers=[cfact], lifted=[lifted])
    return facts, [ex, web]


@pytest.mark.parametrize("url", ["http://localhost:9100/api/x", "http://exsvc:9100/api/x"])
def test_an_n1_a_compose_service_declared_only_in_an_example_file_claims_neither_a_loopback_port_nor_a_service_host(url):
    facts, bundles = example_compose_world(url)
    got, b = run_link(facts, bundles)
    assert b.bind(consumer()) is None and (one(got).status, one(got).reason) == ("confirmed", "key-match")


def test_an_n1_a_root_repository_never_follows_an_absolute_script_token_even_when_such_a_file_record_exists():
    from test_linker_bind_launch import REPOS, cypher_world
    facts, bundles = cypher_world(args=["-c", "exec /cypher-mcp/docker-run.sh"])
    got = make_binder(facts, bundles, repos=REPOS).bind(consumer(anchor="D", site="m1", kind="mcp_tool", key="query", server="cypher"))
    assert got is None


def test_an_n1_a_service_with_a_build_context_and_a_command_binds_by_the_context_only():
    facts, bundles = compose_world(url="http://api:8000/api/x")
    bundles[0].append(launch_fact("K-api", "deploy/compose.yaml", "command", service="api", origin="compose", command="uvicorn", args=["x:app"], line=5))
    facts.launch_facts.append(launch_unit("K-api", "job"))
    got = make_binder(facts, bundles).bind(consumer())
    assert got.units == ("svc",) and [e.entry_id for e in got.entries] == ["M-api"]


def test_an_n1_without_a_build_context_the_command_launch_unit_binds():
    facts, bundles = compose_world(url="http://api:8000/api/x")
    bundles[0][:] = [r for r in bundles[0] if r.get("id") != "M-api"]
    bundles[0].append(launch_fact("K-api", "deploy/compose.yaml", "command", service="api", origin="compose", command="uvicorn", args=["x:app"], line=5))
    facts.launch_facts.append(launch_unit("K-api", "job"))
    got = make_binder(facts, bundles).bind(consumer())
    assert got.units == ("job",) and [e.entry_id for e in got.entries] == ["K-api"]


# --- AN N3: the env value is the union over every Compose service (the P1 rule) ----------------------------------------
def env_world():
    facts, bundles = compose_world(url="", consumer_key="GET /api/x")
    facts.lifted[:] = [url_lifted("s1", "C", "\x00/api/x")]
    bundles[2] = unit_bundle("web", repo="fx-web", records=[*sym_in("C", "web", path="clients/p.py"), call("env1", "C", "os.getenv"),
                                                          vs("env1", "a0", "FX_PEER_BASE"), data("s1", "env1", "a0", "r")])
    return facts, bundles


def test_an_n3_the_variable_set_by_two_compose_services_binds_both_units_and_is_port_shared_each_with_its_own_chain():
    """The consumer's own Compose service cannot be told (the fixture's `web` is image-only, `-p0-compose-binding.md`), so the P1 rule is the union."""
    facts, bundles = env_world()
    bundles[0] += [config_entry("E1", "deploy/compose.yaml", "services.web.environment.FX_PEER_BASE", bind="http://api:8000"),
                   config_entry("E2", "deploy/compose.yaml", "services.other.environment.FX_PEER_BASE", bind="http://job:8300")]
    facts.providers[1]["key"] = "GET /api/x"
    got, b = run_link(facts, bundles)
    assert b.bind(consumer()).units == ("job", "svc")
    link = [l for l in got.links if l.provider == "P"][0]
    assert (link.status, link.reason) == ("possible", "port-shared") and bound_by(link) == [("s1", 0, "E1"), ("s1", 1, "M-api")]


def test_an_n3_a_services_environment_key_path_in_a_non_compose_file_binds_nothing():
    facts, bundles = env_world()
    bundles[0] += [config_entry("E1", "deploy/x.yaml", "services.web.environment.FX_PEER_BASE", fmt="yaml", bind="http://api:8000")]
    assert make_binder(facts, bundles).bind(consumer()) is None


# --- AN N4: callsite scope `unit`, and a pack's config clause naming anything but keyPath ---------------------------------
def test_an_n4_a_unit_scope_callsite_url_binds_a_consumer_in_another_file_of_the_unit():
    import copy
    pk = copy.deepcopy(fastmcp())
    for ub in pk["urlBindings"]:
        for src in ub["urlSources"]:
            if "callsite" in src:
                src["scope"] = "unit"
    facts, bundles = mcp_world("http://localhost:8200/mcp")
    bundles[1] += sym_in("C2", "web", path="web/other.py")
    facts.consumers.append(cons("C2", "mcp_tool", "send_message", "s2"))
    c2 = consumer(site="s2", anchor="C2", kind="mcp_tool", key="send_message")
    assert make_binder(facts, bundles, packs=[pk]).bind(c2).units == ("svc",)
    assert make_binder(facts, bundles).bind(c2) is None                                            # the shipped file scope: no


def test_an_n4_a_config_clause_naming_a_key_other_than_key_path_is_refused_when_the_binder_is_built():
    import copy
    pk = copy.deepcopy(fastmcp())
    pk["urlBindings"][0]["urlSources"][0] = {"config": {"keyPath": {"regex": "x"}, "format": "mcp-json"}}
    facts, bundles = mcp_world("http://localhost:8200/mcp")
    with pytest.raises(ValueError, match="may only name keyPath"):
        make_binder(facts, bundles, packs=[pk])


# --- SEC m3 / m4, each guard by itself ---------------------------------------------------------------------------
def test_sec_m3_a_host_the_writers_rule_cannot_judge_is_forbidden_not_bindable(monkeypatch):
    def boom(value):
        raise ValueError("cannot judge")
    monkeypatch.setattr(valuehash, "forbidden_target", boom)
    assert bind_mod.forbidden_host("localhost") is True
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)
    assert bound_by(one(run_link(facts, bundles)[0])) == []


def test_sec_m4_a_launch_fact_of_the_wrong_shape_is_skipped_alone_and_the_service_still_binds():
    base, _ = compose_world(url="http://api:8000/api/x")
    for extra in ({"line": "x"}, {"port": "x"}):                                    # a sort key / a known-port set that would be poisoned
        facts, bundles = compose_world(url="http://api:8000/api/x")
        bundles[0].append(launch_fact("Lx", "deploy/compose.yaml", "port", service="api", origin="compose", **{"port": 8000, "protocol": "tcp", **extra}))
        got = make_binder(facts, bundles).bind(consumer())
        assert got is not None and got.units == ("svc",), extra
    facts, bundles = compose_world(url="http://api:8000/api/x")
    for r in bundles[0]:
        if r.get("id") == "M-api":
            r["mountPath"] = 5
    assert make_binder(facts, bundles).bind(consumer()) is None


def test_sec_m4_a_launch_fact_whose_service_is_not_text_is_skipped_not_raised():
    facts, bundles = compose_world(url="http://api:8000/api/x")
    bundles[0].append(launch_fact("Lx", "deploy/compose.yaml", "port", service=5, origin="compose", port=8000, protocol="tcp"))
    assert make_binder(facts, bundles).bind(consumer()).units == ("svc",)


def test_a_port_listed_twice_in_one_compose_file_contributes_its_first_fact_only():
    facts, bundles = compose_world(url="http://localhost:7000/api/x")
    bundles[0] += [launch_fact("P-b", "deploy/compose.yaml", "port", service="api", origin="compose", port=7000, protocol="tcp", line=12),
                   launch_fact("P-a", "deploy/compose.yaml", "port", service="api", origin="compose", port=7000, protocol="tcp", line=5)]
    got = make_binder(facts, bundles).bind(consumer())
    assert [e.entry_id for e in got.entries] == ["P-a", "M-api"]


# --- L1: ccg never imports extractors; the forbidden-target rule is ccg's, the writer imports it ------------------------
def test_l1_no_module_under_ccg_imports_extractors():
    root = Path(bind_mod.__file__).resolve().parents[1]
    bad = []
    for f in root.rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) and not n.level else []
            bad += [f"{f.relative_to(root)}: {x}" for x in names if x.split(".")[0] == "extractors"]
    assert bad == []


def test_l1_the_writer_uses_the_ccg_definition_not_its_own():
    from extractors.common import writer_rules as wr
    assert wr._forbidden_target is valuehash.forbidden_target and wr._V4_NETS is valuehash.V4_NETS


# --- m5 (architect ruling): a mention binds only through its server's .mcp.json in the mention's OWN repository --------
def test_m5_another_repositorys_mcp_json_url_entry_never_attributes_a_mention():
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)
    assert bound_by(one(run_link(facts, bundles)[0]))[0] == ("m1", 0, "C1")                          # control: the mention's own repository binds
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp", repo="fx-job"), None, repo="fx-job")
    bundles[2] = unit_bundle("prompts", repo="fx-web", records=bundles[2][1:])                          # the mention moves to another repository
    assert bound_by(one(run_link(facts, bundles)[0])) == []


def test_m5_only_a_mcp_json_path_is_a_servers_configuration_even_in_the_mentions_own_repository():
    for path in ("node_modules/x/README.json", "config/servers.json", "mcp.json"):
        facts, bundles = mention_world(config_entry("C1", path, C1, fmt="mcp-json", bind="http://localhost:8200/mcp"), None)
        assert bound_by(one(run_link(facts, bundles)[0])) == [], path
    for path in (".mcp.json", "sub/.mcp.json"):
        facts, bundles = mention_world(config_entry("C1", path, C1, fmt="mcp-json", bind="http://localhost:8200/mcp"), None)
        assert bound_by(one(run_link(facts, bundles)[0]))[0][2] == "C1", path


def test_m5_a_mention_whose_symbol_is_unknown_binds_through_no_server_entry():
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)
    assert make_binder(facts, bundles).bind(consumer(anchor="NOPE", site="m9", kind="mcp_tool", key="send_message",
                                                     server="falkor-chat-agent-team")) is None


# --- D1: an id must be the one the record's own fields derive; a different record with that id never evicts it ----------------
def real_entry(repo: str, path: str, key_path: str, **kw) -> dict:
    return config_entry(ids.config_entry_id(repo, path, key_path), path, key_path, repo=repo, **kw)


def d1_world(entry: dict, *, repo="fx-svc"):
    facts, bundles = mention_world(entry, None, repo=repo)
    return facts, bundles


def test_d1_a_forged_record_with_an_id_its_fields_do_not_derive_is_rejected_and_the_legitimate_one_is_kept():
    legit = real_entry("fx-svc", ".mcp.json", C1, fmt="mcp-json", bind="http://localhost:8200/mcp")
    facts, bundles = d1_world(legit)
    forged = config_entry(legit["id"], ".mcp.json", "not.a.server.url", repo="fx-svc", fmt="mcp-json", bind="http://localhost:9999/mcp")
    evil = unit_bundle("evil", repo="fx-svc", root="x", lang="config", files=[(".mcp.json", "primary")], records=[forged])
    got, b = run_link(facts, [*bundles, evil], verify_ids=True)
    assert bound_by(one(got))[0] == ("m1", 0, legit["id"]) and got.counts["binding_duplicate_ids"] == 0


def test_d1_a_record_claiming_another_repository_than_its_bundles_is_rejected_even_with_that_repositorys_derived_id():
    legit = real_entry("fx-svc", ".mcp.json", C1, fmt="mcp-json", bind="http://localhost:8200/mcp")
    facts, bundles = d1_world(legit)
    forged = real_entry("fx-svc", ".mcp.json", C1, fmt="mcp-json", bind="http://localhost:9999/mcp")           # id derived for fx-svc ...
    evil = unit_bundle("evil", repo="fx-job", root="x", lang="config", files=[(".mcp.json", "primary")], records=[forged])    # ... in an fx-job bundle
    got, _ = run_link(facts, [*bundles, evil], verify_ids=True)
    assert bound_by(one(got))[0] == ("m1", 0, legit["id"]) and got.counts["binding_duplicate_ids"] == 0


def test_d1_a_record_whose_repo_id_is_not_its_bundles_is_rejected_even_with_an_id_derived_for_the_bundles_repo():
    facts, bundles = mention_world(real_entry("fx-svc", ".mcp.json", C1, fmt="mcp-json", bind="http://localhost:8200/mcp"), None, repo="fx-job")
    bundles[1][-1] = config_entry(ids.config_entry_id("fx-job", ".mcp.json", C1), ".mcp.json", C1, repo="fx-svc", fmt="mcp-json", bind="http://localhost:8200/mcp")
    bundles[2] = unit_bundle("prompts", repo="fx-svc", records=bundles[2][1:])                         # the mention lives in the repo the record claims
    assert bound_by(one(run_link(facts, bundles, verify_ids=True)[0])) == []
    assert bound_by(one(run_link(facts, bundles)[0]))[0][2] == ids.config_entry_id("fx-job", ".mcp.json", C1)  # unverified: it would bind


def test_d1_a_launch_fact_whose_id_is_not_derived_from_its_fields_is_rejected():
    facts, bundles = server_world("http://localhost:8200/api/x")                                           # hand-made readable ids
    assert make_binder(facts, bundles, verify_ids=True)._launch == {}                                      # none derives: none is indexed
    assert set(make_binder(facts, bundles)._launch) == {"L3", "L4"}
    good = launch_fact(ids.launch_fact_id("fx-svc", "scripts/start.sh", "command", 0), "scripts/start.sh", "command", command="uvicorn")
    third = launch_fact(ids.launch_fact_id("fx-svc", "scripts/start.sh", "command", 3), "scripts/start.sh", "command", ordinal=3, command="uvicorn")
    bundles[0] += [good, third]
    assert set(make_binder(facts, bundles, verify_ids=True)._launch) == {good["id"], third["id"]}


def test_d1_the_same_record_in_two_overlapping_unit_bundles_counts_once_and_keeps_binding():
    legit = real_entry("fx-svc", ".mcp.json", C1, fmt="mcp-json", bind="http://localhost:8200/mcp")
    facts, bundles = d1_world(legit)
    twin = unit_bundle("mcpcfg2", repo="fx-svc", root=".mcp.json", lang="config", files=[(".mcp.json", "primary")], records=[dict(legit)])
    got, _ = run_link(facts, [*bundles, twin], verify_ids=True)
    assert bound_by(one(got))[0] == ("m1", 0, legit["id"]) and got.counts["binding_duplicate_ids"] == 0


def test_d1_two_different_records_with_one_derived_id_in_one_repository_still_drop_both_and_are_counted():
    """Residual (reported): the id derives from (repo, path, keyPath) and a bundle's repoId is its own claim, so a forger who matches all three still clashes."""
    legit = real_entry("fx-svc", ".mcp.json", C1, fmt="mcp-json", bind="http://localhost:8200/mcp")
    facts, bundles = d1_world(legit)
    forged = real_entry("fx-svc", ".mcp.json", C1, fmt="mcp-json", bind="http://localhost:9999/mcp")
    evil = unit_bundle("evil", repo="fx-svc", root="x", lang="config", files=[(".mcp.json", "primary")], records=[forged])
    got, _ = run_link(facts, [*bundles, evil], verify_ids=True)
    assert bound_by(one(got)) == [] and got.counts["binding_duplicate_ids"] == 1


# --- P5-2a: the id check is on by default ----------------------------------------------------------------------------
def test_p5_a_plain_binder_rejects_a_forged_id_record():
    facts, bundles = mention_world(url_entry(bind="http://localhost:8200/mcp"), None)       # "C1" is not the id its fields derive
    assert bound_by(one(link(facts, bundles, bindings=Binder(facts, bundles, key=KEY, repos=REPOS, packs=[fastmcp()])))) == []
    assert bound_by(one(run_link(facts, bundles)[0]))[0] == ("m1", 0, "C1")                  # control: the test helper turns the check off


# --- P5-1: one predicate, the extractor's ----------------------------------------------------------------------------
MCP_NAMES = {".mcp.json": True, "sub/.mcp.json": True, ".mcp.example.json": True, ".mcp.local.json": True, "a/b/.mcp.example.json": True,
             "README.json": False, ".mcp.json.bak": False, ".mcp.json.example": False, "mcp.json": False, "x.mcp.json": False,
             "node_modules/x/.mcp.json": False, "a/node_modules/b/.mcp.example.json": False, "node_modules.json": False}


def test_p5_the_mcp_json_predicate_is_one_shared_function_over_a_name_table():
    from ccg import units
    from extractors.config import formats
    assert formats.is_mcp_json is units.is_mcp_json
    for path, want in MCP_NAMES.items():
        assert units.is_mcp_json(path) is want, path
        assert (formats.is_mcp_json(path)) is want, path


def test_p5_a_mention_binds_through_the_real_extractors_example_file_as_possible_example_config():
    from extractors.config import extract as X
    text = b'{"mcpServers": {"falkor-chat-agent-team": {"url": "http://localhost:8200/mcp"}}}'
    for name in (".mcp.example.json", ".mcp.local.json"):
        res = X.extract_config("fx-svc", {name: text}, key=KEY)
        entry = next(r for r in res.records if r["type"] == "configEntry" and r["keyPath"] == C1)
        assert entry["format"] == "mcp-json", name
        role = "example" if "example" in name else "primary"
        facts, bundles = mention_world(entry, None)
        bundles[1] = unit_bundle("mcpcfg", root=name, lang="config", files=[(name, role)], records=[entry])
        link_ = one(run_link(facts, bundles)[0])
        assert bound_by(link_)[0][0:2] == ("m1", 0), name
        assert (link_.status, link_.reason) == (("possible", "example-config") if role == "example" else ("possible", "prompt-mention")), name
