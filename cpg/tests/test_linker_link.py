"""P1-17 dispatch 1: ``link(facts, bundles) -> Links`` (plan ``docs/plans/connected-code-graph-p1-linking.md`` P1-17
behaviours (1), (2), (3), (7), (8), (9), (11), the mention-call-site consumer rule and the bindings seam).

Binding itself (URL, env/Compose, launch) is dispatch 2's ``bind.py``; here the seam is exercised through ``FakeBindings``.
"""
import pytest

from ccg import ids
from ccg.linker import Links, Unmapped, link
from linker_helpers import (FakeBindings, binding, cons, facts, hid, lifted_leaf, mention, prov, shell_site, sym_in)
from packrt_helpers import call


def bundle(*parts):
    return [r for p in parts for r in p]


def one(result: Links):
    assert len(result.links) == 1, result.links
    return result.links[0]


# --- (1) both keys known and matching -------------------------------------------------------------
def test_literal_keys_that_match_give_a_confirmed_key_match_link():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "mcp_tool", "send_message")], [cons("C", "mcp_tool", "send_message", "s1")]), [b]))
    assert (got.status, got.reason, got.matcher) == ("confirmed", "key-match", "exact")
    assert (got.kind, got.key_norm, got.provider, got.consumer) == ("mcp_tool", "send_message", "P", "C")
    assert got.link_id == ids.link_id("mcp_tool", "send_message", "P", "C")
    assert got.to_unit is None and got.param_map == {}


def test_http_template_provider_matches_literal_consumer_and_the_link_key_is_the_providers():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "http", "GET /api/items/{}")], [cons("C", "http", "GET /api/items/7", "s1")]), [b]))
    assert (got.status, got.reason, got.matcher, got.key_norm) == ("confirmed", "key-match", "path-template", "GET /api/items/{}")


def test_keys_of_different_kinds_never_match():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    assert link(facts([prov("P", "mcp_tool", "x")], [cons("C", "http", "x", "s1")]), [b]).links == []


def test_status_is_the_weakest_hop_a_possible_consumer_weakens_a_literal_match():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    c = cons("C", "http", "GET /a", "s1", status="possible", reason="partial-key")
    got = one(link(facts([prov("P", "http", "GET /a")], [c]), [b]))
    assert (got.status, got.reason) == ("possible", "partial-key")


def test_status_is_the_weakest_hop_a_possible_provider_weakens_it_too():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    p = prov("P", "http", "GET /a", status="possible", reason="partial-key")
    got = one(link(facts([p], [cons("C", "http", "GET /a", "s1")]), [b]))
    assert (got.status, got.reason) == ("possible", "partial-key")


def test_a_pack_with_a_link_matcher_for_its_kind_is_honoured_and_a_kind_without_one_links_exact():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    pf = facts([prov("P", "topic", "bus:a.b")], [cons("C", "topic", "bus:a.*", "s1", form="glob")])
    assert one(link(pf, [b])).matcher == "glob"
    pf = facts([prov("P", "http", "GET /a/{}")], [cons("C", "http", "GET /a/1", "s1")])
    assert link(pf, [b], packs=[{"id": "p", "link": [{"kind": "http", "matcher": "exact"}]}]).links == []


def test_pattern_keys_are_matched_by_their_form():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    pf = facts([prov("P", "topic", "kafka:orders.created")], [cons("C", "topic", r"kafka:^orders\..*", "s1", form="regex")])
    got = one(link(pf, [b]))
    assert (got.status, got.reason, got.matcher) == ("confirmed", "key-match", "regex")


# --- one Link per consumer symbol (BF6) ----------------------------------------------------------
def test_two_sites_of_one_symbol_on_one_key_share_one_link_with_two_vias():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    cs = [cons("C", "mcp_tool", "t", "s1", line=3), cons("C", "mcp_tool", "t", "s2", line=9)]
    got = one(link(facts([prov("P", "mcp_tool", "t")], cs), [b]))
    assert [(v.site_id, v.line) for v in got.sites] == [("s1", 3), ("s2", 9)]


def test_two_consumer_symbols_on_one_key_give_two_links():
    b = bundle(sym_in("P", "svc"), sym_in("C1", "web"), sym_in("C2", "web"))
    cs = [cons("C1", "mcp_tool", "t", "s1"), cons("C2", "mcp_tool", "t", "s2")]
    got = link(facts([prov("P", "mcp_tool", "t")], cs), [b])
    assert sorted(l.consumer for l in got.links) == ["C1", "C2"]


def test_link_status_is_the_best_over_its_sites_and_the_reason_is_that_sites():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    cs = [cons("C", "http", "GET /a", "s1", status="possible", reason="partial-key"), cons("C", "http", "GET /a", "s2")]
    got = one(link(facts([prov("P", "http", "GET /a")], cs), [b]))
    assert (got.status, got.reason) == ("confirmed", "key-match")
    assert {v.site_id: (v.status, v.reason) for v in got.sites} == {"s1": ("possible", "partial-key"), "s2": ("confirmed", "key-match")}


def test_the_link_reason_among_equal_best_sites_is_that_of_the_lowest_site_id():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    cs = [cons("C", "http", "GET /a", "s2", status="possible", reason="partial-key"),
          cons("C", "http", "GET /a", "s1", status="possible", reason="partial-key")]
    # a possible provider fact adds nothing to either site; both stay possible/partial-key, ties by siteId
    got = one(link(facts([prov("P", "http", "GET /a")], cs), [b]))
    assert got.reason == "partial-key" and [v.site_id for v in got.sites] == ["s1", "s2"]


# --- (2) partial or name-only ---------------------------------------------------------------------
def test_a_partial_http_key_links_possible_partial_key():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "http", "GET /api/items/{}")], [cons("C", "http", "GET /api/items/{?}", "s1")]), [b]))
    assert (got.status, got.reason, got.key_norm) == ("possible", "partial-key", "GET /api/items/{}")
    assert got.sites[0].consumer_key == "GET /api/items/{?}"


def test_a_provider_with_no_key_is_matched_by_its_symbol_name_only():
    b = bundle(sym_in("P", "svc", name="reindex"), sym_in("C", "web"))
    pf = facts([prov("P", "mcp_tool", None, status="possible", reason="runtime-built")], [cons("C", "mcp_tool", "reindex", "s1")])
    got = one(link(pf, [b]))
    assert (got.status, got.reason, got.matcher) == ("possible", "name-only-provider", "name")


def test_a_provider_with_no_key_and_a_different_name_does_not_link():
    b = bundle(sym_in("P", "svc", name="other"), sym_in("C", "web"))
    pf = facts([prov("P", "mcp_tool", None, status="possible", reason="runtime-built")], [cons("C", "mcp_tool", "reindex", "s1")])
    assert link(pf, [b]).links == []


# --- (3) a fully unknown key ----------------------------------------------------------------------
def test_an_unknown_key_links_every_provider_of_the_kind_none_collapsed():
    b = bundle(sym_in("P1", "svc"), sym_in("P2", "svc"), sym_in("P3", "job"), sym_in("PH", "svc"), sym_in("C", "web"))
    ps = [prov("P1", "mcp_tool", "reindex"), prov("P2", "mcp_tool", "get_item"), prov("P3", "mcp_tool", "reindex"),
          prov("PH", "http", "GET /x")]
    got = link(facts(ps, [cons("C", "mcp_tool", None, "s1")]), [b])
    assert sorted(l.provider for l in got.links) == ["P1", "P2", "P3"]          # two providers of one key, two links
    assert {(l.status, l.reason) for l in got.links} == {("possible", "unknown-key")}
    assert len({l.link_id for l in got.links}) == 3


def test_a_fully_unknown_consumer_lives_only_in_lifted_and_links_every_provider_of_the_packs_kind():
    b = bundle(sym_in("P1", "svc"), sym_in("P2", "svc"), sym_in("C", "web"))
    ps = [prov("P1", "http", "GET /a"), prov("P2", "http", "POST /b")]
    pack = {"id": "fetch", "rules": [{"id": "r", "match": {"lifted": {"lifting": "fetch-role"}},
                                      "emit": [{"consumes": {"kind": "http", "key": "{u}"}}]}], "link": []}
    pf = facts(ps, [], [lifted_leaf("fetch", "fetch-role", "s1", "C")])
    got = link(pf, [b], packs=[pack])
    assert sorted(l.provider for l in got.links) == ["P1", "P2"]
    assert {(l.status, l.reason) for l in got.links} == {("possible", "unknown-key")}


def test_an_unresolved_lifted_leaf_needs_the_packs_to_name_its_kind():
    b = bundle(sym_in("P1", "svc"), sym_in("C", "web"))
    with pytest.raises(ValueError, match="packs"):
        link(facts([prov("P1", "http", "GET /a")], [], [lifted_leaf("fetch", "fetch-role", "s1", "C")]), [b])


def test_a_lifted_leaf_that_is_resolved_or_not_a_leaf_is_not_an_unknown_consumer():
    b = bundle(sym_in("P1", "svc"), sym_in("C", "web"))
    pack = {"id": "fetch", "rules": [{"id": "r", "match": {"lifted": {"lifting": "fetch-role"}},
                                      "emit": [{"consumes": {"kind": "http", "key": "{u}"}}]}]}
    resolved = lifted_leaf("fetch", "fetch-role", "s1", "C", resolved=True)
    inner = lifted_leaf("fetch", "fetch-role", "s2", "C") | {"leaf": False}
    assert link(facts([prov("P1", "http", "GET /a")], [], [resolved, inner]), [b], packs=[pack]).links == []


def test_the_lifted_chain_travels_as_ordered_lifted_via_evidence_per_site():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    hops = [{"kind": "callsite", "id": "s1", "ord": 0}, {"kind": "symbol", "id": "W", "ord": 1}]
    c = cons("C", "http", "GET /a", "s1", evidence={"hops": hops})
    got = one(link(facts([prov("P", "http", "GET /a")], [c]), [b]))
    via = [e for e in got.evidence if e["edge"] == "LIFTED_VIA"]
    assert [(e["siteId"], e["ord"], e["id"]) for e in via] == [("s1", 0, "s1"), ("s1", 1, "W")]


def test_a_shell_dynamic_call_site_gives_the_link_reason_shell_dynamic_over_unknown_key():
    b = bundle(sym_in("P1", "svc"), sym_in("P2", "job"), sym_in("C", "svc", name="<script>", path="svc/dyn.sh")) \
        + [shell_site("s1", "C")]
    ps = [prov("P1", "mcp_tool", "get_item"), prov("P2", "mcp_tool", "reindex")]
    got = link(facts(ps, [cons("C", "mcp_tool", None, "s1")]), [b])
    assert len(got.links) == 2
    assert {(l.status, l.reason) for l in got.links} == {("possible", "shell-dynamic")}


def test_a_shell_dynamic_site_with_a_literal_tool_is_still_shell_dynamic_not_runtime_built():
    b = bundle(sym_in("P", "svc"), sym_in("C", "svc", path="svc/dyn.sh")) + [shell_site("s1", "C")]
    c = cons("C", "mcp_tool", "reindex", "s1", status="possible", reason="runtime-built")
    got = one(link(facts([prov("P", "mcp_tool", "reindex")], [c]), [b]))
    assert (got.status, got.reason) == ("possible", "shell-dynamic")


def test_a_site_that_is_not_shell_dynamic_keeps_unknown_key():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web")) + [call("s1", "C", "callTool")]
    got = one(link(facts([prov("P", "mcp_tool", "x")], [cons("C", "mcp_tool", None, "s1")]), [b]))
    assert got.reason == "unknown-key"


def test_a_known_literal_tool_on_a_non_shell_dynamic_site_is_untouched_by_the_shell_rule():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web")) + [call("s1", "C", "callTool")]
    got = one(link(facts([prov("P", "mcp_tool", "x")], [cons("C", "mcp_tool", "x", "s1")]), [b]))
    assert (got.status, got.reason) == ("confirmed", "key-match")


# --- (7) example files, prompt mentions, documentation ---------------------------------------------
def test_a_link_through_an_example_file_is_possible_example_config():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web", role="example"))
    got = one(link(facts([prov("P", "http", "GET /a")], [cons("C", "http", "GET /a", "s1")]), [b]))
    assert (got.status, got.reason) == ("possible", "example-config")


def test_a_provider_in_an_example_file_is_also_example_config():
    b = bundle(sym_in("P", "svc", role="example"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "http", "GET /a")], [cons("C", "http", "GET /a", "s1")]), [b]))
    assert (got.status, got.reason) == ("possible", "example-config")


def test_example_config_outranks_the_other_reasons_of_a_possible_site():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web", role="example"))
    got = one(link(facts([prov("P", "http", "GET /a/{}")], [cons("C", "http", "GET /a/{?}", "s1")]), [b]))
    assert got.reason == "example-config"


def test_a_documentation_snippet_is_not_a_caller():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web", role="doc"))
    assert link(facts([prov("P", "http", "GET /a")], [cons("C", "http", "GET /a", "s1")]), [b]).links == []


def test_a_tool_mention_links_possible_prompt_mention():
    b = bundle(sym_in("P", "svc"), sym_in("D", "prompts", role="primary")) + [mention("m1", "D", "mcp__cypher__send_message")]
    got = one(link(facts([prov("P", "mcp_tool", "send_message")], []), [b]))
    assert (got.status, got.reason, got.consumer, got.key_norm) == ("possible", "prompt-mention", "D", "send_message")   # the provider's key
    assert [v.site_id for v in got.sites] == ["m1"]


def test_a_bare_tool_name_mention_links_and_a_leading_dot_slash_is_stripped():
    b = bundle(sym_in("P", "svc"), sym_in("D", "prompts")) + [mention("m1", "D", "./x.sh"), mention("m2", "D", "x.sh")]
    got = one(link(facts([prov("P", "mcp_tool", "x.sh")], []), [b]))
    assert [v.site_id for v in got.sites] == ["m1", "m2"]          # one provider, one link, both mentions


def test_a_script_mention_that_names_no_provider_makes_no_link():
    b = bundle(sym_in("P", "svc"), sym_in("D", "prompts")) + [mention("m1", "D", "./other.sh"), mention("m2", "D", "git")]
    assert link(facts([prov("P", "mcp_tool", "x.sh")], []), [b]).links == []


def test_only_mcp_tool_providers_are_mention_targets():
    b = bundle(sym_in("P", "svc"), sym_in("D", "prompts")) + [mention("m1", "D", "health")]
    assert link(facts([prov("P", "http", "health")], []), [b]).links == []


def test_a_mention_in_an_example_prompt_file_is_example_config():
    b = bundle(sym_in("P", "svc"), sym_in("D", "prompts", role="example")) + [mention("m1", "D", "mcp__s__t")]
    got = one(link(facts([prov("P", "mcp_tool", "t")], []), [b]))
    assert got.reason == "example-config"


def test_a_call_that_is_not_a_mention_is_not_read_as_one():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web")) + [call("c1", "C", "mcp__s__t", kind="call")]
    assert link(facts([prov("P", "mcp_tool", "t")], []), [b]).links == []


# --- the bindings seam; (8) never built; (9) O-9 --------------------------------------------------
def test_without_bindings_a_confirmed_literal_match_is_key_match():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "mcp_tool", "t")], [cons("C", "mcp_tool", "t", "s1")]), [b], bindings=None))
    assert got.reason == "key-match" and [e for e in got.evidence if e["edge"] == "BOUND_BY"] == []


def test_a_single_unit_binding_makes_a_confirmed_site_binding_confirmed_with_ordered_bound_by():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    bd = FakeBindings({"s1": binding("svc", entries=("e-url", "e-port", "e-cmd"))})
    got = one(link(facts([prov("P", "mcp_tool", "t")], [cons("C", "mcp_tool", "t", "s1")]), [b], bindings=bd))
    assert (got.status, got.reason) == ("confirmed", "binding-confirmed")
    assert got.sites[0].reason == "binding-confirmed"
    bound = [e for e in got.evidence if e["edge"] == "BOUND_BY"]
    assert [(e["siteId"], e["ord"], e["entryId"]) for e in bound] == [("s1", 0, "e-url"), ("s1", 1, "e-port"), ("s1", 2, "e-cmd")]


def test_the_binding_not_the_key_selects_the_provider_among_same_named_tools():
    b = bundle(sym_in("P1", "svc"), sym_in("P2", "job"), sym_in("C", "web"))
    ps = [prov("P1", "mcp_tool", "reindex"), prov("P2", "mcp_tool", "reindex")]
    got = one(link(facts(ps, [cons("C", "mcp_tool", "reindex", "s1")]), [b], bindings=FakeBindings({"s1": binding("svc")})))
    assert got.provider == "P1"


def test_without_a_binding_same_named_tools_in_two_units_both_link_confirmed():
    b = bundle(sym_in("P1", "svc"), sym_in("P2", "job"), sym_in("C", "web"))
    ps = [prov("P1", "mcp_tool", "reindex"), prov("P2", "mcp_tool", "reindex")]
    got = link(facts(ps, [cons("C", "mcp_tool", "reindex", "s1")]), [b])
    assert sorted((l.provider, l.status, l.reason) for l in got.links) == [("P1", "confirmed", "key-match"), ("P2", "confirmed", "key-match")]


def test_two_units_on_one_port_are_possible_port_shared():
    b = bundle(sym_in("P", "job"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "http", "GET /job/status")], [cons("C", "http", "GET /job/status", "s1")]), [b],
                   bindings=FakeBindings({"s1": binding("svc", "job")})))
    assert (got.status, got.reason) == ("possible", "port-shared")


def test_an_unknown_key_narrowed_by_a_binding_links_the_units_providers_binding_narrowed():
    b = bundle(sym_in("P1", "svc"), sym_in("P2", "job"), sym_in("C", "web"))
    ps = [prov("P1", "mcp_tool", "a"), prov("P2", "mcp_tool", "b")]
    got = one(link(facts(ps, [cons("C", "mcp_tool", None, "s1")]), [b], bindings=FakeBindings({"s1": binding("svc")})))
    assert (got.provider, got.status, got.reason) == ("P1", "possible", "binding-narrowed")


def test_a_binding_into_a_unit_never_built_gives_a_possible_target_not_mapped_link_without_a_provider():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "http", "GET /a")], [cons("C", "http", "GET /a", "s1")]), [b],
                   bindings=FakeBindings({"s1": binding(unmapped=Unmapped(hid("fx-job-unit"), hid("fx-job-unit")))})))
    assert (got.status, got.reason, got.provider, got.to_unit) == ("possible", "target-not-mapped", None, hid("fx-job-unit"))
    assert got.link_id == ids.link_id("http", "GET /a", "unit:" + hid("fx-job-unit"), "C")        # no provider: the consumer's key
    assert got.key_norm == "GET /a" and got.matcher == "none"


def test_a_binding_with_no_chain_entries_is_refused():
    with pytest.raises(ValueError):
        binding("svc", entries=())


def test_o9_a_confirmed_unbound_site_keeps_key_match_beside_a_bound_site_of_the_same_link():
    """The ``siteId`` condition: s2 has no ``BOUND_BY`` of its own, so its ``VIA`` stays ``key-match`` although the link has one."""
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    cs = [cons("C", "mcp_tool", "t", "s1"), cons("C", "mcp_tool", "t", "s2")]
    got = one(link(facts([prov("P", "mcp_tool", "t")], cs), [b], bindings=FakeBindings({"s1": binding("svc")})))
    assert {v.site_id: v.reason for v in got.sites} == {"s1": "binding-confirmed", "s2": "key-match"}
    assert got.reason == "binding-confirmed"          # some confirmed site is


def test_o9_a_possible_bound_site_keeps_its_own_reason_and_does_not_make_the_link_binding_confirmed():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    bd = FakeBindings({"s1": binding("svc")})
    got = one(link(facts([prov("P", "http", "GET /a/{}")], [cons("C", "http", "GET /a/{?}", "s1")]), [b], bindings=bd))
    assert (got.status, got.reason) == ("possible", "partial-key")      # bound but partial: not binding-confirmed
    assert got.sites[0].reason == "partial-key"


def test_o9_a_possible_site_bound_beside_a_confirmed_unbound_site_in_one_symbol_does_not_win_binding_confirmed():
    """Same symbol: s1 bound but possible (partial consumer fact), s2 confirmed unbound. 'Any BOUND_BY' would say binding-confirmed."""
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    cs = [cons("C", "http", "GET /a", "s1", status="possible", reason="partial-key"), cons("C", "http", "GET /a", "s2")]
    got = one(link(facts([prov("P", "http", "GET /a")], cs), [b], bindings=FakeBindings({"s1": binding("svc")})))
    assert (got.status, got.reason) == ("confirmed", "key-match")
    assert {v.site_id: v.reason for v in got.sites} == {"s1": "partial-key", "s2": "key-match"}


def test_the_seam_is_asked_once_per_consumer_site_and_not_for_a_documentation_site():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"), sym_in("D", "web", role="doc", path="web/d.md"))
    bd = FakeBindings({})
    link(facts([prov("P", "http", "GET /a")], [cons("C", "http", "GET /a", "s1"), cons("D", "http", "GET /a", "s9")]), [b], bindings=bd)
    assert bd.asked == ["s1"]


def test_the_key_mismatch_lines_of_the_seam_are_returned_beside_the_links():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    bd = FakeBindings({}, mismatches=["bind-key-mismatch U1"])
    assert link(facts([prov("P", "http", "GET /a")], [cons("C", "http", "GET /a", "s1")]), [b], bindings=bd).bind_key_mismatch == ["bind-key-mismatch U1"]
    assert link(facts([], []), [b]).bind_key_mismatch == []


# --- determinism, registry, counts (11) ----------------------------------------------------------
def test_the_result_is_deterministic_and_independent_of_fact_order():
    b = bundle(sym_in("P1", "svc"), sym_in("P2", "svc"), sym_in("C", "web"))
    ps = [prov("P1", "mcp_tool", "a"), prov("P2", "mcp_tool", "b")]
    cs = [cons("C", "mcp_tool", "a", "s1"), cons("C", "mcp_tool", None, "s2")]
    one_way = link(facts(ps, cs), [b])
    other = link(facts(ps[::-1], cs[::-1]), [b])
    assert one_way.links == other.links


def test_every_link_and_via_reason_is_a_registry_code_valid_on_link_with_the_registry_status():
    """M3: the code, its status and its ``on`` list (a ``Via`` reason: ``Link`` or ``VIA``; ``Link (TO_UNIT)`` is a Link row)."""
    import yaml
    from pathlib import Path
    reg = {r["code"]: r for r in yaml.safe_load((Path(__file__).parents[1] / "ccg/registry/reasons.yaml").read_text())}
    on_link = lambda r: any(t == "Link" or t.startswith("Link ") for t in reg[r]["on"])      # noqa: E731
    on_via = lambda r: on_link(r) or "VIA" in reg[r]["on"]                                   # noqa: E731
    b = bundle(sym_in("P", "svc"), sym_in("P2", "job"), sym_in("C", "web"), sym_in("CE", "web", role="example"),
               sym_in("D", "prompts", path="p/d.md")) + [shell_site("sd", "C"), mention("m1", "D", "t")]
    ps = [prov("P", "http", "GET /a"), prov("P", "http", "GET /a/{}", site="x"), prov("P2", "mcp_tool", "t")]
    cs = [cons("C", "http", "GET /a", "s1"), cons("C", "http", "GET /a/{?}", "s2"), cons("C", "http", None, "s3"),
          cons("CE", "http", "GET /a", "s4"), cons("C", "mcp_tool", None, "sd"),
          cons("C1", "http", "GET /a", "s5"), cons("C1", "http", "GET /a/{?}", "s6"), cons("C1", "http", None, "s7"),
          cons("C2", "http", "GET /a", "s8")]
    b += sym_in("C1", "web", path="web/c1.py") + sym_in("C2", "web", path="web/c2.py")
    bd = FakeBindings({"s1": binding("svc"), "s2": binding("svc", "job"), "s3": binding("svc"),
                       "s4": binding(unmapped=Unmapped(hid("u9"), hid("u9"))), "s8": binding(unmapped=Unmapped(hid("u9"), hid("u9")))})
    got = link(facts(ps, cs), [b], bindings=bd)
    emitted = {l.reason for l in got.links} | {v.reason for l in got.links for v in l.sites}
    assert {"key-match", "binding-confirmed", "example-config", "partial-key", "unknown-key", "binding-narrowed", "port-shared",
            "target-not-mapped", "shell-dynamic", "prompt-mention"} <= emitted
    for l in got.links:
        assert reg[l.reason]["status"] == l.status and on_link(l.reason), (l.reason, reg[l.reason]["on"])
        for v in l.sites:
            assert reg[v.reason]["status"] == v.status and on_via(v.reason), (v.reason, reg[v.reason]["on"])


def test_lifted_role_is_never_a_link_reason_it_stays_a_consumes_reason():
    import yaml
    from pathlib import Path
    reg = {r["code"]: r for r in yaml.safe_load((Path(__file__).parents[1] / "ccg/registry/reasons.yaml").read_text())}
    assert reg["lifted-role"]["on"] == ["CONSUMES"]
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "http", "GET /a")], [cons("C", "http", "GET /a", "s1", reason="lifted-role")]), [b]))
    assert got.reason == "key-match"


def test_link_counts_report_totals_by_kind_status_reason_and_the_unknown_key_fan_out():
    b = bundle(sym_in("P1", "svc"), sym_in("P2", "svc"), sym_in("P3", "svc"), sym_in("C", "web"))
    ps = [prov("P1", "mcp_tool", "a"), prov("P2", "mcp_tool", "b"), prov("P3", "mcp_tool", "c")]
    cs = [cons("C", "mcp_tool", "a", "s1"), cons("C", "mcp_tool", None, "s2")]
    c = link(facts(ps, cs), [b]).counts
    assert c["links"] == 3 and c["by_kind"] == {"mcp_tool": 3}          # P1 is one Link for the keyed and the keyless site
    assert c["by_status"] == {"confirmed": 1, "possible": 2}
    assert c["by_reason"] == {"key-match": 1, "unknown-key": 2}
    assert c["unknown_key_consumers"] == 1 and c["max_fan_out"] == 3 and c["refused_patterns"] == 0


def test_a_keyless_and_a_keyed_site_reaching_one_provider_share_one_link_keyed_by_the_provider():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "mcp_tool", "a")], [cons("C", "mcp_tool", "a", "s1"), cons("C", "mcp_tool", None, "s2")]), [b]))
    assert got.key_norm == "a" and got.link_id == ids.link_id("mcp_tool", "a", "P", "C")
    assert {(v.site_id, v.consumer_key, v.reason) for v in got.sites} == {("s1", "a", "key-match"), ("s2", None, "unknown-key")}
    assert (got.status, got.reason) == ("confirmed", "key-match")


def test_sites_of_one_symbol_with_different_consumer_keys_are_vias_of_one_link():
    """The analyst's repro (B1): ``GET /a/1`` and ``GET /a/2`` against ``GET /a/{}`` are one ``Link`` for the provider."""
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    got = one(link(facts([prov("P", "http", "GET /a/{}")], [cons("C", "http", "GET /a/1", "s1"), cons("C", "http", "GET /a/2", "s2")]), [b]))
    assert got.key_norm == "GET /a/{}" and got.link_id == ids.link_id("http", "GET /a/{}", "P", "C")
    assert [(v.site_id, v.consumer_key) for v in got.sites] == [("s1", "GET /a/1"), ("s2", "GET /a/2")]


def test_one_call_site_with_two_alternative_keys_gives_two_vias_and_one_chain_of_evidence():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    hops = [{"kind": "callsite", "id": "s1", "ord": 0}]
    cs = [cons("C", "http", "GET /a/1", "s1", evidence={"hops": hops}), cons("C", "http", "GET /a/2", "s1", evidence={"hops": hops})]
    got = one(link(facts([prov("P", "http", "GET /a/{}")], cs), [b]))
    assert [v.consumer_key for v in got.sites] == ["GET /a/1", "GET /a/2"]
    assert len([e for e in got.evidence if e["edge"] == "LIFTED_VIA"]) == 1


def test_the_link_matcher_is_that_of_the_best_site_lowest_site_id_on_a_tie():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    cs = [cons("C", "http", "GET /a/{}", "s2"), cons("C", "http", "GET /a/7", "s1")]       # s2 exact, s1 path-template; both confirmed
    got = one(link(facts([prov("P", "http", "GET /a/{}")], cs), [b]))
    assert got.matcher == "path-template"                                                    # s1 is the lowest site id
    cs = [cons("C", "http", "GET /a/{}", "s1"), cons("C", "http", "GET /a/7", "s2")]
    assert one(link(facts([prov("P", "http", "GET /a/{}")], cs), [b])).matcher == "exact"


def test_a_link_into_a_path_outside_every_manifest_repo_is_anchored_unmapped_and_has_no_to_unit():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    bd = FakeBindings({"s1": binding(unmapped=Unmapped(None, hid("ab12")))})
    got = one(link(facts([prov("P", "http", "GET /a")], [cons("C", "http", "GET /a", "s1")]), [b], bindings=bd))
    assert (got.status, got.reason, got.provider, got.to_unit) == ("possible", "target-not-mapped", None, None)
    assert got.link_id == ids.link_id("http", "GET /a", "unmapped:" + hid("ab12"), "C")


def test_the_registry_lets_binding_confirmed_appear_on_a_via():
    import yaml
    from pathlib import Path
    reg = {r["code"]: r for r in yaml.safe_load((Path(__file__).parents[1] / "ccg/registry/reasons.yaml").read_text())}
    assert "Link" in reg["binding-confirmed"]["on"] and "VIA" in reg["binding-confirmed"]["on"]


# --- over the real shell extractor and pack runtime: LNK-20 --------------------------------------
def test_lnk_20_the_fixture_dyn_script_links_every_mcp_tool_provider_possible_shell_dynamic():
    """``eval "mcp-call $TOOL"`` (real extractor, real ``shell-mcp-call`` pack): one keyless consumer, four providers
    including ``fx-job``'s second ``reindex`` (NC-5), none collapsed, each ``possible``/``shell-dynamic``."""
    from ccg.packrt import run_packs
    from extractors.shell import extract_shell
    from packrt_helpers import CPG, pack
    recs = extract_shell(CPG / "fixtures/system1/fx-svc", ["scripts/dyn.sh"], repo_id="fx-svc", unit_id="svc",
                         history=lambda p, s: ("2024-01-01T00:00:00Z", "a" * 40), warnings=[])
    pk = pack("shell-mcp-call")
    pf = run_packs([recs], [pk])
    ps = [prov("G", "mcp_tool", "get_item"), prov("R", "mcp_tool", "reindex"), prov("L", "mcp_tool", "list_orders"),
          prov("JR", "mcp_tool", "reindex")]
    other = bundle(sym_in("G", "svc"), sym_in("R", "svc"), sym_in("L", "svc"), sym_in("JR", "job"))
    got = link(pf._replace(providers=ps), [recs, other], packs=[pk])
    assert sorted(l.provider for l in got.links) == ["G", "JR", "L", "R"]
    assert {(l.status, l.reason, l.consumer) for l in got.links} == {("possible", "shell-dynamic", pf.consumers[0]["anchorId"])}
    assert got.counts["unknown_key_consumers"] == 1 and got.counts["max_fan_out"] == 4


# --- over the real fetch pack: a lifted chain, resolved and fully unknown ------------------------
def _fetch_chain(literal: str | None):
    from packrt_helpers import call as c_, data, file_, param, sym, vs
    recs = [file_("f1", "web/client.ts", lang="javascript", unit="web"),
            sym("get", "get", fid="f1", path="web/client.ts", start=10), param("p0", "get", 0, "path", line=11),
            c_("fe", "get", "fetch", line=12, resolution="external", external="lib/fetch.js:<module>.fetch"),
            data("fe", "p0", "a0", "p0"),
            sym("caller", "caller", fid="f1", path="web/client.ts", start=30),
            c_("c1", "caller", "get", line=31, resolution="resolved", targets=("get",))]
    return recs + ([vs("c1", "a0", literal)] if literal else [])


def test_a_lifted_consumer_links_with_its_chain_in_order_and_an_unresolved_leaf_links_every_http_provider():
    from ccg.packrt import run_packs
    from packrt_helpers import pack
    fetch = pack("fetch")
    providers = bundle(sym_in("P1", "svc"), sym_in("P2", "svc"))
    ps = [prov("P1", "http", "GET /api/x"), prov("P2", "http", "POST /api/y")]
    known = run_packs([_fetch_chain("/api/x")], [fetch])
    got = one(link(known._replace(providers=ps), [_fetch_chain("/api/x"), providers], packs=[fetch]))
    assert (got.provider, got.status, got.reason) == ("P1", "confirmed", "key-match")
    assert [(e["ord"], e["id"]) for e in got.evidence if e["edge"] == "LIFTED_VIA"] == [(0, "c1"), (1, "get"), (2, "fe")]
    unknown = run_packs([_fetch_chain(None)], [fetch])
    assert unknown.consumers == []                                   # the hand-off: only `lifted` holds it
    got = link(unknown._replace(providers=ps), [_fetch_chain(None), providers], packs=[fetch])
    assert sorted(l.provider for l in got.links) == ["P1", "P2"]
    assert {(l.reason, l.consumer) for l in got.links} == {("unknown-key", "caller")}
    assert [e["id"] for e in got.links[0].evidence if e["edge"] == "LIFTED_VIA"] == ["c1", "get", "fe"]


def test_a_chain_that_repeats_an_entry_writes_one_bound_by_edge_at_its_lowest_ord():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    bd = FakeBindings({"s1": binding("svc", entries=("e-a", "e-b", "e-a"))})
    got = one(link(facts([prov("P", "mcp_tool", "t")], [cons("C", "mcp_tool", "t", "s1")]), [b], bindings=bd))
    assert [(e["ord"], e["entryId"]) for e in got.evidence if e["edge"] == "BOUND_BY"] == [(0, "e-a"), (1, "e-b")]


def test_port_evidence_travels_on_the_bound_by_entry_only_when_present():
    from ccg.linker import Binding, BoundEntry
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    bd = FakeBindings({"s1": Binding(("svc",), (BoundEntry("e1", 8200, "env-default"), BoundEntry("e2", 8000, "pack-default", "fastmcp"),
                                                 BoundEntry("e3")))})
    got = one(link(facts([prov("P", "mcp_tool", "t")], [cons("C", "mcp_tool", "t", "s1")]), [b], bindings=bd))
    ev = [e for e in got.evidence if e["edge"] == "BOUND_BY"]
    assert ev[0]["port"] == 8200 and ev[0]["portOrigin"] == "env-default" and "packId" not in ev[0]
    assert ev[1]["packId"] == "fastmcp" and "port" not in ev[2] and "portOrigin" not in ev[2]


def test_two_packs_declaring_different_link_matchers_for_a_kind_is_an_error():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    packs = [{"id": "a", "link": [{"kind": "topic", "matcher": "exact"}]}, {"id": "b", "link": [{"kind": "topic", "matcher": "glob"}]}]
    with pytest.raises(ValueError, match="different link matchers"):
        link(facts([], []), [b], packs=packs)


def test_a_binding_is_asked_once_per_site_even_when_a_site_has_several_consumer_facts():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    bd = FakeBindings({})
    link(facts([prov("P", "http", "GET /a")], [cons("C", "http", "GET /a", "s1"), cons("C", "http", "GET /b", "s1")]), [b], bindings=bd)
    assert bd.asked == ["s1"]


def test_the_link_reason_among_equal_best_sites_with_different_reasons_is_the_lowest_site_ids():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    cs = [cons("C", "http", "GET /a", "s2", status="possible", reason="partial-key"),
          cons("C", "http", "GET /a", "s1", status="possible", reason="prompt-mention")]
    got = one(link(facts([prov("P", "http", "GET /a")], cs), [b]))
    assert got.reason == "prompt-mention"


def test_a_possible_consumer_reason_that_is_not_a_link_reason_is_reported_as_partial_key():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    c = cons("C", "mcp_tool", "t", "s1", status="possible", reason="runtime-built")
    got = one(link(facts([prov("P", "mcp_tool", "t")], [c]), [b]))
    assert (got.status, got.reason) == ("possible", "partial-key")


def test_a_provider_with_a_key_is_never_matched_by_its_symbol_name():
    b = bundle(sym_in("P", "svc", name="reindex"), sym_in("C", "web"))
    assert link(facts([prov("P", "mcp_tool", "something_else")], [cons("C", "mcp_tool", "reindex", "s1")]), [b]).links == []


# --- F6, refused regex patterns, the weaken seam and config-entry anchors (dispatch-2 seam tests) ----
def test_the_packs_error_names_the_lifting_when_packs_were_passed_but_none_consumes_it():
    b = bundle(sym_in("P1", "svc"), sym_in("C", "web"))
    with pytest.raises(ValueError, match=r"^link\(\): no rule of the given packs consumes lifting 'fetch-role' \(pack 'fetch'\)$"):
        link(facts([prov("P1", "http", "GET /a")], [], [lifted_leaf("fetch", "fetch-role", "s1", "C")]), [b], packs=[{"id": "x", "rules": []}])


def test_a_refused_regex_pattern_is_counted_in_the_links_counts():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    pf = facts([prov("P", "topic", "kafka:aaaa")], [cons("C", "topic", "kafka:(a+)+$", "s1", form="regex")])
    got = link(pf, [b])
    assert got.links == [] and got.counts["refused_patterns"] == 1


def test_weaken_must_be_a_possible_reason_valid_on_link():
    from ccg.linker import Binding, BoundEntry
    e = (BoundEntry("e1"),)
    for bad in (("confirmed", "partial-key"), ("possible", "lifted-role"), ("possible", "no-such-code"), ("possible", "name-match")):
        with pytest.raises(ValueError):
            Binding(("svc",), e, weaken=bad)
    Binding(("svc",), e, weaken=("possible", "partial-key"))


def test_a_weaken_chain_gives_a_possible_link_with_its_reason_ranked_after_port_shared():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    pf = facts([prov("P", "mcp_tool", "t")], [cons("C", "mcp_tool", "t", "s1")])
    got = one(link(pf, [b], bindings=FakeBindings({"s1": binding("svc", weaken=("possible", "name-only-provider"))})))
    assert (got.status, got.reason) == ("possible", "name-only-provider")
    both = FakeBindings({"s1": binding("svc", "job", weaken=("possible", "name-only-provider"))})
    b2 = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    assert one(link(pf, [b2], bindings=both)).reason == "port-shared"


def test_a_config_entry_provider_and_consumer_survive_a_one_unit_binding_and_take_the_bundle_units():
    from linker_helpers import cfg_bundle
    b1, b2 = cfg_bundle("svc", "PE"), cfg_bundle("web", "CE", path="web/mcp.json")
    pf = facts([prov("PE", "env_var", "PORT")], [cons("CE", "env_var", "PORT", "site-ce")])
    got = one(link(pf, [b1, b2], bindings=FakeBindings({"site-ce": binding("svc")})))
    assert (got.provider, got.consumer, got.status, got.reason) == ("PE", "CE", "confirmed", "binding-confirmed")
    assert link(pf, [b1, b2], bindings=FakeBindings({"site-ce": binding("job")})).links == []        # the unit is read, not dropped


def test_a_launch_fact_anchor_takes_its_bundles_unit_and_repo_for_its_file():
    from linker_helpers import cfg_bundle
    b1 = cfg_bundle("svc", "PE", launch_id="LF", file_role="example")
    b2 = bundle(sym_in("C", "web"))
    pf = facts([prov("LF", "http", "GET /a")], [cons("C", "http", "GET /a", "s1")])
    got = one(link(pf, [b1, b2], bindings=FakeBindings({"s1": binding("svc")})))
    assert (got.provider, got.status, got.reason) == ("LF", "possible", "example-config")


def test_an_example_config_entry_gives_example_config_and_a_primary_one_does_not():
    from linker_helpers import cfg_bundle
    pf = facts([prov("PE", "env_var", "PORT")], [cons("CE", "env_var", "PORT", "s1")])
    ex = one(link(pf, [cfg_bundle("svc", "PE"), cfg_bundle("web", "CE", path="w/m.json", example=True)]))
    assert (ex.status, ex.reason) == ("possible", "example-config")
    pr = one(link(pf, [cfg_bundle("svc", "PE"), cfg_bundle("web", "CE", path="w/m.json")]))
    assert (pr.status, pr.reason) == ("confirmed", "key-match")


def test_a_config_entry_in_an_example_file_is_example_config_even_when_its_own_flag_is_false():
    from linker_helpers import cfg_bundle
    pf = facts([prov("PE", "env_var", "PORT")], [cons("CE", "env_var", "PORT", "s1")])
    got = one(link(pf, [cfg_bundle("svc", "PE"), cfg_bundle("web", "CE", path="w/m.json", file_role="example")]))
    assert got.reason == "example-config"


def test_a_config_entry_in_a_doc_file_is_not_a_caller():
    from linker_helpers import cfg_bundle
    pf = facts([prov("PE", "env_var", "PORT")], [cons("CE", "env_var", "PORT", "s1")])
    assert link(pf, [cfg_bundle("svc", "PE"), cfg_bundle("web", "CE", path="w/m.md", file_role="doc")]).links == []


# --- F11 / F13 (analyst Pass 2) ---------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["key-match", "binding-confirmed", "verdict-confirmed", "unknown-key", "binding-narrowed"])
def test_f11_weaken_refuses_a_reason_whose_registry_status_is_not_possible_or_that_is_the_matchs_own(bad):
    from ccg.linker import Binding, BoundEntry
    with pytest.raises(ValueError):
        Binding(("svc",), (BoundEntry("e1"),), weaken=("possible", bad))


@pytest.mark.parametrize("good", ["partial-key", "port-shared", "example-config", "name-only-provider", "target-not-mapped"])
def test_f11_weaken_accepts_a_possible_registry_reason_valid_on_link(good):
    from ccg.linker import Binding, BoundEntry
    Binding(("svc",), (BoundEntry("e1"),), weaken=("possible", good))


def test_f13_unmapped_ref_must_be_a_value_free_hex_id():
    for bad in ("U1", "ab12", "../../etc", "G" * 32, hid("x").upper(), hid("x") + "0", ""):
        with pytest.raises(ValueError):
            Unmapped(None, bad)
    Unmapped(None, hid("x"))


def test_f13_unmapped_ref_is_tied_to_the_unit_id_when_there_is_one():
    with pytest.raises(ValueError):
        Unmapped(hid("u1"), hid("other"))
    with pytest.raises(ValueError):
        Unmapped("not-hex", "not-hex")
    Unmapped(hid("u1"), hid("u1"))


def test_f12_the_counts_report_refused_patterns_and_refused_texts_apart():
    b = bundle(sym_in("P", "svc"), sym_in("C", "web"))
    pf = facts([prov("P", "topic", "kafka:" + "a" * 300), prov("P", "topic", "kafka:aaaa", site="x")],
               [cons("C", "topic", "kafka:ab*", "s1", form="regex"), cons("C", "topic", "kafka:(a+)+$", "s2", form="regex")])
    got = link(pf, [b])
    assert got.counts["refused_patterns"] == 1 and got.counts["refused_texts"] == 1


def test_a_prompt_mention_consumer_carries_its_server_to_the_bindings_seam():
    seen = {}

    class Spy(FakeBindings):
        def bind(self, consumer):
            seen[consumer["siteId"]] = consumer.get("server", "absent")
            return None
    b = bundle(sym_in("P", "svc"), sym_in("D", "prompts", path="p/d.md")) + [mention("m1", "D", "mcp__cypher__query"), mention("m2", "D", "./x.sh")]
    link(facts([prov("P", "mcp_tool", "query")], []), [b], bindings=Spy({}))
    assert seen == {"m1": "cypher", "m2": None}


def test_extra_counts_of_the_bindings_are_merged_into_the_links_counts():
    class WithCounts(FakeBindings):
        def counts(self):
            return {"loopback_port_residual": 2}
    b = bundle(sym_in("P", "svc"))
    assert link(facts([], []), [b], bindings=WithCounts({})).counts["loopback_port_residual"] == 2


# --- SEC B1: a hard pair budget across (pattern, literal) pairs of one link run ------------------------
def test_a_link_run_stops_evaluating_regex_pairs_past_its_budget_and_counts_them():
    b = bundle(sym_in("P1", "svc"), sym_in("P2", "svc"), sym_in("P3", "svc"), sym_in("C", "web"))
    provs = [prov(f"P{i}", "topic", f"kafka:t{i}") for i in (1, 2, 3)]
    pf = facts(provs, [cons("C", "topic", "kafka:t.", "s1", form="regex")])
    full = link(pf, [b])
    assert len(full.links) == 3 and full.counts["refused_regex_pairs"] == 0
    got = link(pf, [b], regex_pairs=2)
    assert len(got.links) == 2 and got.counts["refused_regex_pairs"] == 1


def test_the_default_link_budget_is_the_matchers_constant():
    import inspect
    from ccg.linker import matchers
    assert inspect.signature(link).parameters["regex_pairs"].default == matchers.MAX_REGEX_PAIRS == 800
