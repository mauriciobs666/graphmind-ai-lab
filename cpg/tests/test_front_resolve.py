"""P1-26 (3), (3b): the deterministic resolver of the plain-language front over a scripted store (``ccg.front.resolve``).

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-26 behaviours (3), (3b), (7); ``docs/plans/connected-code-graph-p1-ml.md`` §4.2 (span-group draft, ``null``
direction expanded into both directions, the verbatim-substring guard, the per-group join rule, the named resolver tests) and §4.3 (the cap); ``-answer-spec.md`` §3.

The named tests take their question text and gold from the dev file ``cpg/eval/dev/ac24.jsonl`` (dev questions are for scoring and tests; held-out ones are used
nowhere). The draft each test feeds the resolver is the one ``-p1-ml.md`` §4.2 says the interpreter extracts; every span is checked to be a verbatim substring of
the dev question, so the draft is one the fabrication guard lets through.

    cd cpg && .venv/bin/python -m pytest tests/test_front_resolve.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from front_helpers import Cp, FakeStore, Sym

from ccg.answer import normalize_question
from ccg.engine import config
from ccg.engine.reader import GraphDataError
from ccg.front.draft import Draft, Group, Reading, Span
from ccg.front.resolve import CAP, NotBuiltAsk, directions, edit_distance, path_suffix_match, request_anchor, resolve_draft, server_matches

DEV = Path(__file__).resolve().parent.parent / "eval" / "dev" / "ac24.jsonl"
REPO = "graphmind-ai-lab"


def dev(qid: str) -> dict:
    for line in DEV.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["qid"] == qid:
            return row
    raise KeyError(qid)


def draft(*groups, direction="impact", use_case="impact", scope=None, readings=()):
    return Draft(use_case, direction, tuple(Group(tuple(Span(t, k) for t, k in g)) for g in groups), scope=scope, readings=tuple(readings))


def as_gold(outcome) -> dict:
    """The resolved outcome as the dev file's gold ``Request`` shape (ids dropped: the gold carries none)."""
    assert outcome.status == "resolved", outcome
    a = {k: v for k, v in request_anchor(outcome.anchor).items() if k not in ("symbolId", "cpId", "entryId")}
    return {"anchors": [a], "direction": outcome.direction, "scope": None, "useCase": "impact"}


def options_as_gold(outcome) -> list[dict]:
    out = []
    for o in outcome.options:
        anchors = [{k: v for k, v in a.items() if k not in ("symbolId", "cpId", "entryId")} for a in o["anchors"]]
        out.append({"anchors": anchors, "direction": o["direction"], "scope": None, "useCase": o["useCase"]})
    return out


def run(store, d, qid=None, question=None):
    q = normalize_question(dev(qid)["text"] if qid else question)
    for g in d.anchors:
        for s in g.spans:
            assert s.text in q, (s.text, q)                   # a draft the fabrication guard lets through
    return resolve_draft(store, d, q, store.repos)


# ------------------------------------------------------------------------------------------------------------------ the dev world

def dev_store() -> FakeStore:
    """The dev questions' gold anchors, each with distractors that only the join rule removes."""
    syms = [
        Sym("decision-models/decisionmodels/runner.py::Runner", "decision-models/decisionmodels/runner.py", 10, 90),
        Sym("model-bench/modelbench/runner.py::Runner", "model-bench/modelbench/runner.py", 5, 40),                  # same name, other file
        Sym("decision-models/decisionmodels/fit.py::runner", "decision-models/decisionmodels/fit.py", 3, 8),         # same nameNorm, other case
        Sym("falkor-chat/server/falkorchat/api.py::build_router", "falkor-chat/server/falkorchat/api.py", 400, 900),
        Sym("falkor-chat/server/falkorchat/api.py::build_router.check_demo_readiness", "falkor-chat/server/falkorchat/api.py", 598, 640, depth=1),
        Sym("cypher-mcp/server.py::authorize_write", "cypher-mcp/server.py", 540, 600),
        Sym("cypher-mcp/server.py::main", "cypher-mcp/server.py", 610, 700),
        Sym("salesperson/src/api/endpoints.ts::postMessage", "salesperson/src/api/endpoints.ts", 20, 30),
        Sym("salesperson/src/hooks/useChat.ts::postMessage", "salesperson/src/hooks/useChat.ts", 7, 19),
        Sym("falkor-chat/server/falkorchat/mcp_server.py::send_message", "falkor-chat/server/falkorchat/mcp_server.py", 50, 80),
        Sym("mcp-monitor/tests/fake_server.py::send_message", "mcp-monitor/tests/fake_server.py", 5, 9),
        Sym("falkor-chat/server/falkorchat/mcp_server.py::ingest_documents", "falkor-chat/server/falkorchat/mcp_server.py", 90, 120),
        Sym("falkor-chat/server/falkorchat/api.py::post_thread_message", "falkor-chat/server/falkorchat/api.py", 300, 330),
    ]
    cps = [
        Cp("send_message", "mcp_tool", "falkor-chat/server/falkorchat/mcp_server.py::send_message"),
        Cp("send_message", "mcp_tool", "mcp-monitor/tests/fake_server.py::send_message"),                          # same tool, another server
        Cp("ingest_documents", "mcp_tool", "falkor-chat/server/falkorchat/mcp_server.py::ingest_documents"),
        Cp("POST /threads/{}/messages", "http", "falkor-chat/server/falkorchat/api.py::post_thread_message"),
        Cp("GET /threads/{}/messages", "http", "falkor-chat/server/falkorchat/api.py::post_thread_message"),
    ]
    return FakeStore(syms, cps=cps)


# ------------------------------------------------------------------------------------------------------------------ the named resolver tests (-p1-ml.md §4.2)

def test_qw30_symbol_and_file_resolve_by_qname_with_the_file_as_a_path_suffix():
    row = dev("QW-30")
    got = run(dev_store(), draft([("Runner", "symbol"), ("decisionmodels/runner.py", "file")]), "QW-30")
    assert as_gold(got) == row["gold"] and got.lookups == ()


def test_qw27_partial_file_and_line_resolve_to_the_enclosing_element_through_the_full_path():
    """QW-27 with the repo-relative path: the enclosing element (the nested function, narrowest interval) in the depends-on direction."""
    row = dev("QW-27")
    q = "falkor-chat/server/falkorchat/api.py:603 — do que essa função depende?"
    got = run(dev_store(), draft([("falkor-chat/server/falkorchat/api.py:603", "line")], direction="depends-on"), question=q)
    assert as_gold(got) == row["gold"]


def test_qw27_partial_file_line_as_asked_resolves_to_gold():
    """The dev question as asked: a partial path with a line, found by the suffix read after the exact reads miss (``-p1-graph.md`` §5)."""
    row = dev("QW-27")
    store = dev_store()
    got = run(store, draft([("falkorchat/api.py:603", "line")], direction="depends-on"), "QW-27")
    assert as_gold(got) == row["gold"]
    names = store.names()
    assert names.index("file_by_path") < names.index("skipped_by_path") < names.index("file_by_suffix")      # exact first, then the fallback


def test_qw27_a_bare_filename_over_several_files_is_ambiguous_never_a_pick():
    store = FakeStore([Sym("falkor-chat/server/falkorchat/api.py::f", "falkor-chat/server/falkorchat/api.py", 1, 9),
                       Sym("salesperson/api.py::g", "salesperson/api.py", 1, 9)])
    got = resolve_draft(store, draft([("api.py:5", "line")]), "api.py:5", (REPO,))
    assert got.status == "ambiguous" and got.ambiguity_kind == "several-symbols"
    assert [o["anchors"][0]["qname"] for o in got.options] == ["falkor-chat/server/falkorchat/api.py::f", "salesperson/api.py::g"]


def test_qw13_tool_and_server_restrict_the_candidates_to_that_server():
    row = dev("QW-13")
    store = dev_store()
    got = run(store, draft([("send_message", "tool"), ("falkor-chat", "server")]), "QW-13")
    assert as_gold(got) == row["gold"]
    assert got.anchor.cp_id == Cp("send_message", "mcp_tool", "falkor-chat/server/falkorchat/mcp_server.py::send_message").cp_id
    without = run(dev_store(), draft([("send_message", "tool")]), "QW-13")                 # the server span is what removes the other one
    assert without.status == "ambiguous" and without.ambiguity_kind == "several-symbols" and len(without.options) == 2


def test_qw7_file_then_symbol_in_reverse_order_resolve_the_same_way():
    row = dev("QW-7")
    for groups in ([("endpoints.ts", "file"), ("postMessage", "symbol")], [("postMessage", "symbol"), ("endpoints.ts", "file")]):
        got = run(dev_store(), draft(groups), "QW-7")
        assert as_gold(got) == row["gold"], groups


def test_qw47_a_null_direction_is_both_directions_as_ranked_options():
    row = dev("QW-47")
    got = run(dev_store(), draft([("ingest_documents", "tool"), ("MCP", "server")], direction=None), "QW-47")
    assert got.status == "ambiguous" and got.ambiguity_kind == "two-use-cases"
    gold = lambda xs: {json.dumps(x, sort_keys=True) for x in xs}                          # noqa: E731  (the scorer compares options as a set)
    assert gold(options_as_gold(got)) == gold(row["goldOptions"])
    assert [o["direction"] for o in got.options] == ["impact", "depends-on"]                # presentation: impact first
    assert [o["rank"] for o in got.options] == [1, 2]


def test_qw3_resolver_side_a_draft_that_states_impact_resolves_that_one_reading():
    """Resolver side only: the draft handed in already states impact, so one reading comes back. That the interpreter returns a stated direction (not ``null``) for QW-3
    ("blast radius") is dispatch 2's negative test, not this one."""
    row = dev("QW-3")
    got = run(dev_store(), draft([("cypher-mcp/server.py", "file"), ("line 564", "line")]), "QW-3")
    assert as_gold(got) == row["gold"]


def test_qw11_resolver_side_a_draft_that_states_impact_resolves_the_route():
    """Resolver side only (QW-11 "what calls this route"): the interpreter-returns-a-direction negative is dispatch 2's."""
    row = dev("QW-11")
    got = run(dev_store(), draft([("POST /threads/{thread_id}/messages", "route")]), "QW-11")
    assert as_gold(got) == row["gold"]


@pytest.mark.parametrize("qid", ["QW-3", "QW-11"])
def test_the_negatives_never_come_back_with_a_null_direction_from_the_deterministic_path(qid):
    """Their direction is stated in prose, so the parsers do not claim them (``None``: the interpreter's) and never offer both directions on their own."""
    from ccg.front.parsers import parse
    assert parse(normalize_question(dev(qid)["text"])) is None


# ------------------------------------------------------------------------------------------------------------------ directions and options

def test_directions_expand_null_into_both_and_take_the_readings():
    assert directions(draft(direction=None)) == ("impact", "depends-on")
    assert directions(draft(direction="depends-on")) == ("depends-on",)
    assert directions(draft(direction="depends-on", readings=[Reading("impact", "impact")])) == ("impact", "depends-on")
    assert directions(draft(direction=None, use_case="root-cause", readings=[Reading("impact", "depends-on")])) == ("depends-on",)
    assert directions(draft(use_case="security", direction=None)) == ()


def test_four_symbols_times_a_null_direction_are_eight_complete_readings_candidate_major():
    syms = [Sym(f"m{i}.py::run", f"m{i}.py") for i in range(4)]
    got = resolve_draft(FakeStore(syms), draft([("run", "symbol")], direction=None), "run", (REPO,))
    assert got.status == "ambiguous" and got.ambiguity_kind == "several-symbols"
    assert len(got.options) == 8 > CAP
    assert [(o["anchors"][0]["qname"], o["direction"], o["rank"]) for o in got.options] == [
        (f"m{i}.py::run", d, 2 * i + k + 1) for i in range(4) for k, d in enumerate(("impact", "depends-on"))]


def test_complete_readings_are_bounded_by_the_engines_option_limit_in_whole_candidates():
    syms = [Sym(f"m{i:02d}.py::run", f"m{i:02d}.py") for i in range(config.OPTIONS_MAX + 5)]
    got = resolve_draft(FakeStore(syms), draft([("run", "symbol")], direction=None), "run", (REPO,))
    assert len(got.options) == config.OPTIONS_MAX
    assert [o["direction"] for o in got.options[-2:]] == ["impact", "depends-on"]            # the last kept candidate carries both directions
    one = resolve_draft(FakeStore(syms), draft([("run", "symbol")]), "run", (REPO,))
    assert len(one.options) == config.OPTIONS_MAX


def test_several_symbols_with_one_direction_list_every_reading():
    syms = [Sym(f"m{i}.py::run", f"m{i}.py") for i in range(3)]
    got = resolve_draft(FakeStore(syms), draft([("run", "symbol")]), "run", (REPO,))
    assert [o["anchors"][0]["qname"] for o in got.options] == ["m0.py::run", "m1.py::run", "m2.py::run"]
    assert {o["direction"] for o in got.options} == {"impact"}


def test_a_name_with_many_definitions_is_filtered_by_file_over_all_of_them_not_a_capped_list():
    syms = [Sym(f"pkg{i:02d}/mod.py::run", f"pkg{i:02d}/mod.py") for i in range(config.OPTIONS_MAX + 5)]
    got = resolve_draft(FakeStore(syms), draft([("run", "symbol"), ("pkg24/mod.py", "file")]), "run pkg24/mod.py", (REPO,))
    assert got.status == "resolved" and got.anchor.qname == "pkg24/mod.py::run"


# ------------------------------------------------------------------------------------------------------------------ the fabrication guard

def test_a_span_that_is_not_in_the_question_is_dropped_and_an_emptied_group_names_no_anchor():
    store = dev_store()
    got = resolve_draft(store, draft([("Runner", "symbol"), ("decisionmodels/runner.py", "file")]), "what about Runner?", (REPO,))
    assert got.status == "ambiguous" and got.ambiguity_kind == "several-symbols"            # the file span was fabricated: Runner alone is two readings
    got = resolve_draft(store, draft([("Runnr", "symbol")]), "what about Runner?", (REPO,))
    assert (got.status, got.ambiguity_kind, got.options) == ("ambiguous", "no-anchor", ()) and store.names().count("symbol_by_name") == 1


def test_the_guard_is_case_and_byte_exact():
    for span in ("runner", "Runner.", "Runner?"):
        store = dev_store()
        got = resolve_draft(store, draft([(span, "symbol")]), "what breaks if Runner changes", (REPO,))
        assert (got.status, got.ambiguity_kind, store.calls) == ("ambiguous", "no-anchor", []), span


# ------------------------------------------------------------------------------------------------------------------ join rules and lookups

def test_path_suffix_matches_only_at_a_segment_boundary():
    assert path_suffix_match("decision-models/decisionmodels/runner.py", "decisionmodels/runner.py")
    assert path_suffix_match("a/b.py", "a/b.py") and path_suffix_match("a/b.py", "./a/b.py")
    assert not path_suffix_match("x/xdecisionmodels/runner.py", "decisionmodels/runner.py")
    assert not path_suffix_match(None, "a.py") and not path_suffix_match("a.py", "")


def test_server_matching_drops_the_noise_words_and_matches_a_path_segment():
    assert server_matches("falkor-chat/server/x.py", "falkor-chat")
    assert server_matches("falkor-chat/server/x.py", "falkor-chat MCP")
    assert server_matches("anything/x.py", "MCP")                                            # names no server: no restriction
    assert not server_matches("mcp-monitor/tests/x.py", "falkor-chat")
    assert not server_matches("falkor-chatty/x.py", "falkor-chat")


def test_a_named_file_that_is_not_found_is_unresolved_with_one_lookup_per_repo():
    store = FakeStore([Sym("a.py::f", "a.py")], repos=("r1", "r2"))
    got = resolve_draft(store, draft([("nope.py:3", "line")]), "nope.py:3", ("r1", "r2"))
    assert got.status == "unresolved"
    assert got.lookups == ({"path": "nope.py", "repoId": "r1", "state": "not-found"}, {"path": "nope.py", "repoId": "r2", "state": "not-found"})


def test_a_skipped_file_is_unresolved_as_skipped_file_with_its_reason_and_unit():
    unit = "e" * 32
    store = FakeStore([], skipped=[(REPO, "tests/test_a.py", unit, "test")])
    got = resolve_draft(store, draft([("tests/test_a.py:4", "line")]), "tests/test_a.py:4", (REPO,))
    assert got.lookups == ({"path": "tests/test_a.py", "repoId": REPO, "state": "skipped-file", "reason": "test", "unitId": unit},)


def test_a_resolved_anchor_keeps_the_lookups_of_another_named_file_that_failed():
    store = FakeStore([Sym("a.py::f", "a.py", 1, 9)])
    got = resolve_draft(store, draft([("a.py:3", "line")], [("gone.py:1", "line")]), "a.py:3 gone.py:1", (REPO,))
    assert got.status == "resolved" and got.anchor.qname == "a.py::f" and got.lookups == ({"path": "gone.py", "repoId": REPO, "state": "not-found"},)


def test_a_mapped_file_with_no_element_at_the_line_names_no_anchor():
    store = FakeStore([Sym("a.py::f", "a.py", 1, 9)])
    got = resolve_draft(store, draft([("a.py:50", "line")]), "a.py:50", (REPO,))
    assert (got.status, got.ambiguity_kind, got.options) == ("ambiguous", "no-anchor", ())


def test_a_config_file_alone_offers_its_entries():
    store = FakeStore([], config=[(REPO, "compose.yaml", "services.web.ports", 4), (REPO, "compose.yaml", "services.db.image", 9)], extra_files=[(REPO, "compose.yaml")])
    got = resolve_draft(store, draft([("compose.yaml", "file")]), "compose.yaml", (REPO,))
    assert got.status == "ambiguous" and [o["anchors"][0]["keyPath"] for o in got.options] == ["services.db.image", "services.web.ports"]


def test_an_unknown_repo_is_skipped_and_a_file_with_no_known_repo_is_not_built():
    store = FakeStore([Sym("a.py::f", "a.py", 1, 9)])
    got = resolve_draft(store, draft([("a.py:3", "line")]), "a.py:3", ("not-in-store", REPO))
    assert got.status == "resolved"
    with pytest.raises(NotBuiltAsk) as e:
        resolve_draft(store, draft([("a.py:3", "line")]), "a.py:3", ())
    assert e.value.detail == {"surface": "file-without-repos"}


def test_a_malformed_name_row_is_a_graph_data_error():
    class Bad(FakeStore):
        def _symbol_by_name(self, name):
            return [("s", "r", None, 5, "p", 1)]
    with pytest.raises(GraphDataError):
        resolve_draft(Bad([]), draft([("run", "symbol")]), "run", (REPO,))


# ------------------------------------------------------------------------------------------------------------------ unknown names: did-you-mean

def test_an_unknown_symbol_in_a_found_file_suggests_that_files_symbols_by_edit_distance_then_qname_capped():
    syms = [Sym(f"a.py::{n}", "a.py") for n in ("post_mesage", "post_massage", "zz", "get_message", "post", "postmessage_x", "aa")]
    store = FakeStore(syms)
    got = resolve_draft(store, draft([("post_message", "symbol"), ("a.py", "file")]), "post_message in a.py", (REPO,))
    assert got.status == "ambiguous" and got.ambiguity_kind == "unknown-name" and len(got.options) == CAP
    names = [o["anchors"][0]["qname"] for o in got.options]
    assert names == ["a.py::post_massage", "a.py::post_mesage", "a.py::get_message", "a.py::postmessage_x", "a.py::post"]     # distance 1, 1, 3, 3, 8; ties by qname
    assert [edit_distance("post_message", n.partition("::")[2]) for n in names] == [1, 1, 3, 3, 8]


def test_an_unknown_symbol_in_a_missing_file_is_unresolved_on_the_file():
    got = resolve_draft(FakeStore([]), draft([("f", "symbol"), ("gone.py", "file")]), "f in gone.py", (REPO,))
    assert got.status == "unresolved" and got.lookups[0]["state"] == "not-found"


def test_an_unknown_contract_is_unknown_name_with_no_options():
    got = resolve_draft(dev_store(), draft([("DELETE /nope", "route")]), "DELETE /nope", (REPO,))
    assert (got.status, got.ambiguity_kind, got.options) == ("ambiguous", "unknown-name", ())


def test_a_tool_whose_server_matches_none_suggests_the_unrestricted_ones():
    got = resolve_draft(dev_store(), draft([("send_message", "tool"), ("kiro", "server")]), "send_message on kiro", (REPO,))
    assert got.status == "ambiguous" and got.ambiguity_kind == "unknown-name" and len(got.options) == 2


def test_edit_distance():
    assert [edit_distance(a, b) for a, b in [("", ""), ("a", ""), ("kitten", "sitting"), ("abc", "abc"), ("ab", "ba")]] == [0, 1, 3, 0, 2]


# ------------------------------------------------------------------------------------------------------------------ not built

def test_a_scope_is_not_built_with_the_request_echoed():
    with pytest.raises(NotBuiltAsk) as e:
        resolve_draft(dev_store(), draft([("Runner", "symbol")], scope="decision-models"), "Runner in decision-models only", (REPO,))
    assert e.value.detail["request"]["scope"] == "decision-models" and e.value.detail["request"]["anchors"] == [[{"text": "Runner", "kindHint": "symbol"}]]


def test_an_unbuilt_use_case_is_not_built_with_the_request_echoed_and_reads_nothing():
    store = dev_store()
    with pytest.raises(NotBuiltAsk) as e:
        resolve_draft(store, draft([("Runner", "symbol")], use_case="root-cause", direction=None), "why does Runner fail", (REPO,))
    assert e.value.detail == {"request": {"useCase": "root-cause", "direction": None, "anchors": [[{"text": "Runner", "kindHint": "symbol"}]]}, "built": ["impact"]}
    assert store.calls == []


def test_two_groups_that_each_resolve_are_not_built_one_anchor_per_ask():
    with pytest.raises(NotBuiltAsk, match="one anchor"):
        resolve_draft(dev_store(), draft([("authorize_write", "symbol")], [("check_demo_readiness", "symbol")]), "authorize_write and check_demo_readiness", (REPO,))


def test_the_draft_rejects_values_outside_its_enums():
    for bad in (lambda: Span("x", "class"), lambda: Span("", "symbol"), lambda: Group(()), lambda: Draft("rca", None, ()),
                lambda: Draft("impact", "both", ()), lambda: Draft("impact", None, tuple(Group((Span("x", "symbol"),)) for _ in range(4)))):
        with pytest.raises(ValueError):
            bad()


def test_a_file_mapped_in_one_repo_records_no_failed_lookup_for_the_others():
    store = FakeStore([Sym("a.py::f", "a.py", 1, 9, repo="r1")], repos=("r1", "r2"))
    got = resolve_draft(store, draft([("a.py:3", "line")]), "a.py:3", ("r1", "r2"))
    assert got.status == "resolved" and got.lookups == ()


def test_a_name_match_is_exact_on_the_symbol_part_the_case_folded_name_only_finds_candidates():
    got = run(dev_store(), draft([("Runner", "symbol")]), question="who uses Runner")
    assert got.status == "ambiguous" and [o["anchors"][0]["qname"] for o in got.options] == [
        "decision-models/decisionmodels/runner.py::Runner", "model-bench/modelbench/runner.py::Runner"]       # never fit.py::runner
    store = FakeStore([Sym("c.py::ServerConnection.call_tool", "c.py"), Sym("c.py::Other.call_tool", "c.py")])
    got = resolve_draft(store, draft([("ServerConnection.call_tool", "symbol")]), "ServerConnection.call_tool", (REPO,))
    assert got.status == "resolved" and got.anchor.qname == "c.py::ServerConnection.call_tool"


# ------------------------------------------------------------------------------------------------------------------ partial paths: the suffix reads (-p1-graph.md §5)

SUFFIX_READS = ("file_by_suffix", "skipped_by_suffix")


def test_an_exact_repo_relative_path_wins_and_no_suffix_read_runs():
    store = FakeStore([Sym("api.py::root", "api.py", 1, 9), Sym("server/falkorchat/api.py::deep", "server/falkorchat/api.py", 1, 9)])
    got = resolve_draft(store, draft([("api.py:3", "line")]), "api.py:3", (REPO,))
    assert got.status == "resolved" and got.anchor.qname == "api.py::root"
    assert not set(store.names()) & set(SUFFIX_READS)


def test_an_exact_skipped_hit_is_not_mapped_and_the_suffix_reads_do_not_run():
    store = FakeStore([Sym("server/tests/api.py::deep", "server/tests/api.py", 1, 9)], skipped=[(REPO, "tests/api.py", "u" * 32, "test")])
    got = resolve_draft(store, draft([("tests/api.py:3", "line")]), "tests/api.py:3", (REPO,))
    assert got.status == "unresolved" and got.lookups[0]["state"] == "skipped-file" and got.lookups[0]["reason"] == "test"
    assert not set(store.names()) & set(SUFFIX_READS)


def test_the_suffix_is_sent_at_a_segment_boundary_with_the_slash_parameter_and_every_known_repo():
    store = FakeStore([Sym("server/xapi.py::x", "server/xapi.py", 1, 9)], repos=("r1", "r2"))
    got = resolve_draft(store, draft([("api.py:3", "line")]), "api.py:3", ("r1", "r2", "unknown"))
    assert got.status == "unresolved" and {x["repoId"] for x in got.lookups} == {"r1", "r2"}           # xapi.py is not api.py
    sent = [p for n, p in store.calls if n in SUFFIX_READS]
    assert sent == [{"repos": ["r1", "r2"], "sep": "/", "suffix": "/api.py"}] * 2


def test_a_leading_dot_slash_is_removed_from_the_suffix():
    store = FakeStore([Sym("a/b.py::f", "a/b.py", 1, 9)])
    got = resolve_draft(store, draft([("./b.py:3", "line")]), "./b.py:3", (REPO,))
    assert got.status == "resolved"
    assert [p["suffix"] for n, p in store.calls if n == "file_by_suffix"] == ["/b.py"]


def test_one_partial_hit_per_repo_in_two_repos_is_ambiguous_with_both():
    syms = [Sym("server/falkorchat/api.py::f", "server/falkorchat/api.py", 1, 9, repo="r1"), Sym("server/falkorchat/api.py::f", "server/falkorchat/api.py", 1, 9, repo="r2")]
    got = resolve_draft(FakeStore(syms, repos=("r1", "r2")), draft([("falkorchat/api.py:3", "line")]), "falkorchat/api.py:3", ("r1", "r2"))
    assert got.status == "ambiguous" and got.ambiguity_kind == "several-symbols" and {o["anchors"][0]["repoId"] for o in got.options} == {"r1", "r2"}


def test_several_matching_files_are_never_picked_even_when_only_one_has_an_element_at_the_line():
    store = FakeStore([Sym("a/api.py::f", "a/api.py", 1, 9), Sym("b/api.py::g", "b/api.py", 50, 60)])
    got = resolve_draft(store, draft([("api.py:3", "line")]), "api.py:3", (REPO,))
    assert got.status == "ambiguous" and got.ambiguity_kind == "no-anchor"
    assert [o["anchors"][0]["qname"] for o in got.options] == ["a/api.py::f"]


def test_a_bare_partial_file_name_over_several_files_is_ambiguous_with_no_anchor():
    store = FakeStore([Sym("a/api.py::f", "a/api.py", 1, 9), Sym("b/api.py::g", "b/api.py", 1, 9)])
    got = resolve_draft(store, draft([("api.py", "file")]), "api.py", (REPO,))
    assert (got.status, got.ambiguity_kind, got.options) == ("ambiguous", "no-anchor", ())


def test_more_than_options_max_partial_matches_are_ambiguous_truncated_never_a_pick():
    n = config.OPTIONS_MAX + 2
    store = FakeStore([Sym(f"d{i:02d}/z.py::f", f"d{i:02d}/z.py", 1, 9) for i in range(n)])
    got = resolve_draft(store, draft([("z.py:3", "line")]), "z.py:3", (REPO,))
    assert got.status == "ambiguous" and got.ambiguity_kind == "several-symbols" and len(got.options) == config.OPTIONS_MAX
    assert [o["anchors"][0]["path"] for o in got.options] == [f"d{i:02d}/z.py" for i in range(config.OPTIONS_MAX)]       # the lowest by (repo, path)
    assert store.names().count("symbols_of_file") == config.OPTIONS_MAX                                                    # no read for a match past the bound


def test_a_skipped_only_partial_match_is_not_mapped_with_its_reason_not_not_found():
    unit = "e" * 32
    store = FakeStore([], skipped=[(REPO, "tests/api.py", unit, "test")])
    got = resolve_draft(store, draft([("api.py:4", "line")]), "api.py:4", (REPO,))
    assert got.status == "unresolved"
    assert got.lookups == ({"path": "tests/api.py", "repoId": REPO, "state": "skipped-file", "reason": "test", "unitId": unit},)


def test_a_skipped_match_beside_one_file_match_is_reported_as_a_lookup_on_the_resolved_answer():
    unit = "e" * 32
    store = FakeStore([Sym("server/api.py::f", "server/api.py", 1, 9)], skipped=[(REPO, "tests/api.py", unit, "test")])
    got = resolve_draft(store, draft([("api.py:3", "line")]), "api.py:3", (REPO,))
    assert got.status == "resolved" and got.anchor.qname == "server/api.py::f"
    assert got.lookups == ({"path": "tests/api.py", "repoId": REPO, "state": "skipped-file", "reason": "test", "unitId": unit},)


def test_a_partial_path_matching_nothing_is_one_not_found_lookup_per_repo():
    store = FakeStore([Sym("a.py::f", "a.py")], repos=("r1", "r2"))
    got = resolve_draft(store, draft([("nope/a.py:3", "line")]), "nope/a.py:3", ("r1", "r2"))
    assert got.status == "unresolved" and [(x["repoId"], x["state"]) for x in got.lookups] == [("r1", "not-found"), ("r2", "not-found")]


def test_a_malformed_suffix_row_is_a_graph_data_error():
    class Bad(FakeStore):
        def _file_by_suffix(self, repos, sep, suffix):
            return [("r", "p", "f")]
    with pytest.raises(GraphDataError):
        resolve_draft(Bad([]), draft([("nope/a.py:3", "line")]), "nope/a.py:3", (REPO,))


# ------------------------------------------------------------------------------------------------------------------ a group that found nothing is not dropped

def test_an_unknown_symbol_beside_a_resolving_group_is_ambiguous_unknown_name_not_silently_dropped():
    store = FakeStore([Sym("a.py::f", "a.py", 1, 9), Sym("a.py::Typo", "a.py", 20, 30)])
    got = resolve_draft(store, draft([("a.py:3", "line")], [("Typu.baz", "symbol")]), "a.py:3 Typu.baz", (REPO,))
    assert got.status == "ambiguous" and got.ambiguity_kind == "unknown-name"
    assert "a.py::f" in [o["anchors"][0]["qname"] for o in got.options]                      # the reading that did resolve is offered, not lost


def test_an_empty_group_beside_a_resolving_group_is_ambiguous_no_anchor_not_silently_dropped():
    store = FakeStore([Sym("a.py::f", "a.py", 1, 9), Sym("b.py::g", "b.py", 1, 9)])
    got = resolve_draft(store, draft([("a.py:3", "line")], [("b.py:50", "line")]), "a.py:3 b.py:50", (REPO,))
    assert got.status == "ambiguous" and got.ambiguity_kind == "no-anchor"
    assert [o["anchors"][0]["qname"] for o in got.options] == ["a.py::f"]


def test_a_failed_path_beside_a_resolving_group_still_resolves_with_the_lookup():
    store = FakeStore([Sym("a.py::f", "a.py", 1, 9)])
    got = resolve_draft(store, draft([("a.py:3", "line")], [("gone.py:1", "line")]), "a.py:3 gone.py:1", (REPO,))
    assert got.status == "resolved" and got.lookups[0]["path"] == "gone.py"


# ------------------------------------------------------------------------------------------------------------------ did-you-mean distance

def test_did_you_mean_compares_the_last_dotted_segment_of_a_nested_symbol():
    syms = [Sym("a.py::Outer.helpr", "a.py"), Sym("a.py::kelper", "a.py"), Sym("a.py::zzzzzzzz", "a.py")]
    got = resolve_draft(FakeStore(syms), draft([("helper", "symbol"), ("a.py", "file")]), "helper in a.py", (REPO,))
    assert [o["anchors"][0]["qname"] for o in got.options] == ["a.py::Outer.helpr", "a.py::kelper", "a.py::zzzzzzzz"]
    # whole-string distance would put kelper (1) before Outer.helpr (the class prefix adds 6): the segment compare ties them at 1 and the rank key orders them


# ------------------------------------------------------------------------------------------------------------------ which empty groups count as unfound

def _srv_store():
    return FakeStore([Sym("srv/a.py::f", "srv/a.py", 1, 9), Sym("srv/b.py::g", "srv/b.py", 1, 9)], extra_files=[(REPO, "README.md")])


@pytest.mark.parametrize("extra", [[("foo", "other")], [("hello", "server")], [("foo", "other"), ("hello", "server")], [("README.md", "file")], [("srv/b.py", "file")]],
                         ids=["other-only", "server-only", "other-and-server", "mapped-plain-file", "redundant-mapped-code-file"])
def test_a_group_that_names_nothing_to_find_does_not_turn_a_resolving_question_into_an_ask(extra):
    q = "srv/a.py:3 foo hello README.md srv/b.py"
    got = resolve_draft(_srv_store(), draft([("srv/a.py:3", "line")], extra), q, (REPO,))
    assert got.status == "resolved" and got.anchor.qname == "srv/a.py::f"


@pytest.mark.parametrize("extra,kind", [([("Typu.baz", "symbol")], "unknown-name"), ([("DELETE /nope", "route")], "unknown-name"), ([("no_such_tool", "tool")], "unknown-name"),
                                   ([("srv/b.py:50", "line")], "no-anchor")],
                         ids=["unknown-symbol", "unknown-route", "unknown-tool", "line-with-no-element"])
def test_a_name_bearing_group_that_finds_nothing_still_asks(extra, kind):
    q = "srv/a.py:3 Typu.baz DELETE /nope no_such_tool srv/b.py:50"
    got = resolve_draft(_srv_store(), draft([("srv/a.py:3", "line")], extra), q, (REPO,))
    assert (got.status, got.ambiguity_kind) == ("ambiguous", kind)
    assert "srv/a.py::f" in [o["anchors"][0]["qname"] for o in got.options]


def test_an_exact_skipped_hit_in_one_repo_wins_over_a_suffix_file_in_another():
    """r1 holds ``a.py`` as a skipped file, r2 holds ``x/a.py``: ``a.py:3`` is the skipped file (``-p1-graph.md`` §5), never r2's file by suffix."""
    unit = "e" * 32
    store = FakeStore([Sym("x/a.py::f", "x/a.py", 1, 9, repo="r2")], repos=("r1", "r2"), skipped=[("r1", "a.py", unit, "test")])
    got = resolve_draft(store, draft([("a.py:3", "line")]), "a.py:3", ("r1", "r2"))
    assert got.status == "unresolved"
    assert [(x["repoId"], x["state"]) for x in got.lookups] == [("r1", "skipped-file"), ("r2", "not-found")]
    assert not set(store.names()) & {"file_by_suffix", "skipped_by_suffix"}
