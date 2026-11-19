"""P1-26 (1): the deterministic parsers of the plain-language front (``ccg.front.parsers``).

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-26 Files (routes, ``file:line``, ``path::symbol``, qnames; diffs and stack traces only as far as
"not built", P5) and behaviour (1) (a question the parsers fully read makes no model call); ``-p1-ml.md`` §4.2 (spans are verbatim substrings, one group per anchor).

    cd cpg && .venv/bin/python -m pytest tests/test_front_parsers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccg.answer import normalize_question
from ccg.front.parsers import Surface, is_path, parse
from ccg.front.resolve import Draft

DEV = Path(__file__).resolve().parent.parent / "eval" / "dev" / "ac24.jsonl"


def groups(d: Draft) -> list[list[tuple[str, str]]]:
    return [[(s.text, s.kind_hint) for s in g.spans] for g in d.anchors]


def p(q: str):
    return parse(normalize_question(q))


# ------------------------------------------------------------------------------------------------------------------ anchors

@pytest.mark.parametrize("q,want", [
    ("POST /threads/{thread_id}/messages", [[("POST /threads/{thread_id}/messages", "route")]]),
    ("GET /workflow-runs/<run_id>/trace?", [[("GET /workflow-runs/<run_id>/trace", "route")]]),
    ("falkorchat/api.py:603", [[("falkorchat/api.py:603", "line")]]),
    ("falkorchat/api.py:603:12", [[("falkorchat/api.py:603", "line")]]),
    ("(cypher-mcp/server.py:564)", [[("cypher-mcp/server.py:564", "line")]]),
    ("cypher-mcp/server.py line 564", [[("cypher-mcp/server.py", "file"), ("line 564", "line")]]),
    ("cypher-mcp/server.py linha 564", [[("cypher-mcp/server.py", "file"), ("linha 564", "line")]]),
    ("decision-models/decisionmodels/runner.py::Runner", [[("decision-models/decisionmodels/runner.py", "file"), ("Runner", "symbol")]]),
    ("mcp_monitor/client.py::ServerConnection.call_tool()", [[("mcp_monitor/client.py", "file"), ("ServerConnection.call_tool", "symbol")]]),
    ("Services.post_message", [[("Services.post_message", "symbol")]]),
    ("`postMessage()`", [[("postMessage", "symbol")]]),
    ("image-tag.sh", [[("image-tag.sh", "file")]]),
    ("compose.yaml", [[("compose.yaml", "file")]]),
])
def test_each_anchor_form_is_one_group_of_verbatim_spans_with_no_direction(q, want):
    d = p(q)
    assert isinstance(d, Draft) and d.use_case == "impact" and d.direction is None and d.scope is None
    assert groups(d) == want
    nq = normalize_question(q)
    assert all(s.text in nq for g in d.anchors for s in g.spans)


@pytest.mark.parametrize("q,direction", [("impact a/b.py:3", "impact"), ("a/b.py:3 impact", "impact"), ("depends-on a/b.py:3", "depends-on"),
                                         ("a/b.py:3 depends-on", "depends-on"), ("Depends-On: a/b.py:3", "depends-on"), ("IMPACT — a/b.py:3?", "impact")])
def test_a_whole_remainder_use_case_word_states_the_direction(q, direction):
    assert p(q).direction == direction


@pytest.mark.parametrize("q,use_case", [("rca a/b.py:3", "root-cause"), ("code review x.py", "code-review"), ("security", "security"),
                                        ("test coverage of a.py", None)])
def test_an_unbuilt_use_case_word_is_a_draft_of_that_use_case(q, use_case):
    d = p(q)
    assert (d.use_case if d is not None else None) == use_case


def test_several_anchors_are_several_groups_and_more_than_three_is_prose():
    assert groups(p("a.py:3 b.py:4")) == [[("a.py:3", "line")], [("b.py:4", "line")]]
    assert p("a.py:3 b.py:4 c.py:5 d.py:6") is None


@pytest.mark.parametrize("q", ["scope: falkor-chat a.py:3", "repo=falkor-chat a.py:3", "repositório: falkor-chat a.py:3"])
def test_a_scope_key_is_read_as_the_scope_span(q):
    d = p(q)
    assert d.scope == "falkor-chat" and groups(d) == [[("a.py:3", "line")]] and d.scope in normalize_question(q)


# ------------------------------------------------------------------------------------------------------------------ prose is the interpreter's

def test_every_dev_question_is_prose_the_parsers_do_not_claim():
    """No dev question is a bare anchor: each states its direction (or names its anchor) in words, which only the interpreter reads."""
    rows = [json.loads(line) for line in DEV.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 14
    assert [r["qid"] for r in rows if p(r["text"]) is not None] == []


@pytest.mark.parametrize("q", ["what breaks if I change post_message?", "x.py:3 is it safe", "at home (2) what calls a.py:3", "at a.py:3", "runner", "a.py:3 and more", "impact", "GET", "/threads/x",
                               "get /threads/x", "e.g. something", "a.py::not a symbol", "Node.js", "e.g.", "i.e.", "Vue.js", "impact depends-on a.py:3"])
def test_free_prose_or_no_anchor_is_none(q):
    assert p(q) is None


# ------------------------------------------------------------------------------------------------------------------ not built: diffs and stack traces (P5)

@pytest.mark.parametrize("q", ["diff --git a/x.py b/x.py\nindex 1..2\n--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,3 @@\n-a\n+b",
                               "--- a/x.py\n+++ b/x.py\n-a\n+b", "@@ -10 +10,2 @@ def f():"])
def test_a_diff_is_the_code_review_surface(q):
    assert p(q) == Surface("diff", "code-review")


@pytest.mark.parametrize("q", ["Traceback (most recent call last):\n  File \"a/b.py\", line 3, in f\nValueError: x",
                               "  File \"a/b.py\", line 3, in f",
                               "TypeError: x is undefined\n    at render (src/app.js:10:5)\n    at main (src/index.js:2:1)",
                               "at com.x.Y.run(Y.java:12)", "Error: boom\n    at src/app.js:10:5"])
def test_a_stack_trace_is_the_root_cause_surface_even_though_it_holds_file_line_anchors(q):
    assert p(q) == Surface("stack-trace", "root-cause")


# ------------------------------------------------------------------------------------------------------------------ helpers and bounds

def test_is_path_needs_a_known_extension_and_path_characters():
    assert is_path("a/b.py") and is_path("image-tag.sh") and is_path("src/api/endpoints.ts")
    for no in ("Services.post_message", "a/b", "a b.py", ".py", "x" * 600 + ".py", "a/b.unknownext"):
        assert not is_path(no), no


def test_a_huge_token_is_never_an_anchor_and_parsing_stays_linear():
    q = "a" * 100_000 + ".py:3"
    assert p(q) is None
    assert p(("x." * 50_000) + "y") is None


@pytest.mark.parametrize("q", ["depends on a/b.py:3", "a/b.py:3 depends on", "depends on: a/b.py:3", "a/b.py:3 depends on?", "Depends On a/b.py:3", "depends_on a/b.py:3",
                               "dependson a/b.py:3"])
def test_the_spaced_phrase_depends_on_sets_no_direction_it_goes_to_the_interpreter(q):
    """``depends on X`` means what depends on X (impact) and ``X depends on`` the reverse: the phrase is ambiguous, so it is never claimed as depends-on (or any direction)."""
    assert p(q) is None


def test_a_closing_bracket_with_its_opener_in_the_token_is_kept_in_the_route_span():
    for q, span in (("GET /items/{id}", "GET /items/{id}"), ("GET /items/{id}?", "GET /items/{id}"), ("(GET /items/{id})", "GET /items/{id}"),
                    ("POST /a/<int:id>", "POST /a/<int:id>"), ("GET /a/[id]", "GET /a/[id]")):
        d = p(q)
        assert isinstance(d, Draft) and groups(d) == [[(span, "route")]], q
        assert d.anchors[0].spans[0].text in normalize_question(q)
    assert groups(p("(cypher-mcp/server.py:564)")) == [[("cypher-mcp/server.py:564", "line")]]          # an unmatched closer is still stripped
