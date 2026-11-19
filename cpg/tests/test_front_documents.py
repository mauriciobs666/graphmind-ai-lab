"""P1-26 (1), (3), (7): the front from a question to a canonical document, over the real engine builder (``ccg.front.Front``, ``ccg.front.asking_document``).

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-26 behaviours (1) (the parsers run first; what they read makes no model call), (3) (the front builds the
canonical ``ambiguous`` and ``unresolved`` documents, ``options`` with ``rank``), (7) (an unbuilt use case is understood and answered "not built" with the request echoed);
``-answer-spec.md`` §3 (an asking answer ran no query; the mixed case keeps the failed named path in ``lookups``) and §6 (``facts_bytes``); AC-16 (a plain-language
ask that resolves gives the facts of the same structured ask).

    cd cpg && .venv/bin/python -m pytest tests/test_front_documents.py
"""
from __future__ import annotations

import pytest
import test_engine_answer as TA
from engine_answer_helpers import SNAP
from front_helpers import FrontReader

from ccg import answer
from ccg.engine import bfs, config
from ccg.engine import anchors as A
from ccg.engine.answer_build import build_answer
from ccg.front import Front, NotBuiltAsk, asking_document
from ccg.front.resolve import Outcome


def world(flow=True):
    g, ids_ = TA.caller_world(flow=flow)
    return g, ids_


def front_over(g, built=None):
    """A front over ``g``'s repo whose builder opens a fresh reader per answer (as the client's does), recording the readers it opened."""
    built = [] if built is None else built

    def build(anchor, direction):
        r = FrontReader(g)
        built.append(r)
        return build_answer(r, anchor, question=direction)
    return Front((g.repo,), build)


def structured(g, direction="impact"):
    """The structured ask's answer, as ``clients.common.Client`` builds it: resolve on one reader, build on a fresh one."""
    anchor = A.resolve_symbol(FrontReader(g), g.repo, "m.py::A").anchors[0]
    return build_answer(FrontReader(g), anchor, question=direction)


# ------------------------------------------------------------------------------------------------------------------ resolved: the engine's answer

@pytest.mark.parametrize("q,direction", [("impact m.py:5", "impact"), ("depends-on m.py:5", "depends-on"), ("m.py::A impact", "impact"),
                                         ("depends-on m.py::A", "depends-on")])
def test_a_parsed_question_that_resolves_is_the_structured_answer_byte_for_byte(q, direction):
    g, _ = world()
    built = []
    doc = front_over(g, built)(q, reader=FrontReader(g))
    assert answer.dumps(doc) == answer.dumps(structured(g, direction))                      # no provenance: no model read it
    assert answer.facts_bytes(doc) == answer.facts_bytes(structured(g, direction))
    assert len(built) == 1


def test_the_resolvers_reads_stay_out_of_the_answers_recipe():
    g, _ = world()
    resolver_reader = FrontReader(g)
    doc = front_over(g)("impact m.py:5", reader=resolver_reader)
    names = {q["name"] for q in doc["recipe"]["queries"]}
    assert {"file_by_path", "symbols_of_file", "repo_exists"} <= {c[0] for c in resolver_reader.calls}
    assert not names & {"file_by_path", "symbols_of_file", "repo_exists", "symbol_by_name"}


def test_a_failed_named_file_beside_a_resolved_one_is_kept_in_lookups_and_is_not_a_fact():
    g, _ = world()
    doc = front_over(g)("impact m.py:5 gone.py:2", reader=FrontReader(g))
    assert doc["interpretation"]["status"] == "resolved"
    assert doc["interpretation"]["lookups"] == [{"path": "gone.py", "repoId": g.repo, "state": "not-found"}]
    assert answer.facts_bytes(doc) == answer.facts_bytes(structured(g))


# ------------------------------------------------------------------------------------------------------------------ asking documents

def _asking_shape(doc):
    assert doc["items"] == [] and doc["groups"] == [] and doc["staleness"] == {"units": []} and doc["depthBound"] == {"unexplored": 0}
    assert doc["recipe"] == {"algorithm": {"id": "bfs-v1", "version": bfs.algorithm_version()}, "queries": [],
                             "guards": {"visitedSeedsAnchor": True, "levelGuard": config.LEVEL_GUARD}}
    assert doc["stamp"]["snapshotId"] == SNAP and doc["answerSchemaVersion"] == "1.2" and "provenance" not in doc["interpretation"]
    assert answer.is_canonical(answer.dumps(doc))


def test_a_bare_anchor_states_no_direction_and_is_answered_with_both_directions_as_ranked_options():
    g, ids_ = world()
    built = []
    doc = front_over(g, built)("m.py:5", reader=FrontReader(g))
    _asking_shape(doc)
    interp = doc["interpretation"]
    assert (interp["status"], interp["ambiguityKind"]) == ("ambiguous", "two-use-cases")
    anchor = {"type": "symbol", "symbolId": ids_["A"], "qname": "m.py::A", "repoId": g.repo, "path": "m.py"}
    assert interp["options"] == [{"useCase": "impact", "direction": "impact", "anchors": [anchor], "rank": 1},
                                 {"useCase": "impact", "direction": "depends-on", "anchors": [anchor], "rank": 2}]
    assert built == []                                                                      # an asking answer builds nothing


def test_every_option_is_the_request_a_structured_ask_would_send():
    g, _ = world()
    doc = front_over(g)("m.py:5", reader=FrontReader(g))
    for opt in doc["interpretation"]["options"]:
        want = structured(g, opt["direction"])["interpretation"]["request"]
        assert {k: v for k, v in opt.items() if k != "rank"} == want


def test_a_named_file_that_is_not_found_is_the_unresolved_document():
    g, _ = world()
    doc = front_over(g)("impact nope/x.py:3", reader=FrontReader(g))
    _asking_shape(doc)
    assert doc["interpretation"] == {"status": "unresolved", "lookups": [{"path": "nope/x.py", "repoId": g.repo, "state": "not-found"}]}


def test_an_unknown_name_is_ambiguous_never_resolved_or_empty():
    g, _ = world()
    doc = front_over(g)("impact m.py::Nope", reader=FrontReader(g))
    _asking_shape(doc)
    assert doc["interpretation"]["status"] == "ambiguous" and doc["interpretation"]["ambiguityKind"] == "unknown-name"
    assert [o["anchors"][0]["qname"] for o in doc["interpretation"]["options"]] == ["m.py::A"]       # the file's symbols are the did-you-mean list


def test_asking_document_refuses_a_resolved_outcome():
    g, _ = world()
    with pytest.raises(ValueError):
        asking_document(FrontReader(g), Outcome("resolved"))


# ------------------------------------------------------------------------------------------------------------------ not built, and no read before it

@pytest.mark.parametrize("q,detail", [
    ("diff --git a/m.py b/m.py\n@@ -1 +1 @@\n-a\n+b", {"request": {"useCase": "code-review"}, "surface": "diff"}),
    ("Traceback (most recent call last):\n  File \"m.py\", line 5, in A", {"request": {"useCase": "root-cause"}, "surface": "stack-trace"}),
    ("who calls A?", {"surface": "plain-language-question"}),
])
def test_a_question_this_build_cannot_answer_is_not_built_before_any_read(q, detail):
    g, _ = world()
    reader = FrontReader(g)
    f = front_over(g)
    with pytest.raises(NotBuiltAsk) as e:
        f.precheck(q)
    assert e.value.detail == detail
    with pytest.raises(NotBuiltAsk):
        f(q, reader=reader)
    assert reader.calls == []


def test_an_unbuilt_use_case_word_is_not_built_with_the_request_echoed():
    g, _ = world()
    reader = FrontReader(g)
    with pytest.raises(NotBuiltAsk) as e:
        front_over(g)("security m.py:5", reader=reader)
    assert e.value.detail["request"] == {"useCase": "security", "direction": None, "anchors": [[{"text": "m.py:5", "kindHint": "line"}]]}
    assert reader.calls == []


def test_a_scope_is_not_built():
    g, _ = world()
    with pytest.raises(NotBuiltAsk) as e:
        front_over(g)("scope: svc impact m.py:5", reader=FrontReader(g))
    assert e.value.detail["request"]["scope"] == "svc"


def test_the_question_is_normalized_before_parsing_so_spacing_variants_give_one_answer():
    g, _ = world()
    a = front_over(g)("impact   m.py:5", reader=FrontReader(g))
    b = front_over(g)("\timpact m.py:5 \r\n", reader=FrontReader(g))
    assert answer.dumps(a) == answer.dumps(b)


def test_the_question_is_nfc_normalized_before_the_spans_are_cut():
    """``normalize_question`` runs first: a path typed with a decomposed accent is the stored (NFC) path, and a run of spaces inside a route is one space."""
    from front_helpers import Cp, FakeStore, Sym
    store = FakeStore([Sym("app/café.py::f", "app/café.py", 1, 9)], cps=[Cp("POST /threads/{}/messages", "http")])
    f = Front(store.repos, build=lambda anchor, direction: pytest.fail("an ambiguous answer builds nothing"))
    composed = f("app/caf\u00e9.py:3", reader=store)
    decomposed = f("app/cafe\u0301.py:3", reader=store)
    assert decomposed["interpretation"]["ambiguityKind"] == "two-use-cases" and answer.dumps(decomposed) == answer.dumps(composed)
    one = f("POST /threads/{thread_id}/messages", reader=store)
    spaced = f("POST \t  /threads/{thread_id}/messages", reader=store)
    assert spaced["interpretation"]["ambiguityKind"] == "two-use-cases" and answer.dumps(spaced) == answer.dumps(one)


def test_a_partial_path_resolves_through_the_suffix_read_to_the_structured_answer_and_its_reads_stay_out_of_the_recipe():
    g, _ = world()
    resolver_reader = FrontReader(g)
    doc = front_over(g)("impact c.py:12", reader=resolver_reader)                  # the stored path is svc/c.py
    anchor = A.resolve_symbol(FrontReader(g), g.repo, "svc/c.py::C").anchors[0]
    assert answer.facts_bytes(doc) == answer.facts_bytes(build_answer(FrontReader(g), anchor, question="impact"))
    assert {"file_by_suffix", "skipped_by_suffix"} <= {c[0] for c in resolver_reader.calls}
    assert not {"file_by_suffix", "skipped_by_suffix", "file_by_path"} & {q["name"] for q in doc["recipe"]["queries"]}
