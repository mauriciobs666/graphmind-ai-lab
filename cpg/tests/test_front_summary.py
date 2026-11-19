"""P1-26 dispatch 2 (5), (9), (10): the optional aggregate-only summary (``ccg.front.summary``).

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-26 behaviours (5) (aggregate-only, dropped on any added entity; a notice's ``langs`` values are allowed echoes),
(9) (``validate`` first, only a valid document is summarized), (10) (a summary is always followed by ``flow_notice_lines(doc)`` verbatim, assembled by code); ``-answer-spec2.md`` §5.1.
The model is a stub behind a broker double.

    cd cpg && .venv/bin/python -m pytest tests/test_front_summary.py
"""
from __future__ import annotations

import copy
import json

import pytest
from front_model_helpers import FakeBroker
from test_front_documents import structured, world
from test_render import b_notice, notice_empty

from ccg import answer
from ccg.front import ModelError
from ccg.front import interpret as I
from ccg.front.summary import FUNCTION_WORDS, digest, rejected_tokens, summarize
from ccg.render.common import flow_notice_lines


def run_of(*replies, **kw):
    return I.ModelRun(FakeBroker(list(replies), **kw), "shop", ("r",))


def plain():
    g, _ = world()
    return structured(g)


def numbers(doc):
    return {v for v in digest(doc).values() if type(v) is int}


def test_the_model_sees_counts_only_and_a_kept_summary_is_printed_first():
    doc = b_notice()
    text = f"It lists {digest(doc)['items']} items."
    broker = FakeBroker([text])
    s = summarize(doc, I.ModelRun(broker, "shop", ("r",)))
    user = broker.completions[0][2]["messages"][1]["content"]
    assert json.loads(user) == digest(doc) and set(digest(doc)) <= {"status", "direction", "items", "confirmed", "possible", "groups", "groupedMembers", "options"}
    for item in doc["items"]:
        assert item["subject"].get("qname", "\0") not in user and item["subject"].get("path", "\0") not in user
    req = broker.completions[0][1]
    assert (req["component"], req["purpose"]) == ("answer.summarize", "model-call")
    assert s.text == text and not s.dropped and s.lines[0] == text
    assert s.lines[-len(flow_notice_lines(doc)):] == tuple(flow_notice_lines(doc))


def test_a_summary_of_the_digests_own_numbers_and_plain_words_is_kept():
    doc = plain()
    d = digest(doc)
    text = f"{d['items']} items, {d['confirmed']} confirmed and {d['possible']} possible, in the {d['direction']} direction."
    s = summarize(doc, run_of(text))
    assert (s.text, s.dropped, s.lines) == (text, False, (text,))
    assert s.render() == text


def test_the_flow_notice_follows_verbatim_even_when_the_stub_model_omits_it():
    doc = b_notice()
    notice = flow_notice_lines(doc)
    assert notice and all("flow" in n.lower() or n for n in notice)
    d = digest(doc)
    text = f"{d['items']} items."
    s = summarize(doc, run_of(text))
    assert s.lines == (text, *notice) and s.render().endswith("\n".join(notice))


def test_a_model_that_restates_the_notice_in_its_own_words_does_not_replace_the_codes_lines():
    doc = b_notice()
    s = summarize(doc, run_of("The flow analysis was partial, nothing to worry about."))
    assert s.lines[-len(flow_notice_lines(doc)):] == tuple(flow_notice_lines(doc)) and s.lines[-1] == flow_notice_lines(doc)[-1]


def test_a_document_without_notices_has_no_notice_lines():
    s = summarize(plain(), run_of("Nothing is listed."))
    assert s.lines == ("Nothing is listed.",)


ADDED = ["It affects src/billing/pay.py.", "It affects `Runner`.", "It affects Runner.run.", "It affects m.py::Zed and more.", "It affects snake_case_name.",
         "It affects fooBar.", "It affects 999 items.", "It affects v2.", "It affects Billing and more.", "See `qqzz`.", "It affects /threads/{id}/messages."]


@pytest.mark.parametrize("text", ADDED)
def test_a_summary_with_an_added_entity_is_dropped_and_the_notice_still_shows(text):
    doc = b_notice()
    s = summarize(doc, run_of(text))
    assert s.dropped and s.text is None and s.lines == tuple(flow_notice_lines(doc)) and text not in s.render()


@pytest.mark.parametrize("text", ADDED)
def test_an_added_entity_is_dropped_on_a_document_with_no_notice_too(text):
    s = summarize(plain(), run_of(text))
    assert s.dropped and s.lines == ()


PROBED = ["Checkout is affected by 2 items.", "Affects 2 items. Billing breaks.", "The login service is affected.", "Two callers, 50% of them.", "Affects three callers.",
          "It affects \x1b[31m2 items.", "It affects 2 items.\nSee the next line.", "It affects 2\r items.", "It affects 2\x00 items.", "Items\u200b affected.",
          "It affects the items.confirmed.", "It affects a::b.", "It affects items_confirmed.", "It affects the 2/3 items.", "It affects \u0663 items."]


@pytest.mark.parametrize("text", PROBED)
def test_each_string_the_review_probed_is_rejected_by_the_allow_list_and_dropped(text):
    doc = b_notice()
    assert rejected_tokens(text, doc), text
    s = summarize(doc, run_of(text))
    assert s.dropped and s.text is None and s.lines == tuple(flow_notice_lines(doc))


@pytest.mark.parametrize("text, rejected", [
    ("Checkout is affected.", ["Checkout"]), ("The login service is affected.", ["login", "service"]), ("Two items.", ["Two"]),
    ("It affects 50% of items.", ["50", "%"]), ("It affects items.\x1b", ["\\x1b"]), ("It affects items.\nIt affects items.", ["\\n"]),
    ("It affects items.confirmed.", ["s.c"]), ("Items affect a::b.", ["::", "b"]), ("Items are safe.", ["safe"]),
])
def test_the_allow_list_names_what_it_rejects_one_rule_at_a_time(text, rejected):
    out = rejected_tokens(text, plain())
    assert all(any(r in o for o in out) for r in rejected), (text, out)


def test_a_sentence_of_the_digests_words_the_function_words_and_the_digests_numbers_passes():
    doc = plain()
    d = digest(doc)
    for text in (f"{d['items']} items: {d['confirmed']} confirmed and {d['possible']} possible.", "The answer is resolved; there is nothing more.", "It is an impact answer (resolved)."):
        assert rejected_tokens(text, doc) == [], text
    assert "two" not in FUNCTION_WORDS and "one" not in FUNCTION_WORDS                          # a number is written in digits, so it is checked against the digest


def test_the_digests_words_are_accepted_in_singular_and_plural_and_a_notices_langs_are_the_minted_ones():
    doc = notice_empty()
    n = digest(doc)["items"]
    assert rejected_tokens(f"{n} items. Nothing is listed; {n} group, {n} groups, {n} item.", doc) == []
    assert set(doc["notices"][0]["langs"]) <= set(answer.EXT_LANG.values())


def test_what_the_guard_does_not_promise_a_swapped_or_unsupported_count_in_allowed_words_is_kept():
    doc = plain()
    d = digest(doc)
    swapped = f"{d['possible']} items are confirmed and {d['confirmed']} are possible."
    s = summarize(doc, run_of(swapped))
    assert not s.dropped and s.text == swapped                                                  # documented limit: the summary is unscored and not claimed accurate


def test_an_empty_reply_is_dropped_not_printed():
    s = summarize(b_notice(), run_of("   "))
    assert s.dropped and s.text is None


def test_a_notices_langs_values_are_allowed_echoes():
    doc = notice_empty()
    langs = doc["notices"][0]["langs"]
    assert langs == ["javascript", "python"]
    text = f"Nothing is listed for `{langs[0]}` and `{langs[1]}` in the flow analysis."
    s = summarize(doc, run_of(text))
    assert not s.dropped and s.text == text
    assert rejected_tokens("Nothing in `rust`.", doc) == ["rust"]                   # a language the document does not hold is an added entity


def test_the_added_entity_check_reads_numbers_from_the_digest_only():
    doc = plain()
    n = digest(doc)["items"]
    assert rejected_tokens(f"{n} items.", doc) == []
    assert rejected_tokens(f"{n + 100} items.", doc) == [str(n + 100)]


def test_an_invalid_document_is_never_summarized_and_nothing_is_sent():
    doc = plain()
    for broken in (lambda d: d.pop("stamp"), lambda d: d["interpretation"].update(status="nope"), lambda d: d.update(items="x")):
        bad = copy.deepcopy(doc)
        broken(bad)
        broker = FakeBroker(["fine"])
        with pytest.raises(answer.AnswerError):
            summarize(bad, I.ModelRun(broker, "shop", ("r",)))
        assert broker.opens == [] and broker.requests == []
    with pytest.raises(answer.AnswerError):
        summarize({"not": "an answer"}, run_of("x"))


def test_an_unreachable_model_fails_the_summary_rather_than_printing_none():
    with pytest.raises(ModelError):
        summarize(plain(), run_of("x", unavailable=True))


def test_an_ambiguous_document_is_summarized_by_its_option_count():
    g, _ = world()
    from front_helpers import FrontReader
    from ccg.front import Front
    f = Front((g.repo,), lambda a, d: structured(g))
    doc = f("m.py:5", reader=FrontReader(g))
    assert digest(doc) == {"status": "ambiguous", "options": 2}
    assert summarize(doc, run_of("2 options.")).text == "2 options."
