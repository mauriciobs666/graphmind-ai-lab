"""P1-35 dispatch 1: the AC-24 smoke harness (``cpg/eval/tools/ac24_smoke.py``), driven by stubs and synthetic items only.

Contract: ``docs/plans/connected-code-graph-p1-gates.md`` P1-35; ``-p1-ml.md`` §4.3 (protocol, record, custody §6.4); eval protocol §10.4 (scorer), §10.5, §12.3.
No live model is run, no held-out item is read, the real custodian root is never touched: the held-out mode runs on synthetic items in a temporary root. The dev file
``cpg/eval/dev/ac24.jsonl`` is read for the dev entry run (its items are dev items).

    cd cpg && .venv/bin/python -m pytest tests/test_ac24_smoke.py
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest
from front_helpers import FrontReader
from front_model_helpers import FINGERPRINT, FakeBroker, draft_reply
from test_front_documents import world

from ccg.engine.answer_build import build_answer
from ccg.front import Front
from ccg.front import interpret as I
from ccg.front.resolve import RESOLVER_VERSION
from clients.common import Client, ModelUnavailable, NotBuilt

CPG = Path(__file__).resolve().parent.parent
TOOLS = Path(os.environ.get("CCG_TOOLS_DIR", CPG / "eval" / "tools"))
DEV = CPG / "eval" / "dev" / "ac24.jsonl"


def _load():
    spec = importlib.util.spec_from_file_location("ac24_smoke", TOOLS / "ac24_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ac24_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load()

PROV = {"modelFingerprint": FINGERPRINT, "promptHash": "a" * 64, "grammarHash": "b" * 64, "samplingHash": "c" * 32, "resolverVersion": "1.0", "endpoint": "local",
        "temperatureMilli": 0, "quantization": "Q4_K_M", "cacheKey": "d" * 32, "cacheHit": False}
INTERP = {k: PROV[k] for k in S.INTERPRETER_KEYS}
SYM = {"type": "symbol", "repoId": "r", "qname": "m.py::A", "path": "m.py"}
GOLD = {"useCase": "impact", "direction": "impact", "anchors": [{"type": "symbol", "repoId": "r", "qname": "m.py::A", "path": "m.py"}], "scope": None}


def req(direction="impact", qname="m.py::A", rank=None):
    r = {"useCase": "impact", "direction": direction, "anchors": [{**SYM, "qname": qname, "symbolId": "1" * 32}]}
    if rank is not None:
        r["rank"] = rank
    return r


def doc(status="resolved", *, request=None, options=None, kind=None, prov=PROV, hit=False, bypass=False):
    i: dict = {"status": status}
    if request is not None:
        i["request"] = request
    if options is not None:
        i["options"] = options
    if kind is not None:
        i["ambiguityKind"] = kind
    if status == "unresolved":
        i["lookups"] = [{"repoId": "r", "path": "x.py", "result": "not-found"}]
    if prov is not None:
        i["provenance"] = {**prov, "cacheHit": hit, **({"cacheBypass": True} if bypass else {})}
    return {"interpretation": i}


def unamb(qid="U-1", lang="en", gold=GOLD, text="q"):
    return {"qid": qid, "text": text, "lang": lang, "outcome": "unambiguous", "gold": gold}


def amb(gold_options, qid="A-1", lang="en", text="q"):
    return {"qid": qid, "text": text, "lang": lang, "outcome": "ambiguity", "goldOptions": gold_options, "ambiguityKind": "several-symbols"}


G1 = {"useCase": "impact", "direction": "impact", "anchors": [{"type": "symbol", "repoId": "r", "qname": "a.py::X", "path": "a.py"}], "scope": None}
G2 = {**G1, "direction": "depends-on"}


# ------------------------------------------------------------------------------------------------------------------ statistics

@pytest.mark.parametrize("k,n,want", [(23, 24, [0.798, 0.993]), (24, 24, [0.862, 1.0]), (11, 12, [0.646, 0.985])])
def test_wilson_matches_the_figures_of_ep_10_5(k, n, want):
    assert S.wilson(k, n) == want


def test_thresholds_are_the_ceil_rule_in_exact_arithmetic():
    assert S.threshold(S.POOLED_RATE, 24) == 23 and S.threshold(S.LANG_RATE, 12) == 11
    assert S.threshold(S.POOLED_RATE, 25) == 23 and S.threshold(S.POOLED_RATE, 26) == 24


# ------------------------------------------------------------------------------------------------------------------ scorer (EP §10.4)

def test_unambiguous_understood_only_when_resolved_and_equal_to_gold():
    v = S.score_item(unamb(), doc(request=req()))
    assert v["understood"] and v["reason"] == "match" and not v["overAsk"]


@pytest.mark.parametrize("d,reason", [(doc(request=req("depends-on")), "wrong-interpretation"), (doc(request=req(qname="m.py::B")), "wrong-interpretation")])
def test_unambiguous_wrong_direction_or_anchor_is_not_understood(d, reason):
    v = S.score_item(unamb(), d)
    assert not v["understood"] and v["reason"] == reason and not v["overAsk"]


def test_unambiguous_asking_with_gold_among_the_options_is_over_asking():
    v = S.score_item(unamb(), doc("ambiguous", options=[req(rank=1), req("depends-on", rank=2)], kind="several-symbols"))
    assert not v["understood"] and v["overAsk"] and v["reason"] == "over-asking"


def test_anchor_identity_ignores_engine_ids_rank_and_anchor_order():
    two = {**GOLD, "anchors": [GOLD["anchors"][0], {"type": "contract", "kind": "http", "keyNorm": "GET /x"}]}
    d = doc(request={**req(), "anchors": [{"type": "contract", "cpId": "2" * 32, "kind": "http", "keyNorm": "GET /x"}, {**SYM, "symbolId": "9" * 32}]})
    assert S.score_item(unamb(gold=two), d)["understood"]


def test_a_scope_difference_is_not_understood():
    d = doc(request={**req(), "scope": {"repoIds": ["r"]}})
    assert not S.score_item(unamb(), d)["understood"]


@pytest.mark.parametrize("status,kind,n_opts,ok", [("ambiguous", "no-anchor", 0, True), ("unresolved", None, 0, True), ("ambiguous", "unknown-name", 5, True),
                                                    ("ambiguous", "unknown-name", 6, False)])
def test_ambiguous_with_empty_gold_needs_asking_within_the_cap(status, kind, n_opts, ok):
    d = doc(status, options=[req(qname=f"m.py::C{i}", rank=i + 1) for i in range(n_opts)] if status == "ambiguous" else None, kind=kind, prov=None)
    v = S.score_item(amb([]), d)
    assert v["understood"] is ok and v["capExceeded"] is (n_opts > 5)


def test_ambiguous_resolved_silently_is_not_understood():
    v = S.score_item(amb([G1, G2]), doc(request=req()))
    assert not v["understood"] and v["reason"] == "silent-single-interpretation"


def test_ambiguous_needs_every_gold_option_and_status_ambiguous():
    opts = [{**G1, "rank": 1}, {**G2, "rank": 2}]
    assert S.score_item(amb([G1, G2]), doc("ambiguous", options=opts, kind="two-use-cases"))["understood"]
    assert not S.score_item(amb([G1, G2]), doc("ambiguous", options=opts[:1], kind="two-use-cases"))["understood"]
    assert S.score_item(amb([G1, G2]), doc("unresolved"))["reason"] == "unresolved-names-no-reading"


def test_the_emitted_ambiguity_kind_is_reported_and_not_scored():
    v = S.score_item(amb([G1, G2]), doc("ambiguous", options=[{**G1, "rank": 1}, {**G2, "rank": 2}], kind="no-anchor", prov=None))
    assert v["understood"] and v["ambiguityKindEmitted"] == "no-anchor"


def test_the_cap_is_max_of_the_floor_and_each_items_own_gold_size():
    gold7 = [{**G1, "anchors": [{**G1["anchors"][0], "qname": f"a.py::X{i}"}]} for i in range(7)]
    ok7 = [{**g, "rank": i + 1} for i, g in enumerate(gold7)]
    assert S.score_item(amb(gold7), doc("ambiguous", options=ok7, kind="several-symbols"))["understood"]            # 7 <= max(5, 7)
    extra = ok7 + [{**G2, "rank": 8}]
    v = S.score_item(amb(gold7), doc("ambiguous", options=extra, kind="several-symbols"))
    assert not v["understood"] and v["capExceeded"] and v["extras"] == 1                                            # 8 > 7
    pad = [{**G1, "rank": 1}, {**G2, "rank": 2}] + [{**G1, "anchors": [{**G1["anchors"][0], "qname": f"a.py::P{i}"}], "rank": 3 + i} for i in range(4)]
    v2 = S.score_item(amb([G1, G2]), doc("ambiguous", options=pad, kind="several-symbols"))
    assert not v2["understood"] and v2["extras"] == 4 and v2["nOptions"] == 6                                       # 6 > max(5, 2)


def test_a_not_built_outcome_is_not_understood_on_a_built_use_case_item():
    for item in (unamb(), amb([G1, G2])):
        v = S.score_item(item, None)
        assert not v["understood"] and v["notBuilt"] and v["reason"] == "not-built"


def test_the_recorded_unusable_observable_scores_not_understood_on_every_kind_of_item():
    asked = doc("ambiguous", options=[], kind="no-anchor")
    v = S.score_item(unamb(), asked, unusable=True)
    assert v["unusable"] and not v["understood"] and v["overAsk"] and v["reason"] == "unusable-reply"
    v2 = S.score_item(amb([]), asked, unusable=True)
    assert v2["unusable"] and not v2["understood"] and not v2["overAsk"]


def test_a_valid_empty_ask_is_not_unusable_and_is_understood_on_a_gold_empty_item():
    """The collision the reviews found: the same document as an unusable reply, scored by EP 10.4 when the draft was usable."""
    asked = doc("ambiguous", options=[], kind="no-anchor")
    v = S.score_item(amb([]), asked, unusable=False)
    assert v["understood"] and not v["unusable"] and v["reason"] == "asked"
    assert S.score_item(unamb(), asked)["overAsk"] and not S.score_item(unamb(), asked)["unusable"]


def test_e1_a_symbol_in_an_anonymous_scope_scores_as_its_eligible_element():
    gold = {**GOLD, "anchors": [{"type": "symbol", "repoId": "r", "qname": "salesperson/src/ProductList.tsx::ProductList", "path": "salesperson/src/ProductList.tsx"}]}
    inner = {"useCase": "impact", "direction": "impact", "anchors": [{"type": "symbol", "repoId": "r", "symbolId": "1" * 32, "path": "salesperson/src/ProductList.tsx",
                                                                      "qname": "salesperson/src/ProductList.tsx::ProductList.<anon>#1"}]}
    assert S.score_item(unamb(gold=gold), doc(request=inner))["understood"]
    dup = {**inner, "anchors": [{**inner["anchors"][0], "qname": "salesperson/src/ProductList.tsx::ProductList#2"}]}
    assert S.score_item(unamb(gold=gold), doc(request=dup))["understood"]                  # the #k duplicate shares its E: the documented limit
    other = {**inner, "anchors": [{**inner["anchors"][0], "qname": "salesperson/src/ProductList.tsx::Other.<anon>#1"}]}
    assert not S.score_item(unamb(gold=gold), doc(request=other))["understood"]


def test_the_scorer_never_reads_the_question_text():
    item = unamb()
    item["text"] = None
    assert S.score_item(item, doc(request=req()))["understood"]


# ------------------------------------------------------------------------------------------------------------------ items and re-count

def test_load_items_rejects_a_malformed_item_and_sorts_by_qid(tmp_path):
    p = tmp_path / "i.jsonl"
    p.write_text(json.dumps(unamb("B")) + "\n" + json.dumps(unamb("A")) + "\n")
    assert [i["qid"] for i in S.load_items(p)] == ["A", "B"]
    p.write_text(json.dumps({**unamb(), "outcome": "other"}) + "\n")
    with pytest.raises(S.Refusal):
        S.load_items(p)
    p.write_text(json.dumps(unamb("A")) + "\n" + json.dumps(unamb("A")) + "\n")
    with pytest.raises(S.Refusal, match="duplicate"):
        S.load_items(p)


def test_the_recount_reads_the_split_output_not_a_constant():
    items = S.load_items(DEV)
    assert S.recount(items) == {"unambiguous": 12, "ambiguous": 2, "en": 6, "pt": 6}


# ------------------------------------------------------------------------------------------------------------------ the run (stub client)

class Stub:
    """A client double: ``handler(question, bypass, n, log)`` returns the answer (or raises) and may append to the request log, as the broker seam would."""

    def __init__(self, handler, log=None):
        self.handler, self.log, self.calls = handler, log or S.RequestLog(), []

    def ask(self, request=None, *, question=None, cache_bypass=False):
        self.calls.append((question, cache_bypass))
        return self.handler(question, cache_bypass, len(self.calls), self.log)


BODY = {"model": "m", "messages": [{"role": "user", "content": "q"}], "temperature": 0}


def post(log, status=200, body=BODY, reply=None):
    log.entries.append({"method": "POST", "path": S.CHAT_PATH, "body": body, "status": status, **({"reply": reply} if reply is not None else {})})


def get(log, *, unusable=False, bypass=False, prov=PROV):
    """The model-ask's GET read, and the draft observation the recording interpreter appends per ``interpret`` call."""
    log.entries.append({"method": "GET", "path": "/api/v0/models", "body": None, "status": 200})
    log.events.append({"draftNone": unusable, "provenance": {**prov, "cacheHit": False, **({"cacheBypass": True} if bypass else {})}})


def drive(items, handler, *, repeats=5, cache=0, **kw):
    stub = Stub(handler)
    run = S.run_items(items, stub, repeats=repeats, log=stub.log, cache_len=lambda: cache, **kw)
    return run, stub


def good(question, bypass, n, log, *, d=None):
    get(log)
    post(log)
    return d or doc(request=req(), bypass=bypass)


def test_repeat_1_is_uncached_and_repeats_2_to_5_carry_the_bypass_and_every_repeat_counts_one_post():
    run, stub = drive([unamb()], good)
    assert [b for _, b in stub.calls] == [False, True, True, True, True]
    (rec,) = run["items"]
    assert [x["completions"] for x in rec["repeats"]] == [1] * 5            # five POSTs; the five GET reads are not counted
    assert sum(1 for e in stub.log.entries if e["method"] == "GET") == 5 and len(S.completions(stub.log.entries)) == 5


def test_items_run_in_qid_order_item_by_item_whatever_the_input_order():
    run, stub = drive([unamb("B", text="tb"), unamb("A", text="ta")], good, repeats=2)
    assert [q for q, _ in stub.calls] == ["ta", "ta", "tb", "tb"] and [r["qid"] for r in run["items"]] == ["A", "B"]


def test_a_get_read_is_never_a_completion():
    def only_get(q, b, n, log):
        get(log)
        return doc(request=req(), bypass=b)
    with pytest.raises(S.HarnessError, match="no completion request"):
        drive([unamb()], only_get)


def test_a_repeat_that_scores_repeat_1_not_repeat_2():
    def flip(q, b, n, log):
        get(log), post(log)
        return doc(request=req(), bypass=b) if n == 1 else doc(request=req("depends-on"), bypass=b)
    run, _ = drive([unamb()], flip)
    (rec,) = run["items"]
    s = S.summarize(run, [unamb()], mode="dev")
    assert rec["repeats"][0]["understood"] and not rec["repeats"][1]["understood"]
    assert s["unambiguousUnderstood"]["k"] == 1 and s["flipRate"]["k"] == 1 and s["perDrawUnderstood"] == {"k": 1, "n": 5, "wilson95": S.wilson(1, 5)}

    def flip2(q, b, n, log):
        get(log), post(log)
        return doc(request=req("depends-on"), bypass=b) if n == 1 else doc(request=req(), bypass=b)
    run2, _ = drive([unamb()], flip2)
    assert S.summarize(run2, [unamb()], mode="dev")["unambiguousUnderstood"]["k"] == 0


def test_cache_hit_on_repeat_1_scores_not_understood_and_stops_the_run_c5():
    items = [unamb("U-1"), unamb("U-2"), unamb("U-3")]

    def h(q, b, n, log):
        if n == 6:
            return doc(request=req(), hit=True)                   # U-2's repeat 1 is served from the cache
        get(log), post(log)
        return doc(request=req(), bypass=b)
    run, stub = drive(items, h, cache=0)
    assert run["stop"]["code"] == "C5" and run["stop"]["qid"] == "U-2" and run["ran"] == 2 and run["of"] == 3
    assert run["items"][1]["repeats"][0]["understood"] is False and len(run["items"][1]["repeats"]) == 1
    s = S.summarize(run, items, mode="heldout")
    assert s["passed"] is False and s["verdict"] == "smoke gate stopped (C5)"
    assert len(stub.calls) == 6                                    # U-1 five repeats, U-2 repeat 1; U-3 never asked


def test_a_cache_hit_on_a_later_repeat_voids_only_that_repeat():
    def h(q, b, n, log):
        if n == 3:
            return doc(request=req(), hit=True)
        get(log), post(log)
        return doc(request=req(), bypass=b)
    run, _ = drive([unamb()], h)
    reps = run["items"][0]["repeats"]
    assert [x["voided"] for x in reps] == [False, False, True, False, False] and run["stop"] is None
    s = S.summarize(run, [unamb()], mode="dev")
    assert s["voidedRepeats"] == 1 and s["perDrawUnderstood"]["n"] == 4


def test_a_cache_on_the_gate_run_start_is_a_harness_error():
    with pytest.raises(S.HarnessError, match="not empty"):
        drive([unamb()], good, cache=1)


def test_the_bypass_must_reach_the_engine():
    def h(q, b, n, log):
        get(log), post(log)
        return doc(request=req(), bypass=False)                   # ignores cacheBypass
    with pytest.raises(S.HarnessError, match="cacheBypass"):
        drive([unamb()], h)


def test_repeat_1_must_not_carry_the_bypass():
    def h(q, b, n, log):
        get(log), post(log)
        return doc(request=req(), bypass=True)
    with pytest.raises(S.HarnessError, match="repeat 1 carries"):
        drive([unamb()], h, repeats=1)


def test_a_cache_that_reports_false_but_sent_no_request_fails_the_count():
    def liar(q, b, n, log):
        if n == 1:
            get(log), post(log)
        return doc(request=req(), bypass=b)
    with pytest.raises(S.HarnessError, match="no completion"):
        drive([unamb()], liar)


def test_a_logged_retry_passes_and_a_sixth_request_with_a_different_body_fails():
    def retry(q, b, n, log):
        get(log)
        if n == 3:
            post(log, status=503)
        post(log)
        return doc(request=req(), bypass=b)
    run, _ = drive([unamb()], retry)
    assert [x["completions"] for x in run["items"][0]["repeats"]] == [1, 1, 2, 1, 1]

    def transport(q, b, n, log):
        get(log)
        if n == 3:
            post(log, status=None)
        post(log)
        return doc(request=req(), bypass=b)
    assert drive([unamb()], transport)[0]["items"][0]["repeats"][2]["completions"] == 2

    def extra(q, b, n, log):
        get(log), post(log)
        if n == 5:
            post(log, body={**BODY, "temperature": 1})
        return doc(request=req(), bypass=b)
    with pytest.raises(S.HarnessError, match="not a retry"):
        drive([unamb()], extra)

    def other_body_after_failure(q, b, n, log):                    # a failed first exchange, then a request with another body: not the same request retried
        get(log), post(log, status=503), post(log, body={**BODY, "temperature": 1})
        return doc(request=req(), bypass=b)
    with pytest.raises(S.HarnessError, match="not a retry"):
        drive([unamb()], other_body_after_failure)

    def dup_ok(q, b, n, log):                                      # the first exchange succeeded, so a repeat of the body is not a retry
        get(log), post(log), post(log)
        return doc(request=req(), bypass=b)
    with pytest.raises(S.HarnessError, match="not a retry"):
        drive([unamb()], dup_ok)


@pytest.mark.parametrize("key,val", [("modelFingerprint", "other|x|y|Q8|4096"), ("promptHash", "e" * 64), ("grammarHash", "e" * 64), ("samplingHash", "e" * 32),
                                     ("resolverVersion", "9.9"), ("quantization", "Q8_0"), ("endpoint", "hosted")])
def test_a_provenance_that_differs_from_the_preregistered_interpreter_is_a_harness_error_not_a_score(key, val):
    with pytest.raises(S.HarnessError, match=key):
        drive([unamb()], good, expected_interpreter={**INTERP, key: val})


def test_without_a_record_the_first_provenance_is_the_baseline_and_a_change_is_an_error():
    def drift(q, b, n, log):
        get(log), post(log)
        return doc(request=req(), prov={**PROV, "modelFingerprint": "swapped|x|y|Q4|1"} if n == 3 else PROV, bypass=b)
    with pytest.raises(S.HarnessError, match="modelFingerprint"):
        drive([unamb()], drift)
    run, _ = drive([unamb()], good)
    assert run["interpreterObserved"] == INTERP


def test_the_effective_sampler_and_the_request_body_sampler_are_asserted():
    run, _ = drive([unamb()], good, sampler_observer=lambda: {"top_k": 0, "repeat_penalty": 1.0, "min_p": 0.0, "preset": "none"})
    assert run["samplerEffective"]["top_k"] == 0
    with pytest.raises(S.HarnessError, match="effective sampler"):
        drive([unamb()], good, sampler_observer=lambda: {"top_k": 40}, expected_sampler_effective={"top_k": 0})
    with pytest.raises(S.HarnessError, match="effective sampler"):
        drive([unamb()], good, expected_sampler_effective={"top_k": 0})                 # not observed at all
    with pytest.raises(S.HarnessError, match="sampler parameters"):
        drive([unamb()], good, expected_sampler_request={"temperature": 0, "top_p": 1})  # BODY has no top_p
    drive([unamb()], good, expected_sampler_request={"temperature": 0})


def test_a_completion_without_provenance_is_a_harness_error():
    def h(q, b, n, log):
        post(log)
        return doc(request=req(), prov=None)
    with pytest.raises(S.HarnessError, match="no provenance"):
        drive([unamb()], h)


def test_unusable_replies_are_tallied_and_the_dev_recommendation_follows():
    def h(q, b, n, log):
        get(log, unusable=True), post(log)
        return doc("ambiguous", options=[], kind="no-anchor", bypass=b)
    items = [unamb(f"U-{i}", "en" if i % 2 else "pt") for i in range(12)] + [amb([G1, G2], "A-1"), amb([G1, G2], "A-2", "pt")]
    run, _ = drive(items, h, repeats=1)
    s = S.summarize(run, items, mode="dev")
    assert s["unusableObserved"]["count"] == 14 and s["entryPass"] is False and s["recommendation"] == "entry check failed: do not edit the prompt, grammar or resolver after dev2 exists; held-out items are never used for tuning"
    assert s["overAsking"] == {"k": 12, "n": 12, "wilson95": S.wilson(12, 12)}


def test_parser_resolved_items_are_tallied_apart_with_one_repeat_and_no_flip_rate():
    def h(q, b, n, log):
        return doc(request=req(), prov=None)                      # no provenance, no model request
    run, stub = drive([unamb()], h)
    (rec,) = run["items"]
    assert rec["parserResolved"] and rec["modelCalled"] is False and len(rec["repeats"]) == 1 and len(stub.calls) == 1
    s = S.summarize(run, [unamb()], mode="dev")
    assert s["parserResolved"]["count"] == 1 and s["interpreterItems"] == 0 and s["flipRate"] == {"k": 0, "n": 0, "wilson95": None}
    assert s["unambiguousUnderstood"]["k"] == 1


def test_a_not_built_answer_is_scored_not_understood_and_still_checked_for_completions():
    def h(q, b, n, log):
        get(log, bypass=b), post(log)
        raise NotBuilt("unbuilt")
    run, _ = drive([unamb()], h)
    assert all(x["verdict"]["notBuilt"] and not x["understood"] for x in run["items"][0]["repeats"]) and len(run["items"][0]["repeats"]) == 5

    def parser_refused(q, b, n, log):
        raise NotBuilt("a diff")
    run2, stub = drive([unamb()], parser_refused)
    assert len(stub.calls) == 1 and run2["items"][0]["modelCalled"] is False
    assert S.summarize(run2, [unamb()], mode="dev")["notBuilt"] == ["U-1"]


def test_a_not_built_outcome_after_a_model_call_still_passes_the_provenance_and_sampler_checks():
    def wrong_model(q, b, n, log):
        get(log, prov={**PROV, "modelFingerprint": "other|x|y|Q|1"}), post(log)
        raise NotBuilt("unbuilt")
    with pytest.raises(S.HarnessError, match="modelFingerprint"):
        drive([unamb()], wrong_model, expected_interpreter=INTERP)

    def h(q, b, n, log):
        get(log, bypass=b), post(log)
        raise NotBuilt("unbuilt")
    with pytest.raises(S.HarnessError, match="sampler parameters"):
        drive([unamb()], h, expected_sampler_request={"temperature": 0, "top_p": 1})

    def no_flag(q, b, n, log):
        get(log), post(log)                                                                    # the event's provenance lacks cacheBypass on a bypassed repeat
        raise NotBuilt("x")
    with pytest.raises(S.HarnessError, match="without provenance.cacheBypass"):
        drive([unamb()], no_flag)


def test_a_model_call_without_a_draft_observation_is_a_harness_error():
    def h(q, b, n, log):
        log.entries.append({"method": "GET", "path": "/api/v0/models", "body": None, "status": 200})
        post(log)
        return doc(request=req(), bypass=b)                                                   # the rig's interpreter is not the recording one
    with pytest.raises(S.HarnessError, match="draft observation"):
        drive([unamb()], h)


def test_a_model_that_cannot_be_called_is_a_harness_error_not_a_score():
    def h(q, b, n, log):
        raise ModelUnavailable("no broker")
    with pytest.raises(S.HarnessError, match="unavailable"):
        drive([unamb()], h)


# ------------------------------------------------------------------------------------------------------------------ the verdict

def obs(understood, **v):
    verdict = {"understood": understood, "overAsk": False, "unusable": False, "notBuilt": False, "nOptions": None, "extras": None, "capExceeded": False,
               "ambiguityKindEmitted": None, "reason": None, "status": None, **v}
    return {"repeat": 1, "understood": understood, "voided": False, "completions": 1, "verdict": verdict}


def synth(k_en, k_pt, k_amb, *, n_en=12, n_pt=12, n_amb=6, stop=None):
    items, recs = [], []
    for lang, k, n in (("en", k_en, n_en), ("pt", k_pt, n_pt)):
        for i in range(n):
            it = unamb(f"{lang}-{i:02d}", lang)
            items.append(it)
            recs.append({"qid": it["qid"], "lang": lang, "outcome": "unambiguous", "modelCalled": True, "parserResolved": False,
                         "repeats": [obs(i < k, overAsk=i >= k)]})
    for i in range(n_amb):
        it = amb([G1, G2], f"amb-{i}")
        items.append(it)
        recs.append({"qid": it["qid"], "lang": "en", "outcome": "ambiguity", "modelCalled": True, "parserResolved": False,
                     "repeats": [obs(i < k_amb, nOptions=2, extras=0)]})
    return {"items": recs, "stop": stop, "ran": len(recs), "of": len(recs)}, items


@pytest.mark.parametrize("k_en,k_pt,k_amb,passed", [(12, 11, 6, True), (12, 12, 6, True), (11, 11, 6, False),        # pooled 22/24 fails
                                                    (12, 10, 6, False), (10, 12, 6, False),                           # a language at 10/12 fails (pooled 22)
                                                    (12, 11, 5, False)])                                              # one ambiguous item missed
def test_the_heldout_thresholds_pooled_per_language_and_every_ambiguous(k_en, k_pt, k_amb, passed):
    run, items = synth(k_en, k_pt, k_amb)
    s = S.summarize(run, items, mode="heldout")
    assert s["passed"] is passed
    assert s["thresholds"]["pooledMin"] == 23 and s["thresholds"]["perLanguageMin"] == {"en": 11, "pt": 11}


def test_pooled_23_with_a_language_below_11_still_fails():
    run, items = synth(12, 11, 6)
    assert S.summarize(run, items, mode="heldout")["passed"]
    run, items = synth(11, 12, 6)
    assert S.summarize(run, items, mode="heldout")["passed"]


def test_the_thresholds_are_computed_from_the_recount_and_the_report_never_certifies():
    run, items = synth(12, 12, 6)
    s = S.summarize(run, items, mode="heldout")
    assert s["recount"] == {"unambiguous": 24, "ambiguous": 6, "en": 12, "pt": 12}
    assert s["unambiguousUnderstood"] == {"k": 24, "n": 24, "wilson95": [0.862, 1.0]} and "does not certify 0.92" in s["claim"]
    assert s["overAsking"] == {"k": 0, "n": 24, "wilson95": S.wilson(0, 24)}
    run, items = synth(12, 11, 6, n_en=14, n_pt=10)                           # a different split: the ceil rule re-derives, it is not a constant
    assert S.summarize(run, items, mode="heldout")["thresholds"] == {**S.summarize(run, items, mode="heldout")["thresholds"], "pooledMin": 23,
                                                                       "perLanguageMin": {"en": 12, "pt": 9}}


def test_over_asking_is_reported_as_its_own_rate_and_extras_per_ambiguous_item():
    run, items = synth(10, 12, 6)
    s = S.summarize(run, items, mode="heldout")
    assert s["overAsking"]["k"] == 2 and s["overAskedQids"] == ["en-10", "en-11"]
    assert set(s["ambiguousExtras"]) == {f"amb-{i}" for i in range(6)}


def test_the_pooled_composition_counts_parser_resolved_items_and_reports_the_interpreter_only_rate():
    run, items = synth(12, 12, 6)
    for rec in run["items"][:2]:                                                             # two en items answered by the parsers: no model call
        rec["parserResolved"], rec["modelCalled"] = True, False
    run["items"][2]["repeats"][0] = obs(False, overAsk=True)                                  # an interpreter item missed
    s = S.summarize(run, items, mode="heldout")
    assert s["unambiguousUnderstood"]["k"] == 23 and s["parserResolved"]["count"] == 2
    assert s["pooledComposition"] == {"parserResolvedUnambiguous": 2, "interpreterDrawsUnambiguous": 22, "note": s["pooledComposition"]["note"],
                                      "interpreterOnlyUnderstood": {"k": 21, "n": 22, "wilson95": S.wilson(21, 22)}}


@pytest.mark.parametrize("k_unamb,k_amb,ok", [(11, 2, True), (12, 2, True), (10, 2, False), (12, 1, False)])
def test_dev_entry_thresholds(k_unamb, k_amb, ok):
    items = [unamb(f"u{i}", "en") for i in range(12)] + [amb([G1], "a0"), amb([G1], "a1")]
    recs = [{"qid": it["qid"], "lang": "en", "outcome": it["outcome"], "modelCalled": True, "parserResolved": False,
             "repeats": [obs(int(it["qid"][1:]) < k_unamb if it["qid"][0] == "u" else int(it["qid"][1:]) < k_amb)]} for it in items]
    s = S.summarize({"items": recs, "stop": None, "ran": 14, "of": 14}, items, mode="dev")
    assert s["entryPass"] is ok and ("recommendation" in s) is (not ok)


def test_dev_entry_fails_when_more_than_one_reply_is_unusable_even_at_the_threshold():
    items = [unamb(f"u{i}", "en") for i in range(12)] + [amb([G1], "a0"), amb([G1], "a1")]
    recs = [{"qid": it["qid"], "lang": "en", "outcome": it["outcome"], "modelCalled": True, "parserResolved": False,
             "repeats": [obs(True, unusable=(it["qid"] in ("a0", "a1")))]} for it in items]
    s = S.summarize({"items": recs, "stop": None, "ran": 14, "of": 14}, items, mode="dev")
    assert s["unusableObserved"]["count"] == 2 and s["entryPass"] is False


# ------------------------------------------------------------------------------------------------------------------ the real Client/Front seam, end to end

SYSTEM = "shop"
QS = ["if I touch m.py::A who gets hit", "who gets hit if I change m.py::A", "m.py::A, what is affected by a change?"]


def rig_over(g, replies, *, statuses=None, interp=None):
    broker = S.CountingBroker(FakeBroker(replies, statuses=statuses))
    interp = interp or S.recording_interpreter(broker.log)

    def build(anchor, direction):
        return build_answer(FrontReader(g), anchor, question=direction)
    front = Front((g.repo,), build, interpreter=interp, run_factory=I.run_factory(SYSTEM, (g.repo,), broker))
    client = Client(SYSTEM, host="127.0.0.1", port=16399, reader_factory=lambda s, h, p: FrontReader(g), front=front)
    return S.Rig(client, broker.log, lambda: len(interp.cache)), broker, interp


def real_interp():
    return {"endpoint": "local", "modelFingerprint": FINGERPRINT, "quantization": "Q4_K_M", "promptHash": I.prompt_hash(), "grammarHash": I.grammar_hash(),
            "samplingHash": I.sampling_hash(), "resolverVersion": RESOLVER_VERSION}


def e2e_items(g, extra=()):
    gold = {"useCase": "impact", "direction": "impact", "anchors": [{"type": "symbol", "repoId": g.repo, "qname": "m.py::A", "path": "m.py"}], "scope": None}
    return [unamb(f"T-{i}", "en" if i % 2 else "pt", gold, q) for i, q in enumerate(QS)] + list(extra)


def test_end_to_end_through_the_real_client_and_front_five_repeats_with_the_bypass():
    g, _ = world()
    rig, broker, interp = rig_over(g, [draft_reply()])
    items = e2e_items(g)
    run = S.run_items(items, rig.client, repeats=5, log=rig.log, cache_len=rig.cache_len, expected_interpreter=real_interp(),
                      expected_sampler_request=S.sampler_request()["params"])
    assert run["stop"] is None and run["ran"] == 3
    for rec in run["items"]:
        reps = rec["repeats"]
        assert [x["completions"] for x in reps] == [1] * 5 and [x["cacheHit"] for x in reps] == [False] * 5 and [x["understood"] for x in reps] == [True] * 5
        assert [x["cacheBypass"] for x in reps] == [False, True, True, True, True]
    assert len(S.completions(rig.log.entries)) == 15 and len(rig.log.entries) == 30          # 15 GET model reads are in the log and not counted
    assert len(interp.cache) == 3                                                                # only the three repeat-1 draws wrote the cache
    assert run["interpreterObserved"] == real_interp()
    s = S.summarize(run, items, mode="dev")
    assert s["flipRate"] == {"k": 0, "n": 3, "wilson95": S.wilson(0, 3)} and s["perDrawUnderstood"]["k"] == 15


def test_end_to_end_a_duplicate_question_hits_the_cache_on_repeat_1_and_stops_c5():
    g, _ = world()
    rig, _, _ = rig_over(g, [draft_reply()])
    items = e2e_items(g)
    items[1] = {**items[1], "text": items[0]["text"]}                                     # normalizes to the same cache key as the first
    run = S.run_items(items, rig.client, repeats=5, log=rig.log, cache_len=rig.cache_len, expected_interpreter=real_interp())
    assert run["stop"]["code"] == "C5" and run["stop"]["qid"] == "T-1"


def test_end_to_end_a_5xx_then_success_is_a_logged_retry_at_the_front():
    g, _ = world()
    rig, _, _ = rig_over(g, [draft_reply()], statuses=[200, 503])                          # GET ok, first POST 503, retried
    run = S.run_items(e2e_items(g)[:1], rig.client, repeats=1, log=rig.log, cache_len=rig.cache_len, expected_interpreter=real_interp())
    assert run["items"][0]["repeats"][0]["completions"] == 2 and run["items"][0]["repeats"][0]["understood"]


def e2e_run(g, replies, item):
    rig, _, _ = rig_over(g, replies)
    return S.run_items([item], rig.client, repeats=1, log=rig.log, cache_len=rig.cache_len, expected_interpreter=real_interp())


def test_end_to_end_a_reply_that_is_not_json_is_unusable_and_a_valid_draft_naming_nothing_is_not():
    """The planted pair (review B1): the same document, `ambiguous` / `no-anchor` / no options, scored differently by what the interpreter returned."""
    g, _ = world()
    empty_item = amb([], "N-1", text="what happens to it")
    bad = e2e_run(g, ["this is not json"], empty_item)
    v = bad["items"][0]["repeats"][0]["verdict"]
    assert v["unusable"] and not v["understood"] and v["ambiguityKindEmitted"] == "no-anchor" and v["nOptions"] == 0
    valid_empty = e2e_run(g, [draft_reply(groups=(("it", "other"),))], empty_item)
    w = valid_empty["items"][0]["repeats"][0]["verdict"]
    assert not w["unusable"] and w["understood"] and w["ambiguityKindEmitted"] == "no-anchor" and w["nOptions"] == 0 and w["reason"] == "asked"
    u = e2e_run(g, [draft_reply(groups=(("it", "other"),))], unamb("N-2", "en", GOLD, "what happens to it"))
    assert not u["items"][0]["repeats"][0]["verdict"]["unusable"] and u["items"][0]["repeats"][0]["verdict"]["overAsk"]
    schema_invalid = e2e_run(g, ['{"useCase": "impact"}'], empty_item)
    assert schema_invalid["items"][0]["repeats"][0]["verdict"]["unusable"]


def test_the_recording_interpreter_records_every_call_including_cache_hits_with_its_provenance():
    g, _ = world()
    rig, _, interp = rig_over(g, [draft_reply()])
    rig.client.ask(question=QS[0])
    rig.client.ask(question=QS[0])                                                          # the second is a cache hit and is recorded too
    assert [e["draftNone"] for e in rig.log.events] == [False, False] and [e["provenance"]["cacheHit"] for e in rig.log.events] == [False, True]


def test_end_to_end_a_parseable_question_is_parser_resolved_with_no_model_request():
    g, _ = world()
    rig, broker, _ = rig_over(g, [draft_reply()])
    item = e2e_items(g)[0]
    item = {**item, "text": "impact m.py:5"}
    run = S.run_items([item], rig.client, repeats=5, log=rig.log, cache_len=rig.cache_len, expected_interpreter=real_interp())
    (rec,) = run["items"]
    assert rec["parserResolved"] and len(rec["repeats"]) == 1 and rig.log.entries == [] and rec["repeats"][0]["understood"]


def test_end_to_end_the_wrong_model_is_a_harness_error():
    g, _ = world()
    rig, _, _ = rig_over(g, [draft_reply()])
    with pytest.raises(S.HarnessError, match="modelFingerprint"):
        S.run_items(e2e_items(g)[:1], rig.client, repeats=1, log=rig.log, cache_len=rig.cache_len,
                    expected_interpreter={**real_interp(), "modelFingerprint": "qwen/other|qwen|qwen3|Q4_K_M|4096"})


DEV_ITEMS = S.load_items(DEV)
DEV_BY_TEXT = {i["text"]: i for i in DEV_ITEMS}


def dev_answer(q, log, correct=True):
    """The dev items answered by a stub that returns each item's gold (or the wrong direction); a canary question gets a generic usable answer."""
    it = DEV_BY_TEXT.get(q)
    get(log), post(log)
    if it is None or not correct:
        return doc(request=req("depends-on" if it is not None else "impact"))
    if it["outcome"] == "unambiguous":
        return doc(request={**it["gold"], "anchors": [{**a, "symbolId": "1" * 32} if a["type"] == "symbol" else {**a, "cpId": "2" * 32} for a in it["gold"]["anchors"]]})
    return doc("ambiguous", options=[{**o, "rank": i + 1} for i, o in enumerate(it["goldOptions"])], kind=it["ambiguityKind"])


def dev_rig(correct=True):
    stub = Stub(lambda q, b, n, log: dev_answer(q, log, correct))
    return S.Rig(stub, stub.log, lambda: 0), stub


def test_end_to_end_the_dev_file_run_writes_the_record_and_never_the_question_text(tmp_path):
    """The real dev items, answered by a stub that returns each item's gold: the dev entry check passes at 14/14 after the canary."""
    rig, stub = dev_rig()
    out = tmp_path / "dev-run.json"
    res = S.run_dev(DEV, rig, out, home=tmp_path)
    s = res["summary"]
    assert s["entryPass"] is True and s["verdict"] == "dev entry check passed" and s["ran"] == 14
    assert s["unambiguousUnderstood"]["k"] == 12 and s["ambiguousUnderstood"]["k"] == 2 and len(stub.calls) == 14 + len(S.CANARY_QUESTIONS)
    assert res["canary"]["n"] == len(S.CANARY_QUESTIONS) and len(res["canary"]["digest"]) == 64
    text = out.read_text()
    assert not any(i["text"] in text for i in DEV_ITEMS) and json.loads(text)["interpreterObserved"] == INTERP


def test_the_dev_run_refuses_a_set_that_is_not_12_and_2(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(unamb()) + "\n")
    with pytest.raises(S.Refusal, match="not 12 and 2"):
        S.run_dev(p, S.Rig(Stub(good), S.RequestLog(), lambda: 0), tmp_path / "o.json", home=tmp_path)


def test_the_dev_run_refuses_an_items_path_under_the_custodian_area_by_path(tmp_path):
    area = S.custodian_area(tmp_path) / "cf7b2042cce7" / "heldout"
    area.mkdir(parents=True)
    p = area / "ac24.jsonl"
    p.write_text(DEV.read_text())                                                           # dev-shaped content: only the path stops it
    rig, stub = dev_rig()
    with pytest.raises(S.Refusal, match="custodian area"):
        S.run_dev(p, rig, tmp_path / "o.json", home=tmp_path)
    assert stub.calls == []


def test_counting_broker_logs_method_path_body_status_and_the_completions_reply_text():
    b = S.CountingBroker(FakeBroker([draft_reply()], statuses=[200, 503, 200]))
    tok = b.open_run("answer", SYSTEM, "*").token
    b.request(tok, {"target": {"method": "GET", "path": "/api/v0/models"}}, None)
    b.request(tok, {"target": {"method": "POST", "path": S.CHAT_PATH}}, {"x": 1})
    b.request(tok, {"target": {"method": "POST", "path": S.CHAT_PATH}}, {"x": 1})
    assert [(e["method"], e["status"]) for e in b.log.entries] == [("GET", 200), ("POST", 503), ("POST", 200)] and len(S.completions(b.log.entries)) == 2
    assert b.log.entries[2]["reply"] == json.dumps(draft_reply()) and "reply" in b.log.entries[1]


# ------------------------------------------------------------------------------------------------------------------ the canary

def test_the_canary_digest_is_over_the_replies_and_is_stable_and_questions_are_fixed_fx_names():
    def run_with(replies):
        log = S.RequestLog()
        it = iter(replies)
        stub = Stub(lambda q, b, n, lg: (get(lg), post(lg, reply=next(it)), doc(request=req()))[2], log)
        return S.run_canary(stub, log)
    reps = [f"r{i}" for i in range(len(S.CANARY_QUESTIONS))]
    d1, d2 = run_with(reps), run_with(reps)
    assert d1 == d2 and len(d1["digest"]) == 64 and d1["n"] == 6
    assert run_with(reps[:-1] + ["drift"])["digest"] != d1["digest"]
    assert any("fx-" in q for q in S.CANARY_QUESTIONS)
    texts = {i["text"].lower() for i in DEV_ITEMS}
    assert not texts & {q.lower() for q in S.CANARY_QUESTIONS}


def test_the_canary_bypasses_the_cache_and_every_question_must_reach_the_model():
    stub = Stub(good)
    S.run_canary(stub, stub.log)
    assert all(b for _, b in stub.calls)
    log = S.RequestLog()
    with pytest.raises(S.HarnessError, match="did not reach the model"):
        S.run_canary(Stub(lambda q, b, n, lg: doc(request=req()), log), log)


def test_the_canary_runs_through_the_real_client_and_front_and_leaves_the_cache_empty():
    g, _ = world()
    rig, _, interp = rig_over(g, [draft_reply()])
    c = S.run_canary(rig.client, rig.log)
    assert c["n"] == 6 and len(interp.cache) == 0 and len(S.completions(rig.log.entries)) == 6
    other, _, _ = rig_over(g, [draft_reply(direction="depends-on")])
    assert S.run_canary(other.client, other.log)["digest"] != c["digest"]                  # a different reply text is a different digest


# ------------------------------------------------------------------------------------------------------------------ the tree (EP 12.3)

def test_core_files_exclude_eval_packs_tests_and_markdown():
    assert S.is_core_file("ccg/answer.py") and S.is_core_file("clients/cli/main.py") and S.is_core_file("pyproject.toml")
    assert not any(S.is_core_file(f) for f in ("eval/tools/x.py", "packs/p/rules.yaml", "tests/test_x.py", "ccg/README.md", "docs/a.md"))


def test_the_tree_hashes_follow_content_exclusions_and_are_order_independent(tmp_path):
    files = {"ccg/a.py": b"a", "clients/b.py": b"b", "eval/tools/t.py": b"t", "tests/test_a.py": b"x", "ccg/NOTES.md": b"n", "packs/p/r.yaml": b"r1", "packs/p/s.yaml": b"s"}
    for rel, data in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    names = sorted(files)
    core, packs = S.core_tree_hash(names, tmp_path), S.pack_set_hash(names, tmp_path)
    assert S.core_tree_hash(list(reversed(names)), tmp_path) == core and len(core) == 64 and core != packs
    for rel in ("eval/tools/t.py", "tests/test_a.py", "ccg/NOTES.md", "packs/p/r.yaml"):          # excluded from the core hash
        (tmp_path / rel).write_bytes(b"changed")
        assert S.core_tree_hash(names, tmp_path) == core
    (tmp_path / "ccg/a.py").write_bytes(b"A")
    assert S.core_tree_hash(names, tmp_path) != core
    assert S.pack_set_hash(names, tmp_path) != packs                                       # the pack file changed above
    (tmp_path / "clients/b.py").write_bytes(b"b")


def test_the_real_tree_is_observed_read_only_and_is_deterministic():
    a, b = S.observe_tree("HEAD"), S.observe_tree("HEAD")
    assert a == b and len(a["coreTreeHash"]) == 64 and a["coreCommitIsAncestorOfHead"] is True and a["harnessHash"] == S.harness_hash()
    assert "ccg/answer.py" in S.tracked_files() and set(a) >= {"head", "dirty", "clean"}


def test_tree_differences_name_each_reason():
    rec = {"coreTreeHash": "a", "packSetHash": "b", "harnessHash": "c"}
    ok = {"coreTreeHash": "a", "packSetHash": "b", "harnessHash": "c", "clean": True, "coreCommitIsAncestorOfHead": True}
    assert S.tree_differences(rec, ok) == []
    for k, v in (("coreTreeHash", "x"), ("packSetHash", "x"), ("harnessHash", "x"), ("clean", False), ("coreCommitIsAncestorOfHead", False), ("coreCommitIsAncestorOfHead", None)):
        assert len(S.tree_differences(rec, {**ok, k: v})) == 1


# ------------------------------------------------------------------------------------------------------------------ N1 (dev result bound to the tree) and N3 (ignored files)

import subprocess

STAMP = {"coreTreeHash": "2" * 64, "packSetHash": "3" * 64, "harnessHash": "4" * 64, "head": "h" * 40, "dirty": [], "clean": True, "coreCommitIsAncestorOfHead": None}


def test_run_dev_stamps_the_observed_tree_into_its_result(tmp_path):
    rig, _ = dev_rig()
    res = S.run_dev(DEV, rig, tmp_path / "d.json", home=tmp_path, tree_state=lambda c: dict(STAMP))
    assert res["treeStamp"] == STAMP and json.loads((tmp_path / "d.json").read_text())["treeStamp"] == STAMP


def test_run_dev_observes_the_tree_again_after_the_run_and_a_change_is_a_harness_error(tmp_path):
    states = iter([dict(STAMP), {**STAMP, "coreTreeHash": "9" * 64}])
    rig, _ = dev_rig()
    with pytest.raises(S.HarnessError, match="changed during the dev run"):
        S.run_dev(DEV, rig, tmp_path / "d.json", home=tmp_path, tree_state=lambda c: next(states))


def test_cli_dev_passes_its_tree_state_through_to_the_stamp(tmp_path):
    out = tmp_path / "o.json"
    assert S.main(["dev", "--out", str(out)], rig_factory=lambda s: dev_rig(True)[0], home=tmp_path, tree_state=lambda c: dict(STAMP)) == 0
    assert json.loads(out.read_text())["treeStamp"] == STAMP


def stamped_dev(tmp_path, stamp=STAMP):
    rig, _ = dev_rig()
    p = tmp_path / "dev.json"
    S.run_dev(DEV, rig, p, home=tmp_path, tree_state=lambda c: dict(stamp))
    return p


def record_cli(dev, tree, *extra):
    return S.main(["prereg-record", "--dev-result", str(dev), "--eval-commit", EVAL, "--core-commit", "1" * 40, "--heldout-list-sha256", "5" * 64, *extra],
                  tree_state=lambda c: dict(tree))


def test_prereg_record_takes_its_three_hashes_from_the_dev_stamp_when_the_tree_still_equals_it(tmp_path, capsys):
    dev = stamped_dev(tmp_path)
    assert record_cli(dev, STAMP) == 0
    rec = json.loads(capsys.readouterr().out)
    assert (rec["coreTreeHash"], rec["packSetHash"], rec["harnessHash"]) == ("2" * 64, "3" * 64, "4" * 64)


@pytest.mark.parametrize("key", ["coreTreeHash", "packSetHash", "harnessHash"])
def test_prereg_record_refuses_when_the_tree_changed_since_the_dev_run(tmp_path, capsys, key):
    dev = stamped_dev(tmp_path)
    assert record_cli(dev, {**STAMP, key: "9" * 64}) == 4
    assert key in capsys.readouterr().err


def test_prereg_record_refuses_a_dirty_tree_a_dirty_stamp_and_a_missing_stamp(tmp_path, capsys):
    assert record_cli(stamped_dev(tmp_path), {**STAMP, "clean": False, "dirty": [" M cpg/ccg/x.py"]}) == 4 and "not clean" in capsys.readouterr().err
    dirty = {**STAMP, "clean": False, "dirty": [" M cpg/ccg/x.py"]}
    assert record_cli(stamped_dev(tmp_path, dirty), STAMP) == 4 and "not clean" in capsys.readouterr().err
    rig, _ = dev_rig()
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({k: v for k, v in S.run_dev(DEV, rig, Path(os.devnull), home=tmp_path, tree_state=lambda c: dict(STAMP)).items() if k != "treeStamp"}))
    assert record_cli(bare, STAMP) == 4 and "stamp" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--core-tree-hash", "--pack-set-hash", "--harness-hash"])
def test_the_hash_override_flags_no_longer_exist(tmp_path, flag):
    assert record_cli(stamped_dev(tmp_path), STAMP, flag, "a" * 64) == 2


def git(repo, *a):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True)


def throwaway_repo(tmp_path):
    repo = tmp_path / "r"
    (repo / "cpg" / "ccg").mkdir(parents=True)
    (repo / "cpg" / "clients").mkdir()
    (repo / "cpg" / "ccg" / "a.py").write_text("a = 1\n")
    (repo / ".gitignore").write_text("ignored_*.py\n__pycache__/\n")
    git(repo.parent / "r", "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "c")
    return repo


def test_observe_tree_on_a_throwaway_repo_is_clean_then_dirty_and_passes_its_arguments(tmp_path):
    repo = throwaway_repo(tmp_path)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    st = S.observe_tree(head, repo=repo)
    assert st["clean"] is True and st["dirty"] == [] and st["head"] == head and st["coreCommitIsAncestorOfHead"] is True
    (repo / "cpg" / "ccg" / "a.py").write_text("a = 2\n")
    st2 = S.observe_tree("0" * 40, repo=repo)
    assert st2["clean"] is False and st2["coreTreeHash"] != st["coreTreeHash"] and st2["coreCommitIsAncestorOfHead"] is False


def test_an_ignored_python_file_under_ccg_or_clients_makes_the_tree_not_clean_but_pycache_does_not(tmp_path):
    repo = throwaway_repo(tmp_path)
    (repo / "cpg" / "ccg" / "__pycache__").mkdir()
    (repo / "cpg" / "ccg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"x")
    assert S.observe_tree(None, repo=repo)["clean"] is True
    (repo / "cpg" / "clients" / "ignored_shim.py").write_text("x = 1\n")
    st = S.observe_tree(None, repo=repo)
    assert st["clean"] is False and any("ignored_shim.py" in d for d in st["dirty"])


def test_the_real_tree_has_no_ignored_files_under_the_clean_paths_other_than_bytecode():
    """N3 on the real tree: whatever is ignored under CLEAN_PATHS is listed in ``dirty`` (this fails with the path named if a stray ignored file exists)."""
    st = S.observe_tree(None)
    assert not [d for d in st["dirty"] if d.startswith("!!")], st["dirty"]


# ------------------------------------------------------------------------------------------------------------------ the held-out guards (synthetic items, a temp root)

EVAL = "e" * 40
TREE = {"coreTreeHash": "2" * 64, "packSetHash": "3" * 64, "harnessHash": "4" * 64, "head": "h" * 40, "dirty": [], "clean": True, "coreCommitIsAncestorOfHead": True}


def stub_canary():
    log = S.RequestLog()
    return S.run_canary(Stub(good, log), log)


def prereg_record(list_sha="0" * 64, **over):
    r = {"schema": S.PREREG_SCHEMA, "evalCommit": EVAL, "coreCommit": "1" * 40, "coreTreeHash": TREE["coreTreeHash"], "packSetHash": TREE["packSetHash"],
         "harnessHash": TREE["harnessHash"], "interpreter": dict(INTERP), "capFloor": 5, "capRule": S.CAP_RULE, "devExtras": [], "cacheBypass": "cacheBypass",
         "tieReading": None, "recordedBy": "teco", "canary": stub_canary(), "heldoutListSha256": list_sha, "expectedHeldoutCounts": dict(S.EXPECTED_HELDOUT),
         "sampler": {"request": {"params": {"temperature": 0}, "samplingHash": "c" * 32}, "effective": None, "effectiveStatus": "not-observable"}}
    r.update(over)
    return r


class Env:
    def __init__(self, tmp, *, record=True, committed=True, dev_pass=True, items=True, n=(12, 12, 6), root=None, **rec_over):
        self.tmp = tmp
        self.root = root or tmp / "custodian"
        self.marker_dir = tmp / "marker-dir"
        (self.root / "heldout").mkdir(parents=True)
        self.prereg = tmp / "preregistration.jsonl"
        self.other = json.dumps({"id": "ccg-p1-flow-fallback-v1"})
        self.record, self.committed_flag, self.rec_over = record, committed, rec_over
        self.dev = tmp / "dev-run.json"
        self.dev.write_text(json.dumps({"summary": {"entryPass": dev_pass}, "interpreterObserved": INTERP, "canary": stub_canary()}))
        self.items_path = self.root / "heldout" / "ac24.jsonl"
        self.manifest = tmp / "heldout-manifest.json"
        self.has_items = items
        if items:
            rows = ([unamb(f"en-{i}", "en", text=f"text en-{i}") for i in range(n[0])] + [unamb(f"pt-{i}", "pt", text=f"text pt-{i}") for i in range(n[1])]
                    + [amb([G1, G2], f"amb-{i}", text=f"text amb-{i}") for i in range(n[2])])
            self.set_items(rows)
        else:
            self._write_record("0" * 64)
        self.rigs = []

    def set_items(self, rows):
        self.items_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        sha = S._file_sha(self.items_path)
        self.manifest.write_text(json.dumps({"files": [{"name": "ac24.jsonl", "sha256": sha}]}))
        self._write_record(sha)

    def _write_record(self, sha):
        self.line = json.dumps(prereg_record(sha, **self.rec_over), sort_keys=True, separators=(",", ":"))
        self.prereg.write_text(self.other + "\n" + (self.line + "\n" if self.record else ""))
        self.committed = self.other + "\n" + (self.line + "\n" if self.committed_flag else "")

    def rig(self, handler=good, observer=None, cache=0, factory_error=None):
        def make():
            if factory_error:
                raise factory_error
            stub = Stub(handler)
            self.rigs.append(stub)
            return S.Rig(stub, stub.log, lambda: cache, observer)
        return make

    def go(self, *, explicit=True, handler=good, observer=None, cache=0, factory_error=None, eval_commit=EVAL, state=None, harness=TREE["harnessHash"], manifest=True, **kw):
        return S.run_heldout(explicit=explicit, root=self.root, marker_dir=self.marker_dir, prereg_path=self.prereg, dev_result_path=self.dev,
                             rig_factory=self.rig(handler, observer, cache, factory_error), committed_text=lambda p: self.committed, eval_commit=eval_commit,
                             tree_state=lambda core: {**TREE, **(state or {})}, harness_sha=lambda: harness, manifest_path=self.manifest if manifest else None, **kw)

    def marker(self):
        return self.marker_dir / f"ac24-heldout-once-{EVAL[:12]}.json"


def heldout_handler(item_by_text):
    def h(q, b, n, log):
        it = item_by_text.get(q)
        get(log), post(log)
        if it is None or it["outcome"] == "unambiguous":
            return doc(request=req(), bypass=b)
        return doc("ambiguous", options=[{**G1, "rank": 1}, {**G2, "rank": 2}], kind="several-symbols", bypass=b)
    return h


def correct_unamb(item):
    return {**item, "gold": GOLD}


def test_heldout_happy_path_on_synthetic_items_writes_only_under_the_root(tmp_path):
    e = Env(tmp_path)
    rows = [correct_unamb(r) for r in S.load_items(e.items_path) if r["outcome"] == "unambiguous"] + [r for r in S.load_items(e.items_path) if r["outcome"] != "unambiguous"]
    e.set_items(rows)
    res = e.go(handler=heldout_handler({r["text"]: r for r in rows}))
    s = res["summary"]
    assert s["passed"] is True and s["recount"] == {"unambiguous": 24, "ambiguous": 6, "en": 12, "pt": 12}
    assert sorted(p.name for p in e.root.iterdir()) == ["ac24-heldout-run.json", "heldout"] and [p.name for p in e.marker_dir.iterdir()] == [e.marker().name]
    assert json.loads(e.marker().read_text())["state"] == "completed"
    assert sum(len(r["repeats"]) for r in res["items"]) == 150                               # 30 items x 5 repeats
    assert not any(r["text"] in (e.root / "ac24-heldout-run.json").read_text() for r in rows)
    assert s["tree"]["afterAdaptation"] is False and s["tree"]["clean"] is True and s["tree"]["head"] == "h" * 40 and s["tree"]["harnessHash"] == TREE["harnessHash"]
    assert s["pooledComposition"]["parserResolvedUnambiguous"] == 0 and s["pooledComposition"]["interpreterDrawsUnambiguous"] == 24


def test_heldout_refuses_without_the_explicit_flag_and_touches_nothing(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(S.Refusal, match="--heldout"):
        e.go(explicit=False)
    assert not e.marker().exists() and e.rigs == []


@pytest.mark.parametrize("kw,match", [({"record": False}, "holds 0"), ({"committed": False}, "not committed"), ({"dev_pass": False}, "dev entry"),
                                      ({"items": False}, "no held-out AC-24 list")])
def test_heldout_refuses_without_its_preconditions_and_leaves_no_marker(tmp_path, kw, match):
    e = Env(tmp_path, **kw)
    with pytest.raises(S.Refusal, match=match):
        e.go()
    assert not e.marker().exists() and e.rigs == []


def test_heldout_refuses_when_the_preregistration_file_is_absent(tmp_path):
    e = Env(tmp_path)
    e.prereg.unlink()
    with pytest.raises(S.Refusal, match="no preregistration file"):
        e.go()
    assert not e.marker().exists()


def test_heldout_refuses_an_evalcommit_or_dev_interpreter_or_dev_canary_that_is_not_the_records(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(S.Refusal, match="EVAL_COMMIT"):
        e.go(eval_commit="f" * 40)
    good_dev = json.loads(e.dev.read_text())
    e.dev.write_text(json.dumps({**good_dev, "interpreterObserved": {**INTERP, "promptHash": "9" * 64}}))
    with pytest.raises(S.Refusal, match="interpreter"):
        e.go()
    e.dev.write_text(json.dumps({**good_dev, "canary": {**good_dev["canary"], "digest": "9" * 64}}))
    with pytest.raises(S.Refusal, match="canary"):
        e.go()
    assert not e.marker().exists()


def test_a_malformed_dev_result_is_a_clean_refusal_not_a_traceback(tmp_path):
    e = Env(tmp_path)
    e.dev.write_text("{not json")
    with pytest.raises(S.Refusal, match="not valid JSON"):
        e.go()
    e.dev.write_text("[1]")
    with pytest.raises(S.Refusal, match="not a JSON object"):
        e.go()
    e.dev.unlink()
    with pytest.raises(S.Refusal, match="missing"):
        e.go()
    assert not e.marker().exists()


def test_the_default_committed_check_refuses_a_file_outside_any_repository(tmp_path):
    p = tmp_path / "preregistration.jsonl"
    p.write_text(json.dumps(prereg_record()) + "\n")
    with pytest.raises(S.Refusal, match="not committed"):
        S.find_prereg(p)


def test_heldout_runs_once_a_second_invocation_is_refused_whatever_the_first_said(tmp_path):
    e = Env(tmp_path)
    first = e.go()                                                                            # gold GOLD vs the stub's answers: a failing run, reported as run
    assert first["summary"]["passed"] is False
    assert e.marker().exists()
    with pytest.raises(S.Refusal, match="already started"):
        e.go()
    assert len(e.rigs) == 2 and len(e.rigs[1].calls) == len(S.CANARY_QUESTIONS)               # the second run got through pre-flight only (the canary), no item


def test_the_marker_is_bound_to_the_eval_commit_and_a_copied_heldout_list_gets_no_fresh_one(tmp_path):
    e = Env(tmp_path)
    e.go()
    copy_root = tmp_path / "copy"
    (copy_root / "heldout").mkdir(parents=True)
    (copy_root / "heldout" / "ac24.jsonl").write_bytes(e.items_path.read_bytes())
    assert e.marker().name == f"ac24-heldout-once-{EVAL[:12]}.json"
    with pytest.raises(S.Refusal, match="already started"):
        S.run_heldout(explicit=True, root=copy_root, marker_dir=e.marker_dir, prereg_path=e.prereg, dev_result_path=e.dev, rig_factory=e.rig(),
                      committed_text=lambda p: e.committed, eval_commit=EVAL, tree_state=lambda c: dict(TREE), harness_sha=lambda: TREE["harnessHash"],
                      manifest_path=e.manifest)


def test_a_harness_error_after_the_marker_aborts_the_run_and_the_marker_still_refuses_a_rerun(tmp_path):
    e = Env(tmp_path)

    def wrong_model(q, b, n, log):
        get(log), post(log)
        return doc(request=req(), prov={**PROV, "modelFingerprint": "other|x|y|Q|1"} if q.startswith("text") else PROV, bypass=b)
    with pytest.raises(S.HarnessError):
        e.go(handler=wrong_model)
    assert json.loads(e.marker().read_text())["state"] == "aborted"
    with pytest.raises(S.Refusal, match="already started"):
        e.go()


def test_a_split_with_other_counts_is_refused_after_the_marker_for_the_method_owner_to_recompute(tmp_path):
    e = Env(tmp_path, n=(12, 11, 6))
    with pytest.raises(S.Refusal, match="recomputes the thresholds"):
        e.go()
    assert json.loads(e.marker().read_text())["state"] == "aborted" and len(e.rigs[0].calls) == len(S.CANARY_QUESTIONS)


# every check that does not need the items, and the rig, happens before the marker

@pytest.mark.parametrize("kw,exc,match", [
    ({"state": {"coreTreeHash": "9" * 64}}, S.Refusal, "coreTreeHash"),
    ({"state": {"packSetHash": "9" * 64}}, S.Refusal, "packSetHash"),
    ({"harness": "9" * 64}, S.Refusal, "harnessHash"),
    ({"state": {"clean": False, "dirty": [" M cpg/ccg/x.py"]}}, S.Refusal, "uncommitted"),
    ({"state": {"coreCommitIsAncestorOfHead": False}}, S.Refusal, "ancestor"),
    ({"factory_error": RuntimeError("no such system")}, RuntimeError, "no such system"),
    ({"cache": 3}, S.HarnessError, "not empty"),
    ({"handler": lambda q, b, n, log: (get(log), post(log, reply="drifted"), doc(request=req()))[2]}, S.HarnessError, "canary digest"),
])
def test_preflight_failures_leave_no_marker_so_they_do_not_burn_the_run(tmp_path, kw, exc, match):
    e = Env(tmp_path)
    with pytest.raises(exc, match=match):
        e.go(**kw)
    assert not e.marker().exists() and not e.marker_dir.exists()
    assert not (e.root / "ac24-heldout-run.json").exists()


def test_an_adapted_tree_runs_only_with_the_override_and_is_stamped(tmp_path):
    e = Env(tmp_path)
    res = e.go(state={"coreTreeHash": "9" * 64, "clean": False}, after_adaptation=True)
    t = res["summary"]["tree"]
    assert t["afterAdaptation"] is True and len(t["afterAdaptationReasons"]) == 2 and res["summary"]["verdict"].endswith("(after adaptation)")


def test_the_held_out_list_must_match_the_records_hash_and_the_manifests(tmp_path):
    e = Env(tmp_path)
    e.items_path.write_text(e.items_path.read_text() + "\n")                                  # a different byte sequence
    with pytest.raises(S.Refusal, match="not the record's"):
        e.go()
    (tmp_path / "b").mkdir()
    e2 = Env(tmp_path / "b")
    e2.manifest.write_text(json.dumps({"files": [{"name": "ac24.jsonl", "sha256": "9" * 64}]}))
    with pytest.raises(S.Refusal, match="manifest"):
        e2.go()
    e2.manifest.write_text(json.dumps({"files": []}))
    with pytest.raises(S.Refusal, match="manifest"):
        e2.go()
    assert not e.marker().exists() and not e2.marker().exists()


# the record's sampler values reach the checks (through run_heldout, not only through run_items)

def test_the_records_request_sampler_reaches_the_check_and_a_mismatch_aborts(tmp_path):
    e = Env(tmp_path, sampler={"request": {"params": {"temperature": 0, "top_p": 1}, "samplingHash": "c" * 32}, "effective": None, "effectiveStatus": "not-observable"})
    with pytest.raises(S.HarnessError, match="sampler parameters"):
        e.go()
    assert json.loads(e.marker().read_text())["state"] == "aborted"


def test_the_records_effective_sampler_reaches_the_check_and_a_mismatch_is_a_preflight_error(tmp_path):
    smp = {"request": {"params": {"temperature": 0}, "samplingHash": "c" * 32}, "effective": {"top_k": 0, "repeat_penalty": 1.0}, "effectiveStatus": "observed"}
    e = Env(tmp_path, sampler=smp)
    with pytest.raises(S.HarnessError, match="effective sampler"):
        e.go(observer=lambda: {"top_k": 40, "repeat_penalty": 1.1})
    assert not e.marker().exists()
    res = e.go(observer=lambda: {"top_k": 0, "repeat_penalty": 1.0})
    assert res["samplerEffective"] == {"top_k": 0, "repeat_penalty": 1.0}


def test_the_effective_sampler_is_checked_again_during_the_run_so_drift_after_preflight_aborts(tmp_path):
    smp = {"request": {"params": {"temperature": 0}, "samplingHash": "c" * 32}, "effective": {"top_k": 0}, "effectiveStatus": "observed"}
    e = Env(tmp_path, sampler=smp)
    calls = []

    def drifting():
        calls.append(1)
        return {"top_k": 0} if len(calls) == 1 else {"top_k": 40}                            # equal at pre-flight, changed when the run starts
    with pytest.raises(S.HarnessError, match="effective sampler"):
        e.go(observer=drifting)
    assert json.loads(e.marker().read_text())["state"] == "aborted"


def test_a_recorded_effective_sampler_with_no_observer_is_a_preflight_refusal(tmp_path):
    smp = {"request": {"params": {"temperature": 0}, "samplingHash": "c" * 32}, "effective": {"top_k": 0}, "effectiveStatus": "observed"}
    e = Env(tmp_path, sampler=smp)
    with pytest.raises(S.Refusal, match="no observer"):
        e.go()
    assert not e.marker().exists()


def test_the_record_must_be_unique_and_complete(tmp_path):
    p = tmp_path / "p.jsonl"
    line = json.dumps(prereg_record())
    p.write_text(line + "\n" + line + "\n")
    with pytest.raises(S.Refusal, match="holds 2"):
        S.find_prereg(p, committed_text=lambda _: line + "\n" + line + "\n")
    bad_records = [prereg_record(capFloor=4), prereg_record(cacheBypass="noCache"), prereg_record(interpreter={**INTERP, "promptHash": ""}),
                   prereg_record(harnessHash=""), prereg_record(heldoutListSha256=None), prereg_record(canary={}), prereg_record(expectedHeldoutCounts={"unambiguous": 23}),
                   prereg_record(sampler={"request": {}, "effective": None}), prereg_record(coreTreeHash=""), prereg_record(packSetHash=""), prereg_record(coreCommit="")]
    for bad in bad_records:
        t = json.dumps(bad)
        p.write_text(t + "\n")
        with pytest.raises(S.Refusal):
            S.find_prereg(p, committed_text=lambda _, t=t: t + "\n")


def test_load_items_does_not_echo_an_unknown_outcome(tmp_path):
    p = tmp_path / "i.jsonl"
    p.write_text(json.dumps({**unamb(), "outcome": "SECRET-LABEL"}) + "\n")
    with pytest.raises(S.Refusal) as e:
        S.load_items(p)
    assert "SECRET-LABEL" not in str(e.value)


def test_the_custodian_root_is_derived_from_the_eval_commit(tmp_path):
    f = tmp_path / "EVAL_COMMIT"
    f.write_text("cf7b2042cce7bad5d0ab7876940bcea00343d914\n")
    assert S.custodian_root(f, tmp_path) == tmp_path / ".local" / "share" / "ccg-eval" / "cf7b2042cce7"


# ------------------------------------------------------------------------------------------------------------------ the record builder

def dev_result():
    rig, _ = dev_rig()
    return S.run_dev(DEV, rig, Path(os.devnull), home=Path("/nonexistent-home"), tree_state=lambda c: dict(TREE))


def build(res, **kw):
    return S.build_prereg_record(res, eval_commit=EVAL, core_commit="1" * 40, core_tree_hash="2" * 64, pack_set_hash="3" * 64, harness_hash="4" * 64,
                                 heldout_list_sha256="5" * 64, **kw)


def test_the_record_is_built_from_the_dev_runs_provenance_and_is_not_appended(tmp_path):
    res = dev_result()
    prereg = tmp_path / "preregistration.jsonl"
    prereg.write_text('{"id":"x"}\n')
    before = prereg.read_bytes()
    rec = build(res)
    assert prereg.read_bytes() == before
    assert rec["schema"] == "ccg-prereg-ac24-v1" and rec["interpreter"] == INTERP and rec["capFloor"] == 5 and rec["capRule"] == "max(capFloor, |goldOptions_i|)"
    assert rec["cacheBypass"] == "cacheBypass" and rec["tieReading"] is None and rec["recordedBy"] == "teco"
    assert rec["devExtras"] == [{"qid": "QW-19", "options": 3, "extras": 0}, {"qid": "QW-47", "options": 2, "extras": 0}]
    assert rec["sampler"]["request"]["samplingHash"] == I.sampling_hash() and rec["sampler"]["request"]["params"] == dict(I.SAMPLING)
    assert rec["sampler"]["effective"] is None and rec["sampler"]["effectiveStatus"] == "not-observable"
    assert rec["canary"] == res["canary"] and rec["harnessHash"] == "4" * 64 and rec["heldoutListSha256"] == "5" * 64 and rec["expectedHeldoutCounts"] == S.EXPECTED_HELDOUT
    assert rec["lmStudioVersion"] is None and rec["lmStudioVersionStatus"] == "not-obtainable"
    assert rec["repeatProtocol"]["repeats"] == 5 and "draft is None" in rec["unusableObservable"]
    assert "neutral (direction null)" in rec["directionCues"] and rec["resolverRules"] and "refused whole" in rec["unbuiltReadingsRule"]
    assert build(res, sampler_effective={"top_k": 0})["sampler"]["effectiveStatus"] == "observed" and build(res, lm_studio_version="0.3.9")["lmStudioVersionStatus"] == "observed"
    json.dumps(rec)


def test_the_record_builder_refuses_a_dev_run_with_no_provenance_or_no_canary():
    res = dev_result()
    with pytest.raises(S.Refusal, match="no provenance"):
        build({**res, "interpreterObserved": None})
    with pytest.raises(S.Refusal, match="canary"):
        build({**res, "canary": None})


def test_a_record_built_from_a_dev_run_is_accepted_by_find_prereg(tmp_path):
    rec = build(dev_result())
    p = tmp_path / "p.jsonl"
    line = S.canonical_json(rec)
    p.write_text(line + "\n")
    assert S.find_prereg(p, committed_text=lambda _: line + "\n", eval_commit=EVAL)["canary"] == rec["canary"]


# ------------------------------------------------------------------------------------------------------------------ the CLI

def test_cli_dev_exit_codes_and_stdout_carry_no_question_text(tmp_path, capsys):
    out = tmp_path / "o.json"
    assert S.main(["dev", "--out", str(out)], rig_factory=lambda s: dev_rig(True)[0], home=tmp_path) == 0
    shown = capsys.readouterr().out
    assert "dev entry check passed" in shown and not any(i["text"] in shown for i in DEV_ITEMS) and "pooled: 0 parser, 12 interpreter" in shown
    assert S.main(["dev", "--out", str(out)], rig_factory=lambda s: dev_rig(False)[0], home=tmp_path) == 1
    assert "dev entry check failed" in capsys.readouterr().out


def test_cli_canary_and_tree_hashes_print_a_digest_and_the_tree(tmp_path, capsys):
    assert S.main(["canary"], rig_factory=lambda s: dev_rig()[0]) == 0
    assert len(json.loads(capsys.readouterr().out)["digest"]) == 64
    assert S.main(["tree-hashes"], tree_state=lambda c: dict(TREE)) == 0
    assert json.loads(capsys.readouterr().out)["coreTreeHash"] == TREE["coreTreeHash"]


def test_cli_heldout_without_the_flag_is_refused_exit_4_and_usage_errors_exit_2(tmp_path, capsys):
    derived = S.custodian_root(home=tmp_path)
    e = Env(tmp_path, root=derived)
    args = ["heldout", "--prereg", str(e.prereg), "--dev-result", str(e.dev)]
    assert S.main(args, committed_text=lambda p: e.committed, home=tmp_path) == 4
    assert "--heldout" in capsys.readouterr().err and not e.marker().exists()
    assert S.main(["bogus"]) == 2


def test_cli_heldout_refuses_a_custodian_root_other_than_the_derived_one(tmp_path, capsys):
    e = Env(tmp_path, root=S.custodian_root(home=tmp_path))
    other = tmp_path / "elsewhere"
    code = S.main(["heldout", "--heldout", "--custodian-root", str(other), "--prereg", str(e.prereg), "--dev-result", str(e.dev)],
                  committed_text=lambda p: e.committed, home=tmp_path)
    assert code == 4 and "derived" in capsys.readouterr().err and not other.exists()


def good_rig():
    stub = Stub(good)
    return S.Rig(stub, stub.log, lambda: 0)


def test_cli_heldout_runs_in_the_derived_root_with_the_marker_there_and_the_after_adaptation_flag(tmp_path, capsys):
    derived = S.custodian_root(home=tmp_path)
    e = Env(tmp_path, root=derived)
    common = dict(committed_text=lambda p: e.committed, home=tmp_path, eval_commit=EVAL, manifest_path=e.manifest, rig_factory=lambda s: good_rig())
    args = ["heldout", "--heldout", "--prereg", str(e.prereg), "--dev-result", str(e.dev)]
    dirty = lambda c: {**TREE, "clean": False}                                        # noqa: E731
    assert S.main(args, tree_state=dirty, **common) == 4 and "after-adaptation" in capsys.readouterr().err
    assert not (derived / f"ac24-heldout-once-{EVAL[:12]}.json").exists()
    code = S.main([*args, "--after-adaptation"], tree_state=dirty, **common)
    shown = capsys.readouterr().out
    assert code == 1 and "(after adaptation)" in shown and str(derived / "ac24-heldout-run.json") in shown            # the stub's answers fail the gate, reported as run
    assert json.loads((derived / f"ac24-heldout-once-{EVAL[:12]}.json").read_text())["state"] == "completed"
    assert json.loads((derived / "ac24-heldout-run.json").read_text())["summary"]["tree"]["afterAdaptation"] is True


def test_cli_a_malformed_dev_result_exits_4_not_with_a_traceback(tmp_path, capsys):
    bad = tmp_path / "dev.json"
    bad.write_text("{oops")
    code = S.main(["prereg-record", "--dev-result", str(bad), "--eval-commit", EVAL, "--core-commit", "1" * 40, "--heldout-list-sha256", "5" * 64],
                  tree_state=lambda c: dict(TREE))
    assert code == 4 and "not valid JSON" in capsys.readouterr().err


def record_args(dev, **extra):
    return ["prereg-record", "--dev-result", str(dev), "--eval-commit", EVAL, "--core-commit", "1" * 40, "--heldout-list-sha256", "5" * 64,
            *[x for kv in extra.items() for x in kv]]


def test_cli_prereg_record_prints_and_does_not_append(tmp_path, capsys):
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(dev_result()))
    prereg = CPG / "eval" / "preregistration.jsonl"
    before = prereg.read_bytes()
    code = S.main(record_args(dev), tree_state=lambda c: dict(TREE))
    line = capsys.readouterr().out.strip()
    assert code == 0 and json.loads(line)["schema"] == "ccg-prereg-ac24-v1" and prereg.read_bytes() == before


def test_cli_prereg_record_out_never_overwrites_a_file_or_the_registry(tmp_path, capsys):
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(dev_result()))
    existing = tmp_path / "existing.json"
    existing.write_text("keep")
    assert S.main(record_args(dev, **{"--out": str(existing)}), tree_state=lambda c: dict(TREE)) == 4 and existing.read_text() == "keep"
    reg = CPG / "eval" / "preregistration.jsonl"
    before = reg.read_bytes()
    assert S.main(record_args(dev, **{"--out": str(reg)}), tree_state=lambda c: dict(TREE)) == 4 and reg.read_bytes() == before
    fresh_named = tmp_path / "sub" / "preregistration.jsonl"
    fresh_named.parent.mkdir()
    assert S.main(record_args(dev, **{"--out": str(fresh_named)}), tree_state=lambda c: dict(TREE)) == 4 and not fresh_named.exists()
    ok = tmp_path / "new.json"
    assert S.main(record_args(dev, **{"--out": str(ok)}), tree_state=lambda c: dict(TREE)) == 0 and json.loads(ok.read_text())["schema"] == "ccg-prereg-ac24-v1"
    capsys.readouterr()


# --------------------------------------------------------------------------------------------------------------- the draft per item and devAttempts (P1-35)

def test_the_recording_interpreter_records_the_draft_in_the_reply_shape_and_none_for_an_unusable_reply():
    g, _ = world()
    reply = draft_reply("depends-on", (("m.py", "file"), ("A", "symbol")), scope="in the shop", readings=(("impact", "impact"),))
    rig, _, _ = rig_over(g, [reply])
    rig.client.ask(question=QS[0])
    assert rig.log.events[0]["draft"] == reply
    rig2, _, _ = rig_over(g, ['{"useCase": "impact"}'])
    rig2.client.ask(question=QS[0])
    assert rig2.log.events[0]["draftNone"] is True and rig2.log.events[0]["draft"] is None


def test_the_run_records_the_observed_draft_on_each_repeat_and_null_when_none_was_observed():
    stub_draft = draft_reply()

    def handler(question, bypass, n, log):
        get(log)
        post(log)
        log.events[-1]["draft"] = stub_draft
        return doc(request=req(), bypass=bypass)
    run, _ = drive([unamb()], handler, repeats=1, record_drafts=True)
    assert run["items"][0]["repeats"][0]["draft"] == stub_draft
    run2, _ = drive([unamb()], good, repeats=1, record_drafts=True)                                             # an event with no draft key: nothing observed, recorded as null
    assert run2["items"][0]["repeats"][0]["draft"] is None


def test_the_draft_field_changes_no_verdict_and_no_stamped_hash_and_is_not_in_the_record(tmp_path):
    marker = draft_reply((("ZZ-DRAFT-SPAN", "symbol"),))

    def handler(question, bypass, n, log):
        get(log)
        post(log)
        log.events[-1]["draft"] = marker
        return doc(request=req(), bypass=bypass)
    plain, _ = drive([unamb()], good, repeats=1)
    with_draft, _ = drive([unamb()], handler, repeats=1, record_drafts=True)
    strip = lambda r: [{k: v for k, v in x.items() if k != "draft"} for x in r["items"][0]["repeats"]]  # noqa: E731
    assert strip(plain) == strip(with_draft)
    res = dev_result()
    res["items"][0]["repeats"][0]["draft"] = marker
    assert "ZZ-DRAFT-SPAN" not in json.dumps(build(res)) and S.summarize(res, DEV_ITEMS, mode="dev") == res["summary"]


ATTEMPT = {"model": "ministral-3-3b-instruct-2512", "counts": {"unambiguous": 9, "ambiguous": 2}, "unusable": 1, "resultSha256": "a" * 64}


def attempts_file(tmp_path, obj):
    p = tmp_path / "attempts.json"
    p.write_text(json.dumps(obj))
    return p


def record_with_attempts(tmp_path, obj):
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(dev_result()))
    return S.main(record_args(dev, **{"--dev-attempts": str(attempts_file(tmp_path, obj))}), tree_state=lambda c: dict(TREE))


def test_prereg_record_copies_dev_attempts_verbatim_and_omits_the_key_without_the_option(tmp_path, capsys):
    other = {**ATTEMPT, "model": "qwen2.5-3b-instruct", "unusable": 0, "resultSha256": "b" * 64}
    assert record_with_attempts(tmp_path, [ATTEMPT, other]) == 0
    assert json.loads(capsys.readouterr().out)["devAttempts"] == [ATTEMPT, other]
    dev = tmp_path / "dev2.json"
    dev.write_text(json.dumps(dev_result()))
    assert S.main(record_args(dev), tree_state=lambda c: dict(TREE)) == 0
    assert "devAttempts" not in json.loads(capsys.readouterr().out)


MALFORMED = [
    {"a": 1}, [], "x", [ATTEMPT, "x"],
    [{k: v for k, v in ATTEMPT.items() if k != "model"}], [{k: v for k, v in ATTEMPT.items() if k != "counts"}],
    [{k: v for k, v in ATTEMPT.items() if k != "unusable"}], [{k: v for k, v in ATTEMPT.items() if k != "resultSha256"}],
    [{**ATTEMPT, "extra": 1}], [{**ATTEMPT, "model": ""}], [{**ATTEMPT, "model": 3}],
    [{**ATTEMPT, "counts": {}}], [{**ATTEMPT, "counts": [1]}], [{**ATTEMPT, "counts": {"unambiguous": "9"}}], [{**ATTEMPT, "counts": {"unambiguous": -1}}],
    [{**ATTEMPT, "counts": {"unambiguous": True}}], [{**ATTEMPT, "unusable": -1}], [{**ATTEMPT, "unusable": "1"}], [{**ATTEMPT, "unusable": True}],
    [{**ATTEMPT, "resultSha256": "a" * 63}], [{**ATTEMPT, "resultSha256": "G" * 64}], [{**ATTEMPT, "resultSha256": "A" * 64}], [{**ATTEMPT, "resultSha256": 5}],
    [{**ATTEMPT, "promptHash": "c" * 63}], [{**ATTEMPT, "promptHash": "C" * 64}], [{**ATTEMPT, "promptHash": "g" * 64}], [{**ATTEMPT, "promptHash": 5}],
    [{**ATTEMPT, "promptHash": None}], [{**ATTEMPT, "promptHash": ""}],
]


@pytest.mark.parametrize("bad", MALFORMED)
def test_prereg_record_refuses_a_malformed_dev_attempts_file(tmp_path, capsys, bad):
    assert record_with_attempts(tmp_path, bad) == 4
    captured = capsys.readouterr()
    assert captured.out == "" and "dev attempts" in captured.err


def test_prereg_record_carries_an_optional_prompt_hash_through_and_keeps_entries_without_it_valid(tmp_path, capsys):
    hashed = {**ATTEMPT, "promptHash": "c" * 64}
    assert record_with_attempts(tmp_path, [ATTEMPT, hashed]) == 0
    assert json.loads(capsys.readouterr().out)["devAttempts"] == [ATTEMPT, hashed]


def test_prereg_record_refuses_an_unreadable_dev_attempts_file(tmp_path, capsys):
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(dev_result()))
    assert S.main(record_args(dev, **{"--dev-attempts": str(tmp_path / "missing.json")}), tree_state=lambda c: dict(TREE)) == 4


def test_a_run_records_no_draft_unless_asked_and_run_dev_asks(tmp_path):
    def handler(question, bypass, n, log):
        get(log)
        post(log)
        log.events[-1]["draft"] = draft_reply()
        return doc(request=req(), bypass=bypass)
    run, _ = drive([unamb()], handler, repeats=1)                                           # the held-out call shape: no record_drafts
    assert "draft" not in run["items"][0]["repeats"][0]
    res = dev_result()
    assert all("draft" in x for r in res["items"] for x in r["repeats"])
