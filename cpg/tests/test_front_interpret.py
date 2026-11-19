"""P1-26 dispatch 2 (2), (4), (6): the interpreter, its cache, the broker run and the provenance (``ccg.front.interpret``, ``ccg.front.cache``, ``ccg.front.Front``).

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-26 behaviours (1), (2), (4), (6), (7); ``-p1-ml.md`` §4.2 (the draft, the grammar, sampling, provenance, the
fabrication guard) and §4.3 (QW-3 and QW-11 must not come back ``null``); ``-answer-spec2.md`` §2 (``cacheBypass``); ``-egress-spec3.md`` §2.1, §2.5, §10.2 (one run per
question, no direct connection). The model is a stub behind a broker double (``front_model_helpers.FakeBroker``); no live model is run here.

    cd cpg && .venv/bin/python -m pytest tests/test_front_interpret.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import socket
from pathlib import Path

import pytest
from engine_answer_helpers import SNAP
from front_helpers import FrontReader
from front_model_helpers import FINGERPRINT, MODEL, FakeBroker, draft_reply
from test_front_documents import structured, world
from test_front_resolve import dev, dev_store

from ccg import answer
from ccg.broker.client import BrokerClient
from ccg.engine.answer_build import build_answer
from ccg.front import Front, ModelError, NotBuiltAsk
from ccg.front import interpret as I
from ccg.front.cache import InterpretCache, key_of
from ccg.front.resolve import RESOLVER_VERSION

FRONT_DIR = Path(__file__).resolve().parent.parent / "ccg" / "front"
Q = "if I touch m.py::A who gets hit"
SYSTEM = "shop"


def front_for(g, broker, *, interpreter=None, reader_world=None):
    interp = interpreter or I.Interpreter()

    def build(anchor, direction):
        return build_answer(FrontReader(g), anchor, question=direction)
    return Front((g.repo,), build, interpreter=interp, run_factory=I.run_factory(SYSTEM, (g.repo,), broker))


def ask(g, broker, q=Q, *, bypass=False, front=None, reader=None):
    return (front or front_for(g, broker))(q, reader=reader or FrontReader(g), cache_bypass=bypass)


def sha(rel):
    return hashlib.sha256((FRONT_DIR / rel).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ (1) the parsers first

@pytest.mark.parametrize("q", ["impact m.py:5", "m.py::A depends-on", "m.py:5"])
def test_a_question_the_parsers_resolve_makes_no_model_call_and_opens_no_run(q):
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    doc = ask(g, broker, q)
    assert doc["interpretation"]["status"] in ("resolved", "ambiguous") and "provenance" not in doc["interpretation"]
    assert broker.opens == [] and broker.requests == [] and broker.closes == []


def test_the_precheck_lets_free_prose_through_and_still_refuses_a_diff_without_a_run():
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    f = front_for(g, broker)
    f.precheck(Q)
    f.precheck("m.py:5")
    with pytest.raises(NotBuiltAsk):
        f.precheck("diff --git a/m.py b/m.py\n@@ -1 +1 @@")
    with pytest.raises(NotBuiltAsk):
        f.precheck("security m.py:5")
    assert broker.opens == []


# ------------------------------------------------------------------------------------------------------------------ (2), (6) one call, one run

def test_free_prose_is_one_run_a_model_read_and_one_completion_through_the_broker_then_closed():
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    doc = ask(g, broker)
    assert doc["interpretation"]["status"] == "resolved"
    assert broker.opens == [("answer", SYSTEM, "*", None)] and len(broker.opens) == 1
    assert [r[1]["target"] for r in broker.requests] == [{"method": "GET", "path": "/api/v0/models"}, {"method": "POST", "path": "/v1/chat/completions"}]
    assert {r[0] for r in broker.requests} == {"f" * 64} and broker.closes == ["f" * 64]
    get, post = (r[1] for r in broker.requests)
    assert get == {"component": "answer.interpret", "purpose": "model-call", "payloadClass": "none", "endpoint": "http://127.0.0.1:1234", "target": get["target"],
                   "codebases": [g.repo]}
    assert post["component"] == "answer.interpret" and post["purpose"] == "model-call" and post["payloadClass"] == "repo-derived"
    assert broker.requests[0][2] is None


def test_the_run_is_closed_when_the_question_fails():
    g, _ = world()
    broker = FakeBroker([draft_reply(use_case="root-cause")])
    with pytest.raises(NotBuiltAsk):
        ask(g, broker)
    assert len(broker.opens) == 1 and broker.closes == ["f" * 64]


def test_the_request_body_is_one_system_message_one_user_message_and_the_fixed_sampler():
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    ask(g, broker, "  if I touch   m.py::A who gets hit ")
    body = broker.completions[0][2]
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == (FRONT_DIR / "prompts" / "interpret.txt").read_text(encoding="utf-8")
    assert body["messages"][1]["content"] == "if I touch m.py::A who gets hit"                       # the normalized question
    assert (body["model"], body["temperature"], body["top_p"], body["max_tokens"], body["seed"], body["stream"]) == ("qwen/qwen3-4b-2507", 0, 1, 384, 0, False)
    schema = json.loads((FRONT_DIR / "grammar" / "interpret.json").read_text(encoding="utf-8"))
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "interpret_draft", "strict": True, "schema": schema}}


def test_two_model_calls_of_one_question_share_one_run_and_one_model_read():
    g, _ = world()
    broker = FakeBroker([draft_reply(), "a short summary"])
    f = front_for(g, broker)
    got = f.answer(Q, reader=FrontReader(g), summary=True)
    assert got.summary is not None
    assert len(broker.opens) == 1 and len(broker.closes) == 1
    assert [r[1]["target"]["path"] for r in broker.requests] == ["/api/v0/models", "/v1/chat/completions", "/v1/chat/completions"]
    assert [r[1]["component"] for r in broker.requests] == ["answer.interpret", "answer.interpret", "answer.summarize"]


def test_a_parsed_question_with_a_summary_opens_its_run_at_the_summary_only():
    g, _ = world()
    broker = FakeBroker(["one item."])
    got = front_for(g, broker).answer("impact m.py:5", reader=FrontReader(g), summary=True)
    assert len(broker.opens) == 1 and [r[1]["component"] for r in broker.requests] == ["answer.summarize", "answer.summarize"]
    assert got.summary is not None and "provenance" not in got.doc["interpretation"]


def test_each_question_gets_its_own_run():
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    f = front_for(g, broker)
    f(Q, reader=FrontReader(g))
    f("who gets hit if m.py::A moves", reader=FrontReader(g))
    assert len(broker.opens) == 2 and len(broker.closes) == 2


# ------------------------------------------------------------------------------------------------------------------ (4), (f) provenance, hashes, sampling

def test_an_interpreted_answer_is_the_structured_answer_plus_its_provenance():
    g, _ = world()
    doc = ask(g, FakeBroker([draft_reply()]))
    want = structured(g)
    assert answer.facts_bytes(doc) == answer.facts_bytes(want)                    # provenance is outside the facts
    prov = doc["interpretation"]["provenance"]
    answer.validate(doc)
    assert prov == {"modelFingerprint": FINGERPRINT, "promptHash": sha("prompts/interpret.txt"), "grammarHash": sha("grammar/interpret.json"),
                    "samplingHash": answer.sampling_hash(0, {"max_tokens": 384, "seed": 0, "top_p_milli": 1000, "top_k": 0, "repeat_penalty_milli": 1000, "frequency_penalty_milli": 0, "presence_penalty_milli": 0}), "resolverVersion": RESOLVER_VERSION,
                    "endpoint": "local", "temperatureMilli": 0, "quantization": "Q4_K_M", "cacheHit": False,
                    "cacheKey": answer.cache_key(Q, SNAP, FINGERPRINT, sha("prompts/interpret.txt"), sha("grammar/interpret.json"),
                                                 answer.sampling_hash(0, {"max_tokens": 384, "seed": 0, "top_p_milli": 1000, "top_k": 0, "repeat_penalty_milli": 1000, "frequency_penalty_milli": 0, "presence_penalty_milli": 0}), RESOLVER_VERSION)}


def test_an_ambiguous_and_an_unusable_answer_carry_provenance_too():
    g, _ = world()
    amb = ask(g, FakeBroker([draft_reply(direction=None)]))
    assert amb["interpretation"]["status"] == "ambiguous" and amb["interpretation"]["provenance"]["cacheHit"] is False
    bad = ask(g, FakeBroker(["not json"]))
    assert bad["interpretation"]["provenance"]["modelFingerprint"] == FINGERPRINT


@pytest.mark.parametrize("name,value", [("temperature", 0.5), ("top_p", 0.9), ("top_k", 40), ("repeat_penalty", 1.1), ("frequency_penalty", 0.5), ("presence_penalty", 0.5), ("max_tokens", 512), ("seed", 7)])
def test_a_changed_sampling_parameter_changes_the_hash_the_body_and_the_cache_key(monkeypatch, name, value):
    g, _ = world()
    base = ask(g, FakeBroker([draft_reply()]))["interpretation"]["provenance"]
    broker = FakeBroker([draft_reply()])
    monkeypatch.setattr(I, "SAMPLING", {**I.SAMPLING, name: value})
    prov = ask(g, broker)["interpretation"]["provenance"]
    assert prov["samplingHash"] != base["samplingHash"] and prov["cacheKey"] != base["cacheKey"]
    assert broker.completions[0][2][name] == value                          # the hash describes what was sent
    assert prov["temperatureMilli"] == round(I.SAMPLING["temperature"] * 1000)


def test_the_sampling_hash_is_the_answer_modules_over_the_pinned_values():
    assert I.SAMPLING == {"temperature": 0, "top_p": 1, "top_k": 0, "repeat_penalty": 1.0, "frequency_penalty": 0, "presence_penalty": 0, "max_tokens": 384, "seed": 0}
    assert "min_p" not in I.SAMPLING                                                          # not in LM Studio's documented parameter list: not sent
    assert I.sampling_hash() == answer.sampling_hash(0, {"max_tokens": 384, "seed": 0, "top_p_milli": 1000, "top_k": 0, "repeat_penalty_milli": 1000, "frequency_penalty_milli": 0, "presence_penalty_milli": 0})


def test_the_request_body_sends_every_pinned_sampler_value_literally():
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    ask(g, broker)
    body = broker.completions[0][2]
    assert {k: body[k] for k in ("temperature", "top_p", "top_k", "repeat_penalty", "frequency_penalty", "presence_penalty", "max_tokens", "seed")} == \
        {"temperature": 0, "top_p": 1, "top_k": 0, "repeat_penalty": 1.0, "frequency_penalty": 0, "presence_penalty": 0, "max_tokens": 384, "seed": 0}
    assert body["repeat_penalty"] == 1.0 and isinstance(body["repeat_penalty"], float)


def test_the_fingerprint_is_read_from_the_endpoint_at_call_time():
    g, _ = world()
    other = {**MODEL, "quantization": "Q8_0", "loaded_context_length": 8192}
    prov = ask(g, FakeBroker([draft_reply()], models=[other]))["interpretation"]["provenance"]
    assert prov["modelFingerprint"] == "qwen/qwen3-4b-2507|qwen|qwen3|Q8_0|8192" and prov["quantization"] == "Q8_0"


def test_the_bound_model_is_the_recommended_one_when_loaded_else_the_first_loaded_chat_model():
    emb = {**MODEL, "id": "nomic-embed", "type": "embeddings"}
    idle = {**MODEL, "id": "aaa/idle", "state": "not-loaded"}
    zed = {**MODEL, "id": "zzz/other"}
    assert I.pick_model([emb, idle, zed, MODEL])["id"] == "qwen/qwen3-4b-2507"
    assert I.pick_model([emb, idle, zed])["id"] == "zzz/other"
    with pytest.raises(ModelError):
        I.pick_model([emb, idle])


def test_no_loaded_model_fails_the_question():
    g, _ = world()
    with pytest.raises(ModelError, match="no loaded chat model"):
        ask(g, FakeBroker([draft_reply()], models=[{**MODEL, "state": "not-loaded"}]))


# ------------------------------------------------------------------------------------------------------------------ (e) the cache

def test_a_repeat_is_served_from_the_cache_and_records_a_hit():
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    f = front_for(g, broker)
    first = f(Q, reader=FrontReader(g))["interpretation"]["provenance"]
    second_doc = f(Q, reader=FrontReader(g))
    second = second_doc["interpretation"]["provenance"]
    assert len(broker.completions) == 1
    assert (first["cacheHit"], second["cacheHit"]) == (False, True) and {**first, "cacheHit": True} == second
    assert second_doc["interpretation"]["status"] == "resolved"


def test_a_cache_entry_lives_exactly_as_long_as_its_snapshot():
    g, _ = world()
    other = [("6" * 32, 0, answer.EMPTY_VERDICT_LOG_HASH, "1.1", "1.0")]
    broker = FakeBroker([draft_reply()])
    f = front_for(g, broker)
    f(Q, reader=FrontReader(g))
    f(Q, reader=FrontReader(g, other))                                      # a new snapshot: a miss
    assert len(broker.completions) == 2
    f(Q, reader=FrontReader(g))                                             # the old snapshot's entry is gone
    assert len(broker.completions) == 3
    assert f(Q, reader=FrontReader(g))["interpretation"]["provenance"]["cacheHit"] is True


def test_the_cache_holds_one_snapshot_at_a_time():
    c = InterpretCache()
    c.put("k", "s1", {"a": 1})
    assert c.get("k", "s1") == {"a": 1} and len(c) == 1
    assert c.get("k", "s2") is None and len(c) == 0
    assert c.get("k", "s1") is None


def test_the_cache_is_capped_and_the_oldest_entry_goes_first():
    from ccg.front import cache as C
    c = InterpretCache(max_entries=3)
    for i in range(3):
        c.put(f"k{i}", "s", {"i": i})
    c.put("k0", "s", {"i": 0})                                    # a re-put is the newest again
    c.put("k3", "s", {"i": 3})
    assert len(c) == 3 and c.get("k1", "s") is None and all(c.get(k, "s") is not None for k in ("k0", "k2", "k3"))
    with pytest.raises(ValueError):
        InterpretCache(max_entries=0)
    assert C.MAX_ENTRIES == 1024 and I.Interpreter().cache._max == 1024
    big = InterpretCache()
    for i in range(1025):
        big.put(f"k{i}", "s", {})
    assert len(big) == 1024 and big.get("k0", "s") is None and big.get("k1", "s") is not None


def test_a_cached_reply_is_copied_both_ways():
    c = InterpretCache()
    reply = {"a": [1]}
    c.put("k", "s", reply)
    reply["a"].append(2)
    c.get("k", "s")["a"].append(3)
    assert c.get("k", "s") == {"a": [1]}


BASE = dict(normalized_question="q", snapshot_id="1" * 32, model_fingerprint="m", prompt_hash="p", grammar_hash="g", sampling_hash="s", resolver_version="1.0")


@pytest.mark.parametrize("field", list(BASE))
def test_every_input_of_the_cache_key_changes_it(field):
    assert key_of(**{**BASE, field: BASE[field] + "x"}) != key_of(**BASE)
    assert key_of(**BASE) == answer.cache_key(*BASE.values())


def test_the_interpreters_key_follows_the_snapshot_the_prompt_and_the_model(monkeypatch):
    g, _ = world()
    base = ask(g, FakeBroker([draft_reply()]))["interpretation"]["provenance"]["cacheKey"]
    other = [("6" * 32, 0, answer.EMPTY_VERDICT_LOG_HASH, "1.1", "1.0")]
    assert ask(g, FakeBroker([draft_reply()]), reader=FrontReader(g, other))["interpretation"]["provenance"]["cacheKey"] != base
    assert ask(g, FakeBroker([draft_reply()], models=[{**MODEL, "arch": "x"}]))["interpretation"]["provenance"]["cacheKey"] != base
    monkeypatch.setattr(I, "prompt_hash", lambda: "e" * 64)
    assert ask(g, FakeBroker([draft_reply()]))["interpretation"]["provenance"]["cacheKey"] != base


def test_the_bypass_neither_reads_nor_writes_the_cache_and_calls_the_interpreter():
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    f = front_for(g, broker)
    miss = f(Q, reader=FrontReader(g))                                       # warm the cache
    key = miss["interpretation"]["provenance"]["cacheKey"]
    bypassed = f(Q, reader=FrontReader(g), cache_bypass=True)["interpretation"]["provenance"]     # no read: the interpreter is called
    assert len(broker.completions) == 2
    assert bypassed["cacheHit"] is False and bypassed["cacheBypass"] is True and bypassed["cacheKey"] == key
    assert "cacheBypass" not in miss["interpretation"]["provenance"]


def test_a_bypassed_ask_writes_nothing():
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    f = front_for(g, broker)
    f(Q, reader=FrontReader(g), cache_bypass=True)
    assert len(broker.completions) == 1
    after = f(Q, reader=FrontReader(g))["interpretation"]["provenance"]
    assert len(broker.completions) == 2 and after["cacheHit"] is False        # nothing was written by the bypass


def test_a_bypass_does_not_disturb_an_existing_entry():
    g, _ = world()
    broker = FakeBroker([draft_reply()])
    f = front_for(g, broker)
    f(Q, reader=FrontReader(g))
    f(Q, reader=FrontReader(g), cache_bypass=True)
    assert f(Q, reader=FrontReader(g))["interpretation"]["provenance"]["cacheHit"] is True and len(broker.completions) == 2


def test_the_cache_hit_still_passes_the_fabrication_guard():
    g, _ = world()
    f_broker = FakeBroker([draft_reply(groups=(("nowhere/zzz.py", "file"), ("Zed", "symbol")))])
    f = front_for(g, f_broker)
    first, second = (f(Q, reader=FrontReader(g)) for _ in range(2))
    assert first["interpretation"]["status"] == second["interpretation"]["status"] == "ambiguous"
    assert second["interpretation"]["provenance"]["cacheHit"] is True and len(f_broker.completions) == 1


# ------------------------------------------------------------------------------------------------------------------ (b) the reply: invalid is unresolved, never repaired

INVALID = [
    "not json at all",
    "",
    "[]",
    "NaN",
    json.dumps({**draft_reply(), "extra": 1}),
    json.dumps({**draft_reply(), "useCase": "rca"}),
    json.dumps({**draft_reply(), "direction": "both"}),
    json.dumps({k: v for k, v in draft_reply().items() if k != "readings"}),
    json.dumps({**draft_reply(), "anchors": []}),
    json.dumps({**draft_reply(), "anchors": [{"spans": []}]}),
    json.dumps({**draft_reply(), "anchors": [{"spans": [{"text": "A", "kindHint": "id"}]}]}),
    json.dumps({**draft_reply(), "anchors": [{"spans": [{"text": "", "kindHint": "symbol"}]}]}),
    json.dumps({**draft_reply(), "readings": [{"useCase": "impact", "direction": "impact"}] * 4}),
]


@pytest.mark.parametrize("content", INVALID)
def test_a_reply_that_is_not_valid_against_the_grammar_is_unresolved_with_one_call_and_no_repair(content):
    g, _ = world()
    broker = FakeBroker([content, draft_reply()])                           # a second call would be answered validly
    doc = ask(g, broker)
    assert (doc["interpretation"]["status"], doc["interpretation"]["ambiguityKind"], doc["interpretation"]["options"]) == ("ambiguous", "no-anchor", [])
    assert len(broker.completions) == 1 and len(broker.opens) == 1
    assert doc["items"] == [] and doc["interpretation"]["provenance"]["cacheHit"] is False


def test_an_unusable_reply_is_not_cached():
    g, _ = world()
    broker = FakeBroker(["nope", draft_reply()])
    f = front_for(g, broker)
    assert f(Q, reader=FrontReader(g))["interpretation"]["status"] == "ambiguous"
    assert f(Q, reader=FrontReader(g))["interpretation"]["status"] == "resolved" and len(broker.completions) == 2


def test_a_reply_that_validates_but_fails_the_draft_rules_is_unusable_not_an_error():
    assert I.to_draft({**draft_reply(), "anchors": [{"spans": [{"text": "x", "kindHint": "symbol"}]}] * 4}) is None
    assert I.to_draft(draft_reply()) is not None


def test_the_fabrication_guard_runs_on_the_interpreters_draft():
    g, _ = world()
    reader = FrontReader(g)
    doc = ask(g, FakeBroker([draft_reply(groups=(("nowhere/zzz.py", "file"), ("Zed", "symbol")))]), reader=reader)
    assert doc["interpretation"]["status"] == "ambiguous" and doc["interpretation"]["ambiguityKind"] == "no-anchor"
    assert not [c for c in reader.calls if "zzz" in json.dumps(c, default=str) or "Zed" in json.dumps(c, default=str)]      # nothing invented reached the store


def test_a_part_fabricated_group_resolves_on_what_the_question_does_say():
    g, _ = world()
    doc = ask(g, FakeBroker([draft_reply(groups=(("m.py", "file"), ("A", "symbol"), ("Invented", "symbol")))]))
    assert doc["interpretation"]["status"] == "resolved"


@pytest.mark.parametrize("extra", [("foo", "other"), ("hello", "server")])
def test_a_spurious_extra_group_does_not_turn_a_resolvable_question_into_an_ask(extra):
    g, _ = world()
    q = "if I touch m.py::A who gets hit, foo hello"
    doc = ask(g, FakeBroker([draft_reply(groups=((("m.py", "file"), ("A", "symbol")), (extra,)))]), q)
    assert doc["interpretation"]["status"] == "resolved"


# ------------------------------------------------------------------------------------------------------------------ (i), (3) what the draft says

def test_a_null_direction_is_both_directions_as_options_never_one_pick():
    g, _ = world()
    doc = ask(g, FakeBroker([draft_reply(direction=None)]))
    i = doc["interpretation"]
    assert (i["status"], i["ambiguityKind"], [o["direction"] for o in i["options"]]) == ("ambiguous", "two-use-cases", ["impact", "depends-on"])


def test_an_unbuilt_use_case_is_understood_and_not_built_with_the_request_echoed():
    g, _ = world()
    for use_case in ("root-cause", "code-review", "test-coverage", "security"):
        with pytest.raises(NotBuiltAsk) as e:
            ask(g, FakeBroker([draft_reply(use_case=use_case, direction=None)]), "run a security pass on m.py::A")
        assert e.value.detail["request"]["useCase"] == use_case and "not built in this phase" in str(e.value)
        assert e.value.detail["request"]["anchors"] == [[{"text": "m.py", "kindHint": "file"}, {"text": "A", "kindHint": "symbol"}]]


def test_an_unbuilt_use_case_in_readings_refuses_the_whole_ask_and_never_answers_its_built_half_alone():
    g, _ = world()
    q = "do a code review of m.py::A and tell who gets hit"
    for primary_use_case, readings in (("impact", [("code-review", None)]), ("security", [("impact", "impact")]), ("impact", [("impact", "depends-on"), ("root-cause", None)])):
        broker = FakeBroker([draft_reply(use_case=primary_use_case, readings=readings)])
        with pytest.raises(NotBuiltAsk) as e:
            ask(g, broker, q)
        assert "not built in this phase" in str(e.value) and e.value.detail["built"] == ["impact"]
        assert e.value.detail["request"]["readings"] == [{"useCase": u, "direction": d} for u, d in readings]
        assert len(broker.opens) == len(broker.closes) == 1


def test_a_scope_naming_question_is_not_built():
    g, _ = world()
    with pytest.raises(NotBuiltAsk) as e:
        ask(g, FakeBroker([draft_reply(scope="svc")]), "in repo svc if I touch m.py::A who gets hit")
    assert e.value.detail["request"]["scope"] == "svc"


def test_a_scope_the_question_never_said_is_dropped_as_fabricated():
    g, _ = world()
    assert ask(g, FakeBroker([draft_reply(scope="svc")]))["interpretation"]["status"] == "resolved"


def dev_front(broker):
    g, _ = world()
    stub = structured(g)

    def build(anchor, direction):
        return stub
    return Front(("graphmind-ai-lab",), build, interpreter=I.Interpreter(), run_factory=I.run_factory(SYSTEM, ("graphmind-ai-lab",), broker))


@pytest.mark.parametrize("qid,groups", [("QW-3", (("cypher-mcp/server.py", "file"), ("line 564", "line"))),
                                        ("QW-11", (("POST /threads/{thread_id}/messages", "route"),))])
def test_qw3_and_qw11_resolve_when_the_interpreter_states_the_direction_they_imply(qid, groups):
    text = dev(qid)["text"]
    broker = FakeBroker([draft_reply("impact", groups)])
    got = dev_front(broker)(text, reader=dev_store())
    assert got["interpretation"]["status"] == "resolved" and len(broker.completions) == 1
    assert broker.completions[0][2]["messages"][1]["content"] == answer.normalize_question(text)


@pytest.mark.parametrize("qid,groups", [("QW-3", (("cypher-mcp/server.py", "file"), ("line 564", "line"))),
                                        ("QW-11", (("POST /threads/{thread_id}/messages", "route"),))])
def test_a_null_for_qw3_or_qw11_is_visible_as_over_asking(qid, groups):
    """The miss the dev entry check counts: a ``null`` on a question that states a direction comes back as the two-direction ask, not as a silent pick."""
    got = dev_front(FakeBroker([draft_reply(None, groups)]))(dev(qid)["text"], reader=dev_store())
    assert got["interpretation"]["status"] == "ambiguous" and [o["direction"] for o in got["interpretation"]["options"]] == ["impact", "depends-on"]


# ------------------------------------------------------------------------------------------------------------------ (h) a missing or latched broker

def test_an_absent_broker_fails_the_call_and_no_other_connection_is_attempted(monkeypatch, tmp_path):
    g, _ = world()
    sock_path = str(tmp_path / "none.sock")
    attempted = []
    real = socket.socket

    class Recording(real):
        def connect(self, address):
            attempted.append((self.family, address))
            return super().connect(address)

    monkeypatch.setattr(socket, "socket", Recording)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: attempted.append(("create_connection", a)) or (_ for _ in ()).throw(AssertionError("direct")))
    f = Front((g.repo,), lambda a, d: structured(g), interpreter=I.Interpreter(), run_factory=I.run_factory(SYSTEM, (g.repo,), BrokerClient(sock_path)))
    with pytest.raises(ModelError, match="not reachable"):
        f(Q, reader=FrontReader(g))
    assert attempted == [(socket.AF_UNIX, sock_path)]


def test_an_unreachable_broker_fails_at_the_run_and_a_latched_one_at_the_run_or_the_call():
    g, _ = world()
    with pytest.raises(ModelError, match="not reachable"):
        ask(g, FakeBroker(unavailable=True))
    with pytest.raises(ModelError, match="audit-unwritable"):
        ask(g, FakeBroker(open_error="audit-unwritable"))


def test_a_broker_that_latches_mid_run_fails_the_call_without_a_second_run():
    g, _ = world()

    class Latching(FakeBroker):
        def request(self, token, request, body=None):
            from ccg.broker.client import BrokerError
            raise BrokerError("audit-unwritable")
    b = Latching([draft_reply()])
    with pytest.raises(ModelError, match="audit-unwritable"):
        ask(g, b)
    assert len(b.opens) == 1 and b.closes == ["f" * 64]


def test_an_egress_denial_fails_the_call_with_the_rule_and_is_not_retried():
    g, _ = world()
    b = FakeBroker([draft_reply()], allow=False)
    with pytest.raises(ModelError, match="refused by egress policy"):
        ask(g, b)
    assert len(b.requests) == 1


def test_no_run_means_no_system_or_no_repositories():
    g, _ = world()
    with pytest.raises(ModelError, match="no model run"):
        Front((g.repo,), lambda a, d: structured(g), interpreter=I.Interpreter(),
              run_factory=I.run_factory(None, (g.repo,), FakeBroker([draft_reply()])))(Q, reader=FrontReader(g))


# ------------------------------------------------------------------------------------------------------------------ transport retry

def test_a_transport_failure_is_retried_once_and_logged(caplog):
    g, _ = world()
    b = FakeBroker([draft_reply()], statuses=[None])
    with caplog.at_level(logging.WARNING, logger="ccg.front.model"):
        doc = ask(g, b)
    assert doc["interpretation"]["status"] == "resolved" and len(b.requests) == 3        # GET twice, POST once
    assert [r.message for r in caplog.records if "transport failure" in r.message] and len(caplog.records) == 1


def test_a_server_error_is_retried_once_and_a_second_failure_fails_the_call():
    g, _ = world()
    assert ask(g, FakeBroker([draft_reply()], statuses=[503]))["interpretation"]["status"] == "resolved"
    b = FakeBroker([draft_reply()], statuses=[None, 502])
    with pytest.raises(ModelError, match="did not answer"):
        ask(g, b)
    assert len(b.requests) == 2


def test_a_client_error_is_not_retried():
    g, _ = world()
    b = FakeBroker([draft_reply()], statuses=[400])
    with pytest.raises(ModelError, match="HTTP 400"):
        ask(g, b)
    assert len(b.requests) == 1


def test_an_unusable_reply_is_never_retried_whatever_the_transport():
    g, _ = world()
    b = FakeBroker(["garbage"])
    ask(g, b)
    assert len(b.completions) == 1
