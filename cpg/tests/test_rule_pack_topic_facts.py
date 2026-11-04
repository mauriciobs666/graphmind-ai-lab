"""S32: the pack-level emissions of ``kafka`` (spec section 5) and ``paper-inhouse-bus`` (section 6) over the spec's own invented records.

Scope (spec 12.1, S32): the **pack's** emission only -- edge, kind, key text, ``keyForm`` and the reason the key slot's ``onUnknown``
implies. Nothing here asserts a graph element: a ``ContractPoint`` for a pattern key or a keyless ``CONSUMES``, a ``Link``, or a
``Finding`` have no home in graph model D4 until O-15 is ruled, so the literal-key facts (5.2 step 1, 6.2 step 1) are the only
ones read as graph facts; the other emissions are asserted as the rule's output, nothing more.

The evaluator is the smallest one that reads these rules (callsite subject, ``scope.imports``, ``arg``/``yield`` slots, a
``query`` subject through ``test_rule_pack_spec.eval_query``). It exists so the rules' literals -- callee regexes, import gates,
key templates, ``keyForm`` -- are read by something, which is what lets the leaf-mutation harness reject an edit to them.
"""

from __future__ import annotations

import pytest

import test_rule_pack_spec as spec

KAFKA, BUS = spec.PACKS["kafka"], spec.PACKS["paper-inhouse-bus"]


def rule(pack, rid):
    return next(r for r in pack["rules"] if r["id"] == rid)


def const(*values):
    return [{"const": v} for v in values]


def unknown():
    return [{"unknown": "point-derived"}]


def call(callee, imports, **args):
    """A callsite record plus its file's imports; ``args`` maps an argument path (``a0``, ``kw__pattern``) to value alternatives."""
    return {"callee": callee, "imports": imports, "args": {k.replace("__", "."): v for k, v in args.items()}}


def resolved(values, constants=None):
    out = []
    for v in values:
        if "const" in v:
            out.append(v["const"])
        elif "ref" in v:
            out.append((constants or {})[v["ref"]])
    return out


def on_unknown(expr, edge, kind):
    return expr.get("onUnknown", "config-derived" if (edge, kind) == ("provides", "topic") else "partial")


def emit(r, slots, mode_of):
    """The rule's emissions for resolved ``slots`` (name -> value list); an unresolved key slot gives a keyless fact."""
    out = []
    for em in r["emit"]:
        ((name, e),) = em.items()
        refs = [s for s in slots if "{" + s + "}" in e["key"]]
        if any(not slots[s] for s in refs):
            out.append((name, e["kind"], None, e.get("keyForm"), mode_of(refs[0], name, e["kind"])))
            continue
        out.append((name, e["kind"], e["key"].format(**{s: slots[s][0] for s in refs}), e.get("keyForm"), None))
    return out


def run_callsite(pack, rid, c, constants=None):
    r = rule(pack, rid)
    clause = r["match"]["callsite"]
    if not spec.matches(clause["callee"], c["callee"]):
        return []
    if any(a["arg"] not in c["args"] for a in clause.get("args", [])):
        return []
    if not any(spec.matches(i["module"], m) for i in r["scope"]["imports"] for m in c["imports"]):
        return []
    slots = {s: resolved(c["args"].get(x["arg"], []), constants) for s, x in r["extract"].items()}
    return emit(r, slots, lambda s, edge, kind: on_unknown(r["extract"][s], edge, kind))


# ---------------------------------------------------------------- kafka (5.2)

def test_a_producer_call_provides_the_literal_topic_key():
    """5.2 step 1: callsite P, `send("orders.created", ...)` in a file importing kafka."""
    p = call("send", ["kafka"], a0=const("orders.created"), kw__value=unknown())
    assert run_callsite(KAFKA, "produce", p) == [("provides", "topic", "kafka:orders.created", None, None)]


@pytest.mark.parametrize("callee", ["send", "send_and_wait", "produce"])
@pytest.mark.parametrize("module", ["kafka", "aiokafka", "confluent_kafka", "kafka.producer", "aiokafka.producer"])
def test_every_producer_callee_and_import_gate_provides(callee, module):
    got = run_callsite(KAFKA, "produce", call(callee, [module], a0=const("t")))
    assert got == [("provides", "topic", "kafka:t", None, None)]


@pytest.mark.parametrize("callee", ["sender", "sends", "send_all", "flush", "subscribe", "Producer"])
def test_a_callee_outside_the_producer_regex_provides_nothing(callee):
    assert run_callsite(KAFKA, "produce", call(callee, ["kafka"], a0=const("t"))) == []


@pytest.mark.parametrize("module", ["socket", "kafkaesque", "my_kafka", "confluent", "acmebus"])
def test_a_producer_in_a_file_without_a_kafka_import_provides_nothing(module):
    assert run_callsite(KAFKA, "produce", call("send", [module], a0=const("t"))) == []


def test_an_unresolved_producer_topic_is_a_keyless_provides_that_is_config_derived():
    """5.2 / 1.4 (O-16): the publisher's key slot degrades to `config-derived`; the pack states the reason, not a graph fact."""
    got = run_callsite(KAFKA, "produce", call("send", ["kafka"], a0=unknown()))
    assert got == [("provides", "topic", None, None, "config-derived")]


@pytest.mark.parametrize("callee", ["KafkaConsumer", "AIOKafkaConsumer"])
@pytest.mark.parametrize("module", ["kafka", "aiokafka", "kafka.consumer", "aiokafka.consumer"])
def test_a_consumer_constructor_with_a_literal_topic_consumes_its_key(callee, module):
    got = run_callsite(KAFKA, "consume-positional", call(callee, [module], a0=const("orders")))
    assert got == [("consumes", "topic", "kafka:orders", None, None)]


def test_a_consumer_whose_topic_is_read_from_config_consumes_with_no_key():
    """5.2 step 2: `KafkaConsumer(settings.orders_topic)` -- a0 has no resolved alternative, so the fact is keyless and config-derived."""
    got = run_callsite(KAFKA, "consume-positional", call("KafkaConsumer", ["kafka"], a0=unknown()))
    assert got == [("consumes", "topic", None, None, "config-derived")]


@pytest.mark.parametrize("callee,modules", [
    ("KafkaConsumerFactory", ["kafka"]), ("Consumer", ["kafka"]), ("MyKafkaConsumer", ["kafka"]),
    ("KafkaConsumer", ["confluent_kafka"]),  # the positional rule's gate is kafka and aiokafka only
    ("KafkaConsumer", ["socket"]), ("KafkaProducer", ["kafka"]),
])
def test_a_consumer_outside_the_positional_rule_consumes_nothing(callee, modules):
    assert run_callsite(KAFKA, "consume-positional", call(callee, modules, a0=const("orders"))) == []


def test_a_subscribe_with_a_pattern_keyword_consumes_a_regex_key():
    """5.2 step 3: the pattern `^orders\\..*` of the snippet is the key text after `kafka:`, `keyForm regex` (pack output only)."""
    got = run_callsite(KAFKA, "consume-pattern", call("subscribe", ["kafka"], kw__pattern=const(r"^orders\..*")))
    assert got == [("consumes", "topic", r"kafka:^orders\..*", "regex", None)]


@pytest.mark.parametrize("c", [
    call("subscribe", ["kafka"], a0=const("orders")),                       # no kw.pattern
    call("subscribe2", ["kafka"], kw__pattern=const("x")),
    call("unsubscribe", ["kafka"], kw__pattern=const("x")),
    call("subscribe", ["confluent_kafka"], kw__pattern=const("x")),         # gate: kafka and aiokafka
    call("subscribe", ["socket"], kw__pattern=const("x")),
])
def test_a_subscribe_outside_the_pattern_rule_consumes_nothing(c):
    assert run_callsite(KAFKA, "consume-pattern", c) == []


def test_the_pattern_rule_reads_the_pattern_keyword_and_binds_the_slot_it_keys_on():
    r = rule(KAFKA, "consume-pattern")
    assert r["extract"] == {"pattern": {"arg": "kw.pattern"}}
    assert r["match"]["callsite"]["args"] == [{"arg": "kw.pattern"}]
    got = run_callsite(KAFKA, "consume-pattern", call("subscribe", ["aiokafka"], kw__pattern=const("a"), kw__other=const("b")))
    assert got == [("consumes", "topic", "kafka:a", "regex", None)]


def test_the_topic_vocab_entry_example_is_the_producer_key_and_keynorm_states_the_form():
    v = KAFKA["vocab"][0]
    key = rule(KAFKA, "produce")["emit"][0]["provides"]["key"]
    assert v["example"] == key.format(topic="orders.created")
    assert v["keyNorm"] == "kafka:<name>; name as written, case kept (spec §6, topic row)"
    assert v["keyNorm"].startswith(v["transport"] + ":")


# ---------------------------------------------------------------- paper-inhouse-bus (6.2)

CONSTANTS = {"billing/events.py::TOPIC_PAID": "billing.invoice.paid"}
BUS_RECORDS = [
    {"type": "symbol", "id": "T", "name": "InvoiceHandlers", "kind": "type"},
    {"type": "symbol", "id": "H", "name": "handle", "kind": "method", "parentId": "T"},
    {"type": "decorator", "id": "D1", "name": "on", "targetId": "H", "args": [{"arg": "a0", "values": [{"const": "invoice.*"}]}]},
    {"type": "decorator", "id": "D2", "name": "component", "targetId": "T", "args": [{"arg": "a0", "values": [{"const": "billing"}]}]},
]


def bus_rows(records=BUS_RECORDS):
    """The `handler-ns` query's yielded columns, one dict per row (`$alias.args.a0` is that argument's first const)."""
    q = BUS["queries"][0]
    out = []
    for env in spec.eval_query(q, records):
        row = {}
        for col, ref in q["yield"].items():
            alias, fld = ref[1:].split(".", 1)
            rec = env[alias]
            if fld.startswith("args."):
                arg = next((a for a in rec.get("args", []) if a["arg"] == fld[len("args."):]), None)
                row[col] = arg["values"][0]["const"] if arg else None
            else:
                row[col] = rec[fld]
        out.append(row)
    return out


def test_a_publish_through_a_constant_provides_the_literal_key_with_the_pack_transport():
    """6.2 step 1: `bus.publish(TOPIC_PAID, inv)` in a file importing acmebus; a0 is a `ref` resolved to the constant's value."""
    c = call("publish", ["acmebus"], a0=[{"ref": "billing/events.py::TOPIC_PAID"}])
    assert run_callsite(BUS, "publish", c, CONSTANTS) == [("provides", "topic", "inhouse-paper-inhouse-bus:billing.invoice.paid", None, None)]


@pytest.mark.parametrize("c", [
    call("publish", ["acmebus.sub"], a0=const("t")),      # exact module gate: a submodule is not acmebus
    call("publish", ["acmebusx"], a0=const("t")),
    call("publish", ["kafka"], a0=const("t")),
    call("publishes", ["acmebus"], a0=const("t")),
    call("send", ["acmebus"], a0=const("t")),
])
def test_a_publish_outside_the_bus_gate_or_callee_provides_nothing(c):
    assert run_callsite(BUS, "publish", c) == []


def test_the_handler_query_yields_the_handler_the_topic_and_the_class_namespace():
    assert bus_rows() == [{"handler": "H", "topic": "invoice.*", "ns": "billing"}]


def test_the_subscribe_rule_composes_the_namespace_and_the_topic_into_a_glob_key():
    """6.2 step 3: `inhouse-paper-inhouse-bus:billing.invoice.*`, `keyForm glob`, from the yielded columns."""
    r = rule(BUS, "subscribe")
    assert r["match"] == {"query": "handler-ns"} and r["extract"] == {"ns": {"yield": "ns"}, "topic": {"yield": "topic"}}
    (row,) = bus_rows()
    slots = {s: [row[x["yield"]]] for s, x in r["extract"].items()}
    assert emit(r, slots, lambda *a: None) == [("consumes", "topic", "inhouse-paper-inhouse-bus:billing.invoice.*", "glob", None)]


def test_the_subscribe_rule_is_gated_on_the_exact_acmebus_import():
    r = rule(BUS, "subscribe")
    mods = [i["module"] for i in r["scope"]["imports"]]
    assert spec.matches(mods, "acmebus") and not spec.matches(mods, "acmebus.sub") and not spec.matches(mods, "kafka")


def test_the_query_binds_on_and_component_decorators_by_name_only():
    q = BUS["queries"][0]
    dec_name, ns_name = q["from"][0]["where"]["name"], q["from"][3]["where"]["name"]
    for ok in ("on", "bus.on", "b_2.on"):
        assert spec.matches(dec_name, ok)
    for no in ("component", "bus.component", "one", "bus.onx", "bus.on.x"):
        assert not spec.matches(dec_name, no)
    for ok in ("component", "bus.component"):
        assert spec.matches(ns_name, ok)
    for no in ("on", "components", "bus.on"):
        assert not spec.matches(ns_name, no)


def test_the_topic_vocab_entry_example_is_the_publish_key_and_keynorm_states_the_form():
    v = BUS["vocab"][0]
    key = rule(BUS, "publish")["emit"][0]["provides"]["key"]
    assert v["example"] == key.format(topic="billing.invoice.paid")
    assert v["keyNorm"] == "inhouse-paper-inhouse-bus:<name>; name as written, case kept (spec §6, topic row)"
    assert v["keyNorm"].startswith(v["transport"] + ":")
