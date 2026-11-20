"""P1-16 pack runtime over **unresolved arguments** (RCA ``docs/reviews/connected-code-graph-p1-41-rca.md`` problem (a)).

A per-site condition (an argument whose value is ``unknown``) must degrade to a diagnostic or a keyless ``possible`` fact
(rule-pack spec §1.4), never abort the refresh. The guard drives **every** pack of ``ccg.refresh.DEFAULT_PACKS`` over a
bundle synthesized from the pack's own rules (one site per callsite/decorator rule, scope import satisfied, every argument
path the rule names set to ``unknown``), so a new pack or ``onUnknown`` shape is covered without a hand-listed case.
"""
import re
import re._parser as sre

import pytest

from packrt_helpers import call, dec, file_, imp, pack, sym, vs
from ccg.packrt import run_packs
from ccg.refresh import DEFAULT_PACKS

UNKNOWN_EL = {"unknown": "point-derived"}


def sample(matcher) -> str:
    """One string the matcher accepts (exact / prefix / regex; a bare scalar is exact)."""
    if not isinstance(matcher, dict):
        return str(matcher)
    if isinstance(matcher, dict) and "ignoreCase" in matcher:
        matcher = {k: v for k, v in matcher.items() if k != "ignoreCase"}
    ((kind, v),) = matcher.items()
    if kind in ("exact", "prefix", "basename"):
        return v
    if kind == "regex":
        return _sample_re(sre.parse(v))
    raise AssertionError(f"guard cannot sample matcher {kind!r}")


def _sample_re(parsed) -> str:
    out = ""
    for op, arg in parsed:
        name = str(op)
        if name == "LITERAL":
            out += chr(arg)
        elif name == "IN":
            first = arg[0]
            if str(first[0]) == "RANGE":
                out += chr(first[1][0])
            elif str(first[0]) == "LITERAL":
                out += chr(first[1])
            else:
                raise AssertionError(f"class {arg}")
        elif name == "SUBPATTERN":
            out += _sample_re(arg[3])
        elif name == "BRANCH":
            out += _sample_re(arg[1][0])
        elif name in ("MAX_REPEAT", "MIN_REPEAT"):
            lo, _hi, body = arg
            out += _sample_re(body) * max(lo, 0)
        elif name == "ANY":
            out += "x"
        else:
            raise AssertionError(f"regex op {name}")
    return out


def arg_paths(expr: dict) -> set[str]:
    out = {expr["arg"]} if "arg" in expr else set()
    for e in expr.get("firstOf", ()):
        out |= arg_paths(e)
    return out


def site_records(rule: dict) -> list[dict]:
    """A file importing every scope gate, an owner symbol, and one matching site with all-unknown arguments."""
    ((subject, clause),) = [(k, v) for k, v in rule["match"].items()]
    recs = [file_("f1", "svc/a.py"), sym("s", "owner", fid="f1", path="svc/a.py")]
    for i, gate in enumerate((rule.get("scope") or {}).get("imports", ())):
        recs.append(imp("f1", sample(gate["module"])) | {"id": f"imp{i}"})
    paths = {a["arg"] for a in clause.get("args", ())}
    for expr in rule.get("extract", {}).values():
        paths |= arg_paths(expr)
    paths |= {"a0", "a1"}
    callee_or_name = clause.get("callee", clause.get("name"))
    name = sample(callee_or_name) if callee_or_name is not None else "site"
    if subject == "callsite":
        recs.append(call("c1", "s", name, kind=clause.get("callKind", "call")))
        recs += [vs("c1", p, UNKNOWN_EL) for p in sorted(paths)]
    else:
        recs.append(sym("t", "target", fid="f1", path="svc/a.py", parent="s"))
        d = dec("d1", "t", name, *[(p, UNKNOWN_EL) for p in sorted(paths)])
        d["form"] = clause.get("form", "decorator") if isinstance(clause.get("form"), str) else "decorator"
        recs.append(d)
    return recs


def _site_rules(p: dict):
    for rule in p.get("rules", ()):
        m = rule["match"]
        if "anyOf" in m:
            continue
        ((subject, _),) = m.items()
        if subject in ("callsite", "decorator"):
            yield rule


@pytest.mark.parametrize("name", DEFAULT_PACKS)
def test_every_default_pack_degrades_an_all_unknown_site_instead_of_raising(name):
    p = pack(name)
    for rule in _site_rules(p):
        run_packs([site_records(rule)], [p])         # must not raise UnsupportedSubject (RCA (a))


def test_the_guard_reaches_the_kafka_rules_it_exists_for():
    p = pack("kafka")
    got = {r["id"] for r in _site_rules(p)}
    assert {"produce", "consume-positional"} <= got
    # non-vacuity: the all-unknown site is really matched (a fact or a diagnostic comes out of it)
    for rule in _site_rules(p):
        f = run_packs([site_records(rule)], [p])
        assert f.providers or f.consumers or f.diagnostics, rule["id"]


# ---- the degraded fact

def field_read(point, path="a0", field="orders_topic"):
    return {"type": "fieldAccess", "id": f"fa-{point}-{path}", "pointId": point, "path": path, "mode": "read",
            "receiverType": "?", "field": field}


def kafka_send(*, with_field_read=False, arg=UNKNOWN_EL):
    recs = [file_("f1", "svc/a.py"), imp("f1", "kafka", "KafkaProducer"), sym("s", "pub", fid="f1", path="svc/a.py"),
            call("c1", "s", "send"), vs("c1", "a0", arg)]
    return recs + ([field_read("c1")] if with_field_read else [])


KAFKA = pack("kafka")


def test_an_unresolved_topic_publisher_is_a_keyless_possible_runtime_built_provider():
    (f,) = run_packs([kafka_send()], [KAFKA]).providers
    assert (f["kind"], f["key"], f["status"], f["reason"]) == ("topic", None, "possible", "runtime-built")


def test_an_unresolved_topic_read_from_a_field_is_config_derived():
    (f,) = run_packs([kafka_send(with_field_read=True)], [KAFKA]).providers
    assert (f["key"], f["status"], f["reason"]) == (None, "possible", "config-derived")


def test_a_field_read_on_another_argument_does_not_make_it_config_derived():
    recs = kafka_send() + [field_read("c1", path="a1")]
    (f,) = run_packs([recs], [KAFKA]).providers
    assert f["reason"] == "runtime-built"


def test_a_resolved_topic_is_still_confirmed_with_its_key():
    (f,) = run_packs([kafka_send(arg={"const": "orders.created"})], [KAFKA]).providers
    assert (f["key"], f["status"], f["reason"]) == ("kafka:orders.created", "confirmed", "key-match")


def test_an_unresolved_consumer_topic_read_from_a_field_is_config_derived():
    recs = [file_("f1", "svc/a.py"), imp("f1", "kafka", "KafkaConsumer"), sym("s", "sub", fid="f1", path="svc/a.py"),
            call("c1", "s", "KafkaConsumer"), vs("c1", "a0", UNKNOWN_EL), field_read("c1")]
    (f,) = run_packs([recs], [KAFKA]).consumers
    assert (f["key"], f["status"], f["reason"]) == (None, "possible", "config-derived")


def test_a_field_read_does_not_turn_a_runtime_built_slot_into_config_derived():
    import copy
    p = copy.deepcopy(KAFKA)
    (produce,) = [r for r in p["rules"] if r["id"] == "produce"]
    produce["extract"]["topic"]["onUnknown"] = "runtime-built"
    (f,) = run_packs([kafka_send(with_field_read=True)], [p]).providers
    assert (f["key"], f["reason"]) == (None, "runtime-built")
