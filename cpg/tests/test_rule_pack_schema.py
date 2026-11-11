"""S31: ``rule-pack-v1.schema.json``, its validator, the reason registry and the three real packs.

Contract: ``docs/plans/connected-code-graph-rule-pack-spec.md`` (v1.4). The spec test
(``test_rule_pack_spec.py``) makes the *spec's* claims falsifiable and owns ``check_pack``, the
spec-side closure checker. This file tests the *shipped* contract: ``ccg/schemas/rule-pack-v1.schema.json``
plus ``ccg.rulepack.validate_pack`` (the RE2-subset, registry and cross-field checks JSON Schema cannot
carry). Where the two disagree, the schema and validator win (spec §12.1, S31); ``check_pack`` is the
spec's own reading and is not imported here, except for the extraction helpers (``PACKS``, ``BLOCKS``),
so every pack and negative example is taken mechanically from the spec's YAML blocks.

Topic cells (spec v1.4, from ``teco`` ruling O-16): ``PROVIDES topic`` is a cell of ``runtime-built`` and ``config-derived``, so
``onUnknown: config-derived`` is valid on a ``consumes`` or a ``provides`` of kind ``topic``, and the ``provides`` default
``onUnknown`` is ``config-derived`` for ``topic`` (``drop`` for every other kind). ``envPorts.excludeVars`` is optional (§1.6).

Spec §1.4 "A slot has one mode" (N-29): every ``onUnknown`` nested in one slot names the same mode, so a ``firstOf`` whose
nested expressions name different modes is rejected; an expression with none takes the slot's mode.

Leaves left to the P1 conformance test: ``UNEXERCISED_UNTIL_P1`` in ``test_rule_pack_leaf_mutation.py`` (P3-1 harness).
"""

from __future__ import annotations

import copy
import itertools
import json
import re
import sys
from pathlib import Path

import jsonschema
import pytest
import referencing
import yaml

import test_rule_pack_spec as spec
from ccg import rulepack

CPG = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- the reason registry (§1.9)

def parse_on(cell: str) -> list[str]:
    """Spec3 §7.2's ``On`` cell as registry tokens: backticks off, ``A``/``B`` split, ``field`` prefix a kind."""
    tokens: list[str] = []
    for piece in (p.strip() for p in cell.split(",")):
        piece = piece.replace("`", "")
        m = re.fullmatch(r"(?:(field) )?(READS|WRITES)(?:/(READS|WRITES))?", piece)
        if m:
            for edge in filter(None, (m[2], m[3])):
                tokens.append(f"{edge} field" if m[1] else edge)
        else:
            tokens.append(piece)
    return tokens


def spec3_rows(heading: str, until: str) -> list[list[str]]:
    t3 = spec.SPEC3.read_text()
    block = t3[t3.index(heading): t3.index(until)]
    return [[c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in block.splitlines() if ln.startswith("| `")]


def seeded_registry() -> dict[str, dict]:
    """The registry the way §1.9 derives it: spec3 §7.2 and §7.3, then §1.9's table, then O-16's cells."""
    reg: dict[str, dict] = {}
    for code, status, on, *_ in spec3_rows("### 7.2 Status reasons", "### 7.3"):
        reg[code.strip("`")] = {"status": status, "on": parse_on(on)}
    for code, *_ in spec3_rows("### 7.3 Value-set", "### 7.4"):
        row = reg.setdefault(code.strip("`"), {"status": "none", "on": []})
        row["on"].append("valueSet")
    for code, status, on, _used in spec.table_rows("| Code | Status | On (added"):
        row = reg.setdefault(code.strip("`"), {"status": status, "on": []})
        row["on"] += parse_on(on)
    for code in ("runtime-built", "config-derived"):
        reg[code]["on"].append("PROVIDES topic")  # O-16, spec §1.9's closing paragraph
    reg["advisory-unscored"] = {"status": "none", "on": ["Finding"]}  # spec3 §7.4
    reg["binding-confirmed"]["on"].append("VIA")  # P1-17: docs/plans/connected-code-graph-p1-graph.md §2 (O-9)
    for code in ("shell-dynamic", "prompt-mention"):  # P1-17 (3), (7): the linker derives them from the site; lifted-role is NOT widened
        reg[code]["on"].append("Link")
    for code in ("test", "frontend-ignore"):  # spec4: "both codes join the §7 registry ... on header.skippedFiles"
        reg[code] = {"status": "none", "on": ["header.skippedFiles"]}
    return reg


def test_the_registry_file_is_the_seeded_registry():
    shipped = {r["code"]: {"status": r["status"], "on": r["on"]} for r in rulepack.load_registry_rows()}
    seeded = seeded_registry()
    assert set(shipped) == set(seeded)
    for code, row in seeded.items():
        assert shipped[code]["status"] == row["status"], code
        assert sorted(shipped[code]["on"]) == sorted(row["on"]), code


# ---------------------------------------------------------------- helpers

PACKS = spec.all_packs()
NEG = spec.negatives()
NEGATIVE_LINES = list(NEG["shared"]) + [ln for ls in NEG["near-miss"].values() for ln in ls]


def errors(pack: dict, dirname: str | None = None, negatives=None) -> list[str]:
    return rulepack.pack_errors(pack, dirname=dirname if dirname is not None else pack.get("id"), negatives=negatives)


def accepted(pid: str, mutate=None) -> list[str]:
    pack = copy.deepcopy(PACKS[pid])
    if mutate:
        mutate(pack)
    return errors(pack, negatives=NEGATIVE_LINES if pid == "infra-signatures" else None)


# ---------------------------------------------------------------- positive: the seven spec packs validate

@pytest.mark.parametrize("pid", sorted(PACKS))
def test_each_spec_pack_validates(pid):
    assert accepted(pid) == []


def test_validate_pack_raises_a_pack_error_naming_every_problem():
    bad = copy.deepcopy(PACKS["fastapi"])
    bad["packFormat"] = 2
    bad["kind"] = "plugin"
    with pytest.raises(rulepack.PackError) as ei:
        rulepack.validate_pack(bad, dirname="fastapi")
    assert len(ei.value.messages) >= 2


@pytest.mark.parametrize("pid", ["fastapi", "fetch", "fastmcp", "kafka", "paper-inhouse-bus"])
def test_a_variation_with_one_rule_removed_is_valid(pid):
    assert accepted(pid, lambda p: p.update(rules=p["rules"][1:])) == []


# ---------------------------------------------------------------- the spec's own negative lists, against the shipped contract

@pytest.mark.parametrize("name,mutate", spec.NEGATIVES, ids=[n for n, _ in spec.NEGATIVES])
def test_spec_negatives_are_rejected_by_the_schema_validator(name, mutate):
    assert accepted("fastapi", mutate)


@pytest.mark.parametrize("name,pid,mutate", spec.RANGE_NEGATIVES, ids=[n for n, *_ in spec.RANGE_NEGATIVES])
def test_spec_range_negatives_are_rejected_by_the_schema_validator(name, pid, mutate):
    assert accepted(pid, mutate)


# ---------------------------------------------------------------- N-n not in the spec test's lists

def set_emit(**fields):
    def m(p):
        p["rules"][0]["emit"][0]["provides"].update(fields)
    return m


def drop_kind_decl(p):
    p["rules"][0]["emit"][0]["provides"]["kind"] = "widget"


def new_query(**kw):
    def m(p):
        q = p["queries"][0]
        for k, v in kw.items():
            q[k] = v
    return m


def query_binding(i, where):
    def m(p):
        p["queries"][0]["from"][i]["where"] = where
    return m


MORE_NEGATIVES = [
    ("N-2 uppercase id", "fastapi", lambda p: p.update(id="FastAPI")),
    ("N-2 id too long", "fastapi", lambda p: p.update(id="a" * 41)),
    ("N-9 invented reason", "fastapi", set_emit(reason="looks-fine")),
    ("N-9 reason registry status disagrees", "fastapi", set_emit(reason="key-match", status="possible")),
    ("N-9 unmounted reason not in the registry", "fastapi", lambda p: p["mounts"][0]["unmounted"].update(reason="looks-fine")),
    ("N-9 unmounted reason not on PROVIDES", "fastapi", lambda p: p["mounts"][0]["unmounted"].update(reason="unknown-key")),
    ("N-9 unmounted status disagrees with its code", "fastapi", lambda p: p["mounts"][0]["unmounted"].update(status="confirmed")),
    ("N-10 undeclared kind", "fastapi", drop_kind_decl),
    ("N-10 undeclared kind on a link", "fastapi", lambda p: p.update(link=[{"kind": "widget", "matcher": "exact"}])),
    ("N-11 vocab entry without keyNorm", "kafka", lambda p: p["vocab"][0].pop("keyNorm")),
    ("N-13 duplicate rule ids", "fastapi", lambda p: p["rules"][1].update(id="route")),
    ("N-14 write key in a query", "paper-inhouse-bus", new_query(set={"x": 1})),
    ("N-14 secretSighting record", "paper-inhouse-bus", lambda p: p["queries"][0]["from"][1].update(record="secretSighting")),
    ("N-14 five bindings", "paper-inhouse-bus", lambda p: p["queries"][0]["from"].append(copy.deepcopy(p["queries"][0]["from"][1]) | {"as": "z"})),
    ("N-14 join on a non-id field", "paper-inhouse-bus", query_binding(1, {"id": "$dec.name"})),
    ("N-14 no join equality after the first", "paper-inhouse-bus", query_binding(1, {"name": {"regex": "x"}})),
    ("N-14 join to a later alias", "paper-inhouse-bus", query_binding(1, {"id": "$ty.id"})),
    ("N-14 where names a field the record lacks", "paper-inhouse-bus", query_binding(0, {"nme": {"regex": "x"}})),
    ("N-14 yield names a field the record lacks", "paper-inhouse-bus", lambda p: p["queries"][0]["yield"].update(topic="$dec.nme")),
    ("N-14 no yield columns", "paper-inhouse-bus", new_query(**{"yield": {}})),
    ("N-14 record outside spec 5", "paper-inhouse-bus", lambda p: p["queries"][0]["from"][1].update(record="widget")),
    ("N-15 seed without arg", "fetch", lambda p: p["liftings"][0]["seeds"][0].pop("arg")),
    ("N-16 prefix without child", "fastapi", lambda p: p["mounts"][0].pop("child")),
    ("N-17 confirmed with a partial key slot", "fastapi",
     lambda p: (p["rules"][0]["extract"]["route"].update(onUnknown="partial"), set_emit(status="confirmed")(p))),
    ("N-17 confirmed with the default-partial consumes slot", "fetch", lambda p: p["rules"][0]["emit"][0]["consumes"].update(status="confirmed")),
    ("N-17 confirmed beside a possible code", "fastapi", set_emit(status="confirmed", reason="partial-key")),
    ("N-18 level severe", "severity-rules", lambda p: p["rows"][0].update(level="severe")),
    ("N-18 rank 4", "severity-rules", lambda p: p["rows"][0].update(rank=4)),
    ("N-19 transport uppercase", "kafka", lambda p: p["vocab"][0].update(transport="Kafka")),
    ("N-19 other pack's inhouse transport", "paper-inhouse-bus", lambda p: p["vocab"][0].update(transport="inhouse-other")),
    ("N-20 config value", "fastapi", lambda p: p["rules"][0]["extract"].update(v={"config": "value"})),
    ("N-22 class latency", "infra-signatures", lambda p: p["signatures"][0].update(**{"class": "latency"})),
    ("N-22 sample its regex does not match", "infra-signatures", lambda p: p["signatures"][0].update(sample="all fine")),
    ("N-22 sampleSource guessed", "infra-signatures", lambda p: p["signatures"][0].update(sampleSource="guessed")),
    ("N-22 regex matches a near-miss", "infra-signatures", lambda p: p["signatures"][0].update(regex=p["signatures"][0]["regex"] + "|connection")),
    ("N-22 id not runtime-class", "infra-signatures", lambda p: p["signatures"][0].update(id="python-connection-refused-2")),
    ("N-22 id of another runtime", "infra-signatures", lambda p: p["signatures"][0].update(id="node-connection-refused-2")),
    ("N-22 third-party id without the third-party stem", "infra-signatures", lambda p: next(s for s in p["signatures"] if s["id"] == "python-third-party-redis").update(id="python-redis")),
    ("N-22 non-third-party class with a third-party stem", "infra-signatures", lambda p: p["signatures"][0].update(id="python-third-party-x")),
    ("N-24 rank 3 level low", "severity-rules", lambda p: p["rows"][0].update(level="low")),
    ("N-24 id other than derived", "severity-rules", lambda p: p["rows"][0].update(id="INJ-9")),
    ("N-24 id of another rank", "severity-rules", lambda p: p["rows"][0].update(id="INJ-2")),
    ("N-24 id of another finding", "severity-rules", lambda p: p["rows"][0].update(id="SEC-1")),
    ("N-24 second row for one (finding, rank)", "severity-rules", lambda p: p["rows"].append(dict(p["rows"][0]))),
    ("N-25 basename plus exact", "fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(name={"basename": "x", "exact": "y"})),
    ("N-25 ignoreCase on exact", "fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(name={"exact": "x", "ignoreCase": True})),
    ("N-25 targetArg form dotted", "fastmcp", lambda p: p["launch"]["targetArg"][0].update(form="dotted")),
    ("N-25 envPorts adoptedBy any", "fastmcp", lambda p: p["launch"]["envPorts"][0].update(adoptedBy="any")),
    ("N-25 urlBindings scope on a config source", "fastmcp", lambda p: p["urlBindings"][0]["urlSources"][0].update(scope="file")),
    ("N-27 config-derived on a provides of kind http", "fastapi", lambda p: p["rules"][0]["extract"]["route"].update(onUnknown="config-derived")),
    ("N-27 config-derived on a consumes of kind http", "fetch", lambda p: p["rules"][0]["extract"].update(url={"lifted": "url", "onUnknown": "config-derived"})),
    ("N-27 reason config-derived on PROVIDES http", "fastapi", set_emit(reason="config-derived")),
    ("N-27 reason runtime-built on PROVIDES http", "fastapi", set_emit(reason="runtime-built")),
    ("N-27 reason config-derived on CONSUMES http", "fetch", lambda p: p["rules"][0]["emit"][0]["consumes"].update(reason="config-derived")),
    ("N-28 empty firstOf", "fastapi", lambda p: p["rules"][0]["extract"].update(route={"firstOf": []})),
    ("N-28 empty anyOf", "fastapi", lambda p: p["rules"][0].update(match={"anyOf": []})),
    ("N-28 query without a handler column", "paper-inhouse-bus", lambda p: p["queries"][0]["yield"].pop("handler")),
    ("N-28 rule on an unknown query", "paper-inhouse-bus", lambda p: p["rules"][1].update(match={"query": "nope"})),
    ("seed resolution outside the bundle enum", "fetch", lambda p: p["liftings"][0]["seeds"][0]["call"].update(resolution=["external", "nowhere"])),
    ("decorator form outside the bundle enum", "fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(form="macro")),
    ("mount call resolution outside the bundle enum", "fastapi", lambda p: p["mounts"][0]["call"].update(resolution="maybe")),
    ("launch kind outside the bundle enum", "fastapi", lambda p: p["rules"][0].update(match={"launch": {"kind": "spawn"}})),
    ("companion bareDefault is not a verb", "fetch", lambda p: p["liftings"][0]["companions"][0].update(bareDefault="FETCH")),
    ("companion unknown is not a verb", "fetch", lambda p: p["liftings"][0]["companions"][0].update(unknown="any")),
    ("yield names a column the query lacks", "paper-inhouse-bus", lambda p: p["rules"][1]["extract"].update(ns={"yield": "nope"})),
    ("yield on a rule that is not on a query", "fastapi", lambda p: p["rules"][0]["extract"].update(route={"yield": "handler"})),
    ("a lifting role kind that is undeclared", "fetch", lambda p: p["liftings"][0]["role"].update(kind="widget")),
    ("a urlBinding kind that is undeclared", "fastmcp", lambda p: p["urlBindings"][0].update(kinds=["widget"])),
    ("duplicate lifting ids", "fetch", lambda p: p["liftings"].append(copy.deepcopy(p["liftings"][0]))),
    ("duplicate urlBinding ids", "fastmcp", lambda p: p["urlBindings"].append(copy.deepcopy(p["urlBindings"][0]))),
    ("duplicate query ids", "paper-inhouse-bus", lambda p: p["queries"].append(copy.deepcopy(p["queries"][0]))),
    ("onUnknown config-derived nested in a firstOf on a provides of kind http", "fastapi",
     lambda p: p["rules"][0]["extract"].update(route={"firstOf": [{"arg": "a0", "onUnknown": "config-derived"}, {"arg": "kw.path"}]})),
    ("{mount} with no mounts declaration", "fetch", lambda p: p["rules"][0]["emit"][0]["consumes"].update(key="{mount}{url}")),
    ("confirmed on a firstOf with no always-resolved member", "fetch",
     lambda p: (p["rules"][0]["extract"].update(url={"firstOf": [{"lifted": "url"}, {"arg": "a0"}]}), p["rules"][0]["emit"][0]["consumes"].update(status="confirmed"))),
    ("a query alias bound twice", "paper-inhouse-bus", lambda p: p["queries"][0]["from"][2].update(**{"as": "h"})),
    ("yield from an unbound alias", "paper-inhouse-bus", lambda p: p["queries"][0]["yield"].update(topic="$zz.id")),
    ("yield args of a record without args", "paper-inhouse-bus", lambda p: p["queries"][0]["yield"].update(topic="$h.args.a0")),
    ("where compares args of a record without args", "paper-inhouse-bus", query_binding(2, {"id": "$h.parentId", "name": "$h.args.a0"})),
    ("where refers to a field the aliased record lacks", "paper-inhouse-bus", query_binding(1, {"id": "$dec.nope"})),
    ("where joins a non-id field to an id field", "paper-inhouse-bus", query_binding(1, {"name": "$dec.targetId"})),
    ("where joins an id field to a non-id field", "paper-inhouse-bus", query_binding(1, {"id": "$dec.line"})),
    ("lifted rule on an undeclared lifting", "fetch", lambda p: p["rules"][0].update(match={"lifted": {"lifting": "nope"}})),
    ("P4-3 lifting's role.slot renamed: the rule's lifted url dangles", "fetch", lambda p: p["liftings"][0]["role"].update(slot="urlZ")),
    ("P4-3 lifting's role.kind differs from the kind the rule emits", "fetch", lambda p: p["liftings"][0]["role"].update(kind="topic")),
    ("P4-3 lifting without its verb companion: the rule's lifted verb dangles", "fetch", lambda p: p["liftings"][0].pop("companions")),
    ("P4-3 lifted slot in a rule that matches no lifting", "fetch", lambda p: p["rules"][0].update(match={"callsite": {"callee": "fetch"}})),
    ("P4-4 undeclared lifting in the second anyOf clause", "fetch",
     lambda p: p["rules"][0].update(match={"anyOf": [{"lifted": {"lifting": "fetch-url"}}, {"lifted": {"lifting": "nope"}}]})),
    ("P4-4 undeclared query in the second anyOf clause", "paper-inhouse-bus",
     lambda p: p["rules"][1].update(match={"anyOf": [{"query": "handler-ns"}, {"query": "nope"}]})),
    ("P4-4 evidence subject in the second anyOf clause of a non-infra pack", "fastapi",
     lambda p: p["rules"][0].update(match={"anyOf": [p["rules"][0]["match"], {"evidence": {"regex": "x"}}]})),
    ("N-29 mixed nested onUnknown modes [drop, partial] under status confirmed", "fastapi",
     lambda p: (p["rules"][0]["extract"].update(route={"firstOf": [{"arg": "a0", "onUnknown": "drop"}, {"arg": "kw.path", "onUnknown": "partial"}]}),
                p["rules"][0]["emit"][0]["provides"].update(status="confirmed"))),
    ("N-29 mixed nested onUnknown modes [partial, drop]", "fastapi",
     lambda p: p["rules"][0]["extract"].update(route={"firstOf": [{"arg": "a0", "onUnknown": "partial"}, {"arg": "kw.path", "onUnknown": "drop"}]})),
    ("key names a slot the rule does not extract", "fastapi", set_emit(key="{verb} {nope}")),
    ("evidence subject outside an infra pack", "fastapi", lambda p: p["rules"][0].update(match={"evidence": {"regex": "x"}})),
    ("duplicate mount ids", "fastapi", lambda p: p["mounts"].append(copy.deepcopy(p["mounts"][0]))),
    ("duplicate signature ids", "infra-signatures", lambda p: p["signatures"].append(copy.deepcopy(p["signatures"][0]))),
]


@pytest.mark.parametrize("name,pid,mutate", MORE_NEGATIVES, ids=[n for n, *_ in MORE_NEGATIVES])
def test_negative_examples_are_rejected(name, pid, mutate):
    assert accepted(pid, mutate), f"{name} was accepted"


@pytest.mark.parametrize("key", ["packFormat", "id", "version", "kind", "vocabVersion"])
def test_a_pack_without_a_required_header_key_is_rejected(key):
    assert accepted("fastapi", lambda p: p.pop(key))


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "1.0.0.0", "01.0.0", "1.0.x", 1.0, ""])
def test_a_version_that_is_not_semver_is_rejected(version):
    assert accepted("fastapi", lambda p: p.update(version=version))


@pytest.mark.parametrize("version", ["0.1.0", "10.20.30", "1.0.0-rc.1", "1.0.0+build.5"])
def test_semver_versions_are_accepted(version):
    assert accepted("fastapi", lambda p: p.update(version=version)) == []


def test_an_id_that_is_not_its_directory_name_is_rejected():
    assert errors(PACKS["fastapi"], dirname="not-fastapi")


def test_the_regex_mutants_of_8_2_are_rejected_against_the_negatives():
    by_id = {e["id"]: e for e in PACKS["infra-signatures"]["signatures"]}
    mutants = {
        "python-timeout": by_id["python-timeout"]["regex"] + "|timed out",
        "any-oom-kill": by_id["any-oom-kill"]["regex"].replace(r"137\b", "137"),
        "any-oom-kill/bare": by_id["any-oom-kill"]["regex"] + "|oom-kill",
        "python-oom-kill": by_id["python-oom-kill"]["regex"] + "|memory",
        "any-connection-refused": by_id["any-connection-refused"]["regex"] + r"|refused",
        "go-timeout": ".*",
    }
    for name, rx in mutants.items():
        sid = name.split("/")[0]
        assert accepted("infra-signatures", lambda p, sid=sid, rx=rx: next(
            s for s in p["signatures"] if s["id"] == sid).update(regex=rx)), name


@pytest.mark.parametrize("rx", [r"(?=x)abc", r"(?!x)abc", r"(?<=x)abc", r"(?<!x)abc", r"(a)\1", r"(?P<n>a)(?P=n)",
                                r"(?>a)", r"a*+", r"a++", r"a?+", r"a{2}+", r"(?(1)a|b)", r"a" * 257, "("])
def test_regexes_outside_the_re2_subset_are_rejected(rx):
    assert accepted("fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(name={"regex": rx}))


def test_regex_problem_names_each_outside_the_subset_form_directly():
    """Unit level: the schema's maxLength and Python's own parser also catch some of these, so the validator code is pinned here."""
    assert rulepack.regex_problem("a" * 257) == "longer than 256 characters"
    assert rulepack.regex_problem("a" * 256) is None
    assert rulepack.regex_problem(r"(a)?(?(1)b|c)") == "conditional group"
    assert rulepack.regex_problem(r"(?P<n>a)(?P=n)") == "backreference"
    assert rulepack.regex_problem(r"(a)\1") == "backreference"
    assert rulepack.regex_problem(r"(?<=a)b") == "lookaround"
    assert rulepack.regex_problem(r"(?>a)") == "atomic group"
    assert rulepack.regex_problem(r"a*+") == "possessive quantifier"
    assert rulepack.regex_problem(r"a{1,2}+") == "possessive quantifier"
    assert rulepack.regex_problem("(").startswith("not a regex")


_HOSTILE_REGEXES = ["a{99999999999}", "a{1,99999999999}", "(a){,99999999999}", "(" * 10_000 + ")" * 10_000, "a" * 1_000_000]


@pytest.mark.parametrize("rx", _HOSTILE_REGEXES)
def test_regex_problem_never_raises_on_hostile_patterns_and_validate_pack_reports_them(rx):
    why = rulepack.regex_problem(rx)
    assert isinstance(why, str) and why
    assert len(why) <= 200  # a bounded reason: never the pattern echoed back
    assert rx[:50] not in why or len(rx) <= 50
    errs = accepted("fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(name={"regex": rx}))
    assert errs  # over-length ones are refused by the schema's maxLength before the validator code runs
    if len(rx) <= 256:
        assert any(why in e for e in errs)


def test_a_huge_repetition_count_names_its_failure_class():
    assert rulepack.regex_problem("a{99999999999}").startswith("not a regex: OverflowError")


@pytest.mark.parametrize("exc", [OverflowError("big"), RecursionError("deep"), ValueError("bad"), MemoryError("mem")])
def test_regex_problem_turns_any_compiler_exception_into_a_reason(monkeypatch, exc):
    def boom(rx, *a, **k):
        raise exc
    monkeypatch.setattr(rulepack.re, "compile", boom)
    assert rulepack.regex_problem("abc").startswith(f"not a regex: {type(exc).__name__}")


@pytest.mark.parametrize("rx", [r"[(?=]x", r"[(?!a]", r"[a(?=]x", r"[ab(?<=]x", r"[a*+]", r"[*+]+", r"[^]+]", r"[]+]+", r"[a-z]\++"])
def test_regex_syntax_inside_a_character_class_is_literal(rx):
    assert rulepack.regex_problem(rx) is None


@pytest.mark.parametrize("rx", [r"a+?", r"(?:a|b)+", r"[+*?]+", r"\\1", r"(?P<n>a)", r"a" * 256, r"\+\+", r"[a\]]\+"])
def test_regexes_inside_the_re2_subset_are_accepted(rx):
    assert accepted("fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(name={"regex": rx})) == []


# ---------------------------------------------------------------- the three real packs (cpg/packs/*/pack.yaml)

REAL_PACKS = sorted(PACKS)  # all seven (S31 shipped three, S32 the other four)
NEGATIVES_FILE = "negatives.yaml"  # beside infra-signatures/pack.yaml: the spec's section 8 negatives block, for N-22


def spec_block_text(pid: str) -> str:
    """The pack's YAML exactly as the spec writes it: the one fenced block whose first key is ``packFormat`` -- for
    ``infra-signatures``, that header block followed by ``signatures:`` and the section 8.2 list block (the file's body)."""
    for _line, body in spec.yaml_blocks():
        d = yaml.safe_load(body)
        if isinstance(d, dict) and d.get("id") == pid and "packFormat" in d:
            if pid != "infra-signatures":
                return body
            sig_block = next(b for _l, b in spec.yaml_blocks() if isinstance(yaml.safe_load(b), list) and "class" in yaml.safe_load(b)[0])
            return body.rstrip("\n") + "\nsignatures:\n" + sig_block
    raise AssertionError(pid)


def without_comment_lines(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("#")).strip()


@pytest.mark.parametrize("pid", REAL_PACKS)
def test_the_pack_file_validates_and_carries_its_directory_name(pid):
    pack = rulepack.load_pack(CPG / "packs" / pid / "pack.yaml")
    assert pack["id"] == pid


@pytest.mark.parametrize("pid", REAL_PACKS)
def test_the_pack_file_is_the_spec_block_plus_status_paper(pid):
    """Extracted mechanically (spec 10): the file's text, less comments and the one `status: paper` line, is the block."""
    text = (CPG / "packs" / pid / "pack.yaml").read_text()
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert body.count("status: paper") == 1
    body.remove("status: paper")
    assert "\n".join(body).strip() == without_comment_lines(spec_block_text(pid))
    assert rulepack.load_yaml(text) == {**PACKS[pid], "status": "paper"}


@pytest.mark.parametrize("pid", REAL_PACKS)
def test_the_pack_file_is_marked_paper(pid):
    assert rulepack.load_pack(CPG / "packs" / pid / "pack.yaml")["status"] == "paper"


# Packs that are files only, with no block in the rule-pack spec: id -> the files beside `pack.yaml` (P1-46).
NON_SPEC_PACKS = {"shell-mcp-call": {NEGATIVES_FILE}}


def test_the_packs_directory_holds_exactly_the_seven_spec_packs_and_the_files_only_packs():
    """No pack directory beyond the spec's seven and `NON_SPEC_PACKS`, and no file in one beyond its pack (plus its negatives)."""
    assert len(REAL_PACKS) == 7
    on_disk = {p.name for p in (CPG / "packs").iterdir()}
    assert on_disk == set(REAL_PACKS) | set(NON_SPEC_PACKS)
    for pid in on_disk:
        files = {p.name for p in (CPG / "packs" / pid).iterdir()}
        beside = {NEGATIVES_FILE} if pid == "infra-signatures" else NON_SPEC_PACKS.get(pid, set())
        assert files == {"pack.yaml"} | beside, (pid, files)


# ---------------------------------------------------------------- the topic knock-on (O-16)

def kafka_with(mutate):
    p = copy.deepcopy(PACKS["kafka"])
    mutate(p)
    return errors(p)


def test_core_kinds_are_spec3_section_6s_table():
    lines = spec.SPEC3.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| `kind` | `keyNorm` form"))
    body = list(itertools.takewhile(lambda ln: ln.startswith("|"), lines[start + 2:]))
    assert {r.split("|")[1].strip().strip("`") for r in body} == set(rulepack.CORE_KINDS)

def test_the_registry_cells_for_a_topic_publisher_exist_and_spec3_stays_frozen():
    reg = {r["code"]: r for r in rulepack.load_registry_rows()}
    for code in ("runtime-built", "config-derived"):
        assert rulepack.allows(reg[code], "PROVIDES", "topic"), code
        assert not rulepack.allows(reg[code], "PROVIDES", "http"), code
    assert not rulepack.allows(reg["config-derived"], "CONSUMES", "http")
    assert rulepack.allows(reg["config-derived"], "CONSUMES", "topic")
    # spec3 section 7.2 is frozen: its rows do not list PROVIDES for these codes
    for code, _status, on, *_ in spec3_rows("### 7.2 Status reasons", "### 7.3"):
        if code.strip("`") in ("runtime-built", "config-derived"):
            assert "PROVIDES" not in on


def test_the_spec_checker_and_the_shipped_validator_both_accept_config_derived_on_a_topic_publisher():
    """Spec v1.4 (O-16): `onUnknown: config-derived` is valid on a `provides` of kind `topic`; the checker and the validator agree."""
    pack = copy.deepcopy(PACKS["kafka"])
    pack["rules"][0]["extract"]["topic"].update(onUnknown="config-derived")
    assert errors(pack) == []
    spec.check_pack(pack)
    assert rulepack.default_on_unknown("provides", "topic") == "config-derived"


@pytest.mark.parametrize("path,value", [(("id",), "fastapi\n"), (("version",), "0.1.0\n")])
def test_an_anchored_pattern_rejects_a_trailing_newline(path, value):
    """Python's `$` matches before a final newline; the JSON Schema `pattern` means the whole string (ECMA 262), so the validator must too."""
    assert accepted("fastapi", lambda p: p.update({path[0]: value}))


def test_envports_without_excludevars_is_valid():
    """Spec 1.6 (v1.4): `excludeVars` is optional."""
    assert accepted("fastmcp", lambda p: p["launch"]["envPorts"][0].pop("excludeVars")) == []


@pytest.mark.parametrize("edge,kind,expected", [
    ("provides", "topic", "config-derived"), ("provides", "http", "drop"), ("provides", "mcp_tool", "drop"),
    ("writes", "data_entity", "drop"), ("consumes", "topic", "partial"), ("consumes", "http", "partial"),
    ("reads", "data_entity", "partial")])
def test_default_on_unknown_per_edge_and_kind(edge, kind, expected):
    assert rulepack.default_on_unknown(edge, kind) == expected


def test_the_kafka_publisher_with_an_unresolved_topic_degrades_to_a_possible_fact_not_a_drop():
    """The shipped kafka block has `provides topic` with no onUnknown: valid only because PROVIDES topic is a registry cell."""
    assert accepted("kafka") == []
    rule = next(r for r in PACKS["kafka"]["rules"] if r["id"] == "produce")
    assert "onUnknown" not in rule["extract"]["topic"]


@pytest.mark.parametrize("mode", ["config-derived", "runtime-built", "partial", "drop"])
def test_every_unknown_mode_is_valid_on_a_provides_of_kind_topic(mode):
    assert kafka_with(lambda p: p["rules"][0]["extract"]["topic"].update(onUnknown=mode)) == []


@pytest.mark.parametrize("reason,status", [("config-derived", "possible"), ("runtime-built", "possible"), ("partial-key", "possible"),
                                            ("key-match", "confirmed")])
def test_reasons_valid_on_a_topic_provides(reason, status):
    def m(p):
        p["rules"][0]["emit"][0]["provides"].update(reason=reason, status=status)
        p["rules"][0]["extract"]["topic"]["onUnknown"] = "drop" if status == "confirmed" else "config-derived"
    assert kafka_with(m) == []


def test_a_confirmed_topic_publisher_needs_drop_on_its_key_slot():
    confirm = lambda p: p["rules"][0]["emit"][0]["provides"].update(status="confirmed")
    assert kafka_with(confirm)  # default config-derived: the fact is possible when the name is unresolved
    assert kafka_with(lambda p: (confirm(p), p["rules"][0]["extract"]["topic"].update(onUnknown="drop"))) == []


def test_partial_is_partial_key_on_http_and_runtime_built_elsewhere():
    http = lambda p: p["rules"][0]["extract"]["route"].update(onUnknown="partial")
    assert accepted("fastapi", http) == []  # partial-key is registered on PROVIDES
    writes = lambda p: p["rules"][0]["emit"].__setitem__(0, {"writes": {"kind": "http", "key": "{verb} {mount}{route}"}})
    assert accepted("fastapi", lambda p: (http(p), writes(p)))  # partial-key is not on WRITES
    assert accepted("fastapi", lambda p: writes(p)) == []  # writes default is drop


def test_a_nested_firstof_with_one_named_mode_and_one_unnamed_expression_is_valid():
    """N-29: an expression with no onUnknown takes the slot's mode, so [partial, <none>] is one mode."""
    route = lambda p: p["rules"][0]["extract"].update(route={"firstOf": [{"arg": "a0", "onUnknown": "partial"}, {"arg": "kw.path"}]})
    assert accepted("fastapi", route) == []


def test_runtime_built_is_not_valid_for_an_http_publisher_but_is_for_an_http_consumer():
    assert accepted("fastapi", lambda p: p["rules"][0]["extract"]["route"].update(onUnknown="runtime-built"))
    assert accepted("fetch", lambda p: p["rules"][0]["extract"].update(url={"lifted": "url", "onUnknown": "runtime-built"})) == []


def test_config_derived_on_a_slot_no_key_uses_is_rejected():
    assert kafka_with(lambda p: p["rules"][0]["extract"].update(unused={"arg": "a1", "onUnknown": "config-derived"}))


# ---------------------------------------------------------------- loading

def test_a_duplicate_yaml_key_is_rejected_by_the_loader():
    with pytest.raises(Exception, match="duplicate key"):
        rulepack.load_yaml("a: 1\na: 2\n")


def test_a_bare_on_key_is_rejected_because_yaml_1_1_reads_it_as_true():
    with pytest.raises(Exception, match="non-string key"):
        rulepack.load_yaml("on: [x]\n")


def test_a_pack_file_in_the_wrong_directory_is_rejected(tmp_path):
    d = tmp_path / "not-fastapi"
    d.mkdir()
    (d / "pack.yaml").write_text((CPG / "packs/fastapi/pack.yaml").read_text())
    with pytest.raises(rulepack.PackError, match="directory name"):
        rulepack.load_pack(d / "pack.yaml")


def test_registry_rows_have_exactly_code_status_on():
    assert all(set(r) == {"code", "status", "on"} and r["status"] in ("confirmed", "possible", "none")
               for r in rulepack.load_registry_rows())


def test_a_parameterized_code_resolves_to_its_template_row_and_only_that_form():
    assert rulepack.reason_row("severity-rule:INJ-1")["status"] == "none"
    assert rulepack.reason_row("flow-unavailable:python")["status"] == "possible"
    assert rulepack.reason_row("severity-rule:<id>") is None  # the template is not a usable code
    assert rulepack.reason_row("key-match:x") is None
    assert rulepack.reason_row("looks-fine") is None


# ---------------------------------------------------------------- N-26: generated probes (schema-walking, not an enumerated list)

# The only object schemas that are open on purpose: maps whose keys the pack author names (slot names, query columns,
# query field tests). Their values are closed (an expression, a `$alias.field` reference, a matcher) and the validator
# checks the field names against the bundle schema.
OPEN_MAPS = {"#/$defs/rule/properties/extract", "#/$defs/query/properties/yield", "#/$defs/query/properties/from/items/properties/where"}


def object_schemas(node, path="#"):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for k, v in node.items():
            yield from object_schemas(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from object_schemas(v, f"{path}/{i}")


def subschema_validator(pointer: str):
    """A validator for the subschema at ``pointer`` (``#/$defs/x/...``), with the whole schema registered for its references."""
    schema = rulepack.load_schema()
    registry = referencing.Registry().with_resource(schema["$id"], referencing.Resource.from_contents(schema))
    return jsonschema.Draft202012Validator({"$ref": schema["$id"] + ("" if pointer == "#" else pointer)}, registry=registry)


def test_every_object_the_schema_defines_is_closed_except_the_named_maps():
    objects = dict(object_schemas(rulepack.load_schema()))
    assert len(objects) > 60
    assert {p for p, s in objects.items() if s.get("additionalProperties") is not False} == OPEN_MAPS


def test_every_object_the_schema_defines_rejects_one_planted_unknown_key():
    """Generated from the schema: for each object schema, an instance holding only the planted key must fail on that key."""
    objects = dict(object_schemas(rulepack.load_schema()))
    for pointer in objects:
        found = list(subschema_validator(pointer).iter_errors({PLANTED: 1}))
        on_key = [e for e in found if (e.validator == "additionalProperties" and PLANTED in e.message)
                  or (list(e.absolute_path) == [PLANTED])]
        assert on_key, f"{pointer} accepts a planted key"


PLANTED = spec.PLANTED
SHAPES = [1, "x", {"exact": "x"}, [], True, None]


def plant_probe(packs: dict) -> tuple[int, list]:
    """Plant the key at every dict of every pack, with six value shapes; return (mutants tried, accepted ones)."""
    tried, accepted_ = 0, []
    for pid, pack in packs.items():
        negs = NEGATIVE_LINES if pid == "infra-signatures" else None
        for path in spec.dict_positions(pack):
            for shape in SHAPES:
                mutant = copy.deepcopy(pack)
                target = mutant
                for step in path:
                    target = target[step]
                target[PLANTED] = shape
                tried += 1
                if not errors(mutant, negatives=negs):
                    accepted_.append((pid, path, shape))
    return tried, accepted_


def test_a_key_planted_at_every_position_of_every_spec_pack_is_rejected():
    tried, accepted_ = plant_probe(PACKS)
    # the walk must equal an independent count of the YAML mappings in the same blocks (spec test's composer walker)
    assert tried == spec.yaml_mappings_in_pack_blocks() * len(SHAPES)
    assert tried > 1500
    assert not accepted_, f"{len(accepted_)} of {tried} mutants accepted: {accepted_[:5]}"


def test_the_planted_key_probe_can_fail(monkeypatch):
    """Against a schema with one object opened (`mount`), the probe sees it: the probe is not vacuous."""
    lax = copy.deepcopy(rulepack.load_schema())
    for alt in lax["$defs"]["mount"]["oneOf"]:
        del alt["additionalProperties"]
    monkeypatch.setattr(rulepack, "load_schema", lambda: lax)
    _, accepted_ = plant_probe({"fastapi": PACKS["fastapi"]})
    assert accepted_ and all(path[:1] == ("mounts",) for _, path, _ in accepted_)


# Out-of-range probe (P3-2): every scalar leaf of every spec pack is replaced by a value of no valid type or range;
# the only leaves that may accept a garbage string are the free-text ones below. A new leaf that accepts "zz bad!"
# fails this test until it is classified here, so a typed or enumerated leaf cannot be added unguarded.
OUT_OF_RANGE = [None, {"zz": 1}, [], 99999, "zz bad!"]
FREE_STRING_LEAVES = {
    # matcher operands and bare-string matchers: the pack author's text, checked by the RE2 subset where it is a regex
    "formula.configuration.penalties[].when.basename.regex", "formula.secrets.penalties[].when.path.regex",
    "launch.envPorts[].envVar.regex", "launch.envPorts[].excludeVars[].regex", "launch.frameworkDefaults[].command.basename",
    "launch.mountPaths[].app.fromCall.externalTarget.glob", "launch.mountPaths[].call.callee", "launch.portFlags[].command.basename",
    "mounts[].call.callee", "mounts[].constructorPrefix[].call.callee", "mounts[].roots[].call.callee",
    "mounts[].roots[].notWith[].callee", "mounts[].scope.imports[].module.prefix", "liftings[].seeds[].call.callee",
    "queries[].from[].where.name.regex", "rules[].match.callsite.callee", "rules[].match.callsite.callee.regex",
    "rules[].match.decorator.name.regex", "rules[].scope.imports[].module.exact", "rules[].scope.imports[].module.prefix",
    "runtimes[].regex", "urlBindings[].urlSources[].callsite.callee", "urlBindings[].urlSources[].config.keyPath.regex",
    # free text of a declaration
    "launch.frameworkDefaults[].protocol", "launch.portFlags[].flags[]", "mounts[].prefix.default",
    "rules[].emit[].consumes.key", "rules[].emit[].provides.key", "rules[].extract.verb.literal",
    "signatures[].library", "vocab[].example", "vocab[].keyNorm",
}


def leaf_paths(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaf_paths(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaf_paths(v, path + (i,))
    else:
        yield path


def path_pattern(path) -> str:
    return re.sub(r"\.?\[\]", lambda m: "[]", ".".join("[]" if isinstance(p, int) else p for p in path)).replace(".[]", "[]")


def out_of_range_probe(packs: dict) -> tuple[int, dict[str, set]]:
    tried, accepted_ = 0, {}
    for pid, pack in packs.items():
        negs = NEGATIVE_LINES if pid == "infra-signatures" else None
        for path in leaf_paths(pack):
            for shape in OUT_OF_RANGE:
                mutant = copy.deepcopy(pack)
                target = mutant
                for step in path[:-1]:
                    target = target[step]
                target[path[-1]] = shape
                tried += 1
                if not errors(mutant, negatives=negs):
                    accepted_.setdefault(path_pattern(path), set()).add(repr(shape))
    return tried, accepted_


def test_every_scalar_leaf_rejects_a_value_of_no_valid_type_or_range_unless_it_is_free_text():
    tried, accepted_ = out_of_range_probe(PACKS)
    independent = sum(1 for p in PACKS.values() for _ in leaf_paths(p)) * len(OUT_OF_RANGE)
    assert tried == independent and tried > 2500
    assert {k for k in accepted_} == FREE_STRING_LEAVES, (set(accepted_) ^ FREE_STRING_LEAVES)
    assert all(shapes == {"'zz bad!'"} for shapes in accepted_.values()), accepted_


# ---------------------------------------------------------------- N-26, from the schema: every enum, const, range, pattern, length

def subschemas(node, path="#"):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            if k not in ("enum", "const", "required", "dependentRequired"):
                yield from subschemas(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from subschemas(v, f"{path}/{i}")


WRONG_TYPE = {"string": 1, "integer": "zz bad!", "boolean": "zz bad!", "object": "zz bad!", "array": "zz bad!"}


def violations(s: dict):
    """(keyword, instance) pairs: one instance that breaks each constraint `s` states at its own level."""
    if "enum" in s:
        yield "enum", "zz bad!"
    if "const" in s:
        yield "const", "zz bad!"
    if "type" in s:
        yield "type", WRONG_TYPE[s["type"]]
    if "minimum" in s:
        yield "minimum", s["minimum"] - 1
    if "maximum" in s:
        yield "maximum", s["maximum"] + 1
    if "pattern" in s:
        assert not re.search(s["pattern"], "zz bad!"), s["pattern"]
        yield "pattern", "zz bad!"
    if "maxLength" in s:
        yield "maxLength", "a" * (s["maxLength"] + 1)
    if s.get("minLength"):
        yield "minLength", ""
    if s.get("minItems"):
        yield "minItems", []
    if "maxItems" in s:
        yield "maxItems", [{}] * (s["maxItems"] + 1)
    if s.get("minProperties"):
        yield "minProperties", {}
    if "maxProperties" in s:
        yield "maxProperties", {f"k{i}": 1 for i in range(s["maxProperties"] + 1)}


def test_every_enum_const_range_pattern_and_length_the_schema_defines_rejects_a_violation():
    checked = 0
    for pointer, s in subschemas(rulepack.load_schema()):
        for keyword, instance in violations(s):
            errs_ = [e for e in subschema_validator(pointer).iter_errors(instance) if e.validator == keyword and not list(e.absolute_path)]
            assert errs_, f"{pointer} does not enforce {keyword} ({instance!r})"
            checked += 1
    assert checked > 150


SPEC_ENUMS = {
    "#/$defs/link/properties/matcher": spec.ENUMS[("link", "matcher")],
    "#/$defs/signature/properties/class": spec.ENUMS[("class",)],
    "#/$defs/signature/properties/runtime": spec.ENUMS[("runtime",)],
    "#/$defs/runtimeEntry/properties/runtime": spec.ENUMS[("runtime",)],
    "#/$defs/signature/properties/sampleSource": spec.ENUMS[("sampleSource",)],
    "#/$defs/row/properties/level": spec.ENUMS[("level",)],
    "#/$defs/row/properties/finding": spec.ENUMS[("finding",)],
    "#/$defs/emission/properties/finding/properties/finding": spec.ENUMS[("finding",)],
    "#/$defs/fileRole": spec.ENUMS[("fileRole",)],
    "#/properties/status": spec.ENUMS[("status",)],
    "#/properties/kind": spec.KINDS,
    "#/$defs/contractEmission/properties/keyForm": spec.ENUMS[("emit", "keyForm")],
    "#/$defs/expr/properties/onUnknown": spec.ENUMS[("onUnknown",)],
    "#/$defs/expr/properties/target": spec.ENUMS[("target-expr",)],
    "#/$defs/expr/properties/config": spec.ENUMS[("config-expr",)],
    "#/$defs/expr/properties/lifted": spec.ENUMS[("lifted-expr",)],
    "#/$defs/launch/properties/targetArg/items/properties/form": spec.ENUMS[("targetArg", "form")],
    "#/$defs/launch/properties/envPorts/items/properties/adoptedBy": spec.ENUMS[("envPorts", "adoptedBy")],
    "#/$defs/launch/properties/mountPaths/items/properties/pathMatch": spec.ENUMS[("pathMatch",)],
    "#/$defs/urlBinding/properties/urlSources/items/oneOf/1/properties/scope": spec.ENUMS[("scope",)],
    "#/$defs/mount/oneOf/0/properties/roots/items/properties/how": spec.ENUMS[("how",)],
    "#/$defs/mount/oneOf/0/properties/child/properties/resolves": spec.ENUMS[("resolves",)],
    "#/$defs/mount/oneOf/1/properties/kind": spec.ENUMS[("mountKind",)],
    "#/$defs/embedded/properties/language": spec.ENUMS[("language",)],
    "#/$defs/lifting/properties/companions/items/properties/role": spec.ENUMS[("companion",)],
    "#/$defs/emitStatus": spec.ENUMS[("emitStatus",)] - {"none"},  # an emission is confirmed or possible; `none` is a registry status
    "#/$defs/emission/properties/test/properties/kind": {"unit", "integration", "e2e"},
    "#/$defs/emission/properties/mark/properties/as": {"source", "sink", "sanitizer", "auth-check", "sensitive"},
    "#/$defs/emission/properties/finding/properties/target": {"config", "symbol"},
    "#/$defs/verb": {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT", "*"},
}


def at(pointer: str):
    node = rulepack.load_schema()
    for part in pointer[2:].split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


@pytest.mark.parametrize("pointer", sorted(SPEC_ENUMS))
def test_the_schema_enum_equals_the_specs_table(pointer):
    node = at(pointer)
    assert set(node["enum"]) == set(SPEC_ENUMS[pointer]) and len(node["enum"]) == len(SPEC_ENUMS[pointer])


def test_the_http_verbs_enum_is_spec3_section_6_step_1():
    step1 = re.search(r"\(1\) `VERB` is uppercase, from \{([^}]*)\}", spec.SPEC3.read_text())[1]
    assert set(re.findall(r"[A-Z]+", step1)) | {"*"} == set(at("#/$defs/verb")["enum"])


def test_the_schema_keys_equal_the_specs_tables():
    s = rulepack.load_schema()["$defs"]
    assert set(rulepack.load_schema()["properties"]) == spec.TOP
    assert {k for k in s["matcherObject"]["properties"]} == spec.MATCHER_KEYS | {"ignoreCase"}
    assert set(s["expr"]["properties"]) == spec.EXPR | {"upper", "onUnknown", "default"}
    for name, fields in spec.EMITS.items():
        sub = s["emission"]["properties"][name]
        sub = s[sub["$ref"].rsplit("/", 1)[1]] if "$ref" in sub else sub
        assert set(sub["properties"]) == fields, name
    assert set(s["subjectMatch"]["properties"]) == set(spec.SUBJECT_FIELDS)
    for subject in ("decorator", "config", "launch", "symbol", "marker"):
        assert set(s[f"{subject}Clause"]["properties"]) == spec.SUBJECT_FIELDS[subject], subject
    assert set(s["callsiteClause"]["properties"]) == spec.SUBJECT_FIELDS["callsite"]
    assert set(s["callClause"]["properties"]) == spec.SUBJECT_FIELDS["callsite"] - {"target"}
    assert set(s["decoratorClause"]["properties"]["target"]["properties"]) == {"kind", "name", "parentKind"}


def test_the_when_vocabulary_in_the_schema_equals_the_specs_checker_tables():
    s = rulepack.load_schema()["$defs"]
    for kind, feats in spec.KIND_FEATURES.items():
        props = s[f"when-{kind}"]["properties"]
        assert set(props) == feats, kind
        for feat, sch in props.items():
            if feat in spec.BOOL_FEATURES:
                assert sch == {"type": "boolean"}, feat
            elif feat == "class":
                assert set(sch["enum"]) == spec.CLASSES[kind], kind
            elif feat in spec.FEATURES:
                assert set(sch["enum"]) == spec.FEATURES[feat] if "enum" in sch else sch == {"$ref": "#/$defs/fileRole"}, feat
    assert set(s["when-caps"]["properties"]) == spec.FILE_FEATURES
    assert set(s["formula"]["properties"]) == {"caps", *spec.SEV_PREFIX}
    assert set(s["formula"]["required"]) == set(spec.SEV_PREFIX)
    assert "adjust" in s["formula"]["properties"]["secrets"]["properties"]
    assert all("adjust" not in s["formula"]["properties"][k]["properties"] for k in spec.SEV_PREFIX if k != "secrets")
    assert {s["cap"]["properties"]["max"]["maximum"], s["formula"]["properties"]["injection"]["properties"]["base"]["items"]["properties"]["value"]["maximum"]} == {3}


# ---------------------------------------------------------------- kinds, defaults and status, positive controls

def with_rule(emit, extract, pid="fastmcp"):
    def m(p):
        p["rules"][0]["extract"] = extract
        p["rules"][0]["emit"] = [emit]
    return lambda: accepted(pid, m)


def test_a_declared_vocab_kind_is_a_valid_emission_kind():
    def m(p):
        p["vocab"] = [{"kind": "widget", "keyNorm": "w:<name>", "example": "w:a"}]
        p["rules"][0]["emit"][0]["provides"]["kind"] = "widget"
    assert accepted("fastapi", m) == []


def test_reads_and_writes_of_a_data_entity_take_runtime_built_for_an_unresolved_part():
    reads = {"reads": {"kind": "data_entity", "key": "sql:table:{t}"}}
    assert with_rule(reads, {"t": {"arg": "a0"}})() == []  # reads default is partial: runtime-built is on READS
    writes = {"writes": {"kind": "data_entity", "key": "sql:table:{t}"}}
    assert with_rule(writes, {"t": {"arg": "a0", "onUnknown": "partial"}})() == []  # partial on a non-http kind is runtime-built, on WRITES
    assert with_rule(writes, {"t": {"arg": "a0", "onUnknown": "config-derived"}})()  # config-derived is CONSUMES topic / PROVIDES topic only


def test_confirmed_is_allowed_when_the_key_slot_is_always_resolved():
    consumes = lambda extract: with_rule({"consumes": {"kind": "mcp_tool", "key": "{t}", "status": "confirmed"}}, extract)()
    assert consumes({"t": {"target": "name"}}) == []
    assert consumes({"t": {"literal": "x"}}) == []
    assert consumes({"t": {"firstOf": [{"arg": "a0"}, {"target": "name"}]}}) == []
    assert consumes({"t": {"arg": "a0"}})  # consumes defaults to partial: the name may be unresolved
    assert consumes({"t": {"firstOf": [{"arg": "a0"}, {"arg": "kw.name"}]}})
    assert consumes({"t": {"arg": "a0", "onUnknown": "drop"}}) == []


def test_a_registry_row_with_another_key_set_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "reasons.yaml"
    bad.write_text('- {code: x, status: possible, "on": [Link], kinds: [http]}\n')
    monkeypatch.setattr(rulepack, "REGISTRY_PATH", bad)
    with pytest.raises(rulepack.PackError):
        rulepack.load_registry_rows()


def test_a_parameterized_code_needs_a_non_empty_parameter():
    assert rulepack.reason_row("flow-unavailable:") is None
    assert rulepack.reason_row("flow-unavailable") is None


# ---------------------------------------------------------------- the schema states its types (a dropped `type` would otherwise go unprobed)

def schema_nodes(node, ptr="#", role="root"):
    """Every subschema reached through the keywords that hold schemas, with the role it plays."""
    yield ptr, node, role
    for key in ("properties", "$defs"):
        for k, v in node.get(key, {}).items():
            yield from schema_nodes(v, f"{ptr}/{key}/{k}", "value")
    for k, v in node.get("dependentSchemas", {}).items():
        yield from schema_nodes(v, f"{ptr}/dependentSchemas/{k}", "cond")
    if isinstance(node.get("items"), dict):
        yield from schema_nodes(node["items"], f"{ptr}/items", "value")
    if isinstance(node.get("additionalProperties"), dict):
        yield from schema_nodes(node["additionalProperties"], f"{ptr}/additionalProperties", "value")
    if "propertyNames" in node:
        yield from schema_nodes(node["propertyNames"], f"{ptr}/propertyNames", "names")
    for key in ("oneOf", "anyOf"):
        for i, v in enumerate(node.get(key, [])):
            yield from schema_nodes(v, f"{ptr}/{key}/{i}", "alt")


def test_every_schema_node_states_its_type_and_what_the_type_implies():
    seen = 0
    for ptr, s, role in schema_nodes(rulepack.load_schema()):
        if role == "value" or (role == "alt" and set(s) != {"required"}):
            assert {"type", "enum", "const", "$ref", "oneOf", "anyOf"} & set(s), f"{ptr} constrains nothing"
        wants = [("array", {"items", "minItems", "maxItems"}), ("object", {"properties", "minProperties", "maxProperties", "additionalProperties"}),
                 ("integer", {"minimum", "maximum"}), ("string", {"pattern", "minLength", "maxLength"})]
        for typ, keys in wants:
            if keys & set(s) and role not in ("names", "cond") and ptr != "#":
                assert s.get("type") == typ, f"{ptr} uses {sorted(keys & set(s))} without type {typ}"
        seen += 1
    assert seen > 250


# ---------------------------------------------------------------- boundaries the spec states (an oracle independent of the schema under test)

def set_at(pack, path, value):
    node = pack
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value


def verdicts(pid, path, values, extra=None):
    out = {}
    for v in values:
        p = copy.deepcopy(PACKS[pid])
        if extra:
            extra(p)
        set_at(p, path, v)
        out[repr(v)] = bool(errors(p, negatives=NEGATIVE_LINES if pid == "infra-signatures" else None))
    return out


# (pid, path, valid values, invalid values): spec 1.1 id, 1.2 rule id, 1.6 numbers, 9.1 four levels and 0..3 weights
BOUNDARIES = {
    "pack id": ("fastapi", ("id",), ["a", "a" + "b" * 39, "a-b1"], ["", "A", "1a", "a" + "b" * 40, "a_b", "a b", "-a"]),
    "rule id": ("fastapi", ("rules", 0, "id"), ["r", "r-1", "route2"], ["", "R", "1r", "r_1", "r b", "-r"]),
    "mount id": ("fastapi", ("mounts", 0, "id"), ["m", "m-1"], ["", "M", "1m", "m_1"]),
    "regex group": ("fastapi", ("rules", 0, "extract", "verb", "group"), [0, 1, 9], [-1, 10, "1", 1.5, True]),
    "port": ("fastmcp", ("launch", "frameworkDefaults", 0, "port"), [1, 8000, 65535], [0, -1, 65536, "8000", 1.5]),
    "argument index": ("fastmcp", ("launch", "targetArg", 0, "index"), [0, 1, 99], [-1, 100, "0", 0.5]),
    "base weight": ("severity-rules", ("formula", "injection", "base", 0, "value"), [0, 1, 3], [-1, 4, "3", 2.5]),
    "penalty weight": ("severity-rules", ("formula", "injection", "penalties", 0, "value"), [0, 3], [-1, 4]),
    "adjust weight": ("severity-rules", ("formula", "secrets", "adjust", 0, "value"), [0, 3], [-1, 4]),
    "cap max": ("severity-rules", ("formula", "caps", 0, "max"), [0, 3], [-1, 4]),
    "severity row id": ("severity-rules", ("rows", 0, "id"), ["INJ-1"], ["inj-1", "INJ-0", "INJ-5", "XYZ-1", "INJ-", "INJ-1x", "xINJ-1"]),
    "query column": ("paper-inhouse-bus", ("queries", 0, "yield"), [{"handler": "$h.id"}], [{}, {"Handler": "$h.id"}, {"h-x": "$h.id"}, {"handler": "h.id"},
                                                                                        {"handler": "$h"}, {"handler": "$H.id"}, {"handler": "$h.id."}, {"handler": 1}],
                     lambda p: p["rules"].pop(1)),
    "signature id": ("infra-signatures", ("signatures", 0, "id"), ["python-connection-refused"], ["Python-connection-refused", "python", "-connection-refused", "pythonconnection-refused",
                                                                                                  "python-Connection", "go-connection-refused-x"]),
    "signature regex": ("infra-signatures", ("signatures", 0, "regex"), ["ConnectionRefusedError"], [""]),
    "transport": ("kafka", ("vocab", 0, "transport"), ["kafka", "redis-stream", "a1"], ["", "Kafka", "1k", "k_1", "k b"]),
    "pack version": ("fastapi", ("version",), ["0.0.0", "1.2.3"], ["", "1", "1.2", "a.b.c"]),
}


@pytest.mark.parametrize("name", sorted(BOUNDARIES))
def test_boundaries_accept_the_spec_ranges_and_reject_outside_them(name):
    pid, path, good, bad, *extra = BOUNDARIES[name]
    assert not any(verdicts(pid, path, good, *extra).values()), verdicts(pid, path, good, *extra)
    assert all(verdicts(pid, path, bad, *extra).values()), verdicts(pid, path, bad, *extra)


def aliased_query(alias, ref="$dec.args.a0"):
    """Two bindings: `dec` (yielded) and a second one named `alias`, joined to it but never yielded, so only the
    schema's `as` pattern can reject a malformed alias (a yielded alias is also checked by the yield pattern: P4-2)."""
    def m(p):
        p["queries"][0] = {"id": "handler-ns",
                           "from": [{"as": "dec", "record": "decorator", "where": {}},
                                    {"as": alias, "record": "symbol", "where": {"id": "$dec.targetId"}}],
                           "yield": {"handler": "$dec.targetId", "topic": ref}}
        del p["rules"][1]  # the subscribe rule reads the columns this query does not yield
    return m


@pytest.mark.parametrize("alias,ok", [("dec2", True), ("d_1", True), ("", False), ("Dec", False), ("TY", False), ("1d", False),
                                      ("d-x", False), ("d x", False), ("t y", False), ("_d", False), ("h2\n", False)])
def test_a_query_alias_is_a_lowercase_token(alias, ok):
    assert (accepted("paper-inhouse-bus", aliased_query(alias)) == []) is ok


@pytest.mark.parametrize("ref,ok", [("$dec.args.a0", True), ("$dec.args.kw.x", True), ("$dec.targetId", True),
                                    ("xdec.args.a0", False), ("xdec.args.a0$h.id", False), ("$dec.args.a0!! junk", False), ("$dec.args.a0 junk", False),
                                    ("$dec.args.", False), ("$dec.args.a0\n", False), ("$Dec.id", False), ("dec", False), ("$dec", False)])
def test_a_query_yield_reference_is_a_dollar_alias_field_path(ref, ok):
    """P4-2: the validator's `args.` branch checks only that the record has `args`, so the schema pattern is the only guard."""
    assert (accepted("paper-inhouse-bus", aliased_query("h", ref)) == []) is ok


@pytest.mark.parametrize("bad", ["arg0", "a-1", "p0", "r", "ret", "a", "kw.", "kw", ".a0", "a0.", "kw.1x", "recv-x", "A0", "a0 b", ""])
def test_argument_paths_outside_spec_5_1_are_rejected_and_the_spec_agrees(bad):
    assert not re.fullmatch(spec.ARGPATH, bad)
    assert accepted("fastapi", lambda p: p["rules"][0]["extract"].update(t={"arg": bad}))


@pytest.mark.parametrize("bad", ["kw.p\u00e9", "kw.\u00e9x", "recv.f\u00e9"], ids=["w-in-kw", "w-leading", "w-in-recv-member"])
def test_p5_3_a_non_ascii_word_character_is_rejected_as_the_ecma_schema_would(bad):
    """JSON Schema `pattern` is ECMA 262: `\\w` is ASCII-only. Python's str regex is Unicode unless told otherwise."""
    assert accepted("fastapi", lambda p: p["rules"][0]["extract"].update(t={"arg": bad}))


@pytest.mark.parametrize("bad", ["a1\u0663", "a\u0663", "a12\u0663"], ids=["d-after-ascii", "d-only", "d-trailing"])
def test_p5_3_a_non_ascii_digit_is_rejected_as_the_ecma_schema_would(bad):
    assert accepted("fastapi", lambda p: p["rules"][0]["extract"].update(t={"arg": bad}))


def test_p5_3_a_non_ascii_digit_in_a_version_is_rejected():
    assert accepted("fastapi", lambda p: p.update(version="1.0.\u0663"))


@pytest.mark.parametrize("good", ["a0", "a12", "kw.x", "kw.x.y", "recv", "recv.f", "a0.b.c", "kw._p", "a1.method"])
def test_argument_paths_of_spec_5_1_are_accepted(good):
    assert re.fullmatch(spec.ARGPATH, good)
    assert accepted("fastapi", lambda p: p["rules"][0]["extract"].update(t={"arg": good})) == []


@pytest.mark.parametrize("bad", [{"Widget": 1}, 0])
def test_a_vocab_kind_must_be_a_lowercase_token(bad):
    for kind in ("Widget", "1w", "w-x", "w x", ""):
        assert accepted("fastapi", lambda p, kind=kind: p.update(vocab=[{"kind": kind, "keyNorm": "k", "example": "e"}]))
    for kind in ("w", "w_1"):
        assert accepted("fastapi", lambda p, kind=kind: p.update(vocab=[{"kind": kind, "keyNorm": "k", "example": "e"}])) == []


EXACTLY_ONE = [
    ("empty emission", "fastapi", lambda p: p["rules"][0]["emit"].__setitem__(0, {})),
    ("two emissions in one object", "fastapi", lambda p: p["rules"][0]["emit"][0].update(consumes={"kind": "http", "key": "x"})),
    ("empty match", "fastapi", lambda p: p["rules"][0].update(match={})),
    ("anyOf beside a subject", "fastapi", lambda p: p["rules"][0]["match"].update(anyOf=[{"callsite": {"callee": "x"}}])),
    ("anyOf clause with two subjects", "fastapi", lambda p: p["rules"][0].update(match={"anyOf": [{"callsite": {"callee": "x"}, "symbol": {"kind": "function"}}]})),
    ("anyOf inside anyOf", "fastapi", lambda p: p["rules"][0].update(match={"anyOf": [{"anyOf": [{"callsite": {"callee": "x"}}]}]})),
    ("empty expression", "fastapi", lambda p: p["rules"][0]["extract"].update(t={})),
    ("two expressions in one object", "fastapi", lambda p: p["rules"][0]["extract"].update(t={"arg": "a0", "literal": "x"})),
    ("upper without group", "fastapi", lambda p: p["rules"][0]["extract"].update(t={"literal": "x", "upper": True})),
    ("upper false", "fastapi", lambda p: p["rules"][0]["extract"]["verb"].update(upper=False)),
    ("empty decorator clause", "fastapi", lambda p: p["rules"][0].update(match={"decorator": {}})),
    ("empty decorator target", "fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(target={})),
    ("empty call clause", "fastapi", lambda p: p["mounts"][0].update(call={})),
    ("empty callsite clause", "fetch", lambda p: p["rules"][0].update(match={"callsite": {}})),
    ("empty config clause", "fastmcp", lambda p: p["urlBindings"][0]["urlSources"][0].update(config={})),
    ("empty launch clause", "fastapi", lambda p: p["rules"][0].update(match={"launch": {}})),
    ("empty symbol clause", "fastapi", lambda p: p["rules"][0].update(match={"symbol": {}})),
    ("empty marker clause", "fastapi", lambda p: p["rules"][0].update(match={"marker": {}})),
    ("empty when", "severity-rules", lambda p: p["formula"]["injection"]["base"][0].update(when={})),
    ("empty cap when", "severity-rules", lambda p: p["formula"]["caps"][0].update(when={})),
    ("empty matcher object", "fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(name={})),
    ("non-string matcher operand", "fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(name={"exact": 1})),
    ("empty any-of matcher list", "fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(name=[])),
    ("empty from", "paper-inhouse-bus", lambda p: p["queries"][0].update(**{"from": []})),
    ("from with five bindings", "paper-inhouse-bus", lambda p: p["queries"][0]["from"].append({"as": "z", "record": "symbol", "where": {"id": "$h.parentId"}})),
    ("empty seeds", "fetch", lambda p: p["liftings"][0].update(seeds=[])),
    ("empty flags", "fastmcp", lambda p: p["launch"]["portFlags"][0].update(flags=[])),
    ("empty kinds", "fastmcp", lambda p: p["urlBindings"][0].update(kinds=[])),
    ("empty urlSources", "fastmcp", lambda p: p["urlBindings"][0].update(urlSources=[])),
    ("empty base", "severity-rules", lambda p: p["formula"]["injection"]["base"].clear()),
    ("empty emit", "fastapi", lambda p: p["rules"][0].update(emit=[])),
    ("empty firstOf", "fastapi", lambda p: p["rules"][0]["extract"].update(t={"firstOf": []})),
    ("firstOf entry empty", "fastapi", lambda p: p["rules"][0]["extract"].update(t={"firstOf": [{}]})),
    ("urlSource with config and callsite", "fastmcp", lambda p: p["urlBindings"][0]["urlSources"][0].update(callsite={"callee": "x"}, arg="a0")),
    ("urlSource callsite without arg", "fastmcp", lambda p: p["urlBindings"][0]["urlSources"][1].pop("arg")),
    ("mount with call and class-decorator kind", "fastapi", lambda p: p["mounts"][0].update(kind="class-decorator")),
    ("class-decorator mount without decorator", "fastapi", lambda p: p.update(mounts=[{"id": "c", "kind": "class-decorator", "prefix": {"arg": "a0"}}])),
    ("class-decorator mount with a call", "fastapi", lambda p: p.update(mounts=[{"id": "c", "kind": "class-decorator", "decorator": {"name": "X"}, "prefix": {"arg": "a0"}, "call": {"callee": "x"}}])),
    ("mount without call", "fastapi", lambda p: p["mounts"][0].pop("call")),
    ("mount without id", "fastapi", lambda p: p["mounts"][0].pop("id")),
    ("root without how", "fastapi", lambda p: p["mounts"][0]["roots"][0].pop("how")),
    ("constructorPrefix without arg", "fastapi", lambda p: p["mounts"][0]["constructorPrefix"][0].pop("arg")),
    ("unmounted without reason", "fastapi", lambda p: p["mounts"][0]["unmounted"].pop("reason")),
    ("child without resolves", "fastapi", lambda p: p["mounts"][0]["child"].pop("resolves")),
    ("prefix without arg", "fastapi", lambda p: p["mounts"][0]["prefix"].pop("arg")),
    ("lifting without role", "fetch", lambda p: p["liftings"][0].pop("role")),
    ("lifting role without slot", "fetch", lambda p: p["liftings"][0]["role"].pop("slot")),
    ("companion without arg", "fetch", lambda p: p["liftings"][0]["companions"][0].pop("arg")),
    ("seed without call", "fetch", lambda p: p["liftings"][0]["seeds"][0].pop("call")),
    ("rule without match", "fastapi", lambda p: p["rules"][0].pop("match")),
    ("rule without id", "fastapi", lambda p: p["rules"][0].pop("id")),
    ("emission without key", "fastapi", lambda p: p["rules"][0]["emit"][0]["provides"].pop("key")),
    ("emission without kind", "fastapi", lambda p: p["rules"][0]["emit"][0]["provides"].pop("kind")),
    ("args entry without arg", "fastapi", lambda p: p["rules"][0]["match"]["decorator"].update(args=[{"value": "x"}])),
    ("scope import without module", "fastapi", lambda p: p["rules"][0]["scope"]["imports"][0].pop("module")),
    ("frameworkDefault without protocol", "fastmcp", lambda p: p["launch"]["frameworkDefaults"][0].pop("protocol")),
    ("frameworkDefault without port", "fastmcp", lambda p: p["launch"]["frameworkDefaults"][0].pop("port")),
    ("frameworkDefault without command", "fastmcp", lambda p: p["launch"]["frameworkDefaults"][0].pop("command")),
    ("portFlags without flags", "fastmcp", lambda p: p["launch"]["portFlags"][0].pop("flags")),
    ("portFlags without command", "fastmcp", lambda p: p["launch"]["portFlags"][0].pop("command")),
    ("targetArg without index", "fastmcp", lambda p: p["launch"]["targetArg"][0].pop("index")),
    ("targetArg without form", "fastmcp", lambda p: p["launch"]["targetArg"][0].pop("form")),
    ("envPorts without envVar", "fastmcp", lambda p: p["launch"]["envPorts"][0].pop("envVar")),
    ("envPorts without adoptedBy", "fastmcp", lambda p: p["launch"]["envPorts"][0].pop("adoptedBy")),
    ("mountPaths without call", "fastmcp", lambda p: p["launch"]["mountPaths"][0].pop("call")),
    ("mountPaths without path", "fastmcp", lambda p: p["launch"]["mountPaths"][0].pop("path")),
    ("mountPaths without app", "fastmcp", lambda p: p["launch"]["mountPaths"][0].pop("app")),
    ("mountPaths without pathMatch", "fastmcp", lambda p: p["launch"]["mountPaths"][0].pop("pathMatch")),
    ("mountPaths app without fromCall", "fastmcp", lambda p: p["launch"]["mountPaths"][0]["app"].pop("fromCall")),
    ("mountPaths app without arg", "fastmcp", lambda p: p["launch"]["mountPaths"][0]["app"].pop("arg")),
    ("urlBinding without id", "fastmcp", lambda p: p["urlBindings"][0].pop("id")),
    ("urlBinding without kinds", "fastmcp", lambda p: p["urlBindings"][0].pop("kinds")),
    ("urlBinding without urlSources", "fastmcp", lambda p: p["urlBindings"][0].pop("urlSources")),
    ("vocab without kind", "kafka", lambda p: p["vocab"][0].pop("kind")),
    ("vocab without example", "kafka", lambda p: p["vocab"][0].pop("example")),
    ("link without kind", "fastapi", lambda p: p["link"][0].pop("kind")),
    ("link without matcher", "fastapi", lambda p: p["link"][0].pop("matcher")),
    ("query without id", "paper-inhouse-bus", lambda p: p["queries"][0].pop("id")),
    ("query without from", "paper-inhouse-bus", lambda p: p["queries"][0].pop("from")),
    ("query without yield", "paper-inhouse-bus", lambda p: p["queries"][0].pop("yield")),
    ("binding without as", "paper-inhouse-bus", lambda p: p["queries"][0]["from"][0].pop("as")),
    ("binding without record", "paper-inhouse-bus", lambda p: p["queries"][0]["from"][0].pop("record")),
    ("binding without where", "paper-inhouse-bus", lambda p: p["queries"][0]["from"][0].pop("where")),
    ("runtime entry without regex", "infra-signatures", lambda p: p["runtimes"][0].pop("regex")),
    ("runtime entry without runtime", "infra-signatures", lambda p: p["runtimes"][0].pop("runtime")),
    ("signature without sample", "infra-signatures", lambda p: p["signatures"][0].pop("sample")),
    ("signature without regex", "infra-signatures", lambda p: p["signatures"][0].pop("regex")),
    ("signature without class", "infra-signatures", lambda p: p["signatures"][0].pop("class")),
    ("signature without runtime", "infra-signatures", lambda p: p["signatures"][0].pop("runtime")),
    ("signature without id", "infra-signatures", lambda p: p["signatures"][0].pop("id")),
    ("cap without max", "severity-rules", lambda p: p["formula"]["caps"][0].pop("max")),
    ("cap without when", "severity-rules", lambda p: p["formula"]["caps"][0].pop("when")),
    ("formula entry without value", "severity-rules", lambda p: p["formula"]["injection"]["base"][0].pop("value")),
    ("formula entry without when", "severity-rules", lambda p: p["formula"]["injection"]["base"][0].pop("when")),
    ("formula kind without base", "severity-rules", lambda p: p["formula"]["injection"].pop("base")),
    ("row without level", "severity-rules", lambda p: p["rows"][0].pop("level")),
    ("row without rank", "severity-rules", lambda p: p["rows"][0].pop("rank")),
    ("row without finding", "severity-rules", lambda p: p["rows"][0].pop("finding")),
    ("row without id", "severity-rules", lambda p: p["rows"][0].pop("id")),
    ("severity pack without formula", "severity-rules", lambda p: p.pop("formula")),
    ("severity pack without rows", "severity-rules", lambda p: p.pop("rows")),
    ("infra pack without runtimes", "infra-signatures", lambda p: p.pop("runtimes")),
    ("infra pack without signatures", "infra-signatures", lambda p: p.pop("signatures")),
    ("severity pack with runtimes", "severity-rules", lambda p: p.update(runtimes=[])),
    ("severity pack with signatures", "severity-rules", lambda p: p.update(signatures=[])),
    ("infra pack with formula", "infra-signatures", lambda p: p.update(formula=PACKS["severity-rules"]["formula"])),
    ("infra pack with rows", "infra-signatures", lambda p: p.update(rows=[])),
    ("framework pack with runtimes", "fastapi", lambda p: p.update(runtimes=[])),
    ("framework pack with signatures", "fastapi", lambda p: p.update(signatures=[])),
    ("framework pack with rows", "fastapi", lambda p: p.update(rows=[])),
    ("embedded without language", "fastapi", lambda p: p.update(embedded=[{"call": {"callee": "x"}, "arg": "a0"}])),
    ("embedded without call", "fastapi", lambda p: p.update(embedded=[{"arg": "a0", "language": "sql"}])),
    ("embedded without arg", "fastapi", lambda p: p.update(embedded=[{"call": {"callee": "x"}, "language": "sql"}])),
]


@pytest.mark.parametrize("name,pid,mutate", EXACTLY_ONE, ids=[n for n, *_ in EXACTLY_ONE])
def test_a_malformed_shape_is_rejected(name, pid, mutate):
    assert accepted(pid, mutate), f"{name} was accepted"


# ---------------------------------------------------------------- constructs none of the seven spec packs uses

SYNTH = {
    "packFormat": 1, "id": "synth", "version": "1.0.0", "kind": "security", "vocabVersion": "1.0",
    "vocab": [{"kind": "widget", "keyNorm": "w:<name>", "example": "w:a", "transport": "inhouse-synth"}],
    "link": [{"kind": "widget", "matcher": "regex"}],
    "mounts": [{"id": "cls", "kind": "class-decorator", "decorator": {"name": {"exact": "Controller"}}, "prefix": {"arg": "a0"},
                "scope": {"path": {"prefix": "src"}, "notPath": {"glob": "**/gen/**"}, "role": ["primary"]}, "recordConditions": True}],
    "embedded": [{"call": {"callee": "execute"}, "arg": "a0", "language": "sql", "scope": {"imports": [{"module": "db", "names": ["connect"]}]}}],
    "rules": [{
        "id": "everything",
        "match": {"anyOf": [{"callsite": {"callee": "x", "args": [{"arg": "a0", "value": {"glob": "a*"}}], "target": "t"}},
                            {"symbol": {"kind": "function"}}, {"config": {"keyPath": "k"}}, {"marker": {"framework": "pytest"}},
                            {"launch": {"command": {"basename": "x"}, "args": [{"arg": "a0"}]}}, {"decorator": {"name": "d", "target": {"kind": "function"}}}]},
        "scope": {"role": ["primary", "example", "doc"], "imports": [{"module": {"prefix": "db"}}]},
        "extract": {"v": {"field": "name"}, "w": {"config": "keyPath", "default": "dflt"}, "x": {"arg": "a0", "onUnknown": "runtime-built"}},
        "emit": [{"test": {"kind": "unit"}}, {"mark": {"as": "sink", "class": "exec"}},
                 {"finding": {"finding": "injection", "class": "host-compromise", "target": "config"}}, {"launches": {"target": "x"}},
                 {"reads": {"kind": "data_entity", "key": "sql:table:{v}"}}, {"writes": {"kind": "widget", "key": "w:{w}", "keyForm": "glob"}},
                 {"consumes": {"kind": "widget", "key": "w:{x}", "status": "possible", "reason": "runtime-built"}}],
    }],
}
EMPTY_STRING_LEAVES = [
    ("vocab", 0, "keyNorm"), ("vocab", 0, "example"), ("rules", 0, "emit", 1, "mark", "class"), ("rules", 0, "emit", 2, "finding", "class"),
    ("rules", 0, "emit", 3, "launches", "target"), ("rules", 0, "emit", 4, "reads", "key"), ("rules", 0, "emit", 6, "consumes", "reason"),
    ("rules", 0, "extract", "v", "field"),
]


def synth(mutate=None):
    p = copy.deepcopy(SYNTH)
    if mutate:
        mutate(p)
    return errors(p)


def test_a_pack_using_every_emission_expression_and_subject_validates():
    assert synth() == []


@pytest.mark.parametrize("path", EMPTY_STRING_LEAVES, ids=[".".join(map(str, p)) for p in EMPTY_STRING_LEAVES])
def test_an_empty_string_where_text_is_required_is_rejected(path):
    assert synth(lambda p: set_at(p, path, ""))


@pytest.mark.parametrize("name,mutate", [
    ("test kind outside unit/integration/e2e", lambda p: p["rules"][0]["emit"][0]["test"].update(kind="smoke")),
    ("mark as outside the five", lambda p: p["rules"][0]["emit"][1]["mark"].update(**{"as": "taint"})),
    ("finding outside the four", lambda p: p["rules"][0]["emit"][2]["finding"].update(finding="latency")),
    ("finding target outside config/symbol", lambda p: p["rules"][0]["emit"][2]["finding"].update(target="file")),
    ("mark without class", lambda p: p["rules"][0]["emit"][1]["mark"].pop("class")),
    ("finding without target", lambda p: p["rules"][0]["emit"][2]["finding"].pop("target")),
    ("launches without target", lambda p: p["rules"][0]["emit"][3]["launches"].pop("target")),
    ("unknown key in a launches emission", lambda p: p["rules"][0]["emit"][3]["launches"].update(extra=1)),
    ("embedded language outside sql/cypher", lambda p: p["embedded"][0].update(language="mongo")),
    ("scope role outside File.role", lambda p: p["rules"][0]["scope"].update(role=["test"])),
    ("keyForm outside the four", lambda p: p["rules"][0]["emit"][5]["writes"].update(keyForm="fuzzy")),
    ("config expression value", lambda p: p["rules"][0]["extract"].update(w={"config": "value"})),
    ("class-decorator prefix without arg", lambda p: p["mounts"][0]["prefix"].pop("arg")),
    ("class-decorator mount with a child", lambda p: p["mounts"][0].update(child={"arg": "a0", "resolves": "factory-call"})),
    ("recordConditions not a boolean", lambda p: p["mounts"][0].update(recordConditions="yes")),
    ("link matcher fuzzy", lambda p: p["link"][0].update(matcher="fuzzy")),
    ("reason not on a reads of data_entity", lambda p: p["rules"][0]["emit"][4]["reads"].update(reason="key-match")),
    ("a declared kind used under another id", lambda p: p["rules"][0]["emit"][5]["writes"].update(kind="gadget")),
], ids=lambda x: x if isinstance(x, str) else "")
def test_synthetic_malformed_shapes_are_rejected(name, mutate):
    assert synth(mutate), name


@pytest.mark.parametrize("pid,path", [
    ("fastapi", ("rules", 0, "emit", 0, "provides", "key")), ("fastapi", ("rules", 0, "emit", 0, "provides", "reason")),
    ("fastapi", ("mounts", 0, "unmounted", "reason")), ("fastmcp", ("launch", "frameworkDefaults", 0, "protocol")),
    ("fastmcp", ("launch", "portFlags", 0, "flags", 0)), ("fetch", ("liftings", 0, "role", "slot")),
    ("infra-signatures", ("signatures", 6, "library")), ("paper-inhouse-bus", ("rules", 1, "extract", "ns", "yield")),
    ("fetch", ("rules", 0, "emit", 0, "consumes", "key")),
], ids=lambda x: x if isinstance(x, str) else ".".join(map(str, x)))
def test_an_empty_string_in_the_other_required_text_fields_is_rejected(pid, path):
    assert accepted(pid, lambda p: set_at(p, path, ""))


# ---------------------------------------------------------------- differential: at least as strict as the spec's own checker

def spec_rejects(pack: dict) -> bool:
    try:
        spec.check_pack(pack)
    except Exception:  # an AssertionError, or the KeyError/TypeError of a missing or mistyped key: all are rejections
        return True
    return False


def single_edits(pack: dict):
    """Every mutant with one key deleted, or one scalar replaced by a value of no valid type or range."""
    for path in spec.dict_positions(pack):
        node = pack
        for step in path:
            node = node[step]
        for key in list(node):
            mutant = copy.deepcopy(pack)
            target = mutant
            for step in path:
                target = target[step]
            del target[key]
            yield ("delete", path, key), mutant
    for path in leaf_paths(pack):
        for shape in OUT_OF_RANGE:
            mutant = copy.deepcopy(pack)
            set_at(mutant, path, shape)
            yield ("replace", path, shape), mutant


def test_the_validator_rejects_every_single_edit_the_specs_own_checker_rejects():
    checked = rejected_by_spec = 0
    missed = []
    for pid, pack in PACKS.items():
        negs = NEGATIVE_LINES if pid == "infra-signatures" else None
        for what, mutant in single_edits(pack):
            checked += 1
            if spec_rejects(mutant):
                rejected_by_spec += 1
                if not errors(mutant, negatives=negs):
                    missed.append((pid, what))
    assert checked > 3000 and rejected_by_spec > 1000
    assert missed == [], missed[:8]


def test_the_differential_probe_can_fail(monkeypatch):
    """Against a validator that accepts everything, the probe lists the edits the spec checker rejects: it is not vacuous."""
    monkeypatch.setattr(sys.modules[__name__], "errors", lambda *a, **k: [])
    pack = PACKS["fastapi"]
    gone = [w for w, m in single_edits(pack) if spec_rejects(m) and not errors(m)]
    assert len(gone) > 100


# ---------------------------------------------------------------- targeted gaps found by mutating the schema and the validator

def fresh_query(bindings, yield_):
    def m(p):
        p["queries"][0] = {"id": "handler-ns", "from": bindings, "yield": yield_}
        del p["rules"][1]
    return m


def test_a_query_alias_bound_twice_is_rejected_on_that_ground_alone():
    twice = [{"as": "a", "record": "decorator", "where": {}}, {"as": "a", "record": "symbol", "where": {"id": "$a.targetId"}}]
    assert accepted("paper-inhouse-bus", fresh_query(twice, {"handler": "$a.id"}))
    once = [twice[0], {**twice[1], "as": "b"}]
    assert accepted("paper-inhouse-bus", fresh_query(once, {"handler": "$b.id"})) == []


def test_a_join_to_a_field_the_aliased_record_lacks_is_rejected_though_it_is_an_id_field():
    lacks = [{"as": "a", "record": "decorator", "where": {}}, {"as": "b", "record": "symbol", "where": {"id": "$a.parentId"}}]
    assert accepted("paper-inhouse-bus", fresh_query(lacks, {"handler": "$b.id"}))


def test_the_query_record_enum_is_the_bundle_record_types_less_header_and_secret_sightings():
    bundle = {r for r in json.loads((CPG / "ccg/schemas/bundle-v1.schema.json").read_text())["properties"]["type"]["enum"]}
    got = set(at("#/$defs/query/properties/from/items/properties/record")["enum"])
    assert got == bundle - {"header", "secretSighting"}


@pytest.mark.parametrize("kind", sorted(spec.SEV_PREFIX))
def test_every_formula_section_of_every_kind_rejects_a_missing_or_out_of_range_entry_field(kind):
    base = PACKS["severity-rules"]["formula"][kind]
    for section in (s for s in ("base", "adjust", "penalties") if s in base):
        for mutate in (lambda e: e.pop("value"), lambda e: e.pop("when"), lambda e: e.update(value=-1), lambda e: e.update(value=4),
                       lambda e: e.update(when={}), lambda e: e.update(extra=1)):
            assert accepted("severity-rules", lambda p, mutate=mutate, section=section: mutate(p["formula"][kind][section][0])), (kind, section)
        emptied = accepted("severity-rules", lambda p, section=section: p["formula"][kind].update({section: []}))
        assert bool(emptied) is (section != "penalties"), (kind, section)  # no penalties is a valid rule; no base or adjust is not
    assert accepted("severity-rules", lambda p: p["formula"][kind].pop("base"))


def test_a_formula_entry_may_hold_the_boundary_weights():
    for v in (0, 3):
        assert accepted("severity-rules", lambda p, v=v: [e.update(value=v) for k in spec.SEV_PREFIX for s in ("base", "adjust", "penalties")
                                                          for e in p["formula"][k].get(s, [])]) == []


def test_a_row_rank_below_zero_is_rejected():
    assert accepted("severity-rules", lambda p: p["rows"][0].update(rank=-1, id="INJ-5"))


def test_an_empty_signature_regex_is_rejected_without_the_negatives_too():
    p = copy.deepcopy(PACKS["infra-signatures"])
    p["signatures"][0]["regex"] = ""
    assert rulepack.pack_errors(p, dirname="infra-signatures")


@pytest.mark.parametrize("name,mutate", [
    ("urlSource with neither config nor callsite", lambda p: p["urlBindings"][0]["urlSources"].__setitem__(0, {})),
    ("urlSource with only an arg", lambda p: p["urlBindings"][0]["urlSources"].__setitem__(0, {"arg": "a0"})),
    ("urlSource with only a callsite", lambda p: p["urlBindings"][0]["urlSources"].__setitem__(0, {"callsite": {"callee": "x"}})),
    ("urlSource with a scope and no callsite", lambda p: p["urlBindings"][0]["urlSources"].__setitem__(0, {"scope": "file", "arg": "a0"})),
    ("child without arg", lambda p: p["mounts"][0]["child"].pop("arg")),
    ("root without call", lambda p: p["mounts"][0]["roots"][0].pop("call")),
    ("constructorPrefix without call", lambda p: p["mounts"][0]["constructorPrefix"][0].pop("call")),
    ("unmounted without status", lambda p: p["mounts"][0]["unmounted"].pop("status")),
    ("lifted clause without lifting", lambda p: p["rules"].__setitem__(0, {**p["rules"][0], "match": {"lifted": {}}})),
    ("lifting without id", lambda p: p["liftings"][0].pop("id")),
    ("lifting without seeds", lambda p: p["liftings"][0].pop("seeds")),
    ("lifting role without kind", lambda p: p["liftings"][0]["role"].pop("kind")),
    ("companion without role", lambda p: p["liftings"][0]["companions"][0].pop("role")),
    ("rule without emit", lambda p: p["rules"][0].pop("emit")),
], ids=lambda x: x if isinstance(x, str) else "")
def test_more_required_keys(name, mutate):
    pid = "fastmcp" if name.startswith("urlSource") else "fetch" if name.startswith(("lift", "companion")) else "fastapi"
    if name == "lifted clause without lifting":
        pid = "fetch"
    assert accepted(pid, mutate), name


@pytest.mark.parametrize("name,mutate", [
    ("mark without as", lambda p: p["rules"][0]["emit"][1]["mark"].pop("as")),
    ("finding without finding", lambda p: p["rules"][0]["emit"][2]["finding"].pop("finding")),
    ("finding without class", lambda p: p["rules"][0]["emit"][2]["finding"].pop("class")),
    ("test without kind", lambda p: p["rules"][0]["emit"][0]["test"].pop("kind")),
    ("class-decorator mount without id", lambda p: p["mounts"][0].pop("id")),
    ("class-decorator mount without kind", lambda p: p["mounts"][0].pop("kind")),
    ("class-decorator mount without prefix", lambda p: p["mounts"][0].pop("prefix")),
    ("class-decorator mount with an extra key", lambda p: p["mounts"][0].update(extra=1)),
    ("reads without key", lambda p: p["rules"][0]["emit"][4]["reads"].pop("key")),
    ("writes without kind", lambda p: p["rules"][0]["emit"][5]["writes"].pop("kind")),
    ("consumes without key", lambda p: p["rules"][0]["emit"][6]["consumes"].pop("key")),
    ("consumes without kind", lambda p: p["rules"][0]["emit"][6]["consumes"].pop("kind")),
], ids=lambda x: x if isinstance(x, str) else "")
def test_more_required_keys_of_constructs_no_spec_pack_uses(name, mutate):
    assert synth(mutate), name


# ---------------------------------------------------------------- S31 fix round (Pass 4)

def test_every_data_file_under_ccg_is_covered_by_a_package_data_glob():
    """P4-1: an installed (non-editable) wheel ships exactly the globs of pyproject's package-data; the expected set is the tree."""
    import tomllib
    declared = tomllib.loads((CPG / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["ccg"]
    pkg = CPG / "ccg"
    data = {p.relative_to(pkg) for p in pkg.rglob("*") if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts}
    assert data, "the tree has data files (schemas, registry, store DDL)"
    shipped = {p.relative_to(pkg) for g in declared for p in pkg.glob(g) if p.is_file()}
    assert data <= shipped, sorted(map(str, data - shipped))


# ================================================================ S32: the four remaining packs and their checks
# kafka, paper-inhouse-bus, infra-signatures (+ its negatives file) and severity-rules, as files under cpg/packs/.

PACK_DIR = CPG / "packs"
CLASSES = rulepack.signature_classes()


def stage(tmp_path: Path, pid: str, mutate=None, *, negatives="keep") -> Path:
    """Copy ``packs/<pid>/`` into ``tmp_path`` (dirname preserved), applying ``mutate`` to the loaded pack; returns pack.yaml.
    ``negatives``: "keep" copies the negatives file, "drop" leaves it out, any other value is written as its text."""
    d = tmp_path / pid
    d.mkdir(parents=True)
    pack = rulepack.load_yaml((PACK_DIR / pid / "pack.yaml").read_text())
    if mutate:
        mutate(pack)
    (d / "pack.yaml").write_text(yaml.safe_dump(pack, sort_keys=False))
    src = PACK_DIR / pid / NEGATIVES_FILE
    if negatives == "keep" and src.exists():
        (d / NEGATIVES_FILE).write_text(src.read_text())
    elif negatives not in ("keep", "drop"):
        (d / NEGATIVES_FILE).write_text(negatives)
    return d / "pack.yaml"


def load_errors(tmp_path: Path, pid: str, mutate=None, **kw) -> list[str]:
    try:
        rulepack.load_pack(stage(tmp_path, pid, mutate, **kw))
    except rulepack.PackError as e:
        return e.messages
    return []


def test_the_pack_file_negatives_are_the_spec_negatives_block_verbatim():
    text = (PACK_DIR / "infra-signatures" / NEGATIVES_FILE).read_text()
    block = next(b for _l, b in spec.yaml_blocks() if "near-miss" in b and b.lstrip().startswith("# negatives"))
    assert without_comment_lines(text) == without_comment_lines(block)
    assert rulepack.load_yaml(text) == spec.negatives()


def test_load_pack_runs_the_near_miss_checks_of_the_negatives_file(tmp_path):
    """Carried P4-6: load_pack passes the pack directory's negatives to the validator, so N-22 holds for the file itself."""
    def broaden(p):
        next(s for s in p["signatures"] if s["id"] == "python-timeout").update(regex="TimeoutError|timed out")
    errs = load_errors(tmp_path, "infra-signatures", broaden)
    assert any("signatures" in e and "matches the negative" in e for e in errs), errs


def test_the_unmutated_infra_pack_loads_with_its_negatives(tmp_path):
    assert load_errors(tmp_path, "infra-signatures") == []


@pytest.mark.parametrize("name", ["python-timeout", "any-oom-kill", "any-connection-refused", "python-oom-kill", "go-timeout"])
def test_a_broadened_regex_of_the_8_2_list_is_rejected_by_load_pack_through_the_file_negatives(tmp_path, name):
    broadened = {
        "python-timeout": lambda rx: rx + "|timed out",
        "any-oom-kill": lambda rx: rx.replace(r"137\b", "137"),
        "any-connection-refused": lambda rx: rx + "|refused",
        "python-oom-kill": lambda rx: rx + "|memory",
        "go-timeout": lambda rx: ".*",
    }[name]

    def edit(p):
        s = next(s for s in p["signatures"] if s["id"] == name)
        s["regex"] = broadened(s["regex"])
    assert any("matches the negative" in e for e in load_errors(tmp_path, "infra-signatures", edit))


def test_an_infra_pack_without_a_negatives_file_is_rejected_not_silently_unchecked(tmp_path):
    errs = load_errors(tmp_path, "infra-signatures", negatives="drop")
    assert any("negatives" in e for e in errs), errs


@pytest.mark.parametrize("name,text", [
    ("not a mapping", "- a\n- b\n"),
    ("empty file", ""),
    ("no shared", "near-miss: {connection-refused: [x], dns-failure: [x], timeout: [x], oom-kill: [x], tls-error: [x], third-party-exception: [x]}\n"),
    ("empty shared", "shared: []\nnear-miss: {connection-refused: [x], dns-failure: [x], timeout: [x], oom-kill: [x], tls-error: [x], third-party-exception: [x]}\n"),
    ("shared line not a string", "shared: [1]\nnear-miss: {connection-refused: [x], dns-failure: [x], timeout: [x], oom-kill: [x], tls-error: [x], third-party-exception: [x]}\n"),
    ("no near-miss", "shared: [x]\n"),
    ("near-miss not a mapping", "shared: [x]\nnear-miss: [x]\n"),
    ("a class without near-misses", "shared: [x]\nnear-miss: {connection-refused: [x], dns-failure: [x], timeout: [x], oom-kill: [x], tls-error: [x]}\n"),
    ("a class with an empty list", "shared: [x]\nnear-miss: {connection-refused: [], dns-failure: [x], timeout: [x], oom-kill: [x], tls-error: [x], third-party-exception: [x]}\n"),
    ("an unknown class", "shared: [x]\nnear-miss: {connection-refused: [x], dns-failure: [x], timeout: [x], oom-kill: [x], tls-error: [x], third-party-exception: [x], latency: [x]}\n"),
    ("a near-miss line not a string", "shared: [x]\nnear-miss: {connection-refused: [1], dns-failure: [x], timeout: [x], oom-kill: [x], tls-error: [x], third-party-exception: [x]}\n"),
])
def test_a_malformed_negatives_file_is_rejected(tmp_path, name, text):
    errs = load_errors(tmp_path, "infra-signatures", negatives=text)
    assert errs and all("negatives" in e for e in errs), (name, errs)


def test_a_negatives_file_with_only_the_required_keys_is_accepted(tmp_path):
    ok = "shared: [x]\nnear-miss: {" + ", ".join(f"{c}: [x]" for c in CLASSES) + "}\n"
    assert load_errors(tmp_path, "infra-signatures", negatives=ok) == []


def test_every_near_miss_class_of_the_file_is_a_signature_class():
    near = rulepack.load_yaml((PACK_DIR / "infra-signatures" / NEGATIVES_FILE).read_text())["near-miss"]
    assert set(near) == set(CLASSES)


# ---------------------------------------------------------------- section 10 negatives that apply to the four new packs, against the files

def lines_of(name: str) -> list:
    return [t for t in (MORE_NEGATIVES + [(n, p, m) for n, p, m in spec.RANGE_NEGATIVES]) if t[0] == name]


NEW_PACKS = ("kafka", "paper-inhouse-bus", "infra-signatures", "severity-rules")


def on_rule(i, **fields):
    def m(p):
        p["rules"][i]["emit"][0][next(iter(p["rules"][i]["emit"][0]))].update(fields)
    return m


# (name, pid, mutate, fragment the message must carry): each applies the spec's N-n to a pack that carries the construct.
FILE_NEGATIVES = [
    ("N-1 packFormat", "kafka", lambda p: p.update(packFormat=2), "packFormat"),
    ("N-1 packFormat absent", "severity-rules", lambda p: p.pop("packFormat"), "packFormat"),
    ("N-2 id is not the directory", "paper-inhouse-bus", lambda p: p.update(id="kafka"), "directory"),
    ("N-2 uppercase id", "infra-signatures", lambda p: p.update(id="Infra-Signatures"), "id"),
    ("N-3 kind plugin", "kafka", lambda p: p.update(kind="plugin"), "kind"),
    ("N-4 emit []", "kafka", lambda p: p["rules"][0].update(emit=[]), "emit"),
    ("N-5 two subjects", "kafka", lambda p: p["rules"][0].update(match={"callsite": {}, "decorator": {}}), "match"),
    ("N-6 lookahead", "kafka", lambda p: p["rules"][0]["match"]["callsite"].update(callee={"regex": "(?=x)abc"}), "lookaround"),
    ("N-6 backreference", "kafka", lambda p: p["rules"][1]["match"]["callsite"].update(callee={"regex": r"(a)\1"}), "backreference"),
    ("N-7 regex of 257 characters", "kafka", lambda p: p["rules"][0]["match"]["callsite"].update(callee={"regex": "a" * 257}), "regex"),
    ("N-7 signature regex of 257 characters", "infra-signatures", lambda p: p["signatures"][0].update(regex="a" * 257, sample="a" * 257), "signatures.0.regex"),
    ("N-8 two matchers", "kafka", lambda p: p["rules"][0]["match"]["callsite"].update(callee={"exact": "x", "glob": "y"}), "callee"),
    ("N-9 invented reason", "kafka", on_rule(0, reason="looks-fine"), "reason"),
    ("N-10 undeclared kind", "kafka", on_rule(0, kind="widget"), "widget"),
    ("N-10 undeclared kind on a link", "paper-inhouse-bus", lambda p: p.update(link=[{"kind": "widget", "matcher": "exact"}]), "widget"),
    ("N-11 vocab entry without keyNorm", "kafka", lambda p: p["vocab"][0].pop("keyNorm"), "keyNorm"),
    ("N-12 arg grammar", "kafka", lambda p: p["rules"][0]["extract"].update(t={"arg": "arg0"}), "arg"),
    ("N-13 duplicate rule ids", "kafka", lambda p: p["rules"][1].update(id="produce"), "duplicate"),
    ("N-13 duplicate rule ids (bus)", "paper-inhouse-bus", lambda p: p["rules"][1].update(id="publish"), "duplicate"),
    ("N-14 write key in a query", "paper-inhouse-bus", lambda p: p["queries"][0].update(**{"set": {"x": 1}}), "queries"),
    ("N-14 secretSighting record", "paper-inhouse-bus", lambda p: p["queries"][0]["from"][1].update(record="secretSighting"), "record"),
    ("N-14 five bindings", "paper-inhouse-bus", lambda p: p["queries"][0]["from"].append({"as": "z", "record": "symbol", "where": {"id": "$ns.id"}}), "from"),
    ("N-14 join on a non-id field", "paper-inhouse-bus", lambda p: p["queries"][0]["from"][1].update(where={"id": "$dec.name"}), "join"),
    ("N-14 no join equality after the first", "paper-inhouse-bus", lambda p: p["queries"][0]["from"][1].update(where={"name": {"regex": "x"}}), "join equality"),
    ("N-14 no handler column when a rule anchors on it", "paper-inhouse-bus", lambda p: p["queries"][0]["yield"].pop("handler"), "handler"),
    ("N-17 confirmed with the default-config-derived key slot", "kafka", on_rule(0, status="confirmed"), "confirmed"),
    ("N-17 confirmed beside a possible code", "kafka", on_rule(0, status="confirmed", reason="partial-key"), "partial-key"),
    ("N-18 level severe", "severity-rules", lambda p: p["rows"][0].update(level="severe"), "level"),
    ("N-18 rank 4", "severity-rules", lambda p: p["rows"][0].update(rank=4), "rank"),
    ("N-19 transport uppercase", "kafka", lambda p: p["vocab"][0].update(transport="Kafka"), "transport"),
    ("N-19 other pack's inhouse transport", "paper-inhouse-bus", lambda p: p["vocab"][0].update(transport="inhouse-other"), "inhouse-paper-inhouse-bus"),
    ("N-20 config value", "kafka", lambda p: p["rules"][0]["extract"].update(v={"config": "value"}), "config"),
    ("N-21 repoId", "kafka", lambda p: p.update(repoId="x"), "repoId"),
    ("N-21 repoId (infra)", "infra-signatures", lambda p: p.update(repoId="x"), "repoId"),
    ("N-21 repoId (severity)", "severity-rules", lambda p: p.update(repoId="x"), "repoId"),
    ("N-22 class latency", "infra-signatures", lambda p: p["signatures"][0].update(**{"class": "latency"}), "class"),
    ("N-22 sample its regex does not match", "infra-signatures", lambda p: p["signatures"][3].update(sample="all fine"), "sample"),
    ("N-22 sampleSource guessed", "infra-signatures", lambda p: p["signatures"][0].update(sampleSource="guessed"), "sampleSource"),
    ("N-22 regex matches a near-miss", "infra-signatures", lambda p: p["signatures"][0].update(regex=p["signatures"][0]["regex"] + "|connection"), "matches the negative"),
    ("N-22 regex matches a shared negative", "infra-signatures", lambda p: p["signatures"][0].update(regex=p["signatures"][0]["regex"] + "|KeyError"), "matches the negative"),
    ("N-23 link matcher", "kafka", lambda p: p.update(link=[{"kind": "topic", "matcher": "fuzzy"}]), "matcher"),
    ("N-24 level disagrees with rank", "severity-rules", lambda p: p["rows"][0].update(level="low"), "level"),
    ("N-24 second row for one (finding, rank)", "severity-rules", lambda p: p["rows"].append(dict(p["rows"][0])), "second row"),
    ("N-24 id other than the derived one", "severity-rules", lambda p: p["rows"][1].update(id="INJ-1"), "derived"),
    ("N-25 matcher with two operands", "kafka", lambda p: p["rules"][0]["scope"]["imports"][0].update(module={"prefix": "kafka", "exact": "kafka"}), "module"),
    ("N-27 config-derived on a consumes of kind http (a kind swap)", "kafka", lambda p: (p["vocab"].clear(), p["link"].clear(), p["rules"][1]["emit"][0]["consumes"].update(kind="http")), "config-derived"),
    ("N-27 reason not valid on the edge and kind", "kafka", on_rule(0, reason="unknown-key"), "not valid on"),
    ("N-28 empty firstOf", "kafka", lambda p: p["rules"][0]["extract"].update(topic={"firstOf": []}), "firstOf"),
    ("N-28 empty anyOf", "kafka", lambda p: p["rules"][0].update(match={"anyOf": []}), "anyOf"),
    ("N-28 query without a handler column", "paper-inhouse-bus", lambda p: p["queries"][0]["yield"].pop("handler"), "handler"),
]


@pytest.mark.parametrize("name,pid,mutate,fragment", FILE_NEGATIVES, ids=[n for n, *_ in FILE_NEGATIVES])
def test_a_section_10_negative_is_rejected_for_the_new_pack_file(tmp_path, name, pid, mutate, fragment):
    errs = load_errors(tmp_path, pid, mutate)
    assert errs, f"{name} was accepted"
    assert any(fragment in e for e in errs), (fragment, errs)


@pytest.mark.parametrize("name,pid,mutate", [t for t in spec.RANGE_NEGATIVES if t[1] in NEW_PACKS] + [t for t in MORE_NEGATIVES if t[1] in NEW_PACKS],
                         ids=lambda v: v if isinstance(v, str) and " " in v else None)
def test_the_spec_and_s31_negative_lists_hold_for_the_new_pack_files(tmp_path, name, pid, mutate):
    assert load_errors(tmp_path, pid, mutate), f"{name} was accepted by the file"


@pytest.mark.parametrize("pid", NEW_PACKS)
def test_a_pack_file_in_another_directory_is_rejected(tmp_path, pid):
    d = tmp_path / "elsewhere"
    d.mkdir()
    for f in (PACK_DIR / pid).iterdir():
        (d / f.name).write_text(f.read_text())
    with pytest.raises(rulepack.PackError, match="directory name"):
        rulepack.load_pack(d / "pack.yaml")


# ---------------------------------------------------------------- the severity pack file: the 44 vectors, X1 to X11 and the mutation check

def sev_formula() -> dict:
    return rulepack.load_pack(PACK_DIR / "severity-rules" / "pack.yaml")["formula"]


def sev_rows() -> list[dict]:
    return rulepack.load_pack(PACK_DIR / "severity-rules" / "pack.yaml")["rows"]


def test_the_severity_pack_file_passes_the_44_vectors_and_x1_to_x11():
    assert len(spec.vector_rows()) == 44 and len(spec.EXTRA_VECTORS) == 11
    assert spec.run_vectors(sev_formula()) == []


def test_the_severity_pack_file_names_the_derived_row_for_every_vector():
    rows = {(r["finding"], r["rank"]): r["id"] for r in sev_rows()}
    for vid, kind, f, want, row in spec.vector_rows() + spec.extra_vector_rows():
        got = spec.rank(sev_formula(), kind, f)
        assert got == want and rows[(kind, got)] == row, vid
    assert {row for *_, row in spec.vector_rows()} == {r["id"] for r in sev_rows()}


def test_every_mutant_of_the_severity_pack_file_fails_a_vector():
    survivors = [n for n, m, order in spec.mutants(sev_formula()) if not spec.run_vectors(m, order)]
    assert not survivors, survivors


WHEN_VALUES = {"source": ["network", "local"], "sink": ["exec", "data"], "sanitizer": ["none", "partial"], "entry": ["network", "internal"],
               "operation": ["write", "read"], "authCheck": ["none", "partial"], "state": ["head", "removed"],
               "fileRole": ["primary", "example", "doc"], "configuration": ["host-compromise", "exposed-service", "exposed-app", "hygiene"],
               "secrets": ["K", "P", "G"]}


def formula_entries(formula: dict):
    """(label, list-in-formula-getter, index, entry) for every caps, base, adjust and penalties entry."""
    for i, e in enumerate(formula["caps"]):
        yield f"caps[{i}]", (lambda f: f["caps"]), i, e
    for kind in ("injection", "authorization", "configuration", "secrets"):
        for section, entries in formula[kind].items():
            for i, e in enumerate(entries):
                yield f"{kind}.{section}[{i}]", (lambda f, k=kind, s=section: f[k][s]), i, e


def when_mutants(formula: dict):
    """Every `when` of the pack, edited one clause at a time: the clause dropped; an enumerated value swapped for another of
    its set; a boolean flipped; a path or basename matcher widened to match anything and narrowed to match nothing; then the
    whole entry removed."""
    for label, get, i, e in formula_entries(formula):
        m = copy.deepcopy(formula)
        del get(m)[i]
        yield f"{label} removed", m
        for k, v in e["when"].items():
            kind = label.split(".")[0]
            options = ([not v] if isinstance(v, bool)
                       else [{"regex": ".*"}, {"regex": "zz-never-zz"}] if isinstance(v, dict)
                       else [o for o in WHEN_VALUES.get(k, WHEN_VALUES.get(kind, [])) if o != v] if k != "class" else
                       [o for o in WHEN_VALUES[kind] if o != v])
            m = copy.deepcopy(formula)
            del get(m)[i]["when"][k]
            yield f"{label}.when.{k} dropped", m
            for o in options:
                m = copy.deepcopy(formula)
                get(m)[i]["when"][k] = o
                yield f"{label}.when.{k}={o}", m


def test_the_when_mutant_generator_covers_every_entry_and_every_clause():
    f = sev_formula()
    entries = list(formula_entries(f))
    clauses = sum(len(e["when"]) for *_, e in entries)
    names = [n for n, _ in when_mutants(f)]
    assert len(entries) == 1 + (4 + 2) + (4 + 2) + (4 + 1) + (6 + 1 + 1)  # caps, then base and penalties of each kind (secrets adds adjust)
    assert sum(n.endswith(" removed") for n in names) == len(entries)
    assert sum(n.endswith(" dropped") for n in names) == clauses
    assert len(names) > len(entries) + clauses  # value swaps and matcher edits beyond the drops


# `when` mutants no vector can tell apart: the clause is redundant with another in the same entry or unreachable.
EQUIVALENT_WHEN_MUTANTS: set[str] = set()


def test_every_when_clause_of_the_severity_pack_file_is_pinned_by_a_vector():
    survivors = sorted(n for n, m in when_mutants(sev_formula()) if not spec.run_vectors(m))
    assert set(survivors) == EQUIVALENT_WHEN_MUTANTS, survivors


@pytest.mark.parametrize("pattern,instance,ok", [("\\d", "٣", False), ("\\d", "3", True), ("^\\w$", "é", False), ("^\\w$", "e", True)])
def test_p5_3_the_pattern_keyword_is_ascii_for_both_the_anchored_and_the_unanchored_form(pattern, instance, ok):
    """No shipped schema pattern is unanchored today; the unanchored branch still follows ECMA so a future one cannot drift."""
    v = rulepack._Validator({"type": "string", "pattern": pattern})
    assert v.is_valid(instance) is ok
