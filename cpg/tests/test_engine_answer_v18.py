"""P1-39b carry-over (g): V17-V20 (``ccg.answer.validate``) re-run over **real walk output**, not planted documents.

Every world the engine's walk tests use (``tests/test_engine_flow.py``, ``tests/test_engine_group.py`` and the builder tests' own), rewritten to 32-hex ids, is built into an answer
by ``ccg.engine.answer_build`` from every symbol, contract point and config entry as anchor, in both columns, with every unit's ``flowAvailable`` forced true and forced false,
over the shipped table (V-both, ``table=None``) and (internally) the three other tables ``as-built`` (caller-only), ``V-flow`` and ``V-union`` (``ccg.engine.flow.candidate_table``). The builder validates before it
returns, so a document is a pass; the tests assert the population is not trivial (the ``function`` state, ``flow-unavailable`` labels and a candidate walk's callee state occur).
The stored slice bundles (``slice`` marker: absent artifacts skip, reported unrun) are walked the same way.
"""
from __future__ import annotations

import copy

import pytest
from engine_answer_helpers import AnswerReader, hexify

from ccg import answer
from ccg.engine import anchors, flow
from ccg.engine import answer_build as AB
from ccg.engine.reader import GraphDataError

import test_engine_answer as TA
import test_engine_flow as TF
import test_engine_group as TG

TABLES = ["V-both", "as-built", "V-flow", "V-union"]
SHIPPED = "V-both"          # P1-39d: the shipped table is V-both (``table=None``); the caller-only ``as-built`` table and the V-flow and V-union tables are the candidates


def table_arg(name):
    """The ``table=`` of ``build_answer`` for a table name: the shipped one is the default, the others are passed explicitly."""
    return None if name == SHIPPED else flow.candidate_table(name)


def raw_worlds():
    """Name -> a fresh un-hexified ``G``."""
    out = {
        "caller": TA.caller_world_g, "flagfalse2": TA.flag_false_two_callers_g, "link": TA.link_world_g, "depends": TA.depends_world_g, "fold": TA.fold_world_g,
        "chain-js": lambda: TF.caller_chain(flow=False), "chain-split": lambda: TF.caller_chain(flow=False, caller_flow=True, caller_lang="python"),
        "points": lambda: TF.point_rows_graph(False), "points-true": lambda: TF.point_rows_graph(True), "callee-go": lambda: TF.callee_graph(False),
        "callee-go-true": lambda: TF.callee_graph(True), "two-site": lambda: TF.two_site_caller(name_site_first=True)[0], "reach": TF.reach_graph,
        "callee-reach": TF.callee_reach_graph, "callee-chain": lambda: TF.callee_chain(depth=3), "via": TF.via_graph,
        "dep-site": lambda: TF.dep_graph("CallSite"), "dep-ret": lambda: TF.dep_graph("Ret"), "dep-cond": lambda: TF.dep_graph("Cond"),
        "mixed": TG.mixed_graph, "candidates": lambda: TG.candidates_graph(lang="javascript", cand_lang="javascript", n_sites=2), "candidates-py": TG.candidates_graph,
        "downstream": TG.downstream_graph, "arg-row": TG.arg_row_graph,
    }
    return out


def force_flag(g, flag):
    for n in g.rows.nodes.get("Unit", []):
        if n.get("buildStatus") == "fresh":
            n["flowAvailable"] = flag


def anchors_of(g):
    out = []
    for n in g.rows.nodes.get("Symbol", []):
        out.append(anchors.SymbolAnchor(n["symbolId"], n["qname"], n["repoId"], n["path"], n["unitId"], n["lineStart"], n["lineEnd"]))
    for n in g.rows.nodes.get("ContractPoint", []):
        out.append(anchors.ContractAnchor(n["cpId"], n["kind"], n["keyNorm"]))
    for n in g.rows.nodes.get("ConfigEntry", []):
        if n.get("keyPath") is not None:
            out.append(anchors.ConfigAnchor(n["entryId"], n["repoId"], n["path"], n["keyPath"], n["unitId"], n["line"]))
    return out


def run_all(worlds, tables=TABLES, monkeypatch=None):
    """Every (world, flag, table, column, anchor) document. A refusal is recorded with the validator's own message (captured from ``ccg.answer.validate`` through a spy: the
    builder's error names only the rule)."""
    docs, refused = [], []
    for name, make in worlds.items():
        for flag in (True, False):
            for table in tables:
                g, ids_ = hexify(make())
                force_flag(g, flag)
                for a in anchors_of(g):
                    for question in ("impact", "depends-on"):
                        seen = []
                        orig = answer.validate

                        def spy(doc, orig=orig, seen=seen):
                            try:
                                orig(doc)
                            except answer.AnswerError as exc:
                                seen.append(str(exc))
                                raise
                        AB.answer.validate = spy
                        try:
                            doc = AB.build_answer(AnswerReader(g), a, question=question, table=table_arg(table))
                        except GraphDataError as exc:
                            refused.append((name, flag, table, question, type(a).__name__, str(exc), seen[0] if seen else ""))
                            continue
                        finally:
                            AB.answer.validate = orig
                        docs.append((name, flag, table, question, a, doc))
    return docs, refused


@pytest.fixture(scope="module")
def population():
    return run_all(raw_worlds())


def test_every_shipped_table_document_over_every_world_anchor_column_and_flag_builds_and_validates_v17_to_v20(population):
    """The shipped table is V-both (``table=None``): every document is ``"1.2"`` at algorithm ``1.2`` and validates under V20 and V18's ``"1.2"`` rules; the caller-only table, passed
    explicitly, builds and validates as well (a ``"1.2"`` document with the ``1.1`` record's version), so nothing but V-flow and V-union is ever refused."""
    docs, refused = population
    shipped = [d for d in docs if d[2] == SHIPPED]
    assert len(shipped) > 300 and len([d for d in docs if d[2] == "as-built"]) == len(shipped)
    assert [r for r in refused if r[2] in (SHIPPED, "as-built")] == []
    for *_, doc in shipped:
        answer.validate(doc)
        assert doc == answer.canonicalize(doc) and doc["answerSchemaVersion"] == "1.2"
    for *_, doc in [d for d in docs if d[2] == "as-built"]:
        answer.validate(doc)
        assert doc["answerSchemaVersion"] == "1.2"


def test_a_candidate_table_refuses_exactly_where_v18_does_not_define_its_added_hops_and_nowhere_else(population):
    """V18 (``-answer-spec2.md`` §5) encodes the shipped table (V-both): V-flow's added rows label a hop that does not leave ``function`` (clause (c)), and so do V-union's (it holds them
    as well). Flag-false units only; the shipped table and the caller-only one are never refused, and nothing else is: another shipped table needs §5 (and V18) revised first (spec2 §1.2)."""
    docs, refused = population
    candidate = [r for r in refused if r[2] != SHIPPED]
    assert candidate, "the candidate tables added hops no world reached"
    for name, flag, table, question, kind, msg, detail in candidate:
        assert flag is False and msg == "the built answer does not validate (rule V18)", (name, table, msg)
        assert detail.split(": ", 2)[2][:3] == "(c)", (table, detail)
    assert {r[2] for r in candidate} == {"V-flow", "V-union"}


def test_the_population_is_not_trivial_the_function_state_the_labels_and_a_candidate_walks_callee_state_all_occur(population):
    docs, _ = population
    seen = {"function": 0, "flow": 0, "site": 0, "unit": 0, "wiring": 0, "group": 0, "link": 0}
    by_table = {t: 0 for t in TABLES}
    for name, flag, table, question, a, doc in docs:
        by_table[table] += len(doc["items"])
        if table != SHIPPED:
            continue
        seen["group"] += len(doc["groups"])
        for it in doc["items"] + [m for g in doc["groups"] for m in g["members"]]:
            for h in it["chain"]:
                seen["function"] += h["state"] == "function"
                seen["flow"] += str(h.get("reason", "")).startswith("flow-unavailable:")
                seen["site"] += h["state"] == "site"
                seen["unit"] += h["kind"] == "Unit"
                seen["link"] += "lifted" in h
                seen["wiring"] += h["kind"] == "ConfigEntry" and h.get("edgeType") == "BOUND_BY" and h.get("dir") == "->"
    assert all(v > 0 for v in seen.values()), seen
    assert all(v > 0 for v in by_table.values()), by_table
    callee = [h for name, flag, table, question, a, doc in docs if table == SHIPPED and question == "impact" and flag is False for it in doc["items"] for h in it["chain"]
              if (h.get("edgeType"), h.get("dir")) == ("HAS_CALL", "->")]
    assert callee, "no flag-false V-both answer holds a callee run"                      # P1-39d: the shipped walk lists callees as well


def test_a_candidate_table_lists_at_least_what_the_caller_only_one_does_and_the_v_union_what_each_does(population):
    docs, _ = population
    got = {}
    for name, flag, table, question, a, doc in docs:
        got.setdefault((name, flag, question, getattr(a, "symbol_id", None) or getattr(a, "cp_id", None) or a.entry_id), {})[table] = {
            (it["subject"].get("symbolId") or it["subject"].get("entryId") or it["subject"]["unitId"]) for it in doc["items"] + [m for g in doc["groups"] for m in g["members"]]}
    checked = 0
    for tables in got.values():
        if set(tables) == set(TABLES):
            for t in ("V-flow", "V-both"):
                assert tables["as-built"] <= tables[t] <= tables["V-union"]
            checked += 1
    assert checked > 300


def test_a_document_carries_the_version_its_walk_definition_maps_to_and_candidate_version_only_when_none_does(population):
    docs, _ = population
    want = {SHIPPED: "1.2", "as-built": "1.1", "V-flow": AB.CANDIDATE_VERSION, "V-union": AB.CANDIDATE_VERSION}
    assert {d[2] for d in docs} == set(want)
    for name, flag, table, question, a, doc in docs:
        assert doc["recipe"]["algorithm"]["version"] == want[table]


def test_the_shipped_walk_over_random_flag_mixed_worlds_builds_documents_whose_callee_runs_v18_accepts():
    """Mixed-flag, mixed-language worlds (``test_engine_flow.mixed_world``) through the builder, which validates: a callee run in a flag-false unit is accepted by V18's ``"1.2"``
    rules (``CALLEE_ROWS``, (b)-(g)). A V18 refusal (``BuildError``) is a failure; the random worlds' other ``GraphDataError`` kinds (hexified ids with no Unit row) are skipped."""
    docs = runs = refused = 0
    for seed in range(40):
        g, ids_ = hexify(TF.mixed_world(seed)[0])
        for a in anchors_of(g):
            for question in ("impact", "depends-on"):
                try:
                    doc = AB.build_answer(AnswerReader(g), a, question=question)
                except AB.BuildError:
                    raise
                except GraphDataError:
                    refused += 1
                    continue
                docs += 1
                runs += any((h.get("edgeType"), h.get("dir")) == ("HAS_CALL", "->") for it in doc["items"] for h in it["chain"])
    assert docs > 200 and runs > 20, (docs, runs, refused)


# ------------------------------------------------------------------------------------------------------------------ the stored slice bundles (marker ``slice``)

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "slice"))
from packrt_helpers import CPG, S14A  # noqa: E402
from test_packrt_conformance import SHA256  # noqa: E402

slice_only = [pytest.mark.slice, pytest.mark.skipif(not S14A.is_dir() or not list(S14A.glob("*.bundle.jsonl")),
                                                   reason="stored s14a slice bundles absent (cpg/.ccg-work is gitignored): UNRUN, not green")]


def _marked(fn):
    for m in slice_only:
        fn = m(fn)
    return fn


def _load(name):
    spec = importlib.util.spec_from_file_location("slice_" + name, CPG / "spikes/vertical-slice" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["slice_" + name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def slice_rows():
    for name, sha in SHA256.items():
        assert hashlib.sha256((S14A / name).read_bytes()).hexdigest() == sha, f"{name}: not the artifact the oracle was pinned to"
    bundles = [[json.loads(line) for line in (S14A / name).read_text().splitlines()] for name in sorted(SHA256)]
    return _load("load").build_rows(bundles, _load("assemble").assemble(bundles))


@pytest.fixture(scope="module")
def slice_population(slice_rows):
    docs, refused = [], []
    on_link = {e["t"] for e in slice_rows.edges[("Link", "ON", "ContractPoint")]}
    qn = {s["qname"]: s for s in slice_rows.nodes["Symbol"]}
    anchor_list = [anchors.ContractAnchor(c["cpId"], c["kind"], c["keyNorm"]) for c in slice_rows.nodes["ContractPoint"] if c["cpId"] in on_link]
    for q in ("salesperson/src/api/client.ts::apiFetch", "salesperson/src/api/client.ts::buildUrl"):
        n = qn[q]
        anchor_list.append(anchors.SymbolAnchor(n["symbolId"], n["qname"], n["repoId"], n["path"], n["unitId"], n["lineStart"], n["lineEnd"]))
    for flag in (False, True):
        rows = copy.copy(slice_rows)
        rows.nodes = {**slice_rows.nodes, "Unit": [{**u, "flowAvailable": flag} for u in slice_rows.nodes["Unit"]]}
        reader = AnswerReader(SimpleNamespace(rows=rows))      # one index per flag state: building it is the slow part; the recorded calls are reset per build
        for table in TABLES:
            for a in anchor_list:
                for question in ("impact", "depends-on"):
                    reader.calls = []
                    seen = []
                    orig = answer.validate

                    def spy(doc, orig=orig, seen=seen):
                        try:
                            orig(doc)
                        except answer.AnswerError as exc:
                            seen.append(str(exc))
                            raise
                    AB.answer.validate = spy
                    try:
                        doc = AB.build_answer(reader, a, question=question, table=table_arg(table))
                    except GraphDataError as exc:
                        refused.append((flag, table, question, str(exc), seen[0] if seen else ""))
                        continue
                    finally:
                        AB.answer.validate = orig
                    docs.append((flag, table, question, a, doc))
    return docs, refused


@_marked
def test_every_shipped_table_answer_over_the_stored_slice_validates_in_both_columns_and_both_flag_states(slice_population):
    docs, refused = slice_population
    shipped = [d for d in docs if d[1] == SHIPPED]
    assert [r for r in refused if r[1] in (SHIPPED, "as-built")] == []
    assert len(shipped) == 2 * 13 * 2 and all(doc == answer.canonicalize(doc) for *_, doc in shipped)
    health = next(d for d in shipped if d[3].__class__.__name__ == "ContractAnchor" and d[3].key_norm == "GET /shop/api/health" and d[0] is False and d[2] == "impact")[4]
    (item,) = health["items"]
    assert item["subject"]["qname"] == "salesperson/src/api/endpoints.ts::getHealth" and item["status"] == "confirmed" and item["distance"] == 1
    link = next(h for h in item["chain"] if h["kind"] == "Link")
    assert len(link["lifted"]) == 4 and next(h for h in item["chain"] if "candidateCount" in h)["candidateCount"] == 1


@_marked
def test_the_flag_false_slice_answers_carry_the_function_state_and_the_flow_unavailable_labels_the_real_data_gives(slice_population):
    docs, _ = slice_population
    hops = [h for flag, table, question, a, doc in docs if table == SHIPPED and flag is False for it in doc["items"] + [m for g in doc["groups"] for m in g["members"]] for h in it["chain"]]
    assert any(h["state"] == "function" for h in hops)
    assert {h["reason"] for h in hops if str(h.get("reason", "")).startswith("flow-unavailable")} == {"flow-unavailable:javascript"}
    assert any(h["state"] == "path" and h["accessPath"] == "r" for flag, table, question, a, doc in docs if flag is True for it in doc["items"] for h in it["chain"])


@_marked
def test_a_slice_document_is_refused_only_by_v18_c_in_flag_false_v_flow_or_v_union_and_the_shipped_walk_lists_callees_there(slice_population):
    docs, refused = slice_population
    assert all(r[0] is False and r[1] in ("V-flow", "V-union") and r[3] == "the built answer does not validate (rule V18)" and r[4].split(": ", 2)[2][:3] == "(c)" for r in refused)
    callee = [it for flag, table, question, a, doc in docs if table == SHIPPED and flag is False and question == "impact" for it in doc["items"]
              if (it["chain"][-1].get("edgeType"), it["chain"][-1].get("dir")) == ("CALLS", "->")]
    assert callee and all(it["status"] == "possible" for it in callee)
