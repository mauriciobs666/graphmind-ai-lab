"""P1-09 behaviour (5): no connected pair is lost by the projection fix, on the slice's stored raw facts.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` P1-09 (5); the carve-out is `-p1-ml.md` §3.4. The shipped projection is the slice's
S14a facts (`cpg/.ccg-work/vertical-slice/s14a/<tag>/cpg/*.facts.jsonl`); the fixed one is `extract.sc`'s output on the same CPG
(`cpg/.ccg-work/p1-09/fixed-<tag>.facts.jsonl`, a bounded `joern --script`; regenerate with
`cpg/.ccg-work/p1-09/tools/` as REPORT.md records). Both are gitignored: **a skip is unrun, never green**.

Rules checked (per unit, over the whole fact files):
- every shipped `d` record is in the fixed facts (a multiset subset): the fix only adds data records;
- every shipped `cb` record is in the fixed facts, OR it is one of the **rewritten synthetic-cond edges**: `stats.cbRewrites` lists
  `[method, pointKind, pointId, syntheticCond, [replacement conds]]`; the replacement set is one edge, several, or none (an ungoverned
  edge dropped), and each replacement is in the fixed `cb` records for that point;
- a shipped edge missing from the fixed facts without a rewrite entry fails; so does a rewrite entry whose replacement is not emitted.
The pair-level corollary (no golden pair the shipped projection connects is unconnected by the fixed one) is `tests/test_projection_golden.py`
plus the scorer recompute in `cpg/.ccg-work/p1-09/REPORT.md`; edge-level subset implies it for data (reach grows) and, for control,
holds modulo the listed rewrites.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pytest

from ccg import atrest

S14A = atrest.WORK_ROOT / "vertical-slice" / "s14a"
FIXED = atrest.WORK_ROOT / "p1-09"
TAGS = ("fcs", "sp")


def load(path: Path) -> tuple[dict[str, collections.Counter], dict | None]:
    recs: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    stats = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o["k"] == "stats":
            stats = o
        elif o["k"] in ("d", "cb"):
            recs[o["k"]][json.dumps(o, sort_keys=True, separators=(",", ":"))] += 1
    return recs, stats


def regressions(shipped: dict, fixed: dict, fixed_stats: dict) -> dict[str, list]:
    """The violations of the rules above (empty lists = no connected pair lost) plus the rewrite census."""
    out: dict[str, list] = {"d-lost": [], "cb-lost": [], "rewrite-missing": []}
    out["d-lost"] = sorted((shipped["d"] - fixed["d"]).elements())[:5]
    rewrites = {(r[0], r[1], r[2], r[3]): r[4] for r in fixed_stats.get("cbRewrites", [])}
    census = collections.Counter()
    for rec in (shipped["cb"] - fixed["cb"]):
        o = json.loads(rec)
        key = (o["m"], o["pk"], o["pid"], o["cond"])
        if key not in rewrites:
            out["cb-lost"].append(rec)
            continue
        repl = rewrites[key]
        census["none" if not repl else "one" if len(repl) == 1 else "several"] += 1
        for c in repl:
            want = json.dumps({"k": "cb", "m": o["m"], "pk": o["pk"], "pid": o["pid"], "cond": c}, sort_keys=True, separators=(",", ":"))
            if want not in fixed["cb"]:
                out["rewrite-missing"].append(want)
    out["census"] = [dict(census)]
    return out


#: the cond labels a synthetic cond can carry: the iterator is an UNKNOWN node coded `iteratorNonEmptyOrException`; a try-lowering cond is a
#: statement's value node (a call, a returned name or literal). A comparison or boolean operator node is never synthetic.
SYNTH_LABELS = {"CALL", "IDENTIFIER", "LITERAL", "UNKNOWN"}
OPERATOR_CODE = ("not ", )


def shape_problems(conds: dict, rewrites: list) -> list[str]:
    """An independent shape check of the `cbRewrites` the extractor attests about itself (it uses the facts' own `c` records, not the script's
    predicates): every rewritten cond is a `c` fact of its method; it is the iterator cond or carries a statement-like label; every replacement
    is a different `c` fact of the same method that is not itself rewritten. It cannot prove a cond IS a try-lowering one (that needs the source
    tree); it catches a rewrite of a missing, operator-shaped or self-referential cond."""
    synth = {(r[0], r[3]) for r in rewrites}
    out = []
    for m, _pk, _pid, cond, repl in rewrites:
        c = conds.get((m, cond))
        if c is None:
            out.append(f"{m}/{cond}: not a cond fact")
            continue
        if c["code"] != "iteratorNonEmptyOrException" and (c["lab"] not in SYNTH_LABELS or c["code"].startswith(OPERATOR_CODE)):
            out.append(f"{m}/{cond}: shape {c['lab']} {c['code'][:30]!r}")
        for x in repl:
            if x == cond or (m, x) not in conds or (m, x) in synth:
                out.append(f"{m}/{cond}: bad replacement {x}")
    return out


def conds_of(path: Path) -> dict:
    return {(o["m"], o["id"]): o for o in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()) if o["k"] == "c"}


# --- the comparator on planted records (always runs) ----------------------------------------------------------------
def d(i, **kw):
    return json.dumps({"k": "d", "m": "1", "uid": i, **kw}, sort_keys=True, separators=(",", ":"))


def cb(pk, pid, cond):
    return json.dumps({"k": "cb", "m": "1", "pk": pk, "pid": pid, "cond": cond}, sort_keys=True, separators=(",", ":"))


def C(*recs):
    return collections.Counter(recs)


def test_a_superset_with_a_listed_rewrite_is_clean():
    shipped = {"d": C(d("a")), "cb": C(cb("r", "p", "S"), cb("r", "q", "R"))}
    fixed = {"d": C(d("a"), d("b")), "cb": C(cb("r", "p", "C"), cb("r", "q", "R"))}
    stats = {"cbRewrites": [["1", "r", "p", "S", ["C"]]]}
    r = regressions(shipped, fixed, stats)
    assert not r["d-lost"] and not r["cb-lost"] and not r["rewrite-missing"]
    assert r["census"] == [{"one": 1}]


def test_the_ungoverned_and_several_cases_are_listed_with_their_replacement_sets():
    shipped = {"d": C(), "cb": C(cb("r", "p", "S"), cb("r", "q", "S"))}
    fixed = {"d": C(), "cb": C(cb("r", "q", "C1"), cb("r", "q", "C2"))}
    stats = {"cbRewrites": [["1", "r", "p", "S", []], ["1", "r", "q", "S", ["C1", "C2"]]]}
    r = regressions(shipped, fixed, stats)
    assert r["census"] == [{"none": 1, "several": 1}] and not r["cb-lost"] and not r["rewrite-missing"]


@pytest.mark.parametrize("shipped,fixed,stats,key", [
    ({"d": C(d("a")), "cb": C()}, {"d": C(), "cb": C()}, {}, "d-lost"),                                   # a data record lost
    ({"d": C(d("a"), d("a")), "cb": C()}, {"d": C(d("a")), "cb": C()}, {}, "d-lost"),                      # a duplicate lost (multiset)
    ({"d": C(), "cb": C(cb("r", "p", "S"))}, {"d": C(), "cb": C()}, {}, "cb-lost"),                       # an edge dropped, no rewrite entry
    ({"d": C(), "cb": C(cb("r", "p", "S"))}, {"d": C(), "cb": C()}, {"cbRewrites": [["1", "r", "p", "S", ["C"]]]}, "rewrite-missing"),
    ({"d": C(), "cb": C(cb("r", "p", "S"))}, {"d": C(), "cb": C()}, {"cbRewrites": [["2", "r", "p", "S", []]]}, "cb-lost"),  # entry of another method
], ids=["d-dropped", "d-duplicate", "cb-dropped", "replacement-not-emitted", "wrong-method-entry"])
def _c(i, code, lab="CALL"):
    return {("1", i): {"k": "c", "m": "1", "id": i, "code": code, "lab": lab}}


def test_shape_check_accepts_the_two_synthetic_shapes_and_flags_the_rest():
    conds = {**_c("S", "x = f()"), **_c("I", "iteratorNonEmptyOrException", "UNKNOWN"), **_c("R", "keep is None")}
    assert not shape_problems(conds, [["1", "r", "p", "S", ["R"]], ["1", "r", "q", "I", ["R"]], ["1", "r", "z", "S", []]])
    assert shape_problems(conds, [["1", "r", "p", "GONE", ["R"]]])                                   # not a cond fact
    assert shape_problems({**conds, **_c("N", "not a", "CALL")}, [["1", "r", "p", "N", ["R"]]])       # operator-shaped
    assert shape_problems({**conds, **_c("B", "a", "BLOCK")}, [["1", "r", "p", "B", ["R"]]])          # a label no synthetic cond has
    assert shape_problems(conds, [["1", "r", "p", "S", ["S"]]])                                       # replaces itself
    assert shape_problems(conds, [["1", "r", "p", "S", ["I"]], ["1", "r", "q", "I", ["R"]]])          # replacement is itself rewritten
    assert shape_problems(conds, [["1", "r", "p", "S", ["NOPE"]]])                                    # replacement is no cond


@pytest.mark.parametrize("shipped,fixed,stats,key", [
    ({"d": C(d("a")), "cb": C()}, {"d": C(), "cb": C()}, {}, "d-lost"),                                   # a data record lost
    ({"d": C(d("a"), d("a")), "cb": C()}, {"d": C(d("a")), "cb": C()}, {}, "d-lost"),                      # a duplicate lost (multiset)
    ({"d": C(), "cb": C(cb("r", "p", "S"))}, {"d": C(), "cb": C()}, {}, "cb-lost"),                       # an edge dropped, no rewrite entry
    ({"d": C(), "cb": C(cb("r", "p", "S"))}, {"d": C(), "cb": C()}, {"cbRewrites": [["1", "r", "p", "S", ["C"]]]}, "rewrite-missing"),
    ({"d": C(), "cb": C(cb("r", "p", "S"))}, {"d": C(), "cb": C()}, {"cbRewrites": [["2", "r", "p", "S", []]]}, "cb-lost"),  # entry of another method
], ids=["d-dropped", "d-duplicate", "cb-dropped", "replacement-not-emitted", "wrong-method-entry"])
def test_the_comparator_flags_a_lost_pair(shipped, fixed, stats, key):
    assert regressions(shipped, fixed, stats)[key]


# --- the slice units ------------------------------------------------------------------------------------------------
def _pair(tag: str):
    shipped = sorted((S14A / tag / "cpg").glob("*.facts.jsonl")) if (S14A / tag / "cpg").is_dir() else []
    fixed = FIXED / f"fixed-{tag}.facts.jsonl"
    if len(shipped) != 1 or not fixed.is_file():
        pytest.skip(f"slice facts or fixed-projection facts absent for {tag}: {S14A / tag / 'cpg'} / {fixed} (UNRUN, not green)")
    stamp = fixed.with_name(fixed.name + ".sha256")
    now = hashlib.sha256((FIXED.parent.parent / "extractors" / "joern" / "extract.sc").read_bytes()).hexdigest()
    assert stamp.is_file() and stamp.read_text().strip() == now, (
        f"{fixed.name} was not produced by the current extract.sc (stamp {stamp.name} missing or different): regenerate it "
        "(extractors/joern/README.md, 'Regenerating the recorded facts')")
    return shipped[0], fixed


@pytest.mark.slice
@pytest.mark.parametrize("tag", TAGS)
def test_the_fixed_projection_loses_no_data_record_and_no_unlisted_edge(tag):
    shipped_p, fixed_p = _pair(tag)
    s, _ = load(shipped_p)
    f, fstats = load(fixed_p)
    assert sum(s["d"].values()) > 0 and sum(s["cb"].values()) > 0
    r = regressions(s, f, fstats)
    assert not r["d-lost"], r["d-lost"]
    assert not r["cb-lost"], r["cb-lost"][:5]
    assert not r["rewrite-missing"], r["rewrite-missing"][:5]


@pytest.mark.slice
@pytest.mark.parametrize("tag", TAGS)
def test_the_rewrites_have_the_shape_of_synthetic_conds_by_an_independent_check(tag):
    _, fixed_p = _pair(tag)
    _, fstats = load(fixed_p)
    assert not shape_problems(conds_of(fixed_p), fstats.get("cbRewrites", []))


@pytest.mark.slice
def test_the_javascript_unit_rewrites_nothing_and_the_python_unit_does():
    """4b is python-only: no `cbRewrites` for `sp`; `fcs` carries rewrites, including ungoverned (dropped) ones."""
    census = {}
    for tag in TAGS:
        shipped_p, fixed_p = _pair(tag)
        s, _ = load(shipped_p)
        f, fstats = load(fixed_p)
        census[tag] = regressions(s, f, fstats)["census"][0]
    assert census["sp"] == {}
    assert census["fcs"].get("none", 0) > 0 and census["fcs"].get("one", 0) > 0
