"""P1-04 behaviour (3) — the ported extractor reproduces the vertical slice's raw facts.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-04 (3); the rule is P1 index D9 ("Equivalence before
improvement"). The oracle is the slice's S14a raw facts, in the gitignored work area
`cpg/.ccg-work/vertical-slice/s14a/<unit>/cpg/*.facts.jsonl`. These tests carry the `slice` marker and skip when the
artifact is absent: **a skip is unrun, never green.** The run summary must read `0 skipped`.

Two tiers. Without Docker: the oracle's sha256 pin, the ported scripts' bodies equal to the slice's, and the stage
`run.stage_unit` builds at the pinned commit equal to the slice's stage tree, file for file. With Docker (also marked
`live`): the unit is extracted through `run.extract_unit` and its raw facts compared to the oracle as canonical JSON.

Canonical JSON: every record with sorted keys, the `stats` record reduced to its counts (its timings and histogram
differ in every run), and the records compared as a multiset in sorted order. Joern returns `d` and `cb` edges in
an order that varies from run to run (the slice's own S14a, run1, run2 and dev outputs differ line by line and agree as
multisets), so file order carries no information; the normalizer re-sorts by the spec's keys (`extract.sc` header).

Run: `cd cpg && .venv/bin/python -m pytest tests/slice -m "slice" -q` (add `-m "slice and live"` for the Docker tier alone).
"""
from __future__ import annotations

import collections
import filecmp
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

import overlay_5d
from ccg import atrest, ids
from ccg.units import Unit
from extractors.joern import run

CPG = Path(__file__).resolve().parent.parent.parent
REPO = CPG.parent
S14A = atrest.WORK_ROOT / "vertical-slice" / "s14a"
PINNED = "defca60ff16534aa12a74e35856de9be98bfc710"
SPIKE = CPG / "spikes" / "vertical-slice"
GOLDEN = CPG / "tests" / "golden" / "system-units.json"

#: tag -> (rootPath, language, sha256 of the S14a facts file recorded 2026-11-10, S14a stats counts, source files)
SLICE = {
    "fcs": ("falkor-chat/server", "python", "de52bc6891dbe39233d1feaa3334e9b2a0a2a0b8de5074d3692879070c876fa1",
            {"methods": 6254, "cdgRaw": 48022, "cdgSelf": 2751, "dataRaw": 24760}, 106),
    "sp": ("salesperson", "javascript", "585b11df0d5496c75258988be9268e006c536be40f47ceeed23a350117f57fdf",
           {"methods": 232, "cdgRaw": 2242, "cdgSelf": 5, "dataRaw": 637}, 74),
}
COUNT_KEYS = ("methods", "cdgRaw", "cdgSelf", "dataRaw")


def unit_of(tag: str) -> Unit:
    root, lang = SLICE[tag][:2]
    return Unit(ids.unit_id("graphmind-ai-lab", root, lang), "graphmind-ai-lab", root, lang)


def oracle(tag: str) -> Path:
    found = sorted((S14A / tag / "cpg").glob("*.facts.jsonl")) if (S14A / tag / "cpg").is_dir() else []
    if len(found) != 1:
        pytest.skip(f"slice artifact absent: {S14A / tag / 'cpg'} (UNRUN, not green; regenerate at {PINNED})")
    return found[0]


def need_commit() -> None:
    r = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", PINNED + "^{commit}"], capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"the pinned commit {PINNED} is not in this clone (UNRUN, not green)")


def canonical(path: Path, skip: tuple[str, ...] = (), overlay: str | None = None) -> tuple[list[str], dict]:
    """(sorted canonical JSON of every record but `stats`, the stats counts). The timings and histogram are dropped; records of the
    kinds in `skip` are dropped too (P1-09: `d` and `cb` are governed by the subset rules of `slice/test_projection_not_regressed.py`).
    `overlay` is a unit tag of the 5d overlay (`tests/overlay_5d.py`): the live script's `rt` on `cs` facts and the hand-listed
    export-block `const` facts are removed first, so what is compared is the oracle's own content."""
    recs, stats = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o["k"] == "stats":
            assert stats is None, "two stats records"
            stats = {k: int(o[k]) for k in COUNT_KEYS}
        elif o["k"] not in skip:
            recs.append(o)
    assert stats is not None, "no stats record"
    if overlay is not None:
        recs = overlay_5d.strip(recs, overlay)
    return sorted(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for o in recs), stats


#: P1-09 exemptions: the projection kinds, and the one count that derives from them (`cdgRaw` counts raw CDG edges and stays equal).
PROJECTION_KINDS = ("d", "cb")
PROJECTION_COUNT = "dataRaw"
#: the slice's own lines P1-09 replaced in `extract.sc` (everything else in the slice's body must survive verbatim). LIMIT: this only sees REMOVED
#: slice lines; a line INSERTED into a non-projection region that changes a non-projection fact is caught only by the optional (gitignored
#: facts) `test_the_fixed_projection_keeps_every_non_projection_fact_of_the_slice`, which is skipped when its facts are absent.
REPLACED_SLICE_LINES = [
    "  def objFields(b: Block): Option[List[(String, Expression)]] = {",
    '      case c: Call if isCallSite(c) => if (curOwn.contains(c.id)) List((c, "r", fields)) else Nil',
    "        objFields(b) match {",
    "              case h :: t => fs.filter(_._1 == h).flatMap { case (_, v) => defsOf(v, t, seen, depth + 1) }",
    "              case Nil => fs.flatMap { case (_, v) => defsOf(v, Nil, seen, depth + 1) }",
    "              case _ => Nil",
    "  // use positions of an expression: flatten object literals into fields (cap 3 fields after the head)",
    "  def usePositions(e: Expression, fields: List[String]): List[(List[String], Expression)] = e match {",
    "      objFields(b) match {",
    "        case Some(fs) if fs.nonEmpty => fs.flatMap { case (k, v) => usePositions(v, fields :+ k) }",
    # P1-05 5d (in slice order): the `cs` emission now builds the fact first (`rt` is added to it), and the module-constant scan also
    # reads the export-declaration blocks of a JS/TS program. Every other `cs` / `const` line survives verbatim.
    '      emit(ujson.Obj("k" -> "cs", "m" -> m.id, "id" -> c.id, "name" -> c.name, "mfn" -> c.methodFullName,',
    '        "out" -> ujson.Arr.from(outRepo), "args" -> ujson.Arr.from(args)))',
    '      val tops = kids(m.block).collect { case a: Call if isOp(a) && isAssign(a) => a }',
    '        usePositions(a, Nil).foreach { case (fs, ee) => emitData("cs", c.id, head, fs, ee) }',
    '        usePositions(e, Nil).foreach { case (fs, ee) => emitData("r", r.id, "ret", fs, ee) }',
    '              emit(ujson.Obj("k" -> "cb", "m" -> m.id, "pk" -> pk, "pid" -> pid, "cond" -> k.id))',
    '    "dataRaw" -> dataRaw, "histogram" -> ujson.Obj.from(histogram.toList.sortBy(_._1).map { case (k, v) => k -> ujson.Num(v) })))',
]


def body(path: Path) -> list[str]:
    """A Scala script without its leading comment lines (the header differs by design: a path and a provenance line)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(lines) if not l.startswith("//"))
    return lines[i:]


# --- the comparator -----------------------------------------------------------------------------------------------


def _facts(tmp_path, name, records):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


STATS = {"k": "stats", "loadMs": "1", "projectionMs": "2", "totalMs": "3", "methods": 2, "cdgRaw": 3, "cdgSelf": 1,
         "dataRaw": 4, "histogram": {"<1ms": 2}}
BASE = [{"k": "m", "id": "1", "name": "a"}, {"k": "d", "m": "1", "th": "a0"}, {"k": "d", "m": "1", "th": "a1"}, STATS]


def test_canonical_ignores_record_order_key_order_and_the_stats_timings(tmp_path):
    shuffled = [BASE[2], {"name": "a", "id": "1", "k": "m"}, BASE[1], dict(STATS, loadMs="999", totalMs="1", histogram={})]
    assert canonical(_facts(tmp_path, "a", BASE)) == canonical(_facts(tmp_path, "b", shuffled))


@pytest.mark.parametrize("mutate", [
    lambda r: r[:1] + r[2:],                                                   # a dropped record
    lambda r: r[:3] + [r[2]] + r[3:],                                          # a duplicated record (a set would hide it)
    lambda r: [dict(r[0], name="b")] + r[1:],                                  # a changed value
    lambda r: r[:3] + [dict(STATS, methods=3)],                                # a changed count
    lambda r: r[:3] + [dict(STATS, cdgSelf=0)],
    lambda r: r[:1] + [dict(r[1], th="a9")] + r[2:],
], ids=["dropped", "duplicated", "changed-value", "methods-count", "cdgSelf-count", "changed-field"])
def test_canonical_tells_a_real_difference_from_noise(tmp_path, mutate):
    assert canonical(_facts(tmp_path, "a", BASE)) != canonical(_facts(tmp_path, "b", mutate(BASE)))


#: every record kind `extract.sc` emits, written here from the script's `"k" ->` literals and checked against the
#: oracle's own kinds below, so a kind the comparator ignored (or a new one) turns a test red.
KINDS = ("m", "p", "c", "r", "cs", "const", "deco", "d", "cb", "t", "f")
ONE_OF_EACH = [{"k": k, "id": f"id-{k}", "v": k} for k in KINDS]


def test_the_kind_list_is_the_one_the_script_emits_and_the_oracle_carries():
    emitted = set(re.findall(r'"k" -> "(\w+)"', (run.HERE / "extract.sc").read_text()))
    assert emitted - {"stats"} == set(KINDS)
    for tag in SLICE:
        p = S14A / tag / "cpg"
        found = sorted(p.glob("*.facts.jsonl")) if p.is_dir() else []
        if found:
            seen = {json.loads(l)["k"] for l in found[0].read_text().splitlines()}
            assert seen - {"stats"} <= set(KINDS), seen


@pytest.mark.parametrize("kind", KINDS)
def test_a_dropped_record_of_every_kind_is_a_difference(tmp_path, kind):
    full = ONE_OF_EACH + [STATS]
    cut = [r for r in full if r["k"] != kind]
    assert len(cut) == len(full) - 1
    assert canonical(_facts(tmp_path, "a", full)) != canonical(_facts(tmp_path, "b", cut))


@pytest.mark.parametrize("kind", KINDS)
def test_a_changed_value_in_a_record_of_every_kind_is_a_difference(tmp_path, kind):
    full = ONE_OF_EACH + [STATS]
    changed = [dict(r, v="other") if r["k"] == kind else r for r in full]
    assert canonical(_facts(tmp_path, "a", full)) != canonical(_facts(tmp_path, "b", changed))


def test_canonical_refuses_a_file_without_or_with_two_stats_records(tmp_path):
    with pytest.raises(AssertionError):
        canonical(_facts(tmp_path, "a", BASE[:3]))
    with pytest.raises(AssertionError):
        canonical(_facts(tmp_path, "b", BASE + [STATS]))


# --- tier 1: no Docker --------------------------------------------------------------------------------------------


@pytest.mark.slice
@pytest.mark.parametrize("tag", sorted(SLICE))
def test_the_oracle_is_the_artifact_recorded_with_its_sha256_and_counts(tag):
    p = oracle(tag)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == SLICE[tag][2]
    _, stats = canonical(p)
    assert stats == SLICE[tag][3]


@pytest.mark.slice
@pytest.mark.parametrize("tag", sorted(SLICE))
def test_the_slice_units_have_no_same_language_descendant_so_staging_changes_nothing(tag):
    doc = json.loads(GOLDEN.read_text())
    units = [Unit(u["unitId"], u["repoId"], u["rootPath"], u["language"]) for u in doc["units"]]
    assert unit_of(tag) in units
    assert run.descendant_roots(unit_of(tag), units) == []


@pytest.mark.slice
@pytest.mark.parametrize("tag", sorted(SLICE))
def test_the_ported_stager_builds_the_slices_stage_tree_at_the_pinned_commit(tag, tmp_path):
    oracle(tag)
    need_commit()
    slice_stage = S14A / tag / "stage"
    if not slice_stage.is_dir():
        pytest.skip(f"slice stage absent: {slice_stage} (UNRUN, not green)")
    n = run.stage_unit(REPO, PINNED, unit_of(tag), [], tmp_path / "stage")
    assert n == SLICE[tag][4]
    cmp = filecmp.dircmp(slice_stage, tmp_path / "stage")

    def same(c) -> bool:
        return not (c.left_only or c.right_only or c.diff_files or c.funny_files) and all(same(s) for s in c.subdirs.values())

    assert same(filecmp.dircmp(slice_stage, tmp_path / "stage")), (cmp.left_only, cmp.right_only, cmp.diff_files)


@pytest.mark.slice
def test_the_ported_astgen_body_is_the_slices_unchanged():
    slice_file = SPIKE / "astgen.sc"
    if not slice_file.is_file():
        pytest.skip(f"{slice_file} absent (UNRUN, not green)")
    assert body(run.HERE / "astgen.sc") == body(slice_file)


@pytest.mark.slice
def test_the_ported_extract_body_differs_from_the_slices_only_in_the_projection_lines():
    """P1-09: `extract.sc` is the slice's plus the projection fix. The slice lines it no longer contains are exactly the 17 above
    (14 of the projection: `defsOf`, `objFields`, `usePositions` and its two callers, the `cb` emission, the stats line; 3 of 5d: the
    `cs` emission that gains `rt`, the constant scan that reads export blocks); every other slice line survives verbatim, so an edit to the fact kinds `m p c r cs const deco t f` shows up here."""
    slice_file = SPIKE / "extract.sc"
    if not slice_file.is_file():
        pytest.skip(f"{slice_file} absent (UNRUN, not green)")
    ported = set(body(run.HERE / "extract.sc"))
    gone = [l for l in body(slice_file) if l not in ported]
    assert gone == REPLACED_SLICE_LINES


def test_the_body_comparison_sees_an_edit(tmp_path):
    a = tmp_path / "a.sc"
    a.write_text("// header\n//\nimport x\n@main def main() = 1\n")
    b = tmp_path / "b.sc"
    b.write_text("// other header\nimport x\n@main def main() = 2\n")
    assert body(a) != body(b)
    b.write_text("// other header\nimport x\n@main def main() = 1\n")
    assert body(a) == body(b)


#: the fixed projection's raw facts for each slice unit, produced by a bounded `joern --script extract.sc` on the slice's own CPG
#: (`cpg/.ccg-work/p1-09/fixed-<tag>.facts.jsonl`, gitignored); the Docker-free stand-in for the live comparison below.
FIXED = atrest.WORK_ROOT / "p1-09"


@pytest.mark.slice
@pytest.mark.parametrize("tag", sorted(SLICE))
def test_the_fixed_projection_keeps_every_non_projection_fact_of_the_slice(tag):
    fixed = FIXED / f"fixed-{tag}.facts.jsonl"
    if not fixed.is_file():
        pytest.skip(f"fixed-projection facts absent: {fixed} (UNRUN, not green)")
    stamp = fixed.with_name(fixed.name + ".sha256")
    assert stamp.is_file() and stamp.read_text().strip() == hashlib.sha256(run.EXTRACT_SC.read_bytes()).hexdigest(), \
        f"{fixed.name} was not produced by the current extract.sc: regenerate it (extractors/joern/README.md)"
    got, got_stats = canonical(fixed, PROJECTION_KINDS, overlay=tag)
    want, want_stats = canonical(oracle(tag), PROJECTION_KINDS)
    assert got == want and len(got) > 0
    keep = [k for k in COUNT_KEYS if k != PROJECTION_COUNT]
    assert {k: got_stats[k] for k in keep} == {k: want_stats[k] for k in keep} == {k: SLICE[tag][3][k] for k in keep}
    assert got_stats[PROJECTION_COUNT] > want_stats[PROJECTION_COUNT]


# --- the 5d overlay: `rt` on `cs` facts and the export-block `const` facts (`tests/overlay_5d.py`) ----------------------------


def _write(tmp_path, name, records):
    return _facts(tmp_path, name, records + [STATS])


CS = {"k": "cs", "id": "10", "m": "1", "name": "f", "dt": "DYNAMIC_DISPATCH", "in": [], "out": [], "args": []}


def test_the_overlay_removes_rt_from_cs_facts_and_nothing_else(tmp_path):
    live = _write(tmp_path, "live", [dict(CS, rt="ANY")])
    assert canonical(live, overlay="fcs") == canonical(_write(tmp_path, "oracle", [CS]))


@pytest.mark.parametrize("change", [
    lambda c: dict(c, name="g"), lambda c: dict(c, dt="STATIC_DISPATCH"), lambda c: dict(c, **{"in": [{"id": "2", "full": "x"}]}),
], ids=["name", "dispatch", "in"])
def test_the_overlay_still_sees_any_other_difference_in_a_cs_fact(tmp_path, change):
    live = _write(tmp_path, "live", [change(dict(CS, rt="ANY"))])
    assert canonical(live, overlay="fcs") != canonical(_write(tmp_path, "oracle", [CS]))


def test_the_overlay_removes_exactly_the_listed_const_facts(tmp_path):
    listed = overlay_5d.extras()["units"]["sp"]["constFacts"]
    assert len(listed) == 8
    other = {"k": "const", "m": "1", "name": "NOT_LISTED", "l": 1, "e": {"o": "lit", "c": "1"}}
    live = _write(tmp_path, "live", [dict(f) for f in listed] + [other])
    assert canonical(live, overlay="sp") == canonical(_write(tmp_path, "oracle", [other]))          # an unlisted const is a difference left in
    assert canonical(live, overlay="sp") != canonical(_write(tmp_path, "empty", []))


def test_a_listed_const_fact_that_the_live_facts_lack_is_an_error_not_a_pass(tmp_path):
    listed = overlay_5d.extras()["units"]["sp"]["constFacts"]
    live = _write(tmp_path, "live", [dict(f) for f in listed[:-1]])
    with pytest.raises(AssertionError, match="listed const facts missing"):
        canonical(live, overlay="sp")


def test_a_listed_const_fact_is_removed_once_so_a_duplicate_stays_a_difference(tmp_path):
    f = overlay_5d.extras()["units"]["sp"]["constFacts"][0]
    live = _write(tmp_path, "live", [dict(x) for x in overlay_5d.extras()["units"]["sp"]["constFacts"]] + [dict(f)])
    assert canonical(live, overlay="sp") != canonical(_write(tmp_path, "empty", []))


def test_without_the_overlay_an_rt_is_a_difference(tmp_path):
    assert canonical(_write(tmp_path, "a", [dict(CS, rt="ANY")])) != canonical(_write(tmp_path, "b", [CS]))


@pytest.mark.slice
@pytest.mark.parametrize("tag", sorted(SLICE))
def test_the_hand_listed_rt_of_every_site_is_what_the_live_script_wrote(tag):
    """The table that feeds `assemble` (`tests/data/p1_05_5d_extras.json`) against the facts of the fixed script run on the stored
    CPG (`.ccg-work/p1-09/fixed-<tag>.facts.jsonl`): same fact ids, same `rt`, and every listed `const` fact is there."""
    fixed = FIXED / f"fixed-{tag}.facts.jsonl"
    if not fixed.is_file():
        pytest.skip(f"fixed-projection facts absent: {fixed} (UNRUN, not green)")
    stamp = fixed.with_name(fixed.name + ".sha256")
    assert stamp.is_file() and stamp.read_text().strip() == hashlib.sha256(run.EXTRACT_SC.read_bytes()).hexdigest(), \
        f"{fixed.name} was not produced by the current extract.sc: regenerate it (extractors/joern/README.md)"
    recs = [json.loads(l) for l in fixed.read_text(encoding="utf-8").splitlines()]
    live = overlay_5d.rt_of(recs)
    assert {o["dt"] for o in recs if o["k"] == "cs" and "rt" in o} == {"DYNAMIC_DISPATCH"}      # a static call carries no `rt`
    table = overlay_5d.rt_table(tag)
    assert {k: live.get(k) for k in table} == table
    assert (len(table) > 0) == (tag == "fcs")                  # `salesperson`: no name-linked site, so no rt is needed
    consts = {json.dumps(r, sort_keys=True) for r in recs if r["k"] == "const"}
    assert all(json.dumps(f, sort_keys=True) in consts for f in overlay_5d.extras()["units"][tag]["constFacts"])


# --- tier 2: Docker -----------------------------------------------------------------------------------------------


#: P1-50 (5d gate F4): per unit, the number of `cs` facts the live script tags with `rt` and the sha256 of the sorted `(fact id, rt)` list, as the live
#: run below derives them (the table lists only the 1,479 `falkor-chat/server` sites whose targets `rt` does not back; the script tags every dynamic call, 22,805 there).
RT_PINS = {"fcs": (22805, "c0a5329bd37e0c90b08f30e7314adfdfbbd9ec2792ea30e7dadbd8929c5e9df8"),
           "sp": (448, "bda5b6a751bc8e5c85523a253957cb1640d5e96c785ca11aaa2561b2df827715")}


@pytest.mark.slice
@pytest.mark.live
@pytest.mark.parametrize("tag", sorted(SLICE))
def test_the_ported_extractor_reproduces_the_slices_raw_facts(tag, tmp_path, monkeypatch):
    monkeypatch.setenv("CCG_AUDIT_DIR", str(tmp_path / "audit"))     # never the real cpg/.audit/sandbox.jsonl
    expected = oracle(tag)
    need_commit()
    area = tmp_path / ".ccg-work"
    area.mkdir()
    rf = run.extract_unit(REPO, unit_of(tag), [unit_of(tag)], PINNED, area / tag, area=area)
    got, got_stats = canonical(rf.facts, PROJECTION_KINDS, overlay=tag)
    want, want_stats = canonical(expected, PROJECTION_KINDS)
    assert {k: v for k, v in got_stats.items() if k != PROJECTION_COUNT} == {k: v for k, v in want_stats.items() if k != PROJECTION_COUNT}
    assert want_stats == SLICE[tag][3]
    assert len(got) == len(want)
    assert got == want
    live = [json.loads(x) for x in rf.facts.read_text(encoding="utf-8").splitlines()]
    rt_live = overlay_5d.rt_of(live)
    table = overlay_5d.rt_table(tag)
    assert {i: rt_live.get(i) for i in table} == table, "the live `rt` of a listed site differs from the hand-listed table"
    assert (len(rt_live), hashlib.sha256(json.dumps(sorted(rt_live.items()), separators=(",", ":")).encode()).hexdigest()) == RT_PINS[tag], \
        "the live run's `rt`-bearing facts changed: a wrong receiver base on a site outside the table is otherwise invisible"
    assert rf.source_files == SLICE[tag][4]
    assert len(rf.sandbox_audit_ids) == (3 if SLICE[tag][1] == "javascript" else 2)
