"""P1-09 projection goldens: the fixed `extract.sc` projection, judged in the scorer's vocabulary.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` P1-09; rules: R1 (`defsOf` follows REACHING_DEF from an earlier same-name
non-target identifier), R2 (constructor receiver defined by the constructor call), R3 as a projection rule for the frontend's
iterator-protocol calls (with its negatives), the python causes listed in the plan (comprehension/generator scopes, a call that is
itself a CDG source, dict/array/spread literals) and 4b synthetic-cond transparency (`-p1-ml.md` §3.4).

Two tiers. Tier 1 (always): `tests/golden/projection/facts.<lang>.jsonl` is the raw-facts output of `extract.sc` on the golden fixtures
(Joern 4.0.579, recorded by `cpg/.ccg-work/p1-09/tools/build_goldens.py`); every golden pair is judged from it with
`s28_core` (`build_method`, `resolve_endpoint`, `reach`), so a pair is "connected" exactly as P0's scorer says. `CCG_PROJ_FACTS_PY` /
`CCG_PROJ_FACTS_JS` point the same assertions at another facts file (the mutation runs: a mutant `extract.sc` is run by hand on the
staged fixtures and its facts are judged here). Tier 2 (`live`, skipped unless `CCG_PROJ_LIVE=1`; **a skip is unrun, never green**):
the fixtures are rebuilt from the goldens, run through a bounded Joern (`JAVA_OPTS=-Xmx1500m`, `timeout -s KILL 150` each) and the
resulting facts compared with the stored ones. It cannot run under a low `ulimit -v` (the JVM reserves address space).

Run: `cd cpg && .venv/bin/python -m pytest tests/test_projection_golden.py -q`.
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CPG = Path(__file__).resolve().parent.parent
GOLD = CPG / "tests" / "golden" / "projection"
sys.path.insert(0, str(CPG / "spikes" / "frontend-fitness"))
import s28_core as C  # noqa: E402

BEHAVIOURS = ["r1-reaching-def", "r2-ctor-receiver", "r3-synthetic-call", "call-point-cond", "comprehension-scope",
              "container-literal", "bound-method-alias", "cond-transparency", "for-header-join", "paren-join"]
FACTS = {"python": Path(os.environ.get("CCG_PROJ_FACTS_PY") or GOLD / "facts.python.jsonl"),
         "javascript": Path(os.environ.get("CCG_PROJ_FACTS_JS") or GOLD / "facts.javascript.jsonl")}


def golden(name: str) -> dict:
    return json.loads((GOLD / f"{name}.json").read_text(encoding="utf-8"))


def fixture_text(fx: dict) -> str:
    """The fixture file as staged: `startLine - 1` leading lines (blank, `class C:` first when classHeader), then the lines."""
    pad = [""] * (fx["startLine"] - 1)
    if fx["classHeader"]:
        pad[0] = "class C:"
    return "\n".join(pad + fx["lines"]) + "\n"


class Judge:
    """facts + staged sources for one language, built once."""

    def __init__(self, lang: str, tmp: Path, facts: Path | None = None):
        self.lang = lang
        self.F = C.load_facts(facts or FACTS[lang], lang)
        self.src = tmp / lang
        self.src.mkdir(parents=True, exist_ok=True)
        for b in BEHAVIOURS:
            for fx in golden(b)["fixtures"]:
                if fx["lang"] == lang:
                    (self.src / fx["name"]).write_text(fixture_text(fx), encoding="utf-8")
        self.S = C.Src(self.src)
        self._m: dict[str, list] = {}

    def methods(self, fn: str):
        if fn not in self._m:
            self._m[fn] = [C.build_method(self.lang, self.F, self.S, mid, C.file_of(self.F, m))
                           for mid, m in self.F["m"].items() if m["file"] == fn]
        return self._m[fn]

    def status(self, p: dict, fn_lang: str):
        lo, hi = p["span"]
        ms = [M for M in self.methods(p["fixture"]) if M.l <= hi and M.le >= lo - 1]
        for M in sorted(ms, key=lambda M: M.le - M.l):
            a, _ = C.resolve_endpoint(M, p["source"])
            b, _ = C.resolve_endpoint(M, p["sink"])
            if a is None or b is None:
                continue
            if p["kind"] == "data":
                return ((a, b) in M.data) if p["direct"] else (C.reach(M, a, b) is not None)
            return (b, a) in M.ctl
        return None


@pytest.fixture(scope="module")
def judges(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("proj")
    return {lang: Judge(lang, tmp) for lang in FACTS}


def _cases():
    out = []
    for b in BEHAVIOURS:
        d = golden(b)
        lang_of = {fx["name"]: fx["lang"] for fx in d["fixtures"]}
        for i, p in enumerate(d["pairs"]):
            out.append(pytest.param(b, p, lang_of[p["fixture"]], id=f"{b}-{i:02d}-{p['fixture'][:24]}"))
    return out


@pytest.mark.parametrize("behaviour,pair,lang", _cases())
def test_golden_pair(judges, behaviour, pair, lang):
    got = judges[lang].status(pair, lang)
    assert got is not None, f"endpoint unresolved in the facts: {pair['source']} -> {pair['sink']}"
    assert got == pair["connected"], (
        f"{behaviour}: {pair['cause']} {pair.get('note', '')} {json.dumps(pair['source'])} -> {json.dumps(pair['sink'])}: "
        f"expected connected={pair['connected']}")


def test_the_stored_facts_were_produced_by_the_current_extract_script():
    """The goldens judge RECORDED facts; this ties them to `extract.sc`: an edit of the script turns it red until the facts are regenerated
    (recipe: `extractors/joern/README.md`, "Regenerating the recorded facts") and the stamp rewritten."""
    import hashlib
    stamp = (GOLD / "extract.sc.sha256").read_text(encoding="utf-8").strip()
    now = hashlib.sha256((CPG / "extractors" / "joern" / "extract.sc").read_bytes()).hexdigest()
    assert stamp == now, "extract.sc changed since the golden facts were recorded: regenerate facts.*.jsonl and extract.sc.sha256"


def test_behaviour_sizes_are_the_plan_sets():
    """The goldens cover the sets the plan names (counts are of the goldens, not of a corpus run)."""
    n = {b: collections.Counter(p["cause"].split(":")[0] if p["cause"] != "hand" else "hand" for p in golden(b)["pairs"]) for b in BEHAVIOURS}
    assert n["comprehension-scope"]["projection"] == 12          # the 12 comprehension/generator-scope pairs
    assert n["call-point-cond"]["projection"] == 10              # the 10 "call is itself a CDG source" pairs
    assert n["container-literal"]["projection"] == 4             # dict/array/spread literal pairs
    assert sum(1 for p in golden("bound-method-alias")["pairs"] if not p["connected"]) == 1  # recorded not fixed
    # R3: the 17 counterfactual-verified synthetic-call pairs, plus negatives that must stay unconnected
    assert n["r3-synthetic-call"]["projection"] == 17
    assert sum(1 for p in golden("r3-synthetic-call")["pairs"] if not p["connected"]) == 6
    # 4b: the 8 `definition` positives and the one negative (test_repository 4706)
    defs = [p for p in golden("cond-transparency")["pairs"] if p["cause"].startswith("definition")]
    assert sum(p["connected"] for p in defs) == 8 and sum(not p["connected"] for p in defs) == 1


def test_no_projected_edge_ends_at_a_synthetic_cond(judges):
    """4b: in the golden methods no projected `controlledBy` edge ends at a synthetic cond (the closed recognizer list)."""
    d = golden("cond-transparency")
    synth = d["syntheticConds"]
    assert len(synth) >= 10
    j = judges["python"]
    seen = 0
    for sc in synth:
        for M in j.methods(sc["fixture"]):
            ids = {c[0] for c in M.conds if c[1] == sc["line"] and c[2] == C.collapse(sc["code"])}
            for key in ids:
                seen += 1
                ends = [(pt, cd) for pt, cd in M.ctl if cd == key]
                assert not ends, f"{sc}: edge(s) still end at the synthetic cond: {[C.point_label(M, e[0]) for e in ends]}"
    assert seen >= 8, "the synthetic conds were not found in the facts (the check would pass vacuously)"


def test_ungoverned_synthetic_cond_edges_are_dropped(judges):
    """4b ungoverned case: `ungoverned(fn)` has a try-lowering cond with no real cond above it; its points carry no edge to it."""
    j = judges["python"]
    ms = [M for M in j.methods("cond_synth.py") if M.name == "ungoverned"]
    assert len(ms) == 1
    assert not ms[0].ctl, [C.point_label(ms[0], p) for p, _ in ms[0].ctl]


# ------------------------------------------------------------------------------------------------------ committed mutants
#: `extract.sc` mutants (old text occurs once; replaced by new). Each is run through a bounded Joern on the staged fixtures (live tier below)
#: and the goldens judged on the mutant's facts: the behaviours named in `kills` must then have a failing pair. They are the P1-09 mutants of
#: the plan (transparency off kills all 8 `definition` positives) and of the 4f / JS-recognizer conditions of the data-scientist note (C9, C10).
MUTANTS = {
    "transparency_off": ("python", 'if (lang == "python") cdgSources.filter(n => isTryLowering(n) || isCompIterator(n)).map(_.id).toSet else Set.empty',
                         "Set.empty[Long]", {"cond-transparency"}),
    "r1_off": ("python", "              case _ => defsOf(s, fields, seen, depth + 1)", "              case _ => Nil", {"r1-reaching-def"}),
    "r3_off": ("python", "  def isSynthLowering(c: Call): Boolean = {", "  def isSynthLowering(c: Call): Boolean = if (true) false else {", {"r3-synthetic-call"}),
    "call_point_twin_off": ("python", '              if (pk == "c" && pid != k.id && csIds.contains(pid)) cbEdges += (("cs", pid, k.id))', "              ()", {"call-point-cond"}),
    "for_header_data_off": ("python", '      if (code(n) == "iteratorNonEmptyOrException" && !isCompIterator(n)) {', "      if (false) {", {"for-header-join"}),
    "for_header_data_includes_comprehension": ("python", '      if (code(n) == "iteratorNonEmptyOrException" && !isCompIterator(n)) {',
                                               '      if (code(n) == "iteratorNonEmptyOrException") {', {"for-header-join"}),
    # the lowering-temp name regex alone separates `it.next(it)` (receiver at index 1 like the lowering's) from `_iterator_N.next()`
    "js_name_regex_removed": ("javascript", '        case "javascript" => c.name == "next" && r.name.matches("_iterator_\\\\d+")', '        case "javascript" => c.name == "next"', {"r3-synthetic-call"}),
    "js_r3_off": ("javascript", '        case "javascript" => c.name == "next" && r.name.matches("_iterator_\\\\d+")', '        case "javascript" => false', {"r3-synthetic-call"}),
    # a user `it.next()` becomes transparent: receiver at index 0 accepted and the lowering-temp name not required
    "js_user_next_transparent": ("javascript", [('        case "javascript" => c.name == "next" && r.name.matches("_iterator_\\\\d+")', '        case "javascript" => c.name == "next"'),
                                               ('a.argumentIndex == (if (lang == "python") 0 else 1)', "a.argumentIndex == 0")], {"r3-synthetic-call"}),
}


def edits_of(name: str) -> list[tuple[str, str]]:
    """The mutant's (old, new) edits: a single flat (old, new) or an explicit list of pairs."""
    _, *rest, _ = MUTANTS[name]
    return rest[0] if isinstance(rest[0], list) else [(rest[0], rest[1])]


def _failing_behaviours(judges_by_lang: dict) -> set[str]:
    bad = set()
    for b in BEHAVIOURS:
        d = golden(b)
        lang_of = {fx["name"]: fx["lang"] for fx in d["fixtures"]}
        for p in d["pairs"]:
            lang = lang_of[p["fixture"]]
            if judges_by_lang[lang].status(p, lang) != p["connected"]:
                bad.add(b)
    return bad


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_replaces_text_that_occurs_exactly_once(name):
    """A mutant whose old text is absent (or ambiguous) would be a no-op: the live run below would then 'survive' silently."""
    text = (CPG / "extractors" / "joern" / "extract.sc").read_text(encoding="utf-8")
    for old, new in edits_of(name):
        assert text.count(old) == 1 and old != new


def test_for_header_data_mutant_at_fact_level_fails_the_goldens(tmp_path):
    """Always-run counterpart of `for_header_data_off`: the stored facts without the `d` records into a `for` header cond fail the 4f goldens."""
    keep = []
    its = set()
    rows = [json.loads(l) for l in (GOLD / "facts.python.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r["k"] == "c" and r["code"] == "iteratorNonEmptyOrException":
            its.add(r["id"])
    dropped = [r for r in rows if r["k"] == "d" and r["uk"] == "c" and r["uid"] in its]
    assert dropped, "no data record into an iterator cond: the mutant would be a no-op"
    mutant = tmp_path / "mut.python.facts.jsonl"
    mutant.write_text("\n".join(json.dumps(r) for r in rows if r not in dropped) + "\n", encoding="utf-8")
    js = {lang: Judge(lang, tmp_path / "m", mutant if lang == "python" else None) for lang in FACTS}
    assert "for-header-join" in _failing_behaviours(js)


def test_the_js_negative_and_the_comprehension_negative_are_in_the_goldens():
    """C9 / C10: a user `it.next()` stays opaque in TS, and no data flows into a comprehension iterator cond."""
    notes = [p.get("note", "") for b in ("r3-synthetic-call", "for-header-join") for p in golden(b)["pairs"] if not p["connected"]]
    assert any("JS negative" in n for n in notes) and any("COMPREHENSION iterator" in n for n in notes)


# --------------------------------------------------------------------------------------------------------- live tier
def _joern_env():
    home = Path(os.environ.get("JOERN_HOME") or Path.home() / "joern" / "joern-cli")
    return home


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("CCG_PROJ_LIVE") != "1", reason="UNRUN: set CCG_PROJ_LIVE=1 (bounded Joern, no ulimit -v)")
def test_live_extract_reproduces_the_stored_facts(tmp_path):
    home = _joern_env()
    if not (home / "joern-parse").exists():
        pytest.skip(f"UNRUN: no Joern at {home}")
    sc = CPG / "extractors" / "joern" / "extract.sc"
    env = dict(os.environ, JAVA_OPTS="-Xmx1500m")
    for lang, front in (("python", "pythonsrc"), ("javascript", "jssrc")):
        src = tmp_path / f"src-{lang}"
        src.mkdir()
        for b in BEHAVIOURS:
            for fx in golden(b)["fixtures"]:
                if fx["lang"] == lang:
                    (src / fx["name"]).write_text(fixture_text(fx), encoding="utf-8")
        cpg_bin, facts = tmp_path / f"{lang}.cpg.bin", tmp_path / f"{lang}.facts.jsonl"
        subprocess.run(["timeout", "-s", "KILL", "150", str(home / "joern-parse"), str(src), "--language", front, "-o", str(cpg_bin)],
                       check=True, env=env, capture_output=True, cwd=tmp_path)
        subprocess.run(["timeout", "-s", "KILL", "150", str(home / "joern"), "--script", str(sc), "--param", f"cpgFile={cpg_bin}",
                        "--param", f"out={facts}", "--param", f"lang={lang}"], check=True, env=env, capture_output=True, cwd=tmp_path)

        def canon(path):
            return sorted(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines())
                          if r["k"] in {"m", "p", "r", "c", "cs", "d", "cb", "t"})
        assert canon(facts) == canon(GOLD / f"facts.{lang}.jsonl")
        shutil.rmtree(src)


@pytest.fixture(scope="module")
def mutant_cpgs(tmp_path_factory):
    """One bounded `joern-parse` per language of the staged goldens' fixtures (reused by every mutant)."""
    if os.environ.get("CCG_PROJ_LIVE") != "1":
        pytest.skip("UNRUN: set CCG_PROJ_LIVE=1 (bounded Joern, no ulimit -v)")
    home = _joern_env()
    if not (home / "joern-parse").exists():
        pytest.skip(f"UNRUN: no Joern at {home}")
    tmp = tmp_path_factory.mktemp("mutants")
    env = dict(os.environ, JAVA_OPTS="-Xmx1500m")
    out = {}
    for lang, front in (("python", "pythonsrc"), ("javascript", "jssrc")):
        src = tmp / f"src-{lang}"
        src.mkdir()
        for b in BEHAVIOURS:
            for fx in golden(b)["fixtures"]:
                if fx["lang"] == lang:
                    (src / fx["name"]).write_text(fixture_text(fx), encoding="utf-8")
        subprocess.run(["timeout", "-s", "KILL", "150", str(home / "joern-parse"), str(src), "--language", front, "-o", str(tmp / f"{lang}.cpg.bin")],
                       check=True, env=env, capture_output=True, cwd=tmp)
        out[lang] = (src, tmp / f"{lang}.cpg.bin")
    return tmp, out


@pytest.mark.live
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_live_mutant_is_killed(name, mutant_cpgs, tmp_path):
    """Run the mutated `extract.sc` (bounded Joern: -Xmx1500m, timeout -s KILL 150) and judge the goldens on its facts."""
    tmp, cpgs = mutant_cpgs
    lang, kills = MUTANTS[name][0], MUTANTS[name][-1]
    text = (CPG / "extractors" / "joern" / "extract.sc").read_text(encoding="utf-8")
    for old, new in edits_of(name):
        assert text.count(old) == 1
        text = text.replace(old, new)
    sc = tmp_path / "extract.sc"
    sc.write_text(text, encoding="utf-8")
    facts = tmp_path / f"{lang}.facts.jsonl"
    env = dict(os.environ, JAVA_OPTS="-Xmx1500m")
    subprocess.run(["timeout", "-s", "KILL", "150", str(_joern_env() / "joern"), "--script", str(sc), "--param", f"cpgFile={cpgs[lang][1]}",
                    "--param", f"out={facts}", "--param", f"lang={lang}"], check=True, env=env, capture_output=True, cwd=tmp_path)
    judges_by_lang = {lg: Judge(lg, tmp_path / "j", facts if lg == lang else None) for lg in FACTS}
    assert kills <= _failing_behaviours(judges_by_lang), f"mutant {name} survived: {kills} still pass"
