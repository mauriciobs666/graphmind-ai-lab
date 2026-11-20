"""P1-50 Part A, a live golden: a real Joern-narrowed `self.m()` with an override elsewhere in the unit.

`tests/golden/self_rule/facts.jsonl` is the raw facts `extract.sc` wrote through live Joern 4.0.579 (pysrc2cpg) over `tests/golden/self_rule/src/`. Joern's
`in` for the three `self.<m>()` calls (recorded below by hand from that file, not computed):

* `Base.run` -> `self.step()`: `in` is `[Base.step]` only, although `Sub(Base).step` overrides it in another file of the unit (Joern did not link it).
  This is the narrowing that made the old rule (count the candidates) confirm a site a subclass can intercept.
* `Base.other` -> `self.only_here()`: `in` is `[Base.only_here]`, and no other class defines it: the rule confirms.
* `P.run` -> `self.step()`: the override `Q.step` is in the same file, so Joern lists both.

The opt-in row (`CCG_SELFRULE_LIVE=1`) re-runs the probe and compares the `in` lists.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from extractors.joern import run as joern_run
from extractors.joern.normalize import assemble_bundle, core
from normalize_assembly_helpers import git_repo

GOLD = Path(__file__).resolve().parent / "golden" / "self_rule"
UUID = "11111111-1111-4111-8111-111111111111"
#: (file, callee) -> Joern's candidate fullNames, as recorded
RECORDED_IN = {
    ("base.py", "step"): ["base.py:<module>.Base.step"],
    ("base.py", "only_here"): ["base.py:<module>.Base.only_here"],
    ("same_file.py", "step"): ["same_file.py:<module>.P.step", "same_file.py:<module>.Q.step"],
}
WANT = {                                                   # (file, callee) -> (status, reason, qname targets)
    ("base.py", "step"): ("possible", "name-match", []),
    ("base.py", "only_here"): ("confirmed", "resolved-call", ["base.py::Base.only_here"]),
    ("same_file.py", "step"): ("possible", "name-match", []),
}


def facts():
    return core.load_facts(GOLD / "facts.jsonl")


def test_the_recorded_candidates_are_what_the_golden_facts_hold():
    f = facts()
    meth = {m["id"]: m for m in f["m"]}
    got = {}
    for c in f["cs"]:
        if c["name"] in ("step", "only_here"):
            got[(meth[c["m"]]["file"], c["name"])] = sorted(x["full"] for x in c["in"])
    assert got == {k: sorted(v) for k, v in RECORDED_IN.items()}


def test_a_real_narrowed_self_call_with_an_override_elsewhere_in_the_unit_stays_possible(tmp_path):
    src = {p.name: p.read_text(encoding="utf-8") for p in (GOLD / "src").glob("*.py")}
    repo = tmp_path / "repo"
    commit = git_repo(repo, src)
    records, report = assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts(), stage=repo, astgen_dir=None, repo=repo,
                                               commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC, built_at="2026-01-01T00:00:00Z")
    syms = {r["id"]: r for r in records if r["type"] == "symbol"}
    got = {}
    for r in records:
        if r["type"] == "callsite" and r["calleeName"] in ("step", "only_here"):
            got[(syms[r["symbolId"]]["path"], r["calleeName"])] = (r["status"], r["reason"], sorted(syms[t["symbolId"]]["qname"] for t in r["targets"]))
    assert got == WANT
    assert len(report["selfResolvedSites"]) == 1


@pytest.mark.skipif(os.environ.get("CCG_SELFRULE_LIVE") != "1", reason="UNRUN: set CCG_SELFRULE_LIVE=1 (bounded local Joern, one JVM)")
def test_the_recorded_facts_are_what_the_shipped_script_writes_now(tmp_path):
    shutil.copytree(GOLD / "src", tmp_path / "src")
    joern_env = Path(__file__).resolve().parent.parent.parent / "skills" / "joern-cpg" / "scripts" / "joern-env.sh"
    script = (f". {joern_env} && cd {tmp_path} && timeout -s KILL 200 joern-parse src --language pythonsrc -o p.cpg.bin >parse.log 2>&1 && "
              f"timeout -s KILL 200 joern --script {joern_run.EXTRACT_SC} --param cpgFile=p.cpg.bin --param out=facts.jsonl --param lang=python >ex.log 2>&1")
    subprocess.run(["bash", "-c", script], check=True, env={**os.environ, "JAVA_OPTS": "-Xmx1500m"})
    key = lambda p: sorted(json.dumps(json.loads(x), sort_keys=True) for x in Path(p).read_text(encoding="utf-8").splitlines() if '"k": "stats"' not in x and '"k":"stats"' not in x)
    assert key(tmp_path / "facts.jsonl") == key(GOLD / "facts.jsonl")
