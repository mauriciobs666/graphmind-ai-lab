"""P1-05 5d: the recorded draw for the `self` rule's hand audit (`tests/data/p1_05_5d_self_audit_draw.json`).

The audit itself is the `data-scientist`'s independent gate (plan § 5d, "`self` rule acceptance"); this only pins that the file the
auditor starts from is the pinned draw: `random.Random(20261119).sample` over the sorted population, at a recorded commit, with the
script and normalizer it was rebuilt with. It does not (and cannot) re-derive the population without the rebuilt bundles.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from extractors.joern import run
from extractors.joern.normalize import assemble_bundle

DRAW = json.loads((Path(__file__).resolve().parent / "data" / "p1_05_5d_self_audit_draw.json").read_text(encoding="utf-8"))


def test_the_drawn_ids_are_the_seeded_sample_of_the_sorted_population():
    pop = DRAW["populationIds"]
    assert pop == sorted(pop) and len(set(pop)) == len(pop) == DRAW["population"]
    assert [d["callSiteId"] for d in DRAW["drawn"]] == random.Random(DRAW["seed"]).sample(pop, DRAW["drawSize"])
    assert DRAW["seed"] == 20261119 and DRAW["drawSize"] == len(DRAW["drawn"]) == 30 == len({d["callSiteId"] for d in DRAW["drawn"]})


def test_the_population_is_the_sum_of_the_per_unit_counts():
    assert sum(u["selfResolved"] for u in DRAW["units"].values()) == DRAW["population"]
    assert all(d["unitId"] in DRAW["units"] for d in DRAW["drawn"])


def test_the_draw_names_the_commit_the_script_and_the_normalizer_it_was_rebuilt_with():
    assert re.fullmatch(r"[0-9a-f]{40}", DRAW["builtCommit"])
    assert DRAW["normalizerVersion"] == assemble_bundle.NORMALIZER_VERSION
    assert DRAW["extractScriptSha256"] == hashlib.sha256(run.EXTRACT_SC.read_bytes()).hexdigest(), \
        "extract.sc changed since the draw: rebuild the population and redraw (ids are content hashes)"


def test_every_drawn_row_is_a_self_call_to_a_method_of_the_enclosing_class():
    for d in DRAW["drawn"]:
        cls = d["enclosing"].rsplit(".", 1)[0]                           # `path::Class.method` -> `path::Class`
        assert d["confirmedTarget"] == f"{cls}.{d['callee']}", d
        assert d["path"] == d["enclosing"].split("::")[0] and d["line"] > 0
