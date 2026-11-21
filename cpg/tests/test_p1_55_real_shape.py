"""P1-55: the keyless-URL rule on a **real** record (`docs/plans/connected-code-graph-p1-linking.md` P1-55; `REVIEW.md`'s real-record rule).

``tests/data/p1_55_unknown_url.json`` is the callsite, ``valueSet``, ``data``, symbol, file and import records of the one stored proving-repository
site whose URL names no literal segment (`decision-models/decisionmodels/lmstudio.py:366`, `urllib.request.Request(f"{self._base_url}{_MODELS_ROUTE}")`),
extracted by Joern and never hand-built; regenerate with ``tests/data/p1_55_unknown_url_regen.py`` (its docstring has the command). The census
below is the guard that the predicate matches at least one real record: a rule that matched none would leave this site `GET /{?}`.
"""
import json
from pathlib import Path

from ccg.packrt import run_packs
from ccg.packrt.httpkey import names_no_literal_segment, normalize_key
from ccg.rulepack import load_pack

CPG = Path(__file__).resolve().parents[1]
RECORDS = json.loads((CPG / "tests" / "data" / "p1_55_unknown_url.json").read_text())["records"]
PY_HTTP = load_pack(CPG / "packs" / "python-http-client" / "pack.yaml")


def test_the_golden_url_valueset_is_two_unknown_pieces():
    (vs,) = [r for r in RECORDS if r["type"] == "valueSet" and r["arg"] == "a0"]
    assert vs["values"] == [{"template": [{"unknown": "runtime-built"}, {"unknown": "runtime-built"}]}]


def test_the_real_unknown_url_site_is_one_keyless_possible_runtime_built_consumer():
    out = run_packs([RECORDS], [PY_HTTP])
    (c,) = out.consumers
    assert (c["siteId"], c["key"], c["status"], c["reason"]) == ("eb54631ae0b54ed9b6ba5f1e480b4ba4", None, "possible", "runtime-built")


def test_census_the_predicate_matches_at_least_one_real_lifted_alternative():
    """Over the real run's own lifted URL alternatives (the extractor's output, not a written string), the predicate matches the site: a
    predicate that matched no real record fails here."""
    out = run_packs([RECORDS], [PY_HTTP])
    matched = [a for l in out.lifted for a in l["alternatives"] if names_no_literal_segment(normalize_key(a, "consumer"))]
    assert matched == ["\x01\x01"]
