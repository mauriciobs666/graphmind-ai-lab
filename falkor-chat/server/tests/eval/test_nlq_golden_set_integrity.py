"""Structural integrity checks for the NL-query-generation golden set (K-055 M6,
`docs/plans/workflow-nl-query-generation-ml.md` §4, unit U29b).

**Genuinely network/DB-free** — this file checks the JSONL fixture's own
structure (unique ids, valid `dataset`/`shape`/`expected.type` values, size and
per-shape-per-dataset stratification), never live correctness of an individual
answer against `reference`/`ws:nlq-eval`. It needs neither FalkorDB nor a live
LLM to pass, so it runs in the default offline suite. `DATASET_REGISTRY` is
imported directly from `falkorchat.querygen` (a pure module, no I/O) so a stale
hardcoded dataset-key list here can never silently drift from what the shipped
mechanism actually allowlists.

**Known, designed v1 limitation carried in this golden set — read before adding
or reviewing a pair.** The shipped `query_graph_data` mechanism
(`docs/plans/workflow-nl-query-generation.md` §3.1/§3.6) supports exactly one
`MATCH (var:label)` node pattern per request — no relationship-pattern
traversal (`(a)-[:REL]->(b)`) at all. This is a genuine, reviewed v1 scope
decision, not an oversight or a corpus defect. Two shapes in this golden set are
therefore **structurally unanswerable by the shipped mechanism today, on
purpose**:

- `relationship-traversal` (knowledge_base only) — e.g. "who did Marlowe
  Robotics acquire" requires walking a `RELATES_TO` edge; the mechanism cannot
  express that at all.
- `conflicting-facts` (knowledge_base only) — per FR-6, the two conflicting
  values (e.g. Marlowe Robotics' employee count, 62 vs. 140) are modeled as two
  separate `RELATES_TO{label:"has"}` edges to standalone value-entities, not as
  a property on the subject node, so this shape is *also* a relationship-
  traversal question underneath, and equally unreachable by a single-node-
  pattern `MATCH`.

Both shapes are included anyway (per the ml note §6's own guidance) because the
shape taxonomy requires them and a mechanism claiming "arbitrarily phrased"
query support should be measured against its actual limits, not have them
quietly excluded. **A 0% (or near-0%) Layer 1 score on these two shapes is the
expected, correct outcome of a passing run — not a regression, not a corpus
defect, and not something a later scoring/reporting unit should misdiagnose.**
Every pair's own `rationale` field also states this inline.

**Workspace/graph each dataset's pairs assume** — the harness must point at the
right graph per `dataset`, this is not automatic:

- `"catalog"` → `DATASET_REGISTRY["catalog"].graph_key == "reference"`, the
  global, workspace-independent graph.
- `"knowledge_base"` → `DATASET_REGISTRY["knowledge_base"].graph_key is None`,
  which `querygen`'s own module docstring/plan §3.3 documents as resolving to
  `f"ws:{ctx.ws}"` **at call time** — for every pair in this file that means the
  harness must run with `ctx.ws == "nlq-eval"` (the workspace
  `scripts/seed_nlq_eval_corpus.py` populated), not whatever workspace a test
  happens to default to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from falkorchat.querygen import DATASET_REGISTRY

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN_PATH = _EVAL_DIR / "nlq_golden_set.jsonl"

_REQUIRED_FIELDS = ("id", "dataset", "question", "shape", "expected", "rationale")
_VALID_SHAPES = {
    "single-fact",
    "filter-list",
    "not-found",
    "conflicting-facts",
    "aggregation",
    "relationship-traversal",
    "compound-filter",
}
_VALID_EXPECTED_TYPES = {"scalar", "set", "not_found"}

# Shapes expressible per dataset (brief: relationship-traversal/conflicting-facts
# are knowledge_base-only per the corpus; compound-filter is catalog-only per the
# schema — Entity has no numeric property, so min/max aggregation is also
# catalog-only, but plain count-aggregation is common to both, so "aggregation"
# itself is not dataset-restricted).
_DATASET_ONLY_SHAPES = {
    "compound-filter": "catalog",
    "relationship-traversal": "knowledge_base",
    "conflicting-facts": "knowledge_base",
}

# Per-dataset, per-shape minimum pair counts (brief: >=3 per shape per dataset
# where expressible, except conflicting-facts, relaxed to >=2 because the corpus
# has exactly one conflict scenario to draw pairs from).
_SHAPE_MINIMUMS = {
    "catalog": {
        "single-fact": 3,
        "filter-list": 3,
        "not-found": 3,
        "aggregation": 3,
        "compound-filter": 3,
    },
    "knowledge_base": {
        "single-fact": 3,
        "filter-list": 3,
        "not-found": 3,
        "aggregation": 3,
        "relationship-traversal": 3,
        "conflicting-facts": 2,
    },
}

_DATASET_SIZE_RANGES = {
    "catalog": (20, 25),
    "knowledge_base": (15, 20),
}


def _load_golden_set() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not _GOLDEN_PATH.exists():
        return rows
    with _GOLDEN_PATH.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"{_GOLDEN_PATH.name}:{line_no}: invalid JSON — {exc}"
                ) from exc
    return rows


_GOLDEN_ROWS = _load_golden_set()
_GOLDEN_IDS = [row.get("id", f"<row {i}>") for i, row in enumerate(_GOLDEN_ROWS)]


def test_golden_set_file_exists_and_nonempty() -> None:
    assert _GOLDEN_PATH.exists(), f"missing fixture: {_GOLDEN_PATH}"
    assert _GOLDEN_ROWS, f"{_GOLDEN_PATH.name} has no rows"


def test_golden_ids_are_unique() -> None:
    assert len(_GOLDEN_IDS) == len(set(_GOLDEN_IDS)), (
        f"duplicate golden ids found: "
        f"{[i for i in _GOLDEN_IDS if _GOLDEN_IDS.count(i) > 1]}"
    )


@pytest.mark.parametrize("row", _GOLDEN_ROWS, ids=_GOLDEN_IDS)
def test_golden_row_has_required_fields(row: dict[str, Any]) -> None:
    for field in _REQUIRED_FIELDS:
        assert field in row, f"{row.get('id', '?')}: missing field {field!r}"


@pytest.mark.parametrize("row", _GOLDEN_ROWS, ids=_GOLDEN_IDS)
def test_golden_row_dataset_is_registered(row: dict[str, Any]) -> None:
    assert row["dataset"] in DATASET_REGISTRY, (
        f"{row['id']}: dataset {row['dataset']!r} is not a key in "
        f"falkorchat.querygen.DATASET_REGISTRY ({sorted(DATASET_REGISTRY)})"
    )


@pytest.mark.parametrize("row", _GOLDEN_ROWS, ids=_GOLDEN_IDS)
def test_golden_row_shape_is_in_taxonomy(row: dict[str, Any]) -> None:
    assert row["shape"] in _VALID_SHAPES, (
        f"{row['id']}: shape {row['shape']!r} is not one of {_VALID_SHAPES}"
    )


@pytest.mark.parametrize("row", _GOLDEN_ROWS, ids=_GOLDEN_IDS)
def test_golden_row_shape_matches_dataset_restriction(row: dict[str, Any]) -> None:
    shape = row["shape"]
    restricted_to = _DATASET_ONLY_SHAPES.get(shape)
    if restricted_to is not None:
        assert row["dataset"] == restricted_to, (
            f"{row['id']}: shape {shape!r} is only expressible against "
            f"{restricted_to!r} (dataset schema constraint), found "
            f"dataset={row['dataset']!r}"
        )


@pytest.mark.parametrize("row", _GOLDEN_ROWS, ids=_GOLDEN_IDS)
def test_golden_row_expected_type_is_valid(row: dict[str, Any]) -> None:
    expected = row["expected"]
    assert isinstance(expected, dict) and "type" in expected, (
        f"{row['id']}: 'expected' must be an object with a 'type' field"
    )
    assert expected["type"] in _VALID_EXPECTED_TYPES, (
        f"{row['id']}: expected.type {expected['type']!r} is not one of "
        f"{_VALID_EXPECTED_TYPES}"
    )
    if expected["type"] == "scalar":
        assert "value" in expected, f"{row['id']}: scalar expected needs a 'value'"
    if expected["type"] == "set":
        assert isinstance(expected.get("values"), list) and expected["values"], (
            f"{row['id']}: set expected needs a non-empty 'values' list"
        )


@pytest.mark.parametrize("row", _GOLDEN_ROWS, ids=_GOLDEN_IDS)
def test_conflicting_facts_expected_has_at_least_two_values(row: dict[str, Any]) -> None:
    """FR-6 / ml note §3: a conflicting-facts pair's expected answer must itself
    be a set of >=2 values — scoring correct only if the mechanism surfaces all
    of them, never a single silently-picked value."""
    if row["shape"] != "conflicting-facts":
        return
    expected = row["expected"]
    assert expected["type"] == "set", (
        f"{row['id']}: a conflicting-facts pair's expected.type must be 'set', "
        f"found {expected['type']!r}"
    )
    assert len(expected["values"]) >= 2, (
        f"{row['id']}: a conflicting-facts pair needs >=2 expected values, "
        f"found {expected['values']!r}"
    )


@pytest.mark.parametrize("dataset", sorted(_DATASET_SIZE_RANGES))
def test_dataset_size_in_expected_range(dataset: str) -> None:
    lo, hi = _DATASET_SIZE_RANGES[dataset]
    count = sum(1 for row in _GOLDEN_ROWS if row.get("dataset") == dataset)
    assert lo <= count <= hi, (
        f"{dataset}: expected {lo}-{hi} golden pairs, found {count}"
    )


@pytest.mark.parametrize(
    "dataset,shape,minimum",
    [
        (dataset, shape, minimum)
        for dataset, shapes in _SHAPE_MINIMUMS.items()
        for shape, minimum in shapes.items()
    ],
)
def test_shape_stratification_meets_minimum(dataset: str, shape: str, minimum: int) -> None:
    count = sum(
        1
        for row in _GOLDEN_ROWS
        if row.get("dataset") == dataset and row.get("shape") == shape
    )
    assert count >= minimum, (
        f"{dataset}/{shape}: expected >={minimum} golden pairs, found {count}"
    )
