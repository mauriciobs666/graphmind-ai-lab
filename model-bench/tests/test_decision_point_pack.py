"""typed-decision-models U3b.1 — the decision-point pack contract: plan v1.3 §3.5's `validate_pack`
refusals (§5's U3b row), keyed on the `decisionPoint` key's presence, and the declared per-kind and
side-check metrics (`classification.score_answer`/`aggregate`; plan2 §4.1 U3b).

Offline throughout. Each contract pack is built under `tmp_path` from the U1d.2 fixture's manifest
and prompt, with synthetic `eval` items and the real guard pack's 85 human-curated items as the
`side-check` stratum. Nothing here writes under `model-bench/results/`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from modelbench import packs
from modelbench.decisions import MIN_EVAL_ITEMS
from modelbench.packs import STRATA, load_pack, validate_pack

_TESTS = Path(__file__).resolve().parent
_FIXTURE_PACK = _TESTS / "fixtures" / "packs" / "decision_point_guard"
_SHIPPED_PACKS = _TESTS.parent / "packs"
_REAL_GUARD_PACK = _SHIPPED_PACKS / "guard-judge-understanding"


def _eval_item(n: int, *, expected: bool = False, path: str = "understanding") -> dict[str, Any]:
    """One synthetic `eval` item, its own condition so no input hash repeats."""
    return {
        "itemId": f"ev-{n:03d}",
        "path": path,
        "stratum": "eval",
        "origin": "synthetic",
        "condition": f"generated condition {n}",
        "understanding": {"request": f"request {n}"} if path == "understanding" else {},
        "turns": [] if path == "understanding" else [{"displayName": "A", "text": f"turn {n}"}],
        "expected": expected,
    }


def _side_check_items() -> list[dict[str, Any]]:
    """The real guard pack's 85 human-curated items, stamped `side-check` — and, like the real
    ones, carrying no `origin` (the stratum is exempt from the origin refusal)."""
    rows = list(load_pack(_REAL_GUARD_PACK).iter_items())
    return [{**row, "stratum": "side-check"} for row in rows]


def _contract_pack(
    root: Path,
    *,
    evals: int = MIN_EVAL_ITEMS,
    side_check: bool = True,
    manifest_edit: Callable[[dict[str, Any]], None] | None = None,
    rows_edit: Callable[[list[dict[str, Any]]], None] | None = None,
) -> packs.Pack:
    """A guard decision-point pack under `root`: `evals` synthetic `eval` items, then (by default)
    the 85 side-check items — v1.3 §5's accepted shape at the defaults."""
    pack_dir = root / "pack"
    shutil.copytree(_FIXTURE_PACK, pack_dir)
    manifest = json.loads((pack_dir / "pack.json").read_text())
    if manifest_edit is not None:
        manifest_edit(manifest)
    (pack_dir / "pack.json").write_text(json.dumps(manifest))
    rows = [_eval_item(n, expected=n % 2 == 0) for n in range(evals)]
    if side_check:
        rows.extend(_side_check_items())
    if rows_edit is not None:
        rows_edit(rows)
    (pack_dir / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return load_pack(pack_dir)


def _only(problems: list[str], needle: str) -> str:
    """The one problem naming `needle`, with nothing else wrong — each refusal is tested alone."""
    assert len(problems) == 1, problems
    assert needle in problems[0], problems
    return problems[0]


# ==================================================================================================
# 1. The accepted shape, and the keying on `decisionPoint`
# ==================================================================================================


def test_200_eval_items_and_the_85_item_side_check_are_accepted(tmp_path):
    """v1.3 §5's U3b row: "It accepts 200 eval + 85 side-check items"."""
    pack = _contract_pack(tmp_path)
    assert sum(1 for _ in pack.iter_items()) == MIN_EVAL_ITEMS + 85
    assert validate_pack(pack) == []


def test_every_shipped_pack_validates_unchanged_and_none_is_a_decision_point_pack():
    """The refusals key on `decisionPoint`'s presence (v1.3 §3.5), so no pack shipped before it —
    none carries the key, labels or strata — gains a problem."""
    roots = sorted(p for p in _SHIPPED_PACKS.iterdir() if (p / "pack.json").is_file())
    assert len(roots) == 5
    for root in roots:
        pack = load_pack(root)
        assert "decisionPoint" not in pack.manifest, root.name
        assert validate_pack(pack) == [], root.name


def test_the_decision_point_key_alone_switches_every_refusal_on(tmp_path):
    """The real guard pack validates without the key, and with it is refused on each of the six
    rules it breaks — even with an empty block, since no refusal keys on a field inside it."""
    pack_dir = tmp_path / "guard"
    shutil.copytree(_REAL_GUARD_PACK, pack_dir)
    assert validate_pack(load_pack(pack_dir)) == []
    manifest = json.loads((pack_dir / "pack.json").read_text())
    manifest["decisionPoint"] = {}
    (pack_dir / "pack.json").write_text(json.dumps(manifest))
    problems = validate_pack(load_pack(pack_dir))
    for needle in (
        "decisionPoint.incumbent is absent",
        "needs a labels block",
        "85 item(s) have no stratum",
        "0 item(s) in stratum 'eval'",
        "needs its 'side-check' stratum",
        "metrics.verdictMetrics must be exactly",
    ):
        assert sum(needle in p for p in problems) == 1, (needle, problems)
    assert len(problems) == 6, problems


@pytest.mark.parametrize("block", [None, [], "guard"])
def test_a_decision_point_value_that_is_no_object_is_refused_and_still_keys_the_contract(
    tmp_path, block
):
    def edit(manifest):
        manifest["decisionPoint"] = block

    problems = validate_pack(_contract_pack(tmp_path, manifest_edit=edit))
    assert any("decisionPoint must be an object" in p for p in problems), problems
    assert any("decisionPoint.incumbent is absent" in p for p in problems), problems


# ==================================================================================================
# 2. Each refusal, alone (v1.3 §3.5; §5's U3b row)
# ==================================================================================================


def test_199_eval_items_are_refused(tmp_path):
    """FR-11, AC-13: the floor is `MIN_EVAL_ITEMS`, inclusive — 200 passes (above), 199 does not."""
    problem = _only(validate_pack(_contract_pack(tmp_path, evals=199)), "MIN_EVAL_ITEMS")
    assert "199 item(s) in stratum 'eval'" in problem and "AC-13" in problem


def test_side_check_items_never_count_toward_the_eval_floor(tmp_path):
    """FR-15: 199 `eval` items and 85 side-check ones are 284 items, still under the floor."""
    pack = _contract_pack(tmp_path, evals=199)
    assert sum(1 for _ in pack.iter_items()) == 284
    _only(validate_pack(pack), "199 item(s) in stratum 'eval'")


def test_the_labeler_being_the_incumbent_is_refused(tmp_path):
    def edit(manifest):
        manifest["labels"]["labeler"]["model"] = manifest["decisionPoint"]["incumbent"]

    _only(validate_pack(_contract_pack(tmp_path, manifest_edit=edit)), "AC-4")


@pytest.mark.parametrize("inputs", ["real", "mixed", "", None])
def test_labels_inputs_other_than_synthetic_is_refused(tmp_path, inputs):
    def edit(manifest):
        manifest["labels"]["inputs"] = inputs

    _only(validate_pack(_contract_pack(tmp_path, manifest_edit=edit)), "labels.inputs")


@pytest.mark.parametrize("origin", ["human", "production", "", None])
def test_an_eval_item_with_an_origin_other_than_synthetic_is_refused(tmp_path, origin):
    def edit(rows):
        rows[7]["origin"] = origin

    problem = _only(validate_pack(_contract_pack(tmp_path, rows_edit=edit)), "AC-7")
    assert "1 'eval' item(s)" in problem and "'ev-007'" in problem


def test_a_side_check_item_is_exempt_from_the_origin_refusal_by_stratum(tmp_path):
    """v1.3 §3.5: the side-check stratum is human-curated, so any origin (or none) is fine."""

    def edit(rows):
        for row in rows:
            if row["stratum"] == "side-check":
                row["origin"] = "human"

    assert validate_pack(_contract_pack(tmp_path, rows_edit=edit)) == []


def test_a_missing_labels_block_is_refused(tmp_path):
    def edit(manifest):
        del manifest["labels"]

    _only(validate_pack(_contract_pack(tmp_path, manifest_edit=edit)), "needs a labels block")


@pytest.mark.parametrize("source", ["human", "llm-reviewed", "", None])
def test_a_labels_source_other_than_llm_unreviewed_is_refused(tmp_path, source):
    def edit(manifest):
        manifest["labels"]["source"] = source

    _only(validate_pack(_contract_pack(tmp_path, manifest_edit=edit)), "labels.source")


def test_an_item_with_no_stratum_is_refused_and_never_counts_as_eval(tmp_path):
    """No default (v1.3 §3.5): a stratum-less item is refused, and is not an `eval` item — at
    exactly 200 items, dropping one stratum also takes the pack under the floor."""

    def edit(rows):
        del rows[3]["stratum"]

    problems = validate_pack(_contract_pack(tmp_path, rows_edit=edit))
    assert len(problems) == 2, problems
    assert "1 item(s) have no stratum" in problems[0] and "'ev-003'" in problems[0]
    assert "199 item(s) in stratum 'eval'" in problems[1]


def test_a_stratum_outside_the_two_is_refused(tmp_path):
    def edit(rows):
        rows.append({**_eval_item(999), "stratum": "calibration"})

    problem = _only(validate_pack(_contract_pack(tmp_path, rows_edit=edit)), "stratum outside")
    assert "'ev-999'" in problem


def test_strata_is_exactly_eval_and_side_check():
    """Bound to v1.3 §3.5's set, both ways (AGENTS constant rule): a shrink reddens the accepted
    shape above, a widen reddens `test_a_stratum_outside_the_two_is_refused`."""
    assert STRATA == frozenset({"eval", "side-check"})


@pytest.mark.parametrize("incumbent", [None, ""])
def test_a_null_or_empty_incumbent_on_guard_judge_is_refused(tmp_path, incumbent):
    """`BAR_KIND_BY_ROLE["guard-judge"] == "speed-up"`: the speed-up bar needs an incumbent."""

    def edit(manifest):
        manifest["decisionPoint"]["incumbent"] = incumbent

    _only(validate_pack(_contract_pack(tmp_path, manifest_edit=edit)), "speed-up point")


def test_a_guard_pack_with_no_side_check_is_refused(tmp_path):
    problem = _only(validate_pack(_contract_pack(tmp_path, side_check=False)), "side-check")
    assert "AC-17" in problem


def test_a_role_with_no_bar_kind_is_refused(tmp_path):
    """A pack cannot pick its own bar: a role absent from `BAR_KIND_BY_ROLE` is no decision point.
    `chat-responder` is a known role whose sampling contract still holds on `itemId`."""

    def edit(manifest):
        manifest["role"] = "chat-responder"
        manifest.pop("scorer")

    problems = validate_pack(_contract_pack(tmp_path, side_check=False, manifest_edit=edit))
    _only([p for p in problems if "BAR_KIND_BY_ROLE" in p], "is not a decision point")


@pytest.fixture
def new_ability_join_in(monkeypatch):
    """U4 adds the `new-ability` rows; until then a test adds one to `packs`' own reference so the
    `new-ability` and other-role branches can be exercised now."""
    monkeypatch.setattr(
        packs, "BAR_KIND_BY_ROLE", {"guard-judge": "speed-up", "join-in-judge": "new-ability"}
    )


def _join_in(manifest):
    manifest["role"] = "join-in-judge"
    manifest.pop("scorer")
    manifest["decisionPoint"]["incumbent"] = None


def _decision_problems(pack) -> list[str]:
    """`join-in-judge` is not yet a model-bench role, so the sampling axis also objects; keep
    only the decision-point contract's own problems."""
    return packs._decision_point_problems(pack)


def test_a_new_ability_pack_with_a_null_incumbent_and_no_side_check_passes(
    tmp_path, new_ability_join_in
):
    pack = _contract_pack(tmp_path, side_check=False, manifest_edit=_join_in)
    assert _decision_problems(pack) == []


def test_a_non_null_incumbent_on_a_new_ability_role_is_refused(tmp_path, new_ability_join_in):
    def edit(manifest):
        _join_in(manifest)
        manifest["decisionPoint"]["incumbent"] = "qwen/qwen3-4b-2507"

    pack = _contract_pack(tmp_path, side_check=False, manifest_edit=edit)
    _only(_decision_problems(pack), "new-ability point and has none")


def test_deleting_the_incumbent_on_a_new_ability_role_still_refuses(tmp_path, new_ability_join_in):
    """Absent is not null: a new-ability pack states `incumbent: null`, never omits it."""

    def edit(manifest):
        _join_in(manifest)
        del manifest["decisionPoint"]["incumbent"]

    pack = _contract_pack(tmp_path, side_check=False, manifest_edit=edit)
    _only(_decision_problems(pack), "decisionPoint.incumbent is absent")


def test_a_side_check_stratum_on_any_other_role_is_refused(tmp_path, new_ability_join_in):
    pack = _contract_pack(tmp_path, manifest_edit=_join_in)
    _only(_decision_problems(pack), "belongs to guard-judge only")


# ==================================================================================================
# 3. Deleting the key each refusal reads still refuses (v1.3 §3.5, §5's U3b row)
# ==================================================================================================


def _del(*path: str) -> Callable[[dict[str, Any]], None]:
    def edit(manifest):
        node = manifest
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return edit


@pytest.mark.parametrize(
    ("deleted", "needle"),
    [
        (("labels",), "needs a labels block"),
        (("labels", "source"), "labels.source"),
        (("labels", "labeler"), "labels.labeler.model"),
        (("labels", "labeler", "model"), "labels.labeler.model"),
        (("labels", "inputs"), "labels.inputs"),
        (("decisionPoint", "incumbent"), "decisionPoint.incumbent is absent"),
    ],
)
def test_deleting_a_manifest_key_a_refusal_reads_still_refuses(tmp_path, deleted, needle):
    _only(validate_pack(_contract_pack(tmp_path, manifest_edit=_del(*deleted))), needle)


def test_deleting_an_eval_items_origin_still_refuses(tmp_path):
    def edit(rows):
        del rows[0]["origin"]

    _only(validate_pack(_contract_pack(tmp_path, rows_edit=edit)), "'ev-000'")


def test_deleting_the_labeler_still_refuses_when_the_incumbent_is_absent_too(tmp_path):
    """Two absent keys never compare equal-by-`None` into a pass: both are refused."""

    def edit(manifest):
        del manifest["labels"]["labeler"]
        del manifest["decisionPoint"]["incumbent"]

    problems = validate_pack(_contract_pack(tmp_path, manifest_edit=edit))
    assert len(problems) == 2, problems
    assert any("labels.labeler.model" in p for p in problems)
    assert any("decisionPoint.incumbent is absent" in p for p in problems)


def test_a_refusal_names_at_most_five_items_and_the_full_count(tmp_path):
    def edit(rows):
        for row in rows[:12]:
            row["origin"] = "human"

    problem = _only(validate_pack(_contract_pack(tmp_path, rows_edit=edit)), "12 'eval' item(s)")
    assert "'ev-004'" in problem and "'ev-005'" not in problem and "(12 in all)" in problem


# ==================================================================================================
# 4. Declared per-kind and side-check metrics (v1.3 §3.5 "Declared outcomes"; plan2 §4.1 U3b)
# ==================================================================================================

from modelbench.results import ItemTiming  # noqa: E402
from modelbench.scoring import classification  # noqa: E402


def _timing() -> ItemTiming:
    return ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)


def _score(item: dict[str, Any], answer: bool | None, pack, *, failed_call: str | None = None):
    return classification.score_answer(
        item, answer, "pass", _timing(), pack=pack, decided_by=None, probability=None,
        failed_call=failed_call, reason=None, input_tokens=None,
    )


@pytest.fixture
def contract_pack(tmp_path):
    return _contract_pack(tmp_path)


@pytest.mark.parametrize(
    ("label", "path", "per_kind", "pooled"),
    [
        (False, "understanding", "falseAdvanceRateByUnderstanding", "falseAdvanceRate"),
        (False, "turns", "falseAdvanceRateByTurns", "falseAdvanceRate"),
        (True, "understanding", "falseSuspendRateByUnderstanding", "falseSuspendRate"),
        (True, "turns", "falseSuspendRateByTurns", "falseSuspendRate"),
    ],
)
def test_an_eval_item_declares_its_per_kind_metric_and_the_pooled_one(
    contract_pack, label, path, per_kind, pooled
):
    """Chosen from the item's own label and path — never its `tier`, which a generated item lacks.
    A `False` label's error is advancing; a `True` label's is suspending."""
    item = _eval_item(1, expected=label, path=path)
    for answer in (True, False):
        result = _score(item, answer, contract_pack)
        event = int(answer != label)
        assert result.scoreable == {per_kind: True, pooled: True}, (label, path, answer)
        assert result.counts == {per_kind: event, pooled: event}, (label, path, answer)
        assert result.detail == {"path": path}


@pytest.mark.parametrize(
    ("tier", "metric", "event_when_advanced"),
    [
        ("clear_suspend", "sideCheckFalseAdvanceRate", 1),
        ("clear_advance", "sideCheckFalseSuspendRate", 0),
        ("boundary", "sideCheckFalseAdvanceRateBoundary", 1),
    ],
)
def test_a_side_check_item_declares_only_its_side_check_metric_by_tier(
    contract_pack, tier, metric, event_when_advanced
):
    """The old pack's tier mapping, prefixed (FR-15): never a verdict or pooled metric."""
    item = next(
        r for r in contract_pack.iter_items() if r["stratum"] == "side-check" and r["tier"] == tier
    )
    advanced = _score(item, True, contract_pack)
    suspended = _score(item, False, contract_pack)
    assert advanced.scoreable == suspended.scoreable == {metric: True}
    assert (advanced.counts, suspended.counts) == (
        {metric: event_when_advanced}, {metric: 1 - event_when_advanced}
    )
    assert advanced.detail == {"path": item["path"], "tier": tier}


def test_the_side_check_table_keeps_the_old_packs_tiers_one_to_one():
    """Bound to `_METRIC_BY_TIER` (a second declaration): same keys, each value that tier's old
    metric prefixed — a moved, dropped or added row reddens."""
    assert classification._SIDE_CHECK_METRIC_BY_TIER == {
        tier: "sideCheck" + metric[0].upper() + metric[1:]
        for tier, metric in classification._METRIC_BY_TIER.items()
    }


def test_a_failed_decision_call_declares_nothing_on_either_stratum(contract_pack):
    """Plan2 §4.1: the `failed_call` branch is U1d's and unchanged — nothing scoreable."""
    side = next(r for r in contract_pack.iter_items() if r["stratum"] == "side-check")
    for item in (_eval_item(1), side):
        result = _score(item, None, contract_pack, failed_call="decision")
        assert (result.scoreable, result.counts) == ({}, {})


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda item: item.pop("stratum"), "stratum None"),
        (lambda item: item.update(stratum="calibration"), "stratum 'calibration'"),
        (lambda item: item.update(expected="false"), "label must be a bool"),
        (lambda item: item.update(path="mixed"), "not an evidence kind"),
    ],
)
def test_an_item_the_contract_refuses_raises_rather_than_being_scored(contract_pack, edit, match):
    """Nothing is defaulted at score time either: `validate_pack` refuses such a pack first."""
    item = _eval_item(1)
    edit(item)
    with pytest.raises(ValueError, match=match):
        _score(item, True, contract_pack)


def test_the_old_pack_keeps_one_tier_metric_per_item_and_its_aggregate_layout():
    """No `decisionPoint` key, no change: the real guard pack's items still declare their tier's
    one metric, and `aggregate` keeps its boundary and path-split figures."""
    pack = load_pack(_REAL_GUARD_PACK)
    results = [_score(item, True, pack) for item in pack.iter_items()]
    assert all(
        r.scoreable == {classification._METRIC_BY_TIER[r.detail["tier"]]: True} for r in results
    )
    names = [m.name for m in classification.aggregate(results, pack=pack).perClass]
    assert names == [
        "falseAdvanceRate", "falseSuspendRate", "advanceRecall", "falseAdvanceRateBoundary",
        "falseAdvanceRateByUnderstanding", "falseAdvanceRateByTurns",
        "falseSuspendRateByUnderstanding", "falseSuspendRateByTurns",
    ]


def test_a_decision_point_aggregate_reads_every_figure_off_the_declarations(contract_pack):
    """Every eval item advances: each `False` label is a false advance, no `True` label is a false
    suspend. The four per-kind cells partition the 200 eval items; the side check's 85 stay out of
    them and out of the pooled rates."""
    rows = list(contract_pack.iter_items())
    results = [_score(item, True, contract_pack) for item in rows]
    agg = classification.aggregate(results, pack=contract_pack)
    by_name = {m.name: (m.successes, m.n) for m in agg.perClass}
    assert list(by_name) == [
        "falseAdvanceRate", "falseSuspendRate", "advanceRecall",
        "falseAdvanceRateByUnderstanding", "falseAdvanceRateByTurns",
        "falseSuspendRateByUnderstanding", "falseSuspendRateByTurns",
        "sideCheckFalseAdvanceRate", "sideCheckFalseSuspendRate",
        "sideCheckFalseAdvanceRateBoundary",
    ]
    assert by_name["falseAdvanceRate"] == (100, 100)
    assert by_name["falseSuspendRate"] == (0, 100)
    assert by_name["advanceRecall"] == (100, 100)
    assert by_name["falseAdvanceRateByUnderstanding"] == (100, 100)
    assert by_name["falseAdvanceRateByTurns"] == (0, 0)
    assert by_name["falseSuspendRateByUnderstanding"] == (0, 100)
    assert by_name["falseSuspendRateByTurns"] == (0, 0)
    assert by_name["sideCheckFalseAdvanceRate"] == (40, 40)
    assert by_name["sideCheckFalseSuspendRate"] == (0, 30)
    assert by_name["sideCheckFalseAdvanceRateBoundary"] == (15, 15)
    assert agg.n == MIN_EVAL_ITEMS + 85


def test_each_per_kind_cell_holds_exactly_its_own_label_and_path(contract_pack):
    """A quarter of the eval items per (label, path): every item suspends, so each `True`-label
    cell counts all of its 50 items and each `False`-label cell none of its 50."""
    items = [
        _eval_item(n, expected=n % 2 == 0, path=("understanding", "turns")[n % 4 // 2])
        for n in range(200)
    ]
    results = [_score(item, False, contract_pack) for item in items]
    by_name = {
        m.name: (m.successes, m.n)
        for m in classification.aggregate(results, pack=contract_pack).perClass
    }
    assert by_name["falseAdvanceRateByUnderstanding"] == (0, 50)
    assert by_name["falseAdvanceRateByTurns"] == (0, 50)
    assert by_name["falseSuspendRateByUnderstanding"] == (50, 50)
    assert by_name["falseSuspendRateByTurns"] == (50, 50)


def test_the_scorer_keys_on_the_decision_point_key_even_when_its_block_is_empty(tmp_path):
    """The scorer reads the same keying as `validate_pack`: the key's presence, never its value,
    so an empty block still scores by stratum (`validate_pack` refuses that pack separately)."""

    def edit(manifest):
        manifest["decisionPoint"] = {}

    pack = _contract_pack(tmp_path, manifest_edit=edit)
    result = _score(_eval_item(1, expected=False), True, pack)
    assert result.scoreable == {"falseAdvanceRateByUnderstanding": True, "falseAdvanceRate": True}


# ==================================================================================================
# 5. Items the scorer cannot score are refused before a run (impl review P12-m1)
# ==================================================================================================


def _first(stratum: str) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    return lambda rows: next(r for r in rows if r["stratum"] == stratum)


@pytest.mark.parametrize(
    ("stratum", "edit", "needle"),
    [
        ("eval", lambda r: r.update(expected="false"), "no bool label"),
        ("eval", lambda r: r.update(expected=None), "no bool label"),
        ("eval", lambda r: r.pop("expected"), "no bool label"),
        ("eval", lambda r: r.update(path="mixed"), "path outside"),
        ("eval", lambda r: r.pop("path"), "path outside"),
        ("side-check", lambda r: r.update(tier="weird"), "tier outside"),
        ("side-check", lambda r: r.pop("tier"), "tier outside"),
        ("side-check", lambda r: r.pop("path"), "have no path"),
    ],
)
def test_an_item_the_scorer_would_raise_on_is_refused_by_validate_pack(
    tmp_path, stratum, edit, needle
):
    """Each case is refused alone, and the scorer does raise on that same item — so the refusal
    is exactly the scorer's own precondition, not a stricter guess."""
    broken: dict[str, Any] = {}

    def rows_edit(rows):
        row = _first(stratum)(rows)
        edit(row)
        broken.update(row)

    pack = _contract_pack(tmp_path, rows_edit=rows_edit)
    problem = _only(validate_pack(pack), needle)
    assert repr(broken["itemId"]) in problem
    with pytest.raises((ValueError, KeyError)):
        _score(broken, True, pack)


def test_the_item_shape_checks_are_the_classification_scorers_only(tmp_path):
    """Keyed on the declared scorer: a pack naming none skips them (the decision-point contract's
    other refusals still apply), so a future scorer's packs are not held to this one's fields."""

    def manifest_edit(manifest):
        manifest.pop("scorer")

    def rows_edit(rows):
        rows[0]["path"] = "mixed"

    pack = _contract_pack(tmp_path, manifest_edit=manifest_edit, rows_edit=rows_edit)
    assert validate_pack(pack) == []


# ==================================================================================================
# 6. A guard pack's verdict metrics are exactly the four per-kind rates (architect ruling, P12-m2)
# ==================================================================================================

from modelbench import report  # noqa: E402

_FOUR = classification.PER_KIND_VERDICT_METRICS


def test_the_four_are_the_per_kind_tables_values_in_its_order():
    assert _FOUR == tuple(classification._PER_KIND_METRIC.values())
    assert len(set(_FOUR)) == 4


def test_the_four_in_any_order_are_accepted(tmp_path):
    def edit(manifest):
        manifest["metrics"]["verdictMetrics"] = list(reversed(_FOUR))

    assert validate_pack(_contract_pack(tmp_path, manifest_edit=edit)) == []


@pytest.mark.parametrize(
    "verdict",
    [
        pytest.param(["falseAdvanceRate", *_FOUR[1:]], id="pooled-for-per-kind"),
        pytest.param(list(_FOUR[:3]), id="one-missing"),
        pytest.param([*_FOUR, "falseAdvanceRate"], id="one-extra"),
        pytest.param([*_FOUR, _FOUR[0]], id="duplicate"),
        pytest.param(["falseAdvanceRate", "falseSuspendRate"], id="the-old-pooled-pair"),
        pytest.param("falseAdvanceRateByTurns", id="not-a-list"),
    ],
)
def test_any_other_guard_verdict_metrics_are_refused(tmp_path, verdict):
    def edit(manifest):
        manifest["metrics"]["verdictMetrics"] = verdict

    _only(validate_pack(_contract_pack(tmp_path, manifest_edit=edit)), "verdictMetrics")


@pytest.mark.parametrize("empty", [False, True])
def test_a_missing_or_empty_verdict_metrics_is_refused_by_this_rule_too(tmp_path, empty):
    """The sampling axis also objects to these; this rule does not lean on it."""

    def edit(manifest):
        if empty:
            manifest["metrics"]["verdictMetrics"] = []
        else:
            del manifest["metrics"]["verdictMetrics"]

    problems = validate_pack(_contract_pack(tmp_path, manifest_edit=edit))
    assert any("metrics.verdictMetrics must be exactly" in p for p in problems), problems


def test_the_verdict_metrics_rule_is_the_guards_only(tmp_path, new_ability_join_in):
    """The new-ability rule is U4's: a join-in pack is not held to the guard's four."""

    def edit(manifest):
        _join_in(manifest)
        manifest["metrics"]["verdictMetrics"] = ["agreementYes"]

    pack = _contract_pack(tmp_path, side_check=False, manifest_edit=edit)
    assert _decision_problems(pack) == []


@pytest.mark.parametrize("name", _FOUR)
def test_each_per_kind_rate_ranks_lower_is_better(name):
    """In `report._LOWER_IS_BETTER` in the same change that made them verdict metrics: the lower
    rate wins, as for the pooled pair."""
    assert name in report._LOWER_IS_BETTER
    assert report._better(name, 0.1, 0.2) is True
