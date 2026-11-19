"""`decisionmodels.outcome` (plan4 §3.11 "Outcome assembly", §5 U6f.9a; note §Q2.0; plan2 §3.2).

Expected pairs are written by hand from the plan's rules; none is read off the implementation."""

from __future__ import annotations

import ast
import re
from itertools import product
from pathlib import Path

import pytest

from decisionmodels import ladder as ladder_module
from decisionmodels import outcome as outcome_module
from decisionmodels.fit import Selection
from decisionmodels.fitstats import LadderResult
from decisionmodels.outcome import assemble_outcome, validate_quality_target

NO_WORSE = "D_c<=0"
NONREJECT = "mcnemar-nonreject"


def ladder(
    *,
    family: str = "handoff",
    rung: int | None = 1,
    target: object = NO_WORSE,
    agreement: bool = False,
    latency: bool = False,
) -> LadderResult:
    return LadderResult(
        family=family,
        rung=rung,
        qualityTarget=target,
        policy=None,
        key=None,
        agreementUnmet=agreement,
        latencyUnmet=latency,
        barValues={},
    )


_SPEED_UP = {"D_c<=0", "mcnemar-nonreject"}


def assemble(lad: LadderResult, sel: Selection, kind: str | None = None) -> tuple[str, str]:
    """`assemble_outcome` with the bar kind a caller (U6f.9b, from the role) would pass: unless
    given, `speed-up` for the speed-up targets and `new-ability` for the rest."""
    if kind is None:
        kind = "speed-up" if str(lad.qualityTarget) in _SPEED_UP else "new-ability"
    return assemble_outcome(lad, sel, kind)


def selection(*, family: str = "handoff", no_eligible: bool = False) -> Selection:
    return Selection(candidate=family, textId="t1", noEligibleText=no_eligible)


# --- precedence: ranking over agreement over latency ---


@pytest.mark.parametrize(
    ("agreement", "latency", "expected"),
    [
        (False, False, "within-margin"),
        (True, False, "fit-infeasible: agreement"),
        (False, True, "fit-infeasible: latency"),
        (True, True, "fit-infeasible: agreement"),
    ],
)
def test_eligible_texts_agreement_outranks_latency(agreement, latency, expected):
    got = assemble(ladder(agreement=agreement, latency=latency), selection())
    assert got == (expected, NO_WORSE)


@pytest.mark.parametrize(("agreement", "latency"), list(product([False, True], repeat=2)))
def test_no_eligible_text_is_ranking_whatever_the_flags_or_rung(agreement, latency):
    got = assemble(
        ladder(rung=1, agreement=agreement, latency=latency), selection(no_eligible=True)
    )
    assert got == ("fit-infeasible: ranking", NO_WORSE)


# --- feasible names: within-margin only at the best rung and target ---


@pytest.mark.parametrize(
    ("family", "rung", "target", "expected"),
    [
        ("handoff", 1, "D_c<=0", "within-margin"),
        ("handoff", 2, "mcnemar-nonreject", "below-margin"),
        ("handoff", 3, "D_c<=0", "below-margin"),  # margin dropped, s <= s_2
        ("handoff", 3, "mcnemar-nonreject", "below-margin"),
        ("decision-only", 1, "D_c<=0", "within-margin"),
        ("decision-only", 2, "mcnemar-nonreject", "below-margin"),
        ("handoff", 1, "T=0.94", "within-margin"),
        ("decision-only", 1, "T=0.94", "within-margin"),
        ("handoff", 1, "T=0.93", "below-margin"),  # rung 1 at a relaxed T
        ("handoff", 1, "T=0.90", "below-margin"),
        ("handoff", 2, "T=0.94", "below-margin"),  # the s <= s_350 ceiling without the cushion
        ("handoff", 3, "T=0.92", "below-margin"),
    ],
)
def test_feasible_outcome_name(family, rung, target, expected):
    lad = ladder(family=family, rung=rung, target=target)
    got = assemble(lad, selection(family=family))
    assert got == (expected, target)


# --- rung None (fit note §2) and rung 4 ---


def test_rung_none_with_latency_unmet_is_latency_infeasible():
    got = assemble(ladder(rung=None, target=NONREJECT, latency=True), selection())
    assert got == ("fit-infeasible: latency", NONREJECT)


def test_rung_none_with_agreement_unmet_is_agreement_infeasible():
    got = assemble(ladder(rung=None, target=NONREJECT, agreement=True), selection())
    assert got == ("fit-infeasible: agreement", NONREJECT)


def test_rung_none_with_both_unmet_names_agreement():
    got = assemble(ladder(rung=None, target=NONREJECT, agreement=True, latency=True), selection())
    assert got == ("fit-infeasible: agreement", NONREJECT)


def test_rung_none_with_no_unmet_bar_is_refused():
    with pytest.raises(ValueError, match="rung None"):
        assemble(ladder(rung=None, target=NONREJECT), selection())


def test_rung_none_decision_only_keeps_its_target():
    got = assemble(
        ladder(family="decision-only", rung=None, target=NO_WORSE, latency=True),
        selection(family="decision-only"),
    )
    assert got == ("fit-infeasible: latency", NO_WORSE)


def test_rung_none_still_ranking_when_no_eligible_text():
    got = assemble(ladder(rung=None, target=NONREJECT, latency=True), selection(no_eligible=True))
    assert got == ("fit-infeasible: ranking", NONREJECT)


def test_new_ability_rung_four_agreement():
    got = assemble(
        ladder(rung=4, target="max-min-agreement", agreement=True, latency=True), selection()
    )
    assert got == ("fit-infeasible: agreement", "max-min-agreement")


def test_guard_rung_four_latency():
    got = assemble(ladder(rung=4, latency=True), selection())
    assert got == ("fit-infeasible: latency", NO_WORSE)


# --- refusals ---


def test_selection_for_the_other_family_is_refused():
    with pytest.raises(ValueError, match="decision-only"):
        assemble(ladder(family="handoff"), selection(family="decision-only"))


@pytest.mark.parametrize("bad", ["T=0.95", "D_c<0", "", "max-min", None, 0.94, ["T=0.94"], True])
def test_unknown_quality_target_is_refused_naming_it(bad):
    with pytest.raises(ValueError, match="qualityTarget") as err:
        assemble(ladder(target=bad), selection())
    assert repr(bad) in str(err.value)


def test_unknown_target_refused_even_when_ranking_would_win():
    with pytest.raises(ValueError, match="qualityTarget"):
        assemble(ladder(target="T=0.95"), selection(no_eligible=True))


# --- validate_quality_target: closed sets, per bar kind ---


@pytest.mark.parametrize("t", [NO_WORSE, NONREJECT])
def test_speed_up_targets_accepted_for_speed_up(t):
    validate_quality_target(t, "speed-up")


@pytest.mark.parametrize(
    "t", ["T=0.94", "T=0.93", "T=0.92", "T=0.91", "T=0.90", "max-min-agreement"]
)
def test_new_ability_targets_accepted_for_new_ability(t):
    validate_quality_target(t, "new-ability")


@pytest.mark.parametrize(
    ("t", "kind"),
    [
        (NO_WORSE, "new-ability"),
        (NONREJECT, "new-ability"),
        ("T=0.94", "speed-up"),
        ("max-min-agreement", "speed-up"),
    ],
)
def test_target_from_the_other_bar_kind_is_refused(t, kind):
    with pytest.raises(ValueError, match=re.escape(repr(t))):
        validate_quality_target(t, kind)


def test_unknown_bar_kind_is_refused():
    with pytest.raises(ValueError, match="bar kind"):
        validate_quality_target(NO_WORSE, "other")


def test_guard_ladder_with_a_new_ability_target_is_refused():
    with pytest.raises(ValueError, match="speed-up"):
        assemble(ladder(target="T=0.94"), selection(), "speed-up")


@pytest.mark.parametrize("t", [NO_WORSE, NONREJECT])
def test_new_ability_ladder_with_a_speed_up_target_is_refused(t):
    with pytest.raises(ValueError, match="new-ability"):
        assemble(ladder(rung=2, target=t), selection(), "new-ability")


def test_bar_kind_is_required():
    with pytest.raises(TypeError):
        assemble_outcome(ladder(), selection())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        validate_quality_target(NO_WORSE)  # type: ignore[call-arg]


# --- combinations no ladder emits are refused as ladder bugs ---


@pytest.mark.parametrize(
    ("target", "kind"),
    [(NO_WORSE, "speed-up"), ("max-min-agreement", "new-ability"), ("T=0.92", "new-ability")],
)
def test_rung_four_with_no_unmet_bar_is_refused(target, kind):
    with pytest.raises(ValueError, match="rung 4 with no unmet bar"):
        assemble(ladder(rung=4, target=target), selection(), kind)


def test_new_ability_max_min_agreement_with_agreement_met_is_refused():
    with pytest.raises(ValueError, match="max-min-agreement"):
        assemble(ladder(rung=4, target="max-min-agreement", latency=True), selection())


def test_guard_rung_four_with_latency_met_is_refused():
    with pytest.raises(ValueError, match="latencyUnmet"):
        assemble(ladder(rung=4, agreement=True), selection())


def test_rung_one_at_nonreject_is_refused():
    with pytest.raises(ValueError, match="rung 1"):
        assemble(ladder(rung=1, target=NONREJECT), selection())


def test_ladder_bugs_are_refused_even_when_ranking_would_win():
    with pytest.raises(ValueError, match="rung 4 with no unmet bar"):
        assemble(ladder(rung=4), selection(no_eligible=True))


def test_every_closed_outcome_is_the_plan2_set():
    assert set(outcome_module.FIT_OUTCOMES) == {
        "within-margin",
        "below-margin",
        "fit-infeasible: ranking",
        "fit-infeasible: agreement",
        "fit-infeasible: latency",
    }


# --- the ladders' targets and the bench's closed set (reads source; tests are outside FR-8) ---

_LADDER_SRC = Path(ladder_module.__file__).read_text(encoding="utf-8")
_BENCH = Path(__file__).resolve().parents[2] / "model-bench" / "modelbench" / "decisions.py"


def _ladder_emittable_targets() -> set[str]:
    """Every target a ladder can emit: the `TARGET_*` constants, plus the `T=` f-string's range."""
    tree = ast.parse(_LADDER_SRC)
    targets = {
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id.startswith("TARGET_") for t in node.targets)
        and isinstance(node.value, ast.Constant)
    }
    assert 'f"T={float(t):.2f}"' in _LADDER_SRC  # the one computed target; its range below
    t_rungs = [f"T={t / 100:.2f}" for t in ladder_module_t_values()]
    return targets | set(t_rungs)


def ladder_module_t_values() -> list[int]:
    pattern = r"T_RUNGS\s*=\s*tuple\(Fraction\(t, 100\) for t in \(([\d, ]+)\)\)"
    match = re.search(pattern, _LADDER_SRC)
    assert match, "T_RUNGS definition not found in ladder.py"
    return [int(x) for x in match.group(1).split(",")]


def test_every_target_the_ladders_emit_is_accepted():
    emitted = _ladder_emittable_targets()
    assert emitted == {
        "D_c<=0",
        "mcnemar-nonreject",
        "max-min-agreement",
        "T=0.94",
        "T=0.93",
        "T=0.92",
        "T=0.91",
        "T=0.90",
    }  # hand-written, so a ladder gaining a target reddens here
    for t in emitted:
        kind = "speed-up" if t in _SPEED_UP else "new-ability"
        validate_quality_target(t, kind)


def _bench_discrepancies(src: str) -> list[str]:
    """Where the bench's closed sets, as written in `src`, differ from the copy: both directions."""
    block = re.search(r"QUALITY_TARGETS_BY_BAR_KIND[^=]*=(.*?)\n\)\n", src, re.S)
    assert block, "QUALITY_TARGETS_BY_BAR_KIND not found in the bench"
    outcomes = re.search(r"FIT_OUTCOMES[^=]*=(.*?)\n\)\n", src, re.S)
    assert outcomes, "FIT_OUTCOMES not found in the bench"
    quoted = lambda text: set(re.findall(r'"([^"]+)"', text))  # noqa: E731
    problems = []
    if quoted(outcomes.group(1)) != set(outcome_module.FIT_OUTCOMES):
        problems.append("outcomes")
    kinds = {"speed-up", "new-ability"}
    if quoted(block.group(1)) - kinds != set(outcome_module.QUALITY_TARGETS):
        problems.append("target union")
    for kind, targets in outcome_module.QUALITY_TARGETS_BY_BAR_KIND.items():
        entry = re.search(rf'"{kind}":\s*frozenset\(\s*\{{(.*?)\}}', block.group(1), re.S)
        if not entry or quoted(entry.group(1)) != set(targets):
            problems.append(kind)
    return problems


def _bench_source() -> str:
    if not _BENCH.is_file():
        pytest.skip("model-bench/ is absent from this checkout")
    return _BENCH.read_text(encoding="utf-8")


def test_closed_sets_equal_the_benchs():
    assert _bench_discrepancies(_bench_source()) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ('"max-min-agreement"', '"max-min-agreement", "T=0.89"', ["target union", "new-ability"]),
        ('"within-margin",', '"within-margin", "brand-new",', ["outcomes"]),
        ('"fit-infeasible: latency"', '"fit-infeasible: lat"', ["outcomes"]),
        (
            '{"D_c<=0", "mcnemar-nonreject"}',
            '{"D_c<=0", "mcnemar-nonreject", "T=0.94"}',
            ["speed-up"],
        ),
    ],
)
def test_a_perturbed_bench_is_detected(old, new, expected):
    src = _bench_source()
    assert src.count(old) == 1, old
    assert _bench_discrepancies(src.replace(old, new)) == expected


def test_moving_a_target_between_kinds_in_the_bench_is_detected():
    src = _bench_source()
    moved = src.replace(
        '{"D_c<=0", "mcnemar-nonreject"}', '{"D_c<=0", "mcnemar-nonreject", "T=0.94"}'
    )
    moved = moved.replace('"T=0.94", "T=0.93"', '"T=0.93"')
    assert _bench_discrepancies(moved) == ["speed-up", "new-ability"]
