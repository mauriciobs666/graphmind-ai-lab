"""Seed generation (typed-decision-models U5.3; plan `docs/plans/typed-decision-models4.md` §4.1,
§3.6; method note `docs/plans/typed-decision-models-ml.md` v1.11 §Q1.1, §Q1.3).

The expected counts below are derived by hand from the note, not read from `points/`, so an edit
to a point's design that departs from the note reddens here:

- guard: evidence kind 1:1, intended answer 1:1, near-miss a third (§Q1.1); a third of seeds on
  production's condition and the rest over at least 8 others; turn counts 1–10 on `turns` seeds;
  the `missing` form given (intended answer, difficulty) on `understanding` seeds — `empty` 5 :
  `immaterial` 1; `immaterial` 1 : `indirect` 1; `listed` 7 : `unlisted` 1; `listed` 7 :
  `ambiguous` 1;
- message type: five labels 1:1, near-miss a third, opening:follow-up 40:60 (§Q1.5);
- join-in: answer 1:1, near-miss a third; cue given the answer — `true` over cue_01, cue_02,
  cue_04, `false` over all five;
- round 1 is ⌈1.1 × (eval + calibration)⌉ with calibration = eval (§Q1.3 table, plan4 §3.6).

Apportionment is unbiased systematic rounding (`seeds` module docstring): each allocation gives a
value its quota rounded down or up, and the expected count equals the quota at every draw size.
So a count is pinned exactly where its quota is an integer (guard's 880: 440 per kind, 220 per
(kind, answer)) and to {⌊q⌋, ⌈q⌉} elsewhere (220 × 2/3 = 146.67 → 146 or 147 clear).
`allocate`'s expectation is checked exactly over evenly spaced starts, and means over many RNG
seeds at sizes 1–7 check that the draw composes it.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from decisionmodels import seeds
from decisionmodels.points import ROLES, load_point

COMPONENT = Path(__file__).resolve().parents[1]
RECORDED_RNG_SEED = 20261031

ROUND_ONE = {"guard-judge": 880, "message-type-classifier": 660, "join-in-judge": 440}
TARGET = {"guard-judge": 800, "message-type-classifier": 600, "join-in-judge": 400}
MESSAGE_TYPES = ("order", "question", "complaint", "escalation", "other")
PRODUCTION_CONDITION = "the user has provided enough information to research their request"
#: Note v1.11 §Q1.1's `missingForm` table: (intended answer, difficulty) -> {form: weight}.
MISSING_FORMS = {
    (True, "clear"): {"empty": 5, "immaterial": 1},
    (True, "near-miss"): {"immaterial": 1, "indirect": 1},
    (False, "clear"): {"listed": 7, "unlisted": 1},
    (False, "near-miss"): {"listed": 7, "ambiguous": 1},
}
TRUE_CUES = ("cue_01", "cue_02", "cue_04")
ALL_CUES = ("cue_01", "cue_02", "cue_03", "cue_04", "cue_05")


def _draw(role: str, count: int | None = None, rng_seed: int = RECORDED_RNG_SEED, **kw):
    kw.setdefault("design_stratum", None)
    return seeds.draw_seeds(role, count or ROUND_ONE[role], rng_seed, **kw)


def _axes(draw) -> list[dict]:
    return [dict(seed.axes) for seed in draw.seeds]


def _rounded(count: int, quota: Fraction) -> bool:
    """`count` is `quota` rounded down or up."""
    return math.floor(quota) <= count <= math.ceil(quota)


def _digest(draw) -> str:
    """Over the record minus `pythonVersion`, so the pin reddens when the seeds change, not
    merely when the interpreter's minor version does."""
    record = {k: v for k, v in draw.to_record().items() if k != "pythonVersion"}
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _round_trip(draw) -> dict:
    return json.loads(json.dumps(draw.to_record()))


# --- sizes ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_round_one_is_a_tenth_over_eval_plus_calibration(role: str) -> None:
    assert seeds.target_count(role) == TARGET[role]
    assert seeds.round_one_count(role) == ROUND_ONE[role]


# --- reproduction --------------------------------------------------------------------------------


def _redraw(recorded: dict):
    return seeds.draw_seeds(
        recorded["role"], recorded["count"], recorded["rngSeed"],
        design_stratum=recorded["designStratum"],
    )


@pytest.mark.parametrize("role", ROLES)
def test_a_draw_reproduces_exactly_from_its_recorded_rng_seed(role: str) -> None:
    recorded = _round_trip(_draw(role))
    assert _redraw(recorded).to_record() == recorded


@pytest.mark.parametrize("stratum", ["understanding", "turns"])
def test_a_restricted_draw_reproduces_from_its_own_record(stratum: str) -> None:
    """A top-up draw is the restricted case (Pass 3 P3-F2): it must go through the record."""
    recorded = _round_trip(_draw("guard-judge", 60, design_stratum=stratum))
    assert recorded["designStratum"] == stratum
    again = _redraw(recorded).to_record()
    assert again == recorded
    forms = {seed["axes"]["missingForm"] for seed in recorded["seeds"]}
    turns = {seed["axes"]["turnCount"] for seed in recorded["seeds"]}
    assert (None in forms) == (stratum == "turns") and (None in turns) == (stratum != "turns")


def test_a_draw_records_its_python_version() -> None:
    """Reproduction holds on the same minor version only (Pass 3 P3-F4)."""
    expected = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert _draw("join-in-judge", 3).to_record()["pythonVersion"] == expected


def test_reproduction_does_not_depend_on_the_interpreters_hash_seed() -> None:
    """Set or dict-order dependence would make a draw differ between processes."""
    program = (
        "import json, hashlib\n"
        "from decisionmodels import seeds\n"
        "d = seeds.draw_seeds('guard-judge', 120, 7, design_stratum=None).to_record()\n"
        "print(hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest())\n"
    )
    outputs = set()
    for hash_seed in ("1", "2", "random"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(COMPONENT)}
        result = subprocess.run(
            [sys.executable, "-c", program], cwd=COMPONENT, env=env,
            capture_output=True, text=True, check=True,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


#: A regression pin, not a contract vector: recorded from this implementation so that a change to
#: the algorithm, to a point's design, or to how this interpreter's `random` shuffles — any of
#: which changes what a recorded RNG seed reproduces — is a visible, deliberate re-pin rather than
#: a silent drift.
PINNED_DIGESTS = {
    "guard-judge": "341d881590bb6e5c3130279189fac0184a6929d82e80d02400faa3f7abb29c62",
    "message-type-classifier": "299beb51853c00fe108c964dcacfbf6d60ab1b82198af4a349a14a673e1fe67f",
    "join-in-judge": "26109599c5b1ef0184d2c232dc922c6626305e487c3abda9c37cc872d241c036",
}


@pytest.mark.parametrize("role", ROLES)
def test_the_recorded_draw_is_pinned(role: str) -> None:
    assert _digest(_draw(role, 50)) == PINNED_DIGESTS[role]


@pytest.mark.parametrize("role", ROLES)
def test_different_rng_seeds_give_different_draws(role: str) -> None:
    one, two = _draw(role), _draw(role, rng_seed=RECORDED_RNG_SEED + 1)
    assert [s.axes for s in one.seeds] != [s.axes for s in two.seeds]
    assert {s.seed_id for s in one.seeds}.isdisjoint({s.seed_id for s in two.seeds})


def test_a_seed_matching_two_conditional_cases_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass 4 P4-F3: the overlap guard can fail. A scratch guard design gains a `missingForm` case
    whose `when` overlaps the existing (`true`, `clear`) case."""
    point = load_point("guard-judge")
    definition = json.loads(json.dumps(point.definition))
    missing_form = next(a for a in definition["design"]["conditional"]
                        if a["axis"] == "missingForm")
    missing_form["cases"].append(
        {"when": {"evidenceKind": "understanding", "intendedAnswer": True},
         "values": [{"value": "empty", "weight": 1}]})
    scratch = dataclasses.replace(point, definition=definition)
    monkeypatch.setattr(seeds, "load_point", lambda role: scratch)
    with pytest.raises(ValueError, match="more than one case"):
        seeds.draw_seeds("guard-judge", 40, 1, design_stratum=None)


def test_a_seed_id_collision_within_a_draw_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The within-draw guard can fail (Pass 3 P3-F3)."""
    monkeypatch.setattr(seeds, "_seed_id", lambda *args: "sd_000000000001")
    with pytest.raises(RuntimeError, match="collision"):
        seeds.draw_seeds("guard-judge", 2, 1, design_stratum=None)


# --- apportionment -------------------------------------------------------------------------------


@pytest.mark.parametrize("weights", [(1, 1), (2, 1), (9, 2, 2, 2), (3, 1), (7, 1), (2, 3), (1,)])
@pytest.mark.parametrize("total", [0, 1, 2, 3, 5, 7, 11, 880])
def test_every_share_is_its_quota_rounded_down_or_up(weights: tuple, total: int) -> None:
    import random

    for rng_seed in range(40):
        shares = seeds.allocate(total, weights, random.Random(rng_seed))
        assert sum(shares) == total
        for share, weight in zip(shares, weights, strict=True):
            assert _rounded(share, Fraction(total * weight, sum(weights)))


class _GridStart:
    """A stand-in RNG whose one `random()` returns grid point `(k + 0.5) / n`, or `value`."""

    def __init__(self, k: int, n: int, value: float | None = None) -> None:
        self.value = (k + 0.5) / n if value is None else value

    def random(self) -> float:
        return self.value


#: Pass 4 P4-F2: weight vectors for the exact test, the design's shapes among them.
GRID_WEIGHTS = [(1,), (1, 1), (2, 1), (3, 1), (5, 1), (7, 1), (2, 3), (9, 2, 2, 2),
                (1, 1, 1, 1, 1), (1,) * 10]


@pytest.mark.parametrize("weights", GRID_WEIGHTS, ids=lambda w: "w" + "-".join(map(str, w)))
def test_every_share_is_its_quota_exactly_on_average_over_evenly_spaced_starts(
    weights: tuple,
) -> None:
    """Pass 4 P4-F2: the expectation exactly, with no sampling error. `N = 12·W` evenly spaced
    starts stand in for a uniform `rng.random()`, so the mean share over them is the expected
    share, and it must equal the quota as a `Fraction`. A start biased by a few percent, or
    largest remainder, misses it at some total."""
    weight_sum = sum(weights)
    grid = 12 * weight_sum
    for total in range(30):
        sums = [0] * len(weights)
        for k in range(grid):
            shares = seeds.allocate(total, weights, _GridStart(k, grid))
            sums = [a + b for a, b in zip(sums, shares, strict=True)]
        means = [Fraction(n, grid) for n in sums]
        assert means == [Fraction(total * w, weight_sum) for w in weights], (total, means)


@pytest.mark.parametrize("weights", GRID_WEIGHTS, ids=lambda w: "w" + "-".join(map(str, w)))
def test_each_integer_start_owns_exactly_its_width_of_random(weights: tuple) -> None:
    """The start is `⌊rng.random()·W⌋` (module docstring), so the allocation is constant across
    each `[j/W, (j+1)/W)`. Probing both edges catches a start scaled by as little as 0.1%, which
    falls between the grid's points above."""
    weight_sum = sum(weights)
    edge = 1e-9
    for total in range(30):
        for j in range(weight_sum):
            at = [seeds.allocate(total, weights, _GridStart(0, 1, (j + d) / weight_sum))
                  for d in (edge, 0.5, 1 - edge)]
            assert at[0] == at[1] == at[2], (total, j, at)


@functools.cache
def _many_draws(role: str, size: int, stratum: str | None) -> tuple[tuple[dict, ...], ...]:
    """Each draw's seed axes, for `MEAN_DRAWS` draws with RNG seeds 0, 1, …"""
    return tuple(
        tuple(dict(seed.axes)
              for seed in seeds.draw_seeds(role, size, rng_seed, design_stratum=stratum).seeds)
        for rng_seed in range(MEAN_DRAWS)
    )


def _per_draw_counts(role: str, size: int, stratum: str | None, key, value) -> list[int]:
    return [sum(1 for axes in draw if key(axes) == value)
            for draw in _many_draws(role, size, stratum)]


MEAN_DRAWS = 1500
#: The mean count must lie within this many standard errors of the expected count, where the
#: standard error is the case's own per-draw spread over `MEAN_DRAWS` (Pass 4 P4-F2). The RNG
#: seeds are fixed, so the bound is tight without being flaky; a case whose count never varies
#: must hit its expectation exactly. `allocate`'s own expectation is pinned exactly, on a grid.
MEAN_STANDARD_ERRORS = 4


#: (role, design stratum, what to count, expected share of a draw), every share by hand from the
#: note: guard `turns` near-miss 1/3, production condition 9/27, one turn count 1/10; guard
#: `understanding` `empty` ½·⅔·⅚ = 5/18, `immaterial` ½·⅔·⅙ + ½·⅓·½ = 5/36, `unlisted`
#: ½·⅔·⅛ = 1/24, `ambiguous` ½·⅓·⅛ = 1/48;
#: message-type opening 2/5, one label 1/5; join-in cue_03 ½·⅕ = 1/10, cue_01 ½·⅓ + ½·⅕ = 4/15.
EXPECTED_SHARES = [
    ("guard-judge", "turns", lambda a: a["difficulty"], "near-miss", Fraction(1, 3)),
    ("guard-judge", "turns", lambda a: a["condition"], PRODUCTION_CONDITION, Fraction(9, 27)),
    ("guard-judge", "turns", lambda a: a["turnCount"], 1, Fraction(1, 10)),
    ("guard-judge", "understanding", lambda a: a["missingForm"], "empty", Fraction(5, 18)),
    ("guard-judge", "understanding", lambda a: a["missingForm"], "immaterial", Fraction(5, 36)),
    ("guard-judge", "understanding", lambda a: a["missingForm"], "unlisted", Fraction(1, 24)),
    ("guard-judge", "understanding", lambda a: a["missingForm"], "ambiguous", Fraction(1, 48)),
    ("message-type-classifier", None, lambda a: a["threadPosition"], "opening", Fraction(2, 5)),
    ("message-type-classifier", None, lambda a: a["intendedAnswer"], "other", Fraction(1, 5)),
    ("join-in-judge", None, lambda a: a["cueId"], "cue_03", Fraction(1, 10)),
    ("join-in-judge", None, lambda a: a["cueId"], "cue_01", Fraction(4, 15)),
]


@pytest.mark.parametrize("size", range(1, 8))
@pytest.mark.parametrize(
    "case", EXPECTED_SHARES, ids=[f"{r}-{s}-{v}" for r, s, _, v, _ in EXPECTED_SHARES]
)
def test_small_draws_hit_every_expected_count(case, size: int) -> None:
    """Note v1.10 §Q1.1: the expected count equals the quota at every draw size."""
    role, stratum, key, value, share = case
    counts = _per_draw_counts(role, size, stratum, key, value)
    mean = Fraction(sum(counts), MEAN_DRAWS)
    variance = sum((n - mean) ** 2 for n in counts) / (MEAN_DRAWS - 1)
    miss = mean - share * size
    # |miss| ≤ k·SE, squared to stay exact: miss² ≤ k²·variance / draws.
    assert miss**2 * MEAN_DRAWS <= MEAN_STANDARD_ERRORS**2 * variance, (
        float(mean), float(share * size), float(variance))


# --- coverage: guard -----------------------------------------------------------------------------


def test_guard_cells_and_design_strata_match_the_note() -> None:
    axes = _axes(_draw("guard-judge"))
    assert Counter(a["evidenceKind"] for a in axes) == {"understanding": 440, "turns": 440}
    cells = Counter((a["evidenceKind"], a["intendedAnswer"]) for a in axes)
    assert cells == {(k, v): 220 for k in ("understanding", "turns") for v in (True, False)}
    joint = Counter((a["evidenceKind"], a["intendedAnswer"], a["difficulty"]) for a in axes)
    for kind in ("understanding", "turns"):
        for answer in (True, False):
            assert _rounded(joint[(kind, answer, "clear")], Fraction(220 * 2, 3))
            assert joint[(kind, answer, "near-miss")] == 220 - joint[(kind, answer, "clear")]


def test_guard_design_stratum_is_the_evidence_kind() -> None:
    for seed in _draw("guard-judge").seeds:
        assert seed.design_stratum == seed.axes["evidenceKind"]


def test_guard_conditions_are_a_third_production_and_the_rest_spread_over_eight_or_more() -> None:
    axes = _axes(_draw("guard-judge"))
    by_text = Counter(a["condition"] for a in axes)
    assert _rounded(by_text.pop(PRODUCTION_CONDITION), Fraction(880 * 9, 27))  # 293.33
    assert len(by_text) >= 8
    assert all(_rounded(n, Fraction(880 * 2, 27)) for n in by_text.values())  # 65.19
    assert sum(by_text.values()) == 880 - sum(1 for a in axes if a["condition"] ==
                                              PRODUCTION_CONDITION)
    ids = {a["conditionId"] for a in axes if a["condition"] == PRODUCTION_CONDITION}
    assert len(ids) == 1


def test_guard_turn_counts_span_one_to_ten_on_turns_seeds_only() -> None:
    axes = _axes(_draw("guard-judge"))
    turns = Counter(a["turnCount"] for a in axes if a["evidenceKind"] == "turns")
    assert turns == {n: 44 for n in range(1, 11)}
    assert all(a["turnCount"] is None for a in axes if a["evidenceKind"] == "understanding")


def test_guard_missing_forms_follow_the_answer_and_difficulty() -> None:
    """Note v1.11 §Q1.1 "Guard `missing` forms": the form is drawn given (answer, difficulty) on
    `understanding` seeds only, at the table's weights."""
    axes = _axes(_draw("guard-judge"))
    assert all(a["missingForm"] is None for a in axes if a["evidenceKind"] == "turns")
    understanding = [a for a in axes if a["evidenceKind"] == "understanding"]
    for (answer, difficulty), weights in MISSING_FORMS.items():
        group = [a for a in understanding
                 if (a["intendedAnswer"], a["difficulty"]) == (answer, difficulty)]
        forms = Counter(a["missingForm"] for a in group)
        assert set(forms) == set(weights), (answer, difficulty, forms)
        for form, weight in weights.items():
            assert _rounded(forms[form], Fraction(len(group) * weight, sum(weights.values())))


def test_guard_domains_are_balanced() -> None:
    domains = Counter(a["domainId"] for a in _axes(_draw("guard-judge")))
    assert len(domains) == 12
    assert all(_rounded(n, Fraction(880, 12)) for n in domains.values())


def test_a_design_stratum_restriction_keeps_the_rest_of_the_design() -> None:
    """The top-up sampler (plan4 §3.6): restricted to the stratum, never to an intent."""
    axes = _axes(_draw("guard-judge", 100, design_stratum="turns"))
    assert {a["evidenceKind"] for a in axes} == {"turns"}
    assert Counter(a["intendedAnswer"] for a in axes) == {True: 50, False: 50}
    joint = Counter((a["intendedAnswer"], a["difficulty"]) for a in axes)
    for answer in (True, False):
        assert _rounded(joint[(answer, "near-miss")], Fraction(50, 3))
    assert Counter(a["turnCount"] for a in axes) == {n: 10 for n in range(1, 11)}


def test_draw_seeds_has_no_intent_parameter() -> None:
    import inspect

    params = set(inspect.signature(seeds.draw_seeds).parameters)
    assert params == {"role", "count", "rng_seed", "design_stratum"}


# --- coverage: message type and join-in ----------------------------------------------------------


def test_message_type_cells_match_the_note() -> None:
    axes = _axes(_draw("message-type-classifier"))
    assert Counter(a["intendedAnswer"] for a in axes) == {label: 132 for label in MESSAGE_TYPES}
    joint = Counter((a["intendedAnswer"], a["difficulty"]) for a in axes)
    sizes = (("clear", 88), ("near-miss", 44))
    assert joint == {(lab, d): n for lab in MESSAGE_TYPES for d, n in sizes}
    # Opening is 2/5 within each (label, difficulty): 88 → 35.2, 44 → 17.6.
    positions = Counter((a["intendedAnswer"], a["difficulty"], a["threadPosition"]) for a in axes)
    for label in MESSAGE_TYPES:
        for difficulty, n in sizes:
            opening = positions[(label, difficulty, "opening")]
            assert _rounded(opening, Fraction(n * 2, 5))
            assert positions[(label, difficulty, "follow-up")] == n - opening
    assert all(seed.design_stratum is None for seed in _draw("message-type-classifier").seeds)


def test_join_in_cells_match_the_note() -> None:
    axes = _axes(_draw("join-in-judge"))
    assert Counter(a["intendedAnswer"] for a in axes) == {True: 220, False: 220}
    joint = Counter((a["intendedAnswer"], a["difficulty"]) for a in axes)
    for answer in (True, False):
        assert _rounded(joint[(answer, "clear")], Fraction(220 * 2, 3))
    assert all(_rounded(n, Fraction(440, 10)) for n in
               Counter(a["purposeId"] for a in axes).values())
    assert all(seed.design_stratum is None for seed in _draw("join-in-judge").seeds)


def test_join_in_cue_is_drawn_given_the_answer() -> None:
    """Note v1.11 §Q1.1 "Join-in cue given the answer"."""
    axes = _axes(_draw("join-in-judge"))
    true_cues = Counter(a["cueId"] for a in axes if a["intendedAnswer"] is True)
    false_cues = Counter(a["cueId"] for a in axes if a["intendedAnswer"] is False)
    assert set(true_cues) == set(TRUE_CUES)
    assert all(_rounded(n, Fraction(220, 3)) for n in true_cues.values())
    assert false_cues == {cue: 44 for cue in ALL_CUES}
    assert all(a["cue"] for a in axes)


@pytest.mark.parametrize("role", ("message-type-classifier", "join-in-judge"))
def test_a_point_without_a_design_stratum_refuses_a_restriction(role: str) -> None:
    with pytest.raises(ValueError, match="no design stratum"):
        _draw(role, 10, design_stratum="turns")


def test_an_unknown_design_stratum_is_refused() -> None:
    with pytest.raises(ValueError, match="not a evidenceKind"):
        _draw("guard-judge", 10, design_stratum="transcript")


@pytest.mark.parametrize(
    ("count", "rng_seed"), [(0, 1), (-3, 1), (True, 1), (2.0, 1), (5, "7"), (5, True), (5, None)]
)
def test_bad_arguments_are_refused(count, rng_seed) -> None:
    with pytest.raises(ValueError):
        seeds.draw_seeds("guard-judge", count, rng_seed, design_stratum=None)


def test_an_unknown_role_is_refused() -> None:
    with pytest.raises(ValueError, match="no decision point"):
        seeds.draw_seeds("entity-matcher", 5, 1, design_stratum=None)


@pytest.mark.parametrize("count", [1, 2, 7])
def test_small_draws_are_exact_in_size(count: int) -> None:
    for role in ROLES:
        assert len(_draw(role, count).seeds) == count


# --- ids -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_every_id_is_in_the_non_word_form(role: str) -> None:
    draw = _draw(role)
    ids = [seed.seed_id for seed in draw.seeds]
    ids += [v for seed in draw.seeds for k, v in seed.axes.items() if k.endswith("Id")]
    assert ids
    bad = [i for i in ids if not seeds.ID_PATTERN.fullmatch(i)]
    assert bad == []
    assert len({seed.seed_id for seed in draw.seeds}) == len(draw.seeds)


def _defined_entries(role: str) -> list[dict]:
    design = load_point(role).design
    entries = [entry for axis in design["balanced"] for entry in axis["values"]]
    entries += [entry for axis in design["conditional"] for case in axis["cases"]
                for entry in case["values"]]
    return entries


@pytest.mark.parametrize("role", ROLES)
def test_every_defined_id_is_in_the_non_word_form_and_names_one_text(role: str) -> None:
    by_id: dict[str, set[str]] = {}
    for entry in _defined_entries(role):
        if "id" in entry:
            assert seeds.ID_PATTERN.fullmatch(entry["id"]), entry["id"]
            by_id.setdefault(entry["id"], set()).add(entry["text"])
    assert all(len(texts) == 1 for texts in by_id.values())
    texts = [next(iter(t)) for t in by_id.values()]
    assert len(set(texts)) == len(texts)


@pytest.mark.parametrize("word", ["bakery", "domain", "near-miss", "dom07", "dom_", "Dom_07"])
def test_the_id_form_excludes_words(word: str) -> None:
    assert not seeds.ID_PATTERN.fullmatch(word)


def test_a_seed_record_carries_its_id_and_design_stratum() -> None:
    record = _draw("guard-judge", 3).to_record()
    assert record["rngSeed"] == RECORDED_RNG_SEED
    for seed in record["seeds"]:
        assert set(seed) == {"seedId", "role", "designStratum", "axes"}
        assert seed["designStratum"] == seed["axes"]["evidenceKind"]
