"""`decisionmodels/split.py` (typed-decision-models U5.10a; plan4 §3.6 "Cells, then splits",
plan2 §3.5, note §Q1.3/§Q1.4): random subsampling with surplus counted, splits by seed within a
cell, the side-check drop first and the eval-against-calibration/train shingle drop last, never
refilled. Expected values are written here, never read from the constants they check."""

from __future__ import annotations

import random

import pytest

from decisionmodels import shingles, split
from decisionmodels.split import Candidate

SIZES = {"eval": 3, "calibration": 2, "train": 1}


def words(tag: str, count: int = 9) -> str:
    """Nine distinct tokens per tag, so two tags share no 8-token span."""
    return " ".join(f"{tag}{i}" for i in range(count))


def cand(seed: int, cell: object = ("a", "x"), text: str | None = None) -> Candidate:
    return Candidate(f"s{seed:04d}", cell, words(f"w{seed}") if text is None else text)


def pool(count: int, cell: object = ("a", "x"), start: int = 0) -> list[Candidate]:
    return [cand(start + i, cell) for i in range(count)]


def result_ids(result: split.SplitResult) -> list[str]:
    return [*result.eval, *result.calibration, *result.train]


# --- subsample_cells --------------------------------------------------------------------------


def test_a_full_cell_is_subsampled_at_random_not_first_n() -> None:
    members = pool(300)
    kept, surplus = split.subsample_cells({("a", "x"): members}, 100, random.Random(1))
    chosen = {c.seed_id for c in kept[("a", "x")]}
    assert len(chosen) == 100 and surplus == {("a", "x"): 200}
    first_hundred = {c.seed_id for c in members[:100]}
    assert chosen != first_hundred
    assert len(chosen - first_hundred) > 40          # a uniform draw leaves ~67 of 100 outside


def test_the_subsample_ignores_the_input_order_and_follows_the_rng() -> None:
    members = pool(60)
    once, _ = split.subsample_cells({("a", "x"): members}, 20, random.Random(5))
    again, _ = split.subsample_cells({("a", "x"): members[::-1]}, 20, random.Random(5))
    other, _ = split.subsample_cells({("a", "x"): members}, 20, random.Random(6))
    ids = lambda kept: sorted(c.seed_id for c in kept[("a", "x")])  # noqa: E731
    assert ids(once) == ids(again) != ids(other)


def test_a_short_cell_keeps_every_item_and_counts_no_surplus() -> None:
    kept, surplus = split.subsample_cells({("a", "x"): pool(7), ("a", "y"): pool(10, start=50)},
                                          10, random.Random(0))
    assert len(kept[("a", "x")]) == 7 and surplus[("a", "x")] == 0
    assert len(kept[("a", "y")]) == 10 and surplus[("a", "y")] == 0


# --- assign_splits ----------------------------------------------------------------------------


def test_splits_are_cut_within_each_cell_by_the_sizes_in_order() -> None:
    cells = {("a", "x"): pool(8), ("b", "y"): pool(8, ("b", "y"), start=100)}
    out = split.assign_splits(cells, SIZES, random.Random(3))
    for cell in cells:
        counts = {name: sum(c.cell == cell for c in out[name]) for name in split.SPLITS}
        assert counts == {"eval": 3, "calibration": 2, "train": 1}
    # 8 members, 6 placed: the rest of the cell is simply not in a split
    assert len({c.seed_id for name in split.SPLITS for c in out[name]}) == 12


def test_a_short_cell_fills_eval_first_then_calibration_then_train() -> None:
    for count, expected in [(2, (2, 0, 0)), (3, (3, 0, 0)), (4, (3, 1, 0)), (5, (3, 2, 0)),
                            (6, (3, 2, 1)), (9, (3, 2, 1))]:
        out = split.assign_splits({("a", "x"): pool(count)}, SIZES, random.Random(0))
        assert tuple(len(out[name]) for name in split.SPLITS) == expected, count


def test_the_assignment_is_random_within_a_cell() -> None:
    members = pool(30)
    firsts = {tuple(c.seed_id for c in split.assign_splits(
        {("a", "x"): members}, {"eval": 5, "calibration": 5, "train": 0}, random.Random(s))["eval"])
        for s in range(5)}
    assert len(firsts) == 5
    assert all(sorted(f) != [c.seed_id for c in members[:5]] for f in firsts)


# --- make_splits ------------------------------------------------------------------------------


def test_make_splits_cuts_each_cell_to_its_target_and_counts_the_surplus() -> None:
    cells = {("a", "x"): 20, ("a", "y"): 4, ("b", "x"): 6}
    candidates, start = [], 0
    for cell, count in cells.items():
        candidates += pool(count, cell, start)
        start += 100
    result = split.make_splits(candidates, sizes=SIZES, rng_seed=11)
    assert result.surplus == {("a", "x"): 14, ("a", "y"): 0, ("b", "x"): 0}
    assert result.sizes == {("a", "x"): {"eval": 3, "calibration": 2, "train": 1},
                            ("a", "y"): {"eval": 3, "calibration": 1, "train": 0},
                            ("b", "x"): {"eval": 3, "calibration": 2, "train": 1}}
    assert len(result.eval) == 9 and len(result.calibration) == 5 and len(result.train) == 2
    assert result.rng_seed == 11
    ids = result_ids(result)
    assert len(ids) == len(set(ids)) == 16
    assert dict(result.dropped) == {"shingleSideCheck": (), "shingleSplit": ()}


def test_the_same_rng_seed_reproduces_the_split_and_another_changes_it() -> None:
    candidates = pool(40)
    once = split.make_splits(candidates, sizes=SIZES, rng_seed=7)
    assert once == split.make_splits(list(reversed(candidates)), sizes=SIZES, rng_seed=7)
    other = split.make_splits(candidates, sizes=SIZES, rng_seed=8)
    assert (once.eval, once.calibration, once.train) != (other.eval, other.calibration,
                                                         other.train)


def test_the_three_sizes_reach_their_own_splits() -> None:
    result = split.make_splits(pool(40), sizes={"eval": 7, "calibration": 3, "train": 11},
                               rng_seed=2)
    assert (len(result.eval), len(result.calibration), len(result.train)) == (7, 3, 11)
    assert result.surplus == {("a", "x"): 19}


def test_an_eval_item_sharing_a_span_with_a_calibration_item_is_dropped_and_not_refilled() -> None:
    shared = "alpha bravo charlie delta echo foxtrot golf hotel"
    candidates = pool(12)
    candidates = [Candidate(c.seed_id, c.cell, c.text) for c in candidates]
    sizes = {"eval": 4, "calibration": 4, "train": 0}
    plain = split.make_splits(candidates, sizes=sizes, rng_seed=4)
    # plant the span in one eval and one calibration item, and in a surplus one
    planted = {plain.eval[0]: shared, plain.calibration[0]: "x " + shared + " y",
               next(c.seed_id for c in candidates
                    if c.seed_id not in result_ids(plain)): shared + " z"}
    changed = [Candidate(c.seed_id, c.cell, planted.get(c.seed_id, c.text)) for c in candidates]
    result = split.make_splits(changed, sizes=sizes, rng_seed=4)
    assert result.eval == plain.eval[1:] and result.calibration == plain.calibration
    assert result.dropped["shingleSplit"] == (plain.eval[0],)
    assert result.surplus == plain.surplus == {("a", "x"): 4}   # the surplus did not refill eval
    assert result.sizes[("a", "x")] == {"eval": 3, "calibration": 4, "train": 0}


@pytest.mark.parametrize("other_split", ["calibration", "train"])
def test_the_eval_drop_pools_calibration_and_train_across_cells(other_split: str) -> None:
    """The span sits in an eval item of cell A and a calibration (or train) item of cell B."""
    shared = "alpha bravo charlie delta echo foxtrot golf hotel"
    a, b = Candidate("sA", "A", shared), Candidate("sB", "B", "x " + shared)
    sizes = {"eval": 1, "calibration": 0, "train": 0} | {other_split: 1}
    seen = 0
    for seed in range(40):
        result = split.make_splits([a, b, Candidate("sA2", "A", words("a2")),
                                    Candidate("sB2", "B", words("b2"))], sizes=sizes,
                                   rng_seed=seed)
        # cells hold two items each; the planted pair is in different cells
        planted_eval = [s for s in result.eval if s in ("sA", "sB")]
        planted_other = [s for s in getattr(result, other_split) if s in ("sA", "sB")]
        assert not (planted_eval and planted_other), "a span shared across cells was kept"
        if len(result.dropped["shingleSplit"]) == 1:
            seen += 1
    assert seen > 0


def test_make_splits_is_reproducible_under_candidate_order_over_several_cells() -> None:
    span = "alpha bravo charlie delta echo foxtrot golf hotel"
    side = frozenset(shingles.shingle_hashes(span))
    candidates = []
    for n, cell in enumerate([("a", "x"), (None, "y"), ("b", "x")]):
        candidates += [Candidate(f"s{n}{i:03d}", cell, words(f"c{n}x{i}")) for i in range(12)]
    candidates += [Candidate("s9001", ("a", "x"), "p " + span),
                   Candidate("s9000", ("b", "x"), "q " + span),
                   Candidate("s9002", (None, "y"), span + " india"),
                   Candidate("s9003", ("a", "x"), words("dup") + " " + span)]
    sizes = {"eval": 3, "calibration": 2, "train": 1}
    baseline = split.make_splits(candidates, sizes=sizes, rng_seed=9, side_check=side)
    assert baseline.dropped["shingleSideCheck"] == ("s9000", "s9001", "s9002", "s9003")
    for order_seed in range(6):
        shuffled = candidates[:]
        random.Random(order_seed).shuffle(shuffled)
        assert split.make_splits(shuffled, sizes=sizes, rng_seed=9, side_check=side) == baseline


def test_eval_items_are_not_checked_against_one_another() -> None:
    shared = "alpha bravo charlie delta echo foxtrot golf hotel"
    candidates = [Candidate("s1", "c", shared), Candidate("s2", "c", shared + " india")]
    result = split.make_splits(candidates, sizes={"eval": 2, "calibration": 0, "train": 0},
                               rng_seed=1)
    assert sorted(result.eval) == ["s1", "s2"] and result.dropped["shingleSplit"] == ()


def test_an_eval_shingle_shared_with_train_is_dropped_too() -> None:
    shared = "alpha bravo charlie delta echo foxtrot golf hotel"
    candidates = [Candidate("s1", "c", shared), Candidate("s2", "c", "x " + shared)]
    for seed in range(20):
        result = split.make_splits(candidates, sizes={"eval": 1, "calibration": 0, "train": 1},
                                   rng_seed=seed)
        assert len(result.eval) == 0 and len(result.train) == 1
        assert len(result.dropped["shingleSplit"]) == 1
        assert result.sizes["c"] == {"eval": 0, "calibration": 0, "train": 1}


def test_the_eval_drop_looks_only_at_the_final_calibration_and_train_not_at_the_surplus() -> None:
    """A is checked against what the cell holds after the subsample: the surplus is not in a
    split, so a span shared with it drops nothing. (A drop run before the subsample, over the
    whole pool, would drop A on every seed where C is left over.)"""
    shared = "alpha bravo charlie delta echo foxtrot golf hotel"
    candidates = [Candidate("sA", "c", shared), Candidate("sB", "c", words("b")),
                  Candidate("sC", "c", "x " + shared)]
    sizes = {"eval": 1, "calibration": 1, "train": 0}
    leftovers = 0
    for seed in range(60):
        result = split.make_splits(candidates, sizes=sizes, rng_seed=seed)
        left = set(c.seed_id for c in candidates) - set(result_ids(result)) - set(
            result.dropped["shingleSplit"])
        if left in ({"sA"}, {"sC"}):
            leftovers += 1
            assert result.dropped["shingleSplit"] == ()
            assert len(result.eval) == 1 and len(result.calibration) == 1
    assert leftovers > 5


def test_the_side_check_drop_runs_first_so_a_hit_never_takes_a_place_in_the_cell() -> None:
    span = "alpha bravo charlie delta echo foxtrot golf hotel"
    side = frozenset(shingles.shingle_hashes(span))
    hit = Candidate("s0000", "c", "intro " + span)
    candidates = [hit, *[Candidate(f"s{i}", "c", words(f"w{i}")) for i in range(1, 7)]]
    sizes = {"eval": 3, "calibration": 3, "train": 0}
    for seed in range(15):
        result = split.make_splits(candidates, sizes=sizes, rng_seed=seed, side_check=side)
        assert result.dropped["shingleSideCheck"] == ("s0000",)
        assert "s0000" not in result_ids(result)
        # six survivors fill the six places; a drop after the subsample would leave 5 on some seed
        assert (len(result.eval), len(result.calibration)) == (3, 3)
        assert result.surplus == {"c": 0}


def test_without_a_side_check_set_nothing_is_side_check_dropped() -> None:
    result = split.make_splits(pool(5), sizes=SIZES, rng_seed=0)
    assert result.dropped["shingleSideCheck"] == ()


def test_the_dropped_mapping_carries_exactly_the_two_plan_keys() -> None:
    result = split.make_splits(pool(5), sizes=SIZES, rng_seed=0)
    assert set(result.dropped) == {"shingleSideCheck", "shingleSplit"}


def test_a_repeated_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="more than one item"):
        split.make_splits([cand(1), cand(1)], sizes=SIZES, rng_seed=0)


@pytest.mark.parametrize("sizes", [{"eval": 1, "calibration": 1}, {"eval": 1, "calibration": 1,
                                   "train": 0, "extra": 1}, {"eval": -1, "calibration": 1,
                                                             "train": 0}])
def test_sizes_must_be_the_three_splits_and_non_negative(sizes: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="sizes"):
        split.make_splits(pool(3), sizes=sizes, rng_seed=0)


def test_no_empty_input_is_a_valid_split_of_nothing() -> None:
    result = split.make_splits([], sizes=SIZES, rng_seed=0)
    assert result_ids(result) == [] and result.surplus == {} and result.sizes == {}


