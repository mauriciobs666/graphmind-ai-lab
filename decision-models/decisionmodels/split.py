"""Cells, then splits (plan4 `docs/plans/typed-decision-models4.md` §3.6 "Cells, then splits
(U5.10)", amending plan2 `typed-decision-models2.md` §3.5; note
`docs/plans/typed-decision-models-ml.md` §Q1.3 and §Q1.4), as pure functions `finalize` calls.

A **cell** is (design stratum, label); a **candidate** is one accepted, labelled item reduced to
its `seed_id`, its cell and its evidence text (`shingles.evidence_text`). One item per seed, so a
split by seed is a split by item, and a seed id repeated among candidates is an error.

`make_splits` runs the steps in the plan's order:

1. **Side-check drop, first, on every generated item**: a candidate whose evidence shares an
   8-token span with a side-check item (`dropped.shingleSideCheck`).
2. **Subsample each cell at random to its target** (`sizes` summed): `subsample_cells` draws with
   `rng.sample` over the cell's seed-sorted candidates, never the first N. A cell under its target
   keeps all; the surplus items of a full cell stay unused and are counted per cell.
3. **Assign splits within each cell by seed** (`assign_splits`): a seed-shuffled cell is cut into
   eval, then calibration, then train, each up to its size. A short cell therefore fills eval
   first (note §Q1.3, "On a stakeholder 'no'").
4. **The eval-against-calibration/train shingle drop, last**: an eval item sharing an 8-token span
   with any calibration or train item is dropped and counted (`dropped.shingleSplit`). Nothing
   is refilled, so a cell's printed n shows the loss.

Cells and candidates are visited in a fixed order (cells by `str`, candidates by seed id) so a
recorded `rng_seed` reproduces the split from the same candidates on the same Python minor
version, as `random.Random` does.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from . import shingles

SPLITS = ("eval", "calibration", "train")


@dataclass(frozen=True)
class Candidate:
    seed_id: str
    cell: Hashable
    text: str


@dataclass(frozen=True)
class SplitResult:
    """`eval`, `calibration` and `train` hold seed ids. `dropped` has `shingleSideCheck` and
    `shingleSplit` (the seed ids dropped by each rule, in drop order). `surplus` counts, per cell,
    the items the subsample left unused. `sizes` is the printed n: per cell, each split's final
    count."""

    rng_seed: int
    eval: tuple[str, ...]
    calibration: tuple[str, ...]
    train: tuple[str, ...]
    dropped: Mapping[str, tuple[str, ...]]
    surplus: Mapping[Hashable, int]
    sizes: Mapping[Hashable, Mapping[str, int]] = field(default_factory=dict)


def _by_cell(candidates: Iterable[Candidate]) -> dict[Hashable, list[Candidate]]:
    seen: set[str] = set()
    cells: dict[Hashable, list[Candidate]] = {}
    for candidate in candidates:
        if candidate.seed_id in seen:
            raise ValueError(f"seed {candidate.seed_id} has more than one item")
        seen.add(candidate.seed_id)
        cells.setdefault(candidate.cell, []).append(candidate)
    return dict(sorted(cells.items(), key=lambda entry: str(entry[0])))


def subsample_cells(
    cells: Mapping[Hashable, Sequence[Candidate]], target: int, rng: random.Random
) -> tuple[dict[Hashable, list[Candidate]], dict[Hashable, int]]:
    """Each cell cut at random to `target`: (kept, surplus per cell). A cell at or under the
    target is kept whole with no surplus."""
    kept: dict[Hashable, list[Candidate]] = {}
    surplus: dict[Hashable, int] = {}
    for cell in sorted(cells, key=str):
        members = sorted(cells[cell], key=lambda c: c.seed_id)
        kept[cell] = rng.sample(members, target) if len(members) > target else members
        surplus[cell] = len(members) - len(kept[cell])
    return kept, surplus


def assign_splits(
    cells: Mapping[Hashable, Sequence[Candidate]], sizes: Mapping[str, int], rng: random.Random
) -> dict[str, list[Candidate]]:
    """Each cell shuffled on `rng` and cut into `eval`, `calibration`, `train` of at most their
    `sizes`, in that order (so a short cell fills eval first). Within a cell the split is by seed;
    a cell never contributes more than its size to a split."""
    out: dict[str, list[Candidate]] = {name: [] for name in SPLITS}
    for cell in sorted(cells, key=str):
        members = sorted(cells[cell], key=lambda c: c.seed_id)
        rng.shuffle(members)
        start = 0
        for name in SPLITS:
            out[name] += members[start:start + sizes[name]]
            start += sizes[name]
    return out


def make_splits(
    candidates: Iterable[Candidate], *, sizes: Mapping[str, int], rng_seed: int,
    side_check: frozenset[str] | set[str] = frozenset(),
) -> SplitResult:
    """Module docstring's four steps over `candidates`, with `sizes` the per-cell count of each
    split (`eval`, `calibration`, `train`) and `side_check` the side-check shingle hashes."""
    if set(sizes) != set(SPLITS) or any(size < 0 for size in sizes.values()):
        raise ValueError(f"sizes must give non-negative counts for exactly {SPLITS}")
    rng = random.Random(rng_seed)
    everything = _by_cell(candidates)

    side_drops = sorted(c.seed_id for members in everything.values() for c in members
                        if shingles.hits_side_check(c.text, side_check))
    gone = set(side_drops)
    surviving = {cell: [c for c in members if c.seed_id not in gone]
                 for cell, members in everything.items()}

    kept, surplus = subsample_cells(surviving, sum(sizes.values()), rng)
    assigned = assign_splits(kept, sizes, rng)

    others: set[str] = set()
    for name in ("calibration", "train"):
        for candidate in assigned[name]:
            others |= shingles.shingle_hashes(candidate.text)
    split_drops = [c.seed_id for c in assigned["eval"]
                   if not shingles.shingle_hashes(c.text).isdisjoint(others)]
    dropped_eval = set(split_drops)
    eval_kept = [c for c in assigned["eval"] if c.seed_id not in dropped_eval]

    final = {"eval": eval_kept, "calibration": assigned["calibration"],
             "train": assigned["train"]}
    count = {name: {cell: 0 for cell in everything} for name in SPLITS}
    for name, members in final.items():
        for candidate in members:
            count[name][candidate.cell] += 1
    return SplitResult(
        rng_seed=rng_seed,
        eval=tuple(c.seed_id for c in eval_kept),
        calibration=tuple(c.seed_id for c in assigned["calibration"]),
        train=tuple(c.seed_id for c in assigned["train"]),
        dropped={"shingleSideCheck": tuple(side_drops), "shingleSplit": tuple(split_drops)},
        surplus=surplus,
        sizes={cell: {name: count[name][cell] for name in SPLITS} for cell in everything})
