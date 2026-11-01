"""Seed generation per decision point (method note `docs/plans/typed-decision-models-ml.md`
§Q1.1 and §Q1.3 "design stratum"; plan `docs/plans/typed-decision-models4.md` §3.6).

A **seed** describes one item for the generator: the point's seed axes (`axes`), plus a `seedId`
and the seed's **design stratum** (note §Q1.3: a table cell's non-label part — guard's evidence
kind, and none for join-in and message type). Neither `seedId` nor the design stratum goes into a
request; the stratum is visible to the guard generator only as the `evidenceKind` axis it is
drawn from. `axes` keeps every `<axis>Id` for the store and diagnostics; a generation request
carries only the text axes (note §Q1.2 "Blind": the request writer projects the ids out).

`draw_seeds(role, count, rng_seed, design_stratum=...)` is a pure function of its arguments, the
point's `point.json` design and the interpreter's `random` module:

1. **Stratified axes** (`design.stratified`, in order) are allocated **hierarchically**: `count`
   over the first axis's weights, each share over the second's, and so on. Restricting to a design
   stratum (the top-up of plan4 §3.6) keeps only that value of the design stratum axis; the other
   axes keep their whole design. Nothing restricts the intended answer: intent is a sampling
   variable, never a label (note §Q1.1).
2. **Balanced axes** (`design.balanced`: condition, domain, thread purpose) are allocated over all
   `count` seeds by weight, then shuffled, so they are spread independently of the strata.
3. **Conditional axes** (`design.conditional`) are allocated by weight over the seeds matching
   each of their cases (`when`: one or more axis values, all required), then shuffled; a seed
   matching no case carries `None`. Guard's `turnCount` (`turns` seeds) and `missingForm`
   (`understanding` seeds, given intended answer and difficulty), and join-in's `cue` (given the
   intended answer) are conditional.
4. The seeds are shuffled, and each gets its `seedId`.

**Every allocation is unbiased systematic rounding** (note §Q1.1 "Apportionment is unbiased"),
at every hierarchical level and for every axis kind. Splitting `n` over integer weights `w_i`
with sum `W`: each value gets `floor(n·w_i / W)`, its quota rounded down. The remainders
`r_i = (n·w_i) mod W` are laid end to end, in definition order, on `[0, L·W)`, where `L` is the
number of leftover units. One integer start `s` is drawn uniformly from `0 … W−1` (as
`⌊rng.random()·W⌋`, one call per allocation, made even when `L` is 0). The points
`s, s + W, …, s + (L−1)·W` each fall in one remainder's interval, and that value gets one extra
unit. Because `r_i < W`, no value gets more than one, so every count is its quota rounded down or
up. Every edge is an integer, so `r_i` of the `W` starts give the extra unit: its probability is
`r_i / W`, and the expected count equals the quota exactly at every draw size; deterministic
largest remainder would instead always give a one-seed draw to the heaviest value.

**Ids are minted in a non-word form** (`ID_PATTERN`: lowercase letters, an underscore, digits,
such as `dom_07` or `sd_004821937562`). No prose uses such a token, so the generated-text screen
treats one anywhere in an item as a hit (note §Q1.2 "Hits"). An id'd axis value is stored as two
fields: `<axis>Id` (the id) and `<axis>` (its text, which the generator needs).

A `seedId` is a function of the role, the RNG seed, the design-stratum restriction and the seed's
position, so two draws with different RNG seeds do not share ids. Reusing an RNG seed reproduces
the same seeds, ids included; keeping RNG seeds distinct across a campaign's draws is the
caller's (U5.8, U5.9).

**Reproduction holds on the same Python minor version.** Python guarantees the seeder and
`random()` across versions, but `shuffle` draws through `_randbelow`, which it does not. The draw
record therefore carries `pythonVersion` (`major.minor`), and the recorded seeds, not a re-draw,
are the source of truth.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .points import Point, load_point

#: The non-word id form (module docstring).
ID_PATTERN = re.compile(r"[a-z]{2,4}_[0-9]{2,}")

#: note §Q1.3: round 1 generates about 1.1x the target (eval + calibration).
ROUND_ONE_FACTOR_TENTHS = 11

_SEED_ID_PREFIX = "sd_"
_SEED_ID_DIGITS = 12


@dataclass(frozen=True)
class Seed:
    """One seed: `axes` is the generator's seed line minus its request key."""

    seed_id: str
    role: str
    design_stratum: str | None
    axes: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        """The JSON-ready record: `seedId`, `role`, `designStratum`, `axes`."""
        return {
            "seedId": self.seed_id,
            "role": self.role,
            "designStratum": self.design_stratum,
            "axes": dict(self.axes),
        }


@dataclass(frozen=True)
class SeedDraw:
    """One draw: its arguments, recorded, and the seeds it produced."""

    role: str
    rng_seed: int
    count: int
    design_stratum: str | None
    seeds: tuple[Seed, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "rngSeed": self.rng_seed,
            "count": self.count,
            "designStratum": self.design_stratum,
            "pythonVersion": python_version(),
            "seeds": [seed.to_record() for seed in self.seeds],
        }


def python_version() -> str:
    """The running interpreter's `major.minor`, recorded with every draw (module docstring)."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def target_count(role: str) -> int:
    """A point's eval + calibration target (note §Q1.3 table: calibration equals eval)."""
    cells = load_point(role).definition["cells"]
    return 2 * cells["evalPerCell"] * cells["count"]


def round_one_count(role: str) -> int:
    """Round 1's seed count, ⌈1.1 × target⌉ (plan4 §3.6), in integers."""
    return math.ceil(ROUND_ONE_FACTOR_TENTHS * target_count(role) / 10)


def allocate(total: int, weights: Sequence[int], rng: random.Random) -> list[int]:
    """`total` split over `weights` by unbiased systematic rounding (module docstring): each share
    is its quota rounded down or up, and its expected value is the quota exactly."""
    weight_sum = sum(weights)
    shares = [total * weight // weight_sum for weight in weights]
    start = int(rng.random() * weight_sum)

    def points_below(edge: int) -> int:
        """How many of the points `start + k·weight_sum` (k ≥ 0) lie below `edge`, in integers."""
        return max(0, -((start - edge) // weight_sum))

    edge = 0
    for index, weight in enumerate(weights):
        remainder = total * weight % weight_sum
        shares[index] += points_below(edge + remainder) - points_below(edge)
        edge += remainder
    if sum(shares) != total:
        raise RuntimeError("allocation did not place every unit")
    return shares


def _stratified(
    axes: Sequence[Mapping[str, Any]], total: int, rng: random.Random
) -> list[tuple[dict[str, Any], int]]:
    """Hierarchical allocation over the stratified axes: (axis values, count) per stratum."""
    if not axes:
        return [({}, total)]
    head, rest = axes[0], axes[1:]
    values = head["values"]
    shares = allocate(total, [entry["weight"] for entry in values], rng)
    strata: list[tuple[dict[str, Any], int]] = []
    for entry, share in zip(values, shares, strict=True):
        for inner, count in _stratified(rest, share, rng):
            strata.append(({head["axis"]: entry["value"], **inner}, count))
    return strata


def _balanced_values(values: Sequence[Mapping[str, Any]], total: int, rng: random.Random) -> list:
    shares = allocate(total, [entry.get("weight", 1) for entry in values], rng)
    pool = [entry for entry, share in zip(values, shares, strict=True) for _ in range(share)]
    rng.shuffle(pool)
    return pool


def _assign_conditional(
    seeds_axes: Sequence[dict[str, Any]], axis: Mapping[str, Any], rng: random.Random
) -> None:
    """One conditional axis: per case, its values allocated over the seeds matching every axis
    value in `when`; a seed matching no case carries `None`, and one matching two is an error."""
    name = axis["axis"]
    matched: set[int] = set()
    has_ids = any("id" in entry for case in axis["cases"] for entry in case["values"])
    for axes in seeds_axes:
        if has_ids:
            axes[f"{name}Id"] = None
        axes[name] = None
    for case in axis["cases"]:
        when = case["when"]
        meeting = [
            index for index, axes in enumerate(seeds_axes)
            if all(axes[key] == value for key, value in when.items())
        ]
        if matched.intersection(meeting):
            raise ValueError(f"{name}: a seed matches more than one case")
        matched.update(meeting)
        pool = _balanced_values(case["values"], len(meeting), rng)
        for index, entry in zip(meeting, pool, strict=True):
            _set_axis(seeds_axes[index], name, entry["value"] if "id" not in entry else entry)


def _set_axis(axes: dict[str, Any], name: str, entry: Any) -> None:
    if isinstance(entry, Mapping) and "id" in entry:
        axes[f"{name}Id"] = entry["id"]
        axes[name] = entry["text"]
    else:
        axes[name] = entry


def _restricted(point: Point, design_stratum: str | None) -> list[Mapping[str, Any]]:
    design = point.design
    stratified = list(design["stratified"])
    axis_name = design["designStratumAxis"]
    if design_stratum is None:
        return stratified
    if axis_name is None:
        raise ValueError(f"{point.role} has no design stratum to restrict to")
    for index, axis in enumerate(stratified):
        if axis["axis"] == axis_name:
            kept = [entry for entry in axis["values"] if entry["value"] == design_stratum]
            if not kept:
                raise ValueError(f"{design_stratum!r} is not a {axis_name} of {point.role}")
            stratified[index] = {"axis": axis_name, "values": kept}
            return stratified
    raise ValueError(f"{point.role}'s design stratum axis {axis_name!r} is not stratified")


def _seed_id(role: str, rng_seed: int, design_stratum: str | None, position: int) -> str:
    material = f"{role}|{rng_seed}|{design_stratum or ''}|{position}".encode()
    number = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return f"{_SEED_ID_PREFIX}{number % 10**_SEED_ID_DIGITS:0{_SEED_ID_DIGITS}d}"


def draw_seeds(
    role: str, count: int, rng_seed: int, *, design_stratum: str | None
) -> SeedDraw:
    """`count` seeds for `role`, drawn from the point's design with `random.Random(rng_seed)`,
    restricted to `design_stratum` when it is not `None` (module docstring)."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"count must be a positive integer, not {count!r}")
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
        raise ValueError(f"rng_seed must be an integer, not {rng_seed!r}")
    point = load_point(role)
    design = point.design
    rng = random.Random(rng_seed)

    seeds_axes: list[dict[str, Any]] = []
    for stratum, stratum_count in _stratified(_restricted(point, design_stratum), count, rng):
        seeds_axes += [dict(stratum) for _ in range(stratum_count)]

    for axis in design["balanced"]:
        pool = _balanced_values(axis["values"], count, rng)
        for axes, entry in zip(seeds_axes, pool, strict=True):
            _set_axis(axes, axis["axis"], entry)

    for axis in design["conditional"]:
        _assign_conditional(seeds_axes, axis, rng)

    rng.shuffle(seeds_axes)
    stratum_axis = design["designStratumAxis"]
    seeds = tuple(
        Seed(
            seed_id=_seed_id(role, rng_seed, design_stratum, position),
            role=role,
            design_stratum=None if stratum_axis is None else axes[stratum_axis],
            axes=axes,
        )
        for position, axes in enumerate(seeds_axes)
    )
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise RuntimeError("seedId collision within one draw")
    return SeedDraw(
        role=role, rng_seed=rng_seed, count=count, design_stratum=design_stratum, seeds=seeds
    )
