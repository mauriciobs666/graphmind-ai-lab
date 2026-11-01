"""Decision-point bar statistics (typed-decision-models plan v1.3 §3.6; plan2 §3.6; plan3 §3.6).

**This module implements `docs/plans/typed-decision-models-ml.md` (the note) §Q3–§Q6 and no other
source.** Every formula and constant below is that note's, cited by section; the plans fix only
the API. `stats.py` keeps its own single source (`small-model-benchmarking-ml.md`): this module
may call its primitives (`wilson_interval`, `mover_d_interval`, `percentile`,
`mean_bootstrap_interval`) as building blocks, never defines a second percentile estimator, and
**never calls `stats.verdict` or `stats.holm_steps`** (note §Q3.1–§Q3.2: Holm divides α and Rule
7's floor demotion turns a worsening into `holds`, so either would make `holds` easier).

Every argument that shapes a decision is required, with no default (v1.3 §3.6).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from math import comb, exp, lgamma, log, log1p
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from modelbench import stats
from modelbench.decisions import ANSWER_SPACE_BY_ROLE, BAR_KIND_BY_ROLE, MIN_EVAL_ITEMS
from modelbench.results import ItemResult
from modelbench.scoring.agreement import per_class_agreement_metrics
from modelbench.scoring.classification import PER_KIND_VERDICT_METRICS

# --- pinned constants (plan2 §3.6), each the note's -----------------------------------------------

#: Note §Q3.1: FR-1's α — one-sided, per cell, unadjusted.
FR1_ALPHA: float = 0.05
#: Note §Q3.3: the power at which the detectable difference is stated (MDD₈₀).
MDD_POWER: float = 0.80
#: Note §Q3.3: the detectable-difference search grid, in percentage points.
MDD_GRID_PP: float = 0.1
#: Note §Q4.1, §Q4.2: bootstrap resamples for the FR-2 ratio and the FR-10 mean.
BOOTSTRAP_B: int = 10_000
#: Note §Q3.4: the z of `U`'s one-sided 95% MOVER-D bound, Φ⁻¹(0.95).
_Z_ONE_SIDED_95: float = 1.6448536269514722
#: Note §Q1.3: a guard cell's floor ("100 (floor 50)"), printed beside each cell, never refused.
GUARD_CELL_METHOD_MIN: int = 50
#: Note §Q1.3: an FR-10 class's floor — every n ≥ 52 keeps a class truly at 0.85 passing with
#: probability ≤ 0.20 (§Q5.3); 60 is the design size. Printed beside each class, never refused.
FR10_CLASS_METHOD_MIN: int = 52


def _require_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int count, not {value!r}")
    return value


# --- FR-1 parity (note §Q3) -----------------------------------------------------------------------


def p_worse(b: int, c: int) -> float:
    """Note §Q3.1: the one-sided exact McNemar p-value for "the candidate's error rate is higher",
    `Σ_{i=b}^{m} C(m,i) / 2^m` with `m = b + c`, and `1` when `m = 0`. `b` = items the candidate
    gets wrong and the incumbent right, `c` the reverse. Exact integer arithmetic; the one
    division is Python's correctly rounded `int / int`."""
    b = _require_count("b", b)
    c = _require_count("c", c)
    m = b + c
    if m == 0:
        return 1.0
    return sum(comb(m, i) for i in range(b, m + 1)) / 2**m


def _require_probability(name: str, value: object, *, open_low: bool, open_high: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number, not {value!r}")
    low_ok = value > 0 if open_low else value >= 0
    high_ok = value < 1 if open_high else value <= 1
    if not (low_ok and high_ok):
        raise ValueError(f"{name} {value!r} is outside its range")
    return float(value)


def _critical_counts(n: int, alpha: float) -> list[int | None]:
    """Note §Q3.3: `crit(m)`, for `m = 0…n`, is the smallest `b` with `p_worse(b, m − b) ≤ α`, or
    `None` when no `b` rejects (every `m < 5` at α = 0.05). The tail is accumulated from `b = m`
    downward, by the same correctly rounded `int / 2^m` comparison `p_worse` makes."""
    table: list[int | None] = []
    for m in range(n + 1):
        tail, crit = 0, None
        for b in range(m, -1, -1):
            tail += comb(m, b)
            if tail / 2**m > alpha:
                break
            crit = b
        table.append(crit)
    return table


def _binomial_pmf(k: int, n: int, p: float, log_fact: list[float]) -> float:
    """`Bin(k; n, p)`, its terms in log space (`math.lgamma`, note §Q3.3)."""
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    return exp(
        log_fact[n] - log_fact[k] - log_fact[n - k] + k * log(p) + (n - k) * log1p(-p)
    )


def _power(n: int, delta: float, psi_hat: float, crit: list[int | None], log_fact) -> float:
    """Note §Q3.3: `power(δ) = Σ_{m=0}^{n} Bin(m; n, ψ) · P(Bin(m, q) ≥ crit(m))`, with
    `ψ = max(ψ̂, δ)` and `q = (ψ + δ) / (2ψ)`."""
    psi = max(psi_hat, delta)
    q = (psi + delta) / (2 * psi)
    total = 0.0
    for m in range(n + 1):
        c = crit[m]
        if c is None:
            continue
        reject = sum(_binomial_pmf(i, m, q, log_fact) for i in range(c, m + 1))
        total += _binomial_pmf(m, n, psi, log_fact) * reject
    return total


def mdd(n: int, psi_hat: float, *, alpha: float, power_target: float) -> float | None:
    """Note §Q3.3: the smallest true worsening `δ` (a proportion, `0.131` = 13.1 pp) the
    one-sided exact McNemar test detects with probability `power_target`, at the cell's
    `n = floor(n_c / designEffect)` and observed discordance `ψ̂ = m / n_c`.

    The search steps δ up the `MDD_GRID_PP` grid and returns the **first** grid point whose power
    reaches the target — a ceiling, the direction that keeps "could only detect ≥ N" true. `None`
    below `n = 5`, where no worsening can reach significance (note §Q3.3 "Unattainable"); the
    search is total from there, since at δ = 1 every one of the `n ≥ 5` items is candidate-worse.

    `power_target` is `MDD_POWER` for the plain limit and `MDD_POWER ** (1 / (K + 1))` for the
    reuse-adjusted one (note §Q3.5); `alpha` is `FR1_ALPHA`. Neither has a default (v1.3 §3.6).
    """
    n = _require_count("n", n)
    psi_hat = _require_probability("psi_hat", psi_hat, open_low=False, open_high=False)
    alpha = _require_probability("alpha", alpha, open_low=True, open_high=True)
    power_target = _require_probability(
        "power_target", power_target, open_low=True, open_high=True
    )
    crit = _critical_counts(n, alpha)
    if all(c is None for c in crit):
        return None
    log_fact = [lgamma(k + 1) for k in range(n + 1)]
    steps = round(100 / MDD_GRID_PP)
    for step in range(1, steps + 1):
        delta = step / steps
        if _power(n, delta, psi_hat, crit, log_fact) >= power_target:
            return delta
    raise AssertionError(f"mdd: no grid point reached power {power_target} at n = {n}")


# --- one FR-1 cell (note §Q3.1, §Q3.3–§Q3.5) ------------------------------------------------------

CellState = Literal["holds", "worse"]


@dataclass(frozen=True)
class ParityLine:
    """One FR-1 cell's figures, the fields of note §Q3.4's printed line, and its state.

    `e1`/`e2` are the candidate's and incumbent's errors over the `n` paired items; `b`/`c` the
    discordant counts; `U` the upper end of the one-sided 95% MOVER-D interval on the candidate's
    excess error rate (descriptive — `p_worse` decides); `psi_hat = (b + c) / n`; `n_effective =
    floor(n / designEffect)`, the `n` the limits are stated at; `mdd` the MDD₈₀ and `mdd_reuse`
    the reuse-adjusted limit (`None` at K = 0, note §Q3.5). `mdd`/`mdd_reuse` are `None` below
    five effective items, and `U`/`psi_hat` are `None` on an empty cell. `state` is `worse` iff
    `p_worse ≤ α` (note §Q3.1)."""

    e1: int
    e2: int
    n: int
    n_effective: int
    b: int
    c: int
    p_worse: float
    U: float | None
    psi_hat: float | None
    mdd: float | None
    mdd_reuse: float | None
    state: CellState


def _require_pair(pair: object) -> tuple[bool, bool]:
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(v, bool) for v in pair)
    ):
        raise ValueError(f"a pair is (candidate errs, incumbent errs), two bools, not {pair!r}")
    return pair


def parity_cell(
    pairs: Sequence[tuple[bool, bool]], *, design_effect: float, k: int, alpha: float
) -> ParityLine:
    """Note §Q3.1/§Q3.3–§Q3.5 over one cell's paired items, each `(candidate errs, incumbent
    errs)` — what `pair_items` yields per verdict metric. `k` is the arm's reuse count K (plan2
    §3.6: the group's K, the same for every replicate). The decision is the exact test's alone;
    `design_effect` (`>= 1.0`, refused otherwise, a NaN included) reaches only the limits."""
    if not (isinstance(design_effect, (int, float)) and design_effect >= 1.0):
        raise ValueError(f"design_effect must be >= 1.0, not {design_effect!r}")
    k = _require_count("k", k)
    alpha = _require_probability("alpha", alpha, open_low=True, open_high=True)
    checked = [_require_pair(pair) for pair in pairs]
    a = sum(1 for cand, inc in checked if cand and inc)
    b = sum(1 for cand, inc in checked if cand and not inc)
    c = sum(1 for cand, inc in checked if inc and not cand)
    n = len(checked)
    d = n - a - b - c
    p = p_worse(b, c)
    n_effective = math.floor(n / design_effect)
    psi_hat = (b + c) / n if n else None
    limit = reuse = upper = None
    if n:
        upper = stats.mover_d_interval(a, b, c, d, z=_Z_ONE_SIDED_95)[1]
        limit = mdd(n_effective, psi_hat, alpha=alpha, power_target=MDD_POWER)
        if k >= 1:
            reuse = mdd(
                n_effective, psi_hat, alpha=alpha, power_target=MDD_POWER ** (1 / (k + 1))
            )
    return ParityLine(
        e1=a + b,
        e2=a + c,
        n=n,
        n_effective=n_effective,
        b=b,
        c=c,
        p_worse=p,
        U=upper,
        psi_hat=psi_hat,
        mdd=limit,
        mdd_reuse=reuse,
        state="worse" if p <= alpha else "holds",
    )


# --- latency estimators (note §Q4.1, §Q4.2) -------------------------------------------------------


def _require_bootstrap(B: int, levels: tuple[Fraction, Fraction]) -> None:
    if isinstance(B, bool) or not isinstance(B, int) or B < 1:
        raise ValueError(f"B must be a positive int, not {B!r}")
    if levels[0] >= levels[1]:
        raise ValueError(f"levels must be ordered lower then upper, not {levels!r}")


def speedup_ratio(
    x: Sequence[float],
    y: Sequence[float],
    *,
    seed: int,
    B: int,
    levels: tuple[Fraction, Fraction],
) -> stats.BootstrapResult:
    """Note §Q4.1, FR-2: `R = Σx_i / Σy_i` over the paired set — a ratio of means, not a mean of
    per-item ratios — with a paired percentile bootstrap: `B` index resamples drawn with
    `random.Random(seed)`, each item's `(x_i, y_i)` kept together, and the interval read by
    `stats.percentile` at `levels`. `x`/`y` are the incumbent's and the candidate's wall clocks in
    `itemId` order (`PairedItems.latency`); `seed` is the pack's declared seed. Unadjusted: FR-2 is
    one bar. The bar itself is `decisions.bar_holds(R, FR2_MIN_SPEEDUP, direction="at-least")`.
    """
    _require_bootstrap(B, levels)
    if not x or len(x) != len(y):
        raise ValueError(f"speedup_ratio needs equal, non-empty arms, not {len(x)} and {len(y)}")
    if not all(math.isfinite(v) and v >= 0 for v in x) or not all(
        math.isfinite(v) and v > 0 for v in y
    ):
        raise ValueError("speedup_ratio needs finite latencies, every candidate one positive")
    n = len(x)
    rng = random.Random(seed)
    ratios = []
    for _ in range(B):
        draw = [rng.randrange(n) for _ in range(n)]
        ratios.append(sum(x[i] for i in draw) / sum(y[i] for i in draw))
    return stats.BootstrapResult(
        point=sum(x) / sum(y),
        lo=stats.percentile(ratios, level=levels[0]),
        hi=stats.percentile(ratios, level=levels[1]),
        B=B,
        seed=seed,
    )


def mean_latency(
    y: Sequence[float], *, seed: int, B: int, levels: tuple[Fraction, Fraction]
) -> stats.BootstrapResult:
    """Note §Q4.2, FR-10: the candidate's mean `ȳ` over its paired-set items (in `itemId` order),
    with `stats.mean_bootstrap_interval(..., support=None)` as its interval. The bar itself is
    `decisions.bar_holds(ȳ, FR10_MEAN_LATENCY_MS_MAX, direction="at-most")`."""
    _require_bootstrap(B, levels)
    lo, hi = stats.mean_bootstrap_interval(y, B=B, seed=seed, levels=levels, support=None)
    return stats.BootstrapResult(point=sum(y) / len(y), lo=lo, hi=hi, B=B, seed=seed)


# --- bar states over replicates (plan2 §3.6; note §Q4.3, §Q6) -------------------------------------

MET = "met"
NOT_MET = "not met"
LOAD_CHANGED = "not assessable (load changed mid-run)"
SCORED_N_BELOW_MIN = f"not assessable (scored n < {MIN_EVAL_ITEMS})"


UNSTABLE = "unstable across replicates — not met"
_NOT_ASSESSABLE_PREFIX = "not assessable ("


def _not_assessable_reason(state: str) -> str | None:
    """The reason inside a per-replicate `not assessable (<reason>)` state, else `None`."""
    if state.startswith(_NOT_ASSESSABLE_PREFIX) and state.endswith(")"):
        reason = state[len(_NOT_ASSESSABLE_PREFIX) : -1]
        return reason or None
    return None


def combine_replicates(states: Sequence[str], load_count: int) -> str:
    """Plan2 §3.6: one bar's overall state over its replicates, each `MET`, `NOT_MET` or a
    `not assessable (<reason>)` state — `SCORED_N_BELOW_MIN` (note §Q4.3), `LOAD_CHANGED` (an
    `assumed` replicate, note §Q6 "Load rule"), or a caller's own reason. `load_count` is the
    group's replicates that count as a load (an `assumed` one never does).

    `LOAD_CHANGED` replicates are left out of the conjunction; over the rest, in order:
    1. all `met` and `load_count ≥ 2`: `met`;
    2. some `met` and some `not met`: `UNSTABLE`;
    3. otherwise, any `not met`: `not met` — decided, so replicate 2 may be skipped (note §Q2.0);
    4. otherwise `not assessable (<reason>)`: each not-assessable replicate's reason, in first-seen
       order, else `only N load replicate(s) (FR-11a needs 2)`.
    """
    if isinstance(load_count, bool) or not isinstance(load_count, int) or load_count < 0:
        raise ValueError(f"load_count must be a non-negative int, not {load_count!r}")
    counted: list[str] = []
    for state in states:
        if state == LOAD_CHANGED:
            continue
        if state not in (MET, NOT_MET) and _not_assessable_reason(state) is None:
            raise ValueError(f"combine_replicates: {state!r} is not a replicate's bar state")
        counted.append(state)
    if load_count > len(counted):
        raise ValueError(
            f"combine_replicates: {load_count} loads among {len(counted)} non-assumed replicates"
        )
    met, not_met = MET in counted, NOT_MET in counted
    if counted and all(state == MET for state in counted) and load_count >= 2:
        return MET
    if met and not_met:
        return UNSTABLE
    if not_met:
        return NOT_MET
    reasons: list[str] = []
    for state in counted:
        reason = _not_assessable_reason(state)
        if reason is not None and reason not in reasons:
            reasons.append(reason)
    if not reasons:
        reasons.append(f"only {load_count} load replicate(s) (FR-11a needs 2)")
    return f"{_NOT_ASSESSABLE_PREFIX}{'; '.join(reasons)})"


# --- pairing the incumbent and a candidate replicate (note §Q4.3; plan2 §3.3, §3.6; plan3 §3.6) ---

#: A speed-up role's FR-1 cells — the eval stratum's verdict metrics. An item whose incumbent
#: declares one is an eval item; any other is a side-check item. Keyed exactly on
#: `BAR_KIND_BY_ROLE`'s `speed-up` rows; the value is the scorer's own declaration.
_VERDICT_CELLS_BY_ROLE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"guard-judge": PER_KIND_VERDICT_METRICS}
)
#: A failed chat call's `withheldFor` (`results.ItemTiming`); `"load"` is the residency guard's.
_CHAT_WITHHELD = ("timeout", "no_response")
_OVER_LIMIT = "over-limit"
SIDE_CHECK_DECISION_FAILURE = "side-check decision failure"
#: The pack's two strata (v1.3 §3.5), as `pair_items`' `strata` carries them per `itemId`.
_EVAL, _SIDE_CHECK = "eval", "side-check"

Arm = Literal["incumbent", "candidate"]


@dataclass(frozen=True)
class PairedItems:
    """One candidate replicate paired against the incumbent run, by `itemId` (note §Q3.1, §Q4.3).

    - `outcomes[metric]`: `(candidate errs, incumbent errs)` per paired item, ordered by `itemId` —
      a cell's `parity_cell` input. Every verdict cell of the role is here, as `()` when no
      item paired in it; so is every other metric the incumbent declares (pooled, side check).
    - `latency`: `(itemId, x_i, y_i)` over the paired **eval** items, ordered by `itemId` (note
      §Q4.1) — `x_i` the incumbent's wall clock, `y_i` the candidate's per note §Q4.3.
    - `scored_n`: the paired scored eval n (FR-11, note §Q4.3 "Not assessable").
    - `drops[(arm, reason)]`: every dropped **eval** item, counted on the arm that caused it, by
      reason — `"load"`, `"timeout"` or `"no_response"` (note §Q4.3 "Counts"). These print beside
      the bars, and FR-11 counts eval items, so a side-check drop never enters them.
    - `side_check_drops[(arm, reason)]`: the same for side-check items, plus
      `SIDE_CHECK_DECISION_FAILURE` (plan2 §3.3 "Side check"). They print in the side-check
      section only, which does not affect pass/fail (v1.3 §3.5).
    - `over_limit[metric]`: the paired over-limit candidate items, `0` included, for every
      `outcomes` metric (plan3 §3.6, `overLimit k/n` beside each cell).
    """

    outcomes: Mapping[str, tuple[tuple[bool, bool], ...]]
    latency: tuple[tuple[str, float, float], ...]
    scored_n: int
    drops: Mapping[tuple[Arm, str], int]
    side_check_drops: Mapping[tuple[Arm, str], int]
    over_limit: Mapping[str, int]


def _by_item_id(items: Sequence[ItemResult], arm: Arm) -> dict[str, ItemResult]:
    by_id: dict[str, ItemResult] = {}
    for item in items:
        if item.itemId in by_id:
            raise ValueError(f"pair_items: item {item.itemId!r} appears twice in the {arm} arm")
        by_id[item.itemId] = item
    return by_id


def _count(counter: dict[tuple[Arm, str], int], arm: Arm, reason: str) -> None:
    counter[(arm, reason)] = counter.get((arm, reason), 0) + 1


def _wall_clock(item: ItemResult, arm: Arm) -> float:
    if item.timing is None or item.timing.wallClockMs is None:
        raise ValueError(f"pair_items: {arm} item {item.itemId!r} has no wall clock to pair")
    return item.timing.wallClockMs


def _withheld(item: ItemResult) -> str | None:
    return None if item.timing is None else item.timing.withheldFor


def _concordant(candidate: ItemResult) -> bool:
    """Whether the candidate item pairs as concordant (`True`) or on its own answer (`False`).

    Plan3 §3.6's two over-limit rules come first, each explicit: a `handoff`-surface over-limit
    item pairs as any hand-off; a `decisions`-surface one as a guard decision failure. Then
    plan2 §3.3: a kept decision failure is concordant. Any other over-limit or failed shape is
    out of contract and raises."""
    detail = candidate.detail
    if detail.get("reason") == _OVER_LIMIT:
        if detail.get("decidedBy") == "handoff":
            return False
        if detail.get("failedCall") == "decision":
            return True
        raise ValueError(f"pair_items: over-limit item {candidate.itemId!r} has no pairing rule")
    failed = detail.get("failedCall")
    if failed == "decision":
        return True
    if failed is not None:
        raise ValueError(
            f"pair_items: a speed-up candidate never keeps a {failed!r} failure "
            f"({candidate.itemId!r}); its hand-off failure is withheld"
        )
    return False


def pair_items(
    incumbent: Sequence[ItemResult],
    candidate: Sequence[ItemResult],
    *,
    role: str,
    strata: Mapping[str, str],
) -> PairedItems:
    """Note §Q4.3's table, for a speed-up point (plan2 §3.6 "Pairing and drops"). Per `itemId`:

    1. `withheldFor="load"` in either arm drops the item pairwise;
    2. a guard chat call withheld (`"timeout"`/`"no_response"`) in the incumbent run or in a
       candidate hand-off drops it pairwise — over-limit hand-offs included;
    3. a guard decision failure — a kept `failedCall="decision"`, or plan3's `decisions`-surface
       over-limit item — is a **concordant pair**: on every metric the incumbent declares, the
       candidate's error is the incumbent's (plan2 §3.3 "Cell", the one sanctioned cell taken
       from the other arm), and `y_i` = its elapsed time + `x_i`. On a side-check item it is
       dropped and counted instead;
    4. any other item, plan3's handed-off over-limit item included, pairs on its own answer,
       with `y_i` = its `wallClockMs`.

    `strata` maps every `itemId` to its pack `stratum` (`"eval"`/`"side-check"`, v1.3 §3.5) —
    the pack's, because a withheld item's records carry no metric to tell its stratum by (impl
    review P16-n2). It splits the drop counts; a scored incumbent item whose declared metrics
    disagree with its stratum (an eval item with no verdict cell, or the reverse) raises.

    Both arms must hold the same item ids, each once. New-ability points pair nothing: their
    drops are `agreement_items`'.
    """
    cells = _VERDICT_CELLS_BY_ROLE.get(role)
    if cells is None or BAR_KIND_BY_ROLE.get(role) != "speed-up":
        raise ValueError(f"pair_items pairs speed-up points only, not role {role!r}")
    inc_by_id = _by_item_id(incumbent, "incumbent")
    cand_by_id = _by_item_id(candidate, "candidate")
    if inc_by_id.keys() != cand_by_id.keys():
        unmatched = sorted(inc_by_id.keys() ^ cand_by_id.keys())
        raise ValueError(f"pair_items: the arms hold different items: {unmatched!r}")
    unknown = sorted(i for i in inc_by_id if strata.get(i) not in (_EVAL, _SIDE_CHECK))
    if unknown:
        raise ValueError(f"pair_items: no eval/side-check stratum for items {unknown!r}")

    # Every verdict cell is present, empty or not, so no FR-1 line can silently vanish (P16-m1).
    outcomes: dict[str, list[tuple[bool, bool]]] = {cell: [] for cell in cells}
    over_limit: dict[str, int] = dict.fromkeys(cells, 0)
    latency: list[tuple[str, float, float]] = []
    drops: dict[tuple[Arm, str], int] = {}
    side_check_drops: dict[tuple[Arm, str], int] = {}

    for item_id in sorted(inc_by_id):
        inc, cand = inc_by_id[item_id], cand_by_id[item_id]
        is_eval = strata[item_id] == _EVAL
        counter = drops if is_eval else side_check_drops
        if inc.detail.get("failedCall") is not None:
            raise ValueError(f"pair_items: incumbent item {item_id!r} carries a failed call")
        inc_why, cand_why = _withheld(inc), _withheld(cand)
        if "load" in (inc_why, cand_why):
            for arm, why in (("incumbent", inc_why), ("candidate", cand_why)):
                if why == "load":
                    _count(counter, arm, "load")
            continue
        if inc_why in _CHAT_WITHHELD or cand_why in _CHAT_WITHHELD:
            for arm, why in (("incumbent", inc_why), ("candidate", cand_why)):
                if why in _CHAT_WITHHELD:
                    _count(counter, arm, why)
            continue
        inc_errs = {
            metric: inc.scored_outcome(metric) for metric in inc.scoreable if inc.scoreable[metric]
        }
        if not inc_errs:
            raise ValueError(
                f"pair_items: incumbent item {item_id!r} is neither scored nor withheld"
            )
        if is_eval != any(metric in cells for metric in inc_errs):
            raise ValueError(
                f"pair_items: item {item_id!r} is in stratum {strata[item_id]!r} but its "
                f"incumbent declares {sorted(inc_errs)!r}"
            )
        x = _wall_clock(inc, "incumbent")
        if _concordant(cand):
            if not is_eval:
                _count(counter, "candidate", SIDE_CHECK_DECISION_FAILURE)
                continue
            pairs = {metric: (err, err) for metric, err in inc_errs.items()}
            y = _wall_clock(cand, "candidate") + x
        else:
            cand_metrics = {m for m in cand.scoreable if cand.scoreable[m]}
            if cand_metrics != inc_errs.keys():
                raise ValueError(
                    f"pair_items: item {item_id!r} declares {sorted(cand_metrics)!r} in the "
                    f"candidate arm and {sorted(inc_errs)!r} in the incumbent's"
                )
            pairs = {metric: (cand.scored_outcome(metric), err) for metric, err in inc_errs.items()}
            y = _wall_clock(cand, "candidate")
        is_over_limit = cand.detail.get("reason") == _OVER_LIMIT
        for metric, pair in pairs.items():
            outcomes.setdefault(metric, []).append(pair)
            over_limit[metric] = over_limit.get(metric, 0) + int(is_over_limit)
        if is_eval:
            latency.append((item_id, x, y))

    return PairedItems(
        outcomes={metric: tuple(pairs) for metric, pairs in outcomes.items()},
        latency=tuple(latency),
        scored_n=len(latency),
        drops=drops,
        side_check_drops=side_check_drops,
        over_limit=over_limit,
    )


@dataclass(frozen=True)
class AgreementItems:
    """One new-ability candidate replicate's FR-10 input (note §Q4.3, §Q5.1): a single arm, so
    nothing pairs, but the drops are counted as `pair_items` counts them.

    - `outcomes[metric]`: whether each scored item agreed, per class metric (`agreeWhenYes`,
      `agree[<label>]`, …), ordered by `itemId` — `k_j/n_j` is `sum`/`len`. Every class of the
      role's answer space is here, as `()` when no item scored in it, so no FR-10 line can
      silently vanish.
    - `latency`: `(itemId, y_i)` over the scored items, ordered by `itemId` — note §Q4.2's
      `mean_latency` input, `y_i` the item's wall clock (a kept failure's runs to its raise).
    - `scored_n`: the scored eval n (FR-11, note §Q4.3 "Not assessable").
    - `drops[("candidate", reason)]`: every dropped item, by reason. Only `"load"` drops one: a
      kept failed call is scored fail-closed, and an over-limit hand-off on the hand-off LLM's
      answer.
    - `over_limit[metric]`: the scored over-limit items per class, `0` included (plan3 §3.6,
      `overLimit k/n` beside each class line).
    """

    outcomes: Mapping[str, tuple[bool, ...]]
    latency: tuple[tuple[str, float], ...]
    scored_n: int
    drops: Mapping[tuple[Arm, str], int]
    over_limit: Mapping[str, int]


def agreement_items(candidate: Sequence[ItemResult], *, role: str) -> AgreementItems:
    """Note §Q4.3's table, for a new-ability point (plan2 §3.6: "New-ability points pair nothing,
    but use the same drop accounting over the candidate run"). Per item, by `itemId`:

    1. `withheldFor="load"` drops it, counted;
    2. any other item is scored on its own record, one class metric declared: a kept failed call
       (`failedCall` `"decision"`/`"handoff"`) already carries the fail-closed answer's count, and
       an over-limit hand-off the hand-off LLM's (`scoring.agreement`).

    A chat call withheld (`"timeout"`/`"no_response"`) raises: a new ability keeps its failed
    calls (`HANDOFF_FAILURE_BY_ROLE`), so such a record is not a candidate run of this role. So
    does a repeated `itemId`, an item declaring other than exactly one of the role's classes, or
    a scored item with no wall clock. Every item of a valid new-ability pack is `eval` (FR-15)."""
    if BAR_KIND_BY_ROLE.get(role) != "new-ability":
        raise ValueError(f"agreement_items takes new-ability points only, not role {role!r}")
    classes = per_class_agreement_metrics(ANSWER_SPACE_BY_ROLE[role])
    by_id = _by_item_id(candidate, "candidate")
    outcomes: dict[str, list[bool]] = {metric: [] for metric in classes}
    over_limit: dict[str, int] = dict.fromkeys(classes, 0)
    latency: list[tuple[str, float]] = []
    drops: dict[tuple[Arm, str], int] = {}
    for item_id in sorted(by_id):
        item = by_id[item_id]
        why = _withheld(item)
        if why == "load":
            _count(drops, "candidate", "load")
            continue
        if why is not None:
            raise ValueError(
                f"agreement_items: item {item_id!r} is withheld for {why!r}; a new ability keeps "
                "a failed call and scores it fail-closed"
            )
        declared = [metric for metric in item.scoreable if item.scoreable[metric]]
        if len(declared) != 1 or declared[0] not in outcomes:
            raise ValueError(
                f"agreement_items: item {item_id!r} declares {sorted(declared)!r}, not exactly one "
                f"of {list(classes)!r}"
            )
        metric = declared[0]
        outcomes[metric].append(bool(item.scored_outcome(metric)))
        over_limit[metric] += int(item.detail.get("reason") == _OVER_LIMIT)
        latency.append((item_id, _wall_clock(item, "candidate")))
    return AgreementItems(
        outcomes={metric: tuple(agreed) for metric, agreed in outcomes.items()},
        latency=tuple(latency),
        scored_n=len(latency),
        drops=drops,
        over_limit=over_limit,
    )
