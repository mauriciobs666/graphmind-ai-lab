"""The guard ladder and the new-ability ladders (plan4 §4.4 U6f.6-U6f.7; note §Q2.1-§Q2.2, fit
note §1-§2, §5): pure, stdlib + `fitstats` only.

`guard_ladder` reads one text's `ScoredCall`s over the calibration items and returns a
`LadderResult`. Two families share one search: the hand-off family (every band `l <= h` over the
distinct calibration `p` plus `{0, 1}`) and the decision-only family (the empty band `l == h == t`,
`s = 0`; fit note §1). Everything that decides a rung is an exact `Fraction` or an int: the
`s`-vs-ceiling tests, the `0.05` margin (`Fraction(1, 20)`) and the McNemar tail.

Scoring, per item (§3.11, note §Q2.1, fit note §1):
- the hand-off call is wrong unless its outcome is `ok` and its answer is the label: a `failed` (or
  over-limit) hand-off is fail-closed. Its elapsed time enters `t̄_{L|H}` and `t̄_L` whatever the
  outcome;
- the decision call is `ok` (a `p` in [0, 1]), `over-limit` (no `p`: a hand-off in every band on
  the hand-off family, concordant and no hand-off on the decision-only family) or `failed`
  (concordant on both families: it adds 0 to every `D_c` and is never a hand-off).

`b_c`/`c_c` per cell count decided items where the model errs and the LLM does not / the reverse;
`D_c = b_c - c_c`, `m_c = b_c + c_c`. A cell is whichever strings `cells` carries.

Rungs, first attainable (the hand-off family; on the decision-only family rung 4 is never
reported, see `_fallback`):
1. every `D_c <= 0`, `s <= s_2 - 1/20`; minimise `(s, max D_c, band width)`, `D_c<=0`;
2. every cell's one-sided exact McNemar not rejecting (`p_worse > 1/20`), `s <= s_2 - 1/20`;
   minimise `(max D_c, max m_c, s)`, `mcnemar-nonreject`;
3. rung 1 then rung 2 again with `s <= s_2` (the first that is attainable);
4. every `D_c <= 0` at the smallest `s`, latency ignored, as rung 1's key.
Exact ties go to the sweep's first setting (smaller `l`, then smaller `h`).
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from decisionmodels.fitstats import (
    FAMILIES,
    LadderResult,
    ScoredCall,
    ceiling_s2,
    ceiling_s350,
    mcnemar_worse_p_exact,
    mean_handoff_ms,
)

MARGIN = Fraction(1, 20)
ALPHA = Fraction(1, 20)
TARGET_NO_WORSE = "D_c<=0"
TARGET_NONREJECT = "mcnemar-nonreject"


@dataclass(frozen=True)
class _Item:
    cell: str
    label: bool
    llm_wrong: bool
    hand_seconds: Fraction
    decision: str  # "ok" | "over-limit" | "failed"
    p: float | None


@dataclass(frozen=True)
class _Setting:
    l: float  # noqa: E741
    h: float
    n_handoff: int
    handoff_seconds: Fraction
    b: tuple[int, ...]
    c: tuple[int, ...]


def _collect(
    calls: Iterable[ScoredCall],
    labels: Mapping[str, bool],
    cells: Mapping[str, str],
    text_id: str,
) -> tuple[dict[str, _Item], list[ScoredCall], list[ScoredCall]]:
    if set(labels) != set(cells):
        raise ValueError("guard_ladder: labels and cells must name the same items")
    hand, dec = index_calls(calls, labels, text_id, "guard_ladder")
    items: dict[str, _Item] = {}
    for item_id, label in labels.items():
        h, d = hand[item_id], dec[item_id]
        p: float | None = None
        if d.outcome == "ok":
            p = d.answer
            if isinstance(p, bool) or not isinstance(p, int | float) or not 0 <= p <= 1:
                raise ValueError(f"guard_ladder: decision p must be in [0, 1], got {p!r}")
        items[item_id] = _Item(
            cells[item_id],
            label,
            not (h.outcome == "ok" and h.answer is label),
            Fraction(h.elapsedSeconds),
            d.outcome,
            p,
        )
    return items, list(hand.values()), list(dec.values())


def _enumerate(items: dict[str, _Item], family: str, cell_names: list[str]) -> list[_Setting]:
    """Every setting of the family, with per-cell `b`/`c` from prefix sums over the `p` order."""
    scored = sorted((it for it in items.values() if it.p is not None), key=lambda it: it.p)  # type: ignore[arg-type,return-value]
    ps = [it.p for it in scored]
    values = sorted({0.0, 1.0, *ps})  # type: ignore[arg-type]
    n_cells = len(cell_names)
    index = {name: i for i, name in enumerate(cell_names)}

    def prefix(side_yes: bool, want_b: bool) -> list[list[int]]:
        out = [[0] * n_cells]
        for it in scored:
            row = list(out[-1])
            model_wrong = (not it.label) if side_yes else it.label
            if model_wrong != it.llm_wrong and model_wrong == want_b:
                row[index[it.cell]] += 1
            out.append(row)
        return out

    no_b, no_c = prefix(False, True), prefix(False, False)
    yes_b, yes_c = prefix(True, True), prefix(True, False)
    seconds = [Fraction(0)]
    for it in scored:
        seconds.append(seconds[-1] + it.hand_seconds)
    over = [it for it in items.values() if it.decision == "over-limit"]
    over_n = len(over) if family == "handoff" else 0
    over_seconds = sum((it.hand_seconds for it in over), Fraction(0)) if family == "handoff" else 0
    below = [bisect_left(ps, v) for v in values]  # type: ignore[type-var]
    total = len(scored)
    settings: list[_Setting] = []
    for i, lo in enumerate(values):
        for j in range(i, i + 1) if family == "decision-only" else range(i, len(values)):
            hi = values[j]
            lo_n, hi_n = below[i], below[j]
            settings.append(
                _Setting(
                    lo,
                    hi,
                    hi_n - lo_n + over_n,
                    seconds[hi_n] - seconds[lo_n] + over_seconds,
                    tuple(no_b[lo_n][k] + yes_b[total][k] - yes_b[hi_n][k] for k in range(n_cells)),
                    tuple(no_c[lo_n][k] + yes_c[total][k] - yes_c[hi_n][k] for k in range(n_cells)),
                )
            )
    return settings


def guard_ladder(
    calls: Iterable[ScoredCall],
    *,
    labels: Mapping[str, bool],
    cells: Mapping[str, str],
    text_id: str,
    family: str,
) -> LadderResult:
    """The guard ladder for one text over `family` (`handoff` or `decision-only`).

    `labels` (item -> true answer, `True` = yes) and `cells` (item -> FR-13 cell) name the
    calibration items; every item needs one hand-off call and one decision call for `text_id`
    (`ScoredCall`s of other texts are ignored). Raises `ValueError` on a malformed input, and on the
    hand-off family when no rung is attainable (only possible when an item's `p` is exactly 1, which
    no band can hand off; note §Q2.1's "rung 4 always exists" assumes it cannot be).
    """
    if family not in FAMILIES:
        raise ValueError(f"guard_ladder: family must be one of {FAMILIES}, got {family!r}")
    items, hand_calls, dec_calls = _collect(calls, labels, cells, text_id)
    if not items:
        raise ValueError("guard_ladder: no calibration items")
    cell_names = sorted(set(cells.values()))
    n_items = len(items)
    t_l = mean_handoff_ms(hand_calls)
    t_d = 1000 * sum((Fraction(c.elapsedSeconds) for c in dec_calls), Fraction(0)) / len(dec_calls)
    stuck = sum(
        1
        for it in items.values()
        if it.decision == "ok" and it.p == 1 and not it.label and not it.llm_wrong
    )
    settings = _enumerate(items, family, cell_names)

    def share(st: _Setting) -> Fraction:
        return Fraction(st.n_handoff, n_items)

    def t_lh(st: _Setting) -> Fraction:
        # an empty hand-off set has no t̄_{L|H} of its own: every hand-off call, as fit note §1
        return 1000 * st.handoff_seconds / st.n_handoff if st.n_handoff else t_l

    def s2(st: _Setting) -> Fraction:
        return ceiling_s2(t_l, t_d, t_lh(st))

    def no_worse(st: _Setting) -> bool:
        return all(b <= c for b, c in zip(st.b, st.c, strict=True))

    def nonreject(st: _Setting) -> bool:
        return all(mcnemar_worse_p_exact(b, c) > ALPHA for b, c in zip(st.b, st.c, strict=True))

    def max_d(st: _Setting) -> int:
        return max(b - c for b, c in zip(st.b, st.c, strict=True))

    def max_m(st: _Setting) -> int:
        return max(b + c for b, c in zip(st.b, st.c, strict=True))

    def key_no_worse(st: _Setting) -> tuple[Fraction | int, ...]:
        return (share(st), max_d(st), Fraction(st.h) - Fraction(st.l))

    def key_nonreject(st: _Setting) -> tuple[Fraction | int, ...]:
        return (max_d(st), max_m(st), share(st))

    def best(pool: Iterable[_Setting], key) -> _Setting | None:  # type: ignore[no-untyped-def]
        return min(pool, key=lambda st: (key(st), st.l, st.h), default=None)

    def within(limit_margin: Fraction):  # type: ignore[no-untyped-def]
        return lambda st: share(st) <= s2(st) - limit_margin

    def search(quality, key, latency) -> _Setting | None:  # type: ignore[no-untyped-def]
        return best((st for st in settings if quality(st) and latency(st)), key)

    def result(
        st: _Setting,
        rung: int | None,
        target: str,
        key,
        agreement_unmet: bool = False,  # type: ignore[no-untyped-def]
    ) -> LadderResult:
        s = share(st)
        ceiling = s2(st)
        policy = {"l": st.l, "h": st.h, "s": s} if family == "handoff" else {"t": st.l, "s": s}
        return LadderResult(
            family=family,
            rung=rung,
            qualityTarget=target,
            policy=policy,
            key=key(st),
            agreementUnmet=agreement_unmet,
            latencyUnmet=not s <= ceiling,
            barValues={
                "D_c": {n: st.b[k] - st.c[k] for k, n in enumerate(cell_names)},
                "m_c": {n: st.b[k] + st.c[k] for k, n in enumerate(cell_names)},
                "p_worse": {
                    n: mcnemar_worse_p_exact(st.b[k], st.c[k]) for k, n in enumerate(cell_names)
                },
                "s": s,
                "s2": ceiling,
                "tLH_ms": t_lh(st),
                "tL_ms": t_l,
                "td_ms": t_d,
                "stuckCount": stuck,
            },
        )

    def anywhere(_st: _Setting) -> bool:
        return True

    at_margin, at_ceiling = within(MARGIN), within(Fraction(0))
    for rung, quality, key, target, latency in (
        (1, no_worse, key_no_worse, TARGET_NO_WORSE, at_margin),
        (2, nonreject, key_nonreject, TARGET_NONREJECT, at_margin),
        (3, no_worse, key_no_worse, TARGET_NO_WORSE, at_ceiling),
        (3, nonreject, key_nonreject, TARGET_NONREJECT, at_ceiling),
    ):
        found = search(quality, key, latency)
        if found is not None:
            return result(found, rung, target, key)
    if family == "handoff":
        found = search(no_worse, key_no_worse, anywhere)
        if found is not None:
            return result(found, 4, TARGET_NO_WORSE, key_no_worse)

    # fit note §2: no rung reached. The hand-off family gets here only when an item is stuck.
    def key_agreement(st: _Setting) -> tuple[Fraction | int, ...]:
        return (max_d(st), max_m(st), share(st)) if family == "handoff" else (max_d(st), max_m(st))

    return _fallback(
        settings, result, no_worse, nonreject, key_no_worse, key_nonreject, key_agreement, best
    )


def _fallback(  # type: ignore[no-untyped-def]
    settings, result, no_worse, nonreject, key_no_worse, key_nonreject, key_agreement, best
) -> LadderResult:
    """Fit note §2: no rung reached, `rung` `None`, latency ignored.

    The first quality target attainable (`D_c<=0`, else `mcnemar-nonreject`) with its rung-1/2
    minimiser (the latency-only miss); otherwise `fit-infeasible: agreement`: the setting minimising
    `(max D_c, max m_c)` (decision-only) or `(max D_c, max m_c, s)` (hand-off, where `s` varies and
    belongs in the key), the first on an exact tie. On the hand-off family rung 4 has already taken
    any `D_c<=0` setting, so only the non-rejection and agreement steps can fire there.
    """
    found = best((st for st in settings if no_worse(st)), key_no_worse)
    if found is not None:
        return result(found, None, TARGET_NO_WORSE, key_no_worse)
    found = best((st for st in settings if nonreject(st)), key_nonreject)
    if found is not None:
        return result(found, None, TARGET_NONREJECT, key_nonreject)
    found = best(settings, key_agreement)
    return result(found, None, TARGET_NONREJECT, key_agreement, agreement_unmet=True)


# --- the new-ability ladders (U6f.7; note §Q2.2, fit note §1, §4, §5) ---

T_RUNGS = tuple(Fraction(t, 100) for t in (94, 93, 92, 91, 90))
TARGET_MAX_MIN = "max-min-agreement"


@dataclass(frozen=True)
class _NaItem:
    cls: Hashable
    hand_seconds: Fraction
    decision: str  # "ok" | "over-limit" | "failed"
    x: float | None  # predicate p / choice confidence
    llm_right: bool
    fail_closed_right: bool
    regions: tuple[tuple[bool, bool], ...]  # (hand-off?, right?) below l / in [l, h) / from h up


@dataclass(frozen=True)
class _NaSetting:
    policy: tuple[tuple[str, float | int], ...]
    n_handoff: int
    handoff_seconds: Fraction
    right: tuple[int, ...]  # per class


def index_calls(
    calls: Iterable[ScoredCall], labels: Mapping[str, object], text_id: str, who: str
) -> tuple[dict[str, ScoredCall], dict[str, ScoredCall]]:
    hand: dict[str, ScoredCall] = {}
    dec: dict[str, ScoredCall] = {}
    for call in calls:
        if call.itemId not in labels:
            raise ValueError(f"{who}: call for unknown item {call.itemId!r}")
        if call.callKind == "handoff":
            target = hand
        elif call.textId == text_id:
            target = dec
        else:
            continue
        if call.itemId in target:
            raise ValueError(f"{who}: duplicate {call.callKind} call for {call.itemId!r}")
        target[call.itemId] = call
    for name, got in (("hand-off", hand), ("decision", dec)):
        missing = sorted(set(labels) - set(got))
        if missing:
            raise ValueError(f"{who}: items without a {name} call: {missing!r}")
    return hand, dec


def _predicate_item(label: object, hand: ScoredCall, dec: ScoredCall, who: str) -> _NaItem:
    if not isinstance(label, bool):
        raise ValueError(f"{who}: label must be a bool, got {label!r}")
    p: float | None = None
    if dec.outcome == "ok":
        p = dec.answer
        if isinstance(p, bool) or not isinstance(p, int | float) or not 0 <= p <= 1:
            raise ValueError(f"{who}: decision p must be in [0, 1], got {p!r}")
    closed = not label  # fail-closed answers "no"
    llm_right = (hand.answer is label) if hand.outcome == "ok" else closed
    regions = ((False, not label), (True, llm_right), (False, label))
    return _NaItem(label, Fraction(hand.elapsedSeconds), dec.outcome, p, llm_right, closed, regions)


def _choice_item(label: object, hand: ScoredCall, dec: ScoredCall, who: str) -> _NaItem:
    if not isinstance(label, str):
        raise ValueError(f"{who}: label must be a str, got {label!r}")
    conf: float | None = None
    choice: object = None
    if dec.outcome == "ok":
        ans = dec.answer
        if not (
            isinstance(ans, tuple)
            and len(ans) == 2
            and isinstance(ans[0], str)
            and not isinstance(ans[1], bool)
            and isinstance(ans[1], int | float)
            and 0 <= ans[1] <= 1
        ):
            raise ValueError(f"{who}: choice answer must be (choice, confidence), got {ans!r}")
        choice, conf = ans
    llm_right = hand.outcome == "ok" and hand.answer == label  # fail-closed is "no label"
    regions = ((True, llm_right), (True, llm_right), (False, choice == label))
    seconds = Fraction(hand.elapsedSeconds)
    return _NaItem(label, seconds, dec.outcome, conf, llm_right, False, regions)


def _na_settings(
    items: list[_NaItem], classes: list[Hashable], family: str, kind: str
) -> list[_NaSetting]:
    """Every setting, with per-class right counts from prefix sums over the score order.

    A predicate setting is a band `(l, h)` (`l == h` on the decision-only family): below `l` the
    model says no, from `h` up yes, between them a hand-off. A choice setting is `c`: below it a
    hand-off, from it up the model's choice (`c == 0` on the decision-only family). An over-limit
    item is a hand-off in every setting on the hand-off family and fail-closed on the other; a
    failed decision is never a hand-off and always fail-closed.
    """
    index = {cls: i for i, cls in enumerate(classes)}
    scored = sorted((it for it in items if it.x is not None), key=lambda it: it.x)  # type: ignore[arg-type,return-value]
    xs = [it.x for it in scored]
    values = sorted({0.0, 1.0, *xs})  # type: ignore[arg-type]
    total = len(scored)

    right = [[[0] * len(classes)] for _ in range(3)]
    hand_n = [[0] for _ in range(3)]
    hand_sec: list[list[Fraction]] = [[Fraction(0)] for _ in range(3)]
    for it in scored:
        for r in range(3):
            handed, ok = it.regions[r]
            row = list(right[r][-1])
            row[index[it.cls]] += ok
            right[r].append(row)
            hand_n[r].append(hand_n[r][-1] + handed)
            hand_sec[r].append(hand_sec[r][-1] + (it.hand_seconds if handed else 0))

    base = [0] * len(classes)
    base_n, base_sec = 0, Fraction(0)
    for it in items:
        if it.x is not None:
            continue
        if it.decision == "over-limit" and family == "handoff":
            base[index[it.cls]] += it.llm_right
            base_n += 1
            base_sec += it.hand_seconds
        else:
            base[index[it.cls]] += it.fail_closed_right

    def build(policy: tuple[tuple[str, float | int], ...], lo_n: int, hi_n: int) -> _NaSetting:
        spans = ((0, lo_n), (lo_n, hi_n), (hi_n, total))
        counts = list(base)
        n, sec = base_n, base_sec
        for r, (a, b) in enumerate(spans):
            for k in range(len(classes)):
                counts[k] += right[r][b][k] - right[r][a][k]
            n += hand_n[r][b] - hand_n[r][a]
            sec += hand_sec[r][b] - hand_sec[r][a]
        return _NaSetting(policy, n, sec, tuple(counts))

    below = [bisect_left(xs, v) for v in values]  # type: ignore[type-var]
    settings: list[_NaSetting] = []
    if kind == "choice":
        for v, at in zip(values, below, strict=True):
            if family == "handoff" or v == 0:
                settings.append(build((("c", v),), at, at))
        return settings
    for i, lo in enumerate(values):
        for j in range(i, i + 1) if family == "decision-only" else range(i, len(values)):
            if family == "decision-only":
                policy = (("t", lo),)
            else:
                policy = (("l", lo), ("h", values[j]))
            settings.append(build(policy, below[i], below[j]))
    return settings


def _new_ability_ladder(
    calls: Iterable[ScoredCall],
    labels: Mapping[str, object],
    text_id: str,
    family: str,
    kind: str,
) -> LadderResult:
    who = f"{kind}_ladder"
    if family not in FAMILIES:
        raise ValueError(f"{who}: family must be one of {FAMILIES}, got {family!r}")
    calls = list(calls)
    hand, dec = index_calls(calls, labels, text_id, who)
    if not labels:
        raise ValueError(f"{who}: no calibration items")
    build_item = _predicate_item if kind == "predicate" else _choice_item
    items = [build_item(label, hand[i], dec[i], who) for i, label in labels.items()]
    classes = sorted({it.cls for it in items})
    n_class = [sum(1 for it in items if it.cls == cls) for cls in classes]
    n_items = len(items)
    hand_calls = list(hand.values())
    t_l = mean_handoff_ms(hand_calls)
    t_d = 1000 * sum((Fraction(c.elapsedSeconds) for c in dec.values()), Fraction(0)) / len(dec)
    unfixable = {cls: 0 for cls in classes}
    for it in items:
        # fit note §5: decided wrongly at every setting = an `ok` score of exactly 1 whose
        # from-`h`-up (predicate "yes") / kept (choice) answer is wrong
        if it.decision == "ok" and it.x == 1 and not it.regions[2][1]:
            unfixable[it.cls] += 1
    settings = _na_settings(items, classes, family, kind)

    def share(st: _NaSetting) -> Fraction:
        return Fraction(st.n_handoff, n_items)

    def t_lh(st: _NaSetting) -> Fraction:
        return 1000 * st.handoff_seconds / st.n_handoff if st.n_handoff else t_l

    def s350(st: _NaSetting) -> Fraction:
        return ceiling_s350(t_d, t_lh(st))

    def agreement(st: _NaSetting) -> tuple[Fraction, ...]:
        return tuple(Fraction(r, n) for r, n in zip(st.right, n_class, strict=True))

    def min_agreement(st: _NaSetting) -> Fraction:
        return min(agreement(st))

    def result(st: _NaSetting, rung: int, target: str, key: tuple[Fraction, ...]) -> LadderResult:
        s = share(st)
        ceiling = s350(st)
        policy = {**dict(st.policy), "s": s}
        return LadderResult(
            family=family,
            rung=rung,
            qualityTarget=target,
            policy=policy,
            key=key,
            agreementUnmet=rung == 4,
            latencyUnmet=not s <= ceiling,
            barValues={
                "agreement": dict(zip(classes, agreement(st), strict=True)),
                "nClass": dict(zip(classes, n_class, strict=True)),
                "minAgreement": min_agreement(st),
                "s": s,
                "s350": ceiling,
                "tLH_ms": t_lh(st),
                "tL_ms": t_l,
                "td_ms": t_d,
                "unfixableByClass": unfixable,
            },
        )

    def smallest_s(pool: Iterable[_NaSetting]) -> _NaSetting | None:
        return min(pool, key=share, default=None)  # the first on an exact tie

    for limit_margin in (MARGIN, Fraction(0), None):  # rungs 1, 2, 3
        for t in T_RUNGS:
            found = smallest_s(
                st
                for st in settings
                if min_agreement(st) >= t
                and (limit_margin is None or share(st) <= s350(st) - limit_margin)
            )
            if found is not None:
                rung = 3 if limit_margin is None else 1 if limit_margin == MARGIN else 2
                return result(found, rung, f"T={float(t):.2f}", (-t, share(found)))
    found = min(settings, key=lambda st: (-min_agreement(st), share(st)))
    return result(found, 4, TARGET_MAX_MIN, (-min_agreement(found), share(found)))


def predicate_ladder(
    calls: Iterable[ScoredCall],
    *,
    labels: Mapping[str, bool],
    text_id: str,
    family: str,
) -> LadderResult:
    """Note §Q2.2's ladder for a predicate point (join-in): `labels` maps item -> true answer.

    Every item needs one hand-off call and one decision call for `text_id` (others' decision calls
    are ignored); an `ok` decision's answer is `p` in [0, 1], an `ok` hand-off's is a bool. A
    `failed` or over-limit hand-off, an over-limit decision on the decision-only family and a
    `failed` decision all answer "no". Total: it never raises on a stuck item. The policy is
    `{"l", "h", "s"}` (hand-off family) or `{"t", "s"}`. Settings are swept `l` then `h` ascending
    (choice: `c` ascending); settings tied on `s` at a rung go to the first of the sweep.
    """
    return _new_ability_ladder(calls, labels, text_id, family, "predicate")


def choice_ladder(
    calls: Iterable[ScoredCall],
    *,
    labels: Mapping[str, str],
    text_id: str,
    family: str,
) -> LadderResult:
    """Note §Q2.2's ladder for a choice point (message type): `labels` maps item -> true label.

    As `predicate_ladder`, but an `ok` decision's answer is the pair `(choice, confidence)`, an `ok`
    hand-off's is the label string, and fail-closed is "no label" (always wrong). The policy is
    `{"c": minConfidence, "s"}`; the decision-only family is the single point `c = 0`.
    """
    return _new_ability_ladder(calls, labels, text_id, family, "choice")
