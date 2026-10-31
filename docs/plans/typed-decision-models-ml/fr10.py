"""Figures for docs/plans/typed-decision-models-ml.md §Q1.3 (sizes, labelling volume), §Q2.2 and §Q5.

stdlib only. Run from the repo root:  python3 -I docs/plans/typed-decision-models-ml/fr10.py
"""
from math import comb, sqrt

Z = 1.959963984540054  # stats._Z_95


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z / den) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def p_pass(n, a):
    """P(10*k >= 9*n) for k ~ Bin(n, a): the measured-value bar of FR-10."""
    kmin = -(-9 * n // 10)
    return sum(comb(n, k) * a**k * (1 - a) ** (n - k) for k in range(kmin, n + 1))


if __name__ == "__main__":
    As = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    print("§Q5.3 per-class P(measured >= 0.90 | true agreement a)")
    print("n    " + " ".join(f"a={a:.2f}" for a in As))
    for n in [40, 50, 60, 80, 100, 150]:
        print(f"{n:<4} " + " ".join(f"{p_pass(n, a):6.3f}" for a in As))

    print("\n§Q1.3 sawtooth: P(pass | a=0.85) for n=40..70 (bar 10k >= 9n)")
    print(" ".join(f"{n}:{p_pass(n, 0.85):.3f}" for n in range(40, 71)))
    bad = [n for n in range(1, 301) if p_pass(n, 0.85) > 0.20]
    print("largest n <= 300 with P > 0.20:", max(bad), "-> every n >=", max(bad) + 1, "keeps it <= 0.20")

    print("\n§Q5.3 / §Q2.2 conjunction: P(all classes pass) at common true a")
    for n, k in [(60, 5), (100, 2), (40, 5)]:
        print(f"n={n}, {k} classes: " + " ".join(f"a={a:.2f}:{p_pass(n, a) ** k:.3f}"
                                               for a in [0.90, 0.92, 0.93, 0.94, 0.95]))

    print("\n§Q5.3 Wilson half-width at k = ceil(0.9 n)")
    for n in [40, 50, 60, 80, 100]:
        k = -(-9 * n // 10)
        lo, hi = wilson(k, n)
        print(f"n={n} k={k} [{lo:.3f}, {hi:.3f}] half-width {(hi - lo) / 2:.3f}")

    print("\n§Q5.4 worked message-type eval, 60 per label")
    for lab, k in [("order", 58), ("question", 56), ("complaint", 54), ("escalation", 53), ("other", 55)]:
        lo, hi = wilson(k, 60)
        print(f"  {lab:<10} {k}/60 = {k / 60:.3f} [{lo:.3f}, {hi:.3f}] {'met' if 10 * k >= 9 * 60 else 'not met'}")

    print("\n§Q1.3 labelling volume (labeller calls; one variant per cluster in eval and calibration)")
    eval_items = 4 * 100 + 2 * 100 + 5 * 60
    kept = 2 * eval_items  # calibration = eval
    prior_calls = 3 * 200  # plan prior: 200 eval items per point, labelled once
    over, votes = 1.6, 3
    labelled = kept * over
    calls = labelled * votes
    train_kept = 1000 + 1000 + 1500
    train_calls = train_kept * over * votes
    print(f"eval {eval_items}, eval+calibration kept {kept} ({kept / 600:.1f}x the prior's 600 items)")
    print(f"labeller calls without over-generation {kept * votes} ({kept * votes / prior_calls:.0f}x prior {prior_calls})")
    print(f"with {over}x over-generation: {labelled:.0f} items labelled, {calls:.0f} calls ({calls / prior_calls:.1f}x)")
    print(f"U8 train split (kept {train_kept}): +{train_calls:.0f} calls; total {calls + train_calls:.0f} "
          f"({(calls + train_calls) / prior_calls:.0f}x)")
