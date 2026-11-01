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

    print("\n§Q1.3 labelling volume (subagent runs, three votes per item, two-round fill)")
    kept_by_point = {"guard": 2 * 4 * 100, "join-in": 2 * 2 * 100, "message-type": 2 * 5 * 60}  # calibration = eval
    rounds = (1.1, 0.2)  # round 1, then same-design top-up: planning figures, replaced by round 1's P(label | stratum)
    b_gen = 25

    def ceil(x):
        return int(-(-x // 1))

    def volume(b_label):
        labelled = votes = label_runs = gen_runs = 0
        for k in kept_by_point.values():
            for r in rounds:
                n = round(k * r)
                labelled += n
                votes += 3 * n
                label_runs += 3 * ceil(n / b_label)  # each vote re-partitions the round
                gen_runs += ceil(n / b_gen)
        return labelled, votes, label_runs, gen_runs

    kept = sum(kept_by_point.values())
    labelled, votes, label_runs, gen_runs = volume(50)
    print(f"kept {kept} (eval {kept // 2} + calibration {kept // 2}); labelled {labelled}; item-votes {votes}")
    print(f"runs at 50 per label batch: {label_runs} label (sonnet), {gen_runs} generation (opus)")
    pilot_label, pilot_gen = 3 * 2 + 10, 4  # 100 guard items: votes 1-3 at 50, one screen vote at 10
    bounds = [("pessimistic", 143e3, 8e3, 8e3), ("optimistic", 66e3, 1.2e3, 0.6e3)]
    for name, F, c_gen, c_lab in bounds:
        tokens = (label_runs + gen_runs) * F + labelled * c_gen + votes * c_lab
        pilot = (pilot_label + pilot_gen) * F + 100 * c_gen + 400 * c_lab
        print(f"tokens processed, {name} (F={F / 1e3:.0f}k, c_gen={c_gen / 1e3:.1f}k, "
              f"c_label={c_lab / 1e3:.1f}k): {tokens / 1e6:.0f}M; pilot {pilot / 1e6:.1f}M")
    _, _, label_runs_25, _ = volume(25)
    print(f"pilot: {pilot_gen} generation + {pilot_label} label runs; if the screen sets the cap to 25: "
          f"{label_runs_25} label runs")
    train_kept = 1000 + 1000 + 1500
    print(f"U8 train split (kept {train_kept}, {sum(rounds):.1f}x labelled, three votes): "
          f"+{round(train_kept * sum(rounds) * 3)} item-votes")
    from math import erf
    se = sqrt(0.30 / 100)  # paired disagreement gap, 100 items, d ~ 0.15 per comparison
    trigger = {g: 0.5 * (1 + erf((g - 0.10) / (se * sqrt(2)))) for g in (0.0, 0.10, 0.14, 0.20)}
    print(f"pilot screen (gap >= 0.10, normal approx., SE {se:.3f}): P(trigger | true gap) "
          + " ".join(f"{g:.2f}:{p:.2f}" for g, p in trigger.items()))
