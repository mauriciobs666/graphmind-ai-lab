"""Figures for docs/plans/typed-decision-models-ml.md §Q4.4 (break-even hand-off share, worked FR-2 ratio).

stdlib only. Run from the repo root:
    python3 -I docs/plans/typed-decision-models-ml/latency.py model-bench/results/runs
Reads the stored incumbent guard runs guard-judge-understanding-qwen_qwen3-4b-2507-*.json.
"""
import glob
import json
import os
import random
import statistics as st
import sys
from fractions import Fraction

T_D = 20.0  # decision-call warm latency, ms (requirements' Problem section) — an input until U7 measures it
B, SEED = 10_000, 20261030


def pct(sorted_values, level):
    """Hyndman-Fan type 1 by stats.percentile's integer rank expression."""
    x = len(sorted_values)
    rank = max(1, min(x, -(-level.numerator * x // level.denominator)))
    return sorted_values[rank - 1]


def ratio_ci(inc, cand, seed):
    n = len(inc)
    r = random.Random(seed)
    rs = sorted(sum(inc[i] for i in idx) / sum(cand[i] for i in idx)
                for idx in ([r.randrange(n) for _ in range(n)] for _ in range(B)))
    return sum(inc) / sum(cand), pct(rs, Fraction(1, 40)), pct(rs, Fraction(39, 40))


if __name__ == "__main__":
    runs = sorted(glob.glob(os.path.join(sys.argv[1], "guard-judge-understanding-qwen_qwen3-4b-2507-*.json")))
    means = []
    for f in runs:
        d = json.load(open(f))
        items = [it for it in d["items"] if it["timing"].get("withheldFor") is None]
        xs = [it["timing"]["wallClockMs"] for it in items]
        tiers = {}
        for it in items:
            tiers.setdefault(it["detail"]["tier"], []).append(it["timing"]["wallClockMs"])
        tier_means = {k: st.mean(v) for k, v in tiers.items()}
        slow = max(tier_means.values())
        tL = st.mean(xs)
        means.append(tL)
        print(f"{os.path.basename(f)}: n={len(xs)} mean {tL:.0f} median {st.median(xs):.0f} ms; "
              f"tier means { {k: round(v) for k, v in tier_means.items()} }")
        print(f"  FR-2  s_max typical (tL/2-td)/tL = {(tL / 2 - T_D) / tL:.3f}; "
              f"slowest tier (tL/2-td)/{slow:.0f} = {(tL / 2 - T_D) / slow:.3f}")
        print(f"  FR-10 s_max typical (350-td)/tL = {(350 - T_D) / tL:.3f}; "
              f"slowest tier (350-td)/{slow:.0f} = {(350 - T_D) / slow:.3f}")
    print(f"session-to-session change in mean: {means[0]:.0f} -> {means[-1]:.0f} ms "
          f"({100 * (means[0] - means[-1]) / means[0]:.0f}%)")

    # Worked FR-2 ratio — constructed, not measured: incumbent latencies resampled from the newest
    # stored run; candidate = T_D plus the incumbent latency on a random hand-off subset.
    base = sorted(json.load(open(runs[-1]))["items"], key=lambda it: it["itemId"])
    lat = [it["timing"]["wallClockMs"] for it in base]
    g = random.Random(11)
    inc = [g.choice(lat) for _ in range(400)]
    for s in (0.30, 0.45):
        hand = set(g.sample(range(400), round(s * 400)))
        cand = [T_D + (inc[i] if i in hand else 0.0) for i in range(400)]
        R, lo, hi = ratio_ci(inc, cand, SEED)
        print(f"worked FR-2, n=400, s={s:.2f}: R={R:.2f} 95% [{lo:.2f}, {hi:.2f}]")
