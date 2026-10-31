"""Figures for docs/plans/typed-decision-models-ml.md §Q3 (FR-1 parity) and §Q6 (replicates).

stdlib only. Run from the repo root:  python3 -I docs/plans/typed-decision-models-ml/parity.py
"""
from functools import lru_cache
from math import comb

ALPHA = 0.05  # one-sided, per cell, unadjusted (§Q3.1)


@lru_cache(None)
def crit(m, alpha=ALPHA):
    """Smallest b with P(Bin(m, 1/2) >= b) <= alpha; None if no such b (m < 5 at 0.05)."""
    tail = 0.0
    for b in range(m, -1, -1):
        tail += comb(m, b) * 0.5**m
        if tail > alpha + 1e-15:
            return b + 1 if b + 1 <= m else None
    return 0


def p_worse(bw, cw):
    """One-sided exact McNemar p for 'candidate worse' (§Q3.1)."""
    m = bw + cw
    if m == 0:
        return 1.0
    return sum(comb(m, i) for i in range(bw, m + 1)) * 0.5**m


def power(n, delta, psi, alpha=ALPHA):
    """Exact trinomial power of the one-sided test (§Q3.3). psi = per-item discordance."""
    if psi <= 0:
        return 0.0
    q = (psi + delta) / (2 * psi)
    tot = 0.0
    for m in range(n + 1):
        pm = comb(n, m) * psi**m * (1 - psi) ** (n - m)
        c = crit(m, alpha)
        if c is None or pm < 1e-300:
            continue
        tot += pm * sum(comb(m, b) * q**b * (1 - q) ** (m - b) for b in range(c, m + 1))
    return tot


def mdd(n, psi_hat, target=0.80, alpha=ALPHA, step=0.001):
    """First 0.1 pp grid point with power(delta, max(psi_hat, delta)) >= target; None if unattainable."""
    for k in range(1, 1001):
        d = k * step
        if power(n, d, max(psi_hat, d), alpha) >= target:
            return d
    return None


def pp(x):
    return "unattainable" if x is None else f"{100 * x:.1f}"


if __name__ == "__main__":
    print("crit(m), one-sided 0.05:", [(m, crit(m)) for m in range(1, 13)])
    print("MDD unattainable below n=5:", [(n, pp(mdd(n, 0.0))) for n in range(3, 7)])

    print("\n§Q3.3 MDD80 (pp) by n per cell and psi_hat")
    psis = [0.0, 0.05, 0.10, 0.20, 0.30]
    print("n    " + "  ".join(f"psi={p:.2f}" for p in psis))
    for n in [50, 75, 100, 125, 150]:
        print(f"{n:<4} " + "  ".join(f"{pp(mdd(n, p)):>8}" for p in psis))

    print("\n§Q3.5 reuse-adjusted MDD (power target 0.8^(1/(K+1))), K=0..3")
    for n, p in [(50, 0.10), (100, 0.10), (50, 0.0), (50, 0.20), (100, 0.20)]:
        print(f"n={n} psi={p:.2f}: " + " -> ".join(pp(mdd(n, p, target=0.8 ** (1 / (K + 1)))) for K in range(4)))
    print("§Q3.5 miss bound 1-0.8^(K+1), K=0..4:", [round(1 - 0.8 ** (K + 1), 3) for K in range(5)])

    print("\n§Q3.2 / §Q6 false-fail of an equal candidate, four independent cells of 50")
    for psi in [0.05, 0.10, 0.20]:
        cell = power(50, 0.0, psi)
        one = 1 - (1 - cell) ** 4
        two = 1 - (1 - one) ** 2
        print(f"psi={psi:.2f}: per cell {cell:.4f}; one run {100 * one:.2f}%; "
              f"two independent replicates (upper bound) {100 * two:.2f}%")

    print("\n§Q3.6 worked cells (n=50 per cell)")
    for bw, cw in [(4, 1), (6, 2), (5, 0), (7, 1)]:
        ph = (bw + cw) / 50
        print(f"bw={bw} cw={cw} diff={100 * (bw - cw) / 50:+.1f}pp p_worse={p_worse(bw, cw):.4f} "
              f"psi_hat={ph:.2f} MDD80={pp(mdd(50, ph))}pp")
    print(f"same discordance at n=100: bw=8 cw=2 p_worse={p_worse(8, 2):.4f} "
          f"MDD80={pp(mdd(100, 0.10))}pp")
