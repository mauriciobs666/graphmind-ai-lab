"""Pack identity as the report sees it, and the two S1-reachable halves of the pack contract.

Design: `docs/plans/small-model-benchmarking.md` §3.3 and Appendix A. **S1 builds no pack loader**
— that is S2. What S1 needs, and what lives here, is the *reference* a report is handed
(`PackRef`), the pre-registered metric family (`PackMetrics`), and the two rules a report must not
be able to violate whatever a manifest says:

* the **`metrics` block** (§3.3) — `verdictMetrics` non-empty, the `headlineMetric` *key* present
  (omission is not the same statement as `null`, and only the latter is a decision), and a non-null
  headline that is a member of the family;
* the **`sampling` contract's structural half** (§3.3) — `analysisUnit == pairingKey[0]`, because
  the analysis unit is fixed by rule as the outermost component of the pairing key, never chosen by
  a call site.

The `sampling` contract's *second* route — the row-count identity over the data file — needs the
pack's data and belongs to S2's `validate_pack` (§3.3, §5 test 12).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from modelbench.stats import ALPHA_FAMILY


class PackConfigError(ValueError):
    """A manifest that cannot be honestly reported from. Raised, never warned."""


class PackMetrics(NamedTuple):
    """The pre-registered verdict family (§3.3).

    `verdictMetrics` controls **inference** — its length *is* the multiplicity correction's `k`.
    `headlineMetric` controls **presentation** — what a reader is entitled to read as "the" number,
    and it may legitimately be `None` (`guard-judge` has two co-equal errors and no headline).
    """

    verdictMetrics: tuple[str, ...]
    headlineMetric: str | None

    @property
    def k(self) -> int:
        return len(self.verdictMetrics)

    @property
    def alpha_family(self) -> float:
        """The **unadjusted** α — the *floor's*, whatever *k* is (`-ml` v1.6 §7.1, M-ML-6).

        The floor asserts *"below this, nothing can reach significance at any observed outcome"*,
        which is true only at the loosest Holm step a member can face. Computed at α/k it is `7/n`
        and the sentence is false: at n=40 a rank-2 member with b=6, c=0 reaches p=0.031 and clears
        its own 0.05 step, 15.0 pp below the 17.5 pp such a floor would print.
        """
        return ALPHA_FAMILY

    @property
    def alpha_mdd(self) -> float:
        """The **family-adjusted** α, `α/k` — the *MDD's* (§3.3). Not optional, and not the floor's.

        The MDD promises power whatever rank the member draws under Holm, so it takes the tightest
        step. This is also the *pre-registration* α that `stats.verdict`'s third precondition pins.
        """
        return self.alpha_family / self.k


class PackRef(NamedTuple):
    """A pack's identity plus what a report must print and resolve (Appendix A, extended).

    Beyond Appendix A's five fields this carries the `sampling` contract's two declarations
    (§3.3), because `report.py` resolves the analysis-unit id from `analysisUnit` and **no call
    site chooses it**. Appendix A predates §3.3's v1.4 `sampling` block; without these two fields
    the resolution has nowhere to come from.
    """

    packId: str
    packVersion: str
    #: `None` until S2's `load_pack` computes it. `""` was indistinguishable from a hash that
    #: failed to compute, in a field whose whole job is identity (review m-5). Nothing reads it at
    #: S1: the AC-3 banner reads each run's own recorded `fingerprint.packContentHash`, which is
    #: the right source at this stage.
    contentHash: str | None
    role: str
    metrics: PackMetrics
    pairingKey: tuple[str, ...]
    analysisUnit: str

    @property
    def analysisUnitIndex(self) -> int:
        """Where the analysis-unit id sits in an `ItemResult.pairingKey`. Always 0, by the rule."""
        return self.pairingKey.index(self.analysisUnit)

    @property
    def label(self) -> str:
        return f"{self.packId}@{self.packVersion}"


def metrics_from_manifest(block: Mapping[str, Any]) -> PackMetrics:
    """Parse and enforce §3.3's `metrics` block. Raises `PackConfigError`, never repairs."""
    verdicts = block.get("verdictMetrics")
    if not verdicts:
        raise PackConfigError("metrics.verdictMetrics is absent or empty")
    if "headlineMetric" not in block:
        raise PackConfigError(
            "metrics.headlineMetric key is absent; omission is not the same statement as null, "
            "and only null is a decision (plan §3.3)"
        )
    headline = block["headlineMetric"]
    if headline is not None and headline not in verdicts:
        raise PackConfigError(
            f"metrics.headlineMetric {headline!r} is not a member of verdictMetrics "
            f"{tuple(verdicts)!r}"
        )
    return PackMetrics(verdictMetrics=tuple(verdicts), headlineMetric=headline)


def check_sampling_contract(ref: PackRef) -> None:
    """§3.3's structural route: the analysis unit is `pairingKey[0]`, outermost → innermost."""
    if not ref.pairingKey:
        raise PackConfigError("sampling.pairingKey is empty")
    if ref.analysisUnit != ref.pairingKey[0]:
        raise PackConfigError(
            f"sampling.analysisUnit {ref.analysisUnit!r} is not pairingKey[0] "
            f"{ref.pairingKey[0]!r}; the analysis unit is the outermost component of the "
            "pairing key, by rule (plan §3.3)"
        )


def pack_ref_from_manifest(path: Path | str) -> PackRef:
    """Read `pack.json` into a `PackRef`.

    **This is a manifest read, not S2's pack loader.** It does no content hashing, no AST import
    walk, no data-file row-count identity check and no provenance check — those are `load_pack` /
    `validate_pack` (plan §3.6a, S2). It exists because S1 ships `compare`, and a comparison cannot
    resolve its analysis unit or its verdict family without the manifest's `sampling` and `metrics`
    blocks. `contentHash` is `None` here — "not loaded", not "empty" (review m-5/P2-5): at S1 the
    authoritative hash for a comparison is the one each run recorded in its own fingerprint, which
    is what the AC-3 banner reads.
    """
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    sampling = manifest.get("sampling") or {}
    pairing_key = tuple(sampling.get("pairingKey") or ())
    if not pairing_key:
        raise PackConfigError(f"{path}: sampling.pairingKey is absent or empty")
    if "analysisUnit" not in sampling:
        raise PackConfigError(f"{path}: sampling.analysisUnit is absent")
    ref = PackRef(
        packId=manifest["packId"],
        packVersion=manifest["packVersion"],
        contentHash=None,
        role=manifest["role"],
        metrics=metrics_from_manifest(manifest.get("metrics") or {}),
        pairingKey=pairing_key,
        analysisUnit=sampling["analysisUnit"],
    )
    check_sampling_contract(ref)
    return ref
