"""The markdown comparison — where AC-2, AC-3 and AC-4 become visible output.

Design: `docs/plans/small-model-benchmarking.md` §3.4.3, §3.5, §3.7 and §3.9; the statistics and
every string they carry are `docs/plans/small-model-benchmarking-ml.md`'s (§3.2e's three verdicts,
§7.1's resolving-power template, §7.2's rendered line).

Three refusals here are **missing functions, not guarded ones** (§3.5) — a rule you cannot express
is a rule you cannot break under deadline pressure:

* there is no code path that synthesises a headline when `headlineMetric` is `null`;
* there is no path that pools a per-class table into one accuracy figure, because
  `ClassificationAggregates` has no field to hold one;
* there is no parameter through which a caller could choose the analysis unit — it is resolved from
  `PackRef.analysisUnit`, which §3.3 fixes by rule as `pairingKey[0]`.

That last one is the whole of gate finding N-1. `PairedOutcomes.from_units` raising on a repeated
unit id is a **backstop**: it fires only if the id handed to it is the *cluster* key, and 48
conversation ids drawn from 12 scripts are all unique. What closes it is `_unit_ids` below.
"""

from __future__ import annotations

from typing import Sequence

from modelbench import stats
from modelbench.packs import PackConfigError, PackRef, check_sampling_contract
from modelbench.results import BinaryMetric, InvalidRecord, ItemResult, RunResult
from modelbench.roles import unit_kind as unit_kind_for_role

_DESCRIPTIVE_NOTE = (
    "_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not "
    "the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._"
)

_OVERLAP_FOOTNOTE = (
    "_The marginal-overlap line is a **diagnostic**, never the verdict: at this lab's sample sizes "
    "two marginal intervals overlapping is a far stronger condition than their difference covering "
    "zero, and the literal rule cannot fire at all at n<=40 with a baseline >=0.90 (`-ml` §3.1)._"
)

#: `-ml` §7.2's fourth sentence says "the 12 **scripts**" for a conversation-unit pack: the sample
#: is the set of scripts, one conversation each. Everything else names its own unit.
_SAMPLE_NOUN = {"conversation": "scripts", "item": "items", "query": "queries"}


def _pp(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}"


def _fp(run: RunResult, name: str, default: str = "") -> str:
    return str(run.fingerprint.get(name, default))


def _unit_ids(items: Sequence[ItemResult], pack: PackRef) -> list[str]:
    """Resolve each row's analysis-unit id from the pack. **No call site chooses this** (§3.3)."""
    index = pack.analysisUnitIndex
    return [item.pairingKey[index] for item in items]


def _paired_rows(
    a: RunResult, b: RunResult, metric: str, pack: PackRef
) -> tuple[list[str], list[bool], list[bool]]:
    """Items scored by *both* arms, in A's order (`-ml` §4.3's paired-n intersection).

    An item whose precondition was not met in either arm is dropped from the pair rather than
    counted as a failure — a precondition failure must never be laundered into the numerator.
    """
    by_key = {item.pairingKey: item for item in b.items}
    unit_ids: list[str] = []
    a_ok: list[bool] = []
    b_ok: list[bool] = []
    index = pack.analysisUnitIndex
    for item in a.items:
        other = by_key.get(item.pairingKey)
        if other is None:
            continue
        if not item.scoreable.get(metric, True) or not other.scoreable.get(metric, True):
            continue
        unit_ids.append(item.pairingKey[index])
        a_ok.append(item.counts.get(metric, 0) > 0)
        b_ok.append(other.counts.get(metric, 0) > 0)
    return unit_ids, a_ok, b_ok


def resolving_power_line(rp: stats.ResolvingPower, pack: PackRef) -> str:
    """`-ml` §7.1's template, rendered. Four sentences, all mandatory.

    None of them is derivable from a bare `n` — which is the whole of gate B-1's fix: 48 turns, 48
    conversations and 12 scripts would otherwise print the same sentence and mean three different
    things. The exact string this produces for the tool-caller pack is `-ml` §7.2's, and the suite
    asserts against it.
    """
    unit_plural = f"{rp.unit_kind}s"
    sample_noun = _SAMPLE_NOUN.get(rp.unit_kind, unit_plural)
    sentences = [
        (
            f"This pack resolves differences of >={_pp(rp.mdd80)} pp with 80% power at "
            f"n={rp.n_effective:g} effective {unit_plural} ({rp.n_units} units, design effect "
            f"{rp.design_effect:.2f}, {rp.basis}, alpha={rp.alpha:g})."
        ),
        (
            f"Differences below {_pp(rp.observable_floor)} pp cannot reach significance at any "
            "observed outcome."
        ),
    ]
    # The power model is strict dominance, which is the most favourable case, so the figure is a
    # lower bound and must carry its label. Below n_eff = 20 the 2:1 discordance mix reaches 80%
    # power at NO effect size, and there the label stops being a caveat and starts being the
    # finding (`-ml` §7.1).
    best_case = f"Best case — assumes the candidate wins every {rp.unit_kind} the models differ on"
    if rp.n_effective < 20:
        best_case += (
            "; if it loses one for every two it wins, 80% power is not reached at any effect size "
            "at this n."
        )
    else:
        best_case += "."
    sentences.append(best_case)
    sentences.append(
        f"Inference is conditional on the {rp.n_units} {sample_noun} in {pack.label}; "
        "generalization to unwritten scripts is not certified by any interval in this report."
    )
    return " ".join(sentences)


def _comparison_kind(a: RunResult, b: RunResult) -> str:
    """§3.7 — the report says which kind of comparison it is doing, never silently mixing them."""
    if _fp(a, "packVersion") != _fp(b, "packVersion") or _fp(a, "packContentHash") != _fp(
        b, "packContentHash"
    ):
        return "unpaired (different pack version)"
    if a.sessionId is not None and a.sessionId == b.sessionId:
        return "paired, same session"
    return "paired, cross-session"


def _arm_label(run: RunResult) -> str:
    if run.armKind == "deterministic":
        return f"{run.modelKey} — reference arm (deterministic given pack version)"
    return run.modelKey


def _comparison_pair(runs: Sequence[RunResult]) -> tuple[RunResult, RunResult] | None:
    """Two arms to compare, or `None`. Two deterministic arms are never ranked (§3.4.1)."""
    if len(runs) < 2:
        return None
    a, b = runs[0], runs[1]
    if a.armKind == "deterministic" and b.armKind == "deterministic":
        return None
    return a, b


def compare_report(
    runs: Sequence[RunResult],
    *,
    pack: PackRef,
    invalid: Sequence[InvalidRecord] = (),
) -> str:
    """Render the markdown comparison for one pack. Never ranks across roles or packs."""
    check_sampling_contract(pack)
    if pack.metrics.headlineMetric is not None and (
        pack.metrics.headlineMetric not in pack.metrics.verdictMetrics
    ):
        raise PackConfigError("headlineMetric is not a member of verdictMetrics")

    lines: list[str] = [f"# Comparison — {pack.label} ({pack.role})", ""]

    # --- banners: never silent, and never a reason to drop a record -----------------------------
    versions = {_fp(r, "packVersion") for r in runs}
    hashes = {_fp(r, "packContentHash") for r in runs}
    if len(versions) > 1:
        lines += [
            "> **PACK VERSION MISMATCH** — these runs span pack versions "
            + ", ".join(sorted(versions))
            + ". "
            "They are not measuring the same thing; the comparison below is rendered anyway and is "
            "labelled unpaired (AC-3).",
            "",
        ]
    if len(hashes) > 1:
        lines += [
            "> **PACK CONTENT HASH MISMATCH** — same declared version, different bytes: "
            + ", ".join(sorted(h[:8] for h in hashes))
            + ". A declared version can be forgotten; a hash cannot (§3.3).",
            "",
        ]
    schemas = sorted({r.fingerprint.benchSchemaVersion for r in runs})
    if len(schemas) > 1:
        lines += [
            "> **SCHEMA VERSIONS IN THIS COMPARISON** — "
            + ", ".join(str(s) for s in schemas)
            + ". Each record was validated against the contract it was written under; a schema "
            "difference is visible, never silent, and never a reason to drop a record (§3.4.3).",
            "",
        ]
    if invalid:
        lines += ["> **INVALID RESULTS EXCLUDED** (AC-2)", ">"]
        for record in invalid:
            detail = (
                ", ".join(f"`{p.field}` ({p.reason})" for p in record.problems) or record.reason
            )
            lines.append(f"> - `{record.runId or record.path.name}` — {record.reason}: {detail}")
        lines.append("")

    # --- per-arm descriptive table --------------------------------------------------------------
    lines += ["## Arms", "", "| arm | metric | k/n | rate | 95% Wilson |", "|---|---|---|---|---|"]
    for run in runs:
        for metric in run.aggregates.named_metrics():
            if isinstance(metric, BinaryMetric) and metric.n:
                lo, hi = stats.wilson_interval(metric.successes, metric.n)
                lines.append(
                    f"| {_arm_label(run)} | {metric.name} | {metric.successes}/{metric.n} | "
                    f"{metric.successes / metric.n:.3f} | [{lo:.3f}, {hi:.3f}] |"
                )
            elif not isinstance(metric, BinaryMetric):
                lines.append(
                    f"| {_arm_label(run)} | {metric.name} | n={metric.n} | {metric.mean:.4f} | — |"
                )
    lines += ["", _DESCRIPTIVE_NOTE, ""]

    pair = _comparison_pair(runs)
    if pair is None:
        lines += [
            "## Verdicts",
            "",
            "_None: no verdict is computed between two deterministic arms — a deterministic arm is "
            "reproducible from its pack version and arm parameters, so a difference between two of "
            "them is a pack change, not a finding (§3.4.1)._",
            "",
        ]
        return "\n".join(lines) + "\n"

    a, b = pair
    lines += ["## Verdicts", "", f"Comparison kind: **{_comparison_kind(a, b)}** (§3.7).", ""]

    family = list(pack.metrics.verdictMetrics)
    alpha = 0.05 / len(family)
    unit_kind = unit_kind_for_role(pack.role)
    # A basis is only as strong as its weakest arm, and the degradation is fail-safe: any arm whose
    # determinism probe did not run and agree drops the whole comparison to "assumed", which via
    # `-ml` §3.4 Rule 4 moves the decision off McNemar (plan §5 test 12b).
    design_effect = max(a.designEffect, b.designEffect)
    basis = "by-construction" if a.basis == b.basis == "by-construction" else "assumed"

    computed: list[tuple[str, stats.Verdict]] = []
    p_values: list[float] = []
    for metric in family:
        unit_ids, a_ok, b_ok = _paired_rows(a, b, metric, pack)
        outcomes = stats.PairedOutcomes.from_units(unit_kind, list(zip(unit_ids, a_ok, b_ok)))
        rp = stats.resolving_power(
            outcomes.n_units,
            unit_kind=unit_kind,
            design_effect=design_effect,
            basis=basis,
            alpha=alpha,
        )
        v = stats.verdict(
            outcomes,
            resolving=rp,
            metric_name=metric,
            family=family,
            a_label=a.modelKey,
            b_label=b.modelKey,
            bootstrap_seed=20260902,
        )
        computed.append((metric, v))
        p_values.append(v.mcnemar_p)
        lines += [
            f"### {metric}",
            "",
            v.text,
            "",
            f"- marginal Wilson intervals overlap: {'yes' if v.marginal_overlap else 'no'}",
            f"- decided by: {v.decided_by}",
            "",
            resolving_power_line(rp, pack),
            "",
        ]

    if len(family) > 1:
        # Two co-equal verdicts at alpha=0.05 each carry a ~9.75% chance of at least one false
        # "better" under the null, which is the fishing artefact pre-registration exists to prevent
        # (§3.3, `-ml` §3.3). Family-wise control is mandatory, not optional.
        steps = stats.holm_thresholds(p_values, alpha=0.05)
        lines += [
            "### Family-wise error control",
            "",
            f"Holm–Bonferroni across the {len(family)} pre-registered verdict metrics; every "
            f"figure above is computed at alpha={alpha:g}.",
            "",
            "| metric | McNemar p | Holm-adjusted threshold |",
            "|---|---|---|",
        ]
        for (metric, v), step in zip(computed, steps):
            lines.append(f"| {metric} | {v.mcnemar_p:.3f} | {step:.4f} |")
        lines.append("")

    # --- presentation: a headline exists only if the pack declared one ---------------------------
    if pack.metrics.headlineMetric is not None:
        headline = next(v for m, v in computed if m == pack.metrics.headlineMetric)
        lines += [f"**Headline ({headline.metric_name}):** {headline.text}", ""]
    else:
        # No summary line above the verdicts, and no arithmetic combining them (§3.3(i)). The
        # metrics stand side by side in the manifest's declared order, which is how they were
        # rendered above.
        lines += [
            "_This pack declares no headline metric: its verdict metrics are co-equal and are "
            "printed side by side, in the manifest's declared order, with no summary line above "
            "them and no arithmetic combining them (§3.3)._",
            "",
        ]

    exploratory = [
        m
        for run in (a, b)
        for m in run.aggregates.named_metrics()
        if m.name not in family
    ]
    if exploratory:
        lines += ["### Exploratory metrics", ""]
        seen: set[str] = set()
        for metric in exploratory:
            if metric.name in seen:
                continue
            seen.add(metric.name)
            lines.append(f"- `{metric.name}` — exploratory — no significance claim")
        lines.append("")

    lines += [_OVERLAP_FOOTNOTE, ""]
    return "\n".join(lines) + "\n"
