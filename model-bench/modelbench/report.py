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
conversation ids drawn from 12 scripts are all unique. What closes it is `_unit_ids` below, which
`_paired_rows` calls for every row it builds — the resolution has exactly one home and no
parameter reaches it.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

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

_POOLED_FOOTNOTE = (
    "_A count whose denominator is not the analysis unit prints **without an interval**: `-ml` "
    "§4.4's first mandatory consequence is *\"Never print a Wilson interval over a turn-pooled "
    "count\"*, because the turns of one conversation are not independent observations and the "
    "resulting interval is understated several-fold. The honest bound is a one-level cluster "
    "bootstrap over the conversations (`stats.cluster_bootstrap`, Rule 6), which needs the "
    "per-unit observations a stored aggregate does not carry — S2's runner does._"
)

#: `-ml` §7.2's fourth sentence says "the 12 **scripts**" for a conversation-unit pack: the sample
#: is the set of scripts, one conversation each. Everything else names its own unit.
_SAMPLE_NOUN = {"conversation": "scripts", "item": "items", "query": "queries"}

#: How much a basis is worth, weakest first. A comparison takes the weaker of its two arms, and the
#: degradation is one-directional: any arm whose determinism probe did not run and agree drops the
#: whole comparison, never the other way round (plan §5 test 12b, `-ml` §3.4 Rule 4).
_BASIS_STRENGTH = {"assumed": 0, "measured": 1, "by-construction": 2}


def _pp(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}"


def _sentence(clause: str) -> str:
    """A clause promoted to its own sentence. Not `str.capitalize`, which lower-cases the rest —
    it would print "at any holm step" in the one line whose job is to be quotable."""
    return clause[0].upper() + clause[1:] + "."


def _fp(run: RunResult, name: str, default: str = "") -> str:
    return str(run.fingerprint.get(name, default))


def _unit_ids(items: Sequence[ItemResult], pack: PackRef) -> list[str]:
    """Resolve each row's analysis-unit id from the pack. **No call site chooses this** (§3.3)."""
    index = pack.analysisUnitIndex
    return [item.pairingKey[index] for item in items]


class PairedRows(NamedTuple):
    """The paired intersection, plus the tally of everything it had to leave out (`-ml` §4.3).

    The paired `n` printed inside a verdict string shrinks honestly, so the statistic is never
    laundered — but a reader cannot see *that* it shrank, or which arm caused it, unless the
    excluded rows are counted and printed too (§4.3 rule 2, and its paired corollary's
    **`asymmetry`** count). This is also the only place a violated `H <= min(script length)` would
    surface at S1.
    """

    unit_ids: list[str]
    a_ok: list[bool]
    b_ok: list[bool]
    considered: int
    only_in_a: int
    only_in_b: int
    asymmetry_a: int
    asymmetry_b: int
    unscoreable_both: int


def _paired_rows(a: RunResult, b: RunResult, metric: str, pack: PackRef) -> PairedRows:
    """Items scored by *both* arms, in A's order (`-ml` §4.3's paired-n intersection).

    An item whose precondition was not met in either arm is dropped from the pair rather than
    counted as a failure — a precondition failure must never be laundered into the numerator. That
    is `-ml` §10's risk R2, rated **high**: a model that collapses early otherwise scores *better*
    on every conditional count downstream.

    **Which state a row is in is `ItemResult.scored_outcome`'s call, and nothing is inferred here**
    (review P3-1). This function once carried two defaults of its own — a missing scoreability
    declaration read as scoreable, a missing count read as a failure — which combined into a
    verdict of `+100.0 pp, p=0.002` against an arm holding no data whatsoever.
    """
    by_key = {item.pairingKey: item for item in b.items}
    a_units = _unit_ids(a.items, pack)
    unit_ids: list[str] = []
    a_ok: list[bool] = []
    b_ok: list[bool] = []
    only_in_a = asymmetry_a = asymmetry_b = unscoreable_both = 0
    for item, unit_id in zip(a.items, a_units):
        other = by_key.get(item.pairingKey)
        if other is None:
            only_in_a += 1
            continue
        a_outcome = item.scored_outcome(metric)
        b_outcome = other.scored_outcome(metric)
        if a_outcome is None or b_outcome is None:
            if b_outcome is None and a_outcome is not None:
                asymmetry_a += 1
            elif a_outcome is None and b_outcome is not None:
                asymmetry_b += 1
            else:
                unscoreable_both += 1
            continue
        unit_ids.append(unit_id)
        a_ok.append(a_outcome)
        b_ok.append(b_outcome)
    a_keys = {item.pairingKey for item in a.items}
    only_in_b = sum(1 for item in b.items if item.pairingKey not in a_keys)
    return PairedRows(
        unit_ids=unit_ids,
        a_ok=a_ok,
        b_ok=b_ok,
        considered=len(a_keys | {item.pairingKey for item in b.items}),
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        asymmetry_a=asymmetry_a,
        asymmetry_b=asymmetry_b,
        unscoreable_both=unscoreable_both,
    )


def _pairing_tally(rows: PairedRows, unit_plural: str, a_label: str, b_label: str) -> str:
    """§4.3 rule 2's `n/a` tally, printed beside the rate it shaped — always, including when it is
    all zeros, because otherwise a reader cannot tell a shrunken `n` from a full one."""
    return (
        f"- paired n: {len(rows.unit_ids)} of {rows.considered} {unit_plural} "
        f"(`asymmetry`: {rows.asymmetry_a} scoreable for {a_label} only, "
        f"{rows.asymmetry_b} scoreable for {b_label} only; "
        f"{rows.unscoreable_both} unscoreable in both; "
        f"{rows.only_in_a} present in {a_label} only, {rows.only_in_b} in {b_label} only) — §4.3"
    )


def resolving_power_line(rp: stats.ResolvingPower, pack: PackRef) -> str:
    """`-ml` §7.1's template, rendered. Four sentences, all mandatory.

    None of them is derivable from a bare `n` — which is the whole of gate B-1's fix: 48 turns, 48
    conversations and 12 scripts would otherwise print the same sentence and mean three different
    things. The exact string this produces for the tool-caller pack is `-ml` §7.2's, and the suite
    asserts against it.
    """
    unit_plural = f"{rp.unit_kind}s"
    sample_noun = _SAMPLE_NOUN.get(rp.unit_kind, unit_plural)
    if rp.mdd80 is None:
        # Below b_min(alpha_mdd) effective units no effect size attains the power, so the MDD
        # sentence would print a figure the instrument cannot deliver (M-ML-1). The floor sentence
        # is printed either way: it takes the other alpha and can still be attainable when the MDD
        # is not, and where it is not, `floor_clause` says so without quoting a >100 pp threshold.
        sentences = [
            stats.unattainable_clause(rp, unit_plural),
            _sentence(stats.floor_clause(rp)),
        ]
    else:
        sentences = [
            # The stem has one home, in `stats`, because M-ML-7's fix edits it and a second copy
            # here is a scheduled drift (review m-ML-8) — as `provenance`, `floor_clause` and
            # `unattainable_clause` already are.
            f"{stats.mdd_clause(rp, unit_plural)}.",
            _sentence(stats.floor_clause(rp)),
        ]
    # The power model is strict dominance, which is the most favourable case, so the figure is a
    # lower bound and must carry its label. Below n_eff = 20 the 2:1 discordance mix reaches 80%
    # power at NO effect size, and there the label stops being a caveat and starts being the
    # finding (`-ml` §7.1).
    if rp.mdd80 is None:
        # The label qualifies an MDD figure that is not printed; the replacement sentence above
        # already says power is zero at every difference.
        best_case = None
    else:
        best_case = (
            f"Best case — assumes the candidate wins every {rp.unit_kind} the models differ on"
        )
    if best_case is not None and rp.n_effective < 20:
        best_case += (
            f"; if it loses one for every two it wins, {rp.power:.0%} power is not reached at any "
            "effect size at this n."
        )
    elif best_case is not None:
        best_case += "."
    if best_case is not None:
        sentences.append(best_case)
    # `-ml` §4.5.1(ii) publishes this clause for the tool-caller pack, where the sample is a set of
    # written scripts. The claim is right for every unit kind; the noun is not, and `_SAMPLE_NOUN`
    # was already two lines up (n-4, m-ML-5).
    sentences.append(
        f"Inference is conditional on the {rp.n_units} {sample_noun} in {pack.label}; "
        f"generalization to unwritten {sample_noun} is not certified by any interval in this "
        "report."
    )
    return " ".join(sentences)


#: What replaces a verdict when the paired intersection is empty (review P3-1). The two ways in
#: are a scorer that emitted no data for the metric and an arm that could not score a single item;
#: neither is an outcome, and the tally printed under it says which one happened.
_NO_PAIRED_DATA = (
    "**No verdict: no paired data.** No {unit} is scoreable for `{metric}` in both arms, so there "
    "is no paired table, no interval and no verdict. An arm carrying no data for a metric is not "
    "an arm that failed every {unit} of it (`-ml` §4.3)."
)

#: Only the metric's own section has a tally under it; the headline repeats the refusal without
#: pointing at a table that is not beside it.
_NO_PAIRED_DATA_TALLY = " The tally below says where the rows went."


def _decision(v: stats.Verdict | None, step: stats.HolmStep) -> str:
    """What the family table says happened, so no reader has to re-derive it from a threshold."""
    if v is None:
        return "no verdict — no paired data"
    if not step.tested:
        return "not tested (Holm stops here)"
    if v.distinguishable:
        return "distinguishable"
    if v.floor_demoted:
        return "not distinguishable — below the observable floor"
    return "not distinguishable"


def _comparison_kind(a: RunResult, b: RunResult) -> str:
    """§3.7 — the report says which kind of comparison it is doing, never silently mixing them."""
    # Two labels, not one: the banner above says "same declared version, different bytes", and a
    # single "different pack version" line contradicts it in the same report (review m-2).
    if _fp(a, "packVersion") != _fp(b, "packVersion"):
        return "unpaired (different pack version)"
    if _fp(a, "packContentHash") != _fp(b, "packContentHash"):
        return "unpaired (same pack version, different content hash)"
    if a.sessionId is not None and a.sessionId == b.sessionId:
        return "paired, same session"
    return "paired, cross-session"


def _arm_label(run: RunResult) -> str:
    if run.armKind == "deterministic":
        return f"{run.modelKey} — reference arm (deterministic given pack version)"
    return run.modelKey


#: Why no verdict is computed, keyed by cause. One explanation for two causes let a one-arm
#: comparison assert a deterministic-arm reason that is untrue (review M-6).
_NO_VERDICT_REASON = {
    "too-few-arms": (
        "_None: fewer than two arms were selected, so there is nothing to compare. Check "
        "`--models` and `--session` against `model-bench models --tested`; a comparison needs two "
        "stored runs for this pack._"
    ),
    "both-deterministic": (
        "_None: no verdict is computed between two deterministic arms — a deterministic arm is "
        "reproducible from its pack version and arm parameters, so a difference between two of "
        "them is a pack change, not a finding (§3.4.1)._"
    ),
}


def _comparison_pair(
    runs: Sequence[RunResult],
) -> tuple[RunResult, RunResult] | str:
    """Two arms to compare, or the **reason** there are none (§3.4.1)."""
    if len(runs) < 2:
        return "too-few-arms"
    a, b = runs[0], runs[1]
    if a.armKind == "deterministic" and b.armKind == "deterministic":
        return "both-deterministic"
    return a, b


#: The `--negative-control` mode's banner (review P3-4). The mode puts **two copies of one stored
#: record** in the two arms, so `b = c = 0` is arithmetic, not a measurement, and the report it
#: writes is durable and filed next to real comparisons under a filename that differs only in its
#: sequence number. Without this it reads as a validated null — the one output a tool whose value
#: claim is *"it refuses to report a number it cannot stand behind"* cannot afford (`-ml` §9,
#: plan §3.9(5)). It is the first thing in the document because it changes how everything below
#: it is read.
_NEGATIVE_CONTROL_BANNER = (
    "> **NEGATIVE CONTROL (WIRING SMOKE CHECK)** — both arms are the *same stored record*, so "
    "`b = c = 0 by construction` and this comparison **cannot fail**. It proves the mode is "
    "wired; it says nothing about whether the harness is sound. The real negative control is two "
    "**independent** runs of the same model and is an acceptance step, not this (`-ml` §9, "
    "plan §5 test 19a)."
)


def compare_report(
    runs: Sequence[RunResult],
    *,
    pack: PackRef,
    invalid: Sequence[InvalidRecord] = (),
    negative_control: bool = False,
) -> str:
    """Render the markdown comparison for one pack. Never ranks across roles or packs."""
    check_sampling_contract(pack)
    if pack.metrics.headlineMetric is not None and (
        pack.metrics.headlineMetric not in pack.metrics.verdictMetrics
    ):
        raise PackConfigError("headlineMetric is not a member of verdictMetrics")

    lines: list[str] = [f"# Comparison — {pack.label} ({pack.role})", ""]
    if negative_control:
        lines += [_NEGATIVE_CONTROL_BANNER, ""]

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
    pooled_seen = False
    for run in runs:
        for metric in run.aggregates.named_metrics():
            if isinstance(metric, BinaryMetric) and metric.n:
                if metric.unit == unit_kind_for_role(pack.role):
                    lo, hi = stats.wilson_interval(metric.successes, metric.n)
                    interval = f"[{lo:.3f}, {hi:.3f}]"
                else:
                    # `-ml` §4.4: a turn- or call-pooled count's observations are not independent,
                    # so a Wilson interval over them is fiction — measured at 4.5x too narrow on a
                    # representative funnel count (review M-ML-3). The count itself is never
                    # suppressed; only the precision claim is.
                    pooled_seen = True
                    interval = f"— (n is {metric.unit}s; not the analysis unit)"
                lines.append(
                    f"| {_arm_label(run)} | {metric.name} | {metric.successes}/{metric.n} | "
                    f"{metric.successes / metric.n:.3f} | {interval} |"
                )
            elif not isinstance(metric, BinaryMetric):
                lines.append(
                    f"| {_arm_label(run)} | {metric.name} | n={metric.n} | {metric.mean:.4f} | — |"
                )
    lines += ["", _DESCRIPTIVE_NOTE, ""]
    if pooled_seen:
        lines += [_POOLED_FOOTNOTE, ""]

    pair = _comparison_pair(runs)
    if isinstance(pair, str):
        lines += ["## Verdicts", "", _NO_VERDICT_REASON[pair], ""]
        return "\n".join(lines) + "\n"

    a, b = pair
    lines += ["## Verdicts", "", f"Comparison kind: **{_comparison_kind(a, b)}** (§3.7).", ""]

    family = list(pack.metrics.verdictMetrics)
    unit_kind = unit_kind_for_role(pack.role)
    # A basis is only as strong as its weakest arm, and the degradation is fail-safe: any arm whose
    # determinism probe did not run and agree drops the whole comparison to "assumed", which via
    # `-ml` §3.4 Rule 4 moves the decision off McNemar (plan §5 test 12b).
    design_effect = max(a.designEffect, b.designEffect)
    # The **weaker of the two actual bases**, not a collapse to `"assumed"`. The decision rule is
    # unchanged either way — only `by-construction` lets McNemar decide — but printing `assumed`
    # for two genuinely *measured* design effects is false provenance in the one sentence whose
    # entire job is auditability (`-ml` §7.1, review m-ML-4).
    basis = min((a.basis, b.basis), key=_BASIS_STRENGTH.__getitem__)

    # Two passes, because Holm is a property of the **family**: the step a metric is tested at
    # depends on every other member's p-value, so no verdict can be decided until all of them
    # exist. The delivered build ran one pass, decided every metric at the plain Bonferroni
    # `resolving.alpha`, and then printed a Holm table beside verdicts that had not used it —
    # `stats.verdict`'s `alpha_step` was built for exactly this and was passed by nothing (B-1).
    tables: list[tuple[str, stats.PairedOutcomes, stats.ResolvingPower | None]] = []
    p_values: list[float] = []
    tallies: list[str] = []
    for metric in family:
        rows = _paired_rows(a, b, metric, pack)
        outcomes = stats.PairedOutcomes.from_units(
            unit_kind, list(zip(rows.unit_ids, rows.a_ok, rows.b_ok))
        )
        tallies.append(_pairing_tally(rows, f"{unit_kind}s", a.modelKey, b.modelKey))
        # An empty intersection has no resolving power to describe — `n_effective` of zero is not a
        # small sample, it is no sample — so the metric gets no verdict rather than a figure
        # computed from nothing (review P3-1). It stays in the family: *k* is fixed by
        # pre-registration, not by how much data arrived.
        rp = (
            stats.resolving_power(
                outcomes.n_units,
                unit_kind=unit_kind,
                design_effect=design_effect,
                basis=basis,
                # The two αs come from the pack's pre-registered family, which is the only thing
                # that fixes *k*. They are different numbers whenever k > 1, and each bound takes
                # the one that keeps its own sentence true (`-ml` v1.6 §7.1, review M-ML-6).
                alpha_family=pack.metrics.alpha_family,
                alpha_mdd=pack.metrics.alpha_mdd,
            )
            if outcomes.n_units
            else None
        )
        tables.append((metric, outcomes, rp))
        _a, table_b, table_c, _d = outcomes.table
        p_values.append(stats.mcnemar_exact(table_b, table_c))

    steps = stats.holm_steps(p_values, alpha=pack.metrics.alpha_family)

    computed: list[tuple[str, stats.Verdict | None, stats.HolmStep]] = []
    # `strict=True`: a Holm ladder shorter than the family would otherwise truncate the loop and
    # a pre-registered verdict metric would vanish from the report — indistinguishable, to a
    # reader, from one that was never pre-registered (review P2-3).
    for (metric, outcomes, rp), step, tally in zip(tables, steps, tallies, strict=True):
        if rp is None:
            computed.append((metric, None, step))
            lines += [
                f"### {metric}",
                "",
                _NO_PAIRED_DATA.format(unit=unit_kind, metric=metric) + _NO_PAIRED_DATA_TALLY,
                "",
                tally,
                "",
            ]
            continue
        v = stats.verdict(
            outcomes,
            resolving=rp,
            metric_name=metric,
            family=family,
            a_label=a.modelKey,
            b_label=b.modelKey,
            alpha_step=step.threshold,
            holm_tested=step.tested,
            # The pack's declaration, never a literal here: `sampling.seed` is a manifest field
            # (§3.3) and a second copy in the renderer is a second home for the one number that
            # makes a bootstrap-decided verdict reproducible (review P3-5).
            bootstrap_seed=pack.seed,
        )
        computed.append((metric, v, step))
        lines += [
            f"### {metric}",
            "",
            v.text,
            "",
            tally,
            f"- marginal Wilson intervals overlap: {'yes' if v.marginal_overlap else 'no'}",
            # The seed is named only where a resample actually decided: on `mcnemar-exact` no
            # bootstrap ran, and quoting a seed there would claim a reproducibility that is not at
            # issue (review P3-5).
            (
                f"- decided by: {v.decided_by} (seed {pack.seed}, from the pack's "
                "`sampling.seed`)"
                if v.decided_by == "cluster-bootstrap"
                else f"- decided by: {v.decided_by}"
            ),
            "",
            resolving_power_line(rp, pack),
            "",
        ]

    if len(family) > 1:
        # Two co-equal verdicts at alpha=0.05 each carry a ~9.75% chance of at least one false
        # "better" under the null, which is the fishing artefact pre-registration exists to prevent
        # (§3.3, `-ml` §3.3). Family-wise control is mandatory, not optional.
        #
        # The `decision` column is not decoration: a threshold alone is only interpretable under a
        # step-down the table does not show, so a reader comparing p against it can reach the
        # opposite conclusion from the verdict three paragraphs above (B-1, M-ML-2).
        lines += [
            "### Family-wise error control",
            "",
            f"Holm–Bonferroni across the {len(family)} pre-registered verdict metrics, applied: "
            f"the smallest p is tested at alpha/{len(family)}, the next at "
            f"alpha/{len(family) - 1}, and the first non-rejection stops the procedure. Every "
            f"**MDD** above is computed at the family-adjusted alpha="
            f"{pack.metrics.alpha_mdd:g}; every **observable floor** is computed at the unadjusted "
            f"alpha={pack.metrics.alpha_family:g}, the loosest step a member can face, because "
            "that is the only alpha at which the floor's own sentence is true (§7.1).",
            "",
            "| metric | McNemar p | Holm-adjusted threshold | decision |",
            "|---|---|---|---|",
        ]
        for metric, v, step in computed:
            # A member with no paired table has no p-value to print. `mcnemar_exact(0, 0)` returns
            # 1.0 and would render as `1.000`, which reads as a test that was run and found nothing
            # — so the cell says what actually happened instead (review P3-1).
            p_cell = "—" if v is None else f"{v.mcnemar_p:.3f}"
            lines.append(
                f"| {metric} | {p_cell} | {step.threshold:.4f} | {_decision(v, step)} |"
            )
        lines.append("")

    # --- presentation: a headline exists only if the pack declared one ---------------------------
    if pack.metrics.headlineMetric is not None:
        headline = next(v for m, v, _ in computed if m == pack.metrics.headlineMetric)
        headline_text = (
            _NO_PAIRED_DATA.format(unit=unit_kind, metric=pack.metrics.headlineMetric)
            if headline is None
            else headline.text
        )
        lines += [f"**Headline ({pack.metrics.headlineMetric}):** {headline_text}", ""]
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
