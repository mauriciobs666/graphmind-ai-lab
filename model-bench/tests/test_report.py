"""§5 test 5 — `compare_report`, where AC-2/AC-3/AC-4 become visible output.

Every fixture is a hand-built `RunResult`; no LM Studio, no network, no pack on disk (plan §4 S1).
"""

from __future__ import annotations

import json

import pytest
from conftest import (
    BinaryMetric,
    PackMetrics,
    PackRef,
    ToolCallAggregates,
    classification_aggregates,
    deterministic_fields,
    guard_pack,
    item,
    model_fields,
    run,
)

from modelbench import stats
from modelbench.fingerprint import FieldProblem
from modelbench.packs import PackConfigError, metrics_from_manifest
from modelbench.report import compare_report, resolving_power_line
from modelbench.results import InvalidRecord, ItemResult
from modelbench.stats import DuplicateAnalysisUnit, PairedOutcomes

METRIC = "falseAdvanceRate"
PACK_ID = "guard-judge-understanding"


def _arm(run_id: str, correct: int, total: int = 40, **kwargs):
    """One arm: `correct` of `total` items pass `METRIC`, with stable item ids for pairing."""
    items = [item(f"g{i:02d}", correct=i < correct, metric=METRIC) for i in range(total)]
    kwargs.setdefault(
        "fingerprint_fields", model_fields(modelKey=run_id, packId=PACK_ID)
    )
    return run(run_id, items=items, aggregates=classification_aggregates(correct, total), **kwargs)


def _nested_arms(total: int = 40, a_correct: int = 40, b_correct: int = 34):
    """`-ml` §3.1's worked case: perfectly nested, so b = 6 and c = 0."""
    a_items = [item(f"g{i:02d}", correct=i < a_correct, metric=METRIC) for i in range(total)]
    b_items = [item(f"g{i:02d}", correct=i < b_correct, metric=METRIC) for i in range(total)]
    a = run("cand", items=a_items, aggregates=classification_aggregates(a_correct, total),
            fingerprint_fields=model_fields(modelKey="cand", packId=PACK_ID))
    b = run("incumbent", items=b_items, aggregates=classification_aggregates(b_correct, total),
            fingerprint_fields=model_fields(modelKey="incumbent", packId=PACK_ID))
    return [a, b]


# --- AC-3: the pack version / content hash banner -------------------------------------------------


def test_a_pack_version_mismatch_is_bannered_and_the_comparison_still_renders() -> None:
    a, b = _nested_arms()
    b = run(
        "incumbent",
        items=list(b.items),
        aggregates=b.aggregates,
        fingerprint_fields=model_fields(
            modelKey="incumbent", packId=PACK_ID, packVersion="0.9.0"
        ),
    )
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "PACK VERSION MISMATCH" in md
    assert "0.9.0" in md and "1.0.0" in md
    assert METRIC in md


def test_an_identical_version_with_a_different_content_hash_is_also_bannered() -> None:
    """AC-3's nastier case: same declared version, different bytes. A hash cannot be forgotten."""
    a, b = _nested_arms()
    b = run(
        "incumbent",
        items=list(b.items),
        aggregates=b.aggregates,
        fingerprint_fields=model_fields(
            modelKey="incumbent", packId=PACK_ID, packContentHash="f" * 64
        ),
    )
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "PACK CONTENT HASH MISMATCH" in md
    assert METRIC in md


def test_matching_pack_identity_produces_no_banner() -> None:
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "MISMATCH" not in md


# --- AC-2: the excluded-invalid block -------------------------------------------------------------


def test_invalid_records_are_named_with_their_problems(tmp_path) -> None:
    invalid = [
        InvalidRecord(
            path=tmp_path / "runs" / "bad.json",
            runId="bad",
            benchSchemaVersion=1,
            problems=[FieldProblem(field="kvCacheSetting", reason="empty")],
            reason="field",
        )
    ]
    md = compare_report(
        _nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)), invalid=invalid
    )
    assert "INVALID RESULTS EXCLUDED" in md
    assert "bad" in md
    assert "kvCacheSetting" in md
    assert "empty" in md


def test_no_invalid_block_when_every_record_is_valid() -> None:
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "INVALID RESULTS EXCLUDED" not in md


# --- §3.4.3: the schema-versions line -------------------------------------------------------------


def test_records_spanning_schema_versions_are_named_never_dropped(monkeypatch) -> None:
    from modelbench.fingerprint import REQUIRED_BY_SCHEMA

    monkeypatch.setitem(REQUIRED_BY_SCHEMA, 2, REQUIRED_BY_SCHEMA[1])
    a, b = _nested_arms()
    b = run(
        "incumbent",
        items=list(b.items),
        aggregates=b.aggregates,
        fingerprint_fields=model_fields(
            modelKey="incumbent", packId=PACK_ID, benchSchemaVersion=2
        ),
    )
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "SCHEMA VERSIONS IN THIS COMPARISON" in md
    assert "1" in md and "2" in md
    assert METRIC in md


def test_a_single_schema_version_prints_no_such_line() -> None:
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "SCHEMA VERSIONS IN THIS COMPARISON" not in md


# --- AC-4: the decision wording -------------------------------------------------------------------


def test_the_forty_of_forty_case_is_distinguishable() -> None:
    """S1 done-condition 3 / §5 test 6 — the case the *old* marginal-overlap rule got backwards.

    40/40 vs 34/40, perfectly nested: the candidate strictly dominates and the paired difference
    excludes zero, so the correct verdict is *distinguishable* (`-ml` §3.1/§3.2).
    """
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "not distinguishable at this sample size" not in md.lower()
    assert "is better than" in md


def test_a_difference_interval_covering_zero_renders_the_ac4_wording() -> None:
    """`-ml` §3.2c row 3: (33, 6, 1, 0) — +12.5 pp, CI [-1.0, 26.9] pp, p = 0.125."""
    total = 40
    a_items = [item(f"g{i:02d}", correct=i < 39, metric=METRIC) for i in range(total)]
    b_items = [item(f"g{i:02d}", correct=(i < 33 or i == 39), metric=METRIC) for i in range(total)]
    a = run("cand", items=a_items, aggregates=classification_aggregates(39, total),
            fingerprint_fields=model_fields(modelKey="cand", packId=PACK_ID))
    b = run("incumbent", items=b_items, aggregates=classification_aggregates(34, total),
            fingerprint_fields=model_fields(modelKey="incumbent", packId=PACK_ID))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "Not distinguishable at this sample size." in md
    assert "covers zero" in md


def test_the_marginal_overlap_diagnostic_is_printed_and_labelled() -> None:
    """FR-15's literal rule is honoured visibly, and labelled never-the-verdict (`-ml` §3.2)."""
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "marginal Wilson intervals overlap" in md
    assert "diagnostic" in md.lower()


# --- S1 done-condition 5b: the resolving-power line, verbatim -------------------------------------


def test_the_tool_caller_resolving_power_line_is_the_notes_verbatim_string() -> None:
    """S1 done-condition 5b — `-ml` §7.2 has this string, parameterised only by pack@version.

    A line missing the unit, the design effect, the best-case caveat or the conditionality clause
    fails, whatever number it prints. This is the acceptance surface for plan §3.9 point 2.

    **The floor sentence carries `at any Holm step (alpha <= A_family)`** — v1.6 §7.1's template,
    which names *both* αs because they differ whenever k > 1 and a reader shown one cannot tell
    which bound it governs (review M-ML-6). This pack is k=1, so both print 0.05 and only the
    wording moves. §7.2's rendered example still shows the pre-v1.6 sentence; §7.1's template is
    the one v1.6 changed, and it is what this asserts.
    """
    pack = PackRef(
        packId="tool-caller-shop-assistant",
        packVersion="1.0.0",
        contentHash="e" * 64,
        role="tool-caller",
        metrics=PackMetrics(
            verdictMetrics=("cleanThroughTurn4",), headlineMetric="cleanThroughTurn4"
        ),
        pairingKey=("scriptId", "replicate", "turnIndex"),
        analysisUnit="scriptId",
        seed=20260902,
    )
    metric = "cleanThroughTurn4"
    a_items = [
        item(f"S-{i:02d}", correct=i < 12, metric=metric, pairing=(f"S-{i:02d}", "0", "0"))
        for i in range(12)
    ]
    b_items = [
        item(f"S-{i:02d}", correct=i < 6, metric=metric, pairing=(f"S-{i:02d}", "0", "0"))
        for i in range(12)
    ]
    fields = model_fields(packId="tool-caller-shop-assistant")
    a = run("cand", role="tool-caller", items=a_items,
            aggregates=classification_aggregates(12, 12, metric, unit="conversation"),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", role="tool-caller", items=b_items,
            aggregates=classification_aggregates(6, 12, metric, unit="conversation"),
            fingerprint_fields={**fields, "modelKey": "incumbent"})

    md = compare_report([a, b], pack=pack)

    assert (
        "This pack resolves differences of >=57.8 pp with 80% power at n=12 effective "
        "conversations (12 units, design effect 1.00, by-construction, alpha=0.05). Differences "
        "below 50.0 pp cannot reach significance at any observed outcome, at any Holm step "
        "(alpha <= 0.05). Best case — assumes the "
        "candidate wins every conversation the models differ on; if it loses one for every two it "
        "wins, 80% power is not reached at any effect size at this n. Inference is conditional on "
        "the 12 scripts in tool-caller-shop-assistant@1.0.0; generalization to unwritten scripts "
        "is not certified by any interval in this report."
    ) in md


def test_the_mdd_sentence_stem_has_exactly_one_home() -> None:
    """Review m-ML-8 — `stats._mdd_clause` and `report.resolving_power_line` each spelled the stem
    out in full. `provenance`, `floor_clause` and `unattainable_clause` were made public precisely
    so the report could not carry a second copy, and this one was left behind; M-ML-7's fix edits
    exactly this string, so the drift was scheduled rather than hypothetical.

    Asserted as *the report renders the stats module's string*, not as two literals that happen to
    match today.
    """
    rp = stats.resolving_power(
        40, unit_kind="item", design_effect=1.0, basis="by-construction",
        alpha_family=0.05, alpha_mdd=0.05,
    )
    line = resolving_power_line(rp, guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert stats.mdd_clause(rp, "items") + "." in line


def test_the_power_ceiling_sentence_is_dropped_above_n_eff_twenty() -> None:
    """`-ml` §7.1 — the 2:1 row prints only where the caveat becomes the finding (n_eff < 20)."""
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "Best case — assumes the candidate wins every item the models differ on." in md
    assert "80% power is not reached at any effect size at this n" not in md


# --- S1 done-condition 5(c): WHICH key is used as the unit id -------------------------------------


def _clustered_fixture():
    """12 clusters x 4 rows = 48, each row's pairingKey a **unique** (scriptId, replicate) pair.

    S1 has no pack loader, so this is an in-memory `Sequence[ItemResult]` plus a `PackRef`
    (plan S1 done-condition 5(c)).
    """
    metric = "cleanThroughTurn4"
    pack = PackRef(
        packId="tool-caller-clustered",
        packVersion="1.0.0",
        contentHash="e" * 64,
        role="tool-caller",
        metrics=PackMetrics(verdictMetrics=(metric,), headlineMetric=metric),
        pairingKey=("scriptId", "replicate"),
        analysisUnit="scriptId",
        seed=20260902,
    )
    items = [
        item(
            f"S-{s:02d}-{r}",
            correct=(s + r) % 2 == 0,
            metric=metric,
            pairing=(f"S-{s:02d}", str(r)),
        )
        for s in range(12)
        for r in range(4)
    ]
    fields = model_fields(packId="tool-caller-clustered")
    arms = [
        run(name, role="tool-caller", items=items,
            aggregates=classification_aggregates(24, 48, metric, unit="conversation"),
            fingerprint_fields={**fields, "modelKey": name})
        for name in ("cand", "incumbent")
    ]
    return pack, arms, items


def test_the_analysis_unit_id_is_the_cluster_key_and_the_guard_therefore_fires(monkeypatch) -> None:
    """S1 done-condition 5(c), all three assertions (gate finding N-1).

    Asserting only (2) would pass while testing nothing: 48 conversation ids are unique, so the
    *wrong* unit-id choice raises nothing and the fixture goes green on a harness that silently
    produces an anti-conservative verdict. Assertions (1) and (3) are what make it real.
    """
    pack, arms, items = _clustered_fixture()
    original = PairedOutcomes.from_units
    captured: list[list[str]] = []

    def spy(unit_kind, rows):
        materialized = list(rows)
        captured.append([row[0] for row in materialized])
        return original(unit_kind, materialized)

    monkeypatch.setattr(PairedOutcomes, "from_units", spy)

    # (2) Consequence: the repeated unit id makes the paired table unconstructible.
    with pytest.raises(DuplicateAnalysisUnit):
        compare_report(arms, pack=pack)

    # (1) Identity: the argument actually passed is the rows' scriptId values — the OUTERMOST
    # component of pairingKey, resolved from PackRef.analysisUnit. Asserted on the captured
    # argument itself, not on a property inferred from the outcome.
    assert captured, "compare_report never reached PairedOutcomes.from_units"
    passed_unit_ids = captured[0]
    assert passed_unit_ids == [f"S-{s:02d}" for s in range(12) for _ in range(4)]
    assert len(passed_unit_ids) == 48
    assert len(set(passed_unit_ids)) == 12
    assert all(passed_unit_ids.count(u) == 4 for u in set(passed_unit_ids))
    # and it is emphatically NOT the per-conversation id
    assert passed_unit_ids != [it.itemId for it in items]

    # (3) Negative control on the guard itself: the 48 unique conversation ids are ACCEPTED, which
    # proves `from_units` alone does not close this and that assertion (1) is what does.
    conversation_rows = [(f"{it.pairingKey[0]}-{it.pairingKey[1]}", True, False) for it in items]
    accepted = original("conversation", conversation_rows)
    assert accepted.n_units == 48
    assert len(set(accepted.unit_ids)) == 48


def test_no_caller_can_choose_the_analysis_unit() -> None:
    """Plan §3.3 — 'no call site chooses it, and there is no parameter through which one could'.

    Asserted as a **closed** parameter set rather than as the absence of a `unit_kind` name: a
    whitelist fails when *any* new knob appears, which is the only form that catches an analysis
    unit arriving under a name nobody thought to forbid. Growing it is therefore a deliberate act.
    `negative_control` joined it in review P3-4 — it selects a banner, and reads nothing about the
    data.
    """
    import inspect

    params = inspect.signature(compare_report).parameters
    assert set(params) == {"runs", "pack", "invalid", "negative_control"}


# --- S1 done-condition 8: headlineMetric null -----------------------------------------------------


def test_a_null_headline_renders_both_verdicts_and_no_headline() -> None:
    """Plan §3.3(i) — there is no code path that synthesises a headline from `verdictMetrics`."""
    second = "falseSuspendRate"
    total = 40
    a_items = [
        item(f"g{i:02d}", correct=i < 40, metric=METRIC, pairing=(f"g{i:02d}",))
        for i in range(total)
    ]
    b_items = [
        item(f"g{i:02d}", correct=i < 34, metric=METRIC, pairing=(f"g{i:02d}",))
        for i in range(total)
    ]
    for i, it in enumerate(a_items):
        a_items[i] = it.__class__(**{**vars(it), "counts": {METRIC: it.counts[METRIC], second: 1},
                                     "scoreable": {METRIC: True, second: True}})
    for i, it in enumerate(b_items):
        b_items[i] = it.__class__(**{**vars(it), "counts": {METRIC: it.counts[METRIC], second: 1},
                                     "scoreable": {METRIC: True, second: True}})
    fields = model_fields(packId=PACK_ID)
    a = run("cand", items=a_items, aggregates=classification_aggregates(40, total),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=b_items, aggregates=classification_aggregates(34, total),
            fingerprint_fields={**fields, "modelKey": "incumbent"})

    pack = guard_pack(headline=None, verdicts=(METRIC, second))
    md = compare_report([a, b], pack=pack)

    assert METRIC in md and second in md
    # No synthesised headline line, and no summary above the two co-equal verdicts.
    assert "**Headline" not in md
    assert "declares no headline metric" in md
    assert "Holm" in md  # k = 2 makes family-wise error control mandatory (plan §3.3(ii))
    # Review P3-6 — a bare `"alpha=0.025" in md` was satisfied by the **family-wise paragraph**
    # ("computed at the family-adjusted alpha=0.025"), not by the MDD sentence it was placed to
    # guard, so `provenance` printing `alpha_family` where it must print `alpha_mdd` survived the
    # whole suite. The two αs differ only at k>1, and this is the suite's only k=2 α assertion.
    # Asserting the **whole parenthetical** is what binds the number to the bound it governs.
    assert "design effect 1.00, by-construction, alpha=0.025)" in md
    # ...and the floor, in the same report, takes the *other* α — the unadjusted one (M-ML-6).
    assert "at any Holm step (alpha <= 0.05)" in md


def test_a_manifest_omitting_the_headline_key_fails_validation() -> None:
    """Plan §3.3 / S1 done-condition 8 — omission is not the same statement as `null`."""
    with pytest.raises(PackConfigError):
        metrics_from_manifest({"verdictMetrics": ["a"]})
    explicit_null = metrics_from_manifest({"verdictMetrics": ["a"], "headlineMetric": None})
    assert explicit_null.headlineMetric is None


def test_an_empty_verdict_family_fails_validation() -> None:
    with pytest.raises(PackConfigError):
        metrics_from_manifest({"verdictMetrics": [], "headlineMetric": None})


def test_a_headline_outside_the_family_fails_validation() -> None:
    with pytest.raises(PackConfigError):
        metrics_from_manifest({"verdictMetrics": ["a"], "headlineMetric": "b"})


def test_a_metric_outside_the_verdict_family_is_labelled_exploratory() -> None:
    """Plan §3.3 — everything not pre-registered prints `exploratory — no significance claim`."""
    a, b = _nested_arms()
    pack = guard_pack(headline=METRIC, verdicts=(METRIC,))
    a = run("cand", items=list(a.items),
            aggregates=classification_aggregates(40, 40, METRIC).__class__(
                perClass=(*classification_aggregates(40, 40, METRIC).perClass,
                          classification_aggregates(31, 40, "sideMetric").perClass[0]),
                parseFailures=0, n=40),
            fingerprint_fields=model_fields(modelKey="cand", packId=PACK_ID))
    md = compare_report([a, b], pack=pack)
    # Review P3-7 — the two assertions this replaces were `"sideMetric" in md` (also true from the
    # Arms table) and `"exploratory — no significance claim" in md` (true of whichever metric got
    # listed). They asserted the presence of two strings in a document, not the **pairing** between
    # them, so inverting the filter to `m.name in family` — which labels the *pre-registered
    # verdict metrics* "exploratory" and hides the genuinely exploratory ones — left the suite
    # green. The rendered line whole, plus the negative, is what pins the requirement.
    assert "- `sideMetric` — exploratory — no significance claim" in md
    assert f"- `{METRIC}` — exploratory" not in md
    # ...and the pre-registered metric still gets its verdict, which the inversion also removes.
    assert f"### {METRIC}" in md


# --- S1 done-condition 6: the deterministic arm ---------------------------------------------------


def test_a_deterministic_arm_renders_beside_a_model_arm() -> None:
    a = _arm("cand", 34)
    bm25 = run(
        "bm25",
        arm_kind="deterministic",
        items=[item(f"g{i:02d}", correct=i < 20, metric=METRIC) for i in range(40)],
        aggregates=classification_aggregates(20, 40),
        fingerprint_fields=deterministic_fields(packId=PACK_ID),
    )
    md = compare_report([a, bm25], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "reference arm (deterministic given pack version)" in md
    assert "bm25" in md


def test_two_deterministic_arms_are_never_ranked_against_each_other() -> None:
    """Plan §3.4.1 — 'two deterministic arms are **never** the subject of a verdict'."""
    arms = [
        run(
            name,
            arm_kind="deterministic",
            items=[item(f"g{i:02d}", correct=i < n, metric=METRIC) for i in range(40)],
            aggregates=classification_aggregates(n, 40),
            fingerprint_fields=deterministic_fields(
                packId=PACK_ID, armId=name, armParametersHash="b" * 64
            ),
        )
        for name, n in (("bm25", 20), ("bm25-tuned", 30))
    ]
    md = compare_report(arms, pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "is better than" not in md
    assert "no verdict is computed between two deterministic arms" in md


# --- §3.7: which kind of comparison this is -------------------------------------------------------


def test_a_shared_session_is_reported_as_paired_same_session() -> None:
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "paired, same session" in md


def test_different_sessions_are_reported_as_cross_session() -> None:
    a, b = _nested_arms()
    b = run("incumbent", session_id="s2", items=list(b.items), aggregates=b.aggregates,
            fingerprint_fields=model_fields(modelKey="incumbent", packId=PACK_ID))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "paired, cross-session" in md


def test_a_pack_version_difference_makes_the_comparison_unpaired() -> None:
    a, b = _nested_arms()
    b = run("incumbent", items=list(b.items), aggregates=b.aggregates,
            fingerprint_fields=model_fields(
                modelKey="incumbent", packId=PACK_ID, packVersion="0.9.0"))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "unpaired (different pack version)" in md


# --- structural refusals --------------------------------------------------------------------------


def test_a_pack_ref_whose_analysis_unit_is_not_the_outermost_key_is_refused() -> None:
    """Plan §3.3's structural route, applied fail-closed at render time."""
    pack = PackRef(
        packId="p", packVersion="1.0.0", contentHash="e" * 64, role="tool-caller",
        metrics=PackMetrics(verdictMetrics=("m",), headlineMetric="m"),
        pairingKey=("scriptId", "replicate"), analysisUnit="replicate", seed=20260902,
    )
    with pytest.raises(PackConfigError):
        compare_report(_nested_arms(), pack=pack)


def test_the_per_arm_intervals_are_labelled_descriptive() -> None:
    """`-ml` §3.2a — 'explicitly not the comparison instrument, and the report must say so'."""
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "descriptive, not the comparison instrument" in md


def test_every_rate_prints_with_its_denominator() -> None:
    """`-ml` §3.2a — 'never a bare percentage, never without its denominator'."""
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "40/40" in md and "34/40" in md


# --- B-1 / M-ML-2: Holm-Bonferroni is applied, and the table says what it decided -----------------


def _two_metric_arms(a_wins_first: int, a_wins_second: int, total: int = 40):
    """Two co-equal verdict metrics over the same 40 paired items, perfectly nested per metric.

    `a_wins_*` is the number of items arm A gets right and arm B does not, so the paired table is
    `b = a_wins_*, c = 0` — the `guard-judge` shape review B-1 reproduced at k = 2.
    """
    second = "falseSuspendRate"

    def items(first_correct: int, second_correct: int):
        built = []
        for i in range(total):
            built.append(
                item(f"g{i:02d}", correct=True, metric=METRIC).__class__(
                    itemId=f"g{i:02d}",
                    pairingKey=(f"g{i:02d}",),
                    outcome="pass",
                    scoreable={METRIC: True, second: True},
                    counts={
                        METRIC: 1 if i < first_correct else 0,
                        second: 1 if i < second_correct else 0,
                    },
                    latencyMs=1300.0,
                    detail={},
                )
            )
        return built

    fields = model_fields(packId=PACK_ID)
    a = run("cand", items=items(total, total), aggregates=classification_aggregates(total, total),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=items(total - a_wins_first, total - a_wins_second),
            aggregates=classification_aggregates(total - a_wins_first, total),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    return [a, b], guard_pack(headline=None, verdicts=(METRIC, second))


def test_the_family_table_and_the_verdicts_agree_on_every_row() -> None:
    """Review B-1, as rendered output rather than as an assertion about a return value.

    The delivered report printed *"does not reach alpha=0.025 (p=0.031)"* for `falseSuspendRate`
    and, two paragraphs below, its Holm-adjusted threshold as `0.0500` — so a reader applying the
    printed rule concluded it had cleared its step while the verdict said it had not. Whatever the
    decision is, the table has to state it rather than leave the reader to derive it from a
    threshold that is only valid under a step-down the report never showed.

    **Under v1.6 the agreed answer is the other one** (review M-ML-6). `falseSuspendRate` is
    b=6, c=0 → p=0.031 at a rank-2 Holm step of 0.05, and its 15.0 pp is no longer below a floor
    printed at α/k: the fix-round build agreed with the threshold by *demoting* the metric against
    a 17.5 pp floor, which reduced Holm to Bonferroni for the whole `[6/n, 7/n)` band and printed a
    sentence that a p=0.031 outcome had just falsified. The table and the prose still have to
    agree; they now agree on **distinguishable**.
    """
    arms, pack = _two_metric_arms(a_wins_first=8, a_wins_second=6)
    md = compare_report(arms, pack=pack)

    assert "| metric | McNemar p | Holm-adjusted threshold | decision |" in md
    rows = [ln for ln in md.splitlines() if ln.startswith(f"| {METRIC} |")]
    assert rows and "distinguishable" in rows[0] and "not distinguishable" not in rows[0]
    rows = [ln for ln in md.splitlines() if ln.startswith("| falseSuspendRate |")]
    assert rows and "0.0500" in rows[0]
    assert "not distinguishable" not in rows[0]
    # ...and the prose above it must say the same thing
    section = md.split("### falseSuspendRate")[1].split("###")[0]
    assert "is better than" in section
    assert "below 15.0 pp cannot reach significance at any observed outcome, at any Holm step " \
        "(alpha <= 0.05)" in section


def test_a_floor_demotion_is_named_in_both_the_prose_and_the_decision_column() -> None:
    """Rule 7's demote-and-name path, as rendered output rather than as a `Verdict` field.

    It is reachable only on the substitute path now (m-ML-6 raises on the McNemar one), so the
    fixture declares a measured design effect: at DEFF = 2 on 40 items the floor moves to 30.0 pp
    while a 20.0 pp difference still has an interval excluding zero. A reader must be able to see
    *why* the metric was not ranked, in the table and in the prose, without deriving it.
    """
    arms, pack = _two_metric_arms(a_wins_first=8, a_wins_second=6)
    arms = [
        run(r.runId, items=list(r.items), aggregates=r.aggregates, design_effect=2.0,
            basis="measured",
            fingerprint_fields=model_fields(modelKey=r.modelKey, packId=PACK_ID))
        for r in arms
    ]
    md = compare_report(arms, pack=pack)

    rows = [ln for ln in md.splitlines() if ln.startswith(f"| {METRIC} |")]
    assert rows and "not distinguishable — below the observable floor" in rows[0]
    section = md.split(f"### {METRIC}")[1].split("###")[0]
    assert "is below this pack's observable floor" in section
    assert "differences below 30.0 pp cannot reach significance" in section
    assert "is better than" not in section


def test_the_report_refuses_a_short_holm_ladder_rather_than_dropping_a_metric(monkeypatch) -> None:
    """P2-3's consumer half — `zip(tables, steps, tallies)` truncated to the shortest.

    The ladder is the public API S2 wires against, and a metric silently vanishing from a report is
    the one failure this component must not have: a pre-registered verdict metric that is not
    printed is indistinguishable, to a reader, from one that was never pre-registered. `strict=True`
    turns it into a `ValueError` at the point of truncation (3.12 has it).
    """
    arms, pack = _two_metric_arms(a_wins_first=8, a_wins_second=6)
    real = stats.holm_steps
    monkeypatch.setattr(
        stats, "holm_steps", lambda p_values, *, alpha: real(p_values, alpha=alpha)[:-1]
    )
    with pytest.raises(ValueError):
        compare_report(arms, pack=pack)


def test_holm_is_applied_and_not_merely_printed() -> None:
    """`stats.verdict`'s `alpha_step` existed for exactly this and was passed by nothing (B-1).

    `falseSuspendRate` is b=8, c=1 -> p = 0.039: above the plain Bonferroni alpha/k = 0.025 every
    metric was decided at, below its own Holm step of 0.05, and its 17.5 pp is above the 15.0 pp
    unadjusted floor so Rule 7 does not demote it. The step-down is therefore the only thing that
    can make it distinguishable, and the mutation `alpha = resolving.alpha_mdd` is visible here.
    `falseAdvanceRate` is b=8, c=0 -> p = 0.008, which clears the alpha/2 step so Holm does not
    stop before reaching the second metric.
    """
    second = "falseSuspendRate"
    fields = model_fields(packId=PACK_ID)

    def items(is_a: bool):
        built = []
        for i in range(40):
            # falseAdvanceRate: A right everywhere, B wrong on 0..7        -> b=8,  c=0
            first_ok = True if is_a else i >= 8
            # falseSuspendRate: A wrong only on 38, B wrong on 0..7        -> b=8,  c=1
            second_ok = (i != 38) if is_a else (i >= 8)
            built.append(
                ItemResult(
                    itemId=f"g{i:02d}", pairingKey=(f"g{i:02d}",), outcome="pass",
                    scoreable={METRIC: True, second: True},
                    counts={METRIC: int(first_ok), second: int(second_ok)},
                    latencyMs=1300.0, detail={},
                )
            )
        return built

    a = run("cand", items=items(True), aggregates=classification_aggregates(40, 40),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=items(False), aggregates=classification_aggregates(32, 40),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=guard_pack(headline=None, verdicts=(METRIC, second)))
    section = md.split(f"### {second}")[1].split("### ")[0]
    assert "(b=8, c=1)" in section
    assert "is better than" in section
    assert "p=0.039" in section
    row = next(ln for ln in md.splitlines() if ln.startswith(f"| {second} |"))
    assert "0.0500" in row and "| distinguishable |" in row


def test_a_metric_past_the_holm_stop_is_rendered_as_not_tested() -> None:
    """§3.3 — 'stopping at the first non-rejection'; the remainder is marked, never rejected.

    Both metrics land at p = 0.125 (b=6, c=1). The smaller fails its α/2 = 0.025 step, so Holm
    stops and the second is not tested at all — the delivered `holm_thresholds` printed it a 0.05
    threshold with no way for a reader to know it was unusable.
    """
    second = "falseSuspendRate"
    fields = model_fields(packId=PACK_ID)

    def items(a_side: bool):
        built = []
        for i in range(40):
            # b = 6, c = 1 on both metrics
            a_ok = i >= 1 if a_side else (i >= 7 or i == 0)
            built.append(
                ItemResult(
                    itemId=f"g{i:02d}", pairingKey=(f"g{i:02d}",), outcome="pass",
                    scoreable={METRIC: True, second: True},
                    counts={METRIC: int(a_ok), second: int(a_ok)},
                    latencyMs=1300.0, detail={},
                )
            )
        return built

    a = run("cand", items=items(True), aggregates=classification_aggregates(39, 40),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=items(False), aggregates=classification_aggregates(34, 40),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=guard_pack(headline=None, verdicts=(METRIC, second)))
    assert "not tested (Holm stops here)" in md
    assert "is better than" not in md


# --- M-ML-1: no MDD exists below b_min, and the line must not invent one --------------------------


def test_a_pack_below_b_min_says_no_difference_is_resolvable() -> None:
    """Review M-ML-1 — the line rendered *"resolves >=100.0 pp with 80% power"* at zero power.

    n_units = 40 at DEFF = 7 gives n_eff = 5.71, floored to 5, and b_min(0.05) = 6: the McNemar
    rejection region is empty, so `_mcnemar_power` is zero at every δ and the bisection converged
    on its upper bracket.
    """
    a, b = _nested_arms()
    a = run("cand", items=list(a.items), aggregates=a.aggregates, design_effect=7.0,
            basis="measured", fingerprint_fields=model_fields(modelKey="cand", packId=PACK_ID))
    b = run("incumbent", items=list(b.items), aggregates=b.aggregates, design_effect=7.0,
            basis="measured",
            fingerprint_fields=model_fields(modelKey="incumbent", packId=PACK_ID))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "with 80% power" not in md
    assert "100.0 pp" not in md
    assert "No difference is resolvable" in md
    assert "b_min=6" in md
    assert "is better than" not in md
    # The floor sentence is still printed, because it takes the **other** alpha and is a different
    # claim (M-ML-6): here it is unattainable too, and says so instead of quoting the 105.0 pp
    # threshold `6/5.71` would format to — a number no observed difference could ever exceed.
    assert (
        "No observed difference can reach significance at any Holm step (alpha <= 0.05): the "
        "floor of 6 net wins exceeds the 5.71429 effective units available."
    ) in md
    assert "105.0 pp" not in md
    # and the "best case" caveat goes with it: it qualifies an MDD figure that is not printed.
    # Found by reading the rendered line, not by an assertion about a return value.
    assert "Best case" not in md


# --- n-4 / m-ML-5: the conditionality clause names the pack's own sample noun ---------------------


def test_the_conditionality_clause_names_the_packs_own_sample_noun() -> None:
    """`-ml` §4.5.1(ii)'s clause was written for the conversation pack; the claim is right for all
    of them, the noun is not. An item-level pack read *"conditional on the 40 items ...
    generalization to unwritten scripts"*."""
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "conditional on the 40 items" in md
    assert "generalization to unwritten items" in md
    assert "unwritten scripts" not in md


# --- M-3 / m-ML-4: the fail-safe propagation is decision 4's whole justification -----------------


def test_a_weaker_basis_in_either_arm_moves_the_report_off_mcnemar() -> None:
    """Review M-3 — plan-review N-2's mechanism, and at report level nothing held it in place.

    Every report fixture used `design_effect=1.0, basis="by-construction"`, so the clustered branch
    of `verdict()` was exercised only through direct `stats.verdict` calls: forcing
    `basis = "by-construction"` unconditionally in `compare_report` left all 233 tests green.
    """
    a, b = _nested_arms()
    a = run("cand", items=list(a.items), aggregates=a.aggregates, basis="by-construction",
            fingerprint_fields=model_fields(modelKey="cand", packId=PACK_ID))
    b = run("incumbent", items=list(b.items), aggregates=b.aggregates, basis="assumed",
            fingerprint_fields=model_fields(modelKey="incumbent", packId=PACK_ID))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "decided by: cluster-bootstrap" in md
    assert "design effect 1.00, assumed" in md
    # The label says what the two instruments each did: the interval decides, McNemar vetoes
    # (review B-ML-2). At DEFF = 1.00 the widening is a no-op and the veto is the whole of the
    # path's conservatism, which is exactly the corner the blocker was about.
    assert "in conjunction with McNemar's exact test" in md
    assert "may withhold a verdict but never carries one on its own" in md


def test_the_design_effect_is_the_max_of_the_two_arms() -> None:
    """Review M-3 — forcing `design_effect = 1.0` in `compare_report` was green."""
    a, b = _nested_arms()
    a = run("cand", items=list(a.items), aggregates=a.aggregates, design_effect=1.0,
            basis="measured", fingerprint_fields=model_fields(modelKey="cand", packId=PACK_ID))
    b = run("incumbent", items=list(b.items), aggregates=b.aggregates, design_effect=2.0,
            basis="measured",
            fingerprint_fields=model_fields(modelKey="incumbent", packId=PACK_ID))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "design effect 2.00" in md
    assert "design effect 1.00" not in md
    assert "n=20 effective items (40 units" in md


def test_two_measured_bases_at_deff_one_still_do_not_let_mcnemar_decide() -> None:
    """P2-1's report mirror — the seam S2's runner will actually produce.

    `report.py` takes the weaker of the two *actual* bases (m-ML-4), so two measured arms print
    `measured`; Rule 4's branch condition is `by-construction`, so `measured` at a design effect of
    exactly 1.0 must still decide by the substitute. Widening the branch to admit `"measured"` was
    green across the whole delivered suite.
    """
    a, b = _nested_arms()
    a = run("cand", items=list(a.items), aggregates=a.aggregates, design_effect=1.0,
            basis="measured", fingerprint_fields=model_fields(modelKey="cand", packId=PACK_ID))
    b = run("incumbent", items=list(b.items), aggregates=b.aggregates, design_effect=1.0,
            basis="measured",
            fingerprint_fields=model_fields(modelKey="incumbent", packId=PACK_ID))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "design effect 1.00, measured" in md
    assert "decided by: cluster-bootstrap" in md
    assert "decided by: mcnemar-exact" not in md


def test_two_measured_bases_print_measured_not_assumed() -> None:
    """Review m-ML-4 — the decision rule is fail-safe and stays unchanged, but printing `assumed`
    for two genuinely **measured** design effects is false provenance in the one sentence whose
    entire job is auditability (`-ml` §7.1)."""
    a, b = _nested_arms()
    a = run("cand", items=list(a.items), aggregates=a.aggregates, design_effect=2.0,
            basis="measured", fingerprint_fields=model_fields(modelKey="cand", packId=PACK_ID))
    b = run("incumbent", items=list(b.items), aggregates=b.aggregates, design_effect=2.0,
            basis="measured",
            fingerprint_fields=model_fields(modelKey="incumbent", packId=PACK_ID))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "design effect 2.00, measured" in md
    assert "assumed" not in md
    # the decision is unchanged: `measured` is still not `by-construction`
    assert "decided by: cluster-bootstrap" in md


# --- M-5 / M-ML-4: the paired-n intersection, and the trace it must leave -----------------------


def test_a_precondition_failure_is_dropped_from_the_pair_and_printed() -> None:
    """`-ml` §4.3 risk R2 — 'a model that collapses early scores *better* on the conditional
    counts', rated **high**. Replacing the filter with `if False:` left all 233 tests green: no
    fixture in the suite ever set `scoreable=False`, though `conftest.item()` takes the parameter.

    §4.3 rule 2 also requires the excluded items counted in their own tally and printed, and §4.3's
    paired corollary the **`asymmetry`** count — items scoreable for exactly one model — as a
    finding about the arm that could not produce them. `grep -rn asymmetry modelbench/` returned
    nothing.
    """
    total = 40
    a_items = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(total)]
    b_items = [
        item(f"g{i:02d}", correct=i >= 6, metric=METRIC, scoreable=i >= 10)
        for i in range(total)
    ]
    fields = model_fields(packId=PACK_ID)
    a = run("cand", items=a_items, aggregates=classification_aggregates(40, total),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=b_items, aggregates=classification_aggregates(34, total),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    # Arm B could not score its first ten items, and those are exactly the six it got wrong plus
    # four it got right — so the intersection is 30 rows and the discordance vanishes with them.
    # That is R2 in miniature: laundering them in would have made the candidate look better.
    assert "n=30 effective items (30 units" in md
    assert "n=40 effective items" not in md
    assert "paired n: 30 of 40 items" in md
    assert "10 scoreable for cand only" in md
    assert "0 scoreable for incumbent only" in md
    assert "(b=0, c=0" in md


def test_the_paired_n_tally_is_printed_even_when_nothing_was_dropped() -> None:
    """A reader cannot see that `n` shrank unless the tally is there when it did not (M-ML-4)."""
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "paired n: 40 of 40 items" in md
    assert "asymmetry" in md


def test_an_item_present_in_only_one_arm_is_counted_and_named() -> None:
    a_items = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(40)]
    b_items = [item(f"g{i:02d}", correct=i >= 6, metric=METRIC) for i in range(36)]
    fields = model_fields(packId=PACK_ID)
    a = run("cand", items=a_items, aggregates=classification_aggregates(40, 40),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=b_items, aggregates=classification_aggregates(30, 36),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "paired n: 36 of 40 items" in md
    assert "4 present in cand only" in md


# --- P3-1: absence of data is not an outcome ----------------------------------------------------


def _bare(item_id: str, *, scoreable: dict, counts: dict) -> ItemResult:
    """An item whose `scoreable`/`counts` maps are exactly as given — including empty."""
    return ItemResult(
        itemId=item_id,
        pairingKey=(item_id,),
        outcome="pass",
        scoreable=scoreable,
        counts=counts,
        latencyMs=1300.0,
        detail={},
    )


def _pair_of_arms(a_items, b_items):
    fields = model_fields(packId=PACK_ID)
    a = run("cand", items=a_items, aggregates=classification_aggregates(len(a_items), len(a_items)),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=b_items, aggregates=classification_aggregates(0, len(b_items)),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    return a, b


def test_an_arm_carrying_no_data_for_a_metric_is_not_scored_as_failing_every_item() -> None:
    """Review P3-1 (blocker) — the two defaults in `_paired_rows` combined into a false positive.

    `item.scoreable.get(metric, True)` admitted an item that never mentions the metric, and
    `item.counts.get(metric, 0) > 0` then scored it a **loss**. An arm carrying no data at all
    rendered *"cand is better than incumbent … +100.0 pp … p=0.002"* while the §4.3 tally, whose
    entire job is to make dropped rows visible, printed `0 unscoreable in both`.

    A missing declaration is not a declaration: absence routes through the tally, never into the
    numerator's complement (`-ml` §4.3, risk R2).
    """
    a_items = [_bare(f"g{i:02d}", scoreable={METRIC: True}, counts={METRIC: 1}) for i in range(10)]
    b_items = [_bare(f"g{i:02d}", scoreable={}, counts={}) for i in range(10)]
    a, b = _pair_of_arms(a_items, b_items)
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    assert "is better than" not in md
    assert "+100.0 pp" not in md
    assert "paired n: 0 of 10 items" in md
    # Arm B declares nothing for the metric, so every row is an asymmetry — an undeclared metric
    # is exactly as unscoreable as a declared precondition failure, and lands in the same tally.
    assert "10 scoreable for cand only" in md
    assert "No verdict: no paired data" in md


def test_neither_arm_declaring_a_metric_is_unscoreable_in_both() -> None:
    """The other half of the same default: when *both* arms are silent the rows are neither arm's
    finding, and the tally must say so rather than crediting one of them."""
    a_items = [_bare(f"g{i:02d}", scoreable={}, counts={}) for i in range(10)]
    b_items = [_bare(f"g{i:02d}", scoreable={}, counts={}) for i in range(10)]
    a, b = _pair_of_arms(a_items, b_items)
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "10 unscoreable in both" in md
    assert "is better than" not in md


def test_a_metric_with_no_paired_rows_renders_a_refusal_rather_than_raising() -> None:
    """The legitimate case of the same shape: a candidate that collapses so completely that no
    item is scoreable for it. That is real data and a real finding, so it must render — and
    `resolving_power(0 units)` would raise `n_effective must be positive`, aborting the whole
    report including the tally that carries the finding."""
    a_items = [_bare(f"g{i:02d}", scoreable={METRIC: True}, counts={METRIC: 1}) for i in range(10)]
    b_items = [_bare(f"g{i:02d}", scoreable={METRIC: False}, counts={}) for i in range(10)]
    a, b = _pair_of_arms(a_items, b_items)
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    assert "No verdict: no paired data" in md
    assert "paired n: 0 of 10 items" in md
    assert "10 scoreable for cand only" in md
    assert "is better than" not in md
    # The headline may not be synthesised from a metric that has no verdict either.
    assert f"**Headline ({METRIC}):**" in md
    assert "no paired data" in md.split(f"**Headline ({METRIC}):**")[1]


def test_a_no_verdict_metric_is_named_as_such_in_the_family_table() -> None:
    """With k>1 the Holm table has a row per pre-registered member, and a member with no paired
    data must not print a p-value and a threshold as though a test had been run."""
    other = "unsafeAdvanceRate"
    a_items = [
        _bare(f"g{i:02d}", scoreable={METRIC: True, other: True}, counts={METRIC: 1, other: 1})
        for i in range(10)
    ]
    b_items = [
        _bare(f"g{i:02d}", scoreable={METRIC: True}, counts={METRIC: 0}) for i in range(10)
    ]
    a, b = _pair_of_arms(a_items, b_items)
    md = compare_report([a, b], pack=guard_pack(headline=None, verdicts=(METRIC, other)))
    row = [ln for ln in md.splitlines() if ln.startswith(f"| {other} |")]
    assert row == [f"| {other} | — | 0.0500 | no verdict — no paired data |"]


# --- M-6: fewer than two arms is its own reason, not the deterministic one ----------------------


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_arms_prints_its_own_reason(count: int) -> None:
    """Review M-6 — `_comparison_pair` returned `None` for both cases and the report printed one
    explanation for both, so a one-arm comparison asserted a deterministic-arm reason that is
    untrue. The route in is `--models` naming a key with no stored run."""
    arms = _nested_arms()[:count]
    md = compare_report(arms, pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "fewer than two arms" in md
    assert "two deterministic arms" not in md


# --- m-2: the unpaired label names what actually differed --------------------------------------


def test_a_content_hash_only_divergence_is_not_labelled_a_version_difference() -> None:
    """Review m-2 — one report, two adjacent lines contradicting each other: the banner said the
    declared versions matched and the comparison-kind line said they did not."""
    a, b = _nested_arms()
    b = run("incumbent", items=list(b.items), aggregates=b.aggregates,
            fingerprint_fields=model_fields(
                modelKey="incumbent", packId=PACK_ID, packContentHash="f" * 64))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "unpaired (same pack version, different content hash)" in md
    assert "unpaired (different pack version)" not in md


# --- m-5: `PackRef.contentHash` is not yet computable, and says so ------------------------------


def test_pack_ref_content_hash_is_none_until_s2_computes_it(tmp_path) -> None:
    """Review m-5 — `pack_ref_from_manifest` set it to `""` by design and nothing reads it: the
    AC-3 banner correctly reads each run's own `fingerprint.packContentHash`. A field that is
    always the empty string is a trap for the S2 author who fills it in and expects the report to
    use it, because `""` is indistinguishable from a hash that failed to compute.

    `None` makes "not yet computed" expressible. **This makes `PackRef.contentHash` a
    `str | None`, which plan Appendix A's identity triple does not yet allow for** — reported to
    `architect` rather than fixed here.
    """
    from modelbench.packs import pack_ref_from_manifest

    manifest = tmp_path / "pack.json"
    manifest.write_text(json.dumps({
        "packId": "p", "packVersion": "1.0.0", "role": "guard-judge",
        "sampling": {"pairingKey": ["itemId"], "analysisUnit": "itemId", "seed": 20260902},
        "metrics": {"verdictMetrics": ["m"], "headlineMetric": "m"},
    }))
    ref = pack_ref_from_manifest(manifest)
    assert ref.contentHash is None
    assert ref.label == "p@1.0.0"


# --- M-ML-3: a Wilson interval is printed only over the analysis unit ---------------------------


def _toolcall_arm(run_id: str):
    """`-ml` §4.3's real denominators: `cleanThroughTurn4` is per conversation, `restraint` is per
    turn, and the funnel counts are per turn or per call. Figures from review M-ML-3's table."""
    return run(
        run_id,
        role="tool-caller",
        items=[
            item(f"S-{i:02d}", correct=i < 9, metric="cleanThroughTurn4",
                 pairing=(f"S-{i:02d}", "0"))
            for i in range(12)
        ],
        aggregates=ToolCallAggregates(
            cleanThroughTurn=BinaryMetric(
                name="cleanThroughTurn4", successes=9, n=12, unit="conversation"
            ),
            restraint=BinaryMetric(name="restraint", successes=38, n=40, unit="turn"),
            funnel=(
                BinaryMetric(name="nativeCallEmitted", successes=142, n=320, unit="turn"),
            ),
        ),
        fingerprint_fields=model_fields(
            modelKey=run_id, packId="tool-caller-shop-assistant"
        ),
    )


def _toolcall_pack():
    return PackRef(
        packId="tool-caller-shop-assistant", packVersion="1.0.0", contentHash=None,
        role="tool-caller",
        metrics=PackMetrics(
            verdictMetrics=("cleanThroughTurn4",), headlineMetric="cleanThroughTurn4"
        ),
        pairingKey=("scriptId", "replicate"), analysisUnit="scriptId", seed=20260902,
    )


def test_no_wilson_interval_is_printed_over_a_turn_pooled_count() -> None:
    """`-ml` §4.4's first mandatory consequence, verbatim: *"Never print a Wilson interval over a
    turn-pooled count."*

    The Arms table rendered `wilson_interval(successes, n)` for **every** `BinaryMetric`, and
    `ToolCallAggregates.named_metrics()` returns `restraint` and the funnel counts, which §4.3
    defines as turn- and call-denominated. Measured in review M-ML-3: `nativeCallEmitted` at
    142/320 turns printed **[0.390, 0.499]**, a 10.8 pp interval where the honest bound at the
    §4.5.1(i) cap (12 clusters) is ~48.7 pp — understated 4.5x. The `exploratory` label mitigates
    the *verdict* risk and does not cure the *interval*: a printed +-5 pp reads as precision
    whatever it is labelled.
    """
    md = compare_report([_toolcall_arm("cand"), _toolcall_arm("incumbent")], pack=_toolcall_pack())

    clean = next(ln for ln in md.splitlines() if "| cleanThroughTurn4 |" in ln)
    assert "9/12" in clean
    assert "[0.468, 0.911]" in clean  # legitimate: n is conversations, the analysis unit

    for name, k_n in (("restraint", "38/40"), ("nativeCallEmitted", "142/320")):
        row = next(ln for ln in md.splitlines() if f"| {name} |" in ln)
        assert k_n in row  # the count itself is never suppressed
        assert "[" not in row.split("|")[-2]
        assert "n is turns" in row
    assert "[0.390, 0.499]" not in md
    assert "not the analysis unit" in md


def test_the_suppressed_interval_carries_its_reason() -> None:
    md = compare_report([_toolcall_arm("cand"), _toolcall_arm("incumbent")], pack=_toolcall_pack())
    assert "Never print a Wilson interval over a turn-pooled count" in md


def test_a_binary_metric_must_declare_its_denominator_unit() -> None:
    """No default, for the reason `-ml` §3.4 Rule 2 gives about `design_effect`: the anti-
    conservative value here is "the analysis unit", which is what licenses the interval, so a
    default is the caller who forgets clustering all over again. `BinaryMetric` carrying no
    denominator unit is the proximate cause review M-ML-3 names — `report.py` could not tell a
    per-analysis-unit rate from a turn-pooled one."""
    import dataclasses

    field = {f.name: f for f in dataclasses.fields(BinaryMetric)}["unit"]
    assert field.default is dataclasses.MISSING
    with pytest.raises(TypeError):
        BinaryMetric(name="m", successes=1, n=2)


def test_the_denominator_unit_survives_a_disk_round_trip(tmp_path) -> None:
    from modelbench.results import RunResult, store

    original = _toolcall_arm("cand")
    path = store(original, tmp_path)
    restored = RunResult.from_dict(json.loads(path.read_text()))
    assert restored == original
    assert {m.unit for m in restored.aggregates.named_metrics()} == {"conversation", "turn"}


# --- P3-5: the bootstrap seed is the pack's declaration, not a literal in the renderer ----------


def _seed_arms(a_ok: list[bool], b_ok: list[bool]):
    """Two arms over `len(a_ok)` items with exactly the given per-item outcomes, on the fail-safe
    (`basis="assumed"`) path so the seeded cluster bootstrap is what decides."""
    def arm(name, oks, correct):
        items = [
            ItemResult(itemId=f"g{i:02d}", pairingKey=(f"g{i:02d}",),
                       outcome="pass" if ok else "fail", scoreable={METRIC: True},
                       counts={METRIC: 1 if ok else 0}, latencyMs=1300.0, detail={})
            for i, ok in enumerate(oks)
        ]
        return run(name, items=items,
                   aggregates=classification_aggregates(correct, len(oks)), basis="assumed",
                   fingerprint_fields=model_fields(modelKey=name, packId=PACK_ID))
    return [arm("cand", a_ok, sum(a_ok)), arm("incumbent", b_ok, sum(b_ok))]


def test_the_bootstrap_seed_comes_from_the_pack_and_is_printed_beside_the_instrument() -> None:
    """Review P3-5 (major) — `report.py` passed `bootstrap_seed=20260902`, a magic literal
    duplicating the manifest's `sampling.seed` (plan §3.3) that `PackRef` had no field for, so the
    pack's own declaration could not reach the decision. `-ml` §3.2d requires the seed recorded
    "so a report is reproducible"; it was in neither the fingerprint nor the report, and on the
    fail-safe path — which *every* comparison takes until S2's determinism probe lands — the
    bootstrap is what decides. A reader handed a bootstrap-decided verdict could not reproduce it.

    The fingerprint half stays S2's (that is the runner's record, not the reporter's).
    """
    pack = guard_pack(headline=METRIC, verdicts=(METRIC,))._replace(seed=4242)
    a_ok = [True] * 40
    b_ok = [i >= 6 for i in range(40)]
    md = compare_report(_seed_arms(a_ok, b_ok), pack=pack)
    assert "- decided by: cluster-bootstrap (seed 4242, from the pack's `sampling.seed`)" in md
    assert "20260902" not in md


def test_the_printed_seed_is_the_seed_the_interval_was_resampled_at() -> None:
    """The half the rendered line alone cannot prove, and the one P3-5 is actually about.

    Asserting only *"seed 4242"* appears left `bootstrap_seed=20260902` — the literal this finding
    removes — alive: the report printed the pack's seed over an interval resampled at a different
    one, which is precisely the unreproducible verdict. So the seed has to be bound to the number
    it explains.

    Most binary fixtures cannot show it: the percentile bootstrap over ±1/0 unit differences lands
    on a coarse lattice, and at n=40 the displayed bounds are identical at every seed tried
    (measured: one distinct rendered interval across seeds 1–39). At **n=12 with b=5, c=3** the
    lattice is coarse enough for the percentile to move — seeds 1 and 5 render `[-25.0, 58.3]` and
    `[-33.3, 58.3]` pp — which is what makes this assertion possible at all.
    """
    a_ok = [True] * 5 + [False] * 3 + [True] * 4
    b_ok = [False] * 5 + [True] * 3 + [True] * 4
    arms = _seed_arms(a_ok, b_ok)
    pack = guard_pack(headline=METRIC, verdicts=(METRIC,))

    one = compare_report(arms, pack=pack._replace(seed=1))
    five = compare_report(arms, pack=pack._replace(seed=5))

    assert "(seed 1, from the pack's `sampling.seed`)" in one
    assert "(seed 5, from the pack's `sampling.seed`)" in five
    assert "[-25.0, 58.3] pp" in one
    assert "[-33.3, 58.3] pp" in five


def test_the_seed_is_not_printed_where_no_bootstrap_decided_anything() -> None:
    """The seed is provenance for an instrument that ran. On the `mcnemar-exact` path no resample
    happened, so naming a seed there would claim a reproducibility that is not at issue."""
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "- decided by: mcnemar-exact" in md
    assert "seed" not in md


def test_a_manifest_that_declares_no_resample_seed_is_refused(tmp_path) -> None:
    """The other half of P3-5: `PackRef.seed` has no default, so a manifest omitting
    `sampling.seed` must be a named refusal rather than a `KeyError` or a conjured number.

    A default would rebuild the defect it replaces — a seed nobody declared reproduces nothing —
    and it is the same defaulting shape `-ml` §3.4 Rule 2 refuses for `design_effect`.
    """
    from modelbench.packs import pack_ref_from_manifest

    manifest = tmp_path / "pack.json"
    manifest.write_text(json.dumps({
        "packId": "p", "packVersion": "1.0.0", "role": "guard-judge",
        "sampling": {"pairingKey": ["itemId"], "analysisUnit": "itemId"},
        "metrics": {"verdictMetrics": ["m"], "headlineMetric": "m"},
    }))
    with pytest.raises(PackConfigError, match="sampling.seed is absent"):
        pack_ref_from_manifest(manifest)


def test_a_pack_ref_built_in_code_with_an_out_of_family_headline_is_refused() -> None:
    """Review P3-12 — `compare_report`'s headline-membership guard was untested: deleting it left
    the suite green, and the only test of the rule goes through `metrics_from_manifest`, which a
    `PackRef` built in code bypasses entirely. S2 and every fixture here build one that way.

    Without the guard the failure is not a refusal but a bare `StopIteration` with no message, from
    `next(v for m, v, _ in computed if m == pack.metrics.headlineMetric)` — a generator exhausting
    two hundred lines from the rule it violated.
    """
    pack = guard_pack(headline=METRIC, verdicts=(METRIC,))._replace(
        metrics=PackMetrics(verdictMetrics=("someOtherMetric",), headlineMetric=METRIC)
    )
    with pytest.raises(PackConfigError, match="headlineMetric"):
        compare_report(_nested_arms(), pack=pack)


def test_a_manifest_that_declares_no_analysis_unit_is_refused_by_name(tmp_path) -> None:
    """Review P3-15 — removing `pack_ref_from_manifest`'s `"analysisUnit" not in sampling` check
    survived the suite, degrading a named `PackConfigError` into a `KeyError` that `_cmd_compare`
    happens to catch and reports as *"invalid pack: 'analysisUnit'"* — a bare key name where the
    operator needs to be told which declaration is missing and why it matters."""
    from modelbench.packs import pack_ref_from_manifest

    manifest = tmp_path / "pack.json"
    manifest.write_text(json.dumps({
        "packId": "p", "packVersion": "1.0.0", "role": "guard-judge",
        "sampling": {"pairingKey": ["itemId"], "seed": 20260902},
        "metrics": {"verdictMetrics": ["m"], "headlineMetric": "m"},
    }))
    with pytest.raises(PackConfigError, match="sampling.analysisUnit is absent"):
        pack_ref_from_manifest(manifest)
