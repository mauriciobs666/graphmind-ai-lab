"""§5 test 5 — `compare_report`, where AC-2/AC-3/AC-4 become visible output.

Every fixture is a hand-built `RunResult`; no LM Studio, no network, no pack on disk (plan §4 S1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    BinaryMetric,
    ClassificationAggregates,
    PackMetrics,
    PackRef,
    ToolCallAggregates,
    classification_aggregates,
    deterministic_fields,
    embeddings_fields,
    guard_pack,
    item,
    model_fields,
    run,
)

from modelbench import report, stats
from modelbench.fingerprint import FieldProblem
from modelbench.packs import PackConfigError, metrics_from_manifest
from modelbench.report import (
    _SAMPLE_NOUN,
    DuplicateModelInReport,
    _better,
    _metric_value,
    _render_funnel,
    _render_hazard,
    _render_per_turn_position,
    _render_role_caveat,
    _render_speed,
    compare_report,
    rank_report,
    resolving_power_line,
)
from modelbench.results import (
    ContinuousMetric,
    DistributionSummary,
    FunnelCounts,
    GroundingAggregates,
    HazardPoint,
    InvalidRecord,
    ItemResult,
    ItemTiming,
    LatencyBlock,
    RetrievalAggregates,
    RunResult,
    TurnPositionRate,
    load_history,
    store,
)
from modelbench.roles import UNIT_KIND_BY_ROLE
from modelbench.roles import unit_kind as unit_kind_for_role
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


def _agg_from(items, metric: str = METRIC):
    """The aggregate a **consistent** scorer would emit for these items (S1 done-condition 10).

    Every fixture that reaches `compare_report` has to satisfy the `aggregates`-versus-`items`
    cross-check or it is excluded before the behaviour under test is reached, so the fixtures
    derive their stored aggregate from the same items in one pass — which is exactly the contract
    v1.8 §4 S2 puts on the real scorers.
    """
    scored = [it.scored_outcome(metric) for it in items]
    scored = [outcome for outcome in scored if outcome is not None]
    return classification_aggregates(sum(scored), len(scored), metric=metric)


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


def test_a_record_with_no_legible_problems_states_its_reason_once(tmp_path) -> None:
    """An `unparseable` record carries `problems=[]` by construction — nothing about the file was
    legible, so there are no fields to name (`results.load_history`, both `unparseable` paths).

    The detail fell back to `record.reason`, which is already the first half of the same line, so
    every such record rendered as *"unparseable: unparseable"* — a stutter that reads like a
    field named `unparseable` failed for the reason `unparseable`. The reason is printed once and
    the colon belongs to the details that follow it.
    """
    invalid = [
        InvalidRecord(
            path=tmp_path / "runs" / "truncated.json",
            runId=None,
            benchSchemaVersion=None,
            problems=[],
            reason="unparseable",
        )
    ]
    md = compare_report(
        _nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)), invalid=invalid
    )
    assert "> - `truncated.json` — unparseable" in md
    assert "unparseable: unparseable" not in md


def test_a_record_whose_problems_are_legible_still_lists_them_after_its_reason() -> None:
    """The other side of the same line, so removing the suffix outright reddens: a record that
    *does* carry problems keeps `reason: detail`, which is what names the failing fields."""
    invalid = [
        InvalidRecord(
            path=Path("runs/bad.json"),
            runId="bad",
            benchSchemaVersion=1,
            problems=[FieldProblem(field="kvCacheSetting", reason="empty")],
            reason="field",
        )
    ]
    md = compare_report(
        _nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)), invalid=invalid
    )
    assert "> - `bad` — field: `kvCacheSetting` (empty)" in md


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
                    timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None),
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
                    timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None), detail={},
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
                    timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None), detail={},
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


def test_a_metric_past_the_holm_stop_never_carries_a_significance_claim() -> None:
    """P4-2 — `report.py`'s `holm_tested=step.tested` wiring had no regression test at all.

    Hardcoding it to `True` left the whole suite green. The existing stop fixture cannot catch it:
    both its metrics land at p = 0.125, above the rank-2 step of 0.05 anyway, so nothing there
    depends on the keyword. This fixture puts **both** metrics at b=8, c=1 -> p = 0.039, which is
    above the alpha/2 = 0.025 step Holm tests the first at (so Holm stops) and **below** the 0.05
    step the second would face if it were tested. Rendered under the mutation, the report printed

        cand is better than incumbent on falseSuspendRate: +17.5 pp ... McNemar exact p=0.039

    three paragraphs above its own family row reading `not tested (Holm stops here)` — Pass 1's
    blocker verbatim, a significance claim for a metric Holm never tested, contradicted inside one
    document. The sibling wiring `alpha_step=step.threshold` is pinned; this keyword was not.
    """
    second = "falseSuspendRate"
    fields = model_fields(packId=PACK_ID)

    def items(is_a: bool):
        built = []
        for i in range(40):
            # both metrics: A wrong only on item 38, B wrong on 0..7 -> b = 8, c = 1, p = 0.039
            ok = (i != 38) if is_a else (i >= 8)
            built.append(
                ItemResult(
                    itemId=f"g{i:02d}", pairingKey=(f"g{i:02d}",), outcome="pass",
                    scoreable={METRIC: True, second: True},
                    counts={METRIC: int(ok), second: int(ok)},
                    timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None), detail={},
                )
            )
        return built

    a = run("cand", items=items(True), aggregates=classification_aggregates(39, 40),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=items(False), aggregates=classification_aggregates(32, 40),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=guard_pack(headline=None, verdicts=(METRIC, second)))

    # the fixture is at the boundary the finding needs: below the stopped member's own 0.05 step
    assert stats.mcnemar_exact(8, 1) == pytest.approx(0.0390625)
    assert "p=0.039" in md
    row = next(ln for ln in md.splitlines() if ln.startswith(f"| {second} |"))
    assert "| 0.0500 |" in row and "not tested (Holm stops here)" in row
    section = md.split(f"### {second}")[1].split("### ")[0]
    assert "Not tested: Holm–Bonferroni stops at the first non-rejection" in section
    assert "is better than" not in section
    # ...and nowhere else in the document either: the headline repeats the verdict text
    assert "is better than" not in md


def test_the_family_wise_paragraph_attaches_each_alpha_to_the_bound_it_governs() -> None:
    """P4-3 — exchanging the two alphas inside the family-wise paragraph left the suite green.

    The paragraph's whole job is to explain the pair, and under the swap it labels 0.05 the
    family-adjusted alpha and 0.025 the unadjusted one — contradicting the provenance parenthetical
    three paragraphs above, which still prints `alpha=0.025` beside the MDD. P3-6 pinned the
    provenance parenthetical and the floor's own alpha and left this paragraph unpinned, which is
    the third pass running that an alpha-attribution string was found unguarded. So both clauses
    are asserted verbatim, each with the bound it governs.
    """
    arms, pack = _two_metric_arms(a_wins_first=8, a_wins_second=6)
    md = compare_report(arms, pack=pack)

    assert pack.metrics.alpha_mdd == 0.025 and pack.metrics.alpha_family == 0.05
    assert "Every **MDD** above is computed at the family-adjusted alpha=0.025" in md
    assert (
        "every **observable floor** is computed at the unadjusted alpha=0.05, the loosest step a "
        "member can face"
    ) in md
    # ...and the parenthetical the swap would contradict still names the MDD's alpha
    assert "design effect 1.00, by-construction, alpha=0.025)" in md


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


def test_sample_noun_domain_is_exactly_the_declared_unit_kinds():
    """Review Pass 14 P14-3: `_SAMPLE_NOUN` is read through `.get(rp.unit_kind, unit_plural)`, so
    a unit kind missing from the map does not raise — it silently reverts to the generic plural
    (`"items"` for every kind), which is Pass 1's n-4 defect exactly. Nothing pinned the map's
    domain to the set of unit kinds a role can actually declare (`roles.UNIT_KIND_BY_ROLE`'s
    values), so dropping an entry left the full suite green."""
    assert set(_SAMPLE_NOUN) == set(UNIT_KIND_BY_ROLE.values())


@pytest.mark.parametrize(
    "unit_kind,expected_noun", sorted(_SAMPLE_NOUN.items())
)
def test_the_conditionality_clause_uses_each_declared_sample_noun(unit_kind, expected_noun):
    """Drives `resolving_power_line` over every noun `_SAMPLE_NOUN` declares, not just the
    conversation and item packs the rest of this suite happens to build fixtures for — the
    `"query"` (embedder) row had no render assertion anywhere in this file."""
    rp = stats.resolving_power(
        40, unit_kind=unit_kind, design_effect=1.0, basis="by-construction",
        alpha_family=0.05, alpha_mdd=0.05,
    )
    line = resolving_power_line(rp, guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert f"the 40 {expected_noun} in" in line


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
    assert "decided by: conservative envelope" in md
    assert "design effect 1.00, assumed" in md
    # The label says what the two instruments each did: the interval decides, McNemar vetoes
    # (review B-ML-2). At DEFF = 1.00 the widening is a no-op and the veto is the whole of the
    # path's conservatism, which is exactly the corner the blocker was about.
    assert "in conjunction with McNemar's exact test" in md
    # `-ml` v1.8 §3.2e(f) variant 2's closing — *"one"*, not *"a verdict"*: this path declares no
    # clustering, so the rationale that is true here is the unestablished design effect (m-ML-10).
    assert "so it may withhold one but never carries one on its own" in md
    assert "under clustering McNemar rejects too readily" not in md


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
    assert "decided by: conservative envelope" in md
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
    assert "decided by: conservative envelope" in md


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
    a = run("cand", items=a_items, aggregates=_agg_from(a_items),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=b_items, aggregates=_agg_from(b_items),
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
        timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None),
        detail={},
    )


def _pair_of_arms(a_items, b_items):
    """Both arms' aggregates derived from their own items — see `_agg_from`.

    These fixtures used to hard-code `n = len(items)` on both arms, which for an arm that declares
    **nothing** scoreable is precisely review P4-4's inconsistent record: `0/10` rendered beside
    *"no paired data"*. S1 done-condition 10 now excludes such an arm, so the fixture has to be a
    record a scorer could honestly have written, or the P3-1 behaviour below is never reached.
    """
    fields = model_fields(packId=PACK_ID)
    a = run("cand", items=a_items, aggregates=_agg_from(a_items),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=b_items, aggregates=_agg_from(b_items),
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


def test_the_no_paired_data_refusal_points_at_the_tally_only_where_one_is_beside_it() -> None:
    """P4-13 — the `_NO_PAIRED_DATA_TALLY` pointer sentence was untested.

    The metric's own section has the §4.3 tally printed directly under the refusal, so it says
    *"The tally below says where the rows went."* The **headline** repeats the same refusal with no
    table beside it, and must not point at one: a reader sent looking for a tally that is not there
    learns less than one who was told nothing.
    """
    a_items = [_bare(f"g{i:02d}", scoreable={METRIC: True}, counts={METRIC: 1}) for i in range(10)]
    b_items = [_bare(f"g{i:02d}", scoreable={METRIC: False}, counts={}) for i in range(10)]
    a, b = _pair_of_arms(a_items, b_items)
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    section = md.split(f"### {METRIC}")[1].split("###")[0]
    assert "The tally below says where the rows went." in section
    headline = md.split(f"**Headline ({METRIC}):**")[1]
    assert "No verdict: no paired data" in headline
    assert "The tally below says where the rows went." not in headline


def test_an_exploratory_metric_declared_by_both_arms_is_listed_once() -> None:
    """P4-13 — the exploratory dedup was untested, so removing it was green.

    The list is built from **both** arms' aggregates, and both arms normally declare the same
    metrics — so without the dedup every exploratory metric prints twice, which reads as two
    different metrics with one name rather than as one metric measured by two arms.
    """
    fields = model_fields(packId=PACK_ID)
    exploratory = "latencyBudgetHits"

    def arm(name: str, correct: int):
        items = [item(f"g{i:02d}", correct=i < correct, metric=METRIC) for i in range(10)]
        aggs = ClassificationAggregates(
            perClass=(
                BinaryMetric(name=METRIC, successes=correct, n=10, unit="item"),
                BinaryMetric(name=exploratory, successes=3, n=10, unit="item"),
            ),
            parseFailures=0, n=10,
        )
        return run(name, items=items, aggregates=aggs,
                   fingerprint_fields={**fields, "modelKey": name})

    md = compare_report([arm("cand", 10), arm("incumbent", 4)],
                        pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    listed = [
        ln for ln in md.splitlines()
        if ln == f"- `{exploratory}` — exploratory — no significance claim"
    ]
    assert len(listed) == 1


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


# --- P4-4 / S1 done-condition 10, §5 test 11c: aggregates must agree with items -----------------


def _ten_items_declaring_nothing_scoreable():
    return [
        ItemResult(
            itemId=f"g{i:02d}", pairingKey=(f"g{i:02d}",), outcome="pass",
            scoreable={METRIC: False}, counts={},
            timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None), detail={},
        )
        for i in range(10)
    ]


def test_an_arm_whose_aggregate_disagrees_with_its_items_is_excluded_and_named() -> None:
    """§5 test 11c / S1 done-condition 10 — review P4-4, the gate's own reproduction.

    An arm declaring `BinaryMetric(m, successes=0, n=10)` for a metric **no item declares
    scoreable** printed `0/10 = 0.000` in the Arms table — a claim that ten items were scored —
    beside a paired-rows section reading *"No verdict: no paired data … An arm carrying no data
    for a metric is not an arm that failed every item of it"*. One document, two mutually
    exclusive statements about the same metric. `_DESCRIPTIVE_NOTE` does not cover it: it caveats
    the **interval**, and says nothing about the **rate**, which is the half that misreports.

    Plan v1.8 fixes the response: the arm is **excluded and named in the existing
    `INVALID RESULTS EXCLUDED` block**, with the declared and the counted `n`, and the report is
    still produced. Raising would reproduce P4-5's shape — an abort outside `cli.py`'s exit-code
    set that takes the valid arm down with the invalid one — and suppressing only the offending
    row would leave a partially-trusted arm inside the comparison, when the mismatch is evidence
    that this scorer's per-item and aggregate paths disagree.
    """
    fields = model_fields(packId=PACK_ID)
    arms = [
        run(name, items=_ten_items_declaring_nothing_scoreable(),
            aggregates=classification_aggregates(0, 10),
            fingerprint_fields={**fields, "modelKey": name})
        for name in ("cand", "incumbent")
    ]
    md = compare_report(arms, pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    assert "0/10" not in md            # the rate that claimed ten items were scored is gone
    assert "0.000" not in md
    assert "**INVALID RESULTS EXCLUDED** (AC-2)" in md
    for name in ("cand", "incumbent"):
        line = next(ln for ln in md.splitlines() if ln.startswith(f"> - `{name}`"))
        assert f"`{METRIC}` (declared n=10, counted 0)" in line
    # ...and the report is still produced, saying why the arms are gone rather than pointing the
    # reader at the selection flags. Both sentences were individually true before this: the block
    # above said the arms were excluded, and the verdict line said *"fewer than two arms were
    # **selected**, so there is nothing to compare. Check `--models` and `--session`"* — which is
    # the wrong remedy for arms that were selected and then thrown away, and sends a scorer author
    # looking at their command line instead of at their record.
    assert md.startswith("# Comparison — ")
    assert "fewer than two arms were selected" not in md
    assert "Check `--models`" not in md
    assert (
        "None: fewer than two arms remain — 2 arms were excluded above because their stored "
        "aggregates disagree with their own items"
    ) in md


def test_a_genuinely_unselected_comparison_still_points_at_the_selection_flags() -> None:
    """The other side: where nothing was excluded, too-few-arms really is a selection problem and
    `--models` / `--session` really is the thing to check (review M-6's original reason)."""
    md = compare_report(_nested_arms()[:1], pack=guard_pack(headline=METRIC))
    assert "fewer than two arms were selected" in md
    assert "Check `--models` and `--session`" in md
    assert "excluded above" not in md


def test_an_item_declaring_a_count_it_does_not_carry_is_a_mismatch_not_a_traceback() -> None:
    """Plan-gate **G3-7** — DC-10's counting call reproduces the P4-5 shape DC-10 rejects.

    DC-10 counts items "for which `scored_outcome(metric) is not None`", but `scored_outcome` does
    not return `None` for the sibling malformation: a metric declared `scoreable: True` with no
    entry in `counts` **raises `IncompleteItemRecord`**. Nothing on that path catches it —
    `_cmd_compare` catches only `PackConfigError` — so a check written literally from DC-10 turns
    an inconsistent record into a traceback at exit 1, outside §3.6a's closed set, which is exactly
    the response DC-10 rejects raising for.

    So the cross-check treats it as a **mismatch**: the arm is excluded and named, with the
    offending item and metric. `load_history` also quarantines such a record on read (P4-5), and
    the two nets sit at different seams — this one makes `compare_report` total on the input rather
    than trusting its caller to have filtered.
    """
    fields = model_fields(packId=PACK_ID)
    good = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(10)]
    broken = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(9)]
    broken.append(
        ItemResult(
            itemId="g09", pairingKey=("g09",), outcome="pass",
            scoreable={METRIC: True}, counts={},
            timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None), detail={},
        )
    )
    a = run("cand", items=good, aggregates=_agg_from(good),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("halfscored", items=broken, aggregates=classification_aggregates(10, 10),
            fingerprint_fields={**fields, "modelKey": "halfscored"})

    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    line = next(ln for ln in md.splitlines() if ln.startswith("> - `halfscored`"))
    assert f"`{METRIC}` (item 'g09' declares it scoreable and records no count)" in line
    assert "| cand | falseAdvanceRate | 10/10 |" in md      # the sound arm survives
    assert "1 arm was excluded above" in md                 # singular, and not the selection flags
    assert "Check `--models`" not in md


def test_the_cross_check_selects_on_the_denominator_unit_not_the_pairing_key_name() -> None:
    """Plan-gate **G3-6** — DC-10's written selector names two disjoint vocabularies.

    `BinaryMetric.unit` is a *denominator noun* (`item` / `conversation` / `query` / `turn` /
    `call`); `PackRef.analysisUnit` is a *`pairingKey` component name*, which `packs.py` constrains
    to `pairingKey[0]` — `itemId` for this pack. So `metric.unit == pack.analysisUnit` is
    **never** true, and a check written literally from DC-10 selects nothing and silently passes
    everything. Measured here rather than asserted, so the two vocabularies are pinned as distinct
    and this test cannot quietly become vacuous.

    The working predicate, already in use one function over for the Wilson-interval suppression,
    is `metric.unit == unit_kind_for_role(pack.role)`.
    """
    pack = guard_pack(headline=METRIC, verdicts=(METRIC,))
    assert unit_kind_for_role(pack.role) == "item"
    assert pack.analysisUnit == "itemId"
    assert unit_kind_for_role(pack.role) != pack.analysisUnit   # the two vocabularies are disjoint

    # ...and a metric whose denominator is *not* the analysis unit is out of the check's scope,
    # because there is no per-item count to compare a pooled denominator against (`-ml` §4.4).
    fields = model_fields(packId=PACK_ID)
    items = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(10)]
    aggs = ClassificationAggregates(
        perClass=(
            BinaryMetric(name=METRIC, successes=10, n=10, unit="item"),
            BinaryMetric(name=METRIC, successes=3, n=70, unit="turn"),
        ),
        parseFailures=0, n=10,
    )
    a = run("cand", items=items, aggregates=aggs, fingerprint_fields={**fields, "modelKey": "cand"})
    b_items = [item(f"g{i:02d}", correct=i >= 6, metric=METRIC) for i in range(10)]
    b = run("incumbent", items=b_items, aggregates=_agg_from(b_items),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=pack)
    assert "INVALID RESULTS EXCLUDED" not in md
    assert "| cand | falseAdvanceRate | 3/70 |" in md


def test_an_arm_whose_aggregate_matches_its_items_is_not_excluded() -> None:
    """The positive half of §5 test 11c — the check has to discriminate, not exclude everything.

    Without this, `_aggregate_item_mismatches` returning every metric it sees would pass the test
    above and make the tool useless.
    """
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "INVALID RESULTS EXCLUDED" not in md
    assert "| cand | falseAdvanceRate | 40/40 |" in md
    assert "| incumbent | falseAdvanceRate | 34/40 |" in md
    assert "is better than" in md


def test_the_cross_check_counts_scoreable_items_not_correct_ones() -> None:
    """The denominator being cross-checked is *scoreability*, never the score.

    An arm that declares ten items scoreable and gets none of them right is a perfectly consistent
    record — `0/10` is then a real measurement — so a check written against `successes` instead of
    the scoreable count would exclude exactly the arm the tool exists to report on.
    """
    fields = model_fields(packId=PACK_ID)
    items = [item(f"g{i:02d}", correct=False, metric=METRIC) for i in range(10)]
    a = run("cand", items=items, aggregates=classification_aggregates(0, 10),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=[item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(10)],
            aggregates=classification_aggregates(10, 10),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "INVALID RESULTS EXCLUDED" not in md
    assert "| cand | falseAdvanceRate | 0/10 |" in md


def test_a_mismatch_on_a_metric_outside_the_verdict_family_is_not_the_checks_business() -> None:
    """S1 done-condition 10 scopes the check to the pack's `verdictMetrics` family.

    An exploratory metric carries no significance claim and no verdict, so a disagreement there is
    not grounds for throwing away an arm whose pre-registered metrics are sound — and widening the
    scope would make the check refuse records the comparison never reads.
    """
    fields = model_fields(packId=PACK_ID)
    exploratory = "latencyBudgetHits"
    items = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(10)]
    aggs = ClassificationAggregates(
        perClass=(
            BinaryMetric(name=METRIC, successes=10, n=10, unit="item"),
            BinaryMetric(name=exploratory, successes=3, n=10, unit="item"),
        ),
        parseFailures=0, n=10,
    )
    a = run("cand", items=items, aggregates=aggs,
            fingerprint_fields={**fields, "modelKey": "cand"})
    b_items = [item(f"g{i:02d}", correct=i >= 6, metric=METRIC) for i in range(10)]
    b = run("incumbent", items=b_items,
            aggregates=classification_aggregates(4, 10),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "INVALID RESULTS EXCLUDED" not in md
    assert f"`{exploratory}` — exploratory — no significance claim" in md


# --- §4 S1e Table F: the continuous carrier, on the `report.py` side -----------------------------


def _embedder_pack(verdicts: tuple[str, ...] = ("mrr",), headline: str | None = "mrr") -> PackRef:
    return PackRef(
        packId="embedder-graphrag-retrieval", packVersion="1.0.0", contentHash="e" * 64,
        role="embedder", metrics=PackMetrics(verdictMetrics=verdicts, headlineMetric=headline),
        pairingKey=("itemId",), analysisUnit="itemId", seed=20260902,
    )


def _mrr_item(query_id: str, value: float) -> ItemResult:
    return ItemResult(
        itemId=query_id, pairingKey=(query_id,), outcome="pass", scoreable={"mrr": True},
        counts={}, timing=None, measures={"mrr": value}, detail={},
    )


def test_a_continuous_member_declaring_a_measure_it_does_not_carry_is_a_mismatch() -> None:
    """§4 S1e Table F — DC-10's third arithmetic, for a continuous member: the same
    exclude-and-name mechanism as the binary sibling (plan-gate G3-7), read through
    `scored_value` instead of `scored_outcome`."""
    pack = _embedder_pack()
    good_items = [_mrr_item(f"q{i:02d}", 0.5) for i in range(10)]
    broken_items = [_mrr_item(f"q{i:02d}", 0.5) for i in range(9)]
    broken_items.append(
        ItemResult(itemId="q09", pairingKey=("q09",), outcome="pass", scoreable={"mrr": True},
                   counts={}, timing=None, measures={}, detail={})
    )
    agg = RetrievalAggregates(mrr=ContinuousMetric(name="mrr", mean=0.5, n=10, support=(0.0, 1.0)))
    a = run("cand", role="embedder", call_surface="embeddings", items=good_items, aggregates=agg,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    b = run("halfscored", role="embedder", arm_kind="deterministic", items=broken_items,
            aggregates=agg,
            fingerprint_fields=deterministic_fields(packId=pack.packId, armId="halfscored"))

    md = compare_report([a, b], pack=pack)
    line = next(ln for ln in md.splitlines() if ln.startswith("> - `halfscored`"))
    assert "`mrr` (item 'q09' declares it scoreable and records no measure)" in line
    assert "1 arm was excluded above" in md


def test_a_kind_disagreement_continuous_aggregate_binary_items_is_the_same_mismatch_class() -> None:
    """§4 S1e Table F — "the same check is where a kind disagreement surfaces": an aggregate that
    says continuous while the per-item values live in `counts` is DC-10's mismatch exactly,
    because the arm's aggregate path and its per-item path disagree about what was measured."""
    pack = _embedder_pack()
    good_items = [_mrr_item(f"q{i:02d}", 0.5) for i in range(10)]
    wrong_kind_items = [
        ItemResult(itemId=f"q{i:02d}", pairingKey=(f"q{i:02d}",), outcome="pass",
                   scoreable={"mrr": True}, counts={"mrr": 1}, timing=None, measures={},
                   detail={})
        for i in range(10)
    ]
    agg = RetrievalAggregates(mrr=ContinuousMetric(name="mrr", mean=0.5, n=10, support=(0.0, 1.0)))
    a = run("cand", role="embedder", call_surface="embeddings", items=good_items, aggregates=agg,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    b = run("wrongkind", role="embedder", arm_kind="deterministic", items=wrong_kind_items,
            aggregates=agg,
            fingerprint_fields=deterministic_fields(packId=pack.packId, armId="wrongkind"))

    md = compare_report([a, b], pack=pack)
    assert "1 arm was excluded above" in md
    line = next(ln for ln in md.splitlines() if ln.startswith("> - `wrongkind`"))
    assert "`mrr`" in line


def test_reverse_kind_disagreement_binary_aggregate_continuous_items_is_also_a_mismatch() -> None:
    """The reverse direction: an aggregate declares `METRIC` binary while an item scored it
    continuously — `scored_outcome` raises `MetricKindError`, caught by DC-10 on the same route
    (§4 S1e Table F)."""
    fields = model_fields(packId=PACK_ID)
    good_items = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(10)]
    wrong_kind_items = [
        ItemResult(
            itemId=f"g{i:02d}", pairingKey=(f"g{i:02d}",), outcome="pass",
            scoreable={METRIC: True}, counts={},
            timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None),
            measures={METRIC: 1.0}, detail={},
        )
        for i in range(10)
    ]
    a = run("cand", items=good_items, aggregates=_agg_from(good_items),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("wrongkind", items=wrong_kind_items, aggregates=classification_aggregates(10, 10),
            fingerprint_fields={**fields, "modelKey": "wrongkind"})

    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "1 arm was excluded above" in md
    assert f"`{METRIC}`" in md


def test_arms_table_renders_a_distribution_summary_without_reading_mean() -> None:
    """§4 S1e Table F — the shipped bare `else` reads `.mean` and would raise `AttributeError` on
    this type; the fix renders the median and p10, labelled, and no interval."""
    pack = _embedder_pack(verdicts=(), headline=None)
    items = [_mrr_item(f"q{i:02d}", 0.5) for i in range(3)]
    agg = RetrievalAggregates(
        separationZ=DistributionSummary(
            name="separationZ", median=1.5, p10=-0.3, n=40, unit="query", support=None
        )
    )
    a = run("cand", role="embedder", call_surface="embeddings", items=items, aggregates=agg,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    b = run("bm25", role="embedder", arm_kind="deterministic", items=items, aggregates=agg,
            fingerprint_fields=deterministic_fields(packId=pack.packId))

    md = compare_report([a, b], pack=pack)
    assert "| cand | separationZ | n=40 | p50 1.5000, p10 -0.3000 | — |" in md


def test_a_continuous_verdict_member_is_routed_through_continuous_verdict_not_booleanised() -> None:
    """§4 S1e Table F shipped the carrier and stopped at its proof surface — a placeholder
    `MetricKindError` raise, reached because the still-binary family loop's `_paired_rows` call
    read `scored_outcome` on a `measures`-resident metric. **This unit closes that seam**: pass 1
    now resolves `mrr`'s aggregate as continuous (`_metric_kind`), so the family loop routes it
    through `_paired_diffs`/`continuous_verdict()` instead — `scored_outcome` is never called for
    a well-formed continuous member, so the raise this test used to pin is no longer reached.

    Ten paired queries (not one) so the interval `continuous_verdict` computes is a real one — the
    one-unit refusal is
    `test_a_continuous_verdict_member_with_one_paired_unit_prints_a_named_refusal`'s own case.
    """
    pack = _embedder_pack()
    items_a = [_mrr_item(f"q{i:02d}", 0.9) for i in range(10)]
    items_b = [_mrr_item(f"q{i:02d}", 0.5) for i in range(10)]
    agg_a = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=0.9, n=10, support=(0.0, 1.0))
    )
    agg_b = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=0.5, n=10, support=(0.0, 1.0))
    )
    a = run("cand", role="embedder", call_surface="embeddings", items=items_a, aggregates=agg_a,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    b = run("bm25", role="embedder", arm_kind="deterministic", items=items_b, aggregates=agg_b,
            fingerprint_fields=deterministic_fields(packId=pack.packId))

    md = compare_report([a, b], pack=pack)  # must not raise

    section = md.split("### mrr")[1].split("###")[0]
    assert "No verdict: no paired data" not in section
    provenance = "decided by paired bootstrap on per-query differences (B=10000, seed=20260902)"
    assert provenance in section
    # A McNemar verdict prints a percentage-point difference (`+X.X pp`); a continuous one prints
    # a plain decimal (`diff:+.3f`) — the two renderings must not be confusable.
    assert " pp" not in section
    assert "**Headline (mrr):**" in md
    assert "decided by paired bootstrap" in md.split("**Headline (mrr):**")[1].split("\n")[0]


def test_a_continuous_verdict_member_with_zero_paired_units_prints_no_paired_data() -> None:
    """The continuous branch's own empty-intersection guard — parity with the binary branch's
    `rp is None` case (review P3-1) — rather than letting `continuous_verdict` raise on an empty
    `diffs` list."""
    pack = _embedder_pack()
    items_a = [_mrr_item("q1", 0.9)]
    items_b = [ItemResult(itemId="q1", pairingKey=("q1",), outcome="pass",
                           scoreable={"mrr": False}, counts={}, timing=None, measures={},
                           detail={})]
    agg_a = RetrievalAggregates(mrr=ContinuousMetric(name="mrr", mean=0.9, n=1, support=(0.0, 1.0)))
    agg_b = RetrievalAggregates(mrr=ContinuousMetric(name="mrr", mean=0.0, n=0, support=(0.0, 1.0)))
    a = run("cand", role="embedder", call_surface="embeddings", items=items_a, aggregates=agg_a,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    b = run("bm25", role="embedder", arm_kind="deterministic", items=items_b, aggregates=agg_b,
            fingerprint_fields=deterministic_fields(packId=pack.packId))

    md = compare_report([a, b], pack=pack)

    section = md.split("### mrr")[1].split("###")[0]
    assert "No verdict: no paired data" in section
    assert "paired n: 0 of 1" in section


def test_a_continuous_verdict_member_with_one_paired_unit_prints_a_named_refusal() -> None:
    """`continuous_verdict` refuses a one-unit interval (`-ml` §3.4 Rule 8, refusal 4) — but
    `_NO_PAIRED_DATA` would be false here (one unit *is* paired), so the continuous branch guards
    the refusal with its own message rather than letting the `ValueError` escape uncaught."""
    pack = _embedder_pack()
    items_a = [_mrr_item("q1", 1.0)]
    items_b = [_mrr_item("q1", 0.5)]
    agg_a = RetrievalAggregates(mrr=ContinuousMetric(name="mrr", mean=1.0, n=1, support=(0.0, 1.0)))
    agg_b = RetrievalAggregates(mrr=ContinuousMetric(name="mrr", mean=0.5, n=1, support=(0.0, 1.0)))
    a = run("cand", role="embedder", call_surface="embeddings", items=items_a, aggregates=agg_a,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    b = run("bm25", role="embedder", arm_kind="deterministic", items=items_b, aggregates=agg_b,
            fingerprint_fields=deterministic_fields(packId=pack.packId))

    md = compare_report([a, b], pack=pack)  # must not raise

    section = md.split("### mrr")[1].split("###")[0]
    assert "No verdict: one paired query" in section
    assert "paired n: 1 of 1" in section


def test_a_continuous_units_value_is_the_mean_over_its_items_not_a_flattened_pool() -> None:
    """`-ml` §3.2d — a unit's value is its item's `scored_value` when unit ≡ item, and the **mean
    over its items** when a unit spans more than one. Values are chosen so three readings all
    disagree: q1 is one item per arm (diff = 1.0 - 0.0 = 1.0); q2 is three items per arm, unequal
    within each arm so a unit's *first* item is not its mean.

    - **Correct (mean per unit, then averaged over units):** q2 diff = mean(0.0, 0.3, 0.9) -
      mean(0.5, 0.5, 0.5) = 0.4 - 0.5 = -0.1. Over the two units: (1.0 + -0.1) / 2 = **+0.450**.
    - **Wrong — first item, not the mean:** q2 diff = 0.0 - 0.5 = -0.5. Over the two units:
      (1.0 + -0.5) / 2 = +0.250 — a different number this test catches.
    - **Wrong — flattened over all four items, no per-unit grouping:** pairing same-index items
      gives 1.0, -0.5, -0.2, +0.4, averaging to (1.0 - 0.5 - 0.2 + 0.4) / 4 = +0.175 — also
      different, and also caught.
    """
    pack = PackRef(
        packId="embedder-multi-chunk", packVersion="1.0.0", contentHash="f" * 64, role="embedder",
        metrics=PackMetrics(verdictMetrics=("mrr",), headlineMetric="mrr"),
        pairingKey=("itemId", "chunk"), analysisUnit="itemId", seed=20260902,
    )

    def chunk_items(values: dict[str, list[float]]) -> list[ItemResult]:
        return [
            ItemResult(itemId=f"{q}-{i}", pairingKey=(q, str(i)), outcome="pass",
                       scoreable={"mrr": True}, counts={}, timing=None,
                       measures={"mrr": v}, detail={})
            for q, vals in values.items()
            for i, v in enumerate(vals)
        ]

    a_items = chunk_items({"q1": [1.0], "q2": [0.0, 0.3, 0.9]})
    b_items = chunk_items({"q1": [0.0], "q2": [0.5, 0.5, 0.5]})
    agg_a = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=0.55, n=len(a_items), support=(0.0, 1.0))
    )
    agg_b = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=0.375, n=len(b_items), support=(0.0, 1.0))
    )
    a = run("cand", role="embedder", call_surface="embeddings", items=a_items, aggregates=agg_a,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    b = run("bm25", role="embedder", arm_kind="deterministic", items=b_items, aggregates=agg_b,
            fingerprint_fields=deterministic_fields(packId=pack.packId))

    md = compare_report([a, b], pack=pack)

    section = md.split("### mrr")[1].split("###")[0]
    assert "paired n: 2 of 2" in section
    assert "+0.450" in section
    assert "+0.250" not in section
    assert "+0.175" not in section


def _mixed_pack(
    verdicts: tuple[str, ...] = ("mrr", "precisionAt1"), headline: str | None = None
) -> PackRef:
    return PackRef(
        packId="embedder-mixed-family", packVersion="1.0.0", contentHash="a" * 64, role="embedder",
        metrics=PackMetrics(verdictMetrics=verdicts, headlineMetric=headline),
        pairingKey=("itemId",), analysisUnit="itemId", seed=20260902,
    )


def _mixed_item(query_id: str, mrr_value: float, precision_hit: bool) -> ItemResult:
    return ItemResult(
        itemId=query_id, pairingKey=(query_id,), outcome="pass",
        scoreable={"mrr": True, "precisionAt1": True},
        counts={"precisionAt1": int(precision_hit)}, timing=None,
        measures={"mrr": mrr_value}, detail={},
    )


def _mixed_arms(pack: PackRef) -> tuple[RunResult, RunResult]:
    items_a = [_mixed_item(f"q{i:02d}", 0.9, True) for i in range(5)]
    items_b = [_mixed_item(f"q{i:02d}", 0.5, False) for i in range(5)]
    agg_a = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=0.9, n=5, support=(0.0, 1.0)),
        precisionAt1=BinaryMetric(name="precisionAt1", successes=5, n=5, unit="query"),
    )
    agg_b = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=0.5, n=5, support=(0.0, 1.0)),
        precisionAt1=BinaryMetric(name="precisionAt1", successes=0, n=5, unit="query"),
    )
    a = run("cand", role="embedder", call_surface="embeddings", items=items_a, aggregates=agg_a,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    b = run("bm25", role="embedder", arm_kind="deterministic", items=items_b, aggregates=agg_b,
            fingerprint_fields=deterministic_fields(packId=pack.packId))
    return a, b


def test_a_mixed_kind_family_is_refused_whole_not_pruned_to_the_majority_kind() -> None:
    """§3.3 (iv) — a family mixing a continuous and a binary verdict metric is refused *whole*:
    no member is verdicted, nothing is excluded, each member's own block names its resolved kind,
    and the family-wise section states the refusal instead of a false Holm claim."""
    pack = _mixed_pack()
    a, b = _mixed_arms(pack)

    md = compare_report([a, b], pack=pack)

    assert "INVALID RESULTS EXCLUDED" not in md
    mrr_section = md.split("### mrr")[1].split("###")[0]
    precision_section = md.split("### precisionAt1")[1].split("###")[0]
    assert "continuous metric — no verdict" in mrr_section
    assert "binary metric — no verdict" in precision_section
    assert "is better than" not in mrr_section
    assert "is better than" not in precision_section
    assert "paired n: 5 of 5" in mrr_section
    assert "paired n: 5 of 5" in precision_section
    fw_section = md.split("### Family-wise error control")[1].split("###")[0]
    assert "mixes binary and continuous verdict metrics" in fw_section
    assert "Holm–Bonferroni" not in fw_section
    exploratory_section = md.split("### Exploratory metrics")[1]
    assert "- `mrr` — exploratory — no significance claim" in exploratory_section
    assert "- `precisionAt1` — exploratory — no significance claim" in exploratory_section


def test_a_mixed_kind_familys_headline_prints_exploratory_not_no_paired_data() -> None:
    """§3.3 (iv) — the headline fallback is `_NO_PAIRED_DATA` for an unverdicted member with no
    data, which is false for a refused family (there *is* paired data); it prints the same
    exploratory label the family's other numbers get instead."""
    pack = _mixed_pack(headline="mrr")
    a, b = _mixed_arms(pack)

    md = compare_report([a, b], pack=pack)

    assert "**Headline (mrr):** exploratory — no significance claim" in md
    assert "No verdict: no paired data" not in md


def test_an_all_continuous_family_takes_its_correction_in_the_interval_not_a_ladder() -> None:
    """§3.3 (iv) — an all-continuous family with `k > 1` never reaches `holm_steps`: the
    family-wise section states the interval-correction explanation instead of a Holm-ladder
    claim, both members print the bootstrap-decided sentence, and no arm is excluded.

    This test does **not** prove the family-size correction actually reaches each member's
    interval — all four assertions below hold unchanged even if `compare_report` collapsed
    `family` to `[metric]` before calling `stats.continuous_verdict()` (U67, found by mutation
    testing). `test_continuous_verdict_receives_the_whole_family_not_just_the_metric` pins that
    property instead."""
    pack = _embedder_pack(verdicts=("mrr", "separationZ"), headline=None)
    n = 12

    def dual_item(query_id: str, mrr_value: float, sepz_value: float) -> ItemResult:
        return ItemResult(
            itemId=query_id, pairingKey=(query_id,), outcome="pass",
            scoreable={"mrr": True, "separationZ": True}, counts={}, timing=None,
            measures={"mrr": mrr_value, "separationZ": sepz_value}, detail={},
        )

    items_a = [dual_item(f"q{i:02d}", 0.9, 2.0 + 0.01 * i) for i in range(n)]
    items_b = [dual_item(f"q{i:02d}", 0.3, 0.01 * i) for i in range(n)]
    agg_a = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=0.9, n=n, support=(0.0, 1.0)),
        separationZ=DistributionSummary(
            name="separationZ", median=2.05, p10=2.0, n=n, unit="query", support=None
        ),
    )
    agg_b = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=0.3, n=n, support=(0.0, 1.0)),
        separationZ=DistributionSummary(
            name="separationZ", median=0.05, p10=0.0, n=n, unit="query", support=None
        ),
    )
    a = run("cand", role="embedder", call_surface="embeddings", items=items_a, aggregates=agg_a,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    b = run("bm25", role="embedder", arm_kind="deterministic", items=items_b, aggregates=agg_b,
            fingerprint_fields=deterministic_fields(packId=pack.packId))

    md = compare_report([a, b], pack=pack)

    fw_section = md.split("### Family-wise error control")[1].split("###")[0]
    assert "family-wise correction is taken in each metric's own interval" in fw_section
    assert "Holm–Bonferroni" not in fw_section
    assert md.count("decided by paired bootstrap on per-query differences") == 2
    assert "INVALID RESULTS EXCLUDED" not in md


def test_continuous_verdict_receives_the_whole_family_not_just_the_metric() -> None:
    """U67 (found by mutation testing during integration, not by the suite) — the all-continuous
    branch of `compare_report`'s family loop must call `stats.continuous_verdict(family=family,
    ...)` with the **whole** pre-registered family, not `family=[metric]`. Collapsing it sets
    `k = len(family) = 1` inside `_family_ci_levels`, which silently skips the Bonferroni
    correction §3.3 (iv) says a `k > 1` all-continuous family "takes ... in the interval — it has
    nowhere else to put it": the printed interval would be too narrow and the verdict too
    confident, with no visible symptom (`-ml` §3.4 Rule 8, §11.2.2).

    Renders the identical `mrr` data and seed twice — once as the only pre-registered metric
    (`k = 1`, levels `1/40, 39/40`) and once alongside a second all-continuous metric
    (`k = 2`, levels `1/80, 79/80`, per `-ml` §11.2.2's worked table, already pinned directly for
    `_family_ci_levels` in `tests/test_stats.py`). Same seed and identical per-unit `mrr`
    differences mean the two runs bootstrap-resample identically; only the quantile levels differ,
    so the `k = 2` interval must come out strictly wider. `family=[metric]` renders both at the
    `k = 1` levels and this assertion fails.
    """
    n = 12
    mrr_a = [0.9 - 0.01 * i for i in range(n)]
    mrr_b = [0.3 + 0.005 * i for i in range(n)]

    def mrr_only_item(query_id: str, value: float) -> ItemResult:
        return ItemResult(
            itemId=query_id, pairingKey=(query_id,), outcome="pass",
            scoreable={"mrr": True}, counts={}, timing=None,
            measures={"mrr": value}, detail={},
        )

    def mrr_and_sepz_item(query_id: str, mrr_value: float, sepz_value: float) -> ItemResult:
        return ItemResult(
            itemId=query_id, pairingKey=(query_id,), outcome="pass",
            scoreable={"mrr": True, "separationZ": True}, counts={}, timing=None,
            measures={"mrr": mrr_value, "separationZ": sepz_value}, detail={},
        )

    def mrr_ci(md: str) -> tuple[float, float]:
        section = md.split("### mrr")[1].split("###")[0]
        assert "CI" in section, section
        bracket = section.split("CI [", 1)[1].split("]", 1)[0]
        lo_str, hi_str = (part.strip() for part in bracket.split(","))
        return float(lo_str), float(hi_str)

    # k = 1: `mrr` is the only pre-registered metric.
    pack_k1 = _embedder_pack(verdicts=("mrr",), headline="mrr")
    items_a_k1 = [mrr_only_item(f"q{i:02d}", mrr_a[i]) for i in range(n)]
    items_b_k1 = [mrr_only_item(f"q{i:02d}", mrr_b[i]) for i in range(n)]
    agg_a_k1 = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=sum(mrr_a) / n, n=n, support=(0.0, 1.0))
    )
    agg_b_k1 = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=sum(mrr_b) / n, n=n, support=(0.0, 1.0))
    )
    a_k1 = run("cand", role="embedder", call_surface="embeddings", items=items_a_k1,
               aggregates=agg_a_k1,
               fingerprint_fields=embeddings_fields(packId=pack_k1.packId, modelKey="cand"))
    b_k1 = run("bm25", role="embedder", arm_kind="deterministic", items=items_b_k1,
               aggregates=agg_b_k1,
               fingerprint_fields=deterministic_fields(packId=pack_k1.packId))
    lo1, hi1 = mrr_ci(compare_report([a_k1, b_k1], pack=pack_k1))

    # k = 2: `mrr` and `separationZ` both pre-registered, both continuous, identical `mrr` values
    # in identical order — so `_paired_diffs(a, b, "mrr", pack)` is bit-identical to the k=1 run.
    pack_k2 = _embedder_pack(verdicts=("mrr", "separationZ"), headline="mrr")
    items_a_k2 = [mrr_and_sepz_item(f"q{i:02d}", mrr_a[i], 2.0 + 0.01 * i) for i in range(n)]
    items_b_k2 = [mrr_and_sepz_item(f"q{i:02d}", mrr_b[i], 0.01 * i) for i in range(n)]
    agg_a_k2 = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=sum(mrr_a) / n, n=n, support=(0.0, 1.0)),
        separationZ=DistributionSummary(
            name="separationZ", median=2.05, p10=2.0, n=n, unit="query", support=None
        ),
    )
    agg_b_k2 = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=sum(mrr_b) / n, n=n, support=(0.0, 1.0)),
        separationZ=DistributionSummary(
            name="separationZ", median=0.05, p10=0.0, n=n, unit="query", support=None
        ),
    )
    a_k2 = run("cand", role="embedder", call_surface="embeddings", items=items_a_k2,
               aggregates=agg_a_k2,
               fingerprint_fields=embeddings_fields(packId=pack_k2.packId, modelKey="cand"))
    b_k2 = run("bm25", role="embedder", arm_kind="deterministic", items=items_b_k2,
               aggregates=agg_b_k2,
               fingerprint_fields=deterministic_fields(packId=pack_k2.packId))
    lo2, hi2 = mrr_ci(compare_report([a_k2, b_k2], pack=pack_k2))

    assert pack_k1.seed == pack_k2.seed, "the seeds must match for the resample to be comparable"
    width1, width2 = hi1 - lo1, hi2 - lo2
    assert width2 > width1, (
        f"k=2 family's mrr interval [{lo2:+.3f}, {hi2:+.3f}] (width {width2:.4f}) is not wider "
        f"than the k=1 interval [{lo1:+.3f}, {hi1:+.3f}] (width {width1:.4f}) with the same seed "
        "and identical per-unit mrr differences — continuous_verdict() is not being handed the "
        "whole pre-registered family, so the k>1 Bonferroni correction is not reaching the "
        "interval (-ml §3.4 Rule 8, §11.2.2)"
    )


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


@pytest.mark.parametrize("count", [0, 1])
def test_the_negative_control_banner_needs_the_two_arms_its_sentence_describes(count) -> None:
    """P4-1's gate is `>= 2`, because that is what the banner's own sentence asserts.

    *"Both arms are the same stored record"* has no subject at zero arms and is false at one, so
    the gate belongs at the number the sentence needs rather than at the number `_select_arms`
    happens to produce. Relaxing it to `>= 1` survives every CLI test, because `_select_arms`
    returns `[]` or `[r, r]` and never one — but `compare_report` is the public seam S2 wires
    against, and the cross-check above can now remove arms after selection, so the one-arm state
    is a call away rather than a hypothesis.
    """
    md = compare_report(_nested_arms()[:count], pack=guard_pack(headline=METRIC),
                        negative_control=True)
    assert "**NEGATIVE CONTROL REQUESTED, NOT RUN**" in md
    assert "cannot fail" not in md
    assert "fewer than two arms were selected" in md


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


# --- P4-6 to P4-10: the Arms table, the tally's second half, and two arms of one model ---------


def test_a_metric_declared_with_no_observations_is_rendered_not_dropped() -> None:
    """P4-6 — `and metric.n` in the Arms-table guard does two jobs and was tested in neither.

    Relaxing it to `metric.n >= 0` left the whole suite green, which means no fixture anywhere
    constructed a zero-denominator aggregate — and the same clause is the only thing standing
    between `metric.successes / metric.n` and a `ZeroDivisionError`. Rendered, a two-arm
    comparison in which one arm declares the metric with `n=0` printed a **one-row** Arms table
    with nothing saying the second arm was missing.

    **The drop is not the decision.** The table is the report's descriptive half, and a reader
    cannot tell a silently dropped row from an arm that never declared the metric at all — while
    `0/0` with no rate and no interval is a true statement that distinguishes them. It is the same
    call `_POOLED_FOOTNOTE` already makes one column over: *the count itself is never suppressed;
    only the precision claim is.*
    """
    fields = model_fields(packId=PACK_ID)
    scored = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(10)]
    unscored = [
        item(f"g{i:02d}", correct=False, metric=METRIC, scoreable=False) for i in range(10)
    ]
    a = run("cand", items=scored, aggregates=_agg_from(scored),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=unscored, aggregates=_agg_from(unscored),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    assert b.aggregates.named_metrics()[0].n == 0    # the fixture really is the zero-denominator
    assert "| incumbent | falseAdvanceRate | 0/0 | — | — (no observations) |" in md
    assert "| cand | falseAdvanceRate | 10/10 |" in md


def test_the_pairing_tally_counts_the_rows_only_the_second_arm_carried() -> None:
    """P4-7 — the tally's arm-B-only half was entirely untested.

    `only_in_b = 0` and `considered = len(a_keys)` both survived the suite, while the printed
    labels and the other four counters were all pinned. Asymmetric coverage caused by the
    **second** arm is exactly what §4.3 rule 2's tally exists to surface, and with `only_in_b`
    hardcoded to zero the denominator silently under-reports the rows the comparison never saw.
    """
    fields = model_fields(packId=PACK_ID)
    a_items = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(10)]
    b_items = [item(f"g{i:02d}", correct=i >= 4, metric=METRIC) for i in range(12)]
    a = run("cand", items=a_items, aggregates=_agg_from(a_items),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=b_items, aggregates=_agg_from(b_items),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    assert "paired n: 10 of 12 items" in md      # the denominator is the union, not arm A's keys
    assert "0 present in cand only, 2 in incumbent only" in md


def test_the_marginal_overlap_line_renders_the_diagnostic_it_was_given() -> None:
    """P4-8's rendered half — the existing assertion checked only that the label is present.

    Inverting the printed `yes`/`no` left the suite green, so a declared FR-15 output could say
    the opposite of the truth. The line is asserted whole, in both directions.
    """
    pack = guard_pack(headline=METRIC, verdicts=(METRIC,))
    overlapping = compare_report(_nested_arms(), pack=pack)
    assert "- marginal Wilson intervals overlap: yes" in overlapping

    fields = model_fields(packId=PACK_ID)
    a_items = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(40)]
    b_items = [item(f"g{i:02d}", correct=i >= 20, metric=METRIC) for i in range(40)]
    a = run("cand", items=a_items, aggregates=_agg_from(a_items),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", items=b_items, aggregates=_agg_from(b_items),
            fingerprint_fields={**fields, "modelKey": "incumbent"})
    disjoint = compare_report([a, b], pack=pack)
    assert "- marginal Wilson intervals overlap: no" in disjoint


def test_two_arms_of_the_same_model_are_told_apart_everywhere_they_are_named() -> None:
    """P4-10 — plan §5 test 19a is *two independent runs of one model*, and the report could not
    render it.

    Rendered on two runs of one `modelKey` in different sessions, the Arms table had two identical
    `arm` cells, the §4.3 tally read *"0 scoreable for qwen/qwen3-4b-2507 only, 0 scoreable for
    qwen/qwen3-4b-2507 only"*, and a significant verdict would have read *"X is better than X"*.
    `runId` and `sessionId` were never printed anywhere. §5 test 19a is called the highest-value
    single test in the harness, so the report being unreadable for it is the one comparison the
    value claim rests on.

    The suffix is only added where it is needed: an unambiguous comparison must not grow noise.
    """
    fields = model_fields(packId=PACK_ID, modelKey="qwen/qwen3-4b-2507")
    a_items = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(40)]
    b_items = [item(f"g{i:02d}", correct=i >= 6, metric=METRIC) for i in range(40)]
    a = run("morning", items=a_items, aggregates=_agg_from(a_items),
            session_id="s-morning", fingerprint_fields=fields)
    b = run("evening", items=b_items, aggregates=_agg_from(b_items),
            session_id="s-evening", fingerprint_fields=fields)
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    assert "qwen/qwen3-4b-2507 (session s-morning)" in md
    assert "qwen/qwen3-4b-2507 (session s-evening)" in md
    # the verdict sentence must not read "X is better than X"
    assert "qwen/qwen3-4b-2507 is better than qwen/qwen3-4b-2507" not in md
    assert (
        "qwen/qwen3-4b-2507 (session s-morning) is better than "
        "qwen/qwen3-4b-2507 (session s-evening)"
    ) in md
    # ...and the tally's two halves name different arms
    assert "0 scoreable for qwen/qwen3-4b-2507 (session s-morning) only" in md
    assert "0 scoreable for qwen/qwen3-4b-2507 (session s-evening) only" in md


def test_two_arms_sharing_a_model_and_a_session_fall_back_to_the_run_id() -> None:
    """The suffix has to *distinguish*, not merely be present: two runs of one model inside one
    session share their `sessionId`, so printing it twice would leave the two arms as identical as
    the bare model key did. `runId` is unique by construction — it is the record's filename."""
    fields = model_fields(packId=PACK_ID, modelKey="qwen/qwen3-4b-2507")
    a_items = [item(f"g{i:02d}", correct=True, metric=METRIC) for i in range(40)]
    b_items = [item(f"g{i:02d}", correct=i >= 6, metric=METRIC) for i in range(40)]
    a = run("first", items=a_items, aggregates=_agg_from(a_items), session_id="s1",
            fingerprint_fields=fields)
    b = run("second", items=b_items, aggregates=_agg_from(b_items), session_id="s1",
            fingerprint_fields=fields)
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "qwen/qwen3-4b-2507 (run first)" in md
    assert "qwen/qwen3-4b-2507 (run second)" in md


def test_an_unambiguous_comparison_carries_no_disambiguating_suffix() -> None:
    """P4-10's other side — the label must stay the bare model key where it already identifies."""
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "| cand | falseAdvanceRate |" in md
    assert "(session " not in md and "(run " not in md
    assert "cand is better than incumbent" in md


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
            ItemResult(
                itemId=f"g{i:02d}", pairingKey=(f"g{i:02d}",),
                outcome="pass" if ok else "fail", scoreable={METRIC: True},
                counts={METRIC: 1 if ok else 0},
                timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None), detail={},
            )
            for i, ok in enumerate(oks)
        ]
        return run(name, items=items,
                   aggregates=classification_aggregates(correct, len(oks)), basis="assumed",
                   fingerprint_fields=model_fields(modelKey=name, packId=PACK_ID))
    return [arm("cand", a_ok, sum(a_ok)), arm("incumbent", b_ok, sum(b_ok))]


def test_the_decided_by_bullet_names_which_arm_bound_each_bound_and_quotes_no_seed() -> None:
    """§4 S1e Table D / `-ml` v1.11 §3.4 Rule 4 — what replaces the seed parenthetical.

    P3-5's three tests here pinned the retired instrument token, its `(seed N, from the pack's
    `sampling.seed`)` parenthetical, and the binding of that seed to the number it explained.
    Nothing on the paired binary path resamples now, so the seed is provenance for nothing on it and
    the parenthetical is false rather than merely redundant; what the bullet owes instead is the
    audit Rule 4 publishes — **which arm bound each bound**, deterministic once the resample is
    gone. `PackRef.seed` itself stays, its consumer moved to `-ml` §3.2d's continuous bootstrap.

    Both halves are asserted. Naming one arm of a two-arm interval was the defect the token rename
    closes, so the bullet must name *both* arms and say which bound each took; and the seed must
    be absent from the whole rendered page, not merely from this line — a residual over the
    retired parameter counts source lines and cannot see a seed that reaches the reader.
    """
    pack = guard_pack(headline=METRIC, verdicts=(METRIC,))._replace(seed=4242)
    a_ok = [True] * 40
    b_ok = [i >= 6 for i in range(40)]
    md = compare_report(_seed_arms(a_ok, b_ok), pack=pack)

    assert "- decided by: conservative envelope (lower bound: MOVER-D; upper bound: MOVER-D)" in md
    assert "4242" not in md
    assert "seed" not in md


def test_the_decided_by_bullet_names_the_bootstrap_arm_with_the_level_it_was_taken_at() -> None:
    """The other rendering of the same bullet — the exact arm carries its level.

    `(a=4, b=5, c=3, d=0)` is the table where the two arms disagree about which is conservative:
    MOVER-D binds the lower bound at -27.1 pp and the exact paired bootstrap the upper at 58.3 pp,
    so this fixture renders both arms in one line and is the one that would catch a bullet
    hard-coding either name.
    """
    a_ok = [True] * 5 + [False] * 3 + [True] * 4
    b_ok = [False] * 5 + [True] * 3 + [True] * 4
    md = compare_report(_seed_arms(a_ok, b_ok),
                        pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert (
        "- decided by: conservative envelope "
        "(lower bound: MOVER-D; upper bound: exact paired bootstrap, p=0.975)"
    ) in md
    assert "[-27.1, 58.3] pp" in md


def test_the_decided_by_bullet_names_mcnemar_exact_where_one_instrument_decided() -> None:
    """Review P8-2 — the **third** rendering of the same bullet, and the one nothing asserted.

    `_decided_by_line` has exactly three outputs over its domain, discriminated by
    `Verdict.bound_by`: both arms MOVER-D, the two arms split, and `None` — the `mcnemar-exact`
    path, where one instrument produced the whole interval and there is nothing to attribute. The
    first two are pinned by the two tests above. The third's only positive assertion lived in
    `test_the_seed_is_not_printed_where_no_bootstrap_decided_anything`, retired with the seed
    parenthetical, and moved to neither replacement: mutating the branch to
    `return "- decided by: MUTANT"` left the whole suite green.

    It is not an edge case. This is the branch **every** `by-construction` comparison at DEFF 1.00
    takes — the tool-caller pack's own path — so the untested rendering was the one most readers
    would see. `_nested_arms()` is that fixture by default, and the assertion is exact rather than
    a substring so a parenthetical growing back on this branch fails here too.
    """
    md = compare_report(_nested_arms(), pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert "- decided by: mcnemar-exact\n" in md
    assert "conservative envelope" not in md


def test_the_decided_by_bullet_names_a_support_bound_with_its_boundary_value_and_no_level() -> None:
    """`-ml` §3.4 Rule 4a, assertion 10 — the **fourth** rendering of the bullet, and the one
    that closes impl-gate P8-5's finding as collateral.

    `(a=0, b=0, c=38, d=2)` at n=40, DEFF 1.5 is the note's separating case: the composed
    unclamped lower bound is `-1.01124`, outside the support, so the printed `-1.0` is a boundary
    the `√DEFF` widening pushed past the parameter space — not either arm's own bound, and not the
    exact paired bootstrap either, though it is the more negative of the two unclamped arms and
    `P8-1`'s own suggested fix would have named it. A `support bound` token carries no `p=`
    clause, because no level produced it, and renders instead with the support's own boundary
    value: `support bound (-1)`.
    """
    a_ok = [False] * 40
    b_ok = [True] * 38 + [False] * 2
    a, b = _seed_arms(a_ok, b_ok)
    a = run("cand", items=list(a.items), aggregates=a.aggregates, design_effect=1.5,
            basis="measured", fingerprint_fields=model_fields(modelKey="cand", packId=PACK_ID))
    b = run("incumbent", items=list(b.items), aggregates=b.aggregates, design_effect=1.5,
            basis="measured",
            fingerprint_fields=model_fields(modelKey="incumbent", packId=PACK_ID))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))
    assert (
        "- decided by: conservative envelope "
        "(lower bound: support bound (-1); upper bound: MOVER-D)"
    ) in md
    # The false sentence is not confined to the not-distinguishable path (`-ml` §3.4 Rule 4a) —
    # this is a *published, positive* verdict, which is the fact that makes the defect urgent.
    assert "incumbent is better than cand" in md


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


# --------------------------------------------------------------------------------------------
# `Basis` — the `-ml` §7.1 vocabulary, declared in three places (impl review Pass 16, P16-1)
# --------------------------------------------------------------------------------------------
#
# `basis` is what Rule 4 turns on — whether the design effect was *established* by the pairing,
# *measured* from the data, or *assumed* — so it decides which instrument may decide a verdict.
# It was written out three times (`stats.Basis`, `results.Basis`, `report._BASIS_STRENGTH`) with
# nothing binding any pair, and Python does not enforce a `Literal` at runtime: **deleting
# `"measured"` from `stats.Basis` alone left the whole suite green** (Pass 16 §6 P16-1, Appendix
# P.2). Two closures were taken, because the two duplications are different in kind:
#
#  1. `results.Basis` is now an *import* of `stats.Basis` — one home, the way `results.py`
#     already imports `stats.percentile` rather than keeping a second percentile. A second
#     `Literal` on an import edge that already exists is a copy waiting to drift, not a
#     declaration.
#  2. `_BASIS_STRENGTH` cannot be collapsed — it is a *ranking* over the same domain, and its
#     keys are the domain. So it is bound below, and both surviving declarations are bound to a
#     literal transcribed from the note, never to each other: two sets authored in one unit agree
#     by construction (`AGENTS.md`, "A guard's reach lives in an asserted constant"; the
#     `TURN_DISPOSITIONS` probe in `test_convo.py` is the same shape).

#: Transcribed by hand from `docs/plans/small-model-benchmarking-ml.md` §7.1's `ResolvingPower`
#: declaration (v1.17, line `basis: Literal["by-construction", "measured", "assumed"]`). The
#: independent declaration the module's two are each bound to; never derive it from either.
_BASIS_PER_ML_NOTE = {"by-construction", "measured", "assumed"}


def test_the_basis_literal_is_exactly_the_ml_notes_vocabulary() -> None:
    """Leg 1: the type every `basis` field and parameter is annotated with, against the note.

    `results.Basis` is the same object by construction now (it imports it), which is the point —
    the assertion below is what refuses a re-declaration that has drifted by one member.
    """
    from typing import get_args

    from modelbench import results

    assert set(get_args(stats.Basis)) == _BASIS_PER_ML_NOTE
    assert set(get_args(results.Basis)) == _BASIS_PER_ML_NOTE


def test_the_basis_strength_ranking_covers_exactly_the_declared_bases() -> None:
    """Leg 2: `_BASIS_STRENGTH`'s key set, against the same transcript.

    `_comparison_pair`'s consumer reads it as `min(..., key=_BASIS_STRENGTH.__getitem__)`, so a
    basis the ranking does not carry is a `KeyError` in the report path — and a *rank* for a basis
    that no longer exists is an entry nothing can ever select. Both are the same edit here.
    """
    from modelbench.report import _BASIS_STRENGTH

    assert set(_BASIS_STRENGTH) == _BASIS_PER_ML_NOTE


def test_the_basis_ranking_orders_weakest_first_so_a_pair_takes_the_weaker_claim() -> None:
    """The ranking's *values*, driven through the consumer: a comparison's basis is the weaker of
    its two arms', so `assumed` must lose to `measured` and `measured` to `by-construction`.

    Asserted through `min(..., key=...)` — the operation `_comparison_pair`'s caller performs —
    rather than against the integers, which would assert the table against itself.
    """
    from modelbench.report import _BASIS_STRENGTH

    def weaker(a: str, b: str) -> str:
        return min((a, b), key=_BASIS_STRENGTH.__getitem__)

    assert weaker("assumed", "measured") == "assumed"
    assert weaker("measured", "by-construction") == "measured"
    assert weaker("assumed", "by-construction") == "assumed"


# --------------------------------------------------------------------------------------------
# `_NO_VERDICT_REASON` — the causes, bound to the function that produces them (Pass 16, P16-3)
# --------------------------------------------------------------------------------------------


def _det_arm(name: str, correct: int = 20):
    return run(
        name,
        arm_kind="deterministic",
        items=[item(f"g{i:02d}", correct=i < correct, metric=METRIC) for i in range(40)],
        aggregates=classification_aggregates(correct, 40),
        fingerprint_fields=deterministic_fields(
            packId=PACK_ID, armId=name, armParametersHash="b" * 64
        ),
    )


def test_the_no_verdict_reasons_are_exactly_the_causes_comparison_pair_returns() -> None:
    """`_NO_VERDICT_REASON` is looked up as `_NO_VERDICT_REASON[pair]` on every cause
    `_comparison_pair` answers with, so the two are one closed set declared twice — and nothing
    bound them (impl review Pass 16, P16-3). A cause with no reason is a `KeyError` mid-render;
    a reason with no cause is a paragraph nothing can ever print.

    The grid is exhaustive over the two dimensions `_comparison_pair` branches on — how many
    arms it was handed (0, 1, 2) and what kind each is — so it is the function's own domain, not
    a sample of it. A third cause introduced on a *new* dimension is still owed a case here, and
    the equality is what says so out loud.
    """
    from modelbench.report import _NO_VERDICT_REASON, _comparison_pair

    model, det = _arm("cand", 34), _det_arm("bm25")
    grid = [
        [],
        [model],
        [det],
        [model, _arm("other", 30)],
        [model, det],
        [det, model],
        [det, _det_arm("bm25-tuned", 30)],
    ]
    causes = {r for r in (_comparison_pair(runs) for runs in grid) if isinstance(r, str)}
    assert causes == set(_NO_VERDICT_REASON)


def test_each_no_verdict_cause_prints_its_own_reason_and_not_the_other_one() -> None:
    """The per-cause consequence behind the domain equality: one explanation serving two causes
    is what let a one-arm comparison assert a deterministic-arm reason that was untrue (review
    M-6), so each cause is driven end-to-end and the *other* cause's sentence is asserted absent.
    """
    from modelbench.report import _NO_VERDICT_REASON

    pack = guard_pack(headline=METRIC, verdicts=(METRIC,))
    rendered = {
        "too-few-arms": compare_report([_arm("cand", 34)], pack=pack),
        "both-deterministic": compare_report(
            [_det_arm("bm25"), _det_arm("bm25-tuned", 30)], pack=pack
        ),
    }
    assert set(rendered) == set(_NO_VERDICT_REASON), "a cause lost its end-to-end case"
    for cause, md in rendered.items():
        assert _NO_VERDICT_REASON[cause] in md
        for other, text in _NO_VERDICT_REASON.items():
            if other != cause:
                assert text not in md


# ==================================================================================================
# S5 spec §4.4 / §5 Step 6 — the three tool-caller renderers (funnel, per-turn-position, hazard)
# and E5 (the dispatch-failure count surviving storage and reload)
# ==================================================================================================


def _funnel_counts(**overrides) -> FunnelCounts:
    """Every one of the sixteen fields distinct and non-zero by default, so a mutant that
    dropped or transposed one would not coincidentally still match (mirrors `test_results.py`'s
    own `FunnelCounts` round-trip fixture). The last three
    (`argsOmittedRequired`/`argsWrongValue`/`argsBoundaryUnit`) are the 2026-09-14 correction's own
    addition (review `small-model-benchmarking-s5.md` Finding 3, option (a)) — kept distinct from
    every other field's value so a mutant that dropped or transposed one of the three is caught the
    same way the pre-existing thirteen are."""
    base = dict(
        turnsDriven=360,
        unrunnableModelChannel=13,
        unrunnableToolChannel=7,
        turnsScoredAfterUnrunnable=5,
        restraintTurns=40,
        requiredCallTurns=320,
        nativeCallEmitted=142,
        prosePseudoCall=31,
        noAttempt=147,
        turnsWithAnyCall=142,
        dispatchedCalls=167,
        factBearingReturns=118,
        unscoreableReturns=24,
        argsOmittedRequired=9,
        argsWrongValue=12,
        argsBoundaryUnit=6,
    )
    base.update(overrides)
    return FunnelCounts(**base)


def _toolcaller_run(run_id: str, **kwargs) -> RunResult:
    return run(
        run_id,
        role="tool-caller",
        fingerprint_fields=model_fields(
            packId="tool-caller-shop-assistant",
            packVersion="1.0.0",
            packContentHash="a" * 64,
        ),
        **kwargs,
    )


# --- `_render_funnel` -----------------------------------------------------------------------------


def test_render_funnel_prints_the_full_hierarchy_with_real_numbers() -> None:
    """`-ml` §4.3 rule 3's illustrated shape (plan `ml.md:2098-2111`) — every named line present,
    with the fixture's own distinct numbers, so a mutant that dropped a line or mixed up two
    fields would fail here rather than being papered over by a repeated value.

    2026-09-14 correction (review Finding 3, option (a)): also asserts the two new
    `argsOmittedRequired`/`argsWrongValue` lines (`argsBoundaryUnit` is the "of which" annotation
    on the second one, never its own top-level line) and the `allArgsCorrect`-derived denominator
    cross-reference, read off `run.aggregates.funnel` rather than a second, derivable copy stored
    on `FunnelCounts` itself (§7 rule 4)."""
    aggregates = ToolCallAggregates(
        funnelCounts=_funnel_counts(),
        funnel=(BinaryMetric(name="allArgsCorrect", successes=130, n=150, unit="call"),),
    )
    r = _toolcaller_run("cand", aggregates=aggregates)
    lines = _render_funnel(r, "cand")
    text = "\n".join(lines)
    for label, value in (
        ("turns driven", 360),
        ("unrunnable (model channel)", 13),
        ("turns scored after unrunnable", 5),
        ("unrunnable (tool channel)", 7),
        ("restraint turns", 40),
        ("R(t) >= 1", 320),
        ("native call emitted", 142),
        ("prose pseudo-call", 31),
        ("no attempt", 147),
        ("turns with", 142),
        ("dispatched calls", 167),
        ("fact-bearing returns", 118),
        ("unscoreable returns", 24),
        ("args omitted required", 9),
        ("args wrong value", 12),
    ):
        assert any(label in ln and str(value) in ln for ln in lines), (label, value, text)

    # The `allArgsCorrect` denominator annotation on the "args omitted required" line.
    assert any(
        "args omitted required" in ln and "150" in ln and "correct tool" in ln for ln in lines
    ), text
    # The "of which boundary/unit: N" annotation on the "args wrong value" line.
    assert any(
        "args wrong value" in ln and "of which boundary/unit: 6" in ln for ln in lines
    ), text


def test_render_funnel_cross_references_restraint_rate_without_a_second_stored_copy() -> None:
    """§7 rule 4: the restraint "-> k/n" annotation is computed at render time from
    `ToolCallAggregates.restraint`, never a second, derivable integer stored on `FunnelCounts`
    itself (confirmed structurally: `FunnelCounts` carries no restraint-rate field at all)."""
    aggregates = ToolCallAggregates(
        funnelCounts=_funnel_counts(restraintTurns=40),
        restraint=BinaryMetric(name="restraint", successes=38, n=40, unit="turn"),
    )
    r = _toolcaller_run("cand", aggregates=aggregates)
    lines = _render_funnel(r, "cand")
    assert any("38/40" in ln for ln in lines)
    assert not hasattr(FunnelCounts, "restraintRate")


def test_render_funnel_is_empty_when_funnelCounts_is_none() -> None:
    """A `tool-caller` run with no `funnelCounts` (every pre-S5-Step-6 fixture in this file's own
    `_toolcall_arm`) renders no funnel block at all — never a table of zeros."""
    aggregates = ToolCallAggregates(
        restraint=BinaryMetric(name="restraint", successes=1, n=1, unit="turn")
    )
    r = _toolcaller_run("cand", aggregates=aggregates)
    assert _render_funnel(r, "cand") == []


def test_render_funnel_prints_iteration_summary_with_distinct_i_t_and_y_calls_over_y() -> None:
    """§4.4 item 4: the `I(t)` mean/p95 (restricted to replied/cap-hit turns) and the unrestricted
    `Y_calls / Y` ratio, plus the distinctness sentence, all render off `ToolCallAggregates.
    iterationSummary` — with `mean` and `yCalls/y` deliberately DIFFERENT values, so the
    distinctness claim is actually exercised rather than passing on two fields that happen to
    coincide (coordinator finding: a prior version of this suite never populated `iterationSummary`
    on any `test_report.py` fixture at all, so this whole block had zero coverage there)."""
    summary = {
        "n": 5,
        "capHitCount": 0,
        "mean": 3.25,
        "meanCensored": False,
        "p95": 7.5,
        "p95Censored": False,
        "yCalls": 20,
        "y": 8,
    }
    assert summary["mean"] != summary["yCalls"] / summary["y"]  # the fixture itself is distinct
    aggregates = ToolCallAggregates(funnelCounts=_funnel_counts(), iterationSummary=summary)
    r = _toolcaller_run("cand", aggregates=aggregates)
    text = "\n".join(_render_funnel(r, "cand"))
    assert "mean 3.25" in text
    assert "p95 7.50" in text
    assert "20/8" in text  # the unrestricted Y_calls/Y count, printed as a k/n
    assert "2.50" in text  # 20/8, the unrestricted rate itself
    assert "Different statistics" in text


def test_render_funnel_iteration_summary_censored_prefix_is_independent_per_field() -> None:
    """The `>= ` censoring prefix on `mean` and on `p95` is driven by two INDEPENDENT booleans
    (`meanCensored`/`p95Censored`) — never one flag governing both. Asserted with one true and the
    other false, in BOTH directions, mirroring the U142 lesson: a mutant that substituted one
    field's flag for the other's would pass a test that only ever set both flags together."""
    base = {"n": 3, "capHitCount": 1, "mean": 4.0, "p95": 6.0, "yCalls": 9, "y": 3}

    only_mean_censored = {**base, "meanCensored": True, "p95Censored": False}
    aggregates_a = ToolCallAggregates(
        funnelCounts=_funnel_counts(), iterationSummary=only_mean_censored
    )
    text_a = "\n".join(_render_funnel(_toolcaller_run("cand", aggregates=aggregates_a), "cand"))
    assert "mean >= 4.00" in text_a
    assert "p95 >=" not in text_a
    assert "p95 6.00" in text_a

    only_p95_censored = {**base, "meanCensored": False, "p95Censored": True}
    aggregates_b = ToolCallAggregates(
        funnelCounts=_funnel_counts(), iterationSummary=only_p95_censored
    )
    text_b = "\n".join(_render_funnel(_toolcaller_run("cand", aggregates=aggregates_b), "cand"))
    assert "mean 4.00" in text_b
    assert "mean >=" not in text_b
    assert "p95 >= 6.00" in text_b


def test_render_funnel_prints_the_prose_detector_precision_recall_when_populated() -> None:
    """S6 spec §2.5, §5 Step 1: one new line beside the funnel table's existing (a)+(b) partition
    line, printing the detector's own precision/recall figure when `ToolCallAggregates.
    prosePseudoCallDetector` is populated — `precision`/`recall` deliberately DISTINCT values so a
    mutant that printed one where the other belongs would be caught."""
    aggregates = ToolCallAggregates(
        funnelCounts=_funnel_counts(),
        prosePseudoCallDetector={"n": 20, "precision": 0.875, "recall": 0.625},
    )
    text = "\n".join(_render_funnel(_toolcaller_run("cand", aggregates=aggregates), "cand"))
    assert "prose-pseudo-call detector" in text
    assert "precision 0.875" in text
    assert "recall 0.625" in text
    assert "n=20" in text


def test_render_funnel_names_the_prose_detectors_absence_when_none() -> None:
    """The `None` case — no calibration corpus declared — must still print a line naming that
    absence, never silently omit it (S6 spec §5 Step 1's own two literal message shapes)."""
    aggregates = ToolCallAggregates(funnelCounts=_funnel_counts(), prosePseudoCallDetector=None)
    text = "\n".join(_render_funnel(_toolcaller_run("cand", aggregates=aggregates), "cand"))
    assert "prose-pseudo-call detector" in text
    assert "no calibration corpus declared" in text
    assert "unmeasured" in text


def test_funnel_table_opens_the_report_before_the_arms_section() -> None:
    """Rule 3, verbatim: "the report opens with a funnel table, not a metric table" — the funnel
    text must appear strictly before the `## Arms` heading in the full rendered document."""
    aggregates = ToolCallAggregates(funnelCounts=_funnel_counts())
    r = _toolcaller_run("cand", aggregates=aggregates)
    md = compare_report([r], pack=_toolcall_pack())
    funnel_pos = md.index("turns driven")
    arms_pos = md.index("## Arms")
    assert funnel_pos < arms_pos


def test_e5_dispatch_failure_count_survives_storage_and_prints_on_reload(tmp_root) -> None:
    """E5 (dispatch-failure note §5's evaluation table): a stored run carrying a populated
    `funnelCounts.unrunnableToolChannel > 0`, re-read by `compare`, prints the per-arm
    dispatch-failure count — the whole point of §2.3's resolution (no new top-level `RunResult`
    field; the funnel line IS the disclosure line)."""
    aggregates = ToolCallAggregates(funnelCounts=_funnel_counts(unrunnableToolChannel=6))
    written = _toolcaller_run("r-e5", aggregates=aggregates)
    store(written, tmp_root)
    valid, invalid = load_history(tmp_root, packId="tool-caller-shop-assistant")
    assert invalid == []
    assert len(valid) == 1
    md = compare_report(valid, pack=_toolcall_pack())
    tool_channel_line = next(ln for ln in md.splitlines() if "unrunnable (tool channel)" in ln)
    assert "6" in tool_channel_line


# --- `_render_per_turn_position` ------------------------------------------------------------------


def _hazard_pair(*, n_at_0: int, f_at_0: int, c_at_0: int = 0) -> tuple[HazardPoint, ...]:
    return (
        HazardPoint(
            turnIndex=0,
            metric=BinaryMetric(name="hazard", successes=f_at_0, n=n_at_0, unit="conversation"),
            censored=c_at_0,
        ),
    )


def test_render_per_turn_position_one_column_per_arm_with_observed_and_structural_n() -> None:
    """§4.4 item 2: `n` is the OBSERVED count (already censoring-aware); the STRUCTURAL n (the
    run's own total scored-conversation count) is printed beside it — computed from `len(run.
    items)`, never hardcoded, since `PackRef` carries no per-script-length distribution for
    `report.py` to consult (S5 spec §4.4 item 2's own open wiring, this module's synthesis)."""
    position = (
        TurnPositionRate(
            turnIndex=0, metric=BinaryMetric(name="hazard", successes=1, n=12, unit="conversation")
        ),
    )
    items = [item(f"S-{i:02d}", correct=True, metric="cleanThroughTurn4") for i in range(12)]
    a = _toolcaller_run(
        "cand", items=items, aggregates=ToolCallAggregates(perTurnPosition=position)
    )
    b = _toolcaller_run(
        "incumbent", items=items, aggregates=ToolCallAggregates(perTurnPosition=position)
    )
    lines = _render_per_turn_position(_toolcall_pack(), [a, b])
    text = "\n".join(lines)
    assert "1/12" in text
    assert text.count("1/12") >= 2  # one occurrence per arm's own column
    assert "12" in text  # the structural ceiling (len(items) == 12)


def test_render_per_turn_position_marks_a_low_observed_n_as_descriptive() -> None:
    """§4.4 item 2's own literal text: every position with observed `n < 10` is marked
    "descriptive at this n — no significance claim"."""
    position = (
        TurnPositionRate(
            turnIndex=0, metric=BinaryMetric(name="hazard", successes=1, n=3, unit="conversation")
        ),
    )
    items = [item(f"S-{i:02d}", correct=True, metric="cleanThroughTurn4") for i in range(3)]
    a = _toolcaller_run(
        "cand", items=items, aggregates=ToolCallAggregates(perTurnPosition=position)
    )
    lines = _render_per_turn_position(_toolcall_pack(), [a])
    text = "\n".join(lines)
    assert "descriptive at this n — no significance claim" in text


def test_render_per_turn_position_is_empty_when_no_run_carries_position_data() -> None:
    a = _toolcaller_run("cand", aggregates=ToolCallAggregates())
    assert _render_per_turn_position(_toolcall_pack(), [a]) == []


def test_per_turn_position_table_appears_after_the_arms_section() -> None:
    position = (
        TurnPositionRate(
            turnIndex=0, metric=BinaryMetric(name="hazard", successes=1, n=12, unit="conversation")
        ),
    )
    r = _toolcaller_run("cand", aggregates=ToolCallAggregates(perTurnPosition=position))
    md = compare_report([r], pack=_toolcall_pack())
    arms_pos = md.index("## Arms")
    position_pos = md.index("Per-turn position")
    assert arms_pos < position_pos


# --- `_render_hazard` -------------------------------------------------------------------------


def test_render_hazard_prints_both_arms_side_by_side_with_their_own_censored_column() -> None:
    """§4.4 item 3: both arms' curves side by side, each with its OWN `c_t` column reading
    `HazardPoint.censored` directly — never a recomputed rate."""
    a = _toolcaller_run(
        "cand", aggregates=ToolCallAggregates(hazard=_hazard_pair(n_at_0=8, f_at_0=2, c_at_0=1))
    )
    b = _toolcaller_run(
        "incumbent",
        aggregates=ToolCallAggregates(hazard=_hazard_pair(n_at_0=6, f_at_0=1, c_at_0=3)),
    )
    lines = _render_hazard([a, b])
    text = "\n".join(lines)
    assert "2" in text and "8" in text and "1" in text  # cand's f_0, r_0, c_0
    assert "6" in text and "3" in text  # incumbent's r_0, c_0


def test_render_hazard_never_computes_a_cross_arm_difference() -> None:
    """Load-bearing prohibition (§4.4 item 3, `-ml` §4.3 rule 5's closing clause): the two curves
    are conditioned on different, arm-specific risk sets after censoring, so no path may compute
    or print `a_rate - b_rate`. `cand`'s rate at t=0 is 1/2 = 0.5; `incumbent`'s is 1/4 = 0.25 —
    a naive cross-arm difference would be exactly `0.25` / `25.0` (pp). Neither string may appear
    anywhere in the rendered hazard block."""
    a = _toolcaller_run(
        "cand", aggregates=ToolCallAggregates(hazard=_hazard_pair(n_at_0=2, f_at_0=1))
    )
    b = _toolcaller_run(
        "incumbent", aggregates=ToolCallAggregates(hazard=_hazard_pair(n_at_0=4, f_at_0=1))
    )
    lines = _render_hazard([a, b])
    text = "\n".join(lines)
    assert "0.25" not in text
    assert "25.0" not in text
    assert "0.250" not in text


def test_render_hazard_is_empty_when_no_run_carries_hazard_data() -> None:
    a = _toolcaller_run("cand", aggregates=ToolCallAggregates())
    assert _render_hazard([a]) == []


def test_hazard_table_appears_after_the_per_turn_position_table() -> None:
    position = (
        TurnPositionRate(
            turnIndex=0, metric=BinaryMetric(name="hazard", successes=1, n=12, unit="conversation")
        ),
    )
    r = _toolcaller_run(
        "cand",
        aggregates=ToolCallAggregates(
            perTurnPosition=position, hazard=_hazard_pair(n_at_0=12, f_at_0=1)
        ),
    )
    md = compare_report([r], pack=_toolcall_pack())
    position_pos = md.index("Per-turn position")
    hazard_pos = md.index("Hazard")
    assert position_pos < hazard_pos


# ==================================================================================================
# S7 spec §3.5 item 1 — `_render_role_caveat`: `chat-responder`'s deterministic layer never
# measures reply quality, stated once in words. `[]` for every other role (structural self-gate,
# `_render_funnel`'s own pattern) — a widen/shrink pair: every ALREADY-ESTABLISHED role fixture in
# this file stays silent (shrink-catcher), and `chat-responder` alone prints (widen-catcher).
# ==================================================================================================


def _chat_responder_pack(
    verdicts: tuple[str, ...] = ("groundingRate",), headline: str | None = "groundingRate"
) -> PackRef:
    return PackRef(
        packId="chat-responder-grounded-answers", packVersion="0.1.0", contentHash="f" * 64,
        role="chat-responder",
        metrics=PackMetrics(verdictMetrics=verdicts, headlineMetric=headline),
        pairingKey=("itemId",), analysisUnit="itemId", seed=20260917,
    )


def _grounding_aggregates(successes: int, n: int) -> GroundingAggregates:
    return GroundingAggregates(
        checklistPass=BinaryMetric(name="groundingRate", successes=successes, n=n, unit="item"),
        perCheck=(), parseFailures=0,
    )


def _chat_responder_arm(run_id: str, correct: int, total: int = 10) -> RunResult:
    items = [item(f"c{i:02d}", correct=i < correct, metric="groundingRate") for i in range(total)]
    return run(
        run_id, role="chat-responder", items=items,
        aggregates=_grounding_aggregates(correct, total),
        fingerprint_fields=model_fields(
            modelKey=run_id, packId="chat-responder-grounded-answers"
        ),
    )


class TestRenderRoleCaveat:
    def test_prints_for_the_chat_responder_role(self) -> None:
        lines = _render_role_caveat(_chat_responder_pack())
        text = "\n".join(lines)
        assert "Reply quality is not measured by this pack" in text

    def test_is_empty_for_guard_judge(self) -> None:
        assert _render_role_caveat(guard_pack(headline=METRIC, verdicts=(METRIC,))) == []

    def test_is_empty_for_tool_caller(self) -> None:
        assert _render_role_caveat(_toolcall_pack()) == []

    def test_is_empty_for_embedder(self) -> None:
        assert _render_role_caveat(_embedder_pack()) == []

    def test_appears_right_after_the_title_before_the_funnel_table(self) -> None:
        a = _chat_responder_arm("cand", 8)
        b = _chat_responder_arm("incumbent", 6)
        md = compare_report([a, b], pack=_chat_responder_pack())
        title_pos = md.index("# Comparison")
        caveat_pos = md.index("Reply quality is not measured")
        arms_pos = md.index("## Arms")
        assert title_pos < caveat_pos < arms_pos


# ==================================================================================================
# S7 spec §3.5 item 2 — `_render_speed`: `RunResult.latency`'s first renderer, generic and
# role-agnostic, retroactively benefiting every role that carries a populated `LatencyBlock`.
# ==================================================================================================


def _latency_block(**overrides) -> LatencyBlock:
    fields = {
        "latencyMsP50": 1200.0, "latencyMsP95": 1800.0, "latencyMsMax": 2200.0,
        "latencyTimedCount": 40, "latencyItemCount": 40, "latencyWithheldForLoad": 0,
        "latencyWithheldForNoResponse": 0, "statsCoveredCount": 40, "callCount": 40,
        "ttftMsMedian": 150.0, "prefillMsPer1kMedian": 80.0, "tokensPerSecondMedian": 45.5,
        "unexplainedMsMax": 5.0,
    }
    fields.update(overrides)
    return LatencyBlock(**fields)


class TestRenderSpeed:
    def test_prints_a_populated_table_for_a_fully_covered_run(self) -> None:
        import dataclasses

        r = dataclasses.replace(_arm("cand", 34), latency=_latency_block())
        lines = _render_speed([r], {r.runId: r.runId})
        text = "\n".join(lines)
        assert "## Speed" in text
        assert "1200" in text  # p50
        assert "1800" in text  # p95
        assert "150" in text  # TTFT median
        assert "80.0" in text  # prefill ms/1k
        assert "45.5" in text  # tokens/sec median

    def test_uses_insufficient_coverage_wording_for_missing_p50_and_ttft(self) -> None:
        import dataclasses

        block = _latency_block(latencyMsP50=None, ttftMsMedian=None)
        r = dataclasses.replace(_arm("cand", 34), latency=block)
        lines = _render_speed([r], {r.runId: r.runId})
        text = "\n".join(lines)
        assert text.count("— (insufficient coverage)") == 2

    def test_p95_falls_back_to_max_labelled_when_p95_is_absent(self) -> None:
        import dataclasses

        block = _latency_block(latencyMsP95=None, latencyMsMax=2500.0)
        r = dataclasses.replace(_arm("cand", 34), latency=block)
        lines = _render_speed([r], {r.runId: r.runId})
        text = "\n".join(lines)
        assert "2500 (max)" in text

    def test_returns_empty_when_every_runs_latency_is_none(self) -> None:
        a = _arm("cand", 34)
        b = _arm("incumbent", 30)
        assert _render_speed([a, b], {a.runId: a.runId, b.runId: b.runId}) == []

    def test_speed_section_appears_after_arms_and_before_per_turn_position(self) -> None:
        import dataclasses

        position = (
            TurnPositionRate(
                turnIndex=0,
                metric=BinaryMetric(name="hazard", successes=1, n=12, unit="conversation"),
            ),
        )
        r = _toolcaller_run("cand", aggregates=ToolCallAggregates(perTurnPosition=position))
        r = dataclasses.replace(r, latency=_latency_block())
        md = compare_report([r], pack=_toolcall_pack())
        arms_pos = md.index("## Arms")
        speed_pos = md.index("## Speed")
        position_pos = md.index("Per-turn position")
        assert arms_pos < speed_pos < position_pos


# ==================================================================================================
# S7 spec §5 Step 1's own regression half: re-run an EXISTING `compare_report` fixture test
# (`test_the_per_arm_intervals_are_labelled_descriptive` / `test_every_rate_prints_with_its_
# denominator`, both over `_nested_arms()` + `guard_pack`) and confirm every previously-asserted
# line is still present and unchanged, with the new "## Speed" section appearing as a pure
# addition — never a replacement of anything already there.
# ==================================================================================================


def test_regression_existing_guard_judge_report_is_unchanged_and_speed_is_a_pure_addition() -> None:
    import dataclasses

    a, b = _nested_arms()
    a = dataclasses.replace(a, latency=_latency_block())
    b = dataclasses.replace(b, latency=_latency_block(latencyMsP50=900.0))
    md = compare_report([a, b], pack=guard_pack(headline=METRIC, verdicts=(METRIC,)))

    # Every assertion these two pre-existing tests made, unchanged:
    assert "descriptive, not the comparison instrument" in md
    assert "40/40" in md and "34/40" in md

    # ...and the new section is present, as an addition.
    assert "## Speed" in md
    assert md.count("## Speed") == 1


# ==================================================================================================
# Unit A (`docs/plans/small-model-catalog-sweep.md` §3.1–§3.2): `rank_report` and its helpers.
# ==================================================================================================


def test_metric_value_reads_a_binary_metrics_rate() -> None:
    a, b = _nested_arms()
    assert _metric_value(a, METRIC) == 1.0
    assert _metric_value(b, METRIC) == 0.85


def test_metric_value_reads_a_continuous_metrics_mean() -> None:
    pack = _embedder_pack()
    items = [_mrr_item(f"q{i:02d}", 0.5) for i in range(10)]
    agg = RetrievalAggregates(mrr=ContinuousMetric(name="mrr", mean=0.5, n=10, support=(0.0, 1.0)))
    a = run("cand", role="embedder", call_surface="embeddings", items=items, aggregates=agg,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    assert _metric_value(a, "mrr") == 0.5


def test_metric_value_is_none_when_the_run_declares_no_aggregate_for_the_metric() -> None:
    a, _b = _nested_arms()
    assert _metric_value(a, "unrelatedMetric") is None


def test_metric_value_raises_on_a_distribution_summary() -> None:
    """§3.1.1 — no shipped pack's headline/verdictMetrics member resolves to a
    `DistributionSummary` today; silently picking median over p10 would be a guess this module
    refuses elsewhere."""
    pack = _embedder_pack(verdicts=(), headline=None)
    agg = RetrievalAggregates(
        separationZ=DistributionSummary(
            name="separationZ", median=1.5, p10=-0.3, n=40, unit="query", support=None
        )
    )
    a = run("cand", role="embedder", call_surface="embeddings", items=[], aggregates=agg,
            fingerprint_fields=embeddings_fields(packId=pack.packId, modelKey="cand"))
    with pytest.raises(PackConfigError):
        _metric_value(a, "separationZ")


def test_better_honors_lower_is_better_and_is_pinned_against_both_mutations(monkeypatch) -> None:
    """AGENTS.md's guard-testing convention — `_LOWER_IS_BETTER`'s reach is bound by mutating the
    constant alone, both ways: shrinking it must make a `falseAdvanceRate` comparison rank
    backwards, and widening it must make an ordinary higher-is-better metric rank backwards too."""
    assert _better("falseAdvanceRate", 0.1, 0.2) is True  # the lower rate wins
    assert _better("groundingRate", 0.9, 0.8) is True  # the higher rate wins

    monkeypatch.setattr(report, "_LOWER_IS_BETTER", frozenset())
    assert _better("falseAdvanceRate", 0.1, 0.2) is False  # shrink: now ranks backwards

    monkeypatch.setattr(
        report, "_LOWER_IS_BETTER", frozenset({"falseAdvanceRate", "groundingRate"})
    )
    assert _better("groundingRate", 0.9, 0.8) is False  # widen: now ranks backwards


def _rank_pack(
    packId: str = "nlq-structured-query",
    role: str = "nlq-generator",
    verdicts: tuple[str, ...] = ("layer1ExactMatchRate",),
    headline: str | None = "layer1ExactMatchRate",
) -> PackRef:
    return PackRef(
        packId=packId, packVersion="1.0.0", contentHash="9" * 64, role=role,
        metrics=PackMetrics(verdictMetrics=verdicts, headlineMetric=headline),
        pairingKey=("itemId",), analysisUnit="itemId", seed=20260902,
    )


def _rank_arm(
    model_key: str,
    correct: int,
    total: int = 40,
    *,
    metric: str = "layer1ExactMatchRate",
    pack_id: str = "nlq-structured-query",
    latency_p95: float | None = None,
) -> RunResult:
    import dataclasses

    items = [item(f"q{i:02d}", correct=i < correct, metric=metric) for i in range(total)]
    r = run(
        model_key, items=items, aggregates=_agg_from(items, metric=metric),
        fingerprint_fields=model_fields(modelKey=model_key, packId=pack_id),
    )
    if latency_p95 is not None:
        r = dataclasses.replace(r, latency=_latency_block(latencyMsP95=latency_p95))
    return r


def test_rank_report_ranks_a_binary_headline_metric_with_ci_latency_and_footprint() -> None:
    """§5 test 3 — 3+ runs, one binary headline metric, no `reference`: rows sorted correctly,
    each with k/n/rate/Wilson CI, the pack's own restated caveat block-quoted, latency populated
    from `RunResult.latency`, footprint reading `—` when omitted."""
    pack = _rank_pack()
    a = _rank_arm("model-a", 40, 40, latency_p95=1500.0)
    b = _rank_arm("model-b", 30, 40, latency_p95=1800.0)
    c = _rank_arm("model-c", 20, 40)

    md = rank_report([a, b, c], pack=pack)

    section = md.split("### layer1ExactMatchRate")[1]
    a_pos, b_pos, c_pos = (section.index(k) for k in ("model-a", "model-b", "model-c"))
    assert a_pos < b_pos < c_pos
    assert "40/40" in section and "1.000" in section
    assert "30/40" in section and "0.750" in section
    assert "1500" in section
    assert "1800" in section
    assert "—" in section  # model-c's missing latency, and every model's missing footprint
    assert "The true denominator is 34, not 40" in md  # the pack's own restated caveat


def test_rank_report_footprint_column_reads_the_supplied_string_when_present() -> None:
    pack = _rank_pack()
    a = _rank_arm("model-a", 40, 40)
    b = _rank_arm("model-b", 30, 40)

    md = rank_report([a, b], pack=pack, footprints={"model-a": "4.2 GB"})

    section = md.split("### layer1ExactMatchRate")[1]
    assert "4.2 GB" in section
    assert "—" in section  # model-b has no footprint entry


def test_rank_report_emits_a_marker_comment_naming_the_top_row() -> None:
    """§3.5/§5 test 12 — the `<!-- rank-report: ... -->` line Unit C depends on."""
    pack = _rank_pack()
    a = _rank_arm("model-a", 40, 40)
    b = _rank_arm("model-b", 30, 40)

    md = rank_report([a, b], pack=pack)

    assert f"<!-- rank-report: pack={pack.packId} metric=layer1ExactMatchRate top=model-a" in md


def test_rank_report_raises_on_a_duplicate_model_key() -> None:
    """§5 test 8 — mirrors `PairedOutcomes.from_units`'s duplicate-unit-id backstop test shape.
    Deduplication is the caller's job (Unit B); `rank_report` never silently keeps one."""
    pack = _rank_pack()
    a = _rank_arm("model-a", 40, 40)
    a_again = _rank_arm("model-a", 30, 40)

    with pytest.raises(DuplicateModelInReport):
        rank_report([a, a_again], pack=pack)


def test_rank_report_ranks_a_continuous_headline_metric_by_mean() -> None:
    """§5 test 4 — the CI column calls `stats.mean_bootstrap_interval` with `support=(0.0, 1.0)`
    from `ContinuousMetric.support` and renders its own stronger descriptive caveat, distinct from
    `_DESCRIPTIVE_NOTE`."""
    pack = _embedder_pack()
    values_a = [0.9, 0.95, 0.85, 0.9, 0.92, 0.88, 0.91, 0.93, 0.89, 0.9]
    values_b = [0.4, 0.5, 0.45, 0.5, 0.42, 0.48, 0.44, 0.46, 0.43, 0.47]
    items_a = [_mrr_item(f"q{i:02d}", v) for i, v in enumerate(values_a)]
    items_b = [_mrr_item(f"q{i:02d}", v) for i, v in enumerate(values_b)]
    agg_a = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=sum(values_a) / len(values_a), n=10,
                              support=(0.0, 1.0))
    )
    agg_b = RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=sum(values_b) / len(values_b), n=10,
                              support=(0.0, 1.0))
    )
    a = run("model-a", role="embedder", call_surface="embeddings", items=items_a,
            aggregates=agg_a, fingerprint_fields=embeddings_fields(packId=pack.packId,
                                                                     modelKey="model-a"))
    b = run("model-b", role="embedder", call_surface="embeddings", items=items_b,
            aggregates=agg_b, fingerprint_fields=embeddings_fields(packId=pack.packId,
                                                                     modelKey="model-b"))

    md = rank_report([a, b], pack=pack)

    section = md.split("### mrr")[1]
    assert section.index("model-a") < section.index("model-b")
    assert "n=10" in section
    assert "This interval describes this model's own mean" in section
    assert "descriptive, not the comparison instrument" not in section


def test_rank_report_with_no_headline_renders_two_independently_sorted_tables() -> None:
    """§5 test 5 — `guard-judge`'s two co-equal, polarity-corrected metrics: this is also the test
    that would have caught §2.4's polarity defect had it existed in this new code path. A model
    with a LOWER falseAdvanceRate must rank ABOVE one with a higher rate."""
    pack = guard_pack(headline=None, verdicts=("falseAdvanceRate", "falseSuspendRate"))
    # model-a: 2/40 falseAdvanceRate (good), 10/40 falseSuspendRate (bad)
    # model-b: 8/40 falseAdvanceRate (bad), 4/40 falseSuspendRate (good)
    a_advance = [item(f"a{i:02d}", correct=i < 2, metric="falseAdvanceRate") for i in range(40)]
    a_suspend = [item(f"s{i:02d}", correct=i < 10, metric="falseSuspendRate") for i in range(40)]
    b_advance = [item(f"a{i:02d}", correct=i < 8, metric="falseAdvanceRate") for i in range(40)]
    b_suspend = [item(f"s{i:02d}", correct=i < 4, metric="falseSuspendRate") for i in range(40)]
    agg_a = ClassificationAggregates(
        perClass=(
            _agg_from(a_advance, metric="falseAdvanceRate").perClass[0],
            _agg_from(a_suspend, metric="falseSuspendRate").perClass[0],
        ),
        parseFailures=0, n=80,
    )
    agg_b = ClassificationAggregates(
        perClass=(
            _agg_from(b_advance, metric="falseAdvanceRate").perClass[0],
            _agg_from(b_suspend, metric="falseSuspendRate").perClass[0],
        ),
        parseFailures=0, n=80,
    )
    a = run("model-a", items=a_advance + a_suspend, aggregates=agg_a,
            fingerprint_fields=model_fields(modelKey="model-a", packId=PACK_ID))
    b = run("model-b", items=b_advance + b_suspend, aggregates=agg_b,
            fingerprint_fields=model_fields(modelKey="model-b", packId=PACK_ID))

    md = rank_report([a, b], pack=pack)

    assert md.count("### falseAdvanceRate") == 1
    assert md.count("### falseSuspendRate") == 1
    advance_section = md.split("### falseAdvanceRate")[1].split("### falseSuspendRate")[0]
    suspend_section = md.split("### falseSuspendRate")[1]
    # falseAdvanceRate: model-a (2/40 = lower, better) ranks above model-b (8/40)
    assert advance_section.index("model-a") < advance_section.index("model-b")
    # falseSuspendRate: model-b (4/40 = lower, better) ranks above model-a (10/40)
    assert suspend_section.index("model-b") < suspend_section.index("model-a")


def test_rank_report_excludes_a_model_with_no_aggregate_without_dropping_it() -> None:
    """§5 test 6 — a model in `runs` with no aggregate at all for the target metric is excluded
    from that table (never ranked last), and never silently dropped from the whole report."""
    pack = _rank_pack()
    a = _rank_arm("model-a", 40, 40)
    b = _rank_arm("model-b", 30, 40)
    c = run(
        "model-c", items=[], aggregates=RetrievalAggregates(),
        fingerprint_fields=model_fields(modelKey="model-c", packId=pack.packId),
    )

    md = rank_report([a, b, c], pack=pack)

    section = md.split("### layer1ExactMatchRate")[1]
    assert "model-c" not in section


def test_rank_report_excludes_and_names_an_arm_whose_aggregate_disagrees_with_its_items() -> None:
    """§5 test 7 — S1 done-condition 10's cross-check, reused: the same exclude-and-name block
    `compare_report` already renders (`_aggregate_item_mismatches`)."""
    import dataclasses

    pack = _rank_pack()
    a = _rank_arm("model-a", 40, 40)
    items_b = [item(f"q{i:02d}", correct=True, metric="layer1ExactMatchRate") for i in range(10)]
    bad_agg = _agg_from(items_b, metric="layer1ExactMatchRate")
    bad_agg = dataclasses.replace(
        bad_agg, perClass=(dataclasses.replace(bad_agg.perClass[0], n=40, successes=40),)
    )
    b = run(
        "model-b", items=items_b, aggregates=bad_agg,
        fingerprint_fields=model_fields(modelKey="model-b", packId=pack.packId),
    )

    md = rank_report([a, b], pack=pack)

    assert "> **INVALID RESULTS EXCLUDED** (AC-2)" in md
    assert "`model-b` — aggregates disagree with items" in md
    assert "model-b" not in md.split("### layer1ExactMatchRate")[1]


def test_rank_report_version_banner_fires_across_a_multi_model_table() -> None:
    """§5 test 13 — the version/hash/schema banners still fire across an N-arm ranked table
    exactly as they do across two arms."""
    pack = _rank_pack()
    a = _rank_arm("model-a", 40, 40)
    b = _rank_arm("model-b", 30, 40)
    c = _rank_arm("model-c", 20, 40)
    import dataclasses

    new_fields = {**c.fingerprint.fields, "packVersion": "2.0.0"}
    c = dataclasses.replace(
        c, fingerprint=dataclasses.replace(c.fingerprint, fields=new_fields)
    )

    md = rank_report([a, b, c], pack=pack)

    assert "PACK VERSION MISMATCH" in md
    # the ranking is still rendered for every consistent arm, not dropped over the mismatch
    assert "model-a" in md and "model-b" in md and "model-c" in md


def test_rank_report_reference_family_holm_ladder_reaches_all_four_decision_states() -> None:
    """§5 test 9 — `reference` given, a single-metric pack, 3+ candidates: one Holm ladder over
    `N-1` p-values, `correction_k=N-1` passed to each `verdict()` call, at least one fixture
    reaching each of the four decision states."""
    pack = _rank_pack()
    reference = _rank_arm("reference", 40, 40)
    # rejected under Holm: hugely discordant vs. reference -> tiny p
    cand_distinguishable = _rank_arm("cand-strong", 20, 40)
    # tested, but its own p (0.03125) exceeds its Holm step (0.025) once ranked second
    cand_not_distinguishable = _rank_arm("cand-mid", 34, 40)
    # ranked last (p=1.0); never reached because Holm already stopped at cand-mid
    cand_not_tested = _rank_arm("cand-weak", 39, 40)
    # disjoint item ids -> empty paired intersection with the reference
    disjoint_items = [
        item(f"z{i:02d}", correct=True, metric="layer1ExactMatchRate") for i in range(40)
    ]
    cand_no_data = run(
        "cand-no-data", items=disjoint_items,
        aggregates=_agg_from(disjoint_items, metric="layer1ExactMatchRate"),
        fingerprint_fields=model_fields(modelKey="cand-no-data", packId=pack.packId),
    )

    md = rank_report(
        [reference, cand_distinguishable, cand_not_distinguishable, cand_not_tested, cand_no_data],
        pack=pack, reference="reference",
    )

    family_section = md.split("#### Reference-anchored family")[1]
    rows = {
        ln.split("|")[1].strip(): ln
        for ln in family_section.splitlines() if ln.startswith("|")
    }
    assert "distinguishable" in rows["cand-strong"] and "not " not in rows["cand-strong"]
    assert rows["cand-mid"].endswith("not distinguishable |") or (
        "not distinguishable" in rows["cand-mid"] and "not tested" not in rows["cand-mid"]
    )
    assert "not tested (Holm stops here)" in rows["cand-weak"]
    assert "no verdict — no paired data" in rows["cand-no-data"]


def _guard_judge_arm(
    model_key: str,
    advance_correct: int,
    suspend_correct: int,
    total: int = 40,
    *,
    advance_total: int | None = None,
    suspend_total: int | None = None,
):
    """`advance_total`/`suspend_total` default to `total` — override either to build a fixture
    with the pack's own real, asymmetric item counts (40 `falseAdvanceRate` / 30
    `falseSuspendRate`, `-ml` §7.3) rather than the equal-n shape every other fixture here uses."""
    advance_n = advance_total if advance_total is not None else total
    suspend_n = suspend_total if suspend_total is not None else total
    advance = [item(f"a{i:02d}", correct=i < advance_correct, metric="falseAdvanceRate")
               for i in range(advance_n)]
    suspend = [item(f"s{i:02d}", correct=i < suspend_correct, metric="falseSuspendRate")
               for i in range(suspend_n)]
    agg = ClassificationAggregates(
        perClass=(
            _agg_from(advance, metric="falseAdvanceRate").perClass[0],
            _agg_from(suspend, metric="falseSuspendRate").perClass[0],
        ),
        parseFailures=0, n=advance_n + suspend_n,
    )
    return run(model_key, items=advance + suspend, aggregates=agg,
               fingerprint_fields=model_fields(modelKey=model_key, packId=PACK_ID))


def test_rank_report_guard_judge_reference_family_uses_one_combined_ladder(monkeypatch) -> None:
    """§5 test 10 — the guard-judge family is one combined Holm ladder over the flattened
    `2*(N-1)` p-values, not two independent `(N-1)`-entry ladders (a regression test for the
    plan's originally-rejected per-metric default); `correction_k=2*(N-1)` passed to every
    `verdict()` call, and each table's decisions come from a shared 4-entry ladder, not two
    2-entry ones."""
    pack = guard_pack(headline=None, verdicts=("falseAdvanceRate", "falseSuspendRate"))
    reference = _guard_judge_arm("reference", 0, 0)
    cand_a = _guard_judge_arm("cand-a", 2, 10)
    cand_b = _guard_judge_arm("cand-b", 8, 4)

    calls: list[list[float]] = []
    real_holm_steps = stats.holm_steps

    def spy(p_values, *, alpha):
        calls.append(list(p_values))
        return real_holm_steps(p_values, alpha=alpha)

    monkeypatch.setattr(stats, "holm_steps", spy)

    md = rank_report([reference, cand_a, cand_b], pack=pack, reference="reference")

    assert len(calls) == 1  # ONE combined call, never two per-metric calls
    assert len(calls[0]) == 4  # 2 metrics * 2 candidates, not two independent 2-entry ladders

    correction_k_line = "alpha_mdd = 0.05/4"
    assert correction_k_line in md


def test_rank_report_guard_judge_family_k_does_not_shrink_when_a_candidate_has_no_paired_data() -> (
    None
):
    """`-ml` review Pass2-1 (required) — a candidate missing paired data for ONE metric must not
    shrink `k` from 32 to 31 (here, 4 to 3): the empty intersection contributes
    `mcnemar_exact(0, 0) = 1.0` and still consumes a Holm rank, mirroring `compare_report`'s own
    homogeneous-binary handling (`report.py`'s `mcnemar_exact(table_b, table_c)` call, unconditional
    even when `outcomes.n_units == 0`)."""
    pack = guard_pack(headline=None, verdicts=("falseAdvanceRate", "falseSuspendRate"))
    reference = _guard_judge_arm("reference", 0, 0)
    cand_a = _guard_judge_arm("cand-a", 2, 10)
    cand_b_advance = [
        item(f"a{i:02d}", correct=i < 8, metric="falseAdvanceRate") for i in range(40)
    ]
    # disjoint item ids for falseSuspendRate -> empty paired intersection with the reference
    cand_b_suspend = [
        item(f"zz{i:02d}", correct=i < 4, metric="falseSuspendRate") for i in range(40)
    ]
    agg_b = ClassificationAggregates(
        perClass=(
            _agg_from(cand_b_advance, metric="falseAdvanceRate").perClass[0],
            _agg_from(cand_b_suspend, metric="falseSuspendRate").perClass[0],
        ),
        parseFailures=0, n=80,
    )
    cand_b = run(
        "cand-b", items=cand_b_advance + cand_b_suspend, aggregates=agg_b,
        fingerprint_fields=model_fields(modelKey="cand-b", packId=PACK_ID),
    )

    md = rank_report([reference, cand_a, cand_b], pack=pack, reference="reference")

    assert "alpha_mdd = 0.05/4" in md  # k stays 4 (2 metrics * 2 candidates), never shrinks to 3
    suspend_family = md.split("#### Reference-anchored family — falseSuspendRate")[1]
    cand_b_row = next(
        ln for ln in suspend_family.splitlines() if ln.startswith("| cand-b")
    )
    assert "no verdict — no paired data" in cand_b_row


def test_polarity_corrected_reads_a_lower_false_advance_rate_candidate_as_better() -> None:
    """§2.4/§5 test 10 — a candidate with FEWER false-advances than the reference (i.e. the
    reference itself carries more of the undesired event) must render with a POSITIVE
    (candidate-better) signed diff — the fix for `stats.verdict()`'s own polarity-blind
    `diff >= 0` (§2.4), exercised directly against `_polarity_corrected` rather than through a
    full report fixture."""
    from modelbench.report import _polarity_corrected

    # reference had the false-advance event on 8 units the candidate did not (reference worse);
    # the candidate had it on 2 units the reference did not (candidate worse there).
    diff = (8 - 2) / 10  # (b - c) / n, exactly as `stats.verdict` computes it
    signed, _ci = _polarity_corrected("falseAdvanceRate", diff, (0.0, 0.0))
    assert signed > 0  # net: candidate is better -> reads positive

    # an ordinary higher-is-better metric is the opposite: raw diff > 0 means the REFERENCE
    # (the "ok" = success winner) is ahead, so the candidate-reads-positive convention must flip.
    signed_hib, ci_hib = _polarity_corrected("groundingRate", 0.6, (0.1, 0.9))
    assert signed_hib < 0
    assert ci_hib == (-0.9, -0.1)


def test_rank_report_raises_when_the_reference_model_has_no_stored_run() -> None:
    """§5 test 11 — `reference` named but absent from `runs` (or excluded above): a usage-shaped
    error at `rank_report`'s own boundary."""
    pack = _rank_pack()
    a = _rank_arm("model-a", 40, 40)
    b = _rank_arm("model-b", 30, 40)

    with pytest.raises(ValueError, match="no stored"):
        rank_report([a, b], pack=pack, reference="unknown-model")


def test_rank_report_no_reference_resolving_power_sentence_is_alongside_the_pack_line() -> None:
    """§5 test 14 — printed whenever the ranked table has >=2 models; `k` is `N-1` for a
    single-metric pack; no specific model name appears; printed alongside, never instead of, the
    pack's own unmodified `resolving_power_line`."""
    pack = _rank_pack()
    a = _rank_arm("model-a", 40, 40)
    b = _rank_arm("model-b", 30, 40)
    c = _rank_arm("model-c", 20, 40)

    md = rank_report([a, b, c], pack=pack)

    assert "This pack resolves differences of" in md  # the pack's own unmodified sentence
    assert "If this pack's optional reference-anchored family (FR-8) were run" in md
    assert "that family of 2 tests" in md  # k = N-1 = 2
    for name in ("model-a", "model-b", "model-c"):
        assert name not in md.split("If this pack's optional")[1].split(".")[0]


def test_rank_report_no_reference_resolving_power_sentence_doubles_k_for_guard_judge() -> None:
    """§5 test 14 — `k` is `2*(N-1)` for a pack with two co-equal verdict metrics (no headline)."""
    pack = guard_pack(headline=None, verdicts=("falseAdvanceRate", "falseSuspendRate"))
    a = _guard_judge_arm("model-a", 2, 10)
    b = _guard_judge_arm("model-b", 8, 4)
    c = _guard_judge_arm("model-c", 5, 5)

    md = rank_report([a, b, c], pack=pack)

    assert "that family of 4 tests" in md  # k = 2 * (N-1) = 2 * 2 = 4
    assert "jointly across both verdict metrics" in md


def test_rank_report_resolving_power_sentence_uses_each_metrics_own_n_not_pooled() -> None:
    """Code-gate finding (`docs/reviews/small-model-catalog-sweep-impl.md`, Unit A.5) —
    guard-judge's real, already-published `-ml` §7.3 asymmetric item counts (40
    `falseAdvanceRate`, 30 `falseSuspendRate`) must each print their own resolving-power sentence
    pair, never one pooled via `max`/`min` across metrics. Uses the pack's real asymmetric n
    (`advance_total=40, suspend_total=30`), not the suite's other equal-n `_guard_judge_arm`
    fixtures — that symmetry is specifically why the pooling defect was invisible until now."""
    pack = guard_pack(headline=None, verdicts=("falseAdvanceRate", "falseSuspendRate"))
    a = _guard_judge_arm("model-a", 2, 4, advance_total=40, suspend_total=30)
    b = _guard_judge_arm("model-b", 8, 10, advance_total=40, suspend_total=30)
    c = _guard_judge_arm("model-c", 5, 5, advance_total=40, suspend_total=30)

    md = rank_report([a, b, c], pack=pack)

    advance_section = md.split("### falseAdvanceRate")[1].split("### falseSuspendRate")[0]
    suspend_section = md.split("### falseSuspendRate")[1]

    # falseAdvanceRate: its own n=40 figures (floor 15.0pp; published MDD80 21.9pp at pack alpha_mdd
    # 0.025; hypothetical MDD80 24.6pp at the compound family's k=4)
    assert "n=40 effective items" in advance_section
    assert "resolves differences of >=21.9 pp" in advance_section
    assert "Differences below 15.0 pp cannot reach significance" in advance_section
    assert "that family of 4 tests would resolve differences of >=24.6 pp" in advance_section

    # falseSuspendRate: its own n=30 figures (floor 20.0pp; published MDD80 28.7pp; hypothetical
    # MDD80 32.3pp at k=4) — never falseAdvanceRate's n=40 numbers
    assert "n=30 effective items" in suspend_section
    assert "resolves differences of >=28.7 pp" in suspend_section
    assert "Differences below 20.0 pp cannot reach significance" in suspend_section
    assert "that family of 4 tests would resolve differences of >=32.3 pp" in suspend_section
    assert "n=40 effective items" not in suspend_section
    assert ">=21.9 pp" not in suspend_section
    assert ">=24.6 pp" not in suspend_section


def test_rank_report_refuses_a_headline_outside_the_verdict_family() -> None:
    """`analyst` code-gate suggestion (`docs/reviews/small-model-catalog-sweep-impl.md`) — mirrors
    `compare_report`'s own guard on the identical field. Pack-load-time enforcement
    (`metrics_from_manifest`) only protects the real `./run.sh rank` CLI path; every fixture in
    this suite (including `rank_report`'s own) builds a `PackRef`/`PackMetrics` directly, bypassing
    it entirely — which is exactly why `compare_report` carries its own independent check too."""
    pack = PackRef(
        packId="nlq-structured-query", packVersion="1.0.0", contentHash="9" * 64,
        role="nlq-generator",
        metrics=PackMetrics(verdictMetrics=("a",), headlineMetric="b"),
        pairingKey=("itemId",), analysisUnit="itemId", seed=20260902,
    )
    a = _rank_arm("model-a", 40, 40, metric="a", pack_id="nlq-structured-query")

    with pytest.raises(PackConfigError):
        rank_report([a], pack=pack)


def test_rank_report_reference_family_floor_demotion_fires_at_the_candidate_axis_correction_k() -> (
    None
):
    """`analyst` code-gate suggestion (`docs/reviews/small-model-catalog-sweep-impl.md`) — Rule 7's
    observable-floor enforcement is untouched by `correction_k` (`stats.py`'s `verdict()` computes
    `resolving.observable_floor` from `alpha_family`/`n_eff` alone, never from `k`), but nothing
    pinned that by test at `rank_report`'s new candidate-axis scale (`correction_k != len(family)`)
    before this test. Two candidates against one reference gives `correction_k = 2`, not the
    trivial `k = 1` a single-candidate fixture would exercise."""
    import dataclasses

    pack = _rank_pack()
    reference = _rank_arm("reference", 40, 40)
    # a declared design effect > 1.0 moves the decision off the exact mcnemar path onto the
    # conservative envelope, where Rule 7's floor is a guard (demotes) rather than a theorem
    # (raises) — the same worked shape as `stats.py`'s own
    # `test_no_clustered_verdict_is_distinguishable_below_the_observable_floor`.
    reference = dataclasses.replace(reference, designEffect=2.0, basis="measured")
    # (a, b, c, d) = (32, 8, 0, 0) at deff=2.0 is `stats.py`'s own worked floor-demotion fixture
    # (`test_rule_7_is_what_catches_the_case_the_widened_interval_still_misses`): the interval
    # alone excludes zero, but the observed 20.0 pp sits below the deff-widened 30.0 pp floor.
    cand_below_floor = _rank_arm("cand-floor", 32, 40)
    cand_other = _rank_arm("cand-other", 20, 40)  # only present to make correction_k = 2, not 1

    md = rank_report(
        [reference, cand_below_floor, cand_other], pack=pack, reference="reference",
    )

    family_section = md.split("#### Reference-anchored family")[1]
    floor_row = next(
        ln for ln in family_section.splitlines() if ln.startswith("| cand-floor")
    )
    assert "not distinguishable — below the observable floor" in floor_row
    assert "distinguishable |" not in floor_row  # never plain "distinguishable"
