"""§5 test 5 — `compare_report`, where AC-2/AC-3/AC-4 become visible output.

Every fixture is a hand-built `RunResult`; no LM Studio, no network, no pack on disk (plan §4 S1).
"""

from __future__ import annotations

import pytest
from conftest import (
    classification_aggregates,
    deterministic_fields,
    guard_pack,
    item,
    model_fields,
    run,
)

from modelbench.fingerprint import FieldProblem
from modelbench.packs import PackConfigError, PackMetrics, PackRef, metrics_from_manifest
from modelbench.report import compare_report
from modelbench.results import InvalidRecord
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
            aggregates=classification_aggregates(12, 12, metric),
            fingerprint_fields={**fields, "modelKey": "cand"})
    b = run("incumbent", role="tool-caller", items=b_items,
            aggregates=classification_aggregates(6, 12, metric),
            fingerprint_fields={**fields, "modelKey": "incumbent"})

    md = compare_report([a, b], pack=pack)

    assert (
        "This pack resolves differences of >=57.8 pp with 80% power at n=12 effective "
        "conversations (12 units, design effect 1.00, by-construction, alpha=0.05). Differences "
        "below 50.0 pp cannot reach significance at any observed outcome. Best case — assumes the "
        "candidate wins every conversation the models differ on; if it loses one for every two it "
        "wins, 80% power is not reached at any effect size at this n. Inference is conditional on "
        "the 12 scripts in tool-caller-shop-assistant@1.0.0; generalization to unwritten scripts "
        "is not certified by any interval in this report."
    ) in md


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
            aggregates=classification_aggregates(24, 48, metric),
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
    """Plan §3.3 — 'no call site chooses it, and there is no parameter through which one could'."""
    import inspect

    params = inspect.signature(compare_report).parameters
    assert set(params) == {"runs", "pack", "invalid"}


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
    assert "alpha=0.025" in md


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
    assert "sideMetric" in md
    assert "exploratory — no significance claim" in md


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
        pairingKey=("scriptId", "replicate"), analysisUnit="replicate",
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
