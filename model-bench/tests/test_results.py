"""§5 tests 2 and 3 — the two enforcement points (plan §3.4.5, AC-2).

The centre of gravity is the *read* side: AC-2 requires a hand-edited record to be excluded when
history is loaded, not merely rejected when it was written.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest
from conftest import deterministic_fields, embeddings_fields, model_fields, run

from modelbench import results as results_module
from modelbench import stats
from modelbench.fingerprint import Fingerprint
from modelbench.results import (
    _AGGREGATE_BY_KIND,
    _METRIC_DECODERS,
    BENCH_SCHEMA_VERSION,
    INDEX_COLUMNS,
    BinaryMetric,
    ClassificationAggregates,
    ContinuousMetric,
    DistributionSummary,
    ExtractionAggregates,
    GroundingAggregates,
    IncompleteItemRecord,
    InvalidFingerprint,
    ItemResult,
    MetricKindError,
    NonFiniteMeasure,
    RetrievalAggregates,
    RunResult,
    ToolCallAggregates,
    _metric_from_dict,
    _metric_to_dict,
    load_history,
    models_with_stored_results,
    rebuild_index,
    store,
)

PACK = "tool-caller-shop-assistant"


def _run(run_id: str, **kwargs):
    return run(run_id, role="tool-caller", **kwargs)


# --- write refuses ---------------------------------------------------------------------------


def test_store_writes_a_valid_run(tmp_root) -> None:
    path = store(_run("r1"), tmp_root)
    assert path.exists()
    assert json.loads(path.read_text())["runId"] == "r1"


def test_store_refuses_an_invalid_fingerprint(tmp_root) -> None:
    bad = _run("r2", fingerprint_fields=model_fields(kvCacheSetting=""))
    with pytest.raises(InvalidFingerprint) as excinfo:
        store(bad, tmp_root)
    assert "kvCacheSetting" in str(excinfo.value)
    assert not (tmp_root / "results" / "runs" / "r2.json").exists()


def test_store_has_no_bypass_flag() -> None:
    """Plan §3.4.5 — no "save anyway" flag. Asserted on the API surface, not in a comment."""
    assert list(inspect.signature(store).parameters) == ["run", "root"]


def test_store_refuses_a_deterministic_arm_carrying_a_model_field(tmp_root) -> None:
    """S1 done-condition 6 — `{"modelKey": "bm25"}` fails on write, loudly (plan §3.4.1)."""
    bad = _run(
        "r3",
        arm_kind="deterministic",
        fingerprint_fields=deterministic_fields(modelKey="bm25"),
    )
    with pytest.raises(InvalidFingerprint) as excinfo:
        store(bad, tmp_root)
    assert "modelKey (forbidden)" in str(excinfo.value)


def test_store_accepts_a_clean_deterministic_arm(tmp_root) -> None:
    assert store(_run("r4", arm_kind="deterministic"), tmp_root).exists()


def test_store_accepts_a_clean_embeddings_arm(tmp_root) -> None:
    """S1 done-condition 6's third profile — the write surface asserts all three, not two.

    A `model:embeddings` record carries no `runtimeName`, `runtimeVersion`, `temperature` or
    `maxTokens` and is *correct*: that profile forbids them (§3.4.2, §3.4.4a).
    """
    assert store(_run("r5", call_surface="embeddings"), tmp_root).exists()


def test_store_refuses_an_embeddings_arm_carrying_a_chat_only_field(tmp_root) -> None:
    """The forbid half, on the profile the derivation invented it for: an embeddings call has no
    `runtime` object to observe, so a record naming one is claiming an unmeasurable fact."""
    bad = _run(
        "r6",
        call_surface="embeddings",
        fingerprint_fields=embeddings_fields(runtimeName="llama.cpp"),
    )
    with pytest.raises(InvalidFingerprint) as excinfo:
        store(bad, tmp_root)
    assert "runtimeName (forbidden)" in str(excinfo.value)


# --- read quarantines ------------------------------------------------------------------------


def test_load_history_returns_valid_records(tmp_root) -> None:
    store(_run("r1"), tmp_root)
    valid, invalid = load_history(tmp_root, packId=PACK)
    assert [r.runId for r in valid] == ["r1"]
    assert invalid == []


def test_a_hand_edited_record_is_quarantined_on_read(tmp_root) -> None:
    """AC-2 — blank an attested field on disk, the way a human with an editor would."""
    path = store(_run("r1"), tmp_root)
    raw = json.loads(path.read_text())
    raw["fingerprint"]["kvCacheSetting"] = ""
    path.write_text(json.dumps(raw))

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert len(invalid) == 1
    assert invalid[0].reason == "field"
    assert invalid[0].runId == "r1"
    assert [p.field for p in invalid[0].problems] == ["kvCacheSetting"]


def test_a_future_schema_version_is_quarantined_as_unknown_schema(tmp_root) -> None:
    """Plan §3.4.3 — a record from the *future* is the genuinely uninterpretable case."""
    path = store(_run("r1"), tmp_root)
    raw = json.loads(path.read_text())
    raw["fingerprint"]["benchSchemaVersion"] = 99
    path.write_text(json.dumps(raw))

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert invalid[0].reason == "unknown_schema"
    assert invalid[0].benchSchemaVersion == 99


def test_a_boolean_schema_version_is_refused_at_both_enforcement_points(tmp_root) -> None:
    """P2-2 — `True == 1`, so `True in REQUIRED_BY_SCHEMA` was `True` and the two sides disagreed.

    `load_history` learned an `isinstance(schema, bool)` guard in the fix round; `validate()` did
    not. The result: `store()` accepted the record and wrote the file, and the reader it was
    written for immediately quarantined it — with a `bool` landing in an `InvalidRecord` field
    typed `int | None`. Neither side was tested, so removing *either* guard was green.

    A record that cannot be read back is not a record. The two enforcement points must agree, and
    the one that agrees honestly is refusal at write time (plan §3.4.5).
    """
    bad = _run("r1", fingerprint_fields=model_fields(benchSchemaVersion=True))
    assert [p.field for p in bad.fingerprint.validate()] == ["benchSchemaVersion"]
    assert [p.reason for p in bad.fingerprint.validate()] == ["unknown"]
    with pytest.raises(InvalidFingerprint):
        store(bad, tmp_root)
    assert not list((tmp_root / "results").rglob("*.json"))

    # ...and the reader keeps its own guard, because a hand-edited file never passed `store()`
    # (AC-2's whole point). Written past the writer, it comes back quarantined and never valid.
    good = store(_run("r2"), tmp_root)
    raw = json.loads(good.read_text())
    raw["fingerprint"]["benchSchemaVersion"] = True
    good.write_text(json.dumps(raw))
    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert invalid[0].reason == "unknown_schema"
    assert invalid[0].benchSchemaVersion is None  # a bool never reaches an `int | None` field


def test_a_truncated_file_is_quarantined_as_unparseable(tmp_root) -> None:
    store(_run("r1"), tmp_root)
    path = tmp_root / "results" / "runs" / "r1.json"
    path.write_text(path.read_text()[: len(path.read_text()) // 2])

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert invalid[0].reason == "unparseable"
    assert invalid[0].runId is None


def test_an_older_known_schema_record_stays_valid(monkeypatch, tmp_root) -> None:
    """Plan §3.4.3 / S1 done-condition 7 — the FR-3 case that must NOT be silently deleted.

    The current schema moves to 2 and gains a required field. The record stored under schema 1
    never carried it, satisfied the contract it was written under, and must still appear in every
    comparison — the tool's whole value is that a new model lines up against models tested months
    ago. A record that *declares* schema 2 and omits the field is the genuinely invalid one, and
    both directions are asserted in one load so an implementation cannot pass by accepting
    everything.
    """
    from modelbench.fingerprint import REQUIRED_BY_SCHEMA, FieldSpec

    store(_run("old"), tmp_root)

    schema_2 = {
        kind: {**spec, "hypotheticalNewField": FieldSpec(tier="nonempty")}
        for kind, spec in REQUIRED_BY_SCHEMA[1].items()
    }
    monkeypatch.setitem(REQUIRED_BY_SCHEMA, 2, schema_2)
    monkeypatch.setattr("modelbench.results.BENCH_SCHEMA_VERSION", 2)

    # Written straight to disk: `store()` would (correctly) refuse it, and a record that declares a
    # schema it does not satisfy is exactly the hand-edited artifact the read side must catch.
    newer = _run("new", fingerprint_fields=model_fields(benchSchemaVersion=2)).to_dict()
    (tmp_root / "results" / "runs" / "new.json").write_text(json.dumps(newer))

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert [r.runId for r in valid] == ["old"]
    assert [(r.runId, r.reason) for r in invalid] == [("new", "field")]
    assert [p.field for p in invalid[0].problems] == ["hypotheticalNewField"]


def test_load_history_is_scoped_to_one_pack(tmp_root) -> None:
    """FR-20 is structural: there is no API that loads across packs (§3.5)."""
    store(_run("mine"), tmp_root)
    other = _run("theirs", fingerprint_fields=model_fields(packId="embedder-graphrag-retrieval"))
    store(other, tmp_root)

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert [r.runId for r in valid] == ["mine"]
    assert invalid == []
    assert list(inspect.signature(load_history).parameters) == ["root", "packId"]


def test_record_round_trips_through_disk(tmp_root) -> None:
    original = _run(
        "r1",
        items=[
            ItemResult(
                itemId="A-01",
                pairingKey=("A-01", "0"),
                outcome="pass",
                scoreable={"cleanThroughTurn4": True},
                counts={"cleanThroughTurn4": 1},
                latencyMs=1300.0,
                detail={"note": "kept"},
            )
        ],
    )
    path = store(original, tmp_root)
    restored = RunResult.from_dict(json.loads(path.read_text()))
    assert restored == original


def test_bench_schema_version_is_a_separate_constant() -> None:
    """Plan §3.4.3 — never derived from `benchVersion`, never bumped by a release."""
    import modelbench

    assert BENCH_SCHEMA_VERSION == 1
    assert str(BENCH_SCHEMA_VERSION) != modelbench.__version__


# --- the derived views -----------------------------------------------------------------------


def test_rebuild_index_is_regenerable(tmp_root) -> None:
    store(_run("r1"), tmp_root)
    first = rebuild_index(tmp_root).read_text()
    assert rebuild_index(tmp_root).read_text() == first
    assert "r1" in first
    assert "runId,date,role" in first


def test_models_with_stored_results_excludes_deterministic_arms(tmp_root) -> None:
    """FR-17a — BM25 can never be offered as a reference *model* (plan §3.4.1)."""
    store(_run("m"), tmp_root)
    store(_run("d", arm_kind="deterministic"), tmp_root)
    assert models_with_stored_results(tmp_root) == ["qwen/qwen3-4b-2507"]


def test_models_with_stored_results_still_includes_an_embeddings_arm(tmp_root) -> None:
    """`armKind` keeps its two values and every `armKind == "model"` filter is unchanged by the
    profile re-key (§3.4.1) — including this one. An embeddings run is a *model* run; a filter
    that had become profile-aware would silently drop the embedder from `models --tested`."""
    store(_run("e", call_surface="embeddings", fingerprint_fields=embeddings_fields(
        modelKey="text-embedding-qwen3-embedding-0.6b"
    )), tmp_root)
    assert models_with_stored_results(tmp_root) == ["text-embedding-qwen3-embedding-0.6b"]


# --- P3-1: an item's metric outcome is declared, never inferred --------------------------------


def _item(scoreable: dict, counts: dict) -> ItemResult:
    return ItemResult(
        itemId="i1", pairingKey=("i1",), outcome="pass",
        scoreable=scoreable, counts=counts, latencyMs=None, detail={},
    )


def test_an_undeclared_metric_is_not_scoreable() -> None:
    """Review P3-1 — `scoreable.get(metric, True)` read an item that never mentions the metric as
    a scoreable one, and the count default then scored it a loss. Absence is not a declaration."""
    assert _item({}, {}).scored_outcome("falseAdvanceRate") is None


def test_a_declared_precondition_failure_is_not_scoreable() -> None:
    assert _item({"m": False}, {}).scored_outcome("m") is None


def test_a_declared_scoreable_metric_returns_its_recorded_outcome() -> None:
    assert _item({"m": True}, {"m": 1}).scored_outcome("m") is True
    assert _item({"m": True}, {"m": 0}).scored_outcome("m") is False


def test_an_item_that_declares_a_metric_scoreable_and_records_no_count_is_refused() -> None:
    """The self-contradictory record: the arm says it scored the item and supplies no score.
    Reading the absent count as `0` publishes a failure the scorer never observed, which is the
    laundering `-ml` §4.3 forbids, pointed the other way. **This is a contract S2's scorers must
    honour**: a metric declared scoreable carries a count, always."""
    with pytest.raises(IncompleteItemRecord) as excinfo:
        _item({"m": True}, {}).scored_outcome("m")
    assert "m" in str(excinfo.value) and "i1" in str(excinfo.value)


def test_a_fingerprint_dataclass_keeps_absent_distinct_from_null() -> None:
    absent = Fingerprint(
        armKind="model", callSurface="chat", fields=model_fields(kvCacheSetting=...)
    )
    nulled = Fingerprint(
        armKind="model", callSurface="chat", fields=model_fields(kvCacheSetting=None)
    )
    assert [p.reason for p in absent.validate()] == ["absent"]
    assert [p.reason for p in nulled.validate()] == ["null"]


def test_load_history_excludes_and_names_an_item_that_declares_a_count_it_does_not_carry(
    tmp_root,
) -> None:
    """Review P4-5 — the refusal landed at the furthest possible point from its producer.

    `store()` accepts the record (its validation is fingerprint-only) and `load_history` accepted
    it too, so `IncompleteItemRecord` was raised at *report* time, from `report.py`'s only caller,
    where `cli.py` catches nothing but `PackConfigError`. One bad item in one of two otherwise
    valid records aborted the whole `compare` with an uncaught traceback, **exit 1** — outside
    §3.6a's closed set of `0/2/3/4/5` — with **no report written at all** and the valid arm lost
    with it.

    AC-2's actual mechanism is *excluded on read **and named***, which is what this restores: the
    record is a `field` failure like any other, the block prints which item and which metric, and
    every other record still loads.
    """
    def ok(item_id: str) -> ItemResult:
        return ItemResult(itemId=item_id, pairingKey=(item_id,), outcome="pass",
                          scoreable={"m": True}, counts={"m": 1}, latencyMs=1.0, detail={})

    store(_run("good", items=[ok("i1")]), tmp_root)
    store(
        _run("bad", items=[
            ok("i1"),
            ItemResult(itemId="i9", pairingKey=("i9",), outcome="pass",
                       scoreable={"m": True}, counts={}, latencyMs=1.0, detail={}),
        ]),
        tmp_root,
    )

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert [r.runId for r in valid] == ["good"]
    assert [r.runId for r in invalid] == ["bad"]
    assert invalid[0].reason == "field"
    assert [(p.field, p.reason) for p in invalid[0].problems] == [
        ("items[i9].counts.m", "absent")
    ]


# --- M-1 / m-1: which records the pack filter may drop, and which are findings ------------------


@pytest.mark.parametrize("packid_value", ["", None, ...])
def test_a_record_that_cannot_declare_its_pack_is_named_never_silently_dropped(
    tmp_root, packid_value
) -> None:
    """Review M-1 — AC-2's guarantee is "excluded on read **and** named", and this was an absence.

    `packId` is a `REQUIRED_NONEMPTY` field, so a record whose `packId` was blanked or deleted on
    disk failed the `!=` pack test, was skipped before validation ever ran, and appeared in
    **neither** returned list: the comparison quietly lost an arm and the report said nothing. This
    module's own docstring says an unreadable record "is a finding, not an absence".
    """
    path = store(_run("r1"), tmp_root)
    raw = json.loads(path.read_text())
    if packid_value is ...:
        del raw["fingerprint"]["packId"]
    else:
        raw["fingerprint"]["packId"] = packid_value
    path.write_text(json.dumps(raw))

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert [r.runId for r in invalid] == ["r1"]
    assert invalid[0].reason == "field"
    assert "packId" in [p.field for p in invalid[0].problems]


@pytest.mark.parametrize("field_name", ["packId", "kvCacheSetting", "modelKey", "runtimeName"])
def test_the_read_side_quarantines_every_required_field_not_just_one(tmp_root, field_name) -> None:
    """DC-1's read-side test blanked only `kvCacheSetting`; the exhaustive per-field loop ran
    against `Fingerprint.validate()` and never through `load_history`, which is the seam AC-2 is
    actually about (review M-1)."""
    path = store(_run("r1"), tmp_root)
    raw = json.loads(path.read_text())
    raw["fingerprint"][field_name] = ""
    path.write_text(json.dumps(raw))

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert [p.field for p in invalid[0].problems] == [field_name]


def test_a_record_belonging_to_another_pack_is_not_this_packs_exclusion(tmp_root) -> None:
    """Review m-1 — an unknown schema short-circuited the pack filter, so an
    `embedder-graphrag-retrieval` record at `benchSchemaVersion: 99` was reported as an AC-2
    exclusion in a `tool-caller` comparison. Its `packId` is right there and readable."""
    path = store(_run("theirs", fingerprint_fields=model_fields(packId="embedder-x")), tmp_root)
    raw = json.loads(path.read_text())
    raw["fingerprint"]["benchSchemaVersion"] = 99
    path.write_text(json.dumps(raw))

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert invalid == []


def test_an_unparseable_record_is_still_this_packs_finding(tmp_root) -> None:
    """The pack filter stays **off** `unparseable`: a truncated file cannot declare its pack, so
    dropping it would be the silent absence M-1 is about (review m-1's stated boundary)."""
    store(_run("r1"), tmp_root)
    path = tmp_root / "results" / "runs" / "r1.json"
    path.write_text(path.read_text()[:40])
    valid, invalid = load_history(tmp_root, packId="some-other-pack")
    assert [r.reason for r in invalid] == ["unparseable"]


# --- P14-4: the envelope is classified before the body is decoded ------------------------------
#
# The pack filter and `unknown_schema` above are reachable only for a future record whose *body*
# this build still happens to decode — that is, for the one future record whose schema bump was
# not needed. `BENCH_SCHEMA_VERSION`'s own docstring says the integer increments when *the
# on-disk record shape changes in a way a reader must branch on*, so the ordinary future record
# is one this build cannot decode, and that record reached neither branch: `RunResult.from_dict`
# raised first and the whole file landed on `unparseable`, with `runId=None` and
# `benchSchemaVersion=None`.
#
# P14-4 named one instance of this (`_AGGREGATE_BY_KIND` losing a kind). Both breakages below
# reach it through two *independent* decoders and neither involves any drift of that constant, so
# the pin U82 put on it cannot close them: the defect is the ordering, not the table.


def _a_future_schema(raw) -> None:
    """The bump that accompanies any shape change (`BENCH_SCHEMA_VERSION`'s own docstring)."""
    raw["fingerprint"]["benchSchemaVersion"] = 99


def _a_sixth_aggregate_kind(raw) -> None:
    """A kind a later build added — `_AGGREGATE_BY_KIND[d["kind"]]` raises `KeyError`."""
    raw["aggregates"]["kind"] = "grounding2"


def _a_fourth_metric_tag(raw) -> None:
    """A metric tag a later build added — `_metric_from_dict` raises `ValueError`."""
    raw["aggregates"]["mrr"]["type"] = "quantile-sketch"


_UNDECODABLE_BODIES = pytest.mark.parametrize(
    "break_body",
    [_a_sixth_aggregate_kind, _a_fourth_metric_tag],
    ids=["new-aggregate-kind", "new-metric-tag"],
)


def _store_then_edit(tmp_root, run_id: str, *edits, **fingerprint_overrides) -> None:
    """Store a decodable record, then hand-edit the file as a later build would have left it."""
    path = store(
        _run(
            run_id,
            aggregates=RetrievalAggregates(
                mrr=ContinuousMetric(name="mrr", mean=0.5, n=1, support=(0.0, 1.0))
            ),
            fingerprint_fields=model_fields(**fingerprint_overrides),
        ),
        tmp_root,
    )
    raw = json.loads(path.read_text())
    for edit in edits:
        edit(raw)
    path.write_text(json.dumps(raw))


@_UNDECODABLE_BODIES
def test_another_packs_future_record_is_filtered_even_when_its_body_will_not_decode(
    tmp_root, break_body
) -> None:
    """m-1's claim, applied to the record that actually motivates it.

    The comment above the schema branch says another pack's future-schema record is no longer
    surfaced as this pack's exclusion because "its `packId` is right there and readable". It was
    just as readable in this file — a complete, well-formed fingerprint block — and the record
    still landed in a `tool-caller` comparison's AC-2 block as `unparseable`, because the body
    was decoded before anything read the pack.
    """
    _store_then_edit(tmp_root, "theirs", _a_future_schema, break_body, packId="embedder-x")

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert invalid == []


@_UNDECODABLE_BODIES
def test_this_packs_future_record_is_quarantined_as_unknown_schema_not_as_corruption(
    tmp_root, break_body
) -> None:
    """The operator-visible half. "Written under a schema this build does not know" and "this
    file is damaged" call for different actions — upgrade the tool, versus restore the file — and
    the AC-2 block printed the second for both, under a *filename* rather than a run id because
    `runId` came back `None`. Whether this build also chokes on the body is an accident of which
    fields the bump changed, and must not change the diagnosis."""
    _store_then_edit(tmp_root, "r1", _a_future_schema, break_body)

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert [(r.reason, r.runId, r.benchSchemaVersion) for r in invalid] == [
        ("unknown_schema", "r1", 99)
    ]
    assert [(p.field, p.reason) for p in invalid[0].problems] == [
        ("benchSchemaVersion", "unknown")
    ]


@_UNDECODABLE_BODIES
def test_a_known_schema_record_that_will_not_decode_stays_unparseable_but_is_named(
    tmp_root, break_body
) -> None:
    """The half that keeps the fix honest. A record claiming a schema this build *does* know and
    still failing to decode is not a record from the future — it is a damaged or non-conforming
    one, and answering `unknown_schema` would launder it into a tooling-version excuse. It stays
    `unparseable`; what changes is that its identity was readable all along, so the AC-2 line
    names the run instead of the bare filename."""
    _store_then_edit(tmp_root, "r1", break_body)

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert [(r.reason, r.runId, r.benchSchemaVersion) for r in invalid] == [
        ("unparseable", "r1", BENCH_SCHEMA_VERSION)
    ]


@pytest.mark.parametrize("envelope_field", ["runId", "fingerprint"])
def test_a_record_whose_envelope_is_unreadable_is_never_pack_filtered(
    tmp_root, envelope_field
) -> None:
    """The reach of the degraded classification, one behavioural consequence per member: reading
    the envelope first must not widen the silent drop. A record that cannot state its identity
    cannot state its pack either, so it stays m-1's finding — surfaced even under another pack's
    id, with nothing invented for the two fields it never supplied."""

    def _drop(raw) -> None:
        del raw[envelope_field]

    _store_then_edit(tmp_root, "r1", _drop, _a_sixth_aggregate_kind)

    valid, invalid = load_history(tmp_root, packId="some-other-pack")
    assert valid == []
    assert [(r.reason, r.runId, r.benchSchemaVersion) for r in invalid] == [
        ("unparseable", None, None)
    ]


# --- M-2 / m-ML-3: the record seam carries no anti-conservative default -------------------------


@pytest.mark.parametrize("omitted", ["designEffect", "basis"])
def test_run_result_requires_the_design_effect_and_its_basis(omitted: str) -> None:
    """Plan v1.5 §3.5 — "required, no defaults". `-ml` §3.4 Rule 2 removes the `1.0` default from
    `resolving_power` precisely so no caller can assert DEFF = 1 by omission; a default on
    `RunResult` restores it one layer out, at the seam S2's runner constructs.

    It also makes DC-5's clause "`report.py` refuses to render one when the required input is
    absent" true only vacuously: with a default the input can never *be* absent.
    """
    import dataclasses

    field = {f.name: f for f in dataclasses.fields(RunResult)}[omitted]
    assert field.default is dataclasses.MISSING
    assert field.default_factory is dataclasses.MISSING

    kwargs = {
        "runId": "r", "sessionId": None, "role": "guard-judge", "armKind": "model",
        "fingerprint": Fingerprint(armKind="model", callSurface="chat", fields=model_fields()),
        "items": (), "aggregates": ClassificationAggregates(),
        "designEffect": 1.0, "basis": "by-construction",
    }
    kwargs.pop(omitted)
    with pytest.raises(TypeError):
        RunResult(**kwargs)


def test_from_dict_is_the_one_place_the_legacy_fallback_belongs(tmp_root) -> None:
    """Plan v1.5 — `d.get("designEffect", 1.0)` there means "a record written before these fields
    existed": a *reader's* compatibility rule under §3.4.3, not a constructor's default."""
    record = _run("r1").to_dict()
    del record["designEffect"]
    del record["basis"]
    restored = RunResult.from_dict(record)
    assert restored.designEffect == 1.0
    assert restored.basis == "assumed"


# --- n-3: an absent aggregates block is a finding, not something to repair ----------------------


def test_a_record_with_no_aggregates_block_is_quarantined(tmp_root) -> None:
    """Review n-3 — `from_dict` defaulted a missing block to `{"kind": "classification"}`,
    fabricating an empty `ClassificationAggregates` for a record that has none. In a module whose
    thesis is "an unreadable record is a finding, not an absence", this one absence was repaired
    instead of reported."""
    path = store(_run("r1"), tmp_root)
    raw = json.loads(path.read_text())
    del raw["aggregates"]
    path.write_text(json.dumps(raw))

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert [r.reason for r in invalid] == ["unparseable"]


# --- Pass 14 P14-4: `_AGGREGATE_BY_KIND` can lose a kind in green -------------------------------

# The five `Aggregates` subclasses themselves — named directly, not read off `_AGGREGATE_BY_KIND`.
# Parametrizing from the constant under test could only ever lose a case on the exact shrink this
# pins against (review Pass 15 §4's tautology warning); driving from each dataclass's own `kind`
# default is what keeps the source of truth independent of the dict being checked.
_ALL_AGGREGATE_CLASSES = (
    RetrievalAggregates,
    ToolCallAggregates,
    ClassificationAggregates,
    ExtractionAggregates,
    GroundingAggregates,
)


def test_aggregate_by_kind_domain_matches_the_declared_aggregate_classes():
    """Form (i) (review Pass 15 §4): binds `_AGGREGATE_BY_KIND` to an independent declaration of
    the same five kinds — each dataclass's own `kind: Literal[...]` default — rather than to
    anything computed from the dict itself. Dropping `"grounding"` from `_AGGREGATE_BY_KIND`
    disagrees with `GroundingAggregates().kind` either way a bare membership check on the dict
    alone could not."""
    assert set(_AGGREGATE_BY_KIND) == {cls().kind for cls in _ALL_AGGREGATE_CLASSES}


@pytest.mark.parametrize("cls", _ALL_AGGREGATE_CLASSES, ids=lambda c: c().kind)
def test_every_aggregate_kind_round_trips_through_load_history(tmp_root, cls) -> None:
    """Form (ii) (review Pass 15 §4): a behavioural consequence per member — a genuine
    `store()`/`load_history()` round trip, not a membership check — complementing the declarative
    pin above.

    Dropping a kind from `_AGGREGATE_BY_KIND` leaves the full suite green today (§N.1 entry 18:
    834 passed) only because nothing drives a real record of that kind through storage end to
    end — `grounding` is inert until an S7 pack exists. When it is dropped,
    `_aggregates_from_dict`'s `KeyError` is caught by `load_history`'s late `except Exception`
    (the same catch that quarantines a record whose body will not decode) and the record —
    genuinely valid, merely of a kind this build's dict forgot — is reported as `"unparseable"`
    under a schema it does conform to, in the one module whose thesis is that an unreadable
    record is a finding and not an absence. That report is now *named* rather than anonymous
    (P14-4's envelope fix, below), which is why this round trip and not the reason string is
    what catches the drift."""
    aggregates = cls()
    store(_run(f"r-{aggregates.kind}", aggregates=aggregates), tmp_root)
    valid, invalid = load_history(tmp_root, packId=PACK)
    assert [r.reason for r in invalid] == []
    assert len(valid) == 1
    assert valid[0].aggregates == aggregates


# --- m-7: `store()` names the reason instead of raising from pathlib ---------------------------


def test_store_refuses_a_run_id_carrying_a_path_separator(tmp_root) -> None:
    """Review m-7 — plan §3.5 specifies `modelSlug` sanitisation precisely because real model keys
    contain `/` (`qwen/qwen3-4b-2507`), and the slugging is S2's runner. Today an unslugged id
    raised a bare `FileNotFoundError` from `pathlib`, and a segment naming an existing directory
    would have written outside `runs/`."""
    with pytest.raises(ValueError) as excinfo:
        store(_run("pack-qwen/qwen3-4b-2507-01"), tmp_root)
    assert "runId" in str(excinfo.value)
    assert not list((tmp_root / "results").rglob("*.json"))


def test_store_refuses_an_empty_run_id(tmp_root) -> None:
    """P2-4 — the only member of the guard's set that the first clause does not already catch.

    `Path(".").name` is `""`, so `"." != ""` already fails the bare-filename check and that member
    of `{"", ".", ".."}` is unreachable. **`Path("..").name` is `".."`, so `".."` is not** — the
    finding says two-thirds of the set is unreachable and one-third is. None of the three was
    tested, which is why the difference had never been measured: dropping the whole clause was
    green, and so was dropping only the reachable half.
    """
    for empty in ("", ".", ".."):
        with pytest.raises(ValueError) as excinfo:
            store(_run(empty), tmp_root)
        assert "runId" in str(excinfo.value)
    assert not list((tmp_root / "results").rglob("*.json"))


# --- m-4: the two derived views' untested filters ----------------------------------------------


def test_models_with_stored_results_filters_by_role(tmp_root) -> None:
    """Review m-4 — `--role` is a shipped flag whose filter could be deleted entirely in green."""
    store(_run("tc"), tmp_root)
    store(run("gj", role="guard-judge", fingerprint_fields=model_fields(modelKey="other-model")),
          tmp_root)
    assert models_with_stored_results(tmp_root, role="tool-caller") == ["qwen/qwen3-4b-2507"]
    assert models_with_stored_results(tmp_root, role="guard-judge") == ["other-model"]
    assert len(models_with_stored_results(tmp_root)) == 2


def test_the_index_latency_columns_are_p50_and_p95(tmp_root) -> None:
    """Review m-4 — the index test asserted only the header and the runId, so computing
    `latencyMsP95` at the 50th percentile was green."""
    items = [
        ItemResult(itemId=f"i{i}", pairingKey=(f"i{i}",), outcome="pass", scoreable={},
                   counts={}, latencyMs=float(i), detail={})
        for i in range(1, 101)
    ]
    store(_run("r1", items=items), tmp_root)
    text = rebuild_index(tmp_root).read_text()
    header, row = text.splitlines()[0].split(","), text.splitlines()[1].split(",")
    p50 = float(row[header.index("latencyMsP50")])
    p95 = float(row[header.index("latencyMsP95")])
    # R-13 is closed and the definition IS pinned now (§4 S1e Table C): there is one percentile
    # in the package, `stats.percentile`, and it is Hyndman-Fan type 1 (`-ml` §11.2). Over
    # `1..100` the ranks are `ceil(1/2 * 100) = 50` and `ceil(19/20 * 100) = 95`, so the two cells
    # are exact observations of the sample and no longer need a tolerance band. The band was there
    # because two copies of the estimator disagreed; what it protected against — the two columns
    # being the same percentile — is now pinned by the values themselves.
    assert p50 == 50.0
    assert p95 == 95.0


def test_the_index_percentile_is_the_one_in_stats_and_not_a_second_copy() -> None:
    """`-ml` §11.10(3) — asserted as **identity**, not as equal behaviour (§4 S1e Table C).

    Two copies of this formula is what let `latencyMsP95` be computed at the 50th percentile and
    stay green (review M27), so the check that closes it has to deny the second copy rather than
    compare two implementations' output on a sample that happens to agree.
    """
    assert results_module.percentile is stats.percentile
    assert not re.search(
        r"def [A-Za-z_]*(percentile|quantile)", inspect.getsource(results_module)
    ), "`results.py` may define no percentile or quantile helper of its own"


def test_the_index_valid_column_distinguishes_a_usable_record_from_a_quarantined_one(
    tmp_root,
) -> None:
    """Review P3-9 — hardcoding `_index_row(run, valid=True)` survived the whole suite, so a
    regression marking every stored record usable would not have been caught. `index.csv` is the
    only place an operator sees which of a history's runs are usable at a glance.

    The invalid record has to be written by hand: `store()` refuses an incomplete fingerprint and
    has no bypass flag (§3.4.5 point 1), so a blanked field can only arrive by editing the file —
    which is exactly the provenance `load_history` quarantines and the index must flag.
    """
    store(_run("good"), tmp_root)
    store(_run("hand_edited"), tmp_root)
    path = tmp_root / "results" / "runs" / "hand_edited.json"
    raw = json.loads(path.read_text())
    raw["fingerprint"]["kvCacheSetting"] = ""
    path.write_text(json.dumps(raw))

    text = rebuild_index(tmp_root).read_text()
    lines = text.splitlines()
    header = lines[0].split(",")
    by_run = {row.split(",")[header.index("runId")]: row.split(",") for row in lines[1:]}
    assert by_run["good"][header.index("valid")] == "yes"
    assert by_run["hand_edited"][header.index("valid")] == "no"


# --- §4 S1e Table F: the continuous carrier — `measures`, `scored_value`, `DistributionSummary` -


def _citem(*, scoreable: dict, measures: dict, counts: dict | None = None) -> ItemResult:
    return ItemResult(
        itemId="i1", pairingKey=("i1",), outcome="pass", scoreable=scoreable,
        counts=counts or {}, latencyMs=None, measures=measures, detail={},
    )


def test_scored_outcome_raises_on_a_measures_resident_metric() -> None:
    """DC-13(a) — the test that fails if the booleanisation is ever reintroduced.

    A metric declared scoreable whose value lives in `measures` (a continuous instrument) has no
    boolean outcome. Before this table, `scored_outcome` would have read the count default and
    booleanised it; now it refuses loudly, which is what converts "wrong when a pack finally
    declares a continuous verdict metric" into "refuses immediately" — the family loop calls
    `scored_outcome` unconditionally today, so this raise is the whole of that guarantee until a
    kind-aware branch is built (§4 S1e Table F, `-ml` v1.15 §3.2d).
    """
    it = _citem(scoreable={"mrr": True}, measures={"mrr": 0.5})
    with pytest.raises(MetricKindError) as excinfo:
        it.scored_outcome("mrr")
    assert "mrr" in str(excinfo.value) and "measures" in str(excinfo.value)


def test_scored_outcome_is_unaffected_for_a_metric_that_lives_in_counts() -> None:
    """The raise is scoped to `measures`-resident metrics only — the binary path is untouched."""
    it = ItemResult(itemId="i1", pairingKey=("i1",), outcome="pass", scoreable={"m": True},
                     counts={"m": 1}, latencyMs=None, measures={}, detail={})
    assert it.scored_outcome("m") is True


def test_scored_value_has_the_same_three_states_as_scored_outcome() -> None:
    """§4 S1e Table F — `scored_value`'s contract mirrors `scored_outcome`'s, over `measures`."""
    assert _citem(scoreable={}, measures={}).scored_value("mrr") is None
    assert _citem(scoreable={"mrr": False}, measures={}).scored_value("mrr") is None
    assert _citem(scoreable={"mrr": True}, measures={"mrr": 0.75}).scored_value("mrr") == 0.75


def test_scored_value_refuses_a_declared_metric_with_no_measure() -> None:
    """The `measures`-side sibling of the count refusal — an absent measure is not a zero."""
    with pytest.raises(IncompleteItemRecord) as excinfo:
        _citem(scoreable={"mrr": True}, measures={}).scored_value("mrr")
    assert "mrr" in str(excinfo.value) and "i1" in str(excinfo.value)


def test_a_metric_name_present_in_both_maps_is_refused_at_construction() -> None:
    """DC-13(c) — instrument selection is total: a name lives in `counts` XOR `measures`, and the
    ambiguity is refused at construction rather than resolved by which reader is called first."""
    with pytest.raises(MetricKindError) as excinfo:
        _citem(scoreable={"m": True}, measures={"m": 0.5}, counts={"m": 1})
    assert "m" in str(excinfo.value)


def test_a_non_finite_measure_is_refused_at_construction() -> None:
    """DC-13(c) — a NaN or infinity would otherwise propagate through a mean or a percentile and
    arrive as a rendered interval rather than as an error."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(NonFiniteMeasure):
            _citem(scoreable={"m": True}, measures={"m": bad})


def test_a_measure_of_zero_survives_a_round_trip_as_zero_not_absent() -> None:
    """DC-13(b) — the absent-never-zero boundary, asserted on both sides: a `0.0` measurement
    round-trips as `0.0`, and the same item with the key missing raises `IncompleteItemRecord`,
    never reads as a zero. This is the boundary that stops a query nobody judged from being read
    as a query that retrieved nothing relevant."""
    original = _citem(scoreable={"mrr": True}, measures={"mrr": 0.0})
    restored = ItemResult.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.measures["mrr"] == 0.0
    assert restored.scored_value("mrr") == 0.0

    missing = ItemResult.from_dict(
        {**original.to_dict(), "measures": {}}
    )
    with pytest.raises(IncompleteItemRecord):
        missing.scored_value("mrr")


def test_measures_defaults_empty_and_from_dict_reads_a_missing_key_the_same_way() -> None:
    """`from_dict` treats a missing `measures` key as a reader's compatibility rule (§3.4.3): a
    record written before this field existed carries none, exactly as a constructor default
    would — but the two are different mechanisms, so both are asserted."""
    assert ItemResult(
        itemId="i1", pairingKey=("i1",), outcome="pass", scoreable={}, counts={}, latencyMs=None,
        detail={},
    ).measures == {}
    d = _citem(scoreable={}, measures={}).to_dict()
    del d["measures"]
    assert ItemResult.from_dict(d).measures == {}


def test_continuous_metric_requires_support_with_no_default() -> None:
    """§4 S1e Table F — the same discipline `BinaryMetric.unit` already has: the value a forgetful
    caller would omit is exactly the one that licenses the paired bootstrap's clamp."""
    import dataclasses

    field_ = {f.name: f for f in dataclasses.fields(ContinuousMetric)}["support"]
    assert field_.default is dataclasses.MISSING
    assert field_.default_factory is dataclasses.MISSING
    with pytest.raises(TypeError):
        ContinuousMetric(name="mrr", mean=0.5, n=10)  # type: ignore[call-arg]


def test_retrieval_aggregates_named_metrics_returns_separation_raw_and_z() -> None:
    """§4 S1e Table F — this one addition is what makes `sep_z` reach a table at all; today it
    reaches none, whatever the scorer computes."""
    sep_raw = DistributionSummary(
        name="separationRaw", median=0.1, p10=0.02, n=40, unit="query", support=None
    )
    sep_z = DistributionSummary(
        name="separationZ", median=1.2, p10=0.3, n=40, unit="query", support=None
    )
    agg = RetrievalAggregates(separationRaw=sep_raw, separationZ=sep_z)
    assert agg.named_metrics() == (sep_raw, sep_z)


def test_distribution_summary_survives_a_round_trip_as_a_distribution_summary() -> None:
    """DC-13(f) — the type is asserted, not merely its fields: this is the assertion that fails on
    `_decode`'s silent pass-through of an unknown tag, which would otherwise hand back a `dict`."""
    original = DistributionSummary(
        name="separationZ", median=1.5, p10=-0.3, n=12, unit="query", support=(-5.0, 5.0)
    )
    d = _metric_to_dict(original)
    assert d == {
        "type": "distribution", "name": "separationZ", "median": 1.5, "p10": -0.3, "n": 12,
        "unit": "query", "support": [-5.0, 5.0],
    }
    restored = _metric_from_dict(json.loads(json.dumps(d)))
    assert type(restored) is DistributionSummary
    assert restored == original


@pytest.mark.parametrize("kind", ["continuous", "distribution"])
def test_support_round_trips_none_as_none_and_a_pair_as_a_tuple(kind: str) -> None:
    """DC-13(f) — `support` round-trips on **both** continuous types: `None` survives as `None`
    and a bounded pair as a `tuple`, never a `list`."""
    if kind == "continuous":
        bounded = ContinuousMetric(name="mrr", mean=0.5, n=10, support=(0.0, 1.0))
        unbounded = ContinuousMetric(name="sep_raw", mean=0.1, n=10, support=None)
    else:
        bounded = DistributionSummary(
            name="mrr", median=0.5, p10=0.1, n=10, unit="query", support=(0.0, 1.0)
        )
        unbounded = DistributionSummary(
            name="sep_raw", median=0.1, p10=-0.2, n=10, unit="query", support=None
        )
    for original in (bounded, unbounded):
        restored = _metric_from_dict(json.loads(json.dumps(_metric_to_dict(original))))
        assert restored.support == original.support
        assert original.support is None or isinstance(restored.support, tuple)


@pytest.mark.parametrize("kind", ["continuous", "distribution"])
def test_a_stored_metric_dict_with_no_support_key_raises(kind: str) -> None:
    """DC-13(f) — `support` is read with no `.get` fallback, the same rule `BinaryMetric.unit`
    already states: a scorer's declaration is not recomputed by a reader."""
    d = (
        {"type": "continuous", "name": "mrr", "mean": 0.5, "n": 10}
        if kind == "continuous"
        else {"type": "distribution", "name": "mrr", "median": 0.5, "p10": 0.1, "n": 10,
              "unit": "query"}
    )
    with pytest.raises(KeyError):
        _metric_from_dict(d)


def test_an_unrecognised_metric_type_tag_raises_rather_than_returning_a_raw_dict() -> None:
    """§4 S1e Table F, plan-gate P7-1 — this is the site that fails silently today: a third tag
    falls through both `if`s and is returned as a raw `dict`, so a field typed
    `DistributionSummary | None` would hold a `dict` and every later reader would be wrong about a
    type nothing checked. The tag set has one home, and an unrecognised member of it raises."""
    with pytest.raises(ValueError, match="unrecognised metric type"):
        _metric_from_dict({"type": "quantile-sketch", "name": "x"})


def test_a_record_with_an_unrecognised_metric_type_is_quarantined_as_unparseable(tmp_root) -> None:
    """The raise's read-time behaviour: `load_history` surfaces it as `unparseable`, exactly as a
    `KeyError` in `from_dict` already does (`results.py:327`'s own comment)."""
    original = _run("r1", aggregates=RetrievalAggregates(
        mrr=ContinuousMetric(name="mrr", mean=0.5, n=1, support=(0.0, 1.0))
    ))
    path = store(original, tmp_root)
    raw = json.loads(path.read_text())
    raw["aggregates"]["mrr"]["type"] = "quantile-sketch"
    path.write_text(json.dumps(raw))

    valid, invalid = load_history(tmp_root, packId=PACK)
    assert valid == []
    assert [r.reason for r in invalid] == ["unparseable"]


def test_index_row_renders_a_distribution_summary_with_its_median_labelled_p50(tmp_root) -> None:
    """§4 S1e Table F — the shipped bare `else` reads `.mean` and raises `AttributeError` on this
    type; the fix renders the median under a `p50` label, because the cell's continuous form is a
    bare number and a median printed like a mean is §3.5's defect one column over. `p10` is not in
    this cell — the index is a per-run locator, and the Arms table is where a distribution prints.
    """
    agg = RetrievalAggregates(
        separationZ=DistributionSummary(
            name="separationZ", median=1.234, p10=-0.5, n=40, unit="query", support=None
        )
    )
    store(_run("r1", aggregates=agg), tmp_root)
    text = rebuild_index(tmp_root).read_text()
    row = text.splitlines()[1]
    assert "separationZ=p50 1.2340" in row
    assert "p10" not in row


# --------------------------------------------------------------------------------------------
# `_METRIC_DECODERS` and `INDEX_COLUMNS` — two tables pinned only against themselves
# (impl review Pass 16, P16-2 and P16-3)
# --------------------------------------------------------------------------------------------
#
# Both reddened on a shrink and went green on a **widen**, so both read as covered and neither
# was: a fixture happened to use the member that was deleted, and nothing refused an added one.


#: One instance of every member of the `MetricValue` union, hand-written — never built by
#: iterating the union or the decoder table, because parametrising from the thing under test can
#: only ever lose a case (the `_ALL_AGGREGATE_CLASSES` precedent above, review M-4). The
#: assertion below is what refuses a fourth member added to the union and forgotten here.
_ONE_OF_EVERY_METRIC_KIND = (
    BinaryMetric(name="falseAdvanceRate", successes=7, n=40, unit="item"),
    ContinuousMetric(name="mrr", mean=0.5, n=38, support=(0.0, 1.0)),
    DistributionSummary(name="separationZ", median=1.2, p10=0.3, n=40, unit="query", support=None),
)


def test_the_metric_decoders_cover_exactly_the_metric_kinds_the_encoder_emits() -> None:
    """`_METRIC_DECODERS` is the `"type"` tag set's one home, and the *encoder* is the second
    declaration of the same set — `_metric_to_dict` writes the tag, `_metric_from_dict` looks it
    up, and nothing bound them.

    The widen is the consequential direction and it was open: a metric kind added to the union
    and the encoder without a decoder is a `KeyError` on read, which `load_history` quarantines
    as `unparseable` — a record this build wrote, refused by this build, reported as damage. The
    shrink is the mirror: a decoder for a tag nothing can emit.
    """
    from typing import get_args

    from modelbench.results import MetricValue

    assert {type(m) for m in _ONE_OF_EVERY_METRIC_KIND} == set(get_args(MetricValue))
    assert {_metric_to_dict(m)["type"] for m in _ONE_OF_EVERY_METRIC_KIND} == set(_METRIC_DECODERS)


@pytest.mark.parametrize("metric", _ONE_OF_EVERY_METRIC_KIND, ids=lambda m: type(m).__name__)
def test_each_metric_kind_decodes_back_to_its_own_class(metric) -> None:
    """The per-member consequence behind the domain equality: each tag's decoder rebuilds *that*
    class, so a decoder wired to the wrong constructor — a `distribution` read back as a
    `continuous`, which is a median read as a mean (§4 S1e Table F) — reddens here and not only
    in the one round-trip test that happens to use that kind."""
    assert _metric_from_dict(_metric_to_dict(metric)) == metric


#: Transcribed by hand, in order: the fourteen published columns of `results/index.csv` (plan
#: §3.5). `index.csv` is a *consumed artifact* — regenerable, but read by anything pointed at the
#: results directory — so its column names and their order are the contract, and the emitted
#: header was only ever asserted against `INDEX_COLUMNS` itself, which is true of any tuple.
_INDEX_COLUMNS_PER_PLAN_3_5 = (
    "runId", "date", "role", "packId", "packVersion", "packContentHash8", "modelKey",
    "quantization", "armKind", "n", "headlineMetrics", "latencyMsP50", "latencyMsP95", "valid",
)


def test_the_index_columns_are_exactly_the_published_fourteen_in_order() -> None:
    """Leg 1 — the declaration, against the transcript. Order is asserted, not just membership:
    a positional CSV reader is broken by a reordering that a set comparison calls identical."""
    assert INDEX_COLUMNS == _INDEX_COLUMNS_PER_PLAN_3_5


def test_the_written_index_header_is_the_published_fourteen_and_every_cell_is_filled(
    tmp_root,
) -> None:
    """Leg 2 — the artifact on disk, against the same transcript rather than against the
    constant that produced it.

    The second assertion is what catches the widen from the other side: `csv.DictWriter` fills a
    fieldname `_index_row` never emits with the empty string, so a column added to
    `INDEX_COLUMNS` alone appears in every row as a silent blank rather than an error.
    """
    store(_run("r1"), tmp_root)
    lines = rebuild_index(tmp_root).read_text().splitlines()
    assert lines[0].split(",") == list(_INDEX_COLUMNS_PER_PLAN_3_5)
    from modelbench.results import _index_row

    emitted = _index_row(_run("r1"), valid=True)
    assert tuple(emitted) == _INDEX_COLUMNS_PER_PLAN_3_5
