"""§5 tests 2 and 3 — the two enforcement points (plan §3.4.5, AC-2).

The centre of gravity is the *read* side: AC-2 requires a hand-edited record to be excluded when
history is loaded, not merely rejected when it was written.
"""

from __future__ import annotations

import inspect
import json

import pytest
from conftest import deterministic_fields, model_fields, run

from modelbench.fingerprint import Fingerprint
from modelbench.results import (
    BENCH_SCHEMA_VERSION,
    ClassificationAggregates,
    IncompleteItemRecord,
    InvalidFingerprint,
    ItemResult,
    RunResult,
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
    absent = Fingerprint(armKind="model", fields=model_fields(kvCacheSetting=...))
    nulled = Fingerprint(armKind="model", fields=model_fields(kvCacheSetting=None))
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
        "fingerprint": Fingerprint(armKind="model", fields=model_fields()),
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
    # The *definition* is deliberately not pinned here — nearest-rank vs interpolation is plan
    # v1.5 §6's open R-13, and `_percentile` has two copies. What is pinned is that the two columns
    # are different percentiles of the same sample, which is what the surviving mutation denied.
    assert 45.0 <= p50 <= 55.0
    assert 90.0 <= p95 <= 100.0
    assert p95 > p50


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
