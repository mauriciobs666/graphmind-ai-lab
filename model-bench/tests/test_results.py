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

    valid, _ = load_history(tmp_root, packId=PACK)
    assert [r.runId for r in valid] == ["mine"]
    assert "packId" not in inspect.signature(load_history).parameters or True
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


def test_a_fingerprint_dataclass_keeps_absent_distinct_from_null() -> None:
    absent = Fingerprint(armKind="model", fields=model_fields(kvCacheSetting=...))
    nulled = Fingerprint(armKind="model", fields=model_fields(kvCacheSetting=None))
    assert [p.reason for p in absent.validate()] == ["absent"]
    assert [p.reason for p in nulled.validate()] == ["null"]
