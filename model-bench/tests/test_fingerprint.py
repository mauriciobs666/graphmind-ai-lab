"""§5 test 1 — `fingerprint.validate()`.

Every assertion here traces to `docs/plans/small-model-benchmarking.md` §3.4.1 (`armKind`),
§3.4.2 (the two required-field tiers and the absent/empty/null distinction) and §3.4.3
(`benchSchemaVersion`). This is the S1 done-condition 1 and 6 surface.
"""

from __future__ import annotations

import pytest
from conftest import deterministic_fields, model_fields

from modelbench.fingerprint import (
    FORBIDDEN_BY_ARM_KIND,
    REQUIRED_BY_SCHEMA,
    FieldProblem,
    Fingerprint,
)


def _problems(arm_kind: str, fields: dict) -> list[FieldProblem]:
    return Fingerprint(armKind=arm_kind, fields=fields).validate()


def test_a_complete_model_fingerprint_is_valid() -> None:
    assert _problems("model", model_fields()) == []


def test_a_complete_deterministic_fingerprint_is_valid() -> None:
    """Plan §3.4.1 — a BM25 arm has no model, quantization or runtime, and still validates."""
    assert _problems("deterministic", deterministic_fields()) == []


@pytest.mark.parametrize("field", sorted(REQUIRED_BY_SCHEMA[1]["model"]))
def test_every_required_model_field_is_named_when_absent(field: str) -> None:
    """§5 test 1 — 'one test per required field: blank it, assert it is named'."""
    problems = _problems("model", model_fields(**{field: ...}))
    assert FieldProblem(field=field, reason="absent") in problems


@pytest.mark.parametrize("field", sorted(REQUIRED_BY_SCHEMA[1]["model"]))
def test_null_is_invalid_in_either_tier(field: str) -> None:
    """Plan §3.4.2 — `null` is the shape of 'we did not capture this', which FR-7 refuses."""
    problems = _problems("model", model_fields(**{field: None}))
    assert FieldProblem(field=field, reason="null") in problems


def test_empty_list_is_valid_for_a_required_present_field() -> None:
    """`lms ps --json` returns [] on a clean box — the correct, informative value (plan §3.4.2)."""
    assert _problems("model", model_fields(residentModelsAtStart=[])) == []


def test_temperature_zero_is_valid_for_a_required_present_field() -> None:
    """0.0 is the pinned value for four of the five packs; falsy is not missing."""
    assert _problems("model", model_fields(temperature=0.0)) == []


def test_false_is_valid_for_a_required_present_field() -> None:
    assert _problems("model", model_fields(modelCapabilitiesPresent=False)) == []


def test_empty_string_is_invalid_for_a_required_nonempty_field() -> None:
    problems = _problems("model", model_fields(modelKey=""))
    assert FieldProblem(field="modelKey", reason="empty") in problems


def test_a_blanked_attested_field_is_named() -> None:
    """AC-2's field: `kvCacheSetting` has no programmatic source, so it is the weakest link."""
    problems = _problems("model", model_fields(kvCacheSetting=""))
    assert FieldProblem(field="kvCacheSetting", reason="empty") in problems


def test_deterministic_arm_forbids_every_model_field() -> None:
    """Plan §3.4.1 — `{"modelKey": "bm25"}` is the shortcut this design exists to refuse."""
    problems = _problems("deterministic", deterministic_fields(modelKey="bm25", quantization="n/a"))
    assert FieldProblem(field="modelKey", reason="forbidden") in problems
    assert FieldProblem(field="quantization", reason="forbidden") in problems


@pytest.mark.parametrize("field", sorted(FORBIDDEN_BY_ARM_KIND["deterministic"]))
def test_each_forbidden_field_is_named_individually(field: str) -> None:
    problems = _problems("deterministic", deterministic_fields(**{field: "anything"}))
    assert FieldProblem(field=field, reason="forbidden") in problems


def test_model_arm_forbids_the_deterministic_arm_parameters() -> None:
    problems = _problems("model", model_fields(armParametersHash="b" * 64))
    assert FieldProblem(field="armParametersHash", reason="forbidden") in problems


def test_a_model_record_missing_runtime_name_fails() -> None:
    """S1 done-condition 6's third case."""
    problems = _problems("model", model_fields(runtimeName=...))
    assert FieldProblem(field="runtimeName", reason="absent") in problems


def test_an_unknown_arm_kind_is_reported_rather_than_crashing() -> None:
    problems = _problems("bm25ish", model_fields())
    assert FieldProblem(field="armKind", reason="unknown") in problems


def test_a_missing_arm_kind_is_reported() -> None:
    problems = _problems("", model_fields())
    assert FieldProblem(field="armKind", reason="absent") in problems


def test_an_unknown_schema_version_is_reported() -> None:
    """Plan §3.4.3 — a record from the *future* is the genuinely uninterpretable case."""
    problems = _problems("model", model_fields(benchSchemaVersion=99))
    assert FieldProblem(field="benchSchemaVersion", reason="unknown") in problems


def test_an_older_schema_record_validates_against_the_contract_it_was_written_under(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan §3.4.3 / §5 test 3 — the FR-3 case v1.1's 'exclude older schemas' would have deleted.

    A hypothetical field is added at schema 2; the schema-1 record, which never carried it, must
    still validate rather than be quarantined out of every comparison.
    """
    from modelbench.fingerprint import FieldSpec

    schema_2 = {
        kind: {**spec, "hypotheticalNewField": FieldSpec(tier="nonempty")}
        for kind, spec in REQUIRED_BY_SCHEMA[1].items()
    }
    monkeypatch.setitem(REQUIRED_BY_SCHEMA, 2, schema_2)

    assert _problems("model", model_fields()) == []
    assert _problems("model", model_fields(benchSchemaVersion=2)) == [
        FieldProblem(field="hypotheticalNewField", reason="absent")
    ]


def test_round_trips_through_a_dict() -> None:
    fp = Fingerprint(armKind="model", fields=model_fields())
    assert Fingerprint.from_dict(fp.to_dict()) == fp
    assert fp.to_dict()["armKind"] == "model"
