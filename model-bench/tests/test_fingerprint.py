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


# --- M-4: the contracts are pinned against literals, not against themselves ----------------------

#: Plan §3.4.2's schema-1 `model` field set, **transcribed here by hand** rather than derived from
#: the module under test. Review M-4: the suite parametrized over `REQUIRED_BY_SCHEMA[1]["model"]`
#: and `FORBIDDEN_BY_ARM_KIND["deterministic"]`, so deleting an entry deleted its *test case* rather
#: than failing one — dropping `loadedContextLength` gave **230 passed**, three cases silently
#: uncollected and zero failures. A test that parametrizes over the thing it is testing is not a
#: test; the per-field loops below stay, but a shrinking set now fails loudly first.
EXPECTED_MODEL_SCHEMA_1 = {
    "modelKey": "nonempty",
    "modelPublisher": "nonempty",
    "arch": "nonempty",
    "quantization": "nonempty",
    "compatibilityType": "nonempty",
    "maxContextLength": "nonempty",
    "loadedContextLength": "nonempty",
    "modelType": "nonempty",
    "modelCapabilities": "present",
    "modelCapabilitiesPresent": "present",
    "runtimeName": "nonempty",
    "runtimeVersion": "nonempty",
    "lmsCliCommit": "nonempty",
    "residentModelsAtStart": "present",
    "residentModelsAtEnd": "present",
    "temperature": "present",
    "maxTokens": "nonempty",
    "packId": "nonempty",
    "packVersion": "nonempty",
    "packContentHash": "nonempty",
    "benchVersion": "nonempty",
    "benchSchemaVersion": "nonempty",
    "pythonVersion": "nonempty",
    "hostOs": "nonempty",
    "startedAt": "nonempty",
    "endedAt": "nonempty",
    "lmStudioAppVersion": "nonempty",
    "kvCacheSetting": "nonempty",
    "hostRamGb": "nonempty",
    "otherResidentWorkloads": "present",
}

#: Plan §3.4.1's schema-1 `deterministic` field set — a deterministic arm is reproducible from
#: `(packContentHash, armParametersHash, benchVersion)` alone, so it carries pack, tool and host
#: identity and nothing else.
EXPECTED_DETERMINISTIC_SCHEMA_1 = {
    "armId": "nonempty",
    "armParametersHash": "nonempty",
    "packId": "nonempty",
    "packVersion": "nonempty",
    "packContentHash": "nonempty",
    "benchVersion": "nonempty",
    "benchSchemaVersion": "nonempty",
    "pythonVersion": "nonempty",
    "hostOs": "nonempty",
    "startedAt": "nonempty",
    "endedAt": "nonempty",
}


@pytest.mark.parametrize(
    "arm_kind,expected",
    [("model", EXPECTED_MODEL_SCHEMA_1), ("deterministic", EXPECTED_DETERMINISTIC_SCHEMA_1)],
)
def test_the_required_field_contract_is_pinned_by_name_and_by_tier(arm_kind, expected) -> None:
    """Both halves matter: a field that vanishes and a field whose tier is silently relaxed from
    `nonempty` to `present` are the same defect from a reader's point of view (plan §3.4.2)."""
    actual = {name: spec.tier for name, spec in REQUIRED_BY_SCHEMA[1][arm_kind].items()}
    assert actual == expected


def test_the_schema_map_declares_exactly_the_two_arm_kinds() -> None:
    assert set(REQUIRED_BY_SCHEMA) == {1}
    assert set(REQUIRED_BY_SCHEMA[1]) == {"model", "deterministic"}


def test_the_forbidden_sets_are_pinned_against_literals() -> None:
    """The `deterministic` set is the author's **decision 3**, upheld by the gate: §3.4.1's prose
    ("forbids every model field") and its rationale both cover `modelType`,
    `modelCapabilities` and `modelCapabilitiesPresent`, which its enumeration omitted.

    Deriving the set as `frozenset(_MODEL_SCHEMA_1) - frozenset(_DETERMINISTIC_SCHEMA_1)` is the
    better shape — it cannot go stale when a model field is added at schema 2 — but subtracting the
    three catalog fields from it was **green** (230 passed, three cases uncollected). The decision
    was held in place by nothing but the set-difference expression itself, so a future edit
    following the plan's literal enumeration would have reverted it silently.
    """
    assert FORBIDDEN_BY_ARM_KIND["deterministic"] == frozenset(
        set(EXPECTED_MODEL_SCHEMA_1) - set(EXPECTED_DETERMINISTIC_SCHEMA_1)
    )
    assert {"modelType", "modelCapabilities", "modelCapabilitiesPresent"} <= (
        FORBIDDEN_BY_ARM_KIND["deterministic"]
    )
    assert FORBIDDEN_BY_ARM_KIND["model"] == frozenset({"armId", "armParametersHash"})
    assert set(FORBIDDEN_BY_ARM_KIND) == {"model", "deterministic"}


# --- n-2: a frozen record whose hash ignores every value ----------------------------------------


def test_two_fingerprints_differing_in_every_value_do_not_collide() -> None:
    """Review n-2 — `__hash__` hashed only the sorted *field names*, so two fingerprints with the
    same keys and entirely different values shared a hash bucket. Correct (equal objects hash
    equal) but degenerate, in a frozen dataclass whose point is identity."""
    a = Fingerprint(armKind="model", fields=model_fields())
    b = Fingerprint(
        armKind="model",
        fields=model_fields(modelKey="other", quantization="Q8_0", hostRamGb=64),
    )
    assert a != b
    assert hash(a) != hash(b)
    assert len({a, b}) == 2
    assert hash(a) == hash(Fingerprint(armKind="model", fields=model_fields()))


def test_a_frozen_fingerprint_does_not_share_its_mapping_with_the_caller() -> None:
    """`frozen=True` on a dataclass holding a live `dict` the caller still references is frozen in
    name only (review n-2)."""
    supplied = model_fields()
    fp = Fingerprint(armKind="model", fields=supplied)
    supplied["kvCacheSetting"] = ""
    assert fp.get("kvCacheSetting") == "f16"
    assert fp.validate() == []
    with pytest.raises(TypeError):
        fp.fields["kvCacheSetting"] = ""  # type: ignore[index]
