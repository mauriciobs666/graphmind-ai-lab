"""§5 test 1 — `fingerprint.validate()`.

Every assertion here traces to `docs/plans/small-model-benchmarking.md` §3.4.1 (the two
discriminators and the derived `armProfile`), §3.4.2 (the two required-field tiers and the
absent/empty/null distinction), §3.4.3 (`benchSchemaVersion`) and §3.4.4a (the residency element
shape). This is the S1 done-condition 1 and 6 surface.
"""

from __future__ import annotations

import pytest
from conftest import deterministic_fields, embeddings_fields, model_fields

from modelbench.fingerprint import (
    ARM_KINDS,
    CALL_SURFACES,
    FORBIDDEN_BY_ARM_PROFILE,
    REQUIRED_BY_SCHEMA,
    FieldProblem,
    Fingerprint,
)

#: The three schema-1 profiles, paired with the fixture that builds a *valid* record for each.
#: Written here rather than derived from `REQUIRED_BY_SCHEMA` so that a profile appearing or
#: disappearing fails a test rather than silently changing how many run (review M-4).
PROFILE_FIXTURES = {
    "model:chat": model_fields,
    "model:embeddings": embeddings_fields,
    "deterministic": deterministic_fields,
}

MODEL_PROFILES = ("model:chat", "model:embeddings")

#: `(profile, field)` for every required field of both model profiles — 30 + 26 cases.
MODEL_PROFILE_FIELDS = [
    (profile, field)
    for profile in MODEL_PROFILES
    for field in sorted(REQUIRED_BY_SCHEMA[1][profile])
]


def _problems(arm_kind: str, fields: dict, call_surface: str | None = "chat") -> list[FieldProblem]:
    """Validate one record. `arm_kind` stays two-valued (§3.4.1); the surface picks the profile."""
    if arm_kind == "deterministic":
        call_surface = None
    return Fingerprint(armKind=arm_kind, callSurface=call_surface, fields=fields).validate()


def _profile_problems(profile: str, **overrides) -> list[FieldProblem]:
    """Validate the valid record for `profile`, with `overrides` applied to its fields."""
    arm_kind, _, call_surface = profile.partition(":")
    return _problems(
        arm_kind, PROFILE_FIXTURES[profile](**overrides), call_surface=call_surface or None
    )


def test_a_complete_model_fingerprint_is_valid() -> None:
    assert _problems("model", model_fields()) == []


def test_a_complete_deterministic_fingerprint_is_valid() -> None:
    """Plan §3.4.1 — a BM25 arm has no model, quantization or runtime, and still validates."""
    assert _problems("deterministic", deterministic_fields()) == []


@pytest.mark.parametrize("profile,field", MODEL_PROFILE_FIELDS)
def test_every_required_model_field_is_named_when_absent(profile: str, field: str) -> None:
    """§5 test 1 — 'one test per required field: blank it, assert it is named'.

    Both model profiles get the same treatment (§4 S1e Table B): `model:embeddings` has its own
    26-field contract, and a profile whose required set is never exercised field-by-field is a
    contract nothing checks.
    """
    problems = _profile_problems(profile, **{field: ...})
    assert FieldProblem(field=field, reason="absent") in problems


@pytest.mark.parametrize("profile,field", MODEL_PROFILE_FIELDS)
def test_null_is_invalid_in_either_tier(profile: str, field: str) -> None:
    """Plan §3.4.2 — `null` is the shape of 'we did not capture this', which FR-7 refuses."""
    problems = _profile_problems(profile, **{field: None})
    assert FieldProblem(field=field, reason="null") in problems


def test_empty_list_is_valid_for_a_required_present_field() -> None:
    """The residency probe returns [] on a clean box — the correct, informative value (§3.4.2)."""
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


@pytest.mark.parametrize("field", sorted(FORBIDDEN_BY_ARM_PROFILE["deterministic"]))
def test_each_forbidden_field_is_named_individually(field: str) -> None:
    problems = _problems("deterministic", deterministic_fields(**{field: "anything"}))
    assert FieldProblem(field=field, reason="forbidden") in problems


def test_model_arm_forbids_the_deterministic_arm_parameters() -> None:
    problems = _problems("model", model_fields(armParametersHash="b" * 64))
    assert FieldProblem(field="armParametersHash", reason="forbidden") in problems


def test_a_complete_embeddings_fingerprint_is_valid() -> None:
    """S1 done-condition 6 — `model:embeddings` has its own 26-field contract (§3.4.2)."""
    assert _profile_problems("model:embeddings") == []


def test_a_model_embeddings_record_without_runtime_name_validates() -> None:
    """S1 done-condition 6 — that field is not in this profile's required set, so it is not
    absent; the pre-v1.9 sentence "a `model` record missing `runtimeName` fails" was false here."""
    assert _profile_problems("model:embeddings") == []
    assert "runtimeName" not in REQUIRED_BY_SCHEMA[1]["model:embeddings"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtimeName", "llama.cpp"),
        ("runtimeVersion", "1.52.0"),
        ("temperature", 0.0),
        ("maxTokens", 1024),
    ],
)
def test_a_model_embeddings_record_carrying_a_chat_only_field_is_forbidden(field, value) -> None:
    """§3.4.1's derivation earning its keep: an embeddings call has no `runtime` object to observe
    and no sampling parameters to obey, so a record carrying either is claiming something it
    cannot have measured. Nobody wrote these four names down — the set operation did."""
    problems = _profile_problems("model:embeddings", **{field: value})
    assert FieldProblem(field=field, reason="forbidden") in problems


def test_a_model_record_without_a_call_surface_fails_before_any_mapping_is_consulted() -> None:
    """S1 done-condition 6's discriminator case (plan-gate P4-2's mechanical failure).

    The record below is missing every one of the thirty required fields as well, so if the
    required-set mapping had been consulted the answer would be thirty `absent` problems with the
    true one buried among them. Without a `callSurface` there is no profile, so there is no
    contract to report the fields against at all.
    """
    fp = Fingerprint(armKind="model", callSurface=None, fields={"benchSchemaVersion": 1})
    assert fp.validate() == [FieldProblem(field="callSurface", reason="absent")]

    # ...and it is *absent*, not empty or null: the same three-state discipline §3.4.2 applies to
    # the fields applies to the discriminator that decides which fields are read.
    blank = Fingerprint(armKind="model", callSurface="", fields=model_fields())
    assert blank.validate() == [FieldProblem(field="callSurface", reason="absent")]


def test_an_unknown_call_surface_is_reported_rather_than_crashing() -> None:
    """A surface this build has never seen is refused, never guessed at — it would otherwise
    resolve to a profile key no mapping carries and raise out of `validate()`."""
    fp = Fingerprint(armKind="model", callSurface="telepathy", fields=model_fields())
    assert fp.validate() == [FieldProblem(field="callSurface", reason="unknown")]


def test_a_deterministic_record_carrying_a_call_surface_is_refused() -> None:
    """`callSurface` is `None` **iff** `armKind == "deterministic"` (§3.4.1): that arm calls no
    surface, so a value here claims one was called."""
    fp = Fingerprint(armKind="deterministic", callSurface="chat", fields=deterministic_fields())
    assert fp.validate() == [FieldProblem(field="callSurface", reason="forbidden")]


def test_arm_kind_model_is_still_a_valid_membership_answer_after_the_re_key() -> None:
    """§4 S1e Table B — the assertion that catches the mechanical failure: a re-key that leaves
    `ARM_KINDS` derived from the profile mapping refuses *every* model record."""
    assert "model" in ARM_KINDS
    assert _problems("model", model_fields()) == []
    assert _profile_problems("model:embeddings") == []


# --- S1 done-condition 1: the residency **element** shape (§3.4.2, §3.4.4a) ----------------------


@pytest.mark.parametrize("field", ["residentModelsAtStart", "residentModelsAtEnd"])
@pytest.mark.parametrize("extra", ["modelKey", "quantization", "loadedAt"])
def test_a_residency_element_carrying_any_other_key_is_refused(field: str, extra: str) -> None:
    """S1 done-condition 1's element-shape check — the structural fix for the one Table A site
    that fails *silently*.

    §3.4.4a's element is `{id, state}`; the retired `lms ps --json` element, whose only source
    plan v1.8 removed, carried two other keys instead. The field's tier is `present`, which checks
    presence and **never element shape**, so without this check the stale fixture validates, ships
    green and travels into S2, where `residency()` emits `{id, state}` and the two disagree with
    nothing to catch them.

    The check is over the element's **whole key set**, which is what makes it refuse *either*
    retired key — including `modelKey`, which can have no grep residual of its own because it
    keeps its meaning as a required field of the very same record (§7 rule 5(b)'s named
    alternative, and §4 S1e Table A's row). The other retired key is asserted through the rule
    rather than by name: Table A's second residual retires that token from the tree to zero, so
    spelling it here — even in a comment — would leave the residual standing at one.
    """
    element = {"id": "qwen/qwen3-4b-2507", "state": "loaded", extra: "whatever"}
    problems = _problems("model", model_fields(**{field: [element]}))
    assert FieldProblem(field=f"{field}[0].{extra}", reason="forbidden") in problems


@pytest.mark.parametrize(
    "element",
    [
        {"modelKey": "qwen/qwen3-4b-2507", "state": "loaded"},
        {"id": "qwen/qwen3-4b-2507", "bytesOnDisk": 2 << 30},
    ],
    ids=["retired-identity-key-kept", "retired-size-key-kept"],
)
def test_a_half_swapped_residency_element_is_invalid(element: dict) -> None:
    """Each of these swaps one key of the retired pair and keeps the other — the half-application
    §7 rule 5(b) forbids a table's checks from passing. Neither is merely unused; both are
    refused, so a fixture edit that stopped halfway fails here rather than shipping green."""
    assert _problems("model", model_fields(residentModelsAtEnd=[element])) != []


def test_the_current_residency_element_shape_is_valid() -> None:
    """`{id, state}` with the **literal state string** kept — not normalised to a boolean, because
    any state other than the one the 2026-09-03 probe observed is a value this plan has not seen."""
    element = {"id": "qwen/qwen3-4b-2507", "state": "loaded"}
    assert _problems("model", model_fields(residentModelsAtEnd=[element])) == []
    assert _problems("model", model_fields(residentModelsAtEnd=[])) == []


@pytest.mark.parametrize(
    "element,expected",
    [
        ({"id": "", "state": "loaded"}, FieldProblem("residentModelsAtEnd[0].id", "empty")),
        ({"id": "m", "state": ""}, FieldProblem("residentModelsAtEnd[0].state", "empty")),
        ({"id": None, "state": "loaded"}, FieldProblem("residentModelsAtEnd[0].id", "null")),
        ({"id": "m", "state": True}, FieldProblem("residentModelsAtEnd[0].state", "unknown")),
    ],
)
def test_a_residency_element_needs_two_non_empty_strings(element, expected) -> None:
    """`[]` is a real answer for the *snapshot*; an *element* saying nothing is not (§3.4.2's
    absent-versus-empty rule, one level down)."""
    problems = _problems("model", model_fields(residentModelsAtEnd=[element]))
    assert expected in problems


def test_a_residency_snapshot_that_is_not_a_list_of_mappings_is_named() -> None:
    assert FieldProblem("residentModelsAtEnd", "unknown") in _problems(
        "model", model_fields(residentModelsAtEnd={"id": "m", "state": "loaded"})
    )
    assert FieldProblem("residentModelsAtEnd[0]", "unknown") in _problems(
        "model", model_fields(residentModelsAtEnd=["qwen/qwen3-4b-2507"])
    )


def test_a_model_chat_record_missing_runtime_name_fails() -> None:
    """S1 done-condition 6 — absent on `model:chat`, where the field is required."""
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


@pytest.mark.parametrize("profile", sorted(PROFILE_FIXTURES))
def test_round_trips_through_a_dict(profile: str) -> None:
    arm_kind, _, call_surface = profile.partition(":")
    fp = Fingerprint(
        armKind=arm_kind,
        callSurface=call_surface or None,
        fields=PROFILE_FIXTURES[profile](),
    )
    assert Fingerprint.from_dict(fp.to_dict()) == fp
    assert fp.to_dict()["armKind"] == arm_kind
    assert fp.armProfile == profile
    # A discriminator that survived into `fields` would land in every profile's forbidden set,
    # which is the failure `from_dict` strips both names to prevent.
    assert "armKind" not in fp.fields and "callSurface" not in fp.fields
    assert Fingerprint.from_dict(fp.to_dict()).validate() == []


# --- M-4: the contracts are pinned against literals, not against themselves ----------------------

#: Plan §3.4.2's schema-1 `model:chat` field set, **transcribed here by hand** rather than derived
#: from the module under test. Review M-4: the suite parametrized over
#: `REQUIRED_BY_SCHEMA[1]["model:chat"]` and `FORBIDDEN_BY_ARM_PROFILE["deterministic"]`, so
#: deleting an entry deleted its *test case* rather
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
    "residencySource": "nonempty",
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

#: Plan §3.4.2's schema-1 `model:embeddings` field set — `model:chat`'s 30 minus `runtimeName`,
#: `runtimeVersion`, `temperature` and `maxTokens`, 26 in total. **Transcribed by hand, and
#: deliberately not derived from `EXPECTED_MODEL_SCHEMA_1`**: deriving one literal from the other
#: is the M-4 defect these literals exist to prevent — the production set is already a derivation,
#: and a test that repeats it checks nothing.
EXPECTED_EMBEDDINGS_SCHEMA_1 = {
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
    "residencySource": "nonempty",
    "residentModelsAtStart": "present",
    "residentModelsAtEnd": "present",
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
    "profile,expected",
    [
        ("model:chat", EXPECTED_MODEL_SCHEMA_1),
        ("model:embeddings", EXPECTED_EMBEDDINGS_SCHEMA_1),
        ("deterministic", EXPECTED_DETERMINISTIC_SCHEMA_1),
    ],
)
def test_the_required_field_contract_is_pinned_by_name_and_by_tier(profile, expected) -> None:
    """Both halves matter: a field that vanishes and a field whose tier is silently relaxed from
    `nonempty` to `present` are the same defect from a reader's point of view (plan §3.4.2)."""
    actual = {name: spec.tier for name, spec in REQUIRED_BY_SCHEMA[1][profile].items()}
    assert actual == expected


def test_the_schema_map_declares_exactly_the_three_arm_profiles() -> None:
    assert set(REQUIRED_BY_SCHEMA) == {1}
    assert set(REQUIRED_BY_SCHEMA[1]) == {"model:chat", "model:embeddings", "deterministic"}


def test_the_arm_kinds_are_decoupled_from_the_profile_mapping() -> None:
    """§4 S1e Table B's `fingerprint.py:137` row, asserted by value.

    `ARM_KINDS` used to be derived from the forbidden mapping. Re-keying that mapping to profiles
    and leaving the derivation makes the members `{model:chat, model:embeddings, deterministic}`,
    so `armKind == "model"` fails the membership test in `validate()` and **every model record
    returns `FieldProblem("armKind", "unknown")` and refuses on write** — green mapping, dead
    harness. `armKind` keeps its two values; only the mapping key became a profile.
    """
    assert ARM_KINDS == frozenset({"model", "deterministic"})
    assert CALL_SURFACES == frozenset({"chat", "embeddings"})
    assert _problems("model", model_fields()) == []


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
    assert FORBIDDEN_BY_ARM_PROFILE["deterministic"] == frozenset(
        set(EXPECTED_MODEL_SCHEMA_1) - set(EXPECTED_DETERMINISTIC_SCHEMA_1)
    )
    assert {"modelType", "modelCapabilities", "modelCapabilitiesPresent"} <= (
        FORBIDDEN_BY_ARM_PROFILE["deterministic"]
    )
    assert FORBIDDEN_BY_ARM_PROFILE["model:chat"] == frozenset({"armId", "armParametersHash"})
    # The row the derivation earns its keep on (§3.4.1): nobody wrote these four names down.
    assert FORBIDDEN_BY_ARM_PROFILE["model:embeddings"] == frozenset(
        {"armId", "armParametersHash", "runtimeName", "runtimeVersion", "temperature", "maxTokens"}
    )
    assert set(FORBIDDEN_BY_ARM_PROFILE) == {"model:chat", "model:embeddings", "deterministic"}


# --- n-2: a frozen record whose hash ignores every value ----------------------------------------


def test_two_fingerprints_differing_in_every_value_do_not_collide() -> None:
    """Review n-2 — `__hash__` hashed only the sorted *field names*, so two fingerprints with the
    same keys and entirely different values shared a hash bucket. Correct (equal objects hash
    equal) but degenerate, in a frozen dataclass whose point is identity."""
    a = Fingerprint(armKind="model", callSurface="chat", fields=model_fields())
    b = Fingerprint(
        armKind="model",
        callSurface="chat",
        fields=model_fields(modelKey="other", quantization="Q8_0", hostRamGb=64),
    )
    assert a != b
    assert hash(a) != hash(b)
    assert len({a, b}) == 2
    assert hash(a) == hash(Fingerprint(armKind="model", callSurface="chat", fields=model_fields()))

    # ...and the second discriminator is part of that identity: two records with identical fields
    # but different call surfaces are read against different contracts (§3.4.1).
    embeddings = Fingerprint(armKind="model", callSurface="embeddings", fields=model_fields())
    assert a != embeddings
    assert hash(a) != hash(embeddings)


def test_a_frozen_fingerprint_does_not_share_its_mapping_with_the_caller() -> None:
    """`frozen=True` on a dataclass holding a live `dict` the caller still references is frozen in
    name only (review n-2)."""
    supplied = model_fields()
    fp = Fingerprint(armKind="model", callSurface="chat", fields=supplied)
    supplied["kvCacheSetting"] = ""
    assert fp.get("kvCacheSetting") == "f16"
    assert fp.validate() == []
    with pytest.raises(TypeError):
        fp.fields["kvCacheSetting"] = ""  # type: ignore[index]


def test_the_arm_kind_discriminator_keeps_absent_distinct_from_null() -> None:
    """Review P3-10 — the module's first principle is *"absent is not empty, and `null` is
    neither"*, and §3.4.2's three states are tested for the **fields** but not for the field that
    decides which contract the fields are read against.

    `from_dict` defaults a missing `armKind` to `""`, reported as `absent`; changing that default
    to `None` — reported as `null` — survived all 314 tests. A record with no `armKind` key and one
    that says `"armKind": null` are two different failures: the first was never written, the second
    was written by something that had the value and lost it.
    """
    absent = Fingerprint.from_dict({"benchSchemaVersion": 1})
    assert absent.validate() == [FieldProblem(field="armKind", reason="absent")]

    nulled = Fingerprint.from_dict({"armKind": None, "benchSchemaVersion": 1})
    assert nulled.validate() == [FieldProblem(field="armKind", reason="null")]

    # ...and a value that is neither is the third reason, so none of the three collapses.
    bogus = Fingerprint.from_dict({"armKind": "robot", "benchSchemaVersion": 1})
    assert bogus.validate() == [FieldProblem(field="armKind", reason="unknown")]

    # The second discriminator reads the same way. A stored model record with no `callSurface`
    # key is one written before the profile existed or by something that lost it; it has no
    # profile, so it has no contract, and saying so is the only honest answer (§3.4.1).
    surfaceless = Fingerprint.from_dict({"armKind": "model", "benchSchemaVersion": 1})
    assert surfaceless.callSurface is None
    assert surfaceless.validate() == [FieldProblem(field="callSurface", reason="absent")]
