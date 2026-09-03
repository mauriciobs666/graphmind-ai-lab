"""The environment fingerprint, and the rules that make FR-7 mechanical rather than aspirational.

Design: `docs/plans/small-model-benchmarking.md` §3.4 (and §3.4.1–§3.4.3). Three ideas carry the
whole module:

* **`armKind` discriminates** (§3.4.1). A BM25 reference arm has no model, no quantization and no
  runtime, so `validate()` branches on the arm kind and never on field presence. The `deterministic`
  kind *forbids* every model field: `{"modelKey": "bm25", "quantization": "n/a"}` is the shortcut a
  time-pressed implementer reaches for, and it must fail loudly on write rather than quietly become
  a sixth model in the history.
* **Absent is not empty** (§3.4.2). `residentModelsAtStart: []` is the correct value on a clean box
  and the catalog omits `capabilities` entirely for several models, so each required field declares
  a tier: `nonempty` (present *and* truthy) or `present` (`[]`, `0`, `False`, `""` all valid).
  `null` is invalid in both tiers — it is the shape of "we did not capture this", which is the one
  thing FR-7 refuses.
* **A record is validated against the contract it was written under** (§3.4.3). Adding a required
  field at schema 2 must not quarantine every record stored before it; that would be against FR-3
  directly, and the tool's whole value is that a new model's result lines up against models tested
  months ago.

Fields are held in a plain mapping rather than as dataclass attributes precisely so that a *missing
key* stays distinguishable from a key whose value is `None` — a dataclass with `None` defaults
collapses the two states §3.4.2 exists to separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, NamedTuple

ArmKind = Literal["model", "deterministic"]
Tier = Literal["nonempty", "present"]

#: Why a field failed. `unknown` covers a discriminator this build cannot interpret — an
#: unrecognized `armKind` or a `benchSchemaVersion` from the future (plan Appendix A names the
#: other four; see this module's HISTORY entry for why the fifth is needed).
ProblemReason = Literal["absent", "empty", "null", "forbidden", "unknown"]


class FieldSpec(NamedTuple):
    """How strictly one required field is checked (plan §3.4.2)."""

    tier: Tier


class FieldProblem(NamedTuple):
    """One failed field, named with its reason so a report can print it (AC-2)."""

    field: str
    reason: ProblemReason


_NONEMPTY = FieldSpec(tier="nonempty")
_PRESENT = FieldSpec(tier="present")

# --- schema 1 -------------------------------------------------------------------------------
# The auto-captured half (§3.4.2): no human input, and it cannot be wrong without the tool being
# wrong. `loadedContextLength` exists only once a model is loaded, which is why capture ordering is
# part of the contract.
_MODEL_SCHEMA_1: dict[str, FieldSpec] = {
    # model identity, verbatim from the LM Studio catalog — never a normalized alias (§2.3 R-8)
    "modelKey": _NONEMPTY,
    "modelPublisher": _NONEMPTY,
    "arch": _NONEMPTY,
    "quantization": _NONEMPTY,
    "compatibilityType": _NONEMPTY,
    "maxContextLength": _NONEMPTY,
    "loadedContextLength": _NONEMPTY,
    # the raw catalog fields the tool-calling eligibility gate (§3.6) decided on, kept so that
    # decision stays auditable from the stored record rather than re-derived from a moved catalog
    "modelType": _NONEMPTY,
    "modelCapabilities": _PRESENT,
    "modelCapabilitiesPresent": _PRESENT,
    # runtime identity, free from the /api/v0 chat route's `runtime` object
    "runtimeName": _NONEMPTY,
    "runtimeVersion": _NONEMPTY,
    "lmsCliCommit": _NONEMPTY,
    # residency: [] is the correct value on a clean box
    "residentModelsAtStart": _PRESENT,
    "residentModelsAtEnd": _PRESENT,
    # sampling settings: 0.0 is the pinned temperature for four of the five packs
    "temperature": _PRESENT,
    "maxTokens": _NONEMPTY,
    # pack and tool identity
    "packId": _NONEMPTY,
    "packVersion": _NONEMPTY,
    "packContentHash": _NONEMPTY,
    "benchVersion": _NONEMPTY,
    "benchSchemaVersion": _NONEMPTY,
    "pythonVersion": _NONEMPTY,
    "hostOs": _NONEMPTY,
    "startedAt": _NONEMPTY,
    "endedAt": _NONEMPTY,
    # the operator-attested half (§6 R-1): no programmatic source exists on this LM Studio build
    "lmStudioAppVersion": _NONEMPTY,
    "kvCacheSetting": _NONEMPTY,
    "hostRamGb": _NONEMPTY,
    "otherResidentWorkloads": _PRESENT,
}

# A deterministic arm is reproducible from (packContentHash, armParametersHash, benchVersion)
# alone, which is why host state is not merely optional for it but forbidden: recording a KV-cache
# setting beside a BM25 score would imply the score depends on it (§3.4.1).
_DETERMINISTIC_SCHEMA_1: dict[str, FieldSpec] = {
    "armId": _NONEMPTY,
    "armParametersHash": _NONEMPTY,
    "packId": _NONEMPTY,
    "packVersion": _NONEMPTY,
    "packContentHash": _NONEMPTY,
    "benchVersion": _NONEMPTY,
    "benchSchemaVersion": _NONEMPTY,
    "pythonVersion": _NONEMPTY,
    "hostOs": _NONEMPTY,
    "startedAt": _NONEMPTY,
    "endedAt": _NONEMPTY,
}

#: `{schemaVersion: {armKind: {field: FieldSpec}}}` — plan §3.4.3. A record is validated against
#: its own entry here, so an added field at a later version never invalidates an older record.
#: Mutable by design: `model-bench migrate` and the schema-2 regression test both key off it.
REQUIRED_BY_SCHEMA: dict[int, dict[str, dict[str, FieldSpec]]] = {
    1: {"model": _MODEL_SCHEMA_1, "deterministic": _DETERMINISTIC_SCHEMA_1},
}

#: What each arm kind may not carry (§3.4.1). The forbid half is the point of the discriminator.
FORBIDDEN_BY_ARM_KIND: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        # every model field, plus the four operator-attested LM Studio fields
        "deterministic": frozenset(_MODEL_SCHEMA_1) - frozenset(_DETERMINISTIC_SCHEMA_1),
        # a model run has no arm parameters to hash
        "model": frozenset({"armId", "armParametersHash"}),
    }
)

ARM_KINDS: frozenset[str] = frozenset(FORBIDDEN_BY_ARM_KIND)


@dataclass(frozen=True)
class Fingerprint:
    """One run's environment record. `armKind` discriminates; `fields` holds everything else."""

    armKind: str
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        # `frozen=True` on a dataclass holding a live `dict` the caller still references is frozen
        # in name only: the record could change under a report that had already validated it. A
        # copy behind a `MappingProxyType` makes the freeze real (review n-2).
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    @property
    def benchSchemaVersion(self) -> Any:
        return self.fields.get("benchSchemaVersion")

    def get(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)

    def validate(self) -> list[FieldProblem]:
        """Return every field problem; `[]` means valid. Never raises, never warns."""
        if self.armKind not in ARM_KINDS:
            reason: ProblemReason = "unknown"
            if self.armKind is None:
                reason = "null"
            elif not self.armKind:
                reason = "absent"
            return [FieldProblem(field="armKind", reason=reason)]

        schema = self.fields.get("benchSchemaVersion")
        if "benchSchemaVersion" not in self.fields:
            return [FieldProblem(field="benchSchemaVersion", reason="absent")]
        if schema is None:
            return [FieldProblem(field="benchSchemaVersion", reason="null")]
        # `isinstance(schema, bool)` first, because `True == 1`: without it `True` is *in*
        # `REQUIRED_BY_SCHEMA` and validates as schema 1, so `store()` writes a record that
        # `load_history` — which does carry the guard — immediately quarantines, with a bool
        # landing in an `InvalidRecord.benchSchemaVersion` typed `int | None` (review P2-2). The
        # two enforcement points have to agree, and this is the one that can refuse the write.
        if isinstance(schema, bool) or schema not in REQUIRED_BY_SCHEMA:
            return [FieldProblem(field="benchSchemaVersion", reason="unknown")]

        problems: list[FieldProblem] = []
        for name, spec in REQUIRED_BY_SCHEMA[schema][self.armKind].items():
            if name not in self.fields:
                problems.append(FieldProblem(field=name, reason="absent"))
                continue
            value = self.fields[name]
            if value is None:
                problems.append(FieldProblem(field=name, reason="null"))
            elif spec.tier == "nonempty" and not value:
                problems.append(FieldProblem(field=name, reason="empty"))
        for name in sorted(FORBIDDEN_BY_ARM_KIND[self.armKind]):
            if name in self.fields:
                problems.append(FieldProblem(field=name, reason="forbidden"))
        return problems

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Fingerprint":
        fields = {k: v for k, v in d.items() if k != "armKind"}
        return cls(armKind=d.get("armKind", ""), fields=fields)

    def to_dict(self) -> dict[str, Any]:
        return {"armKind": self.armKind, **dict(self.fields)}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fingerprint):
            return NotImplemented
        return self.armKind == other.armKind and dict(self.fields) == dict(other.fields)

    def __hash__(self) -> int:
        """Hash the field *values*, not just their names (review n-2).

        Hashing `tuple(sorted(self.fields))` — the keys alone — satisfied the contract (equal
        objects hash equal) but collided every fingerprint with the same key set, which is every
        fingerprint of the same arm kind. `json` with `sort_keys=True` is what makes it safe: the
        values include lists and nested dicts, so `repr` would order two equal dicts differently by
        insertion order and break `equal -> equal hash`, and `hash(tuple(sorted(items)))` would
        raise on the unhashable ones.
        """
        canonical = json.dumps(dict(self.fields), sort_keys=True, default=repr)
        return hash((self.armKind, canonical))
