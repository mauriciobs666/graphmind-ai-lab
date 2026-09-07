"""The environment fingerprint, and the rules that make FR-7 mechanical rather than aspirational.

Design: `docs/plans/small-model-benchmarking.md` §3.4 (and §3.4.1–§3.4.3). Three ideas carry the
whole module:

* **Two discriminators combine into one profile key** (§3.4.1). A BM25 reference arm has no model,
  no quantization and no runtime, so `validate()` branches on the arm and never on field presence.
  The `deterministic` kind *forbids* every model field: `{"modelKey": "bm25", "quantization":
  "n/a"}` is the shortcut a time-pressed implementer reaches for, and it must fail loudly on write
  rather than quietly become a sixth model in the history. `armKind` keeps its two values; a second
  discriminator `callSurface` (`"chat"` / `"embeddings"`, `None` iff deterministic) joins it, and
  the mapping key is the **derived** `armProfile` — `model:chat`, `model:embeddings` or
  `deterministic`. An embeddings call has no `runtime` object to observe and no sampling parameters
  to obey, so that profile is *forbidden* `runtimeName`, `runtimeVersion`, `temperature` and
  `maxTokens` rather than merely excused from them. Both discriminators are members of no required
  set and are checked before any mapping is consulted.
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
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

ArmKind = Literal["model", "deterministic"]
CallSurface = Literal["chat", "embeddings"]
Tier = Literal["nonempty", "present"]

#: Why a field failed. `unknown` covers a **value** this build cannot interpret, in three
#: families (plan Appendix A): a discriminator it does not recognize (`armKind`, `callSurface`), a
#: `benchSchemaVersion` from the future, and a residency snapshot that is not a list, an element
#: that is not a mapping, or an element value that is not a string.
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
_MODEL_CHAT_SCHEMA_1: dict[str, FieldSpec] = {
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
    # which surface answered the residency and catalog probe (§3.4.2, §3.4.4a). It replaces the
    # retired `lms`-build field one-for-one: nothing in the harness runs that CLI any more, so
    # that field had no source and could only have been kept by defaulting it to `""` — the
    # silently-defaulted fingerprint field FR-7 exists to refuse.
    "residencySource": _NONEMPTY,
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

#: An embeddings call returns no `runtime` object and obeys no sampling parameters (§3.4.4a), so
#: `model:embeddings` is `model:chat`'s 30 fields minus these four — 26 in total (§3.4.2). Derived
#: rather than transcribed: a field added to the chat set reaches this one without a second edit,
#: and §3.4.1's union-minus-mine derivation then *forbids* these four on an embeddings record.
_EMBEDDINGS_HAVE_NO: frozenset[str] = frozenset(
    {"runtimeName", "runtimeVersion", "temperature", "maxTokens"}
)

_MODEL_EMBEDDINGS_SCHEMA_1: dict[str, FieldSpec] = {
    name: spec for name, spec in _MODEL_CHAT_SCHEMA_1.items() if name not in _EMBEDDINGS_HAVE_NO
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

#: `{schemaVersion: {armProfile: {field: FieldSpec}}}` — plan §3.4.3. A record is validated against
#: its own entry here, so an added field at a later version never invalidates an older record.
#: The inner key is the *profile* (§3.4.1), not the arm kind: schema 1 has exactly three.
#: Mutable by design: `model-bench migrate` and the schema-2 regression test both key off it.
REQUIRED_BY_SCHEMA: dict[int, dict[str, dict[str, FieldSpec]]] = {
    1: {
        "model:chat": _MODEL_CHAT_SCHEMA_1,
        "model:embeddings": _MODEL_EMBEDDINGS_SCHEMA_1,
        "deterministic": _DETERMINISTIC_SCHEMA_1,
    },
}


def _forbidden_by_profile(schema: int) -> dict[str, frozenset[str]]:
    """§3.4.1's forbidden set, as **a set operation and never a list**.

    `FORBIDDEN[p] = (⋃ required(q) for every other profile q) − required(p)` — a record may not
    carry a field that only *some other* profile is required to have. Written as the operation so
    that adding a field to any profile forbids it on the others for free: through plan v1.4 this
    was a hand-typed list of fourteen model fields that had already drifted from the field set it
    claimed to complement, and a set difference cannot drift from its own intent. It is also what
    earns `model:embeddings` its four forbidden runtime/sampling fields — nobody wrote those names
    down, and forbidding them is exactly right, because a record carrying either is claiming
    something an embeddings call cannot have measured.
    """
    profiles = REQUIRED_BY_SCHEMA[schema]
    return {
        p: frozenset().union(*(frozenset(profiles[q]) for q in profiles if q != p))
        - frozenset(profiles[p])
        for p in profiles
    }


#: What each arm *profile* may not carry (§3.4.1). The forbid half is the point of the
#: discriminators. Keyed by profile, over schema 1 — the version every live record declares.
FORBIDDEN_BY_ARM_PROFILE: Mapping[str, frozenset[str]] = MappingProxyType(_forbidden_by_profile(1))

#: The two `armKind` values, **decoupled from the forbidden mapping** (§4 S1e Table B). Deriving
#: this from `FORBIDDEN_BY_ARM_PROFILE` — which is what shipped before the re-key — would make its
#: members the three *profiles*, so `armKind == "model"` would fail the membership test below and
#: every model record would return `FieldProblem("armKind", "unknown")` and refuse on write.
#: `armKind` keeps its two values and its meaning; the profile is the mapping key alone.
ARM_KINDS: frozenset[str] = frozenset(p.split(":", 1)[0] for p in REQUIRED_BY_SCHEMA[1])

#: The `callSurface` values, from the same split — the profile suffix, `chat` or `embeddings`.
CALL_SURFACES: frozenset[str] = frozenset(
    p.split(":", 1)[1] for p in REQUIRED_BY_SCHEMA[1] if ":" in p
)

#: A residency snapshot element is `{id, state}` with the **literal `state` string** kept
#: (§3.4.4a) — not normalised to a boolean, because the only state the 2026-09-03 probe observed
#: is the one value and any other is a value this plan has not seen.
RESIDENCY_ELEMENT_KEYS: frozenset[str] = frozenset({"id", "state"})

#: The two fields holding such snapshots. Their tier is `present` — `[]` is the correct, informative
#: answer on a clean box — and `REQUIRED_PRESENT` checks presence and **never element shape**, which
#: is why the shape is checked here as well: without it the retired `lms ps --json` element —
#: whose only source plan v1.8 removed, and whose two keys are *neither* of these — validates,
#: ships green and travels into S2, where the probe emits `{id, state}` and the two disagree with
#: nothing to catch them (plan §3.4.2, S1 done-condition 1). The check is over the element's whole
#: key set, so **any** key that is not `id` or `state` is refused, which is what makes an element
#: half-swapped out of the retired shape fail as loudly as the retired shape itself.
_RESIDENCY_FIELDS: frozenset[str] = frozenset({"residentModelsAtStart", "residentModelsAtEnd"})

#: The discriminators, which are members of no required set and therefore never live in `fields`.
_DISCRIMINATORS: tuple[str, ...] = ("armKind", "callSurface")


def _residency_problems(name: str, value: Any) -> list[FieldProblem]:
    """Every problem in one residency snapshot, each naming the element and key it came from."""
    if not isinstance(value, (list, tuple)):
        return [FieldProblem(field=name, reason="unknown")]
    problems: list[FieldProblem] = []
    for index, element in enumerate(value):
        where = f"{name}[{index}]"
        if not isinstance(element, Mapping):
            problems.append(FieldProblem(field=where, reason="unknown"))
            continue
        for key in sorted(RESIDENCY_ELEMENT_KEYS):
            if key not in element:
                problems.append(FieldProblem(field=f"{where}.{key}", reason="absent"))
            elif element[key] is None:
                problems.append(FieldProblem(field=f"{where}.{key}", reason="null"))
            elif not isinstance(element[key], str):
                problems.append(FieldProblem(field=f"{where}.{key}", reason="unknown"))
            elif not element[key]:
                problems.append(FieldProblem(field=f"{where}.{key}", reason="empty"))
        for key in sorted(set(element) - RESIDENCY_ELEMENT_KEYS):
            problems.append(FieldProblem(field=f"{where}.{key}", reason="forbidden"))
    return problems


@dataclass(frozen=True)
class Fingerprint:
    """One run's environment record. Two discriminators; `fields` holds everything else.

    `callSurface` is **required with no default**: it is what selects the contract the fields are
    read against, and a default would pick one silently for a caller that never decided (§3.4.1).
    It is `None` if and only if `armKind == "deterministic"` — that arm calls no surface at all.
    """

    armKind: str
    callSurface: str | None
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        # `frozen=True` on a dataclass holding a live `dict` the caller still references is frozen
        # in name only: the record could change under a report that had already validated it. A
        # copy behind a `MappingProxyType` makes the freeze real (review n-2).
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    @property
    def armProfile(self) -> str:
        """The mapping key (§3.4.1) — derived from both discriminators, never stored twice."""
        if self.armKind == "deterministic":
            return self.armKind
        return f"{self.armKind}:{self.callSurface}"

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

        # The second discriminator, checked here so that it is decided **before any mapping is
        # consulted** (§3.4.1, S1 done-condition 6): without a `callSurface` there is no profile,
        # so there is no required set to report a record against, and answering with thirty
        # `absent` problems would bury the one that is true.
        #
        # It reports `null` and `absent` as two different failures, for the reason `armKind` does
        # (review P3-10): a stored `"callSurface": null` was written by something that *had* the
        # value and lost it, where a missing key was never written at all, and AC-2's quarantine
        # line prints the reason. `empty` collapses into `absent` here — and only here — because
        # `""` is the missing-key sentinel `from_dict` uses, so on this field the two states are
        # indistinguishable by construction; `null` is not, which is why it stays apart.
        if self.armKind == "deterministic":
            if self.callSurface is not None:
                return [FieldProblem(field="callSurface", reason="forbidden")]
        elif self.callSurface is None:
            return [FieldProblem(field="callSurface", reason="null")]
        elif not self.callSurface:
            return [FieldProblem(field="callSurface", reason="absent")]
        elif self.callSurface not in CALL_SURFACES:
            return [FieldProblem(field="callSurface", reason="unknown")]

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
        for name, spec in REQUIRED_BY_SCHEMA[schema][self.armProfile].items():
            if name not in self.fields:
                problems.append(FieldProblem(field=name, reason="absent"))
                continue
            value = self.fields[name]
            if value is None:
                problems.append(FieldProblem(field=name, reason="null"))
            elif spec.tier == "nonempty" and not value:
                problems.append(FieldProblem(field=name, reason="empty"))
            elif name in _RESIDENCY_FIELDS:
                problems.extend(_residency_problems(name, value))
        for name in sorted(FORBIDDEN_BY_ARM_PROFILE[self.armProfile]):
            if name in self.fields:
                problems.append(FieldProblem(field=name, reason="forbidden"))
        return problems

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Fingerprint":
        # Both discriminators are stripped, for the same reason: a discriminator left in `fields`
        # is a field of no required set and so lands in every profile's *forbidden* set.
        fields = {k: v for k, v in d.items() if k not in _DISCRIMINATORS}
        arm_kind = d.get("armKind", "")
        # A missing key and a stored `null` are two different failures, so the missing-key
        # sentinel is `""` — the same one `armKind` uses one line up — and never `None`, which
        # would collapse the two into one reason (review P5-1). On a *deterministic* record the
        # missing key is not an absence at all: `None` **is** that arm's value (§3.4.1, "`None`
        # iff deterministic"), and it is what `to_dict` omits, so the round trip restores it.
        missing: str | None = None if arm_kind == "deterministic" else ""
        return cls(
            armKind=arm_kind, callSurface=d.get("callSurface", missing), fields=fields
        )

    def to_dict(self) -> dict[str, Any]:
        # `callSurface` is omitted rather than written `null` on a deterministic arm: that arm
        # calls no surface, which is a different fact from "we did not capture this" — the one
        # thing `null` means in this record (§3.4.2).
        surface = {} if self.callSurface is None else {"callSurface": self.callSurface}
        return {"armKind": self.armKind, **surface, **dict(self.fields)}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fingerprint):
            return NotImplemented
        return (
            self.armKind == other.armKind
            and self.callSurface == other.callSurface
            and dict(self.fields) == dict(other.fields)
        )

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
        return hash((self.armKind, self.callSurface, canonical))
