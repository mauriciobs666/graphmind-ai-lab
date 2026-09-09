"""`host.json` — the operator-attested fingerprint half — and the attestation staleness trip-wire.

Design: `docs/plans/small-model-benchmarking.md` §3.4.4 (the schema), §3.4.4a (capture order and
the source-of-truth table), §3.4.5 point 3 (the trip-wire's outcomes) and §3.6a (the `attest`
CLI command's contract). `model-bench/AGENTS.md` names the four operator-attested fields and why
they have no programmatic source on this LM Studio build.

This module owns three things, and nothing past them:

* **The schema, and its writer/reader.** `validate_host_info` returns `list[str]` — `[]` means
  valid — the same shape `Fingerprint.validate()` and `packs.validate_pack` already use.
  `read_host_info` is what `run` (a later, not-yet-built unit) calls at capture-order step 1:
  absent or schema-invalid both raise `HostInfoError`, which is that unit's cue to exit `5` before
  a single model call.
* **`attest(...)`** — the `attest` command's actual work: probe LM Studio, then write `host.json`.
  `cli.py`'s `attest` subcommand is thin argument-parsing wired onto this function.
* **`check_attestation_staleness(...)`** — the trip-wire's pure decision function (plan §3.4.5
  point 3 / §3.4.4a step 6). It is **not wired into a `run` command here** — `run`'s capture-order
  sequence (the warm-up call that produces `runtimeName`/`runtimeVersion` to feed it) is a later
  unit's, out of this unit's fences. This module ships the mechanism and its four outcomes, tested
  directly against hand-built `host.json` states, so that unit has something correct to call
  rather than something to invent.

`attest` is sequenced ahead of the runner (plan §4 S2) because S3's done-condition — "a stored
result with a **complete** fingerprint" — is unreachable until `host.json` exists; that is the
constraint this module's shape serves.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from modelbench import fingerprint
from modelbench.lmstudio import LMStudio

HOST_INFO_FILENAME = "host.json"

#: `host.json`'s only schema version so far (plan §3.4.4). Unlike `fingerprint.py`'s
#: `REQUIRED_BY_SCHEMA`, this file has no migration story of its own — plan §3.4.3's "a record is
#: validated against the contract it was written under" is a *run-record* rule; `host.json` is a
#: single mutable local file `attest` overwrites each time, not an append-only history.
SCHEMA_VERSION = 1

#: The four FR-7 fields with no programmatic source on this LM Studio build
#: (`model-bench/AGENTS.md`, plan §3.4). Exactly the plan's own key spelling — `host.json` is a
#: stored-record shape `run` reads back, so these names are taken from the plan's literal JSON,
#: not invented.
ATTESTED_FIELD_NAMES: tuple[str, ...] = (
    "lmStudioAppVersion",
    "kvCacheSetting",
    "hostRamGb",
    "otherResidentWorkloads",
)

#: Which of the four `attested` fields `validate_host_info` must refuse empty — **derived from
#: `fingerprint.py`'s own `_NONEMPTY` tiering, not hand-typed** (review Pass 13, P13-4). Three of
#: the four (`lmStudioAppVersion`, `kvCacheSetting`, `hostRamGb`) are `_NONEMPTY` in
#: `fingerprint.REQUIRED_BY_SCHEMA[1]["model:chat"]`; `otherResidentWorkloads` is `_PRESENT`, so
#: `[]` stays valid. Hand-writing "these three are non-empty" a second time here is exactly the
#: shape that drifted: the fingerprint tiers a field, and this file agrees only because its author
#: remembered to keep two lists in step. Reading the tier directly means a future field added to
#: `ATTESTED_FIELD_NAMES` — or a tier change on the fingerprint side — is inherited rather than
#: silently missed; `test_attested_field_tiers_agree_between_chat_and_embeddings_profiles`
#: (`tests/test_hostinfo.py`) is the checkable claim that `"model:chat"` is a safe single profile
#: to read this from, and `test_validate_host_info_agrees_with_the_fingerprints_own_attested_
#: field_tiering` drives `validate_host_info`'s actual behavior against these tiers directly,
#: rather than comparing two sets that could agree by coincidence.
ATTESTED_NONEMPTY_FIELD_NAMES: frozenset[str] = frozenset(
    name
    for name in ATTESTED_FIELD_NAMES
    if fingerprint.REQUIRED_BY_SCHEMA[1]["model:chat"][name].tier == "nonempty"
)

#: The trip-wire's three *stored* outcomes (plan §3.4.5 point 3; `RunResult.attestationTripWire`,
#: a later unit's field). The fourth behavioural outcome — a `"compared"` run whose comparands
#: disagree — is not a fourth string: it is `outcome == "compared"` with `stale=True`, because a
#: mismatch exits before a `RunResult` is ever built (plan §3.4.4a step 6) and so is never stored
#: at all.
AttestationOutcome = Literal["first-observation", "compared", "unavailable"]

#: Plan §3.4.5 point 3's message, verbatim.
STALE_MESSAGE = (
    "LM Studio changed since you last attested host.json — re-check the app version and "
    "KV cache setting, then `model-bench attest`."
)


class HostInfoError(RuntimeError):
    """`host.json` is absent or fails schema validation (plan §3.4.4a capture-order step 1) —
    `run`'s cue to exit `5` before any model call. Also raised by `attest` itself if it would
    otherwise write a file that fails its own schema (a defensive check; it should never fire)."""


class AttestProbeFailed(RuntimeError):
    """`attest`'s own probe of LM Studio did not return `"api-v0"`. The message is one of
    §3.4.4a's two distinguishing strings, carried verbatim — `attest` writes no `host.json` in
    either case, the same refusal `run`'s own capture-order step 2 makes for the same probe."""


def host_info_path(root: Path) -> Path:
    """`<root>/host.json` — local, gitignored (plan §3.4.4, `.gitignore`)."""
    return root / HOST_INFO_FILENAME


def validate_host_info(d: Any) -> list[str]:
    """Every schema problem in `d`; `[]` means valid (the shape `Fingerprint.validate()` and
    `packs.validate_pack` already share). Checks presence, type, emptiness where the fingerprint's
    own tiering requires it (`ATTESTED_NONEMPTY_FIELD_NAMES`, review Pass 13 P13-4), that
    `attested` carries no key beyond the four §3.4.4 names (P13-12), and the one non-empty field
    the trip-wire depends on (`observedAtAttestation.residencySource`) — nothing this build cannot
    yet know, such as a specific `apiBaseUrl` format the plan never constrains.
    """
    if not isinstance(d, Mapping):
        return ["host.json is not a JSON object"]
    problems: list[str] = []

    if "schemaVersion" not in d:
        problems.append("schemaVersion: absent")
    elif d["schemaVersion"] != SCHEMA_VERSION:
        problems.append(f"schemaVersion: unknown ({d['schemaVersion']!r})")

    api_base_url = d.get("apiBaseUrl")
    if not isinstance(api_base_url, str) or not api_base_url:
        problems.append("apiBaseUrl: absent or empty")

    attested = d.get("attested")
    if not isinstance(attested, Mapping):
        problems.append("attested: absent or not an object")
    else:
        for name in ATTESTED_FIELD_NAMES:
            if name not in attested:
                problems.append(f"attested.{name}: absent")
                continue
            value = attested[name]
            if value is None:
                problems.append(f"attested.{name}: null")
                continue
            if name == "hostRamGb" and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                problems.append("attested.hostRamGb: not a number")
                continue
            if name == "otherResidentWorkloads" and not isinstance(value, list):
                problems.append("attested.otherResidentWorkloads: not a list")
                continue
            # P13-4: a value can be present, non-null and correctly typed and still be the empty
            # value the fingerprint's own `_NONEMPTY` tier refuses (`""`, `0`) — checked last, and
            # against the derived set, so a field the fingerprint does not tier non-empty
            # (`otherResidentWorkloads`, `_PRESENT`) is correctly left alone.
            if name in ATTESTED_NONEMPTY_FIELD_NAMES and not value:
                problems.append(f"attested.{name}: empty")
        unexpected = sorted(set(attested) - set(ATTESTED_FIELD_NAMES))
        if unexpected:
            problems.append(
                f"attested: unexpected key(s) {unexpected} — only "
                f"{', '.join(ATTESTED_FIELD_NAMES)} are allowed"
            )

    attested_at = d.get("attestedAt")
    if not isinstance(attested_at, str) or not attested_at:
        problems.append("attestedAt: absent or empty")

    observed = d.get("observedAtAttestation")
    if not isinstance(observed, Mapping):
        problems.append("observedAtAttestation: absent or not an object")
    elif not observed.get("residencySource"):
        problems.append("observedAtAttestation.residencySource: absent or empty")

    return problems


def write_host_info(root: Path, data: Mapping[str, Any]) -> Path:
    """Write `data` to `<root>/host.json`, creating `root` if needed. Never validates — callers
    that must not write an invalid file (`attest`) check first."""
    path = host_info_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2) + "\n", encoding="utf-8")
    return path


def read_host_info(root: Path) -> dict[str, Any]:
    """Plan §3.4.4a step 1: absent or schema-invalid both raise `HostInfoError`, so `run`'s
    refusal is one `except` clause rather than two different checks."""
    path = host_info_path(root)
    if not path.is_file():
        raise HostInfoError(f"no host.json at {path} — run `model-bench attest` first")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HostInfoError(f"{path} is not valid JSON: {exc}") from exc
    problems = validate_host_info(data)
    if problems:
        raise HostInfoError(f"{path} fails schema validation: {'; '.join(problems)}")
    return data


def _utc_stamp(now: Callable[[], datetime] | None) -> str:
    moment = (now or (lambda: datetime.now(timezone.utc)))()
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _residency_source_after_a_successful_probe() -> str:
    """The one value `attest` can ever write here: by the time this is called, `attest()` has
    already returned on any probe outcome but `"api-v0"` (review Pass 13, P13-10 — the prior
    signature took a `probe_result` parameter it never read, promising a decision this function
    did not make). `"lmstudio-api-v0"` is this repo's own established literal for the
    `GET /api/v0/models` surface (`tests/conftest.py`'s `MODEL_FIELDS`), reused here rather than
    reinvented."""
    return "lmstudio-api-v0"


def attest(
    root: Path,
    *,
    api_base_url: str,
    attested: Mapping[str, Any],
    client: LMStudio,
    now: Callable[[], datetime] | None = None,
) -> Path:
    """The `attest` command's work (plan §3.6a): probe LM Studio, then write `host.json` matching
    plan §3.4.4's schema exactly, with `observedAtAttestation` carrying **`residencySource` only**
    — neither probed endpoint exposes a `runtime` object and `attest` takes no model argument to
    call one against (plan §3.4.4, plan-gate P4-6).

    `client` is an already-constructed `LMStudio` — this function never decides how to reach the
    network, matching `tests/test_lmstudio.py`'s injected-`opener` seam one layer up. Raises
    `AttestProbeFailed` (host.json is **not** written) when the probe is not `"api-v0"`, with
    §3.4.4a's two distinguishing messages.
    """
    outcome = client.probe()
    if outcome == "unreachable":
        raise AttestProbeFailed(f"LM Studio is not reachable at {api_base_url}")
    if outcome == "v1-only":
        raise AttestProbeFailed(
            f"{api_base_url} serves an OpenAI-compatible API but not LM Studio's native "
            "/api/v0 catalog. model-bench fingerprints from that catalog and will not record "
            "a run it cannot fingerprint (§3.4.4a)."
        )

    host = {
        "schemaVersion": SCHEMA_VERSION,
        "apiBaseUrl": api_base_url,
        "attested": dict(attested),
        "attestedAt": _utc_stamp(now),
        "observedAtAttestation": {"residencySource": _residency_source_after_a_successful_probe()},
    }
    problems = validate_host_info(host)
    if problems:
        # Defensive only — a caller passing a malformed `attested` mapping (e.g. missing one of
        # the four names) should not silently produce a `host.json` that `run` will refuse later
        # with no memory of why; refuse here instead, where the cause is still in scope.
        raise HostInfoError(f"refusing to write an invalid host.json: {'; '.join(problems)}")
    return write_host_info(root, host)


@dataclass(frozen=True)
class AttestationCheck:
    """The trip-wire's verdict (plan §3.4.5 point 3). `updated_host` is the `host.json` dict to
    persist back — only on `"first-observation"`, where step 6 back-fills the runtime keys and
    touches nothing else; `None` on every other outcome, since `"compared"` and `"unavailable"`
    both leave `host.json` exactly as it was read."""

    outcome: AttestationOutcome
    stale: bool
    message: str | None
    updated_host: Mapping[str, Any] | None


def check_attestation_staleness(
    host: Mapping[str, Any],
    *,
    call_surface: Literal["chat", "embeddings"],
    residency_source: str,
    runtime_name: str,
    runtime_version: str,
    now: Callable[[], datetime] | None = None,
) -> AttestationCheck:
    """Plan §3.4.4a capture-order step 6 / §3.4.5 point 3's trip-wire, as a pure function over an
    already-read `host.json` and what a chat-surface warm-up just observed.

    **Not called at all for a `deterministic` arm** — plan §3.4.5: "a deterministic arm skips this
    check entirely." The caller (`run`, a later unit) is expected to enforce that by never invoking
    this function for that arm kind; there is no third `call_surface` value to express it here.

    The three outcomes, plus the mismatch case that is not a fourth outcome string (see
    `AttestationOutcome`'s docstring):

    - **`"unavailable"`** — `call_surface == "embeddings"`. An embeddings warm-up returns no
      `runtime` object at all (§3.4.4a), so the comparison "degenerates to a near-constant" and is
      **not attempted** — unconditionally `"unavailable"`, whatever `host` holds.
    - **`"first-observation"`** — a `model:chat` arm whose `observedAtAttestation` has never
      recorded a `runtimeName` (checked by key, since `runtimeName`/`runtimeVersion`/
      `runtimeObservedAt` are always written together, §3.4.4). Back-fills those three keys into a
      copy of `host` and returns it as `updated_host`; `stale` is always `False` here — a baseline
      is not evidence of anything having changed. `attested` and `attestedAt` are untouched, byte
      for byte — the plan is explicit that the back-fill "attests nothing new".
    - **`"compared"`** — every later `model:chat` run. `stale` compares `runtimeName`,
      `runtimeVersion` **and** `residencySource` against the last-observed values (§3.4.4a: "step
      6 compares runtimeName/runtimeVersion/residencySource ... from the second model:chat run
      onward") — a difference in any one of the three is a mismatch, and `message` carries
      `STALE_MESSAGE` iff `stale`.

    **Precondition, enforced before either branch (review Pass 13, P13-8):** `host` must already
    carry a non-empty `observedAtAttestation.residencySource` — exactly what `read_host_info`
    guarantees and `write_host_info` does not. Without this, a `host` missing
    `observedAtAttestation` entirely back-filled straight into an `updated_host` that itself fails
    `validate_host_info` — a file that, once written, makes every later run exit `5` until the
    operator re-attests. Raises `HostInfoError` instead, naming the precondition this function
    depends on rather than corrupting the state it protects.
    """
    observed_at_attestation = host.get("observedAtAttestation")
    if not isinstance(observed_at_attestation, Mapping) or not observed_at_attestation.get(
        "residencySource"
    ):
        raise HostInfoError(
            "check_attestation_staleness: host carries no observedAtAttestation."
            "residencySource — read it via read_host_info() first, which refuses a host.json "
            "missing it"
        )

    if call_surface == "embeddings":
        return AttestationCheck(outcome="unavailable", stale=False, message=None, updated_host=None)

    observed = dict(observed_at_attestation)
    if "runtimeName" not in observed:
        observed["runtimeName"] = runtime_name
        observed["runtimeVersion"] = runtime_version
        observed["runtimeObservedAt"] = _utc_stamp(now)
        updated_host = {**dict(host), "observedAtAttestation": observed}
        return AttestationCheck(
            outcome="first-observation", stale=False, message=None, updated_host=updated_host
        )

    stale = (
        observed.get("runtimeName") != runtime_name
        or observed.get("runtimeVersion") != runtime_version
        or observed.get("residencySource") != residency_source
    )
    return AttestationCheck(
        outcome="compared",
        stale=stale,
        message=STALE_MESSAGE if stale else None,
        updated_host=None,
    )
