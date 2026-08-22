"""Request models for the REST surface (FastAPI validation boundary).

Responses are plain dicts straight from the service layer — one shape, both
front doors. Only request bodies need declared models.

**One exception, K-031:** the three workflow *structure/diff* read routes declare
`response_model=` (`WorkflowDefStructureOut`, `WorkflowDiffOut`). They are new
surface whose whole point is a stable, hand-diffable contract, so they get a
declared schema; the pre-existing routes are deliberately **not** retrofitted
(FastAPI's `response_model` *filters* undeclared fields, so a wrong model would
silently drop a field the web client reads). That leaves the repo with a mixed
convention — recorded on the standing "per-endpoint response schemas" backlog
entry, not claimed as closed.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Size bounds: RAM is the binding constraint (AGENTS.md rule 6) — message text
# lands in graph memory AND the full-text index, so the boundary caps it.
MAX_NAME_LEN = 200
MAX_TEXT_LEN = 8000
MAX_MENTIONS = 50
# Server-minted ids (runId, msgId, …) are hex uuids; bound path params defensively so a
# pathological id can't reach a query (rule 6 — the boundary caps every string input).
MAX_ID_LEN = 200


class CreateChannelIn(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_NAME_LEN)


class CreateThreadIn(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_NAME_LEN)


class PostMessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LEN)
    # REST mention parity with the MCP tool
    mentions: list[str] | None = Field(None, max_length=MAX_MENTIONS)


# ── §11 Workflow definition publish (M3 Slice 1) ────────────────────────────────
# Size bounds are the RAM guard (rule 6): a def is a handful of steps/transitions
# in the global `reference` graph; `config`/`guard` are opaque strings the caller
# pre-serializes (the service stores them verbatim, never queries inside).
MAX_KEY_LEN = 200
MAX_STEPS = 200
MAX_TRANSITIONS = 500
MAX_CONFIG_LEN = 8000

# `SweepDueWorkflowRunsIn.limit`'s bound (K-028) — declared here, not in
# `services.py`, and imported from there (`services.py`'s own `from .schemas
# import MAX_CONFIG_LEN, MAX_DIFF_PREVIEW` is the precedent this mirrors, not
# reverses: schemas.py is the leaf/boundary module with no imports of its own
# from `services.py`, so the shared constant has to live on this side for
# either module to import the other without a cycle — verified: the reverse
# direction (defining these in `services.py` and importing them here)
# deadlocks at import time in every module-load order, since `services.py`
# itself imports `MAX_CONFIG_LEN`/`MAX_DIFF_PREVIEW` from this module).
DEFAULT_SWEEP_LIMIT = 200
MAX_SWEEP_LIMIT = 1000


class WorkflowStepIn(BaseModel):
    key: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    type: str = Field(min_length=1, max_length=50)
    config: str | None = Field(None, max_length=MAX_CONFIG_LEN)
    # Exactly one step marks itself the start; the service validates + derives it.
    start: bool = False


class WorkflowTransitionIn(BaseModel):
    # `from` is a Python keyword — accept it on the wire via alias.
    from_: str = Field(alias="from", min_length=1, max_length=MAX_KEY_LEN)
    to: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    on: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    guard: str | None = Field(None, max_length=MAX_CONFIG_LEN)
    order: int = 0


class PublishWorkflowDefIn(BaseModel):
    key: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    version: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    name: str = Field(min_length=1, max_length=MAX_NAME_LEN)
    kind: str = Field(min_length=1, max_length=50)
    steps: list[WorkflowStepIn] = Field(min_length=1, max_length=MAX_STEPS)
    transitions: list[WorkflowTransitionIn] = Field(
        default_factory=list, max_length=MAX_TRANSITIONS
    )


# ── §11 Workflow def/snapshot structure + diff responses (K-031) ────────────────
# Read-only observability surface. `config`/`guard` are returned **verbatim** as
# opaque strings (rule 8) — never parsed, never re-serialized: byte fidelity is
# the point, since a diff that round-trips JSON would hide a whitespace-only
# divergence.
#
# The diff's values are **previews, not payloads**: truncated so the response is
# O(differences), never O(def). An operator who needs the full value reads the
# two structure endpoints.
MAX_DIFF_PREVIEW = 200


class WorkflowStepOut(BaseModel):
    key: str
    type: str
    config: str


class WorkflowTransitionOut(BaseModel):
    # `from` is a Python keyword — emit it on the wire via alias (FastAPI
    # serializes `by_alias=True` by default).
    from_: str = Field(alias="from")
    to: str
    on: str
    order: int
    guard: str


class WorkflowDefStructureOut(BaseModel):
    """`GET /workflow-defs/{key}/versions/{version}` and its snapshot mirror.

    `startKeys` is present **only** when a root carries more than one `START`
    edge. The routes declare `response_model_exclude_unset=True` so an *absent*
    key stays absent, rather than the `"startKeys": null` a nullable field would
    otherwise serialize to. `exclude_none` would have been the obvious mechanism
    and is deliberately **not** used: `startKey` is itself nullable, so
    `exclude_none` would silently drop it for a root with no `START` edge — an
    observability endpoint hiding exactly the anomaly it exists to show.

    **`exclude_unset` propagates into nested models** (verified against the
    installed pydantic): a `WorkflowStepOut`/`WorkflowTransitionOut` built from a
    dict missing a *defaulted* field serializes without it. Both nested models
    therefore keep **every field required** — deliberately, not incidentally. Give
    either one a default and it silently becomes omittable on the wire.
    """

    source: str
    key: str
    version: str
    name: str
    kind: str
    startKey: str | None = None
    startKeys: list[str] | None = None
    stepCount: int
    transitionCount: int
    steps: list[WorkflowStepOut]
    transitions: list[WorkflowTransitionOut]


class WorkflowDiffEntry(BaseModel):
    """One enumerated difference. `def` = `reference`, `snapshot` = `ws:{id}`."""

    path: str
    # `def` is a Python keyword — same alias trick as `from` above.
    def_: str | None = Field(alias="def")
    snapshot: str | None


class WorkflowDiffOut(BaseModel):
    """`GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff`.

    `inSync` is the one-glance answer; `differences` is the evidence. One side
    missing is a **200** with the presence flags carrying the story (that is the
    documented post-`pytest`/post-`test_queries.sh` trap, not an error); both
    sides missing is a 404.
    """

    key: str
    version: str
    defPresent: bool
    snapshotPresent: bool
    inSync: bool
    differences: list[WorkflowDiffEntry]
    differenceCount: int


# ── §12 Workflow-run start + human/signal input (K-024 U3, D-B/D-F/D-H) ─────────
# **Which layer owns which bound (plan m-5).** Pydantic bounds only what it can
# see: the *submitted* dict (key count, key length, serialized size) — it never
# sees the stored run `ctx` this merges into, nor does MCP / a direct service
# caller ever reach it. So the **merged** ctx bound, the reserved-key rule (M-2)
# and the parked-step declaration check (D-H) all live in `services.py`; this
# model is the convenience bound at the HTTP door, not the contract.
MAX_RUN_CTX_KEYS = 32
# The run-level step budget a caller may declare (D-H part c: `access-request@v1`
# passes 24; omitting it falls back to the executor's global default of 12).
MAX_RUN_STEPS = 50


def _bounded_flat_dict(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Shared bound for a caller-supplied ctx/input dict (rule 6)."""
    if value is None:
        return None
    if len(value) > MAX_RUN_CTX_KEYS:
        raise ValueError(f"at most {MAX_RUN_CTX_KEYS} keys allowed")
    for key in value:
        if not key or len(key) > MAX_KEY_LEN:
            raise ValueError(f"key must be 1..{MAX_KEY_LEN} characters")
    if len(json.dumps(value, separators=(",", ":"), sort_keys=True)) > MAX_CONFIG_LEN:
        raise ValueError(f"serialized payload exceeds {MAX_CONFIG_LEN} characters")
    return value


class StartWorkflowRunIn(BaseModel):
    """`POST /workflow-runs` — start a run from a snapshot with no chat trigger."""

    defKey: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    version: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    # The run's initial state. Reserved keys (`threadId`, `error`) are rejected by
    # the SERVICE (M-2/F-6) — not here — because MCP and direct callers bypass this.
    ctx: dict[str, Any] | None = None
    trace: bool = False
    # `maxSteps` is a tripwire checked *after* each recorded step, not a hard cap:
    # a run executes at most `maxSteps + 1` steps before failing with
    # "step budget exceeded" (DESIGN §6, QUERIES.md §12.5 note).
    maxSteps: int | None = Field(None, ge=1, le=MAX_RUN_STEPS)

    @field_validator("ctx")
    @classmethod
    def _check_ctx(cls, v):
        return _bounded_flat_dict(v)


class SubmitWorkflowInputIn(BaseModel):
    """`POST /workflow-runs/{runId}/input` — human/signal input for a parked run."""

    input: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def _check_input(cls, v):
        return _bounded_flat_dict(v)


class SweepDueWorkflowRunsIn(BaseModel):
    """`POST /workflow-runs/due` — sweep parked wait-timer runs past their due
    time (K-028, `docs/plans/workflow-timers.md` §3.6)."""

    limit: int = Field(DEFAULT_SWEEP_LIMIT, ge=1, le=MAX_SWEEP_LIMIT)
