"""Shared hand-built fixtures for the suite.

S1 has no pack loader and makes no model calls (`docs/plans/small-model-benchmarking.md` §4 S1),
so every in-memory fixture here is built by hand. S2 adds `pack_fixture()`, resolving the real
on-disk packs under `tests/fixtures/packs/` that `load_pack`/`validate_pack` read (§4 S2) — those
still touch no network and nothing outside `model-bench/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from modelbench.packs import PackMetrics, PackRef
from modelbench.results import (
    BinaryMetric,
    ClassificationAggregates,
    ItemResult,
    ItemTiming,
    RunResult,
    ToolCallAggregates,
)

# A complete, valid `model:chat` fingerprint field set at benchSchemaVersion 1 (plan §3.4.2). Tests
# blank/remove one key at a time from a copy of this, so the baseline must itself be valid.
MODEL_FIELDS: dict[str, Any] = {
    "modelKey": "qwen/qwen3-4b-2507",
    "modelPublisher": "qwen",
    "arch": "qwen3",
    "quantization": "Q4_K_M",
    "compatibilityType": "gguf",
    "maxContextLength": 262144,
    "loadedContextLength": 8192,
    "modelType": "llm",
    "modelCapabilities": ["tool_use"],
    "modelCapabilitiesPresent": True,
    "runtimeName": "llama.cpp",
    "runtimeVersion": "1.52.0",
    "residencySource": "lmstudio-api-v0",
    "residentModelsAtStart": [],
    # `{id, state}` with the literal state string kept — plan §3.4.4a's element shape, and the
    # shape `residency()` will emit in S2. The retired `lms ps --json` element it replaces is now
    # refused by `validate()`, not merely unused.
    "residentModelsAtEnd": [{"id": "qwen/qwen3-4b-2507", "state": "loaded"}],
    "temperature": 0.0,
    "maxTokens": 1024,
    "packId": "tool-caller-shop-assistant",
    "packVersion": "1.0.0",
    "packContentHash": "a" * 64,
    "benchVersion": "0.1.0",
    "benchSchemaVersion": 1,
    "pythonVersion": "3.12.3",
    "hostOs": "Linux-5.15.167.4-microsoft-standard-WSL2",
    "startedAt": "2026-09-03T10:00:00Z",
    "endedAt": "2026-09-03T10:12:00Z",
    "lmStudioAppVersion": "0.3.31",
    "kvCacheSetting": "f16",
    "hostRamGb": 16,
    "otherResidentWorkloads": [],
}

# A complete, valid `deterministic` arm fingerprint (plan §3.4.1): no model fields at all.
DETERMINISTIC_FIELDS: dict[str, Any] = {
    "armId": "bm25",
    "armParametersHash": "b" * 64,
    "packId": "embedder-graphrag-retrieval",
    "packVersion": "1.0.0",
    "packContentHash": "c" * 64,
    "benchVersion": "0.1.0",
    "benchSchemaVersion": 1,
    "pythonVersion": "3.12.3",
    "hostOs": "Linux-5.15.167.4-microsoft-standard-WSL2",
    "startedAt": "2026-09-03T10:20:00Z",
    "endedAt": "2026-09-03T10:20:04Z",
}


def model_fields(**overrides: Any) -> dict[str, Any]:
    """A copy of the valid model field set with `overrides` applied.

    A value of `...` (Ellipsis) *removes* the key, which is how a test expresses "absent" as
    distinct from "empty" and from `null` (plan §3.4.2's three states).
    """
    fields = dict(MODEL_FIELDS)
    for key, value in overrides.items():
        if value is ...:
            fields.pop(key, None)
        else:
            fields[key] = value
    return fields


def embeddings_fields(**overrides: Any) -> dict[str, Any]:
    """A complete, valid `model:embeddings` field set — the chat 26, plan §3.4.2.

    Built by *removing* the four fields that profile forbids rather than by transcribing 26 names:
    the independently written literal that guards against a silently shrinking contract lives in
    `test_fingerprint.py` (review M-4), and a second copy here would be the drift it exists to
    catch, one file over.
    """
    fields = model_fields(**overrides)
    for name in ("runtimeName", "runtimeVersion", "temperature", "maxTokens"):
        if name not in overrides:
            fields.pop(name, None)
    return fields


def deterministic_fields(**overrides: Any) -> dict[str, Any]:
    fields = dict(DETERMINISTIC_FIELDS)
    for key, value in overrides.items():
        if value is ...:
            fields.pop(key, None)
        else:
            fields[key] = value
    return fields


def guard_pack(
    headline: str | None = None, verdicts: tuple[str, ...] = ("falseAdvanceRate",)
) -> PackRef:
    """An item-level pack reference (`sampling.pairingKey == ["itemId"]`, plan §3.3)."""
    return PackRef(
        packId="guard-judge-understanding",
        packVersion="1.0.0",
        contentHash="d" * 64,
        role="guard-judge",
        metrics=PackMetrics(verdictMetrics=verdicts, headlineMetric=headline),
        pairingKey=("itemId",),
        analysisUnit="itemId",
        seed=20260902,
    )


def item(
    item_id: str,
    *,
    correct: bool,
    metric: str = "falseAdvanceRate",
    pairing: tuple[str, ...] | None = None,
    scoreable: bool = True,
) -> ItemResult:
    return ItemResult(
        itemId=item_id,
        pairingKey=pairing if pairing is not None else (item_id,),
        outcome="pass" if correct else "fail",
        scoreable={metric: scoreable},
        counts={metric: 1 if correct else 0},
        timing=ItemTiming(wallClockMs=1300.0, calls=(), withheldFor=None),
        detail={},
    )


def classification_aggregates(
    successes: int, n: int, metric: str = "falseAdvanceRate", unit: str = "item"
):
    return ClassificationAggregates(
        perClass=(BinaryMetric(name=metric, successes=successes, n=n, unit=unit),),
        parseFailures=0,
        n=n,
    )


def run(
    run_id: str,
    *,
    fingerprint_fields: dict[str, Any] | None = None,
    arm_kind: str = "model",
    call_surface: str | None = "chat",
    role: str = "guard-judge",
    items: list[ItemResult] | None = None,
    aggregates: Any = None,
    session_id: str | None = "s1",
    design_effect: float = 1.0,
    basis: str = "by-construction",
    attestation_trip_wire: str | None = "compared",
) -> RunResult:
    from modelbench.fingerprint import Fingerprint

    # `arm_kind` stays — `armKind` keeps its two values (plan §3.4.1) — but the *branch* is
    # profile-aware, because otherwise a `model:embeddings` fixture is not expressible at all.
    if arm_kind == "deterministic":
        call_surface = None
        # `RunResult.attestationTripWire` is None iff armKind == "deterministic" (§4 S1
        # `:3145-3147`) — forced here the same way `call_surface` is, so a caller cannot build an
        # inconsistent fixture by leaving the default in place.
        attestation_trip_wire = None
    profile = arm_kind if call_surface is None else f"{arm_kind}:{call_surface}"

    fields = fingerprint_fields
    if fields is None:
        fields = {
            "model:chat": model_fields,
            "model:embeddings": embeddings_fields,
            "deterministic": deterministic_fields,
        }[profile]()
    items = items if items is not None else []
    if aggregates is None:
        hits = sum(1 for it in items if it.outcome == "pass")
        aggregates = classification_aggregates(hits, len(items))
    return RunResult(
        runId=run_id,
        sessionId=session_id,
        role=role,
        armKind=arm_kind,
        fingerprint=Fingerprint(armKind=arm_kind, callSurface=call_surface, fields=fields),
        items=tuple(items),
        aggregates=aggregates,
        designEffect=design_effect,
        basis=basis,
        attestationTripWire=attestation_trip_wire,
    )


@pytest.fixture()
def tmp_root(tmp_path):
    """A `model-bench` results root under pytest's tmp dir — never the real one."""
    return tmp_path


#: S2's on-disk fixture packs (`model-bench/tests/fixtures/packs/<name>/`), each a real pack
#: directory `load_pack`/`validate_pack` read — never in-memory `PackRef`s, because the row-count
#: identity and the AST import allowlist both need real files (plan §3.3, §4 S2).
PACKS_DIR = Path(__file__).parent / "fixtures" / "packs"


def pack_fixture(name: str) -> Path:
    """The root directory of the named fixture pack under `tests/fixtures/packs/`."""
    return PACKS_DIR / name


__all__ = [
    "BinaryMetric",
    "ClassificationAggregates",
    "ItemResult",
    "PackMetrics",
    "PackRef",
    "RunResult",
    "ToolCallAggregates",
    "classification_aggregates",
    "deterministic_fields",
    "embeddings_fields",
    "guard_pack",
    "item",
    "model_fields",
    "pack_fixture",
    "run",
    "tmp_root",
]
