"""Shared hand-built fixtures for the S1 suite.

S1 has no pack loader and makes no model calls (`docs/plans/small-model-benchmarking.md` §4 S1),
so every fixture here is an in-memory record built by hand. Nothing in this file touches the
network, LM Studio, or any path outside `model-bench/`.
"""

from __future__ import annotations

from typing import Any

import pytest

from modelbench.packs import PackMetrics, PackRef
from modelbench.results import (
    BinaryMetric,
    ClassificationAggregates,
    ItemResult,
    RunResult,
    ToolCallAggregates,
)

# A complete, valid `model` fingerprint field set at benchSchemaVersion 1 (plan §3.4.2). Tests
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
    "lmsCliCommit": "07b7252",
    "residentModelsAtStart": [],
    "residentModelsAtEnd": [{"modelKey": "qwen/qwen3-4b-2507", "sizeBytes": 2 << 30}],
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
        latencyMs=1300.0,
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
    role: str = "guard-judge",
    items: list[ItemResult] | None = None,
    aggregates: Any = None,
    session_id: str | None = "s1",
    design_effect: float = 1.0,
    basis: str = "by-construction",
) -> RunResult:
    from modelbench.fingerprint import Fingerprint

    fields = fingerprint_fields
    if fields is None:
        fields = model_fields() if arm_kind == "model" else deterministic_fields()
    items = items if items is not None else []
    if aggregates is None:
        hits = sum(1 for it in items if it.outcome == "pass")
        aggregates = classification_aggregates(hits, len(items))
    return RunResult(
        runId=run_id,
        sessionId=session_id,
        role=role,
        armKind=arm_kind,
        fingerprint=Fingerprint(armKind=arm_kind, fields=fields),
        items=tuple(items),
        aggregates=aggregates,
        designEffect=design_effect,
        basis=basis,
    )


@pytest.fixture()
def tmp_root(tmp_path):
    """A `model-bench` results root under pytest's tmp dir — never the real one."""
    return tmp_path


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
    "guard_pack",
    "item",
    "model_fields",
    "run",
    "tmp_root",
]
