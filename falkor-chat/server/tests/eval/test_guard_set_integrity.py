"""Structural integrity checks for the guard-judge calibration golden set
(K-027 item 4, `docs/plans/golden-set-expansion-ml.md` §7 step 3 / F4).

Genuinely offline — no FalkorDB, no LLM/judge call, no `pytest.mark.live` marker.
Mirrors `test_golden_set_integrity.py`'s pattern for the *retrieval* golden set
(`golden_retrieval.jsonl`), applied instead to `golden_guards.jsonl`: unique ids,
required schema fields present, `tier`/`path` enum validity, `expected` is a bool,
the `boundary` tier's `expected: false` invariant, and per-stratum-and-path
**minimum** counts expressed as inequalities — not exact literals — so this file
does not need editing the next time the fixture grows (F4). The five *exact*
literal-constant asserts belong to `test_guard_calibration_live.py` only (F3);
this file intentionally never hardcodes the current total.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_EVAL_DIR = Path(__file__).resolve().parent
_GUARDS_PATH = _EVAL_DIR / "golden_guards.jsonl"

_REQUIRED_FIELDS = (
    "id", "tier", "path", "r1_probe", "condition",
    "understanding", "turns", "expected", "label_rationale",
)
_VALID_TIERS = {"clear_advance", "clear_suspend", "boundary"}
_VALID_PATHS = {"understanding", "turns"}


def _load_guard_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _GUARDS_PATH.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"{_GUARDS_PATH.name}:{line_no}: invalid JSON — {exc}"
                ) from exc
    return rows


_GUARD_ROWS = _load_guard_rows()
_GUARD_IDS = [row.get("id", f"<row {i}>") for i, row in enumerate(_GUARD_ROWS)]
_BOUNDARY_ROWS = [r for r in _GUARD_ROWS if r.get("tier") == "boundary"]
_BOUNDARY_IDS = [r.get("id", f"<row {i}>") for i, r in enumerate(_BOUNDARY_ROWS)]


def test_guard_set_file_exists_and_nonempty() -> None:
    assert _GUARDS_PATH.exists(), f"missing fixture: {_GUARDS_PATH}"
    assert _GUARD_ROWS, f"{_GUARDS_PATH.name} has no rows"


def test_guard_ids_are_unique() -> None:
    assert len(_GUARD_IDS) == len(set(_GUARD_IDS)), "duplicate guard ids found"


@pytest.mark.parametrize("row", _GUARD_ROWS, ids=_GUARD_IDS)
def test_guard_row_has_required_fields(row: dict[str, Any]) -> None:
    for field in _REQUIRED_FIELDS:
        assert field in row, f"{row.get('id', '?')}: missing field {field!r}"


@pytest.mark.parametrize("row", _GUARD_ROWS, ids=_GUARD_IDS)
def test_guard_row_tier_is_valid(row: dict[str, Any]) -> None:
    assert row["tier"] in _VALID_TIERS, (
        f"{row['id']}: tier {row['tier']!r} not in {_VALID_TIERS}"
    )


@pytest.mark.parametrize("row", _GUARD_ROWS, ids=_GUARD_IDS)
def test_guard_row_path_is_valid(row: dict[str, Any]) -> None:
    assert row["path"] in _VALID_PATHS, (
        f"{row['id']}: path {row['path']!r} not in {_VALID_PATHS}"
    )


@pytest.mark.parametrize("row", _GUARD_ROWS, ids=_GUARD_IDS)
def test_guard_row_expected_is_bool(row: dict[str, Any]) -> None:
    assert isinstance(row["expected"], bool), (
        f"{row['id']}: expected must be a bool, got {type(row['expected']).__name__}"
    )


@pytest.mark.parametrize("row", _BOUNDARY_ROWS, ids=_BOUNDARY_IDS)
def test_boundary_rows_are_always_expected_false(row: dict[str, Any]) -> None:
    """Boundary labels are policy choices, never gated (archived §4.1/§4.2) — every
    `boundary` row in this fixture is drafted `expected: false` per the designed
    bias-to-suspend operating point (plan §5/§10)."""
    assert row["expected"] is False, (
        f"{row['id']}: boundary row must be expected: false, got {row['expected']!r}"
    )


def _count(tier: str, path: str | None = None) -> int:
    return sum(
        1 for r in _GUARD_ROWS
        if r.get("tier") == tier and (path is None or r.get("path") == path)
    )


def test_stratum_and_path_minimum_counts() -> None:
    """Inequalities, not exact literals (plan §7 step 3 / F4) — mirrors
    `test_golden_set_integrity.py`'s `30 <= len(rows) <= 50` style so this test
    survives the fixture growing further without an edit. Floors match the
    K-027 item 4 target composition (plan §3.4): `clear_advance` 30 (18
    `understanding` / 12 `turns`), `clear_suspend` 40 (24/16), `boundary` 15 (9/6)."""
    assert _count("clear_advance") >= 30, "clear_advance total below the K-027 item 4 floor"
    assert _count("clear_advance", "understanding") >= 18
    assert _count("clear_advance", "turns") >= 12
    assert _count("clear_suspend") >= 40, "clear_suspend total below the K-027 item 4 floor"
    assert _count("clear_suspend", "understanding") >= 24
    assert _count("clear_suspend", "turns") >= 16
    assert _count("boundary") >= 15, "boundary total below the K-027 item 4 floor"
    assert _count("boundary", "understanding") >= 9
    assert _count("boundary", "turns") >= 6
