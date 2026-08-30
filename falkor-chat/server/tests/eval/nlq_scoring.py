"""Pure, network-free scoring/canonicalization functions for the NL-query-
generation golden-set evaluation (K-055 M6, unit U29c,
`docs/plans/workflow-nl-query-generation-ml.md` §3 "Recommended method").

Implements exactly the comparison rules that note specifies — nothing more:

- **Layer 1** (`score_pair`) — the FR-4/AC-4 gate. Compares the mechanism's raw
  structured result (the JSON `QueryGraphDataTool.run()` returns, decoded:
  `{"items": [...]}` or `{"items": [], "finding": "..."}`) against a golden
  `nlq_golden_set.jsonl` row's `expected` field, using exact match after
  canonicalization — never containment, never an LLM judge.
- **Layer 2** (`layer2_contains`) — the secondary, non-gating sanity check: does
  a rendered natural-language answer *contain* the expected value(s), via
  normalized-substring match. Deliberately weaker than Layer 1 by design (§3).
- `wilson_interval` — the Wilson score interval for a binomial proportion, used
  to report Layer 1's 95% CI alongside the point estimate (§5: "never
  substituted for it").

No FalkorDB, no LLM, no file I/O beyond nothing at all — every function here
takes plain Python values and returns plain Python values, so this module is
unit-testable by feeding it hand-built golden rows and hand-built tool results
(`test_nlq_scoring.py`).

**Scalar-extraction shape rule (this module's own call, not stated verbatim in
the ml note).** A "scalar" golden pair expects the mechanism to have answered
with exactly one focused value: exactly one result row, with exactly one
column. Zero rows, more than one row, or more than one column is scored
incorrect rather than guessed at (e.g. "take the first value") — a mechanism
that returns more or less than a single focused fact for a single-fact
question has not actually answered it, and guessing which of several returned
values was "the" answer would let an over-broad or malformed query pass by
accident.

**Set-extraction rule.** A "set" golden pair (filter-list, aggregation-list, or
conflicting-facts) is compared against every value across every row/column,
flattened and canonicalized — this is deliberately column-name-agnostic (the
raw tool result keys are whatever `RETURN` expression text FalkorDB assigned,
e.g. `"p.name"`, `"count(p)"` — `repository.run_readonly_query`'s own
docstring), and it does mean a query that returned an extra, unrequested
column would inflate the actual set and fail an exact-match comparison against
the golden set — the correct outcome, not a scoring bug: FR-2's contract is to
return exactly what was asked, not "a superset that happens to contain it."
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Layer 1's numeric tolerance for float rounding (ml note §3) — never
# string-equal on a formatted price.
_NUMERIC_EPSILON = 0.01

# Wilson interval's default z for a 95% confidence level (ml note §5).
_Z_95 = 1.959963984540054

_WHITESPACE_RE = re.compile(r"\s+")

# Layer 2's not-found instrument: a rendered answer is read as a correct
# abstention if it contains one of these common phrasings. Heuristic and
# necessarily incomplete (a real LLM can phrase abstention arbitrarily) — this
# is the secondary sanity signal, never the FR-4/AC-4 gate, so a false miss
# here does not affect the number that actually gates the capability.
_ABSTENTION_MARKERS = (
    "not found",
    "no matching",
    "couldn't find",
    "could not find",
    "don't have",
    "do not have",
    "no data",
    "unable to find",
    "no information",
    "not available",
    "no record",
    "i'm not sure",
    "i don't know",
    "cannot find",
    "can't find",
)


def _canon_str(value: Any) -> str:
    """Case-fold + whitespace-collapse a value's string form — mirrors this
    codebase's own `Entity.nameNormalized` convention (ml note §3)."""
    return _WHITESPACE_RE.sub(" ", str(value).strip().casefold())


def _scalar_equal(expected: Any, actual: Any) -> bool:
    """One value-vs-value comparison per the ml note's scalar rule: numeric
    epsilon for two numbers, case/whitespace-folded string equality otherwise.
    A numeric `expected` compared against a non-numeric `actual` (or vice
    versa) is a type mismatch, not stringified and matched — the ml note's own
    "never string-equal on a formatted price" caution generalizes to "never
    coerce a differently-typed actual value into matching."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        # A tiny extra slop (1e-9) keeps the epsilon boundary itself inclusive
        # against float representation error (e.g. 25.00 - 24.99 does not
        # represent as exactly 0.01 in IEEE-754) — the epsilon is a tolerance
        # for rounding, not a razor's-edge cutoff.
        return abs(float(expected) - float(actual)) <= _NUMERIC_EPSILON + 1e-9
    if isinstance(expected, (int, float)) or isinstance(actual, (int, float)):
        return False  # one numeric, one not — never coerced
    return _canon_str(expected) == _canon_str(actual)


def _extract_scalar(items: list[dict[str, Any]]) -> tuple[Any, bool]:
    """See module docstring's scalar-extraction rule. Returns `(value, ok)`;
    `ok` is `False` whenever the raw result isn't shaped like exactly one
    focused fact."""
    if len(items) != 1:
        return None, False
    row = items[0]
    if len(row) != 1:
        return None, False
    return next(iter(row.values())), True


def _flatten_values(items: list[dict[str, Any]]) -> set[str]:
    """Every value across every row/column, canonicalized — see module
    docstring's set-extraction rule."""
    return {_canon_str(v) for row in items for v in row.values()}


@dataclass(frozen=True)
class ScoreOutcome:
    """`score_pair`'s result: `correct` is the Layer 1 verdict; `reason` is a
    short human-readable diagnostic (always populated, whether correct or not
    — useful in the persisted per-pair harness record); `extracted` is
    whatever value(s) this outcome's comparison actually extracted from the
    raw tool result, for report/debugging use."""

    correct: bool
    reason: str
    extracted: Any


def score_pair(row: Mapping[str, Any], tool_result: Mapping[str, Any]) -> ScoreOutcome:
    """Layer 1 (the FR-4/AC-4 gate): score one golden `row` (an
    `nlq_golden_set.jsonl` entry — needs `row["expected"]`/`row["shape"]`)
    against `tool_result` (`QueryGraphDataTool.run()`'s JSON-decoded return
    value: `{"items": [...]}`, optionally with a `"finding"` key on
    abstention). Raises `ValueError` for an `expected.type` outside the three
    this golden set's own integrity checks allow (`scalar | set | not_found`)
    — a defect in the caller/fixture, never silently absorbed.
    """
    expected = row["expected"]
    shape = row["shape"]
    etype = expected["type"]
    items = list(tool_result.get("items", []))

    if etype == "not_found":
        correct = len(items) == 0
        reason = (
            "result is genuinely empty"
            if correct
            else f"expected not_found but got {len(items)} row(s): {items!r}"
        )
        return ScoreOutcome(correct, reason, items)

    if etype == "scalar":
        value, ok = _extract_scalar(items)
        if not ok:
            return ScoreOutcome(
                False,
                f"expected a single scalar row/column, got {items!r}",
                None,
            )
        correct = _scalar_equal(expected["value"], value)
        reason = (
            f"{value!r} matches expected {expected['value']!r}"
            if correct
            else f"{value!r} does not match expected {expected['value']!r}"
        )
        return ScoreOutcome(correct, reason, value)

    if etype == "set":
        actual_set = _flatten_values(items)
        expected_set = {_canon_str(v) for v in expected["values"]}
        if shape == "conflicting-facts":
            # Containment, not exact match: surfacing the required values plus
            # something extra is still correct for this one shape (module
            # docstring / ml note §3).
            correct = expected_set.issubset(actual_set)
            reason = (
                "all expected conflicting values present"
                if correct
                else f"missing {sorted(expected_set - actual_set)} from {sorted(actual_set)}"
            )
        else:
            correct = expected_set == actual_set
            reason = (
                "exact set match"
                if correct
                else f"expected {sorted(expected_set)}, got {sorted(actual_set)}"
            )
        return ScoreOutcome(correct, reason, sorted(actual_set))

    raise ValueError(f"unknown expected.type {etype!r}")


def layer2_contains(row: Mapping[str, Any], rendered_answer: str) -> bool:
    """Layer 2 (secondary, non-gating sanity check): does `rendered_answer` —
    the mechanism's final natural-language sentence, not its raw structured
    result — contain the golden `expected` value(s), via normalized-substring
    match (ml note §3's default instrument; no LLM judge).

    - scalar: the expected value's string form must appear in the answer.
    - set (filter-list / aggregation-list / conflicting-facts): EVERY expected
      value must appear — a rendered list answer that drops one item is not a
      complete answer, even as a sanity check.
    - not_found: the answer must contain a recognizable abstention phrase
      (heuristic; see `_ABSTENTION_MARKERS`'s own docstring for why this is
      necessarily approximate and why that's acceptable for a secondary
      signal).
    """
    expected = row["expected"]
    etype = expected["type"]
    normalized = _canon_str(rendered_answer)

    if etype == "not_found":
        return any(marker in normalized for marker in _ABSTENTION_MARKERS)
    if etype == "scalar":
        return _canon_str(expected["value"]) in normalized
    if etype == "set":
        return all(_canon_str(v) in normalized for v in expected["values"])
    raise ValueError(f"unknown expected.type {etype!r}")


def wilson_interval(successes: int, n: int, *, z: float = _Z_95) -> tuple[float, float]:
    """The Wilson score interval for a binomial proportion `successes/n` at
    confidence level implied by `z` (default `z=1.96`, ~95% — ml note §5:
    "Wilson 95% CI reported alongside, never substituted for it"). Returns
    `(lower, upper)`, each clamped to `[0.0, 1.0]`.

    Raises `ValueError` if `n <= 0` — there is no proportion to bound.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    """Parse an `nlq_golden_set.jsonl`-shaped fixture into a list of row
    dicts. No validation beyond JSON parsing — structural integrity is
    `test_nlq_golden_set_integrity.py`'s job, not this function's; the live
    harness that calls this trusts that suite already passed."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
