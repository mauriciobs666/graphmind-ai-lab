"""Guard-judge calibration harness (K-027 item 3).

Builds real `guards.evaluate_guard` calls from `golden_guards.jsonl` rows, per
`docs/archive/plans/m3-guard-calibration.md` §5.1's table with the one addendum
`docs/plans/guard-judge-calibration-ml.md` §4 specifies (`run =
{"ws": "ws:golden-eval", "modelOverrides": {}}`, not the archived doc's literal
`run = {}`, which — post-K-042 — lands in a branch production never reaches).

Pure, network-free plumbing; the live LLM only enters through whatever `judge`
callable a caller injects (`test_guard_calibration_live.py`, or a `StubJudge` here
for the offline suite, `test_guard_calibration.py`).

**NOT a pytest test module itself** — a library `test_guard_calibration.py` (offline)
and `test_guard_calibration_live.py` (live, `pytest.mark.live`) both import from here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from falkorchat.guards import evaluate_guard

_EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_GUARDS_PATH = _EVAL_DIR / "golden_guards.jsonl"

# Prose that deliberately does NOT parse as JSON (F5: the fallback trigger is
# truthiness, not presence — `step_output` must fail to parse to a non-empty dict,
# and `ctx` below carries no `understanding` key, for a `path: "turns"` case to
# actually drive the RECENT-TURNS fallback rather than the primary path).
_TURNS_PATH_STEP_OUTPUT = (
    "I'm still working out what the user needs before I can summarize it."
)


def load_golden_guards(path: Path | None = None) -> list[dict[str, Any]]:
    """Parse `golden_guards.jsonl` (or an override `path`, for tests). Never mutated
    by anything downstream — the fixture is read-only (out-of-scope to expand,
    K-027 item 4)."""
    rows: list[dict[str, Any]] = []
    with (path or GOLDEN_GUARDS_PATH).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_call(row: Mapping[str, Any]) -> dict[str, Any]:
    """One fixture row → the kwargs `guards.evaluate_guard` takes, per archived §5.1's
    table with the ml-note §4 `run` addendum applied."""
    guard = json.dumps({"kind": "llm", "text": row["condition"]})
    if row["path"] == "turns":
        step_output = _TURNS_PATH_STEP_OUTPUT
    else:
        step_output = json.dumps({"understanding": row["understanding"]})
    return {
        "guard": guard,
        "ctx": {},
        "run": {"ws": "ws:golden-eval", "modelOverrides": {}},
        "step_output": step_output,
        "thread": row.get("turns") or [],
    }


class RecordingJudge:
    """Wraps a judge callable and records the extract-then-judge evidence it was
    actually handed, plus its raw (pre-`_coerce_verdict`) output — the thin
    recording wrapper archived §5.1 requires ("in the spirit of the existing
    `StubJudge.calls` pattern"), so the harness can assert the evidence path was
    really taken and compute `coercion_flip_rate` (raw vs. final decision).

    Propagates the inner judge's `accepts_run` capability flag (K-042) so
    `guards.evaluate_guard` forwards `run=` exactly when the real production judge
    would receive it — never unconditionally.
    """

    def __init__(self, inner: Callable[..., Any]) -> None:
        self._inner = inner
        self.accepts_run = bool(getattr(inner, "accepts_run", False))
        self.calls: list[dict[str, Any]] = []

    def __call__(self, condition, *, understanding, recent_turns, ctx, step_output,
                 model=None, run=None):
        kwargs: dict[str, Any] = {
            "understanding": understanding, "recent_turns": recent_turns,
            "ctx": ctx, "step_output": step_output,
        }
        if model is not None:
            kwargs["model"] = model
        if self.accepts_run:
            kwargs["run"] = run
        raw = self._inner(condition, **kwargs)
        self.calls.append({
            "condition": condition, "understanding": understanding,
            "recent_turns": recent_turns, "raw": raw,
        })
        return raw


def path_taken(call: Mapping[str, Any]) -> str:
    """Which evidence tier a recorded call actually carried: "understanding",
    "turns", or "neither" (the last is always a harness bug, never a valid case)."""
    if call["understanding"]:
        return "understanding"
    if call["recent_turns"]:
        return "turns"
    return "neither"


def assert_path_matches(row: Mapping[str, Any], call: Mapping[str, Any]) -> None:
    """Archived §5.1: "This assertion is not optional" — a `path: "turns"` fixture
    row that silently drove the understanding branch (or vice versa) is a worthless
    data point that would still score. Raises `AssertionError` naming the mismatch.
    """
    actual = path_taken(call)
    expected = row["path"]
    assert actual == expected, (
        f"case {row['id']!r}: fixture declares path={expected!r} but the judge "
        f"actually received evidence tier {actual!r} "
        f"(understanding={call['understanding']!r}, recent_turns={call['recent_turns']!r})"
    )


# ── running one case (k replicates) ───────────────────────────────────────────


@dataclass(frozen=True)
class ReplicateResult:
    """One of the `k` replicate calls for one case (archived §5.2).

    `raw_decision` is the judge's own verdict, captured by `RecordingJudge` BEFORE
    `guards._coerce_verdict` runs — `None` when the raw output wasn't a mapping with
    a real-`bool` `True` `decision` (mirrors `_coerce_verdict`'s own strict check).
    `final_decision`/`final_rationale` are the post-coercion `GuardVerdict`.
    """

    raw_decision: bool | None
    raw_rationale: str
    final_decision: bool
    final_rationale: str


@dataclass(frozen=True)
class CaseResult:
    """One fixture case's `k` replicate results, carrying the fixture metadata the
    metric functions below stratify on (`tier`, `path`, `r1_probe`, `expected`)."""

    id: str
    tier: str
    path: str
    r1_probe: bool
    expected: bool
    replicates: list[ReplicateResult] = field(default_factory=list)


def run_case(row: Mapping[str, Any], judge: Callable[..., Any], *, k: int = 3) -> CaseResult:
    """Run one fixture row through the real `guards.evaluate_guard` `k` times.

    Asserts the evidence path was actually taken on every replicate (archived
    §5.1 — "not optional"): a fresh `RecordingJudge` wraps `judge` per call so each
    replicate's recorded evidence is checked independently.
    """
    call = build_call(row)
    replicates: list[ReplicateResult] = []
    for _ in range(k):
        recorder = RecordingJudge(judge)
        verdict = evaluate_guard(
            call["guard"], ctx=call["ctx"], run=call["run"],
            step_output=call["step_output"], thread=call["thread"], judge=recorder,
        )
        assert_path_matches(row, recorder.calls[0])
        raw = recorder.calls[0]["raw"]
        raw_decision = raw.get("decision") if isinstance(raw, Mapping) else None
        raw_decision = raw_decision if isinstance(raw_decision, bool) else None
        raw_rationale = raw.get("rationale") if isinstance(raw, Mapping) else ""
        raw_rationale = raw_rationale if isinstance(raw_rationale, str) else ""
        replicates.append(ReplicateResult(
            raw_decision=raw_decision, raw_rationale=raw_rationale,
            final_decision=verdict.decision, final_rationale=verdict.rationale,
        ))
    return CaseResult(
        id=row["id"], tier=row["tier"], path=row["path"],
        r1_probe=bool(row.get("r1_probe")), expected=bool(row["expected"]),
        replicates=replicates,
    )


# ── metrics (archived §4, §5.2) ───────────────────────────────────────────────


def case_majority(replicates: list[ReplicateResult]) -> bool:
    """2-of-3 (or general majority) final-decision vote across replicates."""
    trues = sum(1 for r in replicates if r.final_decision)
    return trues * 2 > len(replicates)


def false_advance_rate(cases: list[CaseResult]) -> dict[str, Any]:
    """**G1** — archived §4.1: over the 10 `clear_suspend` cases, counted per
    INDIVIDUAL judge call (not per-case majority) — production takes exactly one
    sample, so a case that advances 1-in-3 times is a real 33% false-advance risk
    on the live path that majority-voting would hide."""
    suspend_cases = [c for c in cases if c.tier == "clear_suspend"]
    calls = [r for c in suspend_cases for r in c.replicates]
    advances = sum(1 for r in calls if r.final_decision)
    n_calls = len(calls)
    return {
        "advances": advances, "n_calls": n_calls, "n_cases": len(suspend_cases),
        "rate": advances / n_calls if n_calls else 0.0,
    }


def false_advance_rate_all(cases: list[CaseResult]) -> dict[str, Any]:
    """`FAR_all` — reported, not gated (archived §4.1): every `expected:false` case
    (`clear_suspend` + `boundary`), same per-call counting as G1, for continuity."""
    denom_cases = [c for c in cases if not c.expected]
    calls = [r for c in denom_cases for r in c.replicates]
    advances = sum(1 for r in calls if r.final_decision)
    n_calls = len(calls)
    return {
        "advances": advances, "n_calls": n_calls, "n_cases": len(denom_cases),
        "rate": advances / n_calls if n_calls else 0.0,
    }


def advance_recall(cases: list[CaseResult]) -> dict[str, Any]:
    """**G2** — archived §4.1: over the 11 `clear_advance` cases, per-CASE majority
    (k=3 ⇒ 2-of-3) — this is the "ability" arm; majority voting is legitimate here,
    unlike G1's "risk" arm."""
    advance_cases = [c for c in cases if c.tier == "clear_advance"]
    majorities = [case_majority(c.replicates) for c in advance_cases]
    advances = sum(majorities)
    n = len(advance_cases)
    return {"advances": advances, "n_cases": n, "rate": advances / n if n else 0.0}


def confusion_matrix(cases: list[CaseResult]) -> dict[str, int]:
    """2×2 confusion matrix over the given cases, predicted = per-case majority."""
    tp = fp = tn = fn = 0
    for c in cases:
        predicted = case_majority(c.replicates)
        if c.expected and predicted:
            tp += 1
        elif c.expected and not predicted:
            fn += 1
        elif not c.expected and predicted:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def cohens_kappa(cases: list[CaseResult]) -> dict[str, Any]:
    """Cohen's κ over `cases` (archived §4.2: report over the 21 clear cases only),
    with both raters' marginals and the set prevalence — a bare kappa number is
    uninterpretable per §3.3's prevalence-effect finding. `diagnostic — not a gate`.
    """
    n = len(cases)
    cm = confusion_matrix(cases)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    po = (tp + tn) / n if n else 0.0
    a_pos = (tp + fn) / n if n else 0.0  # rater A (expected) positive rate = prevalence
    b_pos = (tp + fp) / n if n else 0.0  # rater B (judge majority) positive rate
    pe = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return {
        "kappa": kappa, "po": po, "pe": pe, "n": n,
        "rater_a_positive_rate": a_pos, "rater_b_positive_rate": b_pos,
        "prevalence": a_pos, "confusion_matrix": cm,
    }


def coercion_flip_rate(cases: list[CaseResult], *, r1_probe_only: bool = False) -> dict[str, Any]:
    """§4.3: share of calls where the judge said advance (`raw_decision is True`) and
    `_coerce_verdict` overrode it to suspend. Report overall and restricted to
    `r1_probe: true` cases (`r1_probe_only=True`)."""
    selected = [c for c in cases if (c.r1_probe if r1_probe_only else True)]
    calls = [r for c in selected for r in c.replicates]
    flips = sum(1 for r in calls if r.raw_decision is True and not r.final_decision)
    n_calls = len(calls)
    return {"flips": flips, "n_calls": n_calls, "rate": flips / n_calls if n_calls else 0.0}


def flip_rate(cases: list[CaseResult]) -> dict[str, Any]:
    """§5.2: share of CASES whose `k` replicates disagree on final decision — replicate
    instability, reported even when both gates pass."""
    unstable = [c.id for c in cases if len({r.final_decision for r in c.replicates}) > 1]
    n = len(cases)
    return {
        "unstable_ids": unstable, "unstable": len(unstable), "n_cases": n,
        "rate": len(unstable) / n if n else 0.0,
    }


def per_path_breakdown(cases: list[CaseResult]) -> dict[str, dict[str, Any]]:
    """§4.2: confusion matrix + accuracy per evidence path (`understanding`/`turns`)
    — the risk-#4 fallback-path-cost estimate."""
    result: dict[str, dict[str, Any]] = {}
    for path in ("understanding", "turns"):
        subset = [c for c in cases if c.path == path]
        if not subset:
            continue
        cm = confusion_matrix(subset)
        n = len(subset)
        accuracy = (cm["tp"] + cm["tn"]) / n if n else 0.0
        result[path] = {"n_cases": n, "confusion_matrix": cm, "accuracy": accuracy}
    return result


def materiality_probe_failed(cases_by_id: dict[str, CaseResult]) -> bool:
    """§4.2/§7: the judge is pattern-matching `len(missing)` rather than assessing
    sufficiency iff it suspends ALL of `ca-04`/`ca-05`/`ca-08` (materiality probes,
    non-empty `missing` that is immaterial to research) AND advances `cs-04` (the
    control — empty `missing` that IS material). Only this combined bloc is a
    blocker-grade finding (archived §7); a single miss is not."""
    probe_ids = ("ca-04", "ca-05", "ca-08")
    control_id = "cs-04"
    if not all(cid in cases_by_id for cid in (*probe_ids, control_id)):
        return False
    probes_suspended = all(
        not case_majority(cases_by_id[cid].replicates) for cid in probe_ids
    )
    control_advanced = case_majority(cases_by_id[control_id].replicates)
    return probes_suspended and control_advanced
