"""Live guard-judge calibration run (K-027 item 3): is the LLM-as-judge behind the
`intake -> research` fuzzy guard fit to wire live on the shipped local 4B?

Protocol: `docs/archive/plans/m3-guard-calibration.md` (frozen; the gate, the k=3
replicate design, the statistical-honesty caveat). Current-code addendum:
`docs/plans/guard-judge-calibration-ml.md` (the `run` construction, the model/
temperature/provider pre-flight). All deterministic plumbing (fixture assembly,
path assertion, G1/G2/kappa/coercion-flip/flip-rate math) is unit-tested offline in
`test_guard_calibration.py`; this module only drives that plumbing against a REAL
judge and writes the report.

**Marker-gated**, mirroring `test_workflow_live.py`/`test_judge_live.py`:
`pytest.mark.live`, DESELECTED by default (`addopts = -m "not live"`), so the
standard network-free `pytest -q` baseline never touches LM Studio. Run explicitly:

    cd server && .venv/bin/python -m pytest -m live -s tests/eval/test_guard_calibration_live.py

Needs: LM Studio serving `qwen/qwen3-4b-2507` at whatever `FALKORCHAT_OPENCODE_CONFIG`
resolves to (see `_opencode_path()` below). **No FalkorDB dependency** — unlike
`test_workflow_live.py`, this harness never reads a live thread; every `thread` value
comes straight from the fixture (`golden_guards.jsonl`'s own `turns` field, already
`repository.read_thread` shape per archived F4), so `guards.evaluate_guard` is
exercised exactly as production calls it, with zero graph I/O.

**Model config — deliberately NOT `ModelGateway.from_env()`.** This is a pytest test
under `server/tests/`, so it inherits `tests/conftest.py`'s autouse
`_model_config_env` fixture, which redirects `FALKORCHAT_OPENCODE_CONFIG`/
`FALKORCHAT_MODEL_CONFIG` to the offline `tests/data/*.json` dim-4 fixtures for EVERY
test in the suite, unconditionally. Two established precedents in this exact
codebase handle that redirection differently, and this module follows the one that
fits: `test_workflow_live.py`/`test_judge_live.py` (D7 mechanism 2) sidestep the
whole config layer and construct `OpenAICompatibleLLM` directly from env-var
literals — but that means the `config/models.json` `temperature: 0` pin this unit
adds (K-027 item 3 step 3) would never reach the request, defeating half the point
of the exercise. `test_golden_set_integrity.py` (D7 mechanism 1) instead constructs
`ProviderCatalog`/`Overlay` directly from the REAL files, bypassing `.from_env()`'s
env-var indirection entirely — and that is what this module does too, so the
harness actually resolves through `modelconfig._resolve_element` and picks up the
temperature pin. `_AMBIENT_OPENCODE_CONFIG` is captured at MODULE IMPORT time
(pytest collection, before any per-test fixture — including the autouse one —
executes), so it reflects whatever the invoking shell actually had set, not the
redirected value.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from falkorchat import app, modelconfig

from guard_calibration import (
    advance_recall,
    case_majority,
    coercion_flip_rate,
    cohens_kappa,
    confusion_matrix,
    false_advance_rate,
    false_advance_rate_all,
    flip_rate,
    load_golden_guards,
    materiality_probe_failed,
    per_path_breakdown,
    run_case,
)

pytestmark = pytest.mark.live

_EVAL_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _EVAL_DIR.parents[1]
_REPO_ROOT = _SERVER_DIR.parent
_REPORTS_DIR = _REPO_ROOT / "docs" / "test-reports"
_EXAMPLE_OPENCODE_PATH = _REPO_ROOT / "config" / "opencode.example.json"
_GOLDEN_PATH = _EVAL_DIR / "golden_guards.jsonl"

# Captured at collection time — see module docstring's "Model config" section.
_AMBIENT_OPENCODE_CONFIG = os.environ.get("FALKORCHAT_OPENCODE_CONFIG")

MODEL_REF = "lmstudio/qwen/qwen3-4b-2507"
K_REPLICATES = 3

# archived §6.1's caveat, required VERBATIM adjacent to the verdict (archived §8 item 3).
_CAVEAT_SENTENCE = (
    "This gate is a one-sided screen at n=21 hand-labeled cases. A failure is strong "
    "evidence the judge is unfit. A pass means only that no large defect was detected "
    "at a sample size that could not have detected a small one. It is not a "
    "calibration certificate."
)


def _opencode_path() -> str:
    """The real provider file to build the gateway from — the ambient
    `FALKORCHAT_OPENCODE_CONFIG` if the invoking shell set one, else the repo's
    `config/opencode.example.json` (verified reachable on this box during this
    unit's build; see the eventual report's provenance header for what actually
    resolved on the box the run happened on — never assumed here)."""
    return _AMBIENT_OPENCODE_CONFIG or str(_EXAMPLE_OPENCODE_PATH)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT,
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _fetch_quantization(base_url: str, model_id: str) -> str | None:
    """Re-verify quantization LIVE at run time (per the brief — never trust a stale
    note): `curl <root>/api/v0/models`, LM-Studio's own extended endpoint."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(f"{root}/api/v0/models", timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    for entry in data.get("data", []):
        if entry.get("id") == model_id:
            return entry.get("quantization")
    return None


# ── live-dependency gating ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def models() -> modelconfig.ModelGateway:
    """A REAL `ModelGateway` over the real `config/models.json` + a real provider
    file — see the module docstring for why `.from_env()` is deliberately not used.
    Skips (never fails) if the resolved provider/model is unreachable.
    """
    # The example provider file also declares an unrelated `openai` provider whose
    # `{env:OPENAI_API_KEY}` substitution is resolved eagerly for EVERY declared
    # provider at `ModelGateway.__init__` (`_build_providers` — modelconfig.py:560),
    # not lazily per `.llm()` call — so it must be set even though this harness
    # never touches that provider.
    os.environ.setdefault("OPENAI_API_KEY", "unused-in-guard-calibration-eval")

    opencode_path = _opencode_path()
    if not Path(opencode_path).exists():
        pytest.skip(
            f"FALKORCHAT_OPENCODE_CONFIG resolves to {opencode_path!r}, which does "
            f"not exist"
        )
    try:
        catalog = modelconfig.ProviderCatalog.load(opencode_path)
        overlay = modelconfig.Overlay.load(str(modelconfig.DEFAULT_MODEL_CONFIG_PATH))
        gateway = modelconfig.ModelGateway(catalog, overlay)
    except modelconfig.ModelConfigError as exc:
        pytest.skip(f"model config unusable ({opencode_path!r}): {exc!r}")

    try:
        probe = gateway.llm("guard", ws="ws:golden-eval", overrides={})
        probe.complete([{"role": "user", "content": "Reply with the single word: ok"}])
    except Exception as exc:
        pytest.skip(
            f"guard model ({MODEL_REF}) unreachable via {opencode_path!r} ({exc!r}) "
            f"— confirm FALKORCHAT_OPENCODE_CONFIG points at a reachable LM Studio "
            f"instance actually serving {MODEL_REF} (see "
            f"docs/plans/guard-judge-calibration-ml.md §5's pre-flight finding)"
        )
    return gateway


@pytest.fixture(scope="module")
def judge(models: modelconfig.ModelGateway):
    return app._build_llm_judge(models)


# ── the run ───────────────────────────────────────────────────────────────────


def test_guard_judge_calibration_live(
    models: modelconfig.ModelGateway, judge: Any
) -> None:
    """Runs all 26 golden cases x k=3 replicates (78 real judge calls) through the
    real `guards.evaluate_guard`, computes the archived §4 metric set, and writes
    `docs/test-reports/guard-judge-calibration-<date>.md` per archived §8.

    **Non-gating (archived §7's own instruction: a failing verdict is not this
    harness's problem to fix or route around).** This test asserts only that the
    harness ran to completion correctly (right sample sizes, the temperature pin
    reached the request, the report was written) — never a pass/fail threshold on
    the judge's own G1/G2 numbers. The report's VERDICT line carries that
    judgment; a human (teco/architect) acts on it downstream.
    """
    resolved_llm, resolution = models.resolve_llm("guard", ws="ws:golden-eval", overrides={})
    resolved = resolution.primary
    assert resolved.ref == MODEL_REF, (
        f"defaults.guard resolved to {resolved.ref!r}, not {MODEL_REF!r} — "
        f"config/models.json may have drifted since this unit was built"
    )
    assert resolved.params.get("temperature") == 0, (
        "config/models.json's temperature:0 pin for "
        f"{MODEL_REF!r} did not reach the resolved request params "
        f"(got params={resolved.params!r}) — the K-027 item 3 step-3 config change "
        f"is not flowing through modelconfig._resolve_element as expected"
    )

    # LM Studio's `/api/v0/models` `id` field is the bare model id (`resolved.model`,
    # e.g. "qwen/qwen3-4b-2507"), NOT the full `<provider>/<model>` ref (`MODEL_REF`,
    # "lmstudio/qwen/qwen3-4b-2507") — passing the full ref here always misses.
    quantization = _fetch_quantization(resolved.base_url, resolved.model) or "(unknown)"

    rows = load_golden_guards()
    assert len(rows) == 85, f"golden_guards.jsonl has {len(rows)} rows, expected 85"

    cases = [run_case(row, judge, k=K_REPLICATES) for row in rows]
    cases_by_id = {c.id: c for c in cases}
    assert len(cases) == 85
    assert sum(len(c.replicates) for c in cases) == 85 * K_REPLICATES

    clear_cases = [c for c in cases if c.tier in ("clear_advance", "clear_suspend")]
    boundary_cases = [c for c in cases if c.tier == "boundary"]
    assert len(clear_cases) == 70
    assert len(boundary_cases) == 15

    g1 = false_advance_rate(cases)
    g1_all = false_advance_rate_all(cases)
    g2 = advance_recall(cases)
    kappa = cohens_kappa(clear_cases)
    boundary_confusion = confusion_matrix(boundary_cases)
    coercion_overall = coercion_flip_rate(cases)
    coercion_r1 = coercion_flip_rate(cases, r1_probe_only=True)
    flips = flip_rate(cases)
    path_breakdown = per_path_breakdown(clear_cases)
    materiality_failed = materiality_probe_failed(cases_by_id)
    tn05 = cases_by_id.get("tn-05")
    boundary_suspended = sum(
        1 for c in boundary_cases if not case_majority(c.replicates)
    )
    conservatism_rate = boundary_suspended / len(boundary_cases) if boundary_cases else 0.0

    g1_pass = g1["rate"] <= 0.10
    g2_pass = g2["rate"] >= 0.80
    verdict = "wire" if (g1_pass and g2_pass) else "block"

    report = _build_report(
        today=date.today(), resolved=resolved, quantization=quantization,
        cases=cases, g1=g1, g1_all=g1_all, g2=g2, verdict=verdict, kappa=kappa,
        boundary_confusion=boundary_confusion, coercion_overall=coercion_overall,
        coercion_r1=coercion_r1, flips=flips, path_breakdown=path_breakdown,
        materiality_failed=materiality_failed, tn05=tn05,
        conservatism_rate=conservatism_rate, boundary_cases=boundary_cases,
    )
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / f"guard-judge-calibration-{date.today().isoformat()}.md"
    out_path.write_text(report, encoding="utf-8")

    print(
        f"\nwrote {out_path}\n"
        f"G1 false-advance = {g1['rate']:.1%} (n={g1['n_cases']} cases / "
        f"{g1['n_calls']} calls) · G2 advance-recall = {g2['rate']:.1%} "
        f"(n={g2['n_cases']} cases) · VERDICT: {verdict}"
    )


# ── report rendering (archived §8) ────────────────────────────────────────────


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _build_report(
    *, today: date, resolved, quantization: str, cases, g1, g1_all, g2, verdict: str,
    kappa, boundary_confusion, coercion_overall, coercion_r1, flips, path_breakdown,
    materiality_failed: bool, tn05, conservatism_rate: float, boundary_cases,
) -> str:
    lines: list[str] = []
    lines.append("# Guard-judge calibration report (K-027 item 3)")
    lines.append("")
    lines.append(
        "> **Status:** active · **Owner:** `tdd-engineer` · **Tracks:** K-027 item 3 (M3.5)"
    )
    lines.append("")
    lines.append(
        "Supersedes the stale recall=0.818 / false-advance=0.067 numbers currently "
        "quoted in `docs/BACKLOG.md` K-027 item 3 — those numbers predate the "
        "K-027 item 1 parse fix, bypassed `guards.evaluate_guard`, and used the wrong "
        "G1 denominator (full explanation: "
        "`docs/plans/guard-judge-calibration-ml.md` §3)."
    )
    lines.append("")

    # ── provenance ──
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- **model ref:** `{resolved.ref}`")
    lines.append(f"- **quantization (live-verified at run time):** `{quantization}`")
    lines.append(f"- **temperature:** `{resolved.params.get('temperature')}`")
    lines.append(f"- **resolved baseURL:** `{resolved.base_url}`")
    lines.append(f"- **prompt revision (HEAD at run time):** `{_git_head()}`")
    lines.append(f"- **fixture sha256:** `{_sha256(_GOLDEN_PATH)}`")
    lines.append(f"- **k (replicates per case):** {len(cases[0].replicates) if cases else '?'}")
    lines.append(f"- **date:** {today.isoformat()}")
    lines.append("")

    # ── verdict ──
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"**G1 false-advance = {_fmt_pct(g1['rate'])} (n={g1['n_cases']} cases / "
        f"{g1['n_calls']} calls) · G2 advance-recall = {_fmt_pct(g2['rate'])} "
        f"(n={g2['n_cases']} cases) · VERDICT: {verdict}**"
    )
    lines.append("")
    lines.append(f"> {_CAVEAT_SENTENCE}")
    lines.append("")
    lines.append(
        f"`FAR_all` (reported, not gated — all 15 `expected:false` cases): "
        f"{_fmt_pct(g1_all['rate'])} ({g1_all['advances']}/{g1_all['n_calls']} calls)."
    )
    lines.append("")

    # ── kappa, diagnostic ──
    lines.append("## Cohen's kappa — diagnostic, not a gate")
    lines.append("")
    lines.append(
        f"- **kappa:** {kappa['kappa']:.3f} (n={kappa['n']} clear cases; observed "
        f"agreement po={kappa['po']:.3f}, chance agreement pe={kappa['pe']:.3f})"
    )
    lines.append(
        f"- **rater A (human label) positive rate / prevalence:** "
        f"{_fmt_pct(kappa['rater_a_positive_rate'])}"
    )
    lines.append(
        f"- **rater B (judge, per-case majority) positive rate:** "
        f"{_fmt_pct(kappa['rater_b_positive_rate'])}"
    )
    lines.append(
        "- Comparable only across runs on this exact fixture file "
        "(`docs/archive/plans/m3-guard-calibration.md` §3.3's prevalence-effect finding)."
    )
    lines.append("")

    # ── boundary stratum ──
    lines.append("## Boundary stratum (n=5) — reported, never gated")
    lines.append("")
    lines.append(
        f"- **conservatism rate (share suspended):** {_fmt_pct(conservatism_rate)} "
        f"— expected high; a low value is an early warning the bias-to-suspend "
        f"posture is eroding."
    )
    lines.append(
        f"- **boundary confusion matrix (expected always `false`):** "
        f"tp={boundary_confusion['tp']} fp={boundary_confusion['fp']} "
        f"tn={boundary_confusion['tn']} fn={boundary_confusion['fn']}"
    )
    lines.append("")

    # ── per-path breakdown ──
    lines.append("## Per-path breakdown (understanding vs. turns fallback)")
    lines.append("")
    for path, stats in path_breakdown.items():
        cm = stats["confusion_matrix"]
        lines.append(
            f"- **{path}:** n={stats['n_cases']}, accuracy={_fmt_pct(stats['accuracy'])}, "
            f"confusion tp={cm['tp']} fp={cm['fp']} tn={cm['tn']} fn={cm['fn']}"
        )
    lines.append("")

    # ── coercion flip rate ──
    lines.append("## Coercion-flip rate (R-1 live-incidence metric, §4.3)")
    lines.append("")
    lines.append(
        f"- **overall:** {_fmt_pct(coercion_overall['rate'])} "
        f"({coercion_overall['flips']}/{coercion_overall['n_calls']} calls)"
    )
    lines.append(
        f"- **r1_probe cases only:** {_fmt_pct(coercion_r1['rate'])} "
        f"({coercion_r1['flips']}/{coercion_r1['n_calls']} calls)"
    )
    lines.append(
        "- A non-zero rate on `r1_probe` cases is a defect report against "
        "`_NEGATION_CUES`, not a judge failure (archived §4.3/§7)."
    )
    lines.append("")

    # ── flip rate ──
    lines.append("## Flip rate (replicate instability, §5.2)")
    lines.append("")
    lines.append(
        f"- **{_fmt_pct(flips['rate'])}** of cases ({flips['unstable']}/{flips['n_cases']}) "
        f"had disagreeing replicates: {', '.join(flips['unstable_ids']) or '(none)'}"
    )
    lines.append("")

    # ── materiality probe ──
    lines.append("## Materiality-probe check (ca-04/05/08 vs. cs-04)")
    lines.append("")
    if materiality_failed:
        lines.append(
            "**FAILED as a bloc** — the judge suspended all of ca-04/ca-05/ca-08 AND "
            "advanced cs-04. This is a blocker-grade finding per archived §7 "
            "regardless of G1/G2: the judge appears to be pattern-matching "
            "`len(missing)` rather than assessing research sufficiency."
        )
    else:
        lines.append(
            "Passed — the judge did not exhibit the `len(missing)` pattern-match "
            "failure mode across this probe set."
        )
    lines.append("")

    # ── tn-05 in isolation ──
    lines.append("## tn-05 in isolation")
    lines.append("")
    if tn05 is not None:
        maj = sum(1 for r in tn05.replicates if r.final_decision)
        lines.append(
            f"assistant's clarifying question is the last (unanswered) turn — "
            f"{maj}/{len(tn05.replicates)} replicates advanced. An advance here is a "
            "qualitatively distinct fallback failure (the judge crediting its own "
            "question as answered)."
        )
        for i, r in enumerate(tn05.replicates, start=1):
            lines.append(f"  - replicate {i}: final={r.final_decision} — {r.final_rationale!r}")
    else:
        lines.append("tn-05 not found in this run's case set.")
    lines.append("")

    # ── full per-case table ──
    lines.append("## Full per-case table")
    lines.append("")
    lines.append(
        "| id | tier | path | expected | raw decisions | final decisions | rationales |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for c in cases:
        raw = ", ".join(str(r.raw_decision) for r in c.replicates)
        final = ", ".join(str(r.final_decision) for r in c.replicates)
        rationales = " / ".join(r.final_rationale.replace("|", "/") for r in c.replicates)
        lines.append(
            f"| {c.id} | {c.tier} | {c.path} | {c.expected} | {raw} | {final} | {rationales} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"
