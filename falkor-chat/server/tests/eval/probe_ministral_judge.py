"""THROWAWAY PROBE — K-027 item 5 (Ministral re-probe, D13 finding 2), U3.

Not a pytest test module (no `test_*` name — never collected). Run directly:

    cd server && .venv/bin/python tests/eval/probe_ministral_judge.py

Drives the exact, already-reviewed `guard_calibration.py` plumbing
(`test_guard_calibration_live.py`'s own methodology: real `guards.evaluate_guard`
calls, the full 26-row `golden_guards.jsonl`, k=3 replicates, the same G1/G2/kappa
metric functions) against `mistralai/ministral-3-3b` instead of Qwen — via the
workspace-override hard cap (`run["modelOverrides"]["guardModel"]`), never by
editing `config/models.json`. `guard_calibration.build_call` is monkeypatched
(module-global rebind, not a file edit) to inject the override into the `run`
dict every `run_case` call already builds; every other function is used exactly
as `test_guard_calibration_live.py` uses it, unmodified.

Never touches `reference` or any real workspace — `ws:ministral-probe` is a label
carried inside the `run` dict passed to `guards.evaluate_guard`/`_LlmGuardJudge`,
never a graph key any code here selects or writes (confirmed by reading
`guard_calibration.build_call`/`_LlmGuardJudge.__call__` — `run["ws"]` only ever
reaches `ModelGateway.llm(..., ws=...)`, which reads it only through the
`NullWorkspaceOverrides` no-op port here, since `overrides=` is always non-None on
this path and wins first per `_workspace_override_ref`).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _EVAL_DIR.parents[1]
_REPO_ROOT = _SERVER_DIR.parent
sys.path.insert(0, str(_EVAL_DIR))  # `guard_calibration` lives here, imported bare
sys.path.insert(0, str(_SERVER_DIR))  # `falkorchat` package

import guard_calibration as gc  # noqa: E402

from falkorchat import app, modelconfig  # noqa: E402

MODEL_REF = "lmstudio/mistralai/ministral-3-3b"
K_REPLICATES = 3
_EXAMPLE_OPENCODE_PATH = _REPO_ROOT / "config" / "opencode.example.json"


def _opencode_path() -> str:
    return os.environ.get("FALKORCHAT_OPENCODE_CONFIG") or str(_EXAMPLE_OPENCODE_PATH)


def _fetch_quantization(base_url: str, model_id: str) -> str | None:
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


def main() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "unused-in-ministral-probe")
    opencode_path = _opencode_path()
    catalog = modelconfig.ProviderCatalog.load(opencode_path)
    overlay = modelconfig.Overlay.load(str(modelconfig.DEFAULT_MODEL_CONFIG_PATH))
    gateway = modelconfig.ModelGateway(catalog, overlay)

    override_run = {
        "ws": "ws:ministral-probe",
        "modelOverrides": {"guardModel": MODEL_REF},
    }

    # ── preflight: confirm the override resolves to Ministral and is reachable ──
    resolved_llm, resolution = gateway.resolve_llm(
        "guard", ws=override_run["ws"], overrides=override_run["modelOverrides"]
    )
    assert resolution.source == "workspace", (
        f"expected the workspace override to win, got source={resolution.source!r}"
    )
    resolved = resolution.primary
    assert resolved.ref == MODEL_REF, f"resolved to {resolved.ref!r}, not {MODEL_REF!r}"
    reply = resolved_llm.complete(
        [{"role": "user", "content": "Reply with the single word: ok"}]
    )
    print(f"preflight: resolved={resolved.ref!r} baseURL={resolved.base_url!r} "
          f"params={resolved.params!r} reply={reply!r}")

    quantization = _fetch_quantization(resolved.base_url, resolved.model) or "(unknown)"
    print(f"quantization (live-verified): {quantization}")

    # ── monkeypatch build_call so run_case's own internal call picks up the override ──
    _orig_build_call = gc.build_call

    def _ministral_build_call(row):
        call = _orig_build_call(row)
        call["run"] = override_run
        return call

    gc.build_call = _ministral_build_call

    judge = app._build_llm_judge(gateway)

    rows = gc.load_golden_guards()
    assert len(rows) == 26, f"golden_guards.jsonl has {len(rows)} rows, expected 26"

    print(f"running {len(rows)} cases x k={K_REPLICATES} replicates "
          f"({len(rows) * K_REPLICATES} real judge calls) against {MODEL_REF!r} ...")
    cases = []
    for i, row in enumerate(rows, start=1):
        case = gc.run_case(row, judge, k=K_REPLICATES)
        cases.append(case)
        print(f"  [{i}/{len(rows)}] {row['id']} ({row['tier']}/{row['path']}) -> "
              f"final={[r.final_decision for r in case.replicates]}")

    cases_by_id = {c.id: c for c in cases}
    clear_cases = [c for c in cases if c.tier in ("clear_advance", "clear_suspend")]
    boundary_cases = [c for c in cases if c.tier == "boundary"]
    assert len(clear_cases) == 21 and len(boundary_cases) == 5

    g1 = gc.false_advance_rate(cases)
    g1_all = gc.false_advance_rate_all(cases)
    g2 = gc.advance_recall(cases)
    kappa = gc.cohens_kappa(clear_cases)
    boundary_confusion = gc.confusion_matrix(boundary_cases)
    coercion_overall = gc.coercion_flip_rate(cases)
    coercion_r1 = gc.coercion_flip_rate(cases, r1_probe_only=True)
    flips = gc.flip_rate(cases)
    path_breakdown = gc.per_path_breakdown(clear_cases)
    materiality_failed = gc.materiality_probe_failed(cases_by_id)

    g1_pass = g1["rate"] <= 0.10
    g2_pass = g2["rate"] >= 0.80
    verdict = "wire" if (g1_pass and g2_pass) else "block"

    print("\n=== RESULTS (Ministral 3B judge, golden_guards.jsonl, k=3) ===")
    print(f"model ref: {resolved.ref}")
    print(f"quantization: {quantization}")
    print(f"temperature: {resolved.params.get('temperature')!r}")
    print(f"base_url: {resolved.base_url}")
    print(f"G1 false-advance = {g1['rate']:.1%} (n={g1['n_cases']} cases / {g1['n_calls']} calls)")
    print(f"G1_all FAR = {g1_all['rate']:.1%} ({g1_all['advances']}/{g1_all['n_calls']} calls)")
    print(f"G2 advance-recall = {g2['rate']:.1%} (n={g2['n_cases']} cases, {g2['advances']} advanced)")
    print(f"VERDICT: {verdict}  (gates: G1<=10% -> {g1_pass}, G2>=80% -> {g2_pass})")
    print(f"kappa = {kappa['kappa']:.3f} (po={kappa['po']:.3f}, pe={kappa['pe']:.3f}, "
          f"prevalence={kappa['prevalence']:.3f}, judge-positive-rate={kappa['rater_b_positive_rate']:.3f})")
    print(f"boundary conservatism (share suspended) = "
          f"{1 - boundary_confusion['fp']/len(boundary_cases):.1%} "
          f"(confusion tp={boundary_confusion['tp']} fp={boundary_confusion['fp']} "
          f"tn={boundary_confusion['tn']} fn={boundary_confusion['fn']})")
    for path, stats in path_breakdown.items():
        cm = stats["confusion_matrix"]
        print(f"  path={path}: n={stats['n_cases']} accuracy={stats['accuracy']:.1%} "
              f"tp={cm['tp']} fp={cm['fp']} tn={cm['tn']} fn={cm['fn']}")
    print(f"coercion-flip overall = {coercion_overall['rate']:.1%} "
          f"({coercion_overall['flips']}/{coercion_overall['n_calls']})")
    print(f"coercion-flip r1_probe-only = {coercion_r1['rate']:.1%} "
          f"({coercion_r1['flips']}/{coercion_r1['n_calls']})")
    print(f"flip-rate (replicate instability) = {flips['rate']:.1%} "
          f"({flips['unstable']}/{flips['n_cases']}) ids={flips['unstable_ids']}")
    print(f"materiality-probe bloc failed = {materiality_failed}")

    print("\n=== per-case raw table ===")
    for c in cases:
        raw = [r.raw_decision for r in c.replicates]
        final = [r.final_decision for r in c.replicates]
        print(f"{c.id}\t{c.tier}\t{c.path}\texpected={c.expected}\traw={raw}\tfinal={final}")
        for r in c.replicates:
            print(f"    rationale: {r.final_rationale!r}")


if __name__ == "__main__":
    main()
