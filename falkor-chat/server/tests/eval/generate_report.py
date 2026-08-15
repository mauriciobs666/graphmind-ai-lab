#!/usr/bin/env python3
"""Generate the GraphRAG eval metrics report (K-026, `docs/plans/graphrag-eval.md`
§5 Unit 3).

**NOT a pytest test** — manually/CI invoked after both suites have run:

    cd server && .venv/bin/python tests/eval/generate_report.py

Reads:
  - `retrieval_baseline.json` (Unit 2b, **required** — errors clearly and exits
    non-zero if absent, meaning Unit 2b was never run against a seeded `ws:eval`).
  - `judge_calibration.json` (Unit 3, **optional** — its absence is the unambiguous
    "not run" signal, per its own "written only on an actual completed live run"
    contract in `test_judge_live.py`; never fabricated here).
  - `corpus_provenance.json` (Unit 1) and `golden_retrieval.jsonl` (Unit 2a) for
    corpus/golden-set context.

Writes `docs/test-reports/graphrag-eval-<date>.md`, per the plan's exact §5 Unit 3
spec: recall@10/recall@5/MRR baseline numbers; judge-human agreement per axis or an
explicit not-run marker; corpus provenance; golden-set size and the
self-retrieval-guard pass/fail; the D4 small-N caveat verbatim when judge numbers are
present; and, when `judge_calibration.json`'s `sameModelAsAgentUnderTest` is `true`,
the mandatory same-model judge caveat (below) verbatim, adjacent to the judge numbers
— never a trailing footnote.

**Dropped sub-clause (teco decision, per the K-026 Unit 3 dispatch brief).** The
plan's literal text also asks for "whether the required `data-scientist` sign-off has
happened yet." That has no data source: `judge_calibration.json` is written
unconditionally on every completed live run (unlike D6's baseline, whose *committed
presence* is itself read as the sign-off signal) — its mere existence cannot be read
as "sign-off happened." This report does not attempt to compute that status; the
caveat's `[pending / not yet reviewed]` placeholder is emitted **literally**, and
actual sign-off tracking lives in the coordination ledger
(`docs/plans/graphrag-eval-coordination.md`), not in this generated file.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _EVAL_DIR.parents[1]
_REPO_ROOT = _SERVER_DIR.parent
_REPORTS_DIR = _REPO_ROOT / "docs" / "test-reports"

_RETRIEVAL_BASELINE_PATH = _EVAL_DIR / "retrieval_baseline.json"
_JUDGE_CALIBRATION_PATH = _EVAL_DIR / "judge_calibration.json"
_CORPUS_PROVENANCE_PATH = _EVAL_DIR / "corpus_provenance.json"
_GOLDEN_RETRIEVAL_PATH = _EVAL_DIR / "golden_retrieval.jsonl"

# `docs/reviews/graphrag-eval-ml.md` M-1's exact recommended language, emitted
# VERBATIM whenever judge_calibration.json's sameModelAsAgentUnderTest is true —
# never paraphrased, never in a trailing footnote (mirrors this codebase's own
# `m3-guard-calibration.md` report-honesty rule: "the caveat sentence, verbatim,
# adjacent to the verdict — not in a footnote").
_SAME_MODEL_CAVEAT_TEMPLATE = (
    "**Same-model judge limitation.** The judge model (`{judge_model}`) is identical "
    "to the agent-under-test model for this run.\n"
    "- *Calibration numbers* (judge-vs-human agreement on fixed, "
    "independently-authored triples) are largely unaffected by self-preference bias "
    "— the judge did not generate the content it's scoring. Read subject to the "
    "existing small-N caveat (D4), not this one.\n"
    "- *Generation-sub-pass faithfulness/relevance numbers* (the judge scoring its "
    "own model's live output) are structurally exposed to self-preference bias. A "
    "high score here is a same-model directional signal, **not independent "
    "validation**, and must not be read as if a distinct judge produced it. **A "
    "passing calibration number does not license trusting these — they are two "
    "different validity claims.**\n"
    "- Gross/obvious failures (flat contradiction of the retrieved context, "
    "answering the wrong question) likely remain catchable even here — the bias "
    "risk concentrates in borderline/subjective calls, where it becomes "
    "indistinguishable from genuine quality.\n"
    "- `data-scientist` sign-off status on these numbers: "
    "**[pending / not yet reviewed]**.\n"
)

# Plan §3 D4: "a mandatory small-N caveat in the generated report (mirroring
# m3-guard-calibration.md's own honesty section)".
_SMALL_N_CAVEAT = (
    "**Small-N caveat (D4).** Judge-vs-human agreement is raw percent exact-agreement "
    "over a ~10-example calibration set, not Cohen's kappa — `m3-guard-calibration.md` "
    "(this codebase's prior same-model-judge precedent) found kappa badly behaved even "
    "at N~21-26, and a ~10-example set only makes that worse. At N~10, a single "
    "disagreement swings the reported number by 10 points: read it as a directional "
    "signal, not a statistically defensible claim against the method note's ~0.7 "
    "threshold."
)


class ReportError(Exception):
    """A required input is missing or unreadable — surfaced as a clear CLI error."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_retrieval_baseline() -> dict[str, Any]:
    if not _RETRIEVAL_BASELINE_PATH.exists():
        raise ReportError(
            f"{_RETRIEVAL_BASELINE_PATH} does not exist — Unit 2b "
            f"(`pytest tests/eval/test_retrieval_eval.py`) has never been run "
            f"against a seeded ws:eval. Run ./scripts/seed_eval_corpus.sh, then "
            f"the retrieval-eval test suite, before generating a report."
        )
    return _load_json(_RETRIEVAL_BASELINE_PATH)


def _load_judge_calibration() -> dict[str, Any] | None:
    """Optional: absence means "not run (live LLM unreachable at last suite
    execution)" — never fabricated, never treated as an error."""
    if not _JUDGE_CALIBRATION_PATH.exists():
        return None
    return _load_json(_JUDGE_CALIBRATION_PATH)


def _load_corpus_provenance() -> dict[str, Any] | None:
    if not _CORPUS_PROVENANCE_PATH.exists():
        return None
    return _load_json(_CORPUS_PROVENANCE_PATH)


def _self_retrieval_guard_failures(rows: list[dict[str, Any]]) -> list[str]:
    """Re-derive the self-retrieval-inflation guard (method note finding 5/risk 2),
    the same pure string comparison `test_golden_set_integrity.py`'s
    `test_query_is_not_verbatim_self_retrieval` makes against each row's own
    `target_text` — reimplemented here (not imported) because that module's checks
    are pytest test functions, not a library API, and the check itself is a two-line
    comparison not worth extracting into shared production code for one caller.
    Returns the ids of any row that fails either direction.
    """
    failures: list[str] = []
    for row in rows:
        query = row.get("query", "").strip().lower()
        target_text = row.get("target_text", "").strip().lower()
        if query and target_text and (query in target_text or target_text in query):
            failures.append(row["id"])
    return failures


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_retrieval_section(baseline: dict[str, Any]) -> str:
    lines = [
        "## Retrieval baseline (Unit 2b)",
        "",
        f"- **recall@10:** {_fmt_pct(baseline['recall_at_10'])}",
        f"- **recall@5:** {_fmt_pct(baseline['recall_at_5'])}",
        f"- **MRR:** {baseline['mrr']:.4f}",
        f"- **n (golden pairs measured):** {baseline['n']}",
        "",
        "These numbers are the vector-only @1024 baseline over `Services.hybrid_search`"
        " against `ws:eval` (`docs/plans/graphrag-eval.md` §3 D6). Per D6, the first"
        " committed `retrieval_baseline.json` requires an explicit `data-scientist`"
        " methodology sign-off (\"is this a reasonable floor\") before it is treated as"
        " gating for future runs — the test having passed is not, by itself, sufficient.",
    ]
    return "\n".join(lines)


def _render_judge_section(judge: dict[str, Any] | None) -> str:
    if judge is None:
        return (
            "## Judge layer (Unit 3)\n\n"
            "**Not run** — `judge_calibration.json` does not exist, meaning the live"
            " judge test (`pytest -m live tests/eval/test_judge_live.py`) either has"
            " never been run, or its last run was skipped because a live dependency"
            " (FalkorDB, `ws:eval`, LM Studio) was unreachable. No judge-human"
            " agreement numbers are available for this report."
        )

    calibration = judge["calibration"]
    generation = judge["generation"]
    same_model = bool(judge.get("sameModelAsAgentUnderTest"))

    lines = [
        "## Judge layer (Unit 3)",
        "",
        f"- **judge model:** `{judge['judgeModel']}`",
        f"- **agent-under-test model:** `{judge['agentUnderTestModel']}`",
        f"- **run at:** {judge.get('generatedAt', '(unknown)')}",
        "",
        "### Calibration sub-pass",
        "",
        f"- **sample size:** {calibration['sampleSize']} fixed"
        " (`golden_judge_calibration.jsonl`) triples",
        f"- **faithfulness agreement:** {_fmt_pct(calibration['faithfulnessAgreement'])}",
        f"- **relevance agreement:** {_fmt_pct(calibration['relevanceAgreement'])}",
        f"- **parse failures:** {calibration['parseFailures']} / {calibration['sampleSize']}"
        " (a parse failure resolves conservatively — faithfulness=None,"
        " relevance=False — and is counted here, never silently dropped from the"
        " denominator)",
        "",
        _SMALL_N_CAVEAT,
        "",
        "### Generation sub-pass",
        "",
        f"- **sample size:** {generation['sampleSize']} live-generated items"
        " (sampled from `golden_retrieval.jsonl`)",
        f"- **faithful:** {generation['faithfulTrue']} true /"
        f" {generation['faithfulFalse']} false /"
        f" {generation['faithfulAbstained']} abstained (no retrieved context)",
        f"- **relevant:** {generation['relevantTrue']} true /"
        f" {generation['relevantFalse']} false",
        f"- **parse failures:** {generation['parseFailures']} / {generation['sampleSize']}",
        "",
    ]

    if same_model:
        lines.append(_SAME_MODEL_CAVEAT_TEMPLATE.format(judge_model=judge["judgeModel"]))
    else:
        lines.append(
            f"Judge model (`{judge['judgeModel']}`) differs from the agent-under-test"
            f" model (`{judge['agentUnderTestModel']}`) for this run — the same-model"
            " self-preference-bias caveat does not apply."
        )

    return "\n".join(lines)


def _render_corpus_section(
    provenance: dict[str, Any] | None, golden_rows: list[dict[str, Any]]
) -> str:
    guard_failures = _self_retrieval_guard_failures(golden_rows)
    guard_status = (
        "**PASS** — no golden query is a verbatim substring of (or superstring"
        " containing) its own target message text"
        if not guard_failures
        else f"**FAIL** — {len(guard_failures)} row(s) failed:"
        f" {', '.join(guard_failures)}"
    )

    lines = ["## Corpus & golden set", ""]
    if provenance is not None:
        lines += [
            f"- **corpus:** {provenance.get('message_count', '?')} messages across"
            f" {provenance.get('thread_count', '?')} threads (`ws:eval`)",
            f"- **embedding model:** `{provenance.get('embedding_model_ref', '?')}`"
            f" @ dim {provenance.get('embedding_dim', '?')}",
            f"- **seeded at:** {provenance.get('seeded_at', '?')}",
        ]
    else:
        lines.append(
            "- **corpus provenance:** not available"
            " (`server/tests/eval/corpus_provenance.json` is missing — run"
            " `./scripts/seed_eval_corpus.sh`)"
        )
    lines += [
        f"- **golden-retrieval set size:** {len(golden_rows)} pairs"
        " (`golden_retrieval.jsonl`)",
        f"- **self-retrieval-inflation guard:** {guard_status}",
    ]
    return "\n".join(lines)


def build_report(*, today: date | None = None) -> tuple[str, str]:
    """Assemble the report's markdown content. Returns `(filename, content)`.

    Kept separate from `main()`'s I/O so this is unit-testable without touching the
    filesystem beyond the read-only loads above.
    """
    today = today or date.today()
    baseline = _load_retrieval_baseline()
    judge = _load_judge_calibration()
    provenance = _load_corpus_provenance()
    golden_rows = _load_jsonl(_GOLDEN_RETRIEVAL_PATH) if _GOLDEN_RETRIEVAL_PATH.exists() else []

    generated_at = datetime.now(timezone.utc).isoformat()
    sections = [
        "# GraphRAG retrieval + generation evaluation report",
        "",
        "> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-026 (M2.5-quality)",
        "",
        f"Generated {generated_at} by `server/tests/eval/generate_report.py`"
        " (K-026, `docs/plans/graphrag-eval.md` §5 Unit 3). Machine-generated —"
        " re-run the script to refresh rather than hand-editing this file.",
        "",
        _render_retrieval_section(baseline),
        "",
        _render_judge_section(judge),
        "",
        _render_corpus_section(provenance, golden_rows),
        "",
    ]
    content = "\n".join(sections)
    filename = f"graphrag-eval-{today.isoformat()}.md"
    return filename, content


def main() -> int:
    try:
        filename, content = build_report()
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / filename
    out_path.write_text(content, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
