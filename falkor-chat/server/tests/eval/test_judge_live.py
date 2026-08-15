"""Live judge-layer test (K-026, `docs/plans/graphrag-eval.md` §5 Unit 3): the
generation sub-pass + the calibration sub-pass against a REAL LM Studio, writing
`judge_calibration.json` on a completed run.

**Marker-gated**, mirroring `test_workflow_live.py`: `pytest.mark.live`, DESELECTED by
default (`addopts = -m "not live"`), so the standard network-free `pytest -q` baseline
is unaffected. Run explicitly:

    cd server && .venv/bin/python -m pytest -m live -s tests/eval/test_judge_live.py

Needs: FalkorDB up with a seeded `ws:eval` (`./scripts/seed_eval_corpus.sh`) **and**
LM Studio serving the agent-under-test model, the judge model, and an embedding model
at `FALKORCHAT_LIVE_LLM_BASE_URL` (default `http://localhost:1234/v1`). Any of these
being unreachable **skips** (never fails) with a reason naming which dependency.

**Read-only against `ws:eval` (D2).** The generation sub-pass calls
`Services.hybrid_search` only — never a write — so a live run here never mutates the
corpus the retrieval baseline (Unit 2b) depends on. The test asserts this itself: the
`Message` count is captured before and after and must be identical.

**Model config, D7 mechanism 2.** This is a pytest test under `server/tests/`, so it
inherits `conftest.py`'s autouse `_model_config_env` redirection like every other test
in the suite — both LLM clients below are therefore constructed directly from
env-var literals, never via `ModelGateway`/`StaticModelGateway`, which would silently
resolve against `server/tests/data/models.json`'s dim-4 test fixture instead of the
real M2 stack.

**Judge model default, D1 (v4).** `FALKORCHAT_LIVE_JUDGE_MODEL` defaults to
`qwen/qwen3-4b-2507` — the SAME ref as `FALKORCHAT_LIVE_LLM_MODEL` — a deliberate,
stakeholder-directed, hardware-constrained deviation from the method note's "never the
model judging itself" guidance (`docs/plans/graphrag-eval.md` §3 D1,
`docs/reviews/graphrag-eval-ml.md`). This is *not* a new reachability dependency: since
both refs default to the same model, `live_models` below probes each **distinct**
resolved ref exactly once — one loaded model serving two roles, not two live
dependencies (plan §6). An operator who overrides `FALKORCHAT_LIVE_JUDGE_MODEL` back to
a distinct, stronger model still gets an independent probe/skip for it.
`sameModelAsAgentUnderTest` in the written output is derived from comparing the two
resolved refs at run time, never hardcoded — the machine-readable carrier
`generate_report.py` gates its self-preference-bias caveat on.

**Reuses `AgentResponder._build_prompt`** via a throwaway, side-effect-free instance
(`services=None`) rather than duplicating the prompt-construction logic — this already
embeds the module-level `_SYSTEM_PROMPT` constant, so drift-with-production is
structurally impossible (plan §2's corrected finding).

**Non-gating (D1).** This test asserts only that the harness itself worked correctly
(sample sizes, the output file was written, the empty-context faithfulness invariant,
`ws:eval` unmutated) — never a pass/fail threshold on the judge's own agreement or
faithfulness/relevance numbers. Those numbers are descriptive/directional, reported by
`generate_report.py`, and require a `data-scientist` sign-off before being trusted
(D1/D6-mirrored gate) — this test does not, and must not, encode that judgment.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from falkorchat import config, db
from falkorchat.config import CallContext
from falkorchat.embedding import OpenAICompatibleEmbedder
from falkorchat.llm import OpenAICompatibleLLM
from falkorchat.repository import Repository
from falkorchat.responder import AgentResponder
from falkorchat.services import Services

from judge import judge_triple

pytestmark = pytest.mark.live

# Live-test-only knobs (K-042: model choice is not a product env var, FR-20) — same
# pattern as `test_workflow_live.py`, plus this plan's own `FALKORCHAT_LIVE_JUDGE_MODEL`
# (§3 D1). All optional; defaults match the M2 stack / v4's same-model judge default.
LIVE_LLM_BASE_URL = os.environ.get("FALKORCHAT_LIVE_LLM_BASE_URL", "http://localhost:1234/v1")
LIVE_LLM_MODEL = os.environ.get("FALKORCHAT_LIVE_LLM_MODEL", "qwen/qwen3-4b-2507")
LIVE_JUDGE_MODEL = os.environ.get("FALKORCHAT_LIVE_JUDGE_MODEL", "qwen/qwen3-4b-2507")
LIVE_EMBEDDING_MODEL = os.environ.get(
    "FALKORCHAT_LIVE_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b"
)

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN_RETRIEVAL_PATH = _EVAL_DIR / "golden_retrieval.jsonl"
_GOLDEN_CALIBRATION_PATH = _EVAL_DIR / "golden_judge_calibration.jsonl"
_OUTPUT_PATH = _EVAL_DIR / "judge_calibration.json"

# Plan §5 Unit 3: "~15-20 items sampled from golden_retrieval.jsonl (a fixed subset,
# e.g. the first 20 by id, so the sample is stable across runs)".
GENERATION_SAMPLE_SIZE = 20


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _message_count(ws: str) -> int:
    graph = db.workspace_graph(db.connect(), ws)
    res = graph.ro_query("MATCH (m:Message) RETURN count(m)")
    return res.result_set[0][0]


# ── live-dependency gating ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_models(ws_eval: str) -> str:
    """Skip (never fail) unless `ws:eval` is seeded (reused `ws_eval` probe fixture
    from `conftest.py`, shared with Unit 2b) AND every DISTINCT model ref this run
    needs is loadable. Returns the workspace id.

    Probing is deduped by resolved ref (`{LIVE_LLM_MODEL, LIVE_JUDGE_MODEL}` is a set)
    — as of v4 both default to the same ref, so this issues exactly one probe call per
    role in the common case, not two independent live dependencies (plan §6).
    """
    try:
        OpenAICompatibleEmbedder(LIVE_LLM_BASE_URL, LIVE_EMBEDDING_MODEL).embed("probe")
    except Exception as exc:
        pytest.skip(
            f"embedding model unreachable at {LIVE_LLM_BASE_URL} ({exc!r}) — "
            f"start LM Studio and load {LIVE_EMBEDDING_MODEL!r}"
        )

    for model in {LIVE_LLM_MODEL, LIVE_JUDGE_MODEL}:
        try:
            OpenAICompatibleLLM(LIVE_LLM_BASE_URL, model).complete(
                [{"role": "user", "content": "Reply with the single word: ok"}]
            )
        except Exception as exc:
            pytest.skip(
                f"model {model!r} unreachable at {LIVE_LLM_BASE_URL} ({exc!r}) — "
                f"start LM Studio and load it"
            )

    return ws_eval


# ── the test ──────────────────────────────────────────────────────────────────


def test_judge_layer_generation_and_calibration_subpasses(live_models: str) -> None:
    """Runs both sub-passes (plan §5 Unit 3) and writes `judge_calibration.json`.

    Generation sub-pass: for each of the first `GENERATION_SAMPLE_SIZE` golden-
    retrieval queries (by id), embed → `Services.hybrid_search` (read-only) →
    build the real production prompt via `AgentResponder._build_prompt` → the
    agent-under-test model answers → the judge model scores faithfulness/relevance
    of the live-generated triple.

    Calibration sub-pass: the 10 fixed `golden_judge_calibration.jsonl` triples,
    judged directly (no generation) against their human-authored expected labels;
    raw percent exact-agreement per axis (D4), including the null/None-vs-None case
    for a legitimately abstained expectation.
    """
    ws = live_models
    ctx = CallContext(ws=ws, actor="eval-harness")

    count_before = _message_count(ws)

    repo = Repository(db.connect())
    services = Services(repo)
    embedder = OpenAICompatibleEmbedder(LIVE_LLM_BASE_URL, LIVE_EMBEDDING_MODEL)
    agent_llm = OpenAICompatibleLLM(LIVE_LLM_BASE_URL, LIVE_LLM_MODEL)
    judge_llm = OpenAICompatibleLLM(LIVE_LLM_BASE_URL, LIVE_JUDGE_MODEL)
    # Side-effect-free (no I/O in __init__) — used only for its `_build_prompt`
    # helper, so production's prompt-construction logic is never duplicated here.
    responder = AgentResponder(services=None, agent_id=config.AGENT_ID)

    # ── generation sub-pass ──
    golden_rows = sorted(_load_jsonl(_GOLDEN_RETRIEVAL_PATH), key=lambda r: r["id"])
    sample = golden_rows[:GENERATION_SAMPLE_SIZE]

    generation_items: list[dict[str, Any]] = []
    for row in sample:
        q_vec = embedder.embed(row["query"])
        seeds = services.hybrid_search(ctx, q_vec=q_vec, k=10, channel_id=None)
        prompt = responder._build_prompt(row["query"], seeds)
        answer = agent_llm.complete(prompt)
        context_texts = [s["text"] for s in seeds]
        verdict = judge_triple(
            judge_llm, question=row["query"], context=context_texts, answer=answer
        )
        generation_items.append({
            "id": row["id"],
            "seedCount": len(seeds),
            "faithfulness": verdict.faithfulness,
            "relevance": verdict.relevance,
            "parseFailed": verdict.parse_failed,
        })

    # §6 edge case: an empty retrieved context must resolve faithfulness to None,
    # never scored against nothing — `judge.judge_triple` enforces this
    # unconditionally, this just confirms the invariant held for whatever this
    # live run actually retrieved (it may never trigger against a real corpus).
    for item in generation_items:
        if item["seedCount"] == 0:
            assert item["faithfulness"] is None, (
                f"{item['id']}: empty-context item must abstain (faithfulness=None), "
                f"got {item['faithfulness']!r}"
            )

    # ── calibration sub-pass ──
    calibration_rows = _load_jsonl(_GOLDEN_CALIBRATION_PATH)
    calibration_items: list[dict[str, Any]] = []
    for row in calibration_rows:
        verdict = judge_triple(
            judge_llm, question=row["question"], context=row["context"], answer=row["answer"]
        )
        calibration_items.append({
            "id": row["id"],
            "expectedFaithfulness": row["expected_faithfulness"],
            "expectedRelevance": row["expected_relevance"],
            "judgedFaithfulness": verdict.faithfulness,
            "judgedRelevance": verdict.relevance,
            "parseFailed": verdict.parse_failed,
            "faithfulnessAgree": verdict.faithfulness == row["expected_faithfulness"],
            "relevanceAgree": verdict.relevance == row["expected_relevance"],
        })

    n_cal = len(calibration_items)
    assert n_cal == len(calibration_rows)
    faithfulness_agreement = sum(i["faithfulnessAgree"] for i in calibration_items) / n_cal
    relevance_agreement = sum(i["relevanceAgree"] for i in calibration_items) / n_cal
    calibration_parse_failures = sum(i["parseFailed"] for i in calibration_items)

    n_gen = len(generation_items)
    assert n_gen == len(sample)
    generation_parse_failures = sum(i["parseFailed"] for i in generation_items)

    same_model = LIVE_JUDGE_MODEL == LIVE_LLM_MODEL  # D1: derived at run time, never hardcoded

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "agentUnderTestModel": LIVE_LLM_MODEL,
        "judgeModel": LIVE_JUDGE_MODEL,
        "sameModelAsAgentUnderTest": same_model,
        "calibration": {
            "sampleSize": n_cal,
            "faithfulnessAgreement": faithfulness_agreement,
            "relevanceAgreement": relevance_agreement,
            "parseFailures": calibration_parse_failures,
            "items": calibration_items,
        },
        "generation": {
            "sampleSize": n_gen,
            "faithfulTrue": sum(1 for i in generation_items if i["faithfulness"] is True),
            "faithfulFalse": sum(1 for i in generation_items if i["faithfulness"] is False),
            "faithfulAbstained": sum(1 for i in generation_items if i["faithfulness"] is None),
            "relevantTrue": sum(1 for i in generation_items if i["relevance"] is True),
            "relevantFalse": sum(1 for i in generation_items if i["relevance"] is False),
            "parseFailures": generation_parse_failures,
            "items": generation_items,
        },
    }

    # Write only on an actual completed run (never on skip — the fixture above
    # raises before this code is ever reached when a dependency is unreachable).
    _OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(  # -s to see this: a quick human-readable summary alongside the JSON file
        f"\njudge_calibration.json written: "
        f"calibration faithfulness agreement={faithfulness_agreement:.2f} "
        f"({n_cal - calibration_parse_failures}/{n_cal} parsed), "
        f"relevance agreement={relevance_agreement:.2f}; "
        f"generation n={n_gen}, parse failures={generation_parse_failures}, "
        f"sameModelAsAgentUnderTest={same_model}"
    )

    # ── D2: this sub-pass is read-only — ws:eval must be byte-identical after ──
    count_after = _message_count(ws)
    assert count_after == count_before, (
        f"ws:{ws} message count changed from {count_before} to {count_after} — "
        f"the judge live test must never write to the corpus"
    )
