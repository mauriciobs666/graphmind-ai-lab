#!/usr/bin/env python3
"""Run the K-055 (M6) NL-query-generation golden-set evaluation (unit U29c,
`docs/plans/workflow-nl-query-generation-ml.md` §3-5) — the FR-4/AC-4 Layer 1
execution-accuracy gate, plus a secondary Layer 2 rendered-answer sanity check.

**Why a bare script, not a `pytest.mark.live` test module.** `server/tests/
conftest.py`'s autouse `_model_config_env` fixture redirects
`FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` to the offline dim-4
`tests/data/*.json` fixtures for EVERY test under `server/tests/`, including a
live-marked one (`test_golden_set_integrity.py`'s own docstring documents this
exact redirection; `test_guard_calibration_live.py` deliberately bypasses the
whole config layer for the same reason). This script needs the REAL
`config/opencode.json` (or `config/opencode.example.json`) and
`config/models.json` `ModelGateway.from_env()` resolves in production — the
same convention `scripts/seed_eval_corpus.py` already uses for its own real
embedding-model resolution — so it runs as a standalone script, never
collected by pytest, sidestepping the redirection question entirely rather
than working around it.

Needs: FalkorDB up (`./scripts/start_falkordb.sh -d`), `reference` seeded with
the product catalog (`EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh acme &&
./scripts/seed_catalog.sh acme` — **a default offline `pytest -q` run wipes
`reference` at teardown, AGENTS.md; re-seed before running this if you've run
the suite since**), `ws:nlq-eval` seeded (`./scripts/seed_nlq_eval_corpus.sh`,
unit U29's own corpus), and LM Studio serving a chat model at whatever
`FALKORCHAT_OPENCODE_CONFIG` resolves to (defaults to
`config/opencode.example.json`'s `lmstudio` provider, `localhost:1234/v1`, if
`FALKORCHAT_OPENCODE_CONFIG` is not already set in the environment).

Run:

    cd server && .venv/bin/python tests/eval/run_nlq_golden_set_eval.py

**Deliberately sequential, one golden pair at a time — never concurrent.**
Live-verified fact from this lab's own session history (`kaizen_team`,
confirmed independently twice): this LM Studio instance JIT-loads one model at
a time and thrashes badly (`ProviderCallError` / stuck 'processing') under
concurrent requests. `scripts/seed_nlq_eval_corpus.py`'s own module docstring
documents the identical finding for its embed/extract pipeline. 39 pairs, up
to two live LLM calls each (the tool's own internal structured-completion call
plus this script's Layer 2 render call), takes real wall-clock time — expected,
not a bug to work around.

Writes `tests/eval/nlq_eval_results.json` (every pair's raw tool result,
Layer 1/Layer 2 verdicts, and diagnostics — the input the report is generated
from) and prints an accuracy summary. Does **not** itself write the
`docs/test-reports/` markdown report — that's a deliberate, separate step
(inspect the JSON, then write the report) so a partial/exploratory run never
overwrites the last complete one's narrative report by accident.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _EVAL_DIR.parents[1]
_REPO_ROOT = _SERVER_DIR.parent

sys.path.insert(0, str(_SERVER_DIR))
sys.path.insert(0, str(_EVAL_DIR))  # `nlq_scoring` is a bare sibling import

# **Must run BEFORE any `falkorchat` import.** `falkorchat.config` resolves
# `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` into module-level
# constants once at import time (FR-15, "read once, no reload path") — the
# same footgun `tests/conftest.py`'s `_model_config_env` fixture docstring
# documents ("a bare setenv alone would never reach `ModelGateway.from_env()`").
# Setting the env var after `falkorchat` is already imported would be a
# silent no-op, so this call is placed ahead of every `falkorchat.*` import
# below rather than inside `main()`.
os.environ.setdefault("OPENAI_API_KEY", "unused-in-nlq-golden-set-eval")
os.environ.setdefault(
    "FALKORCHAT_OPENCODE_CONFIG", str(_REPO_ROOT / "config" / "opencode.example.json")
)

from falkorchat import db, modelconfig  # noqa: E402
from falkorchat.config import CallContext  # noqa: E402
from falkorchat.repository import Repository  # noqa: E402
from falkorchat.services import Services  # noqa: E402
from falkorchat.tools import QueryGraphDataTool  # noqa: E402

from nlq_scoring import layer2_contains, load_golden_set, score_pair, wilson_interval  # noqa: E402

_GOLDEN_PATH = _EVAL_DIR / "nlq_golden_set.jsonl"
_RESULTS_PATH = _EVAL_DIR / "nlq_eval_results.json"

# Catalog pairs' `ctx.ws` is never actually read (querygen.CATALOG_SCHEMA has a
# fixed graph_key="reference") but CallContext still needs a valid-looking
# value — any already-bootstrapped workspace works (brief's own call).
CATALOG_WS = os.environ.get("NLQ_CATALOG_WS", "acme")
KB_WS = os.environ.get("NLQ_EVAL_WS", "nlq-eval")
ACTOR_ID = "nlq-eval-harness"

# Layer 2's render call: one extra internal LLM call per pair, turning the
# tool's own raw JSON result into a short sentence — mirrors the internal-call
# pattern `QueryGraphDataTool.run()` itself already uses (`llm.complete([...])`
# with a system + user message), rather than driving a full multi-turn
# `salesperson@v4` conversation through the executor (that heavier live e2e
# proof is a separate, later `qa-engineer` unit's job per this unit's brief).
_RENDER_SYSTEM_PROMPT = (
    "You answer a user's question in ONE short sentence, using ONLY the data "
    "in the JSON result below - never invent anything not present in it. If "
    "the JSON's \"items\" list is empty, say plainly that the information "
    "was not found. Always use the exact numeric digits from the JSON "
    "verbatim - never spell a number out as a word.\n\nJSON result:\n{result_json}"
)


def _ctx_for(dataset: str) -> CallContext:
    ws = KB_WS if dataset == "knowledge_base" else CATALOG_WS
    return CallContext(ws=ws, actor=ACTOR_ID)


def _falkordb_reachable() -> bool:
    try:
        db.connect().select_graph(f"ws:{KB_WS}").query("RETURN 1")
        return True
    except Exception:
        return False


def _check_reference_seeded() -> None:
    graph = db.reference_graph(db.connect())
    res = graph.ro_query("MATCH (p:Product) RETURN count(p) AS n")
    n = res.result_set[0][0]
    if n == 0:
        raise SystemExit(
            "ERROR: `reference` graph has 0 Product nodes - the catalog "
            "dataset needs it seeded (a default offline `pytest -q` run wipes "
            "`reference` at teardown, AGENTS.md). Run:\n"
            "  EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh acme && "
            "./scripts/seed_catalog.sh acme"
        )


def _check_kb_seeded() -> None:
    graph = db.workspace_graph(db.connect(), KB_WS)
    res = graph.ro_query("MATCH (e:Entity) RETURN count(e) AS n")
    n = res.result_set[0][0]
    if n == 0:
        raise SystemExit(
            f"ERROR: ws:{KB_WS} has 0 Entity nodes - run "
            f"./scripts/seed_nlq_eval_corpus.sh first (unit U29's own corpus)."
        )


def _render_answer(gateway: Any, ctx: CallContext, question: str, tool_result: dict) -> str:
    llm = gateway.llm("step", ws=ctx.ws)
    prompt = _RENDER_SYSTEM_PROMPT.format(result_json=json.dumps(tool_result))
    return llm.complete([
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ])


def _run_one(tool: QueryGraphDataTool, gateway: Any, row: dict[str, Any]) -> dict[str, Any]:
    ctx = _ctx_for(row["dataset"])
    raw = tool.run({"question": row["question"], "dataset": row["dataset"]}, ctx=ctx, run={})
    tool_result = json.loads(raw)
    layer1 = score_pair(row, tool_result)

    rendered = _render_answer(gateway, ctx, row["question"], tool_result)
    layer2_ok = layer2_contains(row, rendered)

    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "shape": row["shape"],
        "question": row["question"],
        "expected": row["expected"],
        "toolResult": tool_result,
        "layer1Correct": layer1.correct,
        "layer1Reason": layer1.reason,
        "layer1Extracted": layer1.extracted,
        "renderedAnswer": rendered,
        "layer2Correct": layer2_ok,
    }


def _write_results(records: list[dict[str, Any]], model_ref: str) -> None:
    payload = {
        "runAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model_ref,
        "records": records,
    }
    _RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nResults written: {_RESULTS_PATH.relative_to(_REPO_ROOT)}")


def _print_summary(records: list[dict[str, Any]]) -> None:
    n = len(records)
    correct = sum(1 for r in records if r["layer1Correct"])
    lo, hi = wilson_interval(correct, n)
    print(
        f"\nOverall Layer 1 accuracy: {correct}/{n} = {correct / n:.1%}  "
        f"(95% CI [{lo:.1%}, {hi:.1%}])"
    )

    kb = [r for r in records if r["dataset"] == "knowledge_base"]
    if kb:
        kb_correct = sum(1 for r in kb if r["layer1Correct"])
        klo, khi = wilson_interval(kb_correct, len(kb))
        print(
            f"knowledge_base subset: {kb_correct}/{len(kb)} = "
            f"{kb_correct / len(kb):.1%}  (95% CI [{klo:.1%}, {khi:.1%}])"
        )

    not_found = [r for r in records if r["shape"] == "not-found"]
    if not_found:
        false_answers = sum(1 for r in not_found if not r["layer1Correct"])
        print(
            f"not-found false-answer rate: {false_answers}/{len(not_found)} = "
            f"{false_answers / len(not_found):.1%}"
        )

    print("\nPer-shape breakdown:")
    shapes = sorted({r["shape"] for r in records})
    for shape in shapes:
        subset = [r for r in records if r["shape"] == shape]
        c = sum(1 for r in subset if r["layer1Correct"])
        print(f"  {shape:>22}: {c}/{len(subset)} = {c / len(subset):.1%}")


def main() -> None:
    if not _falkordb_reachable():
        raise SystemExit(
            "ERROR: FalkorDB not reachable - start it with "
            "./scripts/start_falkordb.sh -d"
        )

    print("Resolving chat model via ModelGateway.from_env()...")
    gateway = modelconfig.ModelGateway.from_env()
    resolution = gateway.resolve("step")
    model_ref = resolution.primary.ref
    print(f"  step model: {model_ref}")

    _check_reference_seeded()
    _check_kb_seeded()

    conn = db.connect()
    repo = Repository(conn)
    services = Services(repo)
    tool = QueryGraphDataTool(services, models=gateway)

    rows = load_golden_set(_GOLDEN_PATH)
    print(
        f"Scoring {len(rows)} golden pairs sequentially "
        f"(this takes real wall-clock time - never concurrent, see module "
        f"docstring)..."
    )

    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        t0 = time.monotonic()
        record = _run_one(tool, gateway, row)
        elapsed = time.monotonic() - t0
        l1 = "OK  " if record["layer1Correct"] else "MISS"
        l2 = "OK  " if record["layer2Correct"] else "MISS"
        print(
            f"  [{i:2d}/{len(rows)}] {row['id']:>8} "
            f"({row['shape']:>22}) L1={l1} L2={l2} ({elapsed:5.1f}s)"
        )
        records.append(record)

    _write_results(records, model_ref)
    _print_summary(records)


if __name__ == "__main__":
    main()
