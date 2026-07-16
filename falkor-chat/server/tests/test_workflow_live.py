"""Live end-to-end triage-workflow test (M3, K-022 — U14): AC-1…AC-4.

**Marker-gated.** This is the only test in the suite that talks to a REAL LM Studio.
It is `@pytest.mark.live` and DESELECTED by default (`addopts = -m "not live"` in
`pyproject.toml`), so the standard network-free `pytest` baseline is unaffected even
when LM Studio happens to be running. Run it explicitly:

    cd server && .venv/bin/python -m pytest -m live -s

Needs: FalkorDB up (`./scripts/start_falkordb.sh -d`) **and** LM Studio serving a chat
model + an embedding model at `FALKORCHAT_LLM_BASE_URL` (default `localhost:1234`).
Either being unreachable skips (never fails) with a reason.

**Why its own `ws:live` workspace, not `ws:test` or `ws:acme`:**

  * `ws:test` (the `conftest` convention) bootstraps its vector indexes at
    `TEST_EMBEDDING_DIM = 4`. A real embedder returns ~1024 dims, and a wrong-dim
    `vecf32` write is **silently accepted and then drops out of the ANN index**
    (AGENTS.md) — AC-3 would vacuously "pass" while retrieving nothing. Disqualifying.
  * `ws:acme` is the live demo workspace with real data; a test must not write into it.

So this module bootstraps a throwaway `ws:live` at the **probed** live embedding
dimension (never a hardcoded 1024 — the model is whatever LM Studio has loaded). Same
isolated-throwaway approach as `scripts/load_test.sh`'s `ws:load`. Set `KEEP_WS=1` to
keep the graph afterwards for inspection.

**The def under test is the real one.** Rather than duplicating the triage spec here
(which would drift from what ships), the module runs U13's `scripts/seed_workflows.sh`
against `ws:live` — the same publish+materialize service-layer path the served app
relies on. `conftest.py` sets the precedent of shelling out to `bootstrap_schema.sh`.

**Assertions are structural, never on model wording.** The intake→research guard is
judged by a live LLM (bias-to-suspend on ambiguity), so exact phrasing — and even the
number of clarifying rounds — is non-deterministic. The test drives the conversation
like a human would, up to a bounded ceiling, and asserts only invariants the engine
must uphold regardless of what the model says: the `TRIGGERED_BY` edge (AC-1), a
contiguous `StepRun` `NEXT` audit chain, that the run visits research + answer, that a
grounded reply was posted and linked `PRODUCED` (AC-4), and that the run reaches `done`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from falkorchat import config, db
from falkorchat.config import CallContext
from falkorchat.embedding import EmbeddingWorker, LMStudioEmbedder
from falkorchat.executor import GraphTracer, WorkflowExecutor
from falkorchat.llm import LMStudioLLM
from falkorchat.repository import Repository
from falkorchat.services import Services
from falkorchat.tools import build_builtin_registry
from falkorchat.trigger import WorkflowTrigger

pytestmark = pytest.mark.live

LIVE_WS = "live"
AGENT_ID = "assistant"
USER_ID = "u1"
CTX = CallContext(ws=LIVE_WS, actor=USER_ID)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP = _REPO_ROOT / "scripts" / "bootstrap_schema.sh"
_SEED_WORKFLOWS = _REPO_ROOT / "scripts" / "seed_workflows.sh"

# The intake guard is LLM-judged and biased to suspend, so the number of clarifying
# rounds is not deterministic. Drive like a human up to the DS-note 3-round clarifying
# ceiling (`docs/plans/m3-executor-ml.md` Q1), +1 of headroom before we call it a
# finding rather than a flake.
MAX_CLARIFY_ROUNDS = 4

# The seeded corpus AC-3 retrieves over: a small, distinctive conversation the research
# node can ground findings in. Deliberately specific (service name, version, root cause)
# so a retrieval hit is unambiguous and not something the model could confabulate.
CORPUS = [
    "The checkout service started returning 502 errors right after the v4.2 deploy "
    "went out on Tuesday.",
    "We rolled back checkout to v4.1 and the 502 errors stopped immediately.",
    "Root cause of the checkout v4.2 502s: the database connection pool limit was "
    "lowered from 100 to 10 in the v4.2 config change.",
    "Action item: restore the checkout connection pool limit to 100 and re-deploy "
    "v4.2 with the corrected config.",
    "Unrelated: the marketing site redesign is scheduled for next quarter.",
]

# The human side of the triage conversation. Turn 0 is the @mention that starts the run
# (AC-1); the rest are plain replies with NO re-@mention (AC-2 — a reply to a parked run
# must resume it without re-mentioning the agent). Each supplies progressively more
# detail, so a reasonably-calibrated judge should let the guard fire.
TRIGGER_TEXT = "@assistant I need help figuring out what happened with our checkout service."
HUMAN_REPLIES = [
    "The checkout service was returning 502 errors to customers. It started right "
    "after we deployed version 4.2 on Tuesday, and rolling back to 4.1 fixed it. "
    "I want to know the root cause and how to safely re-deploy v4.2.",
    "Nothing else changed that day — just the v4.2 deploy. The 502s hit every "
    "checkout request, and I need the root cause and the fix so we can re-deploy.",
    "That is everything I know: checkout v4.2, 502s on every request, rollback to "
    "v4.1 resolved it. Please research the root cause now.",
]


# ── live-dependency gating ────────────────────────────────────────────────────


def _falkordb_reachable() -> bool:
    try:
        db.connect().select_graph(f"ws:{LIVE_WS}").query("RETURN 1")
        return True
    except Exception:
        return False


def _probe_embedding_dim() -> int:
    """Embed a probe string against the live model to learn the real vector width.

    The workspace's vector indexes must be created at exactly this dimension — a
    mismatch is silently accepted on write and then drops out of ANN (AGENTS.md), which
    would make AC-3 pass while retrieving nothing. So we ask the model rather than
    hardcode 1024.
    """
    return len(LMStudioEmbedder().embed("probe"))


@pytest.fixture(scope="module")
def live_dim() -> int:
    """Skip unless BOTH live deps are reachable; return the probed embedding dim."""
    if not _falkordb_reachable():
        pytest.skip("FalkorDB not reachable — start it with ./scripts/start_falkordb.sh -d")
    try:
        dim = _probe_embedding_dim()
    except Exception as exc:
        pytest.skip(
            f"LM Studio not reachable at {config.EMBEDDING_BASE_URL} ({exc!r}) — "
            f"start it and load a chat + an embedding model"
        )
    return dim


@pytest.fixture(scope="module")
def live_ws(live_dim: int) -> int:
    """A throwaway `ws:live` graph bootstrapped at the probed embedding dimension.

    Dropped first so the vector-index dimension is deterministic (a vector index's
    dimension is fixed at creation and cannot be altered in place — a stale wrong-dim
    index from an earlier run with a different model would silently break AC-3), then
    dropped again at teardown unless `KEEP_WS=1`.
    """
    try:
        db.connect().select_graph(f"ws:{LIVE_WS}").delete()
    except Exception:
        pass  # graph may not exist yet — bootstrap creates it fresh
    subprocess.run(
        ["bash", str(_BOOTSTRAP), LIVE_WS],
        check=True, capture_output=True, text=True,
        env={**os.environ, "EMBEDDING_DIM": str(live_dim)},
    )
    # Publish + materialize the REAL triage def (U13's script) into ws:live, so this
    # test tracks the shipped spec instead of a copy that could drift.
    subprocess.run(
        ["bash", str(_SEED_WORKFLOWS), LIVE_WS],
        check=True, capture_output=True, text=True,
        env={**os.environ, "FALKORCHAT_WS_ID": LIVE_WS},
    )
    yield live_dim
    if not os.environ.get("KEEP_WS"):
        try:
            db.connect().select_graph(f"ws:{LIVE_WS}").delete()
        except Exception:
            pass


# ── the live stack (the same wiring `app._build_default_app` uses) ────────────


def _build_live_stack(dim: int):
    """Wire the real executor + trigger exactly as `app._build_default_app` does when
    `FALKORCHAT_WORKFLOW_ENABLED` is on — real LM Studio LLM, real embedder, real tool
    registry, real graph tracer — but pointed at `ws:live` and driven in-process.

    Driving the trigger directly (rather than over HTTP) keeps the test synchronous and
    race-free: `start_workflow_run`/`resume_workflow_run` drive the executor loop inline,
    so there is nothing to poll for. Black-box REST acceptance is U15's job (qa-engineer).
    """
    from falkorchat.app import _build_llm_judge

    repo = Repository(db.connect())
    services = Services(repo)
    embedder = LMStudioEmbedder()
    registry = build_builtin_registry(services, embedder, agent_id=AGENT_ID)
    executor = WorkflowExecutor(
        services, repo, llm=LMStudioLLM(), guard_judge=_build_llm_judge(LMStudioLLM()),
        tool_registry=registry, tracer=GraphTracer(repo),
    )
    services.set_executor(executor)
    trigger = WorkflowTrigger(
        services, agent_id=AGENT_ID, def_key=config.TRIGGER_DEF_KEY,
        def_version=config.TRIGGER_DEF_VERSION, responder=None,
        trace=True,  # a debug instance — the trace is the diagnostic record (FR-4)
    )
    return repo, services, trigger, EmbeddingWorker(repo, embedder, expected_dim=dim)


def _seed_conversation(
    repo: Repository, services: Services, worker: EmbeddingWorker
) -> str:
    """Seed the AC-3 precondition: a channel/thread of embedded corpus messages, plus
    the (separate) thread the triage run will happen in. Returns the run thread's id.

    The members are registered here rather than via `seed_demo.sh` — that script seeds
    the fixed `ws:acme` demo ids, whereas this workspace is a throwaway that only needs
    the human author and the `assistant` Agent the workflow posts as (the §4 write path
    anchors on the author node and refuses an unknown one).

    Embeddings are written explicitly here because the real write path computes them
    out-of-band on a background task (DESIGN §9) — this test drives no background worker.
    `hybrid_search` is workspace-wide in M2 (K-015), so corpus in a sibling thread is
    retrievable from the run's thread.
    """
    repo.ensure_user(LIVE_WS, user_id=USER_ID, display_name="Alice")
    repo.ensure_agent(LIVE_WS, agent_id=AGENT_ID, name="Assistant")

    channel = services.create_channel(CTX, name="incidents")
    corpus_thread = services.create_thread(
        CTX, channel_id=channel["channelId"], title="checkout incident"
    )
    for text in CORPUS:
        posted = services.post_message(
            CTX, thread_id=corpus_thread["threadId"], text=text
        )
        worker.embed_message(LIVE_WS, msg_id=posted["msgId"], text=text)

    run_thread = services.create_thread(
        CTX, channel_id=channel["channelId"], title="triage"
    )
    return run_thread["threadId"]


def _guard_judgments(graph) -> list[str]:
    """The live judge's verdicts + rationales, straight from the debug trace (FR-4).

    Folded into the failure message so a RED run explains *why* the guard did not fire
    instead of just reporting a status mismatch — this trace is what turned U14's first
    failure into a diagnosed defect rather than a suspected flake.
    """
    rows = graph.ro_query(
        "MATCH (sr:StepRun)-[:TRACED]->(t:TraceEvent) "
        "WHERE t.kind = 'guard_judgment' RETURN t.payload ORDER BY t.at"
    ).result_set
    return [p for (p,) in rows]


def _post_and_trigger(services, trigger, *, thread_id, text, mentions=None):
    """Post a human message and hand it to the trigger, exactly as the API's background
    task does (`api._safe_run_workflow`). Returns the trigger's result."""
    posted = services.post_message(
        CTX, thread_id=thread_id, text=text, mentions=mentions
    )
    result = trigger.maybe_trigger(
        CTX, thread_id=thread_id, msg_id=posted["msgId"], text=text,
        role="user", mentions=mentions or [],
    )
    return posted, result


# ── the test ──────────────────────────────────────────────────────────────────


def test_triage_flow_runs_end_to_end_against_live_llm(live_ws):
    """AC-1…AC-4: an @mention starts a triage run that clarifies, researches, answers
    and reaches `done`, with a `TRIGGERED_BY` anchor and a contiguous StepRun trace."""
    repo, services, trigger, worker = _build_live_stack(live_ws)
    thread_id = _seed_conversation(repo, services, worker)
    graph = db.workspace_graph(db.connect(), LIVE_WS)

    # ── AC-1: the @mention starts a run anchored to the triggering message ──
    trig_msg, started = _post_and_trigger(
        services, trigger, thread_id=thread_id, text=TRIGGER_TEXT, mentions=[AGENT_ID]
    )
    assert started is not None, "the @mention did not start a workflow run"
    run_id = started["runId"]

    [[triggered_by]] = graph.ro_query(
        "MATCH (r:WorkflowRun {runId: $runId})-[:TRIGGERED_BY]->(m:Message) "
        "RETURN m.msgId",
        {"runId": run_id},
    ).result_set
    assert triggered_by == trig_msg["msgId"], "run is not anchored to the trigger message"

    # ── AC-2: intake parks `waiting` (it must not advance on the vague opener) and a
    #    PLAIN human reply — no re-@mention — resumes it until the fuzzy guard fires.
    assert started["status"] == "waiting", (
        f"intake should suspend for clarification, got {started['status']!r}"
    )

    status = started["status"]
    rounds = 0
    for reply in HUMAN_REPLIES:
        if status != "waiting":
            break
        rounds += 1
        _posted, resumed = _post_and_trigger(
            services, trigger, thread_id=thread_id, text=reply  # NO mentions — AC-2
        )
        assert resumed is not None, "a reply to a waiting run did not resume it"
        status = resumed["status"]
        if rounds >= MAX_CLARIFY_ROUNDS:
            break

    # ── AC-4: the run reached a terminal `done` ──
    trail = [s["stepKey"] for s in services.read_workflow_step_runs(CTX, run_id=run_id)]
    assert status == "done", (
        f"run did not complete: status={status!r} after {rounds} clarifying "
        f"round(s). Step trail: {trail}\n"
        f"Live guard judgments: {_guard_judgments(graph)}\n"
        f"(A trail of only 'intake' with every judgment False is the U14 finding: the "
        f"intake node emits prose, so `guards._extract_understanding` resolves to {{}} "
        f"and the judge can never see enough to advance — see the U14 report.)"
    )

    run = services.get_workflow_run(CTX, run_id=run_id)
    assert run["status"] == "done"

    # ── the StepRun NEXT audit chain is contiguous and visits the whole flow ──
    step_runs = services.read_workflow_step_runs(CTX, run_id=run_id)
    visited = [s["stepKey"] for s in step_runs]
    assert visited[0] == "intake", f"the run did not start at intake: {visited}"
    assert "research" in visited, f"the run never reached research (AC-3): {visited}"
    assert visited[-1] == "answer", f"the run did not end at answer (AC-4): {visited}"

    [[chain_len]] = graph.ro_query(
        "MATCH (r:WorkflowRun {runId: $runId})-[:HAS_STEP_RUN]->(sr:StepRun) "
        "WHERE NOT ()-[:NEXT]->(sr) "
        "MATCH p = (sr)-[:NEXT*0..]->(:StepRun) "
        "RETURN max(length(p)) + 1",
        {"runId": run_id},
    ).result_set
    assert chain_len == len(step_runs), (
        f"StepRun NEXT chain is not contiguous: chain walks {chain_len} of "
        f"{len(step_runs)} recorded step-runs"
    )

    # ── AC-4: the answer node posted a real, PRODUCED-linked reply into the thread ──
    produced = graph.ro_query(
        "MATCH (r:WorkflowRun {runId: $runId})-[:HAS_STEP_RUN]->(sr:StepRun) "
        "MATCH (sr)-[:PRODUCED]->(m:Message) "
        "RETURN sr.stepKey, m.text ORDER BY m.createdAt",
        {"runId": run_id},
    ).result_set
    assert produced, "no StepRun-[:PRODUCED]->Message edge — the run posted nothing"
    assert any(step == "answer" for step, _text in produced), (
        f"the answer node never posted a reply (AC-4); posts came from: "
        f"{[s for s, _ in produced]}"
    )

    # The thread itself must carry the agent's turns (visible to participants, FR-5a).
    thread = services.read_thread(CTX, thread_id=thread_id)
    assert any(m["role"] == "assistant" for m in thread), (
        "the workflow posted nothing visible in the thread"
    )
