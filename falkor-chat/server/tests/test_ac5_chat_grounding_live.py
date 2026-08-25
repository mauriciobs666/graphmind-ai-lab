"""Live AC-5 chat-grounding e2e (K-050 M5 Stage 5, FR-2, `docs/plans/document-
ingestion.md` §5 AC-5): ingest a document, `@mention` the agent with a question
the ingested content answers, and assert the answer's `EMITTED` provenance
resolves back to the source chunk/document. This is AC-5's second, real-LLM
altitude — the first (mocked LLM) ships as
`test_responder.py::test_ac5_document_grounded_answer_provenance_resolves_to_chunk_and_document`.

**Marker-gated**, mirroring `test_workflow_live.py`: `pytest.mark.live`, DESELECTED
by default (`addopts = -m "not live"` in `pyproject.toml`), so the standard
offline `pytest -q` baseline is unaffected. Run explicitly:

    cd server && .venv/bin/python -m pytest -m live -s tests/test_ac5_chat_grounding_live.py

Needs: FalkorDB up (`./scripts/start_falkordb.sh -d`) **and** LM Studio serving a
chat model + an embedding model at `FALKORCHAT_LIVE_LLM_BASE_URL` (default
`localhost:1234`). Either being unreachable skips (never fails) with a reason.

**Scope is deliberately narrower than `test_workflow_live.py`**: only `Services` +
`Repository` + `AgentResponder` are exercised — no `WorkflowExecutor`/
`WorkflowTrigger`, no `seed_workflows.sh` (AC-5 is the direct-reply chat-grounding
scenario, not a workflow run). Closer in spirit to `test_services_live.py`'s "real
repository, real Cypher" posture, with a real LLM/embedder standing in for the
network edge that file doesn't need.

**Why its own `ws:live5` workspace**: not `ws:test` (bootstrapped at a fixed dim-4
index — a real embedder's ~1024-dim vectors would silently drop out of the ANN
index on write, AGENTS.md) and not `ws:live` (owned by `test_workflow_live.py`,
module-scoped and torn down independently at the end of that file's run — sharing
it risks a teardown race between two live test modules). Bootstrapped at the
probed live embedding dimension, same pattern as `test_workflow_live.py`'s
`live_ws` fixture. Set `KEEP_WS=1` to keep the graph afterwards for inspection.

**Retrieval is deterministic despite using a real model.** The workspace has
exactly one embedded candidate in each ANN pool at assertion time (one ingested
chunk, zero embedded messages — the trigger message is never embedded, mirroring
the responder's real write path), so the ingested chunk surfaces as a seed
regardless of the real embedding model's precision; `AgentResponder` (unlike
`GraphragRetrieveTool`) applies no τ/cap filter, so `hybrid_search`'s only
candidate is guaranteed to reach the write. This keeps the test's pass/fail
independent of embedding-quality variance across whatever model LM Studio has
loaded.

**Assertion altitude is structural**, on the response shape returned by
`Repository.read_provenance` — never on the model's answer wording (a real LLM's
exact phrasing is non-deterministic), mirroring `test_workflow_live.py`'s own
"assertions are structural" discipline.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from falkorchat import db
from falkorchat.config import CallContext
from falkorchat.embedding import EmbeddingWorker, OpenAICompatibleEmbedder
from falkorchat.llm import OpenAICompatibleLLM
from falkorchat.repository import Repository
from falkorchat.responder import AgentResponder
from falkorchat.services import Services

pytestmark = pytest.mark.live

# Same live-test-only knobs as `test_workflow_live.py` (K-042: model choice is not
# a product-wide env var — these are never read by `config.py`).
LIVE_LLM_BASE_URL = os.environ.get("FALKORCHAT_LIVE_LLM_BASE_URL", "http://localhost:1234/v1")
LIVE_LLM_MODEL = os.environ.get("FALKORCHAT_LIVE_LLM_MODEL", "qwen/qwen3-4b-2507")
LIVE_EMBEDDING_MODEL = os.environ.get(
    "FALKORCHAT_LIVE_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b"
)

LIVE_WS = "live5"
AGENT_ID = "assistant"
USER_ID = "u1"
CTX = CallContext(ws=LIVE_WS, actor=USER_ID)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP = _REPO_ROOT / "scripts" / "bootstrap_schema.sh"


# ── live-dependency gating (mirrors test_workflow_live.py) ────────────────────


def _falkordb_reachable() -> bool:
    try:
        db.connect().select_graph(f"ws:{LIVE_WS}").query("RETURN 1")
        return True
    except Exception:
        return False


def _probe_embedding_dim() -> int:
    """Embed a probe string against the live model to learn the real vector width
    — the workspace's vector indexes must be created at exactly this dimension
    (AGENTS.md: a mismatch is silently accepted on write and drops out of ANN)."""
    return len(OpenAICompatibleEmbedder(LIVE_LLM_BASE_URL, LIVE_EMBEDDING_MODEL).embed("probe"))


@pytest.fixture(scope="module")
def live_dim() -> int:
    """Skip unless BOTH live deps are reachable; return the probed embedding dim."""
    if not _falkordb_reachable():
        pytest.skip("FalkorDB not reachable — start it with ./scripts/start_falkordb.sh -d")
    try:
        dim = _probe_embedding_dim()
    except Exception as exc:
        pytest.skip(
            f"LM Studio not reachable at {LIVE_LLM_BASE_URL} ({exc!r}) — "
            f"start it and load a chat + an embedding model"
        )
    return dim


@pytest.fixture(scope="module")
def live_ws(live_dim: int) -> int:
    """A throwaway `ws:live5` graph bootstrapped at the probed embedding dimension.

    Dropped first (a stale wrong-dim index from an earlier run with a different
    model would silently break this test), then dropped again at teardown unless
    `KEEP_WS=1`.
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
    yield live_dim
    if not os.environ.get("KEEP_WS"):
        try:
            db.connect().select_graph(f"ws:{LIVE_WS}").delete()
        except Exception:
            pass


# ── the test ────────────────────────────────────────────────────────────────


def test_ac5_document_grounded_answer_provenance_resolves_to_chunk_and_document_live(
    live_ws: int,
) -> None:
    dim = live_ws
    repo = Repository(db.connect())
    services = Services(repo)
    embedder = OpenAICompatibleEmbedder(LIVE_LLM_BASE_URL, LIVE_EMBEDDING_MODEL)
    llm = OpenAICompatibleLLM(LIVE_LLM_BASE_URL, LIVE_LLM_MODEL)
    worker = EmbeddingWorker(repo, embedder, expected_dim=dim)

    repo.ensure_user(LIVE_WS, user_id=USER_ID)
    repo.ensure_agent(LIVE_WS, agent_id=AGENT_ID, name="Assistant")
    channel = services.create_channel(CTX, name="c1")
    thread = services.create_thread(CTX, channel_id=channel["channelId"], title="t1")

    # Ingest a document whose content answers the question the trigger asks, then
    # embed its (single) chunk via the real embedder — mirrors the real write path,
    # where chunk embedding runs out-of-band from ingest (DESIGN §9).
    ingested = services.ingest_document(
        CTX, text="The capital of Freedonia is Fredonia City.",
        title="Freedonia Facts",
    )
    chunks = services.list_document_chunks(CTX, document_id=ingested["documentId"])
    assert len(chunks) == 1  # short text — one chunk, so the ANN pool is deterministic
    worker.embed_chunk(LIVE_WS, chunk_id=chunks[0]["chunkId"], text=chunks[0]["text"])

    # The triggering @mention.
    trigger = services.post_message(
        CTX, thread_id=thread["threadId"],
        text="What is the capital of Freedonia?", mentions=[AGENT_ID],
    )

    responder = AgentResponder(services, embedder, llm, worker, agent_id=AGENT_ID, k=10)
    posted = responder.maybe_respond(
        CTX, thread_id=thread["threadId"], msg_id=trigger["msgId"],
        text=trigger["text"], role="user", channel_id=channel["channelId"],
        mentions=[AGENT_ID],
    )

    assert posted is not None

    prov = repo.read_provenance(LIVE_WS, msg_id=posted["msgId"])
    chunk_rows = [p for p in prov if p["seedKind"] == "Chunk"]
    assert chunk_rows, (
        f"expected the ingested chunk to be cited as EMITTED provenance, "
        f"got {prov!r}"
    )
    assert chunk_rows[0]["seedId"] == chunks[0]["chunkId"]
    assert chunk_rows[0]["documentId"] == ingested["documentId"]
    assert chunk_rows[0]["documentTitle"] == "Freedonia Facts"
    assert chunk_rows[0]["role"] is None

    # Reverse read: the ingested chunk is discoverable as "cited by" the answer.
    citing = repo.read_citing_answers(LIVE_WS, seed_id=chunks[0]["chunkId"])
    assert [c["answerMsgId"] for c in citing] == [posted["msgId"]]
