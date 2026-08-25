"""Unit tests for the AgentResponder (K-013), fully mocked — no DB, no network.

Retrieval (`services.hybrid_search` + `post_agent_answer`), the LLM, the embedder,
and the answer-embedding worker are all injected fakes. The responder's contract:

  * trigger = the incoming message @mentions the agent AND is not agent-authored;
  * flow = embed trigger → hybrid_search (channel-scoped) → LLM → post as the agent
    with the retrieved seeds as `EMITTED` provenance in rank order — seeds can be
    `Message`- or `Chunk`-shaped (K-050 M5 Stage 5), discriminated by `seedKind`;
  * failure isolation = embedder/LLM run BEFORE the post, so any failure ⇒ no post;
  * loop guard = an `assistant`-role trigger never responds (no self-answer loop).

**One deliberate exception** (bottom of the file, "AC-5 chat-grounding
integration"): a single live-FalkorDB-backed test using the real `repo`/`conn`
fixtures and a real `Services` instead of `FakeServices` — plan §5's AC-5 row
names the end-to-end scenario (ingest a document, `@mention` the agent with a
question the content answers, the answer's `EMITTED` edge resolves back to the
source chunk/document), which unit-level coverage of the pieces (this file's
`FakeServices`-based Chunk-seed tests above, `test_provenance.py`'s repo-level
Chunk-seed round trip) doesn't by itself prove wired together. Only the LLM and
embedder are mocked — everything else (ingestion, chunk embedding, message
post, retrieval, provenance write/read) is the real stack, mirroring
`test_ingestion.py`'s identical "one deliberate exception" precedent for AC-8.
"""

from __future__ import annotations

import pytest
from conftest import TEST_EMBEDDING_DIM

from falkorchat.config import CallContext
from falkorchat.responder import AgentResponder
from falkorchat.services import Services

CTX = CallContext(ws="test", actor="u1")
AGENT_ID = "bot1"


class StubEmbedder:
    def __init__(self, vector=None, *, fail=False):
        self._vector = vector or [1.0, 0.0, 0.0, 0.0]
        self._fail = fail
        self.seen: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.seen.append(text)
        if self._fail:
            raise RuntimeError("embedder down")
        return list(self._vector)


class StubLLM:
    def __init__(self, answer="the answer", *, fail=False):
        self._answer = answer
        self._fail = fail
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        if self._fail:
            raise RuntimeError("llm down")
        return self._answer


class SpyWorker:
    def __init__(self):
        self.calls: list[tuple] = []

    def embed_message(self, ws, *, msg_id, text):
        self.calls.append((ws, msg_id, text))
        return [0.0]


class FakeServices:
    def __init__(self, seeds=None):
        self._seeds = seeds or []
        self.hybrid_calls: list[dict] = []
        self.post_calls: list[dict] = []
        self._counter = 0

    def hybrid_search(self, ctx, *, q_vec, k=10, limit=10, channel_id=None):
        self.hybrid_calls.append(
            {"ctx": ctx, "q_vec": list(q_vec), "k": k, "channel_id": channel_id}
        )
        return list(self._seeds)

    def post_agent_answer(self, ctx, *, thread_id, text, mentions=None, seeds=None):
        self._counter += 1
        posted = {
            "msgId": f"ag{self._counter}", "threadId": thread_id,
            "authorId": ctx.actor, "text": text, "role": "assistant",
            "seeds": list(seeds or []),
        }
        self.post_calls.append({"ctx": ctx, **posted, "mentions": mentions})
        return posted


def _responder(services, *, embedder=None, llm=None, worker=None, k=10):
    return AgentResponder(
        services,
        embedder or StubEmbedder(),
        llm or StubLLM(),
        worker or SpyWorker(),
        agent_id=AGENT_ID,
        k=k,
    )


# ── happy path: mention → retrieve → LLM → post with provenance ───────────────


def test_mention_triggers_answer_with_provenance_in_rank_order():
    seeds = [
        {"msgId": "s1", "text": "seed one", "role": "user", "score": 0.0, "seedKind": "Message"},
        {"msgId": "s2", "text": "seed two", "role": "user", "score": 0.3, "seedKind": "Message"},
    ]
    services = FakeServices(seeds=seeds)
    llm = StubLLM(answer="grounded reply")
    worker = SpyWorker()
    responder = _responder(services, llm=llm, worker=worker)

    out = responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="what about cats?",
        role="user", channel_id="c1", mentions=[AGENT_ID],
    )

    # retrieval is channel-scoped, using the embedded trigger as the query vector
    assert len(services.hybrid_calls) == 1
    assert services.hybrid_calls[0]["channel_id"] == "c1"
    assert services.hybrid_calls[0]["k"] == 10

    # posted as the agent, LLM answer is the text, seeds = retrieved (msgId, score) in rank order
    assert len(services.post_calls) == 1
    posted = services.post_calls[0]
    assert posted["ctx"].actor == AGENT_ID
    assert posted["ctx"].ws == "test"
    assert posted["text"] == "grounded reply"
    assert posted["seeds"] == [("s1", 0.0), ("s2", 0.3)]
    assert out["text"] == "grounded reply"

    # seed texts flow into the LLM prompt
    prompt_text = " ".join(
        m["content"] for msg in llm.calls for m in msg
    )
    assert "seed one" in prompt_text and "seed two" in prompt_text

    # the answer is self-embedded after the post (grows the retrievable corpus)
    assert worker.calls == [("test", out["msgId"], "grounded reply")]


# ── Chunk-seeded provenance threading (K-050 M5 Stage 5) ───────────────────────


def test_chunk_seeded_answer_threads_chunk_id_into_provenance():
    """A `Chunk`-seeded hit (no `msgId` key at all) must resolve via `chunkId`,
    not silently drop the seed or crash on a missing `msgId`.
    """
    seeds = [
        {"chunkId": "ch1", "text": "chunk text", "documentId": "d1", "seq": 0,
         "score": 0.05, "seedKind": "Chunk", "relatedContext": []},
    ]
    services = FakeServices(seeds=seeds)
    responder = _responder(services)

    responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="what does the doc say?",
        role="user", channel_id="c1", mentions=[AGENT_ID],
    )

    posted = services.post_calls[0]
    assert posted["seeds"] == [("ch1", 0.05)]


def test_mixed_message_and_chunk_seeds_thread_correct_ids_in_rank_order():
    seeds = [
        {"msgId": "s1", "text": "seed one", "role": "user", "score": 0.0,
         "seedKind": "Message"},
        {"chunkId": "ch1", "text": "chunk text", "documentId": "d1", "seq": 0,
         "score": 0.2, "seedKind": "Chunk", "relatedContext": []},
    ]
    services = FakeServices(seeds=seeds)
    responder = _responder(services)

    responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="mixed grounding",
        role="user", channel_id="c1", mentions=[AGENT_ID],
    )

    posted = services.post_calls[0]
    assert posted["seeds"] == [("s1", 0.0), ("ch1", 0.2)]


# ── loop guard: agent-authored messages never trigger ─────────────────────────


def test_assistant_role_message_never_responds():
    services = FakeServices(seeds=[{"msgId": "s1", "text": "x", "role": "user", "score": 0.0, "seedKind": "Message"}])
    llm = StubLLM()
    responder = _responder(services, llm=llm)

    out = responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="i am the agent",
        role="assistant", channel_id="c1", mentions=[AGENT_ID],
    )

    assert out is None
    assert services.hybrid_calls == []
    assert services.post_calls == []
    assert llm.calls == []


# ── no trigger: agent not mentioned ───────────────────────────────────────────


def test_no_mention_no_response():
    services = FakeServices()
    responder = _responder(services)

    out = responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="just chatting",
        role="user", channel_id="c1", mentions=["u2"],
    )

    assert out is None
    assert services.post_calls == []


def test_empty_mentions_no_response():
    services = FakeServices()
    responder = _responder(services)
    out = responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="hi", role="user",
        channel_id="c1", mentions=[],
    )
    assert out is None
    assert services.post_calls == []


# ── failure isolation: nothing posted if the LLM/embedder fails ───────────────


def test_llm_failure_posts_nothing():
    services = FakeServices(seeds=[{"msgId": "s1", "text": "x", "role": "user", "score": 0.0, "seedKind": "Message"}])
    worker = SpyWorker()
    responder = _responder(services, llm=StubLLM(fail=True), worker=worker)

    with pytest.raises(RuntimeError):
        responder.maybe_respond(
            CTX, thread_id="t1", msg_id="m1", text="q", role="user",
            channel_id="c1", mentions=[AGENT_ID],
        )

    assert services.post_calls == []   # no torn thread
    assert worker.calls == []          # no answer to embed


def test_embedder_failure_posts_nothing():
    services = FakeServices()
    responder = _responder(services, embedder=StubEmbedder(fail=True))

    with pytest.raises(RuntimeError):
        responder.maybe_respond(
            CTX, thread_id="t1", msg_id="m1", text="q", role="user",
            channel_id="c1", mentions=[AGENT_ID],
        )

    assert services.hybrid_calls == []  # short-circuits before retrieval
    assert services.post_calls == []


# ── self-embedding of the answer does not re-enter the trigger path ───────────


def test_answer_self_embedding_does_not_re_trigger():
    services = FakeServices(seeds=[{"msgId": "s1", "text": "x", "role": "user", "score": 0.0, "seedKind": "Message"}])
    worker = SpyWorker()
    responder = _responder(services, worker=worker)

    responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="q", role="user",
        channel_id="c1", mentions=[AGENT_ID],
    )

    # exactly one post: embedding the answer is a write, never a new trigger
    assert len(services.post_calls) == 1
    assert len(worker.calls) == 1


# ── K-042 code review Major 2: the `agent`+`embedding`-kind gateway resolution wiring ──
#
# Every test above injects `embedder=`/`llm=` — the pre-K-042 `StaticModelGateway`
# sugar path, which proves backward compatibility but never exercises
# `maybe_respond`'s own `self._models.embedder("embedding", ws=ctx.ws)` /
# `self._models.llm("agent", ws=ctx.ws)` calls. This uses a small recording
# `models=` double instead, so a regression at either call site (a swapped kind, a
# dropped `ws=`, or the retrieval/answer resolution firing in the wrong order) would
# fail here even though every other test in this file is blind to it.

class RecordingGateway:
    """A minimal `ModelGateway`-shaped double: records every `.embedder()`/`.llm()`
    call's `(kind, ws)`, in call order, and hands back the injected stub clients."""

    def __init__(self, embedder, llm):
        self.calls: list[tuple[str, str]] = []
        self._embedder = embedder
        self._llm = llm

    def embedder(self, kind, *, requested=None, ws=None, overrides=None):
        self.calls.append(("embedder", kind, ws))
        return self._embedder

    def llm(self, kind, *, requested=None, ws=None, overrides=None):
        self.calls.append(("llm", kind, ws))
        return self._llm


def test_maybe_respond_resolves_embedding_then_agent_through_the_gateway_in_order():
    seeds = [{"msgId": "s1", "text": "seed one", "role": "user", "score": 0.0, "seedKind": "Message"}]
    services = FakeServices(seeds=seeds)
    embedder = StubEmbedder()
    llm = StubLLM(answer="grounded reply")
    worker = SpyWorker()
    gateway = RecordingGateway(embedder, llm)
    responder = AgentResponder(
        services, worker=worker, agent_id=AGENT_ID, models=gateway
    )

    out = responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="what about cats?",
        role="user", channel_id="c1", mentions=[AGENT_ID],
    )

    assert gateway.calls == [
        ("embedder", "embedding", CTX.ws),
        ("llm", "agent", CTX.ws),
    ]
    assert out["text"] == "grounded reply"
    assert embedder.seen == ["what about cats?"]
    assert llm.calls  # the resolved llm was actually driven


def test_maybe_respond_does_not_resolve_anything_when_not_triggered():
    # The loop guard / no-mention short-circuit must fire BEFORE any resolution.
    services = FakeServices()
    gateway = RecordingGateway(StubEmbedder(), StubLLM())
    responder = AgentResponder(
        services, worker=SpyWorker(), agent_id=AGENT_ID, models=gateway
    )

    out = responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="just chatting",
        role="user", channel_id="c1", mentions=["u2"],
    )

    assert out is None
    assert gateway.calls == []


# ── AC-5: chat-grounding integration (live repo, mocked LLM) ──────────────────
#
# See the module docstring's "one deliberate exception" note. Real `Repository`
# (`repo`/`conn` fixtures) + real `Services`; only the LLM and embedder are
# stubbed. `TEST_EMBEDDING_DIM`-shaped vectors, same convention as
# `test_graphrag.py`.


def _pad(head: list[float]) -> list[float]:
    return (head + [0.0] * TEST_EMBEDDING_DIM)[:TEST_EMBEDDING_DIM]


def test_ac5_document_grounded_answer_provenance_resolves_to_chunk_and_document(repo):
    ws = "test"
    doc_ctx = CallContext(ws=ws, actor="u1")

    repo.ensure_user(ws, user_id="u1")
    repo.ensure_agent(ws, agent_id="bot1", name="Bot")
    services = Services(repo)

    channel = services.create_channel(doc_ctx, name="c1")
    thread = services.create_thread(
        doc_ctx, channel_id=channel["channelId"], title="t1"
    )

    # Ingest a document whose content answers the question the trigger asks.
    ingested = services.ingest_document(
        doc_ctx, text="The capital of Freedonia is Fredonia City.",
        title="Freedonia Facts",
    )
    chunks = services.list_document_chunks(
        doc_ctx, document_id=ingested["documentId"]
    )
    assert len(chunks) == 1  # short text — one chunk, so ANN retrieval is deterministic
    chunk_vec = _pad([1.0])
    repo.set_chunk_embedding(
        ws, chunk_id=chunks[0]["chunkId"], embedding=chunk_vec,
        expected_dim=TEST_EMBEDDING_DIM,
    )

    # The triggering @mention.
    trigger = services.post_message(
        doc_ctx, thread_id=thread["threadId"],
        text="What is the capital of Freedonia?", mentions=["bot1"],
    )

    embedder = StubEmbedder(vector=chunk_vec)  # same vector → the chunk is the top ANN hit
    llm = StubLLM(answer="Fredonia City is the capital of Freedonia.")
    worker = SpyWorker()
    responder = AgentResponder(
        services, embedder, llm, worker, agent_id="bot1", k=10,
    )

    trigger_ctx = CallContext(ws=ws, actor="u1")
    posted = responder.maybe_respond(
        trigger_ctx, thread_id=thread["threadId"], msg_id=trigger["msgId"],
        text=trigger["text"], role="user", channel_id=channel["channelId"],
        mentions=["bot1"],
    )

    assert posted is not None
    assert posted["text"] == "Fredonia City is the capital of Freedonia."

    prov = repo.read_provenance(ws, msg_id=posted["msgId"])
    assert len(prov) == 1
    row = prov[0]
    assert row["seedKind"] == "Chunk"
    assert row["seedId"] == chunks[0]["chunkId"]
    assert row["documentId"] == ingested["documentId"]
    assert row["documentTitle"] == "Freedonia Facts"
    assert row["role"] is None

    # Reverse read: the ingested chunk is discoverable as "cited by" the answer.
    citing = repo.read_citing_answers(ws, seed_id=chunks[0]["chunkId"])
    assert [c["answerMsgId"] for c in citing] == [posted["msgId"]]
