"""Unit tests for the AgentResponder (K-013), fully mocked — no DB, no network.

Retrieval (`services.hybrid_search` + `post_agent_answer`), the LLM, the embedder,
and the answer-embedding worker are all injected fakes. The responder's contract:

  * trigger = the incoming message @mentions the agent AND is not agent-authored;
  * flow = embed trigger → hybrid_search (channel-scoped) → LLM → post as the agent
    with the retrieved seeds as `EMITTED` provenance in rank order;
  * failure isolation = embedder/LLM run BEFORE the post, so any failure ⇒ no post;
  * loop guard = an `assistant`-role trigger never responds (no self-answer loop).
"""

from __future__ import annotations

import pytest

from falkorchat.config import CallContext
from falkorchat.responder import AgentResponder

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
        {"msgId": "s1", "text": "seed one", "role": "user", "score": 0.0},
        {"msgId": "s2", "text": "seed two", "role": "user", "score": 0.3},
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


# ── loop guard: agent-authored messages never trigger ─────────────────────────


def test_assistant_role_message_never_responds():
    services = FakeServices(seeds=[{"msgId": "s1", "text": "x", "role": "user", "score": 0.0}])
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
    services = FakeServices(seeds=[{"msgId": "s1", "text": "x", "role": "user", "score": 0.0}])
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
    services = FakeServices(seeds=[{"msgId": "s1", "text": "x", "role": "user", "score": 0.0}])
    worker = SpyWorker()
    responder = _responder(services, worker=worker)

    responder.maybe_respond(
        CTX, thread_id="t1", msg_id="m1", text="q", role="user",
        channel_id="c1", mentions=[AGENT_ID],
    )

    # exactly one post: embedding the answer is a write, never a new trigger
    assert len(services.post_calls) == 1
    assert len(worker.calls) == 1
