"""The server-side AI agent participant (K-013, DESIGN §M2).

`AgentResponder` turns a triggering chat message into an agent-authored answer
grounded in GraphRAG retrieval, recording the retrieved seeds as `EMITTED`
provenance (QUERIES.md §10).

Trigger (locked, DECISION 2): respond when the incoming message **@mentions the
agent** and is **not itself agent-authored** (`role != "assistant"`). The second
condition is the hard loop-guard — without it an agent answer that mentions the
agent would trigger another answer forever.

Flow: embed the trigger → `services.hybrid_search` (channel-scoped) for ranked
seeds → build the LLM prompt from the retrieved context → call the LLM → post the
answer **as the agent** (`services.post_agent_answer`, role derived `assistant`)
with the retrieved `(msgId, score)` seeds in rank order → self-embed the answer so
it joins the retrievable corpus.

Failure isolation (hard): the embedder, retrieval, and LLM all run **before** the
guarded §4/§10 write. Any failure short-circuits before anything is posted — LLM
latency or failure can never leave a torn thread. The answer's own embedding runs
**after** the post (a write, never a new trigger — embedding ≠ triggering).

Everything the responder depends on is injected (services, embedder, llm, worker),
so it is exercised fully mocked in tests — no DB, no network.
"""

from __future__ import annotations

from typing import Any

from .config import CallContext
from .embedding import Embedder, EmbeddingWorker
from .llm import LLM, ChatMessage

_SYSTEM_PROMPT = (
    "You are a helpful assistant participating in a team chat. Answer the user's "
    "question using the retrieved context below when it is relevant. If the context "
    "does not help, answer from general knowledge. Be concise."
)


class AgentResponder:
    """Retrieval-grounded AI responder that posts as the workspace Agent."""

    def __init__(
        self,
        services: Any,
        embedder: Embedder,
        llm: LLM,
        worker: EmbeddingWorker,
        *,
        agent_id: str,
        k: int = 10,
    ) -> None:
        self._services = services
        self._embedder = embedder
        self._llm = llm
        self._worker = worker
        self._agent_id = agent_id
        self._k = k

    def _should_respond(self, *, role: str, mentions: list[str] | None) -> bool:
        # Loop guard first: never respond to an agent-authored message, even if it
        # @mentions the agent (prevents the self-answer loop). Then the sole
        # trigger: the incoming message @mentions this agent.
        if role == "assistant":
            return False
        return self._agent_id in (mentions or [])

    def _build_prompt(self, question: str, seeds: list[dict[str, Any]]) -> list[ChatMessage]:
        """Turn retrieved seed texts + the question into a chat message list."""
        if seeds:
            context = "\n".join(f"- {s['text']}" for s in seeds)
            user = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            user = question
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def maybe_respond(
        self, ctx: CallContext, *, thread_id: str, msg_id: str, text: str,
        role: str, channel_id: str | None, mentions: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Answer `text` if it triggers the agent; otherwise return None.

        Runs out-of-band from the caller's request (scheduled on FastAPI
        `BackgroundTasks`), so its latency never blocks the poster.
        """
        if not self._should_respond(role=role, mentions=mentions):
            return None

        # Failure isolation: everything that can fail (embed, retrieve, LLM) runs
        # BEFORE the guarded write. A raise here means nothing is posted.
        q_vec = self._embedder.embed(text)
        seeds = self._services.hybrid_search(
            ctx, q_vec=q_vec, k=self._k, channel_id=channel_id
        )
        answer_text = self._llm.complete(self._build_prompt(text, seeds))

        # Post as the agent — swap the actor to the agent id so role derives to
        # `assistant` in the service (never trusted from the caller). Provenance =
        # retrieved (msgId, score) in the retrieval's rank order (score ASC).
        agent_ctx = CallContext(ws=ctx.ws, actor=self._agent_id)
        seed_prov = [(s["msgId"], s["score"]) for s in seeds]
        posted = self._services.post_agent_answer(
            agent_ctx, thread_id=thread_id, text=answer_text, seeds=seed_prov
        )

        # Self-embed the answer AFTER the post so it becomes retrievable. This is
        # a plain write — it never re-enters the trigger path (embedding ≠
        # triggering). Kept last so a persist hiccup can't lose the posted answer.
        self._worker.embed_message(ctx.ws, msg_id=posted["msgId"], text=answer_text)
        return posted
