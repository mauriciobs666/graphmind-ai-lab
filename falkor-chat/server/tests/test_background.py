"""Unit tests for `background.py`'s failure-isolation wrappers (K-041).

K-042 Landing 2 (L2-5, FR-10) adds coverage for the "unresolvable model" case
specifically: `_safe_respond` must swallow whatever `responder.maybe_respond`
raises (never propagate into the caller's `BackgroundTasks`/thread scheduling),
log it at ERROR naming the failing message, and — because the responder's own
failure-isolation ordering (`responder.py`'s docstring: embed → retrieve → LLM
all run BEFORE the guarded write) means a `ModelResolutionError` fires before
`services.post_agent_answer` is ever reached — no reply must be posted.
"""

from __future__ import annotations

import logging

from falkorchat.background import _safe_embed_chunk, _safe_extract, _safe_respond
from falkorchat.config import CallContext
from falkorchat.modelconfig import ModelResolutionError
from falkorchat.responder import AgentResponder

CTX = CallContext(ws="test", actor="u1")
AGENT_ID = "bot1"


class _UnresolvableGateway:
    """Raises `ModelResolutionError` at the very first resolution point
    `maybe_respond` calls (the retrieval embedder) — the same "unresolvable ref"
    failure `test_modelconfig.py` already pins at the resolver layer; this proves
    it reaches `_safe_respond`'s safety net rather than duplicating that
    assertion."""

    def embedder(self, kind, *, requested=None, ws=None, overrides=None):
        raise ModelResolutionError("unknown provider for ref 'nope/thing'")

    def llm(self, kind, *, requested=None, ws=None, overrides=None):
        raise AssertionError("llm() must never be reached — embedder resolution failed first")


class FakeServices:
    """Records `post_agent_answer` calls; `hybrid_search` must never be reached
    (the embedder resolution fails before retrieval)."""

    def __init__(self):
        self.post_calls: list[dict] = []

    def hybrid_search(self, ctx, *, q_vec, k=10, limit=10, channel_id=None):
        raise AssertionError("hybrid_search must never be reached")

    def post_agent_answer(self, ctx, *, thread_id, text, mentions=None, seeds=None):
        self.post_calls.append({"ctx": ctx, "thread_id": thread_id, "text": text})
        return {"msgId": "should-not-happen"}


def test_safe_respond_swallows_an_unresolvable_model_logs_error_and_posts_nothing(caplog):
    services = FakeServices()
    responder = AgentResponder(services, agent_id=AGENT_ID, models=_UnresolvableGateway())
    posted = {
        "threadId": "t1", "msgId": "m1", "text": "hey @bot1", "role": "user",
        "mentions": [AGENT_ID],
    }

    with caplog.at_level(logging.ERROR):
        _safe_respond(responder, CTX, posted)  # must not raise — the safety net itself

    assert services.post_calls == []  # no reply posted

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "background responder failed" in error_records[0].getMessage()
    assert "m1" in error_records[0].getMessage()  # the failing message is named
    # the ERROR log carries the exception (via _log.exception) — the identifier
    # that made the model unresolvable is reachable from the record
    assert error_records[0].exc_info is not None
    assert "nope/thing" in str(error_records[0].exc_info[1])


# ── _safe_embed_chunk (K-050 M5 Stage 2) ─────────────────────────────────────────
#
# Mirrors `_safe_respond`'s failure-isolation contract exactly (this module has
# no dedicated `_safe_embed` unit test to mirror — that path is covered via the
# REST/MCP integration fixtures in test_api.py/test_mcp.py instead — but the
# failure-isolation discipline itself is the same for every `_safe_*` wrapper,
# so it is pinned directly here for the new one).


class _RecordingChunkWorker:
    def __init__(self):
        self.calls: list[tuple] = []

    def embed_chunk(self, ws, *, chunk_id, text):
        self.calls.append((ws, chunk_id, text))


class _FailingChunkWorker:
    def embed_chunk(self, ws, *, chunk_id, text):
        raise RuntimeError(f"boom embedding {chunk_id}")


def test_safe_embed_chunk_calls_the_worker():
    worker = _RecordingChunkWorker()

    _safe_embed_chunk(worker, "test", "c1", "about cats")

    assert worker.calls == [("test", "c1", "about cats")]


def test_safe_embed_chunk_swallows_failure_logs_error_never_raises(caplog):
    worker = _FailingChunkWorker()

    with caplog.at_level(logging.ERROR):
        _safe_embed_chunk(worker, "test", "c1", "about cats")  # must not raise

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "background chunk embed failed" in error_records[0].getMessage()
    assert "c1" in error_records[0].getMessage()
    assert error_records[0].exc_info is not None
    assert "boom embedding c1" in str(error_records[0].exc_info[1])


# ── _safe_extract (K-050 M5 Stage 3) ─────────────────────────────────────────
#
# Mirrors `_safe_embed_chunk`'s failure-isolation contract exactly.


class _RecordingIngestionPipeline:
    def __init__(self):
        self.calls: list[tuple] = []

    def extract_chunk(self, ws, *, chunk_id, document_id, text):
        self.calls.append((ws, chunk_id, document_id, text))


class _FailingIngestionPipeline:
    def extract_chunk(self, ws, *, chunk_id, document_id, text):
        raise RuntimeError(f"boom extracting {chunk_id}")


def test_safe_extract_calls_the_pipeline():
    pipeline = _RecordingIngestionPipeline()

    _safe_extract(pipeline, "test", "c1", "d1", "about cats")

    assert pipeline.calls == [("test", "c1", "d1", "about cats")]


def test_safe_extract_swallows_failure_logs_error_never_raises(caplog):
    pipeline = _FailingIngestionPipeline()

    with caplog.at_level(logging.ERROR):
        _safe_extract(pipeline, "test", "c1", "d1", "about cats")  # must not raise

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "background extract failed" in error_records[0].getMessage()
    assert "c1" in error_records[0].getMessage()
    assert error_records[0].exc_info is not None
    assert "boom extracting c1" in str(error_records[0].exc_info[1])
