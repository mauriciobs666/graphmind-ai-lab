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

from falkorchat.background import (
    _safe_embed_chunk,
    _safe_extract,
    _safe_fuse,
    _safe_respond,
    _schedule_chunk_processing,
)
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
    def __init__(self, repo=None):
        self.calls: list[tuple] = []
        self.repo = repo

    def embed_chunk(self, ws, *, chunk_id, text):
        self.calls.append((ws, chunk_id, text))


class _FailingChunkWorker:
    def __init__(self, repo=None):
        self.repo = repo

    def embed_chunk(self, ws, *, chunk_id, text):
        raise RuntimeError(f"boom embedding {chunk_id}")


def test_safe_embed_chunk_calls_the_worker():
    worker = _RecordingChunkWorker()

    _safe_embed_chunk(worker, "test", "d1", "c1", "about cats")

    assert worker.calls == [("test", "c1", "about cats")]


def test_safe_embed_chunk_swallows_failure_logs_error_never_raises(caplog):
    worker = _FailingChunkWorker()

    with caplog.at_level(logging.ERROR):
        _safe_embed_chunk(worker, "test", "d1", "c1", "about cats")  # must not raise

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "background chunk embed failed" in error_records[0].getMessage()
    assert "c1" in error_records[0].getMessage()
    assert error_records[0].exc_info is not None
    assert "boom embedding c1" in str(error_records[0].exc_info[1])


# ── _safe_embed_chunk / _safe_extract report completion back onto the owning
# Document (K-051) ────────────────────────────────────────────────────────
#
# `embed_worker.repo`/`ingestion_pipeline.repo` is where K-051 sources the
# repository to call `report_document_job_done` on — a worker/pipeline with
# no `.repo` (every fake elsewhere in this module and in test_api.py/
# test_mcp.py that predates K-051) must keep working exactly as before,
# silently skipping the report rather than raising.


class _RecordingProgressRepo:
    def __init__(self):
        self.calls: list[tuple] = []
        self.start_calls: list[tuple] = []

    def report_document_job_done(self, ws, *, document_id, success):
        self.calls.append((ws, document_id, success))

    def start_document_progress(self, ws, *, document_id, total_jobs):
        self.start_calls.append((ws, document_id, total_jobs))


class _RaisingProgressRepo:
    """`report_document_job_done` raises — pins that `_report_document_job`
    swallows a raising repo the same way every sibling `_safe_*` wrapper
    swallows a raising worker/pipeline (K-051 review MINOR 3)."""

    def report_document_job_done(self, ws, *, document_id, success):
        raise RuntimeError(f"boom reporting {document_id}")


def test_safe_embed_chunk_reports_success_to_the_workers_repo():
    progress_repo = _RecordingProgressRepo()
    worker = _RecordingChunkWorker(repo=progress_repo)

    _safe_embed_chunk(worker, "test", "d1", "c1", "about cats")

    assert progress_repo.calls == [("test", "d1", True)]


def test_safe_embed_chunk_reports_failure_to_the_workers_repo():
    progress_repo = _RecordingProgressRepo()
    worker = _FailingChunkWorker(repo=progress_repo)

    _safe_embed_chunk(worker, "test", "d1", "c1", "about cats")  # must not raise

    assert progress_repo.calls == [("test", "d1", False)]


def test_safe_embed_chunk_with_no_repo_on_the_worker_does_not_raise():
    worker = _RecordingChunkWorker()  # repo=None, the pre-K-051 default shape

    _safe_embed_chunk(worker, "test", "d1", "c1", "about cats")  # must not raise

    assert worker.calls == [("test", "c1", "about cats")]


def test_safe_embed_chunk_swallows_a_raising_repo_logs_error_never_raises(caplog):
    worker = _RecordingChunkWorker(repo=_RaisingProgressRepo())

    with caplog.at_level(logging.ERROR):
        _safe_embed_chunk(worker, "test", "d1", "c1", "about cats")  # must not raise

    assert worker.calls == [("test", "c1", "about cats")]  # the embed itself still ran
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "failed to report background embed completion" in error_records[0].getMessage()
    assert "c1" in error_records[0].getMessage()
    assert error_records[0].exc_info is not None
    assert "boom reporting d1" in str(error_records[0].exc_info[1])


# ── _safe_extract (K-050 M5 Stage 3) ─────────────────────────────────────────
#
# Mirrors `_safe_embed_chunk`'s failure-isolation contract exactly.


class _RecordingIngestionPipeline:
    def __init__(self, repo=None):
        self.calls: list[tuple] = []
        self.repo = repo

    def extract_chunk(self, ws, *, chunk_id, document_id, text):
        self.calls.append((ws, chunk_id, document_id, text))


class _FailingIngestionPipeline:
    def __init__(self, repo=None):
        self.repo = repo

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


def test_safe_extract_reports_success_to_the_pipelines_repo():
    progress_repo = _RecordingProgressRepo()
    pipeline = _RecordingIngestionPipeline(repo=progress_repo)

    _safe_extract(pipeline, "test", "c1", "d1", "about cats")

    assert progress_repo.calls == [("test", "d1", True)]


def test_safe_extract_reports_failure_to_the_pipelines_repo():
    progress_repo = _RecordingProgressRepo()
    pipeline = _FailingIngestionPipeline(repo=progress_repo)

    _safe_extract(pipeline, "test", "c1", "d1", "about cats")  # must not raise

    assert progress_repo.calls == [("test", "d1", False)]


def test_safe_extract_with_no_repo_on_the_pipeline_does_not_raise():
    pipeline = _RecordingIngestionPipeline()  # repo=None, the pre-K-051 default shape

    _safe_extract(pipeline, "test", "c1", "d1", "about cats")  # must not raise

    assert pipeline.calls == [("test", "c1", "d1", "about cats")]


def test_safe_extract_swallows_a_raising_repo_logs_error_never_raises(caplog):
    pipeline = _RecordingIngestionPipeline(repo=_RaisingProgressRepo())

    with caplog.at_level(logging.ERROR):
        _safe_extract(pipeline, "test", "c1", "d1", "about cats")  # must not raise

    assert pipeline.calls == [("test", "c1", "d1", "about cats")]  # the extract itself still ran
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "failed to report background extract completion" in error_records[0].getMessage()
    assert "c1" in error_records[0].getMessage()
    assert error_records[0].exc_info is not None
    assert "boom reporting d1" in str(error_records[0].exc_info[1])


# ── _safe_fuse (K-050 M5 Stage 4) ─────────────────────────────────────────────
#
# Mirrors `_safe_extract`'s failure-isolation contract exactly, but at
# per-ENTITY granularity — called from inside `IngestionPipeline.extract_
# chunk`'s own loop, not scheduled per-chunk by api.py/mcp.py (see this
# wrapper's docstring in background.py).


class _RecordingFuser:
    def __init__(self):
        self.calls: list[tuple] = []

    def fuse_entity(self, ws, entity_id, name, type):
        self.calls.append((ws, entity_id, name, type))


class _FailingFuser:
    def fuse_entity(self, ws, entity_id, name, type):
        raise RuntimeError(f"boom fusing {entity_id}")


def test_safe_fuse_calls_the_pipeline():
    pipeline = _RecordingFuser()

    _safe_fuse(pipeline, "test", "e1", "Acme", "Organization")

    assert pipeline.calls == [("test", "e1", "Acme", "Organization")]


def test_safe_fuse_swallows_failure_logs_error_never_raises(caplog):
    pipeline = _FailingFuser()

    with caplog.at_level(logging.ERROR):
        _safe_fuse(pipeline, "test", "e1", "Acme", "Organization")  # must not raise

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "background fuse failed" in error_records[0].getMessage()
    assert "e1" in error_records[0].getMessage()
    assert error_records[0].exc_info is not None
    assert "boom fusing e1" in str(error_records[0].exc_info[1])


# ── _schedule_chunk_processing initializes progress, including the
# zero-chunk edge (K-051 review MINOR 2) ─────────────────────────────────


def _noop_schedule(fn, *args):
    pass


def test_schedule_chunk_processing_initializes_progress_with_total_jobs():
    progress_repo = _RecordingProgressRepo()
    embed_worker = _RecordingChunkWorker(repo=progress_repo)
    pipeline = _RecordingIngestionPipeline(repo=progress_repo)
    chunks = [{"chunkId": "c1", "text": "a"}, {"chunkId": "c2", "text": "b"}]

    _schedule_chunk_processing(
        _noop_schedule, "test", "d1", chunks,
        embed_worker=embed_worker, ingestion_pipeline=pipeline,
    )

    # 2 chunks * 2 wired jobs (embed + extract) each = 4 outstanding jobs.
    assert progress_repo.start_calls == [("test", "d1", 4)]


def test_schedule_chunk_processing_with_zero_chunks_still_initializes_progress_to_zero():
    """K-051 review MINOR 2: a document with an empty chunk list must not be
    silently skipped — `start_document_progress` is what flips it straight
    to 'ready' when `total_jobs == 0`, and that only happens if this method
    actually calls it."""
    progress_repo = _RecordingProgressRepo()
    embed_worker = _RecordingChunkWorker(repo=progress_repo)

    _schedule_chunk_processing(
        _noop_schedule, "test", "d1", [],
        embed_worker=embed_worker, ingestion_pipeline=None,
    )

    assert progress_repo.start_calls == [("test", "d1", 0)]


def test_schedule_chunk_processing_with_no_repo_on_either_worker_does_not_raise():
    # Pre-K-051 fakes with no `.repo` at all — must keep working unmodified.
    embed_worker = _RecordingChunkWorker()
    pipeline = _RecordingIngestionPipeline()
    chunks = [{"chunkId": "c1", "text": "a"}]

    _schedule_chunk_processing(  # must not raise
        _noop_schedule, "test", "d1", chunks,
        embed_worker=embed_worker, ingestion_pipeline=pipeline,
    )
