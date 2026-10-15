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
    _safe_detect_update,
    _safe_embed_chunk,
    _safe_extract,
    _safe_fuse,
    _safe_respond,
    _schedule_chunk_processing,
    _schedule_update_detection,
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


# ── _safe_detect_update / _schedule_update_detection (document-ingestion2
# Stage D, FR-2/AC-2) ─────────────────────────────────────────────────────
#
# Mirrors `_safe_extract`'s failure-isolation contract, but deliberately
# WITHOUT any `_report_document_job` interaction (a detection failure is a
# soft failure, never flips `Document.status`) — so there is no repo.repo
# accessor here at all, just a plain fake repository exposing the three
# methods `_safe_detect_update` actually calls.


class _RecordingUpdateRepo:
    def __init__(self, *, document=None, shortlist=None):
        self.document = document
        self.shortlist = shortlist if shortlist is not None else []
        self.suggestion_calls: list[dict] = []
        self.shortlist_calls: list[dict] = []

    def get_document(self, ws, *, document_id):
        return self.document

    def find_update_shortlist(self, ws, *, bands, title, limit=5):
        self.shortlist_calls.append({"ws": ws, "bands": bands, "title": title})
        return self.shortlist

    def create_or_reopen_supersede_suggestion(
        self, ws, *, new_document_id, candidate_document_id, match_id,
        status, confidence, technique, created_at,
    ):
        self.suggestion_calls.append({
            "ws": ws, "new_document_id": new_document_id,
            "candidate_document_id": candidate_document_id, "status": status,
            "confidence": confidence, "technique": technique,
        })
        return {"created": True, "reopened": False, "matchId": match_id, "status": status}


def test_safe_detect_update_no_document_is_a_noop(caplog):
    repo = _RecordingUpdateRepo(document=None)

    with caplog.at_level(logging.ERROR):
        _safe_detect_update(repo, "test", "d1")  # must not raise

    assert repo.suggestion_calls == []
    assert [r for r in caplog.records if r.levelno == logging.ERROR] == []


def test_safe_detect_update_no_candidates_writes_no_suggestion():
    repo = _RecordingUpdateRepo(
        document={"documentId": "d1", "title": "t", "text": "hello world"},
        shortlist=[],
    )

    _safe_detect_update(repo, "test", "d1")

    assert repo.suggestion_calls == []


def test_safe_detect_update_skips_the_document_itself_as_a_candidate():
    text = "one two three four five six seven"
    repo = _RecordingUpdateRepo(
        document={"documentId": "d1", "title": "t", "text": text},
        shortlist=[{"documentId": "d1", "title": "t", "text": text}],
    )

    _safe_detect_update(repo, "test", "d1")

    assert repo.suggestion_calls == []  # d1 can't be its own candidate


def test_safe_detect_update_writes_a_pending_suggestion_for_the_best_candidate():
    text = " ".join(f"word{i}" for i in range(50))
    near_duplicate = text + " one extra trailing word"
    unrelated = "completely different content that shares nothing at all"
    repo = _RecordingUpdateRepo(
        document={"documentId": "d1", "title": "t", "text": text},
        shortlist=[
            {"documentId": "d2", "title": "t2", "text": unrelated},
            {"documentId": "d3", "title": "t3", "text": near_duplicate},
        ],
    )

    _safe_detect_update(repo, "test", "d1")

    assert len(repo.suggestion_calls) == 1
    call = repo.suggestion_calls[0]
    assert call["new_document_id"] == "d1"
    assert call["candidate_document_id"] == "d3"  # the near-duplicate, not the unrelated one
    assert call["status"] == "pending"
    assert call["technique"] == "shingled_jaccard_overlap"
    assert 0.0 < call["confidence"] <= 1.0


def test_safe_detect_update_below_noise_floor_writes_no_suggestion():
    repo = _RecordingUpdateRepo(
        document={"documentId": "d1", "title": "t", "text": "alpha beta gamma delta epsilon"},
        shortlist=[
            {"documentId": "d2", "title": "t2", "text": "zulu yankee xray whiskey victor"},
        ],
    )

    _safe_detect_update(repo, "test", "d1")

    assert repo.suggestion_calls == []  # zero overlap — well below the noise floor


def test_safe_detect_update_passes_lsh_bands_and_title_to_the_shortlist_lookup():
    repo = _RecordingUpdateRepo(
        document={"documentId": "d1", "title": "My Title", "text": "some content"},
        shortlist=[],
    )

    _safe_detect_update(repo, "test", "d1")

    assert len(repo.shortlist_calls) == 1
    assert repo.shortlist_calls[0]["ws"] == "test"
    assert repo.shortlist_calls[0]["title"] == "My Title"
    assert isinstance(repo.shortlist_calls[0]["bands"], list)
    assert len(repo.shortlist_calls[0]["bands"]) == 8


def test_safe_detect_update_swallows_get_document_failure_logs_error_never_raises(caplog):
    class _FailingRepo:
        def get_document(self, ws, *, document_id):
            raise RuntimeError(f"boom fetching {document_id}")

    with caplog.at_level(logging.ERROR):
        _safe_detect_update(_FailingRepo(), "test", "d1")  # must not raise

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "background update-detection failed" in error_records[0].getMessage()
    assert "d1" in error_records[0].getMessage()
    assert error_records[0].exc_info is not None
    assert "boom fetching d1" in str(error_records[0].exc_info[1])


def test_safe_detect_update_swallows_shortlist_failure_logs_error_never_raises(caplog):
    class _FailingShortlistRepo(_RecordingUpdateRepo):
        def find_update_shortlist(self, ws, *, bands, title, limit=5):
            raise RuntimeError("boom shortlisting")

    repo = _FailingShortlistRepo(
        document={"documentId": "d1", "title": "t", "text": "some content"}
    )

    with caplog.at_level(logging.ERROR):
        _safe_detect_update(repo, "test", "d1")  # must not raise

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "background update-detection failed" in error_records[0].getMessage()


def test_safe_detect_update_swallows_suggestion_write_failure_logs_error_never_raises(caplog):
    text = " ".join(f"word{i}" for i in range(50))

    class _FailingSuggestionRepo(_RecordingUpdateRepo):
        def create_or_reopen_supersede_suggestion(self, *a, **kw):
            raise RuntimeError("boom writing suggestion")

    repo = _FailingSuggestionRepo(
        document={"documentId": "d1", "title": "t", "text": text},
        shortlist=[{"documentId": "d2", "title": "t2", "text": text}],
    )

    with caplog.at_level(logging.ERROR):
        _safe_detect_update(repo, "test", "d1")  # must not raise

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "background update-detection failed" in error_records[0].getMessage()


def test_schedule_update_detection_schedules_safe_detect_update_with_repo():
    scheduled: list[tuple] = []

    def _recording_schedule(fn, *args):
        scheduled.append((fn, args))

    repo = _RecordingUpdateRepo(document=None)

    _schedule_update_detection(_recording_schedule, repo, "test", "d1")

    assert scheduled == [(_safe_detect_update, (repo, "test", "d1"))]


def test_schedule_update_detection_with_no_repo_schedules_nothing():
    scheduled: list[tuple] = []

    def _recording_schedule(fn, *args):
        scheduled.append((fn, args))

    _schedule_update_detection(_recording_schedule, None, "test", "d1")

    assert scheduled == []


# ── K-067 step 2: the extract retry seen from `_safe_extract` (plan
#    `falkor-chat/docs/plans/extract-transient-retry.md` §3.3/§3.5) ──────────────
#
# U-9c is offline (fakes). I-1 and I-4 run against the live `ws:test` graph via
# the `repo`/`conn` fixtures (which wipe only `ws:test`).

import json  # noqa: E402

from falkorchat import db  # noqa: E402
from falkorchat.ingestion import IngestionPipeline  # noqa: E402
from falkorchat.services import Services  # noqa: E402
from falkorchat.transport import ProviderCallError  # noqa: E402


def _transient_503():
    return ProviderCallError("p/m @ url: HTTP 503", kind="http", status=503)


def _extraction_reply(person: str, org: str) -> str:
    # One entity + one relationship whose object is stub-repaired → 2 entities.
    return json.dumps({
        "entities": [{"name": person, "type": "Person"}],
        "relationships": [{"subject": person, "predicate": "founded", "object": org}],
    })


class _CountingScriptedLLM:
    """`complete` pops the next outcome (exception → raised, str → returned);
    the last outcome repeats once the script runs out."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _RecordingSleep:
    def __init__(self, on_sleep=None):
        self.delays: list[float] = []
        self._on_sleep = on_sleep

    def __call__(self, delay):
        self.delays.append(delay)
        if self._on_sleep is not None:
            self._on_sleep()


class _WritePhase503Repo(_RecordingProgressRepo):
    """U-9c: gate open; the first graph write raises a transient-classified 503 —
    a stand-in for a future provider call inside the write loop."""

    def __init__(self):
        super().__init__()
        self.write_attempts = 0

    def chunk_document_is_processing(self, ws, *, chunk_id):
        return True

    def create_entity_with_auto_match(self, ws, **kwargs):
        self.write_attempts += 1
        raise ProviderCallError("x", kind="http", status=503)


def test_u9c_safe_extract_never_retries_a_write_phase_provider_error():
    repo = _WritePhase503Repo()
    llm = _CountingScriptedLLM(_extraction_reply("Alice", "Acme"))
    sleep = _RecordingSleep()
    pipeline = IngestionPipeline(repo, llm, sleep=sleep)

    _safe_extract(pipeline, "ws", "chunk-7", "doc-3", "Alice founded Acme.")

    assert llm.calls == 1
    assert repo.write_attempts == 1
    assert sleep.delays == []
    assert repo.calls == [("ws", "doc-3", False)]  # reported exactly once, as a failure


# ── live helpers (ws:test) ──


def _ingest_one_chunk(repo, text: str) -> tuple[str, str]:
    repo.ensure_user("test", user_id="u1", display_name="Alice")
    receipt = Services(repo).ingest_document(CTX, text=text)
    document_id = receipt["documentId"]
    [chunk] = repo.list_document_chunks("test", document_id=document_id)
    return document_id, chunk["chunkId"]


def _document_progress(conn, document_id: str) -> tuple[str, int]:
    [[status, pending]] = db.workspace_graph(conn, "test").ro_query(
        "MATCH (d:Document {documentId: $id}) RETURN d.status, d.pendingJobs",
        {"id": document_id},
    ).result_set
    return status, pending


def _document_graph_counts(conn, document_id: str) -> dict[str, int]:
    g = db.workspace_graph(conn, "test")
    [[entities]] = g.ro_query(
        "MATCH (:Document {documentId: $id})-[:HAS_CHUNK]->(:Chunk)-[:ABOUT]->(e:Entity) "
        "RETURN count(DISTINCT e)", {"id": document_id},
    ).result_set
    [[about]] = g.ro_query(
        "MATCH (:Document {documentId: $id})-[:HAS_CHUNK]->(:Chunk)-[a:ABOUT]->(:Entity) "
        "RETURN count(a)", {"id": document_id},
    ).result_set
    [[relates]] = g.ro_query(
        "MATCH ()-[r:RELATES_TO {sourceDocumentId: $id}]->() RETURN count(r)",
        {"id": document_id},
    ).result_set
    [[same_as]] = g.ro_query(
        "MATCH (:Document {documentId: $id})-[:HAS_CHUNK]->(:Chunk)-[:ABOUT]->(e:Entity) "
        "MATCH (e)-[s:SAME_AS {status: 'confirmed'}]->(:Entity) RETURN count(s)",
        {"id": document_id},
    ).result_set
    return {"entities": entities, "about": about, "relates_to": relates,
            "same_as_confirmed": same_as}


def _orphan_entity_count(conn) -> int:
    [[orphans]] = db.workspace_graph(conn, "test").ro_query(
        "MATCH (e:Entity) OPTIONAL MATCH (:Chunk)-[a:ABOUT]->(e) "
        "WITH e, count(a) AS n WHERE n = 0 RETURN count(e)"
    ).result_set
    return orphans


# ── I-1: idempotency — N failed attempts + 1 success == 1 clean success ──


def test_i1_retried_extract_leaves_exactly_the_graph_of_a_clean_extract(repo, conn):
    # Distinct texts and entity names per document: identical text would make B
    # auto-supersede A, and shared names would make exact-tier SAME_AS
    # order-dependent — neither is what this test is about.
    doc_a, chunk_a = _ingest_one_chunk(repo, "Alice founded Acme.")
    doc_b, chunk_b = _ingest_one_chunk(repo, "Bruno founded Bolt.")
    repo.start_document_progress("test", document_id=doc_a, total_jobs=1)
    repo.start_document_progress("test", document_id=doc_b, total_jobs=1)

    llm_a = _CountingScriptedLLM(
        _transient_503(), _transient_503(), _extraction_reply("Alice", "Acme"))
    sleep_a = _RecordingSleep()
    _safe_extract(IngestionPipeline(repo, llm_a, sleep=sleep_a),
                  "test", chunk_a, doc_a, "Alice founded Acme.")

    llm_b = _CountingScriptedLLM(_extraction_reply("Bruno", "Bolt"))
    sleep_b = _RecordingSleep()
    _safe_extract(IngestionPipeline(repo, llm_b, sleep=sleep_b),
                  "test", chunk_b, doc_b, "Bruno founded Bolt.")

    assert (llm_a.calls, len(sleep_a.delays)) == (3, 2)
    assert (llm_b.calls, len(sleep_b.delays)) == (1, 0)
    assert _document_progress(conn, doc_a) == ("ready", 0)
    assert _document_progress(conn, doc_b) == ("ready", 0)
    counts_a = _document_graph_counts(conn, doc_a)
    counts_b = _document_graph_counts(conn, doc_b)
    assert counts_a == counts_b
    assert counts_a == {"entities": 2, "about": 2, "relates_to": 1, "same_as_confirmed": 0}
    assert _orphan_entity_count(conn) == 0


# ── I-4: pendingJobs/status during and after retries ──


def test_i4_a_retrying_job_is_still_outstanding_then_a_sibling_failure_fails_the_document(
    repo, conn,
):
    doc, chunk = _ingest_one_chunk(repo, "Carla founded Crux.")
    repo.start_document_progress("test", document_id=doc, total_jobs=2)
    seen_during_sleep: list[tuple[str, int]] = []
    sleep = _RecordingSleep(
        on_sleep=lambda: seen_during_sleep.append(_document_progress(conn, doc)))
    llm = _CountingScriptedLLM(_transient_503(), _extraction_reply("Carla", "Crux"))

    _safe_extract(IngestionPipeline(repo, llm, sleep=sleep),
                  "test", chunk, doc, "Carla founded Crux.")

    assert seen_during_sleep == [("processing", 2)]  # retrying ≠ reported
    assert _document_progress(conn, doc) == ("processing", 1)

    _safe_embed_chunk(_FailingChunkWorker(repo=repo), "test", doc, chunk, "x")

    assert _document_progress(conn, doc) == ("failed", 0)


def test_i4_exhausted_retries_fail_the_document_with_exactly_one_decrement(repo, conn):
    doc, chunk = _ingest_one_chunk(repo, "Dana founded Dyne.")
    repo.start_document_progress("test", document_id=doc, total_jobs=1)
    llm = _CountingScriptedLLM(_transient_503())
    sleep = _RecordingSleep()

    _safe_extract(IngestionPipeline(repo, llm, sleep=sleep),
                  "test", chunk, doc, "Dana founded Dyne.")

    assert llm.calls == 4
    assert len(sleep.delays) == 3
    assert _document_progress(conn, doc) == ("failed", 0)
    assert _orphan_entity_count(conn) == 0


def test_i4_a_sibling_failure_during_the_sleep_stops_the_retry_at_the_real_gate(repo, conn):
    # Not a plan row: pins the real `chunk_document_is_processing` gate end to end
    # (the post-sleep read against live FalkorDB), not only at the repository altitude.
    doc, chunk = _ingest_one_chunk(repo, "Erin founded Ergo.")
    repo.start_document_progress("test", document_id=doc, total_jobs=2)
    sleep = _RecordingSleep(on_sleep=lambda: repo.report_document_job_done(
        "test", document_id=doc, success=False))  # the sibling embed job gives up
    llm = _CountingScriptedLLM(_transient_503(), _extraction_reply("Erin", "Ergo"))

    _safe_extract(IngestionPipeline(repo, llm, sleep=sleep),
                  "test", chunk, doc, "Erin founded Ergo.")

    assert llm.calls == 1  # no attempt 2 once the document is already failed
    assert len(sleep.delays) == 1
    assert _document_progress(conn, doc) == ("failed", 0)
    assert _document_graph_counts(conn, doc)["entities"] == 0


# ── K-067 step 3: the chunk-embed retry seen from `_safe_embed_chunk` (plan
#    `falkor-chat/docs/plans/extract-transient-retry.md` §3.4/§3.5, row I-3) ──────

import pytest  # noqa: E402

from falkorchat.embedding import EmbeddingWorker  # noqa: E402


class _ScriptedEmbedder:
    """`embed` pops the next outcome (exception → raised, vector → returned); the
    last outcome repeats once the script runs out. `seen` records each input."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.seen: list[str] = []

    def embed(self, text):
        self.seen.append(text)
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return list(outcome)


def _chunk_embedding(conn, chunk_id: str):
    [[embedding]] = db.workspace_graph(conn, "test").ro_query(
        "MATCH (c:Chunk {chunkId: $id}) RETURN c.embedding", {"id": chunk_id},
    ).result_set
    return embedding


def test_i3_a_retried_chunk_embed_lands_its_vector_and_readies_the_document(repo, conn):
    # Two documents: only A's job runs, so a gate or write keyed on the wrong
    # chunk/document would either stop A's retry or touch B.
    doc_a, chunk_a = _ingest_one_chunk(repo, "Fabio founded Flux in Lisbon.")
    doc_b, chunk_b = _ingest_one_chunk(repo, "Gina founded Gale in Braga.")
    repo.start_document_progress("test", document_id=doc_a, total_jobs=1)
    repo.start_document_progress("test", document_id=doc_b, total_jobs=1)
    vector = [0.5, 0.25, 0.125, 1.0]  # `ws:test`'s vector index is dim 4
    embedder = _ScriptedEmbedder(_transient_503(), vector)
    sleep = _RecordingSleep()
    worker = EmbeddingWorker(repo, embedder, expected_dim=4, sleep=sleep)

    _safe_embed_chunk(worker, "test", doc_a, chunk_a, "Fabio founded Flux in Lisbon.")

    assert embedder.seen == ["Fabio founded Flux in Lisbon."] * 2
    assert len(sleep.delays) == 1
    assert _chunk_embedding(conn, chunk_a) == pytest.approx(vector)
    assert _document_progress(conn, doc_a) == ("ready", 0)
    assert _chunk_embedding(conn, chunk_b) is None
    assert _document_progress(conn, doc_b) == ("processing", 1)
