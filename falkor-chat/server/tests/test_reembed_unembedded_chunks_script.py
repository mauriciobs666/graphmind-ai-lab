"""Tests for `scripts/reembed_unembedded_chunks.py` (K-102 repair): re-embed Chunks
with no embedding through `EmbeddingWorker.embed_chunk`, dry-run by default, and
never touch `Document.status`."""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import reembed_unembedded_chunks as script  # noqa: E402

from falkorchat.embedding import EmbeddingWorker  # noqa: E402


class _FakeRepo:
    def __init__(self, chunks, failed=()):
        self.chunks = list(chunks)
        self.failed = list(failed)
        self.writes: list[str] = []  # any mutating call lands here

    def list_unembedded_chunks(self, ws):
        return [dict(c) for c in self.chunks]

    def list_failed_documents(self, ws):
        embedded = {c["chunkId"] for c in self.chunks}
        return [
            {**d, "unembeddedChunks": sum(
                1 for c in self.chunks if c["documentId"] == d["documentId"]
            )}
            for d in self.failed
        ]

    def report_document_job_done(self, *a, **k):  # must never be called
        self.writes.append("report_document_job_done")


class _FakeWorker:
    def __init__(self, repo, fail_ids=()):
        self.repo = repo
        self.calls: list[tuple[str, str, str]] = []
        self.fail_ids = set(fail_ids)

    def embed_chunk(self, ws, *, chunk_id, text):
        self.calls.append((ws, chunk_id, text))
        if chunk_id in self.fail_ids:
            raise RuntimeError("provider down")
        self.repo.chunks = [c for c in self.repo.chunks if c["chunkId"] != chunk_id]
        return [0.0]


def _chunk(cid, doc, status="failed"):
    return {"chunkId": cid, "text": f"text-{cid}", "documentId": doc,
            "documentTitle": f"T-{doc}", "documentStatus": status}


def test_dry_run_lists_chunks_and_makes_no_embed_call(capsys):
    repo = _FakeRepo([_chunk("c1", "d1"), _chunk("c2", "d2")],
                     failed=[{"documentId": "d1", "documentTitle": "T-d1"}])
    worker = _FakeWorker(repo)

    result = script.run("agent-team", repo=repo, worker=worker, apply=False)

    out = capsys.readouterr().out
    assert worker.calls == []
    assert "c1" in out and "c2" in out and "dry-run" in out.lower()
    assert result.found == 2 and result.embedded == 0


def test_apply_embeds_each_chunk_through_the_worker_and_never_reports_status(capsys):
    repo = _FakeRepo([_chunk("c1", "d1"), _chunk("c2", "d2")],
                     failed=[{"documentId": "d1", "documentTitle": "T-d1"},
                             {"documentId": "d2", "documentTitle": "T-d2"}])
    worker = _FakeWorker(repo)

    result = script.run("agent-team", repo=repo, worker=worker, apply=True)

    assert worker.calls == [("agent-team", "c1", "text-c1"), ("agent-team", "c2", "text-c2")]
    assert repo.writes == []
    assert result.embedded == 2 and result.errors == []
    out = capsys.readouterr().out
    # both documents are still 'failed' (status untouched) but now fully embedded
    assert "d1" in out and "d2" in out and "still 'failed'" in out


def test_apply_continues_past_a_failing_chunk_and_reports_it(capsys):
    repo = _FakeRepo([_chunk("c1", "d1"), _chunk("c2", "d2")],
                     failed=[{"documentId": "d1", "documentTitle": "T-d1"},
                             {"documentId": "d2", "documentTitle": "T-d2"}])
    worker = _FakeWorker(repo, fail_ids={"c1"})

    result = script.run("agent-team", repo=repo, worker=worker, apply=True)

    assert [c[1] for c in worker.calls] == ["c1", "c2"]
    assert result.embedded == 1
    assert [e[0] for e in result.errors] == ["c1"]
    assert "still unembedded" in capsys.readouterr().out


def test_main_defaults_to_dry_run_and_apply_flag_enables_writes(monkeypatch):
    seen = {}

    def fake_run(ws, *, repo, worker, apply):
        seen["apply"] = apply
        return script.RunResult(found=0, embedded=0, errors=[])

    monkeypatch.setattr(script, "run", fake_run)
    monkeypatch.setattr(script, "_build", lambda ws: (object(), object()))

    assert script.main(["agent-team"]) == 0
    assert seen["apply"] is False
    assert script.main(["agent-team", "--apply"]) == 0
    assert seen["apply"] is True


def test_live_repair_embeds_chunk_without_changing_document_status(repo, conn):
    """Real repository + real EmbeddingWorker (fake embedder): the chunk gets its
    embedding, the dimension guard is the repository's, and the failed document stays failed."""
    from conftest import TEST_EMBEDDING_DIM

    repo.ensure_user("test", user_id="u1", display_name="A")
    repo.create_document(
        "test", document_id="d1", title="T", text="x", source_format="text",
        ingested_by="u1", created_at=1,
        chunks=[{"chunkId": "c0", "text": "hello", "seq": 0}],
    )
    repo.start_document_progress("test", document_id="d1", total_jobs=1)
    repo.report_document_job_done("test", document_id="d1", success=False)

    class _Embedder:
        def embed(self, text):
            return [0.25] * TEST_EMBEDDING_DIM

    worker = EmbeddingWorker(repo, _Embedder(), expected_dim=TEST_EMBEDDING_DIM)

    result = script.run("test", repo=repo, worker=worker, apply=True)

    assert result.embedded == 1
    assert repo.list_unembedded_chunks("test") == []
    assert repo.get_document("test", document_id="d1")["status"] == "failed"
