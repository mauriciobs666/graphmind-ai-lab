"""Tests for `scripts/backfill_document_lsh_bands.py` (K-091 one-off backfill):
stamp `Document.lshBand0..7` on documents ingested before the ingest write stored
them. Dry run by default; writes only with `--apply`; idempotent."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import backfill_document_lsh_bands as script  # noqa: E402

from falkorchat import update_detection as ud  # noqa: E402


def _bands_of(text):
    return ud.lsh_bands(ud.minhash_signature(ud.shingles(ud.normalize_text(text))))


class _FakeRepo:
    def __init__(self, docs):
        self.docs = {d["documentId"]: dict(d) for d in docs}
        self.writes: list[tuple[str, str, list[str]]] = []

    def list_documents_missing_lsh_bands(self, ws):
        return [
            {"documentId": d["documentId"], "title": d["title"], "text": d["text"]}
            for d in self.docs.values() if d.get("lshBands") is None
        ]

    def set_document_lsh_bands(self, ws, *, document_id, lsh_bands):
        self.writes.append((ws, document_id, lsh_bands))
        self.docs[document_id]["lshBands"] = lsh_bands
        return True


def _doc(did, text="some text here"):
    return {"documentId": did, "title": f"T-{did}", "text": text}


def test_dry_run_lists_band_less_documents_and_writes_nothing(capsys):
    repo = _FakeRepo([_doc("d1"), _doc("d2")])

    result = script.run("probe", repo=repo, apply=False)

    out = capsys.readouterr().out
    assert repo.writes == []
    assert "d1" in out and "d2" in out and "dry-run" in out.lower()
    assert result.found == 2 and result.written == 0


def test_apply_writes_the_bands_computed_from_each_documents_text():
    repo = _FakeRepo([_doc("d1", "alpha beta gamma delta epsilon zeta"), _doc("d2", "other")])

    result = script.run("probe", repo=repo, apply=True)

    assert repo.writes == [
        ("probe", "d1", _bands_of("alpha beta gamma delta epsilon zeta")),
        ("probe", "d2", _bands_of("other")),
    ]
    assert result.found == 2 and result.written == 2


def test_apply_bands_come_from_the_normalized_text_the_ingest_path_uses():
    """K-091: the backfill must give an old document the same bands ingest
    would have stored — `normalize_text` first. The text here is one
    normalization changes, so raw-text shingling is distinguishable."""
    text = "Legacy   DOCUMENT body\nWith  Mixed CASE and   gaps in it."
    assert ud.normalize_text(text) != text
    explicit = ud.lsh_bands(ud.minhash_signature(ud.shingles(ud.normalize_text(text))))
    assert explicit != ud.lsh_bands(ud.minhash_signature(ud.shingles(text)))
    repo = _FakeRepo([_doc("d1", text)])

    script.run("probe", repo=repo, apply=True)

    assert repo.writes == [("probe", "d1", explicit)]


def test_apply_is_idempotent_a_second_run_finds_nothing():
    repo = _FakeRepo([_doc("d1")])
    script.run("probe", repo=repo, apply=True)

    again = script.run("probe", repo=repo, apply=True)

    assert again.found == 0 and again.written == 0
    assert len(repo.writes) == 1


def test_main_defaults_to_dry_run_and_takes_several_workspaces(monkeypatch):
    seen = []

    def fake_run(ws, *, repo, apply):
        seen.append((ws, apply))
        return script.RunResult(found=0, written=0)

    monkeypatch.setattr(script, "run", fake_run)
    monkeypatch.setattr(script, "_build", lambda: object())

    assert script.main(["a", "b"]) == 0
    assert seen == [("a", False), ("b", False)]
    seen.clear()
    assert script.main(["a", "--apply"]) == 0
    assert seen == [("a", True)]


def test_live_backfill_stamps_bands_only_on_band_less_documents(repo, conn):
    """Real repository: a pre-K-091 document (written by `create_document`,
    which stores no bands) gets the bands its text yields; a document the
    ingest write already banded is not listed; a second run finds nothing."""
    repo.ensure_user("test", user_id="u1", display_name="A")
    repo.create_document(
        "test", document_id="old", title="Old", text="legacy body text of the doc",
        source_format="text", ingested_by="u1", created_at=1,
        chunks=[{"chunkId": "c0", "text": "legacy body text of the doc", "seq": 0}],
    )
    repo.create_document_with_auto_supersede(
        "test", document_id="new", title="New", text="fresh body",
        text_normalized_hash="h-new", source_format="text", ingested_by="u1",
        created_at=2, chunks=[{"chunkId": "c1", "text": "fresh body", "seq": 0}],
        match_id="m1", lsh_bands=_bands_of("fresh body"),
    )

    assert [d["documentId"] for d in repo.list_documents_missing_lsh_bands("test")] == ["old"]

    result = script.run("test", repo=repo, apply=True)

    assert result.written == 1
    [row] = repo._graph("test").ro_query(
        "MATCH (d:Document {documentId:'old'}) RETURN "
        + ", ".join(f"d.lshBand{i}" for i in range(8))
    ).result_set
    assert row == _bands_of("legacy body text of the doc")
    assert script.run("test", repo=repo, apply=True).found == 0


def test_main_reports_a_failing_workspace_and_carries_on_with_the_rest(monkeypatch, capsys):
    """K-091 N2: a mistyped workspace (no such graph -> `ResponseError: Invalid
    graph operation on empty key`) is reported as one clean line, the other
    workspaces still run, and the exit code is 1."""
    from redis.exceptions import ResponseError

    seen = []

    def fake_run(ws, *, repo, apply):
        seen.append(ws)
        if ws == "nosuch":
            raise ResponseError("Invalid graph operation on empty key")
        return script.RunResult(found=0, written=0)

    monkeypatch.setattr(script, "run", fake_run)
    monkeypatch.setattr(script, "_build", lambda: object())

    assert script.main(["a", "nosuch", "b"]) == 1
    assert seen == ["a", "nosuch", "b"]
    err = capsys.readouterr().err
    assert "ws:nosuch" in err and "empty key" in err and "Traceback" not in err
    # K-091 N2 nit: the same catch sees a real write failure, so the line
    # names what failed without guessing why.
    assert "read or write failed" in err and "no such workspace" not in err
