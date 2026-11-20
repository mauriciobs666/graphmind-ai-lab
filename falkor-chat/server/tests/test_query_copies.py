"""Drift guard for `scripts/test_queries.sh`'s hand copies of shipped queries
(K-091 F5).

A `test_queries.sh` assertion runs a hand copy of a repository query, so it pins
the copy, not the method; the update-detection section's copy of
`create_document_with_auto_supersede` had already drifted once (it lacked
`ingestorIsUser`). Each guarded copy here is compared, whitespace-normalized, to
the text the shipped method actually sends, captured with a recording graph —
no FalkorDB needed.

Scope: the update-detection copies (QUERIES §14.9) that each render from one
shipped method. The rest of the file's copies predate this guard.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from falkorchat.repository import Repository

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test_queries.sh"


def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def shell_copy(name: str, script_text: str) -> str:
    """The value bash assigns to `NAME` in `test_queries.sh` (K-091 F6).

    Exactly one `^NAME=` line must exist — a second one silently replaces the
    first for every later use — and that line is evaluated by bash itself, so
    any quoting slip (an unescaped `$` inside double quotes) shows up as the
    text the suite really sends. Each guarded definition is a self-contained
    literal, so it evaluates without the rest of the script."""
    lines = [ln for ln in script_text.splitlines() if ln.startswith(f"{name}=")]
    assert len(lines) == 1, (
        f"expected exactly one {name}= definition in {_SCRIPT.name}, found {len(lines)}"
    )
    done = subprocess.run(
        ["bash", "-c", f'{lines[0]}\nprintf %s "${name}"'],
        capture_output=True, text=True, check=True, env={"PATH": os.environ["PATH"]},
    )
    return done.stdout


class _Recorder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def query(self, q: str, params: Any = None):
        self.texts.append(q)
        return _Result()

    ro_query = query


class _Result:
    result_set = [[True, "d", False, None, None]]


def _rendered(call) -> list[str]:
    recorder = _Recorder()
    repo = Repository.__new__(Repository)
    repo._graph = lambda ws: recorder  # type: ignore[method-assign]
    call(repo)
    return [_normalize(t) for t in recorder.texts]


_BANDS = [f"b{i}" for i in range(8)]

_SHIPPED = {
    "CREATE_DOC_WITH_AUTO_SUPERSEDE": lambda: _rendered(
        lambda r: r.create_document_with_auto_supersede(
            "w", document_id="d", title="t", text="x", text_normalized_hash="h",
            source_format="text", ingested_by="u1", created_at=1, chunks=[],
            match_id="m", lsh_bands=_BANDS,
        )
    )[0],
    "FIND_SHORTLIST_BANDS": lambda: _rendered(
        lambda r: r.find_update_shortlist("w", bands=_BANDS, title="Some Title")
    )[0],
    "FIND_SHORTLIST_TITLE": lambda: _rendered(
        lambda r: r.find_update_shortlist("w", bands=_BANDS, title="Some Title")
    )[1],
    "LIST_MISSING_BANDS": lambda: _rendered(
        lambda r: r.list_documents_missing_lsh_bands("w")
    )[0],
    "SET_DOC_BANDS": lambda: _rendered(
        lambda r: r.set_document_lsh_bands("w", document_id="d", lsh_bands=_BANDS)
    )[0],
}


@pytest.mark.parametrize("name", sorted(_SHIPPED))
def test_test_queries_copy_matches_the_shipped_query(name):
    assert _normalize(shell_copy(name, _SCRIPT.read_text())) == _SHIPPED[name]()


def test_a_drifted_copy_is_detected():
    """Positive control: the pre-K-091 copy of the ingest write (no
    `ingestorIsUser`, no bands) does not match the shipped query."""
    stale = (
        "CREATE_DOC_WITH_AUTO_SUPERSEDE='OPTIONAL MATCH (u:User {userId: $ingestedBy}) "
        "OPTIONAL MATCH (a:Agent {agentId: $ingestedBy}) WITH u, a, coalesce(u, a) AS "
        "ingestor, (coalesce(u, a) IS NOT NULL) AS ok RETURN ok'"
    )
    assert _normalize(shell_copy("CREATE_DOC_WITH_AUTO_SUPERSEDE", stale)) != (
        _SHIPPED["CREATE_DOC_WITH_AUTO_SUPERSEDE"]()
    )


def test_the_band_or_shortlist_read_is_written_only_in_its_variable():
    """K-091 F6: an inline band-OR shortlist read is a copy no variable guards;
    every use must go through `FIND_SHORTLIST_BANDS`."""
    body = re.compile(r"d\.lshBand0 = \\?\$band0")
    stray = [
        n for n, line in enumerate(_SCRIPT.read_text().splitlines(), 1)
        if body.search(line) and not line.startswith("FIND_SHORTLIST_BANDS=")
    ]
    assert stray == [], f"inline band-OR read at test_queries.sh line(s) {stray}"
