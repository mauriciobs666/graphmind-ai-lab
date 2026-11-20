import collections
import json
from pathlib import Path

import pytest

from coordgraph.parse.ledger import ALIASES, VOCABULARY, status_token

CORPUS = Path(__file__).parent / "data" / "status_cells_2026-10-21.json"

# Every raw variant quoted in graph note 2.5 (and plan section 5), plus real cells seen since.
CASES = [
    ("accepted", "accepted"),
    ("**in-flight** — dispatched at 2026-10-01 10:00", "in-flight"),
    ("in flight", "in-flight"),
    ("**accepted — APPROVE WITH SUGGESTIONS**", "accepted"),
    ("✅ delivered 2026-07-28 (rev 2)", "delivered"),
    ("✅ done 2026-07-29 — all green", "delivered"),
    ("**paused** — question relayed", "paused"),
    ("**relayed**", "paused"),
    ("`queued`", "queued"),
    ("_gated_", "gated"),
    ("GATED", "gated"),
    ("superseded", "abandoned"),
    ("abandoned", "abandoned"),
    ("queued (after U2 gate, U1)", "queued"),
    ("⏳ in-flight", "in-flight"),
    # deliberately not aliased (note 2.5): they read unknown and a human fixes the cell
    ("blocked (on architect fix + re-gate)", "unknown"),
    ("teco-verified", "unknown"),
    ("✅ **teco-verified** on disk", "unknown"),
    ("accepted-checkpoint (closed — no further resumes)", "unknown"),
    ("**NEEDS CHANGES** (2 blockers)", "unknown"),
    ("✅ **approve with suggestions** — 0 blockers", "unknown"),
    ("✅ **PASS WITH DEFECTS** — 22/23", "unknown"),
    ("—", "unknown"),
    ("", "unknown"),
    ("   ", "unknown"),
    ("271k tok / 142 tools", "unknown"),
    ("`a93df6f4e93cf613f`", "unknown"),
]


@pytest.mark.parametrize("cell,token", CASES)
def test_status_token_from_real_cells(cell, token):
    assert status_token(cell) == token


def test_vocabulary_and_alias_table_are_exactly_the_documented_ones():
    assert VOCABULARY == ("queued", "in-flight", "delivered", "gated", "accepted", "abandoned", "paused")
    assert ALIASES == {"relayed": "paused", "done": "delivered", "superseded": "abandoned"}


def test_a_word_merely_starting_like_a_token_is_not_that_token():
    assert status_token("acceptedly") == "unknown"
    assert status_token("pausedness") == "unknown"
    assert status_token("inflight") == "unknown"


def test_corpus_of_frozen_status_cells():
    """A frozen copy of the status cells of every non-archive ledger at 2026-10-21 (986 cells).

    Pinned: no cell raises, the tally is the one the function produces, every cell that
    reads `paused` is a `**relayed**` (no ledger holds a bare `paused` cell), and every
    `unknown` belongs to an archived ledger (no non-archived ledger existed that day).
    """
    cells = json.loads(CORPUS.read_text())["cells"]
    assert len(cells) == 986
    tally = collections.Counter(status_token(c) for _, c in cells)
    assert tally == {
        "accepted": 429, "delivered": 364, "unknown": 104, "in-flight": 38,
        "gated": 25, "queued": 18, "paused": 5, "abandoned": 3,
    }
    assert {c for _, c in cells if status_token(c) == "paused"} == {"**relayed**"}
    assert all(archived for archived, c in cells if status_token(c) == "unknown")
