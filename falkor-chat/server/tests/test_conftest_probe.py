"""Tests for the root `conftest.py`'s own probe machinery (K-046, filed out of
K-026 close — `docs/BACKLOG.md`).

`server/tests/conftest.py`'s `_falkordb_reachable()` is a mere reachability
probe used by the session-scoped `_schema` fixture. This file proves it never
materializes the graph key it probes as a side effect — mirroring the
analogous, already-fixed proof for the eval subtree's twin function
(`tests/eval/test_conftest_probe.py`, K-026 Unit 2b finding B-1).

Imported via `from conftest import _falkordb_reachable` (not a package path):
unlike `tests/eval/`, `tests/` has no `__init__.py`, so pytest's default
`--import-mode=prepend` loads `tests/conftest.py` as a top-level module
literally named `conftest` — the same pattern `test_tools.py`/`test_graphrag.py`
already use for `TEST_EMBEDDING_DIM`.
"""

from __future__ import annotations

from conftest import _falkordb_reachable
from falkorchat import db


def test_falkordb_reachable_probe_does_not_materialize_a_nonexistent_graph_key() -> None:
    ghost_ws = "k046-reachability-probe-does-not-exist"
    conn = db.connect()

    before = set(conn.list_graphs())
    assert f"ws:{ghost_ws}" not in before, (
        f"test precondition violated: ws:{ghost_ws} already exists — pick a "
        f"different probe name"
    )

    try:
        assert _falkordb_reachable(ghost_ws) is True, (
            "a genuinely nonexistent graph key should still read as "
            "'reachable' — the server responded, there's just no such graph "
            "yet"
        )

        after = set(conn.list_graphs())
        assert f"ws:{ghost_ws}" not in after, (
            f"_falkordb_reachable() materialized ws:{ghost_ws} as a side "
            f"effect of merely probing it"
        )
    finally:
        # Defensive cleanup: if the assertion above ever fails because the bug
        # is back, don't leave the junk empty graph key behind either way.
        if f"ws:{ghost_ws}" in set(conn.list_graphs()):
            conn.select_graph(f"ws:{ghost_ws}").delete()
