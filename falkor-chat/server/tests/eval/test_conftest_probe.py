"""Tests for `conftest.py`'s own probe machinery (K-026, `docs/reviews/graphrag-eval.md`
Unit 2b code/content review, finding B-1).

`conftest.py`'s module docstring states `ws_eval` is **probe-only**: it never
bootstraps, seeds, or writes — including its own reachability check. This file
proves that contract for `_falkordb_reachable()` specifically: probing a
genuinely nonexistent graph key must not leave that key behind in
`GRAPH.LIST` (mirrors the existing, analogous proof for
`Repository.read_index_dimension` in
`tests/test_graphrag.py::test_read_index_dimension_none_when_the_graph_key_does_not_exist`).

Imported via the `eval.conftest` package path (not a bare `import conftest`):
`tests/conftest.py` (root) is *also* imported as a top-level module literally
named `conftest` (no `tests/__init__.py` to qualify it), so a bare
`import conftest` here would silently resolve to the *root* conftest's own
`_falkordb_reachable()` — a different function testing a different bug — not
this directory's. `eval.conftest` is the same module object pytest itself
already loaded as this directory's conftest plugin (verified: no duplicate
import/side-effect execution).
"""

from __future__ import annotations

from falkorchat import db

from eval.conftest import _falkordb_reachable


def test_falkordb_reachable_probe_does_not_materialize_a_nonexistent_graph_key() -> None:
    ghost_ws = "b1-reachability-probe-does-not-exist"
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
            "yet; a more specific 'not seeded' skip reason comes from the "
            "dimension check that runs next in the ws_eval fixture"
        )

        after = set(conn.list_graphs())
        assert f"ws:{ghost_ws}" not in after, (
            f"_falkordb_reachable() materialized ws:{ghost_ws} as a side "
            f"effect of merely probing it — violates conftest.py's own "
            f"'probe-only, never writes' contract (B-1)"
        )
    finally:
        # Defensive cleanup: if the assertion above ever fails because the bug
        # is back, don't leave the junk empty graph key behind either way.
        if f"ws:{ghost_ws}" in set(conn.list_graphs()):
            conn.select_graph(f"ws:{ghost_ws}").delete()
