"""An in-memory stand-in for the few statements `sync`/`drift` issue, for offline tests.

It interprets by statement prefix (the Cypher itself is exercised live, `test_sync_live.py`): replace
statements store `(kind, meta, rows)` per path; the reads answer from that dict.
"""
from coordgraph import store
from coordgraph.sync import EXPORT, ORPHAN_SWEEPS


class Result:
    def __init__(self, rows=(), deleted=0):
        self.result_set = [list(r) for r in rows]
        self.nodes_deleted = deleted


class MemGraph:
    def __init__(self):
        self.sources: dict[str, tuple[str, dict, list[dict]]] = {}
        self.writes: list[str] = []  # paths replaced, in order
        self.sync_state: dict = {}
        self.fail_paths: set[str] = set()

    def query(self, q, params=None):
        params = params or {}
        for kind, stmt in store.REPLACE.items():
            if q == stmt:
                path = params["path"]
                if path in self.fail_paths:
                    raise RuntimeError("boom")
                self.sources[path] = (kind, dict(params["meta"]), [dict(r) for r in params["rows"]])
                self.writes.append(path)
                return Result([[len(params["rows"])]])
        if q == store.DELETE_SOURCE:
            gone = self.sources.pop(params["path"], None)
            return Result(deleted=0 if gone is None else 1 + len(gone[2]))
        if q.startswith("MATCH (s:Source) RETURN s.path, s.sha256, s.parserVersion"):
            return Result([[p, m["sha256"], m.get("parserVersion")] for p, (_, m, _) in sorted(self.sources.items())])
        if q.startswith("MATCH (s:Source) RETURN s.path, s.sha256, s.syncedAt"):
            return Result([[p, m["sha256"], m.get("syncedAt")] for p, (_, m, _) in sorted(self.sources.items())])
        if q.startswith("MATCH (s:Source) RETURN count(s)"):
            return Result([[len(self.sources)]])
        if q == EXPORT:
            return Result([[p, dict(m), [dict(r) for r in rows]] for p, (_, m, rows) in sorted(self.sources.items())])
        if q.startswith("MATCH (s:Source {path: $path})-[]->(c) RETURN properties(c)"):
            return Result([[dict(r)] for r in self.sources.get(params["path"], (0, 0, []))[2]])
        if q.startswith("MERGE (m:SyncState"):
            self.sync_state = dict(params)
            return Result()
        for _label, _rel, stmt in ORPHAN_SWEEPS:
            if q == stmt:
                return Result([[0]])
        raise AssertionError(f"unexpected statement: {q[:80]}")

