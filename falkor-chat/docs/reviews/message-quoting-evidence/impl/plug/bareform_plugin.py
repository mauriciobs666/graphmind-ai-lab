import pytest
from falkorchat import repository as R
_orig = R.Repository._graph
class _G:
    def __init__(s, g): s._g = g
    def __getattr__(s, n): return getattr(s._g, n)
    def query(s, q, params=None, *a, **k):
        if "UNWIND [$quotedId] AS qid " in q:
            q = q.replace("UNWIND [$quotedId] AS qid ", "").replace("{msgId: qid}", "{msgId: $quotedId}")
        return s._g.query(q, params, *a, **k)
def _patched(self, ws):
    return _G(_orig(self, ws))
R.Repository._graph = _patched
