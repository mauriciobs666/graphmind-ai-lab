"""Test doubles for the model-backed half of the front (P1-26 dispatch 2): :class:`FakeBroker`, a stand-in for ``ccg.broker.client.BrokerClient`` that answers the two
calls the front makes (``GET /api/v0/models``, ``POST /v1/chat/completions``) and records every run and request, and :func:`draft_reply`, an ``InterpretDraft`` as the model
would return it. No socket is opened anywhere."""
from __future__ import annotations

import json

from ccg.broker.client import BrokerError, BrokerUnavailable, Decision, Response, RunGrant

MODEL = {"id": "qwen/qwen3-4b-2507", "publisher": "qwen", "arch": "qwen3", "quantization": "Q4_K_M", "state": "loaded", "type": "llm", "loaded_context_length": 4096}
FINGERPRINT = "qwen/qwen3-4b-2507|qwen|qwen3|Q4_K_M|4096"


def draft_reply(direction="impact", groups=(("m.py", "file"), ("A", "symbol")), *, use_case="impact", scope=None, readings=()):
    """One group of ``groups`` spans (a flat tuple of ``(text, kindHint)``), or a tuple of such tuples for several groups."""
    if groups and isinstance(groups[0][0], str):
        groups = (groups,)
    return {"useCase": use_case, "direction": direction, "anchors": [{"spans": [{"text": t, "kindHint": k} for t, k in g]} for g in groups], "scope": scope,
            "readings": [{"useCase": u, "direction": d} for u, d in readings]}


class FakeBroker:
    """``replies``: the chat completions' message contents in order (a ``str``, or a ``dict`` encoded as JSON); the last one repeats. ``models``: the endpoint's model list."""

    def __init__(self, replies=(), *, models=(MODEL,), unavailable=False, open_error=None, allow=True, statuses=None):
        self.replies, self.models = list(replies), list(models)
        self.unavailable, self.open_error, self.allow = unavailable, open_error, allow
        self.statuses = list(statuses or [])         # per-request HTTP statuses (None entries: no completed exchange), consumed in order, then 200
        self.opens, self.closes, self.requests = [], [], []
        self._chat = 0

    def open_run(self, run_kind, system_id, codebases, override=None):
        if self.unavailable:
            raise BrokerUnavailable("no socket")
        if self.open_error:
            raise BrokerError(self.open_error)
        self.opens.append((run_kind, system_id, codebases, override))
        return RunGrant(f"run-{len(self.opens)}", "f" * 64, {}, False, "a" * 32)

    def close_run(self, token):
        self.closes.append(token)

    def request(self, token, request, body=None):
        if self.unavailable:
            raise BrokerUnavailable("no socket")
        self.requests.append((token, request, body))
        if not self.allow:
            return Decision("deny", "R6-STICKY-LOCAL", "sticky-local-only", "b" * 32), None
        status = self.statuses.pop(0) if self.statuses else 200
        decision = Decision("allow", "R1-LOCAL-ENDPOINT", None, "c" * 32)
        if status is None:
            return decision, None
        if request["target"]["path"] == "/api/v0/models":
            payload = {"data": self.models}
        else:
            reply = self.replies[min(self._chat, len(self.replies) - 1)]
            self._chat += 1
            payload = {"choices": [{"message": {"role": "assistant", "content": reply if isinstance(reply, str) else json.dumps(reply)}}]}
        return decision, Response(status, "application/json", json.dumps(payload).encode())

    @property
    def completions(self):
        return [r for r in self.requests if r[1]["target"]["path"] == "/v1/chat/completions"]
