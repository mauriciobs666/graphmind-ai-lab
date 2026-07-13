"""Unit tests for the `@mention`→workflow trigger (U11, §6 ordered rule).

`WorkflowTrigger.maybe_trigger` routes an incoming message to **exactly one** of:
  1. nothing — the message is agent-authored (loop-guard);
  2. resume — this thread has a `waiting` run (no re-@mention required, §2.4);
  3. start — the message @mentions the agent and a def is configured;
  4. the held responder — no workflow applies (M2 direct-reply fall-through).

Everything is injected (stub services, stub responder) so the ordered rule is asserted
offline — no DB, no LLM, no executor.
"""

from __future__ import annotations

from falkorchat.config import CallContext
from falkorchat.trigger import WorkflowTrigger

CTX = CallContext(ws="test", actor="u1")


class StubServices:
    def __init__(self, *, waiting=None):
        self._waiting = waiting
        self.calls: list[tuple] = []

    def find_waiting_run_for_thread(self, ctx, *, thread_id):
        self.calls.append(("find_waiting", thread_id))
        return self._waiting

    def resume_workflow_run(self, ctx, *, run_id):
        self.calls.append(("resume", run_id))
        return {"runId": run_id, "status": "running"}

    def start_workflow_run(self, ctx, *, def_key, version, trigger_msg_id, trace):
        self.calls.append(("start", def_key, version, trigger_msg_id, trace))
        return {"runId": "new-run", "status": "waiting"}


class StubResponder:
    def __init__(self):
        self.calls: list[dict] = []

    def maybe_respond(self, ctx, *, thread_id, msg_id, text, role, channel_id, mentions):
        self.calls.append(
            {"thread_id": thread_id, "msg_id": msg_id, "text": text, "role": role,
             "channel_id": channel_id, "mentions": mentions}
        )
        return {"responded": True}


def _trigger(services, *, responder=None):
    return WorkflowTrigger(
        services, agent_id="assistant", def_key="triage", def_version="v1",
        responder=responder,
    )


# ── 1. loop-guard — agent-authored message ───────────────────────────────────

def test_agent_authored_message_is_a_noop_loop_guard():
    svc = StubServices()
    resp = StubResponder()
    out = _trigger(svc, responder=resp).maybe_trigger(
        CTX, thread_id="t1", msg_id="m1", text="hi", role="assistant",
        mentions=["assistant"],
    )
    assert out is None
    assert svc.calls == []        # never even looks for a waiting run
    assert resp.calls == []       # never falls through to the responder


# ── 2. resume-if-waiting — no re-@mention required ───────────────────────────

def test_waiting_run_resumes_without_a_re_mention():
    svc = StubServices(waiting={"runId": "r9", "status": "waiting"})
    resp = StubResponder()
    out = _trigger(svc, responder=resp).maybe_trigger(
        CTX, thread_id="t1", msg_id="m2", text="my username is bob", role="user",
        mentions=[],                       # NOTE: no @mention on the reply
    )
    assert out == {"runId": "r9", "status": "running"}
    assert ("resume", "r9") in svc.calls
    # resume short-circuits — neither start nor responder fires
    assert not any(c[0] == "start" for c in svc.calls)
    assert resp.calls == []


def test_waiting_run_takes_priority_even_when_mentioned():
    # a waiting run owns the thread's reply even if it re-mentions the agent —
    # resume-before-start (§6), never a second run.
    svc = StubServices(waiting={"runId": "r9", "status": "waiting"})
    out = _trigger(svc).maybe_trigger(
        CTX, thread_id="t1", msg_id="m2", text="hi @bot", role="user",
        mentions=["assistant"],
    )
    assert out["runId"] == "r9"
    assert not any(c[0] == "start" for c in svc.calls)


# ── 3. @mention-to-start — no waiting run, agent mentioned ───────────────────

def test_mention_starts_a_run():
    svc = StubServices(waiting=None)
    resp = StubResponder()
    out = _trigger(svc, responder=resp).maybe_trigger(
        CTX, thread_id="t1", msg_id="m3", text="@bot help", role="user",
        mentions=["assistant"],
    )
    assert out == {"runId": "new-run", "status": "waiting"}
    assert ("start", "triage", "v1", "m3", False) in svc.calls
    assert resp.calls == []       # start short-circuits the responder


# ── 4. fall-through to the responder — no workflow applies ───────────────────

def test_falls_through_to_responder_when_not_mentioned():
    svc = StubServices(waiting=None)
    resp = StubResponder()
    out = _trigger(svc, responder=resp).maybe_trigger(
        CTX, thread_id="t1", msg_id="m4", text="just chatting", role="user",
        mentions=["someone_else"],
    )
    assert out == {"responded": True}
    assert resp.calls[0]["text"] == "just chatting"
    assert resp.calls[0]["channel_id"] is None
    assert not any(c[0] == "start" for c in svc.calls)


def test_no_responder_and_no_workflow_is_a_noop():
    svc = StubServices(waiting=None)
    out = _trigger(svc, responder=None).maybe_trigger(
        CTX, thread_id="t1", msg_id="m5", text="hello", role="user", mentions=[],
    )
    assert out is None


def test_mention_without_def_key_falls_through_to_responder():
    # a trigger built without a def key cannot start — an @mention still routes to
    # the responder rather than silently no-op.
    svc = StubServices(waiting=None)
    resp = StubResponder()
    trig = WorkflowTrigger(
        svc, agent_id="assistant", def_key="", def_version="v1", responder=resp
    )
    out = trig.maybe_trigger(
        CTX, thread_id="t1", msg_id="m6", text="@bot", role="user",
        mentions=["assistant"],
    )
    assert out == {"responded": True}
    assert not any(c[0] == "start" for c in svc.calls)
