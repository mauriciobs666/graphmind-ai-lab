"""Integration test for Option B PRODUCED-emission linking (K-023, U11).

An integrated agent-node run (real `Services` + real `ToolRegistry`/`PostMessageTool`,
a stub tool-calling LLM, the live `ws:test` graph) must leave a real
`StepRun -[:PRODUCED]-> Message` edge keyed to the executing step's StepRun — the tool
posts the message and returns its msgId; the executor buffers it and links **after**
`record_step_and_advance` created the StepRun (Option B). No graph/QUERIES change.
"""

from __future__ import annotations

import itertools

from falkorchat import db
from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.llm import ChatResult, ToolCall
from falkorchat.services import Services
from falkorchat.tools import PostMessageTool, ToolRegistry

CTX = CallContext(ws="test", actor="u1")
WS = "test"


class ScriptedChatLLM:
    """intake → final text (advance); answer → post via `post_message` then final text."""

    def __init__(self):
        self._turns = [
            ChatResult(text="ready"),                       # intake: no tool → advance
            ChatResult(text="", tool_calls=[                # answer: post the reply
                ToolCall("c1", "post_message", {"text": "here is your answer"})]),
            ChatResult(text="done"),                        # answer: finish
        ]

    def chat(self, messages, tools):
        return self._turns.pop(0) if self._turns else ChatResult(text="(spent)")


INTAKE_STEP = {"key": "intake", "type": "agent", "config": "{}"}
ANSWER_STEP = {"key": "answer", "type": "agent",
               "config": '{"tools":["post_message"],"systemPrompt":"Answer."}'}
TRANSITIONS = [{"from": "intake", "to": "answer", "on": "ready", "guard": "", "order": 0}]


def _seed(repo):
    repo.ensure_user(WS, user_id="u1", display_name="Alice")
    repo.ensure_agent(WS, agent_id="assistant", name="Bot")
    repo.create_channel(WS, channel_id="c1", name="general", created_at=100)
    repo.create_thread(WS, channel_id="c1", thread_id="t1", title="x", created_at=110)
    repo.post_first_message(
        WS, thread_id="t1", msg_id="trig1", author_id="u1",
        text="please help", role="user", created_at=120,
    )
    repo.materialize_snapshot(
        WS, key="one", version="1", name="One", kind="conversation",
        start_key="intake", steps=[INTAKE_STEP, ANSWER_STEP], transitions=TRANSITIONS,
    )
    repo.start_run(
        WS, run_id="r1", def_key="one", def_version="1", started_at=1000,
        trigger_msg_id="trig1", ctx='{"threadId":"t1"}', trace=False, max_steps=12,
    )


def _executor(repo, services):
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(2000)
    registry = ToolRegistry([PostMessageTool(services, agent_id="assistant")])
    return WorkflowExecutor(
        services, repo, llm=ScriptedChatLLM(), tool_registry=registry,
        guard_judge=None, id_gen=lambda: next(ids), clock=lambda: next(clock),
    )


def test_integrated_agent_node_post_creates_produced_edge_live(wf_repo, conn):
    _seed(wf_repo)
    services = Services(
        wf_repo,
        clock=(lambda c=itertools.count(500): next(c)),
        id_gen=(lambda c=itertools.count(1): f"m{next(c)}"),
    )
    ex = _executor(wf_repo, services)

    status = ex.run(CTX, run_id="r1")

    assert status == "done"
    graph = db.workspace_graph(conn, WS)
    # exactly one StepRun-[:PRODUCED]->Message edge, keyed to the answer step's StepRun
    res = graph.query(
        "MATCH (r:WorkflowRun {runId: 'r1'})-[:HAS_STEP_RUN]->(sr:StepRun) "
        "MATCH (sr)-[:PRODUCED]->(m:Message) "
        "RETURN sr.stepKey AS stepKey, m.text AS text, count(*) AS n"
    )
    assert res.result_set, "no PRODUCED edge was created"
    step_key, text, n = res.result_set[0]
    assert step_key == "answer"
    assert text == "here is your answer"
    assert n == 1


class PlainTextAnswerLLM:
    """intake → final text (advance); answer → plain text, `post_message` NEVER called
    (the K-039 / mention-reply-delivery bug shape, live-reproduced with the real
    `qwen/qwen3-4b-2507`): the model was granted the tool and simply doesn't call it."""

    def __init__(self):
        self._turns = [
            ChatResult(text="ready"),                # intake: no tool → advance
            ChatResult(text="here is your answer"),  # answer: plain text, no tool call
        ]

    def chat(self, messages, tools):
        return self._turns.pop(0) if self._turns else ChatResult(text="(spent)")


def _executor_with_llm(repo, services, llm):
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(2000)
    registry = ToolRegistry([PostMessageTool(services, agent_id="assistant")])
    return WorkflowExecutor(
        services, repo, llm=llm, tool_registry=registry,
        guard_judge=None, id_gen=lambda: next(ids), clock=lambda: next(clock),
    )


def test_implicit_post_when_tool_not_called_still_creates_produced_edge_live(wf_repo, conn):
    # K-039 immediate mitigation: a node granted post_message that ends its turn on
    # plain text (never dispatching the tool) must still result in a real posted
    # Message + PRODUCED edge — not a silently discarded StepRun.output, which is
    # exactly the live-reproduced demo-blocking bug (mention-reply-delivery-rca.md §2).
    _seed(wf_repo)
    services = Services(
        wf_repo,
        clock=(lambda c=itertools.count(500): next(c)),
        id_gen=(lambda c=itertools.count(1): f"m{next(c)}"),
    )
    ex = _executor_with_llm(wf_repo, services, PlainTextAnswerLLM())

    status = ex.run(CTX, run_id="r1")

    assert status == "done"
    graph = db.workspace_graph(conn, WS)
    res = graph.query(
        "MATCH (r:WorkflowRun {runId: 'r1'})-[:HAS_STEP_RUN]->(sr:StepRun) "
        "MATCH (sr)-[:PRODUCED]->(m:Message) "
        "RETURN sr.stepKey AS stepKey, m.text AS text, count(*) AS n"
    )
    assert res.result_set, (
        "no PRODUCED edge was created — the model's plain-text reply was silently "
        "discarded (the exact live-reproduced K-039 failure)"
    )
    step_key, text, n = res.result_set[0]
    assert step_key == "answer"
    assert text == "here is your answer"
    assert n == 1


def test_link_gap_does_not_fail_run(wf_repo, conn, caplog):
    # A missing endpoint (link returns None) is a diagnosable gap logged, never fatal:
    # the run still completes. Simulate by making link_step_emission always miss.
    _seed(wf_repo)
    services = Services(
        wf_repo,
        clock=(lambda c=itertools.count(500): next(c)),
        id_gen=(lambda c=itertools.count(1): f"m{next(c)}"),
    )
    services.link_step_emission = (
        lambda ctx, *, step_run_id, msg_id, tools_used=(): None
    )
    ex = _executor(wf_repo, services)

    status = ex.run(CTX, run_id="r1")

    assert status == "done"          # a missing link never fails the run


# ── K-056 — `toolsUsed` is stamped end-to-end (audit trail only) ─────────────
#
# `Message.toolsUsed` is a pure audit/observability property (the replayed-history
# breadcrumb it originally fed was reverted — see `_assemble_messages`'s docstring
# and `docs/reviews/salesperson-tool-reliability-impl.md`, MAJOR 1). This drives a
# real node that dispatches a domain tool *and* posts, through the real
# executor→services→repository chain, and reads the resulting `Message.toolsUsed`
# back — not just that `_link_emissions` was *called*.

class FakeLookupTool:
    name = "lookup_product_fact"

    @property
    def schema(self):
        return {"type": "function",
                "function": {"name": "lookup_product_fact", "parameters": {}}}

    def run(self, arguments, *, ctx, run):
        return '{"name": "Wireless Mouse Pro", "price": 29.99}'


class LookupThenPostLLM:
    """answer → dispatch the domain tool, then post the reply, then finish."""

    def __init__(self):
        self._turns = [
            ChatResult(text="ready"),                       # intake: no tool → advance
            ChatResult(text="", tool_calls=[                 # answer: look it up
                ToolCall("c1", "lookup_product_fact", {"query": "Wireless Mouse Pro"})]),
            ChatResult(text="", tool_calls=[                 # answer: post the reply
                ToolCall("c2", "post_message", {"text": "It's $29.99."})]),
            ChatResult(text="done"),                         # answer: finish
        ]

    def chat(self, messages, tools):
        return self._turns.pop(0) if self._turns else ChatResult(text="(spent)")


ANSWER_STEP_WITH_LOOKUP = {
    "key": "answer", "type": "agent",
    "config": '{"tools":["lookup_product_fact","post_message"],"systemPrompt":"Answer."}',
}
TRANSITIONS_WITH_LOOKUP = [
    {"from": "intake", "to": "answer", "on": "ready", "guard": "", "order": 0}
]


def _seed_with_lookup(repo):
    repo.ensure_user(WS, user_id="u1", display_name="Alice")
    repo.ensure_agent(WS, agent_id="assistant", name="Bot")
    repo.create_channel(WS, channel_id="c1", name="general", created_at=100)
    repo.create_thread(WS, channel_id="c1", thread_id="t1", title="x", created_at=110)
    repo.post_first_message(
        WS, thread_id="t1", msg_id="trig1", author_id="u1",
        text="what's the price of the Wireless Mouse Pro?", role="user", created_at=120,
    )
    repo.materialize_snapshot(
        WS, key="two", version="1", name="Two", kind="conversation",
        start_key="intake", steps=[INTAKE_STEP, ANSWER_STEP_WITH_LOOKUP],
        transitions=TRANSITIONS_WITH_LOOKUP,
    )
    repo.start_run(
        WS, run_id="r1", def_key="two", def_version="1", started_at=1000,
        trigger_msg_id="trig1", ctx='{"threadId":"t1"}', trace=False, max_steps=12,
    )


def test_a_tool_backed_reply_is_stamped_with_the_tool_name_on_the_message(wf_repo, conn):
    _seed_with_lookup(wf_repo)
    services = Services(
        wf_repo,
        clock=(lambda c=itertools.count(500): next(c)),
        id_gen=(lambda c=itertools.count(1): f"m{next(c)}"),
    )
    registry = ToolRegistry([PostMessageTool(services, agent_id="assistant"),
                             FakeLookupTool()])
    ids = (f"sr{n}" for n in itertools.count(1))
    clock = itertools.count(2000)
    ex = WorkflowExecutor(
        services, wf_repo, llm=LookupThenPostLLM(), tool_registry=registry,
        guard_judge=None, id_gen=lambda: next(ids), clock=lambda: next(clock),
    )

    status = ex.run(CTX, run_id="r1")

    assert status == "done"
    graph = db.workspace_graph(conn, WS)
    res = graph.query(
        "MATCH (r:WorkflowRun {runId: 'r1'})-[:HAS_STEP_RUN]->(sr:StepRun {stepKey:'answer'}) "
        "MATCH (sr)-[:PRODUCED]->(m:Message) "
        "RETURN m.text AS text, m.toolsUsed AS toolsUsed"
    )
    assert res.result_set, "no PRODUCED edge was created"
    text, tools_used = res.result_set[0]
    assert text == "It's $29.99."
    assert tools_used == ["lookup_product_fact"]
