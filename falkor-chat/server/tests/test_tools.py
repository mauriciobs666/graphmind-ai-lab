"""Unit + integration tests for the node-capability tools (U9).

`ToolRegistry` satisfies the exact interface `executor._run_agent_node` calls
(`schema(name)` + `dispatch(name, args, *, ctx, run)`). The built-ins:
  * `post_message` posts as the workflow agent via `services.post_agent_answer` (role
    derived `assistant`) and links `StepRun -[:PRODUCED]-> Message` via
    `services.link_step_emission` when a `stepRunId` is on the run (two-step, non-atomic).
  * `graphrag_retrieve` embeds the query (injected `Embedder`) → `services.hybrid_search`,
    then applies the DS-note Q2 policy (distance cutoff τ, cap 5, abstain on no-hit).
  * `human_handoff` raises `HumanHandoffSignal` (present, not exercised).

Threshold/registry logic is tested with stubs (deterministic, no DB); `post_message`'s §4
write + PRODUCED link and `graphrag_retrieve`'s abstention are exercised against the live
`ws:test` graph (conftest bootstraps schema + wipes per test).
"""

from __future__ import annotations

import itertools
import json

import pytest
from conftest import TEST_EMBEDDING_DIM

from falkorchat.config import CallContext
from falkorchat.services import Services
from falkorchat.tools import (
    GraphragRetrieveTool,
    HumanHandoffSignal,
    HumanHandoffTool,
    PostMessageTool,
    ToolRegistry,
    UnknownToolError,
    build_builtin_registry,
)

CTX = CallContext(ws="test", actor="u1")
WS = "test"


# ── stubs ────────────────────────────────────────────────────────────────────

class StubServices:
    """Records `post_agent_answer` / `link_step_emission` / `hybrid_search` calls."""

    def __init__(self, *, search_rows=None, link_ok=True):
        self._search_rows = search_rows or []
        self._link_ok = link_ok
        self.posted: list[dict] = []
        self.linked: list[dict] = []
        self.searched: list[dict] = []
        self._msg_seq = itertools.count(1)

    def post_agent_answer(self, ctx, *, thread_id, text, mentions=None, seeds=None):
        msg_id = f"m{next(self._msg_seq)}"
        self.posted.append(
            {"actor": ctx.actor, "thread_id": thread_id, "text": text,
             "mentions": mentions}
        )
        return {"msgId": msg_id, "threadId": thread_id, "text": text}

    def link_step_emission(self, ctx, *, step_run_id, msg_id):
        self.linked.append({"step_run_id": step_run_id, "msg_id": msg_id})
        return {"stepRunId": step_run_id, "msgId": msg_id} if self._link_ok else None

    def hybrid_search(self, ctx, *, q_vec, k=10, channel_id=None):
        self.searched.append({"q_vec": q_vec, "k": k, "channel_id": channel_id})
        return list(self._search_rows)


class StubEmbedder:
    def __init__(self, vec=None):
        self._vec = vec or [1.0, 0.0, 0.0, 0.0]
        self.calls: list[str] = []

    def embed(self, text):
        self.calls.append(text)
        return list(self._vec)


class _FakeTool:
    name = "echo"
    schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}

    def run(self, arguments, *, ctx, run):
        return f"echo:{arguments.get('v')}"


# ── ToolRegistry contract (matches _run_agent_node) ──────────────────────────

def test_registry_schema_and_dispatch_roundtrip():
    reg = ToolRegistry([_FakeTool()])
    assert reg.names() == ["echo"]
    assert reg.schema("echo")["function"]["name"] == "echo"
    out = reg.dispatch("echo", {"v": 7}, ctx=CTX, run={})
    assert out == "echo:7"


def test_registry_unknown_tool_raises_unknown_tool_error():
    reg = ToolRegistry([])
    with pytest.raises(UnknownToolError):
        reg.schema("nope")
    with pytest.raises(UnknownToolError):
        reg.dispatch("nope", {}, ctx=CTX, run={})
    # KeyError-compatible so bare-dict stub registries stay consistent (U8)
    assert issubclass(UnknownToolError, KeyError)


def test_registry_jsonifies_nonstring_results():
    class DictTool:
        name = "d"
        schema = {"type": "function", "function": {"name": "d", "parameters": {}}}

        def run(self, arguments, *, ctx, run):
            return {"a": 1}

    reg = ToolRegistry([DictTool()])
    assert json.loads(reg.dispatch("d", {}, ctx=CTX, run={})) == {"a": 1}


def test_build_builtin_registry_registers_all_three():
    reg = build_builtin_registry(StubServices(), StubEmbedder(), agent_id="assistant")
    assert set(reg.names()) == {"post_message", "graphrag_retrieve", "human_handoff"}


# ── post_message (unit, stub services) ───────────────────────────────────────

def test_post_message_posts_as_agent_and_returns_posted_msg_id():
    # Option B (K-023): the tool posts as the agent and returns the msgId in its
    # envelope; it does NOT link the emission (the executor owns PRODUCED linking
    # after the StepRun exists).
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    run = {"runId": "r1", "ctx": json.dumps({"threadId": "t1"})}

    out = json.loads(tool.run({"text": "hello"}, ctx=CTX, run=run))

    # posted as the agent (actor swapped), role derives assistant in the service
    assert svc.posted == [
        {"actor": "assistant", "thread_id": "t1", "text": "hello", "mentions": None}
    ]
    # the tool no longer links — that is the executor's job (Option B)
    assert svc.linked == []
    assert out["posted"]                 # the posted msgId, for the executor to link
    assert out["threadId"] == "t1"
    assert "linked" not in out           # no linked flag — linking moved to the executor


def test_post_message_does_not_link_even_with_step_run_id_on_run():
    # Even if a stepRunId happens to be on the run dict, the tool never links
    # (Option B decouples the tool from audit linking).
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    run = {"ctx": json.dumps({"threadId": "t1"}), "stepRunId": "sr1"}

    tool.run({"text": "hi"}, ctx=CTX, run=run)

    assert len(svc.posted) == 1
    assert svc.linked == []


def test_post_message_reads_thread_from_explicit_run_key():
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    tool.run({"text": "x"}, ctx=CTX, run={"threadId": "tX"})
    assert svc.posted[0]["thread_id"] == "tX"


def test_post_message_errors_without_a_bound_thread():
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    out = tool.run({"text": "x"}, ctx=CTX, run={"runId": "r1"})
    assert "no thread" in out
    assert svc.posted == []


def test_post_message_forwards_mentions():
    svc = StubServices()
    tool = PostMessageTool(svc, agent_id="assistant")
    tool.run({"text": "hey", "mentions": ["u2"]}, ctx=CTX,
             run={"threadId": "t1"})
    assert svc.posted[0]["mentions"] == ["u2"]


# ── graphrag_retrieve (unit, stub services + embedder) ───────────────────────

def _row(msg_id, score, text="ctx"):
    return {"msgId": msg_id, "text": text, "role": "user", "score": score,
            "relatedContext": []}


def test_graphrag_retrieve_embeds_then_applies_tau_cutoff_and_cap():
    rows = [_row("a", 0.0), _row("b", 0.4), _row("c", 0.6), _row("d", 0.9)]
    emb = StubEmbedder([0.5, 0.5, 0.0, 0.0])
    svc = StubServices(search_rows=rows)
    tool = GraphragRetrieveTool(svc, emb, tau=0.5, cap=5)

    out = json.loads(tool.run({"query": "reset password"}, ctx=CTX, run={}))

    assert emb.calls == ["reset password"]            # query embedded, not a raw vector
    assert svc.searched[0]["q_vec"] == [0.5, 0.5, 0.0, 0.0]
    # only seeds within τ=0.5 survive (a,b), ordering preserved; c,d cut
    assert [s["msgId"] for s in out["seeds"]] == ["a", "b"]
    assert "finding" not in out


def test_graphrag_retrieve_caps_at_five():
    rows = [_row(str(i), 0.1) for i in range(8)]  # all pass τ
    tool = GraphragRetrieveTool(StubServices(search_rows=rows), StubEmbedder(),
                                tau=0.5, cap=5)
    out = json.loads(tool.run({"query": "q"}, ctx=CTX, run={}))
    assert len(out["seeds"]) == 5


def test_graphrag_retrieve_abstains_when_nothing_passes_tau():
    rows = [_row("a", 0.7), _row("b", 0.9)]  # all beyond τ
    tool = GraphragRetrieveTool(StubServices(search_rows=rows), StubEmbedder(),
                                tau=0.5)
    out = json.loads(tool.run({"query": "off topic"}, ctx=CTX, run={}))
    assert out["seeds"] == []
    assert out["finding"] == "no relevant context found"


def test_graphrag_retrieve_configurable_tau():
    rows = [_row("a", 0.7)]
    tool = GraphragRetrieveTool(StubServices(search_rows=rows), StubEmbedder(),
                                tau=0.8)  # looser cutoff keeps 0.7
    out = json.loads(tool.run({"query": "q"}, ctx=CTX, run={}))
    assert [s["msgId"] for s in out["seeds"]] == ["a"]


# ── human_handoff (present, not exercised) ───────────────────────────────────

def test_human_handoff_dispatch_signals_suspend():
    reg = ToolRegistry([HumanHandoffTool()])
    with pytest.raises(HumanHandoffSignal) as exc:
        reg.dispatch("human_handoff", {"reason": "needs a person"}, ctx=CTX, run={})
    assert exc.value.reason == "needs a person"


# ── integration: post_message writes the agent message via §4 (durable artifact) ─
# The PRODUCED audit edge is asserted at executor altitude now (Option B — the
# executor links after the StepRun exists); see test_executor_produced.py.

def _configure_services(repo, *, actor="u1"):
    clock = itertools.count(1000)
    ids = (f"id{n}" for n in itertools.count(1))
    return Services(repo, clock=lambda: next(clock), id_gen=lambda: next(ids))


def test_post_message_writes_agent_message_live(repo, conn):
    repo.ensure_user(WS, user_id="u1")
    repo.ensure_agent(WS, agent_id="assistant", name="Bot")
    repo.create_channel(WS, channel_id="c1", name="c1", created_at=1)
    repo.create_thread(WS, channel_id="c1", thread_id="t1", title="t", created_at=1)
    repo.post_first_message(
        WS, thread_id="t1", msg_id="trigger", author_id="u1",
        text="please help", role="user", created_at=10,
    )

    svc = _configure_services(repo)
    tool = PostMessageTool(svc, agent_id="assistant")
    run = {"runId": "r1", "ctx": json.dumps({"threadId": "t1"})}

    out = json.loads(tool.run({"text": "here is your answer"}, ctx=CTX, run=run))

    posted = repo.get_message(WS, msg_id=out["posted"])
    assert posted["text"] == "here is your answer"
    assert posted["role"] == "assistant"            # role derived, agent author
    assert posted["authorId"] == "assistant"


# ── integration: graphrag_retrieve threshold + abstention over seeded ws:test ─

def _pad(head):
    return (head + [0.0] * TEST_EMBEDDING_DIM)[:TEST_EMBEDDING_DIM]


def _seed_embedded_thread(repo):
    repo.ensure_user(WS, user_id="u1")
    repo.create_channel(WS, channel_id="c1", name="c1", created_at=1)
    repo.create_thread(WS, channel_id="c1", thread_id="t1", title="t", created_at=1)
    msgs = [("m1", "about cats", _pad([1.0, 0.0])),
            ("m2", "about dogs", _pad([0.0, 1.0]))]
    first = True
    ts = 10
    for msg_id, text, vec in msgs:
        write = repo.post_first_message if first else repo.post_subsequent_message
        write(WS, thread_id="t1", msg_id=msg_id, author_id="u1",
              text=text, role="user", created_at=ts)
        repo.set_embedding(WS, msg_id=msg_id, embedding=vec, expected_dim=TEST_EMBEDDING_DIM)
        first = False
        ts += 1


def test_graphrag_retrieve_returns_near_seed_live(repo):
    _seed_embedded_thread(repo)
    svc = _configure_services(repo)
    # query vector identical to m1 → distance ~0 (≤ τ); m2 is orthogonal (distance ~1)
    tool = GraphragRetrieveTool(svc, StubEmbedder(_pad([1.0, 0.0])), tau=0.5)
    out = json.loads(tool.run({"query": "cats"}, ctx=CTX, run={}))
    ids = [s["msgId"] for s in out["seeds"]]
    assert "m1" in ids
    assert "m2" not in ids                # orthogonal seed cut by τ


def test_graphrag_retrieve_abstains_when_all_seeds_distant_live(repo):
    _seed_embedded_thread(repo)
    svc = _configure_services(repo)
    # a query orthogonal to BOTH seeds → every distance ~1 > τ → abstain
    tool = GraphragRetrieveTool(svc, StubEmbedder(_pad([0.0, 0.0, 1.0])), tau=0.5)
    out = json.loads(tool.run({"query": "unrelated"}, ctx=CTX, run={}))
    assert out["seeds"] == []
    assert out["finding"] == "no relevant context found"
