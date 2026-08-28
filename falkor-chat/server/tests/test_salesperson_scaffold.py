"""Regression coverage for the shared `salesperson` `WorkflowDef` scaffold (K-052
M6, `docs/plans/workflow-catalog-lookup.md` §2.4/§3.3/§5).

Three properties, per the plan's own "Additional coverage" list:

  1. `SALESPERSON_DEF` publishes clean — `_validate_def_spec`'s invariants (exactly
     one start step, ≥ 1 transition, a structurally-sound `cmp` guard) all pass for
     the real shipped constant, exercised through the real `publish_workflow_def`.
  2. A republish of the byte-identical spec is a clean structural no-op — the
     version-bump discipline §2.5 depends on (each sibling capability bumps the
     def's version rather than re-publishing `v1`) only works if a same-version
     republish is provably harmless, not merely assumed so.
  3. **The safety-critical one**: `ctx.endConversation` never becomes truthy across
     an ordinary, arbitrarily-long multi-turn conversation, so the `assistant ->
     ended` transition never fires and silently ends the demo run mid-conversation.
     Nothing in this def's tool set (`post_message`, `lookup_product_fact`,
     `filter_products`) ever writes to `ctx`, and the ordinary chat-resume path
     (`services.resume_workflow_run` -> `executor.resume` with **no** `run_ctx_json`)
     never merges anything into it either — unlike `submit_workflow_input`'s
     human/API input path. A companion test proves the guard mechanism itself is
     real (not vacuously never-firing because it's broken) by driving the one path
     that *does* set `ctx.endConversation` explicitly.

All three run offline against `ws:test` — no LLM, no network. The `assistant` step's
agent loop is driven by a scripted stub `chat()` (never a real model), mirroring
`test_executor_agent.py`'s stub collaborators.
"""

from __future__ import annotations

import itertools
import json

from falkorchat.config import CallContext
from falkorchat.executor import WorkflowExecutor
from falkorchat.llm import ChatResult, ToolCall
from falkorchat.proof_defs import SALESPERSON_DEF, SALESPERSON_MAX_STEPS
from falkorchat.services import Services

CTX = CallContext(ws="test", actor="u1")

KEY = SALESPERSON_DEF["key"]
#: Published under a **test-only version** (mirrors `test_process_flow.py`'s
#: `v1-test` convention) — `conftest.wf_repo` wipes `reference` at fixture setup,
#: so a finished pytest session leaves the last workflow test's defs behind;
#: publishing the production `salesperson@v1` from here would make
#: `seed_salesperson.sh`'s "already present — no-op" line untrustworthy.
VERSION = "v1-test"
TEST_DEF = {**SALESPERSON_DEF, "version": VERSION}


# ── stubs (mirrors test_executor_agent.py's collaborators) ────────────────────

_SCHEMAS = {
    "post_message": {
        "type": "function",
        "function": {
            "name": "post_message",
            "description": "Post a message to the customer.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    "lookup_product_fact": {
        "type": "function",
        "function": {
            "name": "lookup_product_fact",
            "description": "Look up one product's category/price by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    "filter_products": {
        "type": "function",
        "function": {
            "name": "filter_products",
            "description": "List products by category and/or price range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "minPrice": {"type": "number"},
                    "maxPrice": {"type": "number"},
                },
                "required": [],
            },
        },
    },
}


class StubRegistry:
    """Records dispatches; returns a canned success string per tool. Never called
    by the guard-safety tests below (nothing in this def's tools writes `ctx`) —
    this only makes the agent loop's tool-calling machinery itself runnable."""

    def __init__(self):
        self.dispatched: list[tuple[str, dict]] = []

    def schema(self, name):
        return _SCHEMAS[name]

    def dispatch(self, name, arguments, *, ctx, run):
        self.dispatched.append((name, arguments))
        return json.dumps({"ok": True})


class ScriptedTurnLLM:
    """One simulated customer turn = one `post_message` tool call, then a final
    text — replayed forever so it can drive an arbitrary number of turns. Never
    emits anything that could be mistaken for a `ctx` write: the agent loop itself
    has no mechanism to write `ctx` from a `ChatResult` at all (§2.4 — only
    `submit_workflow_input`'s explicit input-merge path can)."""

    def __init__(self, reply_text: str = "Sure, here you go."):
        self._reply_text = reply_text
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        if self.calls % 2 == 1:
            return ChatResult(
                text="",
                tool_calls=[ToolCall(
                    id=f"call{self.calls}", name="post_message",
                    arguments={"text": self._reply_text},
                )],
            )
        return ChatResult(text=self._reply_text)


def _make_services(repo, *, llm):
    """Real service + real engine, deterministic ids/clocks — mirrors
    `test_process_flow.py`'s `_make_services`."""
    ids = (f"id{n}" for n in itertools.count(1))
    services = Services(repo, clock=lambda: 1000, id_gen=lambda: next(ids))
    sr_ids = (f"sr{n}" for n in itertools.count(1))
    sr_clock = itertools.count(2000)
    services.set_executor(WorkflowExecutor(
        services, repo, llm=llm, tool_registry=StubRegistry(), guard_judge=None,
        id_gen=lambda: next(sr_ids), clock=lambda: next(sr_clock),
    ))
    return services


def _seed_thread(repo, *, channel_id, thread_id, user_id, msg_id, text):
    repo.create_channel(CTX.ws, channel_id=channel_id, name=channel_id, created_at=100)
    repo.create_thread(
        CTX.ws, channel_id=channel_id, thread_id=thread_id, title="chat",
        created_at=110,
    )
    repo.ensure_user(CTX.ws, user_id=user_id, display_name="Customer")
    repo.post_first_message(
        CTX.ws, thread_id=thread_id, msg_id=msg_id, author_id=user_id,
        text=text, role="user", created_at=120,
    )


# ── 1. the def itself publishes clean ──────────────────────────────────────────


def test_salesperson_def_publishes_clean_through_validate_def_spec(wf_repo):
    """`_validate_def_spec`'s invariants all pass for the real shipped constant:
    exactly one start step (`assistant`), >= 1 transition, and a structurally-sound
    `cmp` guard (`guards.validate_cmp`) — exercised through the real
    `publish_workflow_def`, not a copy of the spec."""
    services = _make_services(wf_repo, llm=None)

    pub = services.publish_workflow_def(CTX, **TEST_DEF)

    assert (pub["key"], pub["version"]) == (KEY, VERSION)
    assert pub["stepCount"] == 2       # `assistant` + `ended`
    assert pub["transitionCount"] == 1  # the one `ctx.endConversation` guard

    mat = services.materialize_def(CTX, key=KEY, version=VERSION)
    snap = services.get_snapshot(CTX, key=KEY, version=VERSION)
    assert mat["stepCount"] == 2
    assert snap["start_key"] == "assistant"
    assert {s["key"]: s["type"] for s in snap["steps"]} == {
        "assistant": "agent", "ended": "decision",
    }
    assert len(snap["transitions"]) == 1
    # `guard` (like `config`) is an OPAQUE SERIALIZED STRING on read-back
    # (`falkor-chat/AGENTS.md` rule 8 — "ctx, input, output ... are serialised
    # strings"; the same holds for a step's `config` / a transition's `guard`).
    guard = json.loads(snap["transitions"][0]["guard"])
    assert guard == {
        "kind": "cmp", "path": "ctx.endConversation", "op": "truthy",
    }


# ── 2. a same-version republish is a clean structural no-op ───────────────────


def test_republish_of_a_byte_identical_topology_is_a_clean_no_op(wf_repo):
    """The version-bump discipline (§2.5) three sibling capabilities depend on:
    K-053/K-054/K-055 each republish the FULL cumulative step config at a bumped
    version, relying on the *unchanged* topology across versions never hitting the
    K-034 409 conflict path. Proven here at the more basic level first — republishing
    the exact same (key, version) with byte-identical content must be a harmless,
    structurally-identical no-op, not merely assumed safe."""
    services = _make_services(wf_repo, llm=None)

    first = services.publish_workflow_def(CTX, **TEST_DEF)
    second = services.publish_workflow_def(CTX, **TEST_DEF)

    assert second == first


# ── 3. the safety-critical property: ordinary conversation never ends itself ──


def test_ordinary_multi_turn_conversation_never_fires_the_end_transition(wf_repo):
    """The single most safety-critical property of this scaffold (§2.4/§5): a
    false-positive fire of the `assistant -> ended` transition would silently end
    the demo run mid-conversation. Simulates 10 ordinary customer turns (an
    arbitrary "many" — `SALESPERSON_MAX_STEPS` leaves plenty of headroom) via the
    real chat-resume path (`resume_workflow_run`, no ctx merge) and asserts the run
    never leaves `assistant`."""
    llm = ScriptedTurnLLM()
    services = _make_services(wf_repo, llm=llm)
    services.publish_workflow_def(CTX, **TEST_DEF)
    services.materialize_def(CTX, key=KEY, version=VERSION)
    _seed_thread(
        wf_repo, channel_id="c-multi", thread_id="t-multi", user_id="u-multi",
        msg_id="m-multi", text="hi, do you have wireless mice?",
    )

    started = services.start_workflow_run(
        CTX, def_key=KEY, version=VERSION, trigger_msg_id="m-multi",
        max_steps=SALESPERSON_MAX_STEPS,
    )
    run_id = started["runId"]
    assert started["status"] == "waiting"
    assert wf_repo.get_run(CTX.ws, run_id=run_id)["atStepKey"] == "assistant"

    for turn in range(10):
        out = services.resume_workflow_run(CTX, run_id=run_id)
        assert out["status"] == "waiting", f"ended prematurely on turn {turn}"
        run = wf_repo.get_run(CTX.ws, run_id=run_id)
        assert run["status"] == "waiting"
        assert run["atStepKey"] == "assistant"
        assert json.loads(run["ctx"]).get("endConversation") is None

    # the agent loop really did run every turn (proves the loop wasn't a no-op
    # that trivially "never fires" because nothing executed)
    assert llm.calls == 22  # 1 (start) + 10 (resumes) turns x 2 LLM calls each


def test_ctx_end_conversation_truthy_does_fire_the_end_transition(wf_repo):
    """Sanity companion to the test above: proves the guard mechanism itself is
    real (the demo's own toolset just never triggers it, per the test above) —
    explicitly setting `ctx.endConversation` via the one path that can
    (`submit_workflow_input`'s human/API input merge) does fire the transition to
    the terminal `ended` step."""
    llm = ScriptedTurnLLM()
    services = _make_services(wf_repo, llm=llm)
    services.publish_workflow_def(CTX, **TEST_DEF)
    services.materialize_def(CTX, key=KEY, version=VERSION)
    _seed_thread(
        wf_repo, channel_id="c-end", thread_id="t-end", user_id="u-end",
        msg_id="m-end", text="bye",
    )

    started = services.start_workflow_run(
        CTX, def_key=KEY, version=VERSION, trigger_msg_id="m-end",
        max_steps=SALESPERSON_MAX_STEPS,
    )
    run_id = started["runId"]

    out = services.submit_workflow_input(
        CTX, run_id=run_id, input={"endConversation": True},
    )

    assert out["status"] == "done"
    run = wf_repo.get_run(CTX.ws, run_id=run_id)
    assert run["status"] == "done"
    assert run["atStepKey"] is None
    trail = [sr["stepKey"] for sr in wf_repo.read_step_runs(CTX.ws, run_id=run_id)]
    assert trail[-1] == "ended"
