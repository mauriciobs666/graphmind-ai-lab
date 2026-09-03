"""App-assembly tests: both front doors mounted on one FastAPI process.

Guards the python-sdk #1367 gotcha — if the MCP app's lifespan is not forwarded,
the session manager never initialises and requests to /mcp fail with a 500
("task group not initialized"). Here the endpoint returns a protocol-level
response instead, and REST keeps working alongside it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import warnings
from typing import NamedTuple

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Host, Mount

from falkorchat import config, db
from falkorchat.app import create_app
from falkorchat.config import CallContext
from falkorchat.repository import Repository
from falkorchat.services import MemberIdCollisionError, ServiceError, Services

CTX = lambda: CallContext(ws="test", actor="u1")  # noqa: E731


def test_app_mounts_mcp_and_rest_routes():
    app = create_app(context_provider=CTX, mount_mcp=True)
    mount_paths = [r.path for r in app.routes if isinstance(r, Mount)]
    assert "/mcp" in mount_paths
    # REST endpoints are registered via an included router — assert through the
    # generated OpenAPI schema, which flattens their paths.
    paths = app.openapi()["paths"]
    assert "/channels" in paths
    assert "/threads/{thread_id}/messages" in paths


def test_mcp_lifespan_is_wired_and_rest_coexists(conn):
    app = create_app(context_provider=CTX, mount_mcp=True)
    with TestClient(app) as c:
        # session manager is running: /mcp routes and gives a protocol response,
        # not a 404 (unmounted), 405 (trailing-slash-only mount — QA DEF-1: the
        # documented `POST /mcp` must work, not just `/mcp/`), or 500 (lifespan
        # not forwarded).
        r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert r.status_code != 404
        assert r.status_code != 405
        assert r.status_code < 500
        # REST works on the same process
        assert c.get("/channels").status_code == 200


def test_fresh_tenant_posts_out_of_the_box(conn):
    """Regression: the configured actor must exist before the first write.

    Without startup seeding, the write query's author MATCH found no node and
    the whole write silently no-opped — REST returned 201 but nothing was
    stored. The lifespan now ensures the context actor as a User.
    """
    app = create_app(
        Services(Repository(conn)),  # note: u1 NOT seeded here
        context_provider=CTX,
        mount_mcp=False,
    )
    with TestClient(app) as c:  # with-block runs the lifespan (actor ensure)
        cid = c.post("/channels", json={"name": "general"}).json()["channelId"]
        tid = c.post(
            f"/channels/{cid}/threads", json={"title": "hi"}
        ).json()["threadId"]

        r = c.post(f"/threads/{tid}/messages", json={"text": "hello"})

        assert r.status_code == 201
        msgs = c.get(f"/threads/{tid}/messages").json()
        assert [m["text"] for m in msgs] == ["hello"]  # actually persisted


def test_importing_app_module_never_requires_reachable_falkordb():
    """DEF-2: `import falkorchat.app` runs `create_app()` at module scope; it
    must never touch the network. The QA repro: with a dead port configured,
    uvicorn sat >=90s with zero output because the import itself blocked on
    the eager FalkorDB connect."""
    env = {**os.environ, "FALKORDB_HOST": "10.255.255.1", "FALKORDB_PORT": "6399"}

    proc = subprocess.run(
        [sys.executable, "-c", "import falkorchat.app"],
        env=env, capture_output=True, text=True, timeout=15,
    )

    assert proc.returncode == 0, proc.stderr


# ── K-028 §6 test 6 — the periodic sweep's lifespan wiring ───────────────────────
# The one test in the whole K-028 plan that touches real wall-clock time (proving
# the loop actually ticks); everything needing precise due-time behaviour is
# proven by `test_workflow_timers.py`'s injected clock instead, never by this one.
# Interval kept tiny (10ms) and the wait generous (10x) so this stays fast and
# non-flaky.


class _StubSweepServices:
    """The minimal surface `create_app`'s lifespan touches: `ensure_actor` (no-op,
    so no FalkorDB is needed at all) and a counting `sweep_due_workflow_runs`."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def ensure_actor(self, ctx) -> None:
        pass

    def sweep_due_workflow_runs(self, ctx, *, limit: int):
        self.calls.append(limit)
        return {"checked": 0, "due": 0, "resumed": [], "raced": [], "faulted": []}


def test_sweep_loop_ticks_on_interval_and_cancels_cleanly_at_shutdown():
    # NOTE on what this test can and cannot discriminate (found empirically
    # while mutation-testing this test, not assumed): `TestClient`'s own
    # portal teardown cancels any task still running on its event loop when
    # the `with` block exits, REGARDLESS of whether `create_app`'s own
    # lifespan does its explicit `sweep_task.cancel()` + await — so
    # `sweep_task.cancelled()` being `True` at the end does not, by itself,
    # prove the app's own cancellation code ran (verified: removing that code
    # entirely, the task still ends up `.cancelled() == True` here). What
    # *does* catch a real regression in this harness is `app_.state.
    # sweep_task` actually being set — accessing it below raises
    # `AttributeError` if that line is ever dropped (confirmed by mutation-
    # testing that specific line). The `.cancelled()`/no-propagated-exception
    # checks are kept as a secondary, still-meaningful smoke check (a
    # `CancelledError` escaping the lifespan, or a task left `running` after
    # the `with` block, would both still fail loudly here) even though they
    # don't independently pin the shutdown-cancellation code path.
    services = _StubSweepServices()
    app = create_app(
        services, context_provider=CTX, mount_mcp=False,
        sweep_interval_s=0.01, sweep_limit=7,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with TestClient(app) as c:
            time.sleep(0.1)
            sweep_task = c.app.state.sweep_task  # AttributeError if never stored
            assert not sweep_task.done(), "the loop stopped ticking on its own"
        # The `with` block above returning cleanly (no propagated
        # `CancelledError`) is itself part of the assertion.

    assert services.calls, "the sweep tick never fired within the wait window"
    assert all(limit == 7 for limit in services.calls)
    assert sweep_task.cancelled(), "the task was left dangling, never cancelled"
    # No "task was destroyed but it is pending" (or similar asyncio) warning.
    assert not [
        w for w in caught
        if "was destroyed" in str(w.message) or "was never retrieved" in str(w.message)
    ]


def test_startup_against_unreachable_db_fails_fast_with_clear_error(monkeypatch):
    """DEF-2: building the app offline must work; the *lifespan* makes the
    first connection and must fail within the connect-timeout budget with an
    error naming host:port — not hang for minutes."""
    monkeypatch.setattr(config, "FALKORDB_HOST", "10.255.255.1")
    monkeypatch.setattr(config, "FALKORDB_PORT", 6399)
    monkeypatch.setattr(config, "FALKORDB_CONNECT_TIMEOUT", 1.0, raising=False)

    app = create_app(mount_mcp=False)  # default-services path, built offline

    t0 = time.monotonic()
    with pytest.raises(db.FalkorDBUnreachableError) as exc:
        with TestClient(app):  # lifespan startup → first real connection
            pass
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0
    assert "10.255.255.1:6399" in str(exc.value)


def test_startup_fails_loudly_when_actor_id_collides_with_agent(conn):
    """DEF-1 QA repro (S3): `FALKORCHAT_USER_ID` equal to an existing Agent id
    must abort startup with a clear error — the old MERGE silently created a
    shadow User that eclipsed the Agent in every coalesce(u, a) lookup."""
    repo = Repository(conn)
    repo.ensure_agent("test", agent_id="qabot", name="Bot")
    app = create_app(
        Services(repo),
        context_provider=lambda: CallContext(ws="test", actor="qabot"),
        mount_mcp=False,
    )

    with pytest.raises(MemberIdCollisionError):
        with TestClient(app):  # lifespan runs the actor ensure
            pass

    # the Agent was not eclipsed by a shadow User
    assert repo.resolve_member_kinds("test", ids=["qabot"]) == {"qabot": "Agent"}


def test_default_app_wiring_is_gated_on_enable_agent(monkeypatch):
    """The module-level `app` stays network-free by default: no responder/embedder
    is wired unless FALKORCHAT_ENABLE_AGENT is on. Constructing the LM Studio
    clients is itself offline (no request until a message posts), so the enabled
    path is exercisable here with no live model — we only inspect what gets wired.
    """
    from falkorchat import app as app_mod

    captured: dict = {}

    def fake_create_app(services=None, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return object()  # sentinel: we only inspect the wiring, not the app

    monkeypatch.setattr(app_mod, "create_app", fake_create_app)

    # Disabled (default): the plain app, no responder / embed_worker passed.
    monkeypatch.setattr(app_mod.config, "ENABLE_AGENT", False)
    app_mod._build_default_app()
    assert captured.get("responder") is None
    assert captured.get("embed_worker") is None
    # K-028: this path passes no kwargs at all — `sweep_interval_s` stays at
    # `create_app`'s own `None` default, untouched.
    assert "sweep_interval_s" not in captured

    # Enabled: both are wired and the responder targets the configured agent id.
    monkeypatch.setattr(app_mod.config, "ENABLE_AGENT", True)
    monkeypatch.setattr(app_mod.config, "AGENT_ID", "assistant")
    app_mod._build_default_app()
    assert captured["responder"] is not None
    assert captured["embed_worker"] is not None
    assert captured["responder"]._agent_id == "assistant"


def test_workflow_wiring_is_gated_on_workflow_enabled(monkeypatch):
    """WORKFLOW_ENABLED off (default) → the M2 wiring (responder, no trigger). On →
    the trigger is wired (holding the responder) and no bare responder is passed, so
    the API schedules exactly one handler. Constructing the clients is offline."""
    from falkorchat import app as app_mod

    captured: dict = {}

    def fake_create_app(services=None, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(app_mod, "create_app", fake_create_app)
    monkeypatch.setattr(app_mod.config, "ENABLE_AGENT", True)
    monkeypatch.setattr(app_mod.config, "AGENT_ID", "assistant")

    # WORKFLOW off: responder wired, trigger not.
    monkeypatch.setattr(app_mod.config, "WORKFLOW_ENABLED", False)
    app_mod._build_default_app()
    assert captured.get("responder") is not None
    assert captured.get("trigger") is None
    # K-028: the periodic sweep depends on the executor, which only exists in
    # the branch below — `create_app`'s `sweep_interval_s` default (`None`)
    # must be left untouched here (no keyword passed at all).
    assert "sweep_interval_s" not in captured

    # WORKFLOW on: trigger wired (targets the agent + configured def), responder held
    # by the trigger (not passed to create_app → API schedules only the trigger).
    monkeypatch.setattr(app_mod.config, "WORKFLOW_ENABLED", True)
    monkeypatch.setattr(app_mod.config, "TRIGGER_DEF_KEY", "triage")
    monkeypatch.setattr(app_mod.config, "TRIGGER_DEF_VERSION", "v1")
    monkeypatch.setattr(app_mod.config, "WORKFLOW_SWEEP_INTERVAL_S", 45.0)
    app_mod._build_default_app()
    assert captured.get("trigger") is not None
    assert captured.get("responder") is None
    trig = captured["trigger"]
    assert trig._agent_id == "assistant"
    assert trig._def_key == "triage"
    assert trig._responder is not None       # holds the responder for fall-through
    # K-028 §3.6: the sweep interval is only ever passed inside this same
    # WORKFLOW_ENABLED branch, sourced from config, not hardcoded.
    assert captured.get("sweep_interval_s") == 45.0


def test_build_llm_judge_parses_a_json_verdict():
    """The production judge matches the injected shape `(condition, *, understanding,
    recent_turns, ctx, step_output) -> {decision, rationale}` and parses the JSON verdict."""
    from falkorchat.app import _build_llm_judge

    class StubLLM:
        def __init__(self, text):
            self._text = text
            self.calls = []

        def complete(self, messages):
            self.calls.append(messages)
            return self._text

    llm = StubLLM('{"decision": true, "rationale": "all fields present"}')
    judge = _build_llm_judge(llm)

    verdict = judge(
        "enough info?", understanding={"missing": []}, recent_turns=[],
        ctx={}, step_output="",
    )

    assert verdict["decision"] is True
    assert "present" in verdict["rationale"]
    assert llm.calls  # the llm was actually driven


class _ReplyLLM:
    """Stub LLM returning one canned judge reply."""

    def __init__(self, text):
        self._text = text

    def complete(self, messages):
        return self._text


def _judge_verdict(reply: str) -> dict:
    from falkorchat.app import _build_llm_judge

    return _build_llm_judge(_ReplyLLM(reply))(
        "enough info?", understanding={}, recent_turns=[], ctx={}, step_output=""
    )


def test_build_llm_judge_parses_a_fenced_json_verdict():
    # K-027 item 1: the D13 Ministral shape — a correct verdict wrapped in a
    # ```json fence used to be destroyed by the bare json.loads (26/26 unparseable).
    verdict = _judge_verdict(
        '```json\n{"decision": true, "rationale": "the user named the service"}\n```'
    )

    assert verdict["decision"] is True
    assert "named the service" in verdict["rationale"]


def test_build_llm_judge_parses_an_unlabelled_fenced_verdict():
    verdict = _judge_verdict('```\n{"decision": true, "rationale": "all fields given"}\n```')

    assert verdict["decision"] is True


def test_build_llm_judge_parses_a_prose_wrapped_verdict():
    verdict = _judge_verdict(
        'Here is my verdict:\n{"decision": true, "rationale": "the user gave repro steps"}\n'
        "Let me know if you need more."
    )

    assert verdict["decision"] is True
    assert "repro steps" in verdict["rationale"]


def test_build_llm_judge_still_suspends_on_a_json_reply_that_is_not_a_verdict():
    # JSON that carries no verdict object must not advance — the bias-to-suspend
    # default (and guards._coerce_verdict downstream) still holds.
    assert _judge_verdict("[1, 2, 3]")["decision"] is False
    assert _judge_verdict('"just a string"')["decision"] is False


def test_build_llm_judge_does_not_invent_a_verdict_from_prose():
    # The tolerant parse must not turn an English answer into a decision.
    verdict = _judge_verdict("Yes, the condition is clearly satisfied.")

    assert verdict["decision"] is False


def test_build_llm_judge_does_not_advance_on_a_hypothetical_inline_verdict():
    # gate B-1: the judge narrates a counterfactual and then answers false. Reading
    # the *quoted* object out of the sentence advances a guard that HEAD suspended —
    # the dangerous direction, and `guards._coerce_verdict` cannot catch it because
    # the quoted rationale is clean.
    verdict = _judge_verdict(
        "If the user had named the service I would answer "
        '{"decision": true, "rationale": "named"} but they did not, so I answer false.'
    )

    assert verdict["decision"] is False


def test_build_llm_judge_does_not_advance_on_an_inline_schema_echo():
    verdict = _judge_verdict(
        'The expected reply shape is {"decision": true, "rationale": "..."} — '
        "in this case the condition is not met."
    )

    assert verdict["decision"] is False


def test_build_llm_judge_advances_on_an_own_line_schema_echo():
    # gate N-2 — characterisation of a *declared* residual, not an endorsement.
    # The inline twin above suspends; this one owns its lines, so the conservative
    # parse accepts it and the guard advances. Closing it needs the model's intent,
    # not a parser (see `llm.extract_own_line_json_object`). Pinned so the boundary
    # is visible: if K-027 item 3's calibration counts this in the false-advance
    # rate, *this* test is what changes.
    verdict = _judge_verdict(
        "The reply shape is:\n"
        '{"decision": true, "rationale": "..."}\n'
        "In this case the condition is not met."
    )

    assert verdict["decision"] is True


def test_build_llm_judge_suspends_on_a_single_line_array_wrapped_verdict():
    # gate B-1's second shape: at HEAD a non-dict ⇒ suspend; the permissive
    # extractor lifted the inner object out and advanced. Still suspends.
    assert (
        _judge_verdict('[{"decision": true, "rationale": "named"}]')["decision"]
        is False
    )


def test_build_llm_judge_advances_on_a_multiline_array_wrapped_verdict():
    # gate N-6 — characterisation. The object owns its lines, so it is *asserted*
    # rather than quoted, and the own-line rule accepts it. Asymmetric with the
    # single-line form above purely because of line ownership; pinned so a future
    # narrowing is deliberate.
    verdict = _judge_verdict('[\n{"decision": true, "rationale": "named"}\n]')

    assert verdict["decision"] is True


def test_build_llm_judge_suspends_when_two_candidate_verdicts_disagree():
    # Two own-line objects: which one is the verdict is a guess, and a guess in the
    # advance direction is exactly what the bias-to-suspend design forbids.
    verdict = _judge_verdict(
        "Example of advancing:\n"
        '{"decision": true, "rationale": "all fields given"}\n'
        "My actual answer:\n"
        '{"decision": false, "rationale": "service not named"}'
    )

    assert verdict["decision"] is False


def test_build_llm_judge_ignores_an_own_line_object_without_a_decision_key():
    verdict = _judge_verdict(
        "Here is the evidence I considered:\n"
        '{"request": "access", "known": ["service"]}\n'
        "I cannot decide."
    )

    assert verdict["decision"] is False


def test_build_llm_judge_biases_to_suspend_on_unparseable_output():
    from falkorchat.app import _build_llm_judge

    class StubLLM:
        def complete(self, messages):
            return "I think it is probably fine"      # not JSON

    verdict = _build_llm_judge(StubLLM())(
        "enough info?", understanding={}, recent_turns=[], ctx={}, step_output=""
    )
    # a non-parseable verdict must not advance — guards._coerce_verdict then holds
    assert verdict["decision"] is False


def test_build_llm_judge_returns_an_object_advertising_accepts_run():
    from falkorchat.app import _build_llm_judge

    class StubLLM:
        def complete(self, messages):
            return '{"decision": true, "rationale": "x"}'

    judge = _build_llm_judge(StubLLM())
    assert getattr(judge, "accepts_run", False) is True


def _fake_transport_factory(captured):
    def fake_make_http_transport(*, timeout, headers=None, opener=None, provider="?", model="?"):
        def _transport(url, payload):
            captured["url"] = url
            captured["model"] = payload.get("model")
            return {
                "choices": [{"message": {"content": '{"decision": true, "rationale": "ok"}'}}]
            }
        return _transport
    return fake_make_http_transport


def test_build_llm_judge_resolves_kind_guard_through_a_real_gateway_with_ws_from_run(
    monkeypatch,
):
    from falkorchat.app import _build_llm_judge
    from falkorchat.modelconfig import ModelGateway, Overlay, ProviderCatalog

    catalog = ProviderCatalog(
        {"lmstudio": {"options": {"baseURL": "http://host:1/v1"}}}, path="opencode.json"
    )
    overlay = Overlay({"defaults": {"guard": "lmstudio/qwen/qwen3-4b-2507"}}, path="models.json")
    models = ModelGateway(catalog, overlay)

    captured: dict = {}
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(captured)
    )

    judge = _build_llm_judge(models)
    verdict = judge(
        "enough info?", understanding={}, recent_turns=[], ctx={}, step_output="",
        run={"ws": "acme"},
    )

    assert verdict["decision"] is True
    assert captured["url"] == "http://host:1/v1/chat/completions"
    assert captured["model"] == "qwen/qwen3-4b-2507"


def test_build_llm_judge_honours_a_guard_declared_model_override(monkeypatch):
    from falkorchat.app import _build_llm_judge
    from falkorchat.modelconfig import ModelGateway, Overlay, ProviderCatalog

    catalog = ProviderCatalog(
        {"lmstudio": {"options": {"baseURL": "http://host:1/v1"}}}, path="opencode.json"
    )
    overlay = Overlay({"defaults": {"guard": "lmstudio/default-model"}}, path="models.json")
    models = ModelGateway(catalog, overlay)

    captured: dict = {}
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(captured)
    )

    judge = _build_llm_judge(models)
    judge(
        "enough info?", understanding={}, recent_turns=[], ctx={}, step_output="",
        model="lmstudio/special-model", run={"ws": "acme"},
    )

    assert captured["model"] == "special-model"


def test_build_llm_judge_workspace_override_beats_the_guards_own_declared_model(
    monkeypatch,
):
    # K-042 L2-3 / AC-10, `guard` kind — the B-1 payoff: this is the one consumer
    # that had no workspace carrier before Landing 1's `run["ws"]` stamp, so proving
    # the hard cap reaches it (and beats the guard's OWN declared `model=`, not just
    # the kind default) is what B-1 existed to make possible.
    from falkorchat.app import _build_llm_judge
    from falkorchat.modelconfig import ModelGateway, Overlay, ProviderCatalog

    catalog = ProviderCatalog(
        {"lmstudio": {"options": {"baseURL": "http://host:1/v1"}}}, path="opencode.json"
    )
    overlay = Overlay({"defaults": {"guard": "lmstudio/kind-default"}}, path="models.json")
    models = ModelGateway(catalog, overlay)

    captured: dict = {}
    monkeypatch.setattr(
        "falkorchat.transport.make_http_transport", _fake_transport_factory(captured)
    )

    judge = _build_llm_judge(models)
    verdict = judge(
        "enough info?", understanding={}, recent_turns=[], ctx={}, step_output="",
        model="lmstudio/guard-declared-model",
        run={
            "ws": "acme",
            "modelOverrides": {
                "agentModel": None, "guardModel": "lmstudio/workspace-model",
                "embeddingModel": None, "responderModel": None,
            },
        },
    )

    assert verdict["decision"] is True
    # the guard's own declared model was never called — the override wins outright
    assert captured["model"] == "workspace-model"


def test_web_ui_served_at_root_without_shadowing_rest(tmp_path, conn):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>falkor-chat</title>")

    app = create_app(
        Services(Repository(conn)),
        context_provider=CTX,
        mount_mcp=False,
        web_dir=web,
    )
    with TestClient(app) as c:
        root = c.get("/")
        assert root.status_code == 200
        assert "falkor-chat" in root.text
        # the catch-all static mount must not shadow the REST API
        assert c.get("/channels").status_code == 200


# ── the DS §Q1 judge prompt: CONDITION / CURRENT STATE / RECENT TURNS ─────────
#
# The judge prompt had no RECENT TURNS block at all — so even once `guards` selected the
# fallback tier, the evidence had nowhere to land (Defect A's third link). These pin the
# rendering; the omit rule itself is `guards`' job and is pinned in test_guards.py.

def _turns(n, *, text=None):
    return [
        {"speaker": f"Alice{i}", "role": "user", "text": text or f"turn {i}"}
        for i in range(n)
    ]


def test_judge_prompt_renders_recent_turns_newest_last():
    # T10 — the block exists, is labelled context-only, and preserves chronology.
    from falkorchat.app import _render_judge_user

    user = _render_judge_user("enough info?", {}, _turns(3))

    assert "CONDITION: enough info?" in user
    assert "RECENT TURNS (context only):" in user
    assert "Alice0: turn 0" in user
    assert user.index("turn 0") < user.index("turn 2")   # newest last
    assert "CURRENT STATE" not in user                   # nothing to render


def test_judge_prompt_omits_recent_turns_when_an_understanding_is_present():
    # T11 — the renderer is a dumb renderer: handed no turns, it emits no block.
    from falkorchat.app import _render_judge_user

    user = _render_judge_user("enough info?", {"request": "reset password"}, [])

    assert "RECENT TURNS" not in user
    assert "CURRENT STATE:" in user
    assert "reset password" in user


def test_judge_prompt_is_capped_by_dropping_the_oldest_turns_first():
    # T12 — the newest turn is the one the condition is usually about; it must survive
    # the cap. Oldest-first eviction, then a hard truncation backstop.
    from falkorchat.app import JUDGE_USER_MAX_CHARS, _render_judge_user

    turns = [
        {"speaker": f"S{i:02d}", "role": "user", "text": "x" * 400}
        for i in range(50)
    ]
    user = _render_judge_user("enough info?", {}, turns)

    assert len(user) <= JUDGE_USER_MAX_CHARS
    assert "S49:" in user      # the newest turn survives the cap
    assert "S00:" not in user  # the oldest was evicted first


def test_judge_prompt_cap_holds_at_scale_well_beyond_the_shipped_window():
    # n-2 (K-027 carried finding): the cap loop was rewritten from O(turns^2)
    # (re-joining the whole candidate message on every eviction) to O(turns) —
    # this pins the arithmetic at a scale (300 turns) the shipped RECENT_TURNS_N=6
    # window never reaches, so a reintroduced off-by-one in the new eviction
    # arithmetic would show here even though it's invisible at N=6.
    from falkorchat.app import JUDGE_USER_MAX_CHARS, _render_judge_user

    turns = [
        {"speaker": f"S{i:04d}", "text": "y" * 100}
        for i in range(300)
    ]
    user = _render_judge_user("enough info?", {}, turns)

    assert len(user) <= JUDGE_USER_MAX_CHARS
    assert "S0299:" in user       # the newest turn survives the cap
    assert "S0000:" not in user  # the oldest was evicted first


def test_judge_prompt_survives_a_condition_with_no_evidence_at_all():
    # The degenerate case must still be a well-formed prompt, not a crash: the judge
    # then correctly biases to suspend (that behavior is Defect A's *symptom*, and is
    # the right answer when there genuinely is no evidence).
    from falkorchat.app import _render_judge_user

    user = _render_judge_user("enough info?", {}, [])

    assert user == "CONDITION: enough info?"


# ── salesperson-ui S3: the two wiring switches (plan §4.3 part 4, §4.9) ───────
#
# Both assertions here are on the **route table**, never on a 404 probe: a 404
# passes when a route is absent *and* when it exists but errors, so it is
# evidence that proves less than it appears to (plan §4.9, §6.1).


class RouteEntry(NamedTuple):
    """One registered route: its methods, its fully-prefixed path, and the
    per-route `responses={…}` declaration FastAPI keeps on the route object.

    `methods` is empty for a `Mount`. `responses` is `{}` for anything that
    carries none — which is what makes "every storefront route declares its own
    returns" an assertion rather than a reading exercise (salesperson-ui S8).
    """

    methods: frozenset[str]
    path: str
    responses: dict


def _route_entries(app) -> list[RouteEntry]:
    """Every route registered on `app`, flattened and fully prefixed, duplicates kept.

    Two FastAPI 0.139 facts this has to get right, both of which produce an
    assertion that cannot fail if it gets them wrong:

    1. **An included router is ONE opaque `_IncludedRouter` entry** in `app.routes`
       rather than its `APIRoute`s spliced in, and that wrapper has no `.path` at
       all — so the naive `[r.path for r in app.routes]` reports the whole legacy
       REST surface as zero paths whether the router is mounted or not. Recurse
       through anything carrying a nested router.
    2. **`include_router(prefix=...)` lives on that wrapper**
       (`include_context.prefix`), NOT on the inner routes — so appending a raw
       inner `.path` reports `/join` for a router mounted at `/shop/api`. Thread
       the prefix through the walk and accumulate it, so nested includes compose.

    `Mount`s deliberately stop the walk and appear as their own single entry
    (`/mcp`, and `""` for the `/` static catch-all — how Starlette normalises it).

    A route matching neither shape **raises** rather than being skipped: a helper
    whose entire job is completeness must not under-report silently, which is the
    same failure mode it exists to catch. `starlette.routing.Host` is exactly that
    shape; `create_app` registers none today.
    """
    found: list[RouteEntry] = []

    def walk(routes, prefix: str = "") -> None:
        for route in routes:
            nested = getattr(getattr(route, "original_router", None), "routes", None)
            if nested is not None:
                own = getattr(getattr(route, "include_context", None), "prefix", "")
                walk(nested, prefix + (own or ""))
                continue
            path = getattr(route, "path", None)
            if path is None:
                raise AssertionError(
                    f"route-table helper cannot classify {route!r} — it carries "
                    "neither a nested router nor a `.path`, so it would vanish "
                    "from the table this helper exists to assert on"
                )
            found.append(
                RouteEntry(
                    methods=frozenset(getattr(route, "methods", None) or ()),
                    path=prefix + path,
                    responses=dict(getattr(route, "responses", None) or {}),
                )
            )

    walk(app.routes)
    return found


def _route_paths(app) -> list[str]:
    """`_route_entries`, projected to just the paths (the S3-era view)."""
    return [entry.path for entry in _route_entries(app)]


# `/openapi.json`, `/docs`, `/docs/oauth2-redirect`, `/redoc` — FastAPI's own,
# on every app it builds. Subtracted so the assertions below read as "what this
# app registered", and so a FastAPI upgrade that changes its doc routes does not
# quietly turn an exact-set assertion into a false failure.
_FASTAPI_BUILTIN_PATHS = frozenset(_route_paths(FastAPI()))


def _registered_paths(app) -> list[str]:
    return [p for p in _route_paths(app) if p not in _FASTAPI_BUILTIN_PATHS]


def test_default_deployment_registers_the_legacy_surface_with_one_health_route(tmp_path):
    """The control for the assertion below: the helper genuinely SEES the legacy
    router, the `/` static mount and `/mcp`. Without this, `dev_surface=False`'s
    empty route table would pass against a helper that finds nothing ever."""
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>falkor-chat</title>")

    app = create_app(context_provider=CTX, mount_mcp=True, web_dir=web)

    paths = _registered_paths(app)
    assert "/channels" in paths                     # api.build_router is mounted
    assert "/search" in paths
    assert "/mcp" in [r.path for r in app.routes if isinstance(r, Mount)]
    assert "" in [r.path for r in app.routes if isinstance(r, Mount)]   # `/` static
    # exactly one `/health` — the router's own; the S3 liveness route must not
    # be registered a second time on this configuration
    assert paths.count("/health") == 1
    # The other half of the control — that the helper reports *where* a router is
    # mounted, not merely that it is — needs a prefix, which this app has none of:
    # see the next test.


def test_route_paths_helper_reports_a_routers_mount_prefix_through_nested_includes():
    """The second half of the control: the helper must report where a router is
    mounted, not just that it exists.

    FastAPI 0.139 keeps `include_router(prefix=...)` on the `_IncludedRouter`
    wrapper (`include_context.prefix`), NOT on the inner `APIRoute`s — so a walk
    that appends the raw inner `.path` reports `/join` for a router mounted at
    `/shop/api`. `create_app` itself has no prefixed include, which is exactly why
    this needs its own probe: an assertion over S3's route table cannot fail on a
    prefix bug, and S8's route table is a router at `/shop/api` and a mount at
    `/shop` — i.e. it is ALL prefix. Unfixed, an S8 assertion would read the same
    whether that router were mounted at `/shop/api`, at `/`, or at `/admin`.

    Two levels, because the prefixes must accumulate rather than the innermost
    winning.
    """
    inner = APIRouter()

    @inner.get("/join")
    def join():  # pragma: no cover - never called; only its registration matters
        return {}

    outer = APIRouter()
    outer.include_router(inner, prefix="/api")
    probe = FastAPI()
    probe.include_router(outer, prefix="/shop")

    assert _registered_paths(probe) == ["/shop/api/join"]


def test_route_paths_helper_refuses_a_route_it_cannot_classify():
    """A helper whose whole job is completeness must not under-report silently —
    that is the very failure it exists to prevent. `starlette.routing.Host` is a
    route with neither `original_router` nor `.path`; `create_app` registers none
    today, so this is a tripwire for a future one rather than a live case."""
    probe = FastAPI()
    probe.routes.append(Host("evil.example", app=FastAPI()))

    with pytest.raises(AssertionError, match="cannot classify"):
        _route_paths(probe)


def test_dev_surface_false_registers_no_legacy_router_no_web_mount_and_no_mcp(tmp_path):
    """§4.9 move 1: the storefront deployment does not mount the unauthenticated
    surfaces at all. `dev_surface=False` is dominant — it un-mounts `/mcp` even
    when `mount_mcp=True` is asked for, so the dangerous shape is not expressible.
    """
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>falkor-chat</title>")

    app = create_app(
        context_provider=CTX, mount_mcp=True, web_dir=web, dev_surface=False
    )

    # the whole route table, exactly: one bare liveness route and nothing else
    assert _registered_paths(app) == ["/health"]
    assert [r.path for r in app.routes if isinstance(r, Mount)] == []


def test_dev_surface_false_still_answers_the_health_liveness_probe(conn):
    app = create_app(
        Services(Repository(conn)), context_provider=CTX,
        mount_mcp=False, dev_surface=False,
    )
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_dev_surface_false_health_reports_503_when_falkordb_does_not_answer(conn):
    """Same contract as the router's own `/health` — liveness, not a stub 200."""
    class _DeadServices(Services):
        def ping(self, ctx):  # noqa: ANN001
            raise RuntimeError("FalkorDB unreachable")

    app = create_app(
        _DeadServices(Repository(conn)), context_provider=CTX,
        mount_mcp=False, dev_surface=False,
    )
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503


class _ProviderFailingAfterStartup:
    """Resolves the context once (for the lifespan's `ensure_actor`), then raises.

    Stands in for the post-auth `config.get_context` seam (`docs/SERVER.md` §1.3),
    where resolving the *caller* can genuinely fail. Today's provider returns a
    constant and cannot, which is why the contract needs a stub to be pinned at all.
    """

    def __init__(self, ctx) -> None:  # noqa: ANN001
        self._ctx = ctx
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls == 1:
            return self._ctx
        raise RuntimeError("caller could not be resolved")


def test_dev_surface_false_health_does_not_report_an_auth_failure_as_a_dead_database(conn):
    """The bare `/health` must resolve the context OUTSIDE its `try`, matching
    `api.py`'s route (which takes `ctx` through `Depends`, outside any handler of
    its own). Swallowing a context-resolution failure into the FalkorDB branch
    would report a rejected caller as "FalkorDB unreachable" — a 503 pointing an
    operator at the database when the database is fine."""
    provider = _ProviderFailingAfterStartup(CallContext(ws="test", actor="u1"))
    app = create_app(
        Services(Repository(conn)), context_provider=provider,
        mount_mcp=False, dev_surface=False,
    )
    with TestClient(app) as c:
        with pytest.raises(RuntimeError, match="caller could not be resolved"):
            c.get("/health")
    assert provider.calls == 2  # once at startup, once in the route


@pytest.mark.parametrize(
    "enable_agent, workflow_enabled",
    [(False, False), (True, False), (True, True)],
    ids=["plain-app", "responder-app", "workflow-app"],
)
def test_default_app_derives_both_switches_from_storefront_enabled(
    monkeypatch, enable_agent, workflow_enabled
):
    """§4.9 move 1: `_build_default_app` derives `dev_surface` as
    `not config.STOREFRONT_ENABLED`, alongside `mount_mcp` derived the same way —
    on every one of its three return paths, not just the wired one."""
    from falkorchat import app as app_mod

    captured: dict = {}

    def fake_create_app(services=None, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(app_mod, "create_app", fake_create_app)
    monkeypatch.setattr(app_mod.config, "ENABLE_AGENT", enable_agent)
    monkeypatch.setattr(app_mod.config, "WORKFLOW_ENABLED", workflow_enabled)
    monkeypatch.setattr(app_mod.config, "AGENT_ID", "assistant")

    monkeypatch.setattr(app_mod.config, "STOREFRONT_ENABLED", False)
    app_mod._build_default_app()
    assert captured.get("dev_surface") is True
    assert captured.get("mount_mcp") is True

    monkeypatch.setattr(app_mod.config, "STOREFRONT_ENABLED", True)
    app_mod._build_default_app()
    assert captured.get("dev_surface") is False
    assert captured.get("mount_mcp") is False


class _SpyResponder:
    """Stands in for `AgentResponder` (deferred-imported inside
    `_build_default_app`, so a module attribute swap reaches it)."""

    def __init__(self, services, *, worker=None, agent_id=None, models=None):  # noqa: ANN001
        self._agent_id = agent_id
        self.calls: list = []

    def maybe_respond(self, ctx, **kwargs):  # noqa: ANN001
        self.calls.append(kwargs)
        return {"replied": True}


class _NoWaitingRunServices:
    """No parked run in this thread — so `maybe_trigger` reaches step 3/4."""

    def find_waiting_run_for_thread(self, ctx, *, thread_id):  # noqa: ANN001
        return None


def _drive_unmentioning_message(trigger):
    """Send the wired trigger an unmentioning, non-resuming message.

    Swaps the trigger's live `Services` for a stub so this stays offline; the
    trigger under test is otherwise exactly the one `_build_default_app` wired.
    """
    trigger._services = _NoWaitingRunServices()
    return trigger.maybe_trigger(
        CallContext(ws="test", actor="p1"), thread_id="t1", msg_id="m1",
        text="how much is the blue mug?", role="member", mentions=[],
    )


def test_trigger_responder_fall_through_is_gated_on_its_own_flag(monkeypatch):
    """§4.3 part 4 / S3(a): `FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH=0` wires
    `WorkflowTrigger(responder=None)`, making the M2 responder's workspace-wide
    retrieval structurally unreachable from a participant's message."""
    from falkorchat import app as app_mod
    from falkorchat import responder as responder_mod

    captured: dict = {}

    def fake_create_app(services=None, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(app_mod, "create_app", fake_create_app)
    monkeypatch.setattr(responder_mod, "AgentResponder", _SpyResponder)
    monkeypatch.setattr(app_mod.config, "ENABLE_AGENT", True)
    monkeypatch.setattr(app_mod.config, "WORKFLOW_ENABLED", True)
    monkeypatch.setattr(app_mod.config, "AGENT_ID", "assistant")
    monkeypatch.setattr(app_mod.config, "TRIGGER_DEF_KEY", "salesperson")
    monkeypatch.setattr(app_mod.config, "TRIGGER_DEF_VERSION", "v7")

    # Flag ON (the default) — unchanged M3 behaviour: the trigger holds the
    # responder and an unhandled message falls through to it.
    monkeypatch.setattr(app_mod.config, "TRIGGER_RESPONDER_FALLTHROUGH", True)
    app_mod._build_default_app()
    trigger = captured["trigger"]
    assert isinstance(trigger._responder, _SpyResponder)
    assert _drive_unmentioning_message(trigger) == {"replied": True}
    assert len(trigger._responder.calls) == 1

    # Flag OFF — no responder is wired at all, so the same message reaches none.
    monkeypatch.setattr(app_mod.config, "TRIGGER_RESPONDER_FALLTHROUGH", False)
    app_mod._build_default_app()
    trigger = captured["trigger"]
    assert trigger._responder is None
    assert _drive_unmentioning_message(trigger) is None


def test_trigger_responder_fall_through_defaults_to_on():
    """The flag is opt-OUT: an existing deployment that sets nothing keeps the
    M2 fall-through it has today."""
    assert config.TRIGGER_RESPONDER_FALLTHROUGH is True


def test_storefront_is_disabled_by_default():
    assert config.STOREFRONT_ENABLED is False


# ── salesperson-ui S8: the storefront wiring (plan §4.7, §4.9, §5.1) ─────────
#
# Route-table assertions again, never 404 probes: a 404 passes when a route is
# absent *and* when it exists but errors.


def _storefront_app(tmp_path=None, **kwargs):
    return create_app(
        context_provider=CTX, mount_mcp=False, dev_surface=False,
        storefront=True, **kwargs,
    )


def test_storefront_and_dev_surface_together_are_not_expressible():
    """§4.9's route-table assertion, keyed on the **parameter** rather than on
    `config.STOREFRONT_ENABLED`.

    Keying it on the module constant would have made the guard dead in exactly
    the configuration the suite tests: `config.py` resolves every flag at
    *import* time, so every `create_app(storefront=True, dev_surface=False)`
    here leaves `config.STOREFRONT_ENABLED` `False` and would skip it.
    """
    with pytest.raises(ValueError, match="dev_surface=False"):
        create_app(context_provider=CTX, storefront=True)
    with pytest.raises(ValueError, match="dev_surface=False"):
        create_app(context_provider=CTX, storefront=True, dev_surface=True)


def test_the_storefront_route_table_is_the_eleven_api_routes_and_liveness():
    """The whole route table, exactly. Two things it pins that a probe cannot:
    that the router landed at `/shop/api` and not at `/` (the helper threads the
    prefix), and that §4.9's un-mounted surfaces really are absent.
    """
    app = _storefront_app()

    assert sorted(_registered_paths(app)) == sorted([
        "/health",
        "/shop/api/health",
        "/shop/api/session",
        "/shop/api/state",
        "/shop/api/messages",   # GET
        "/shop/api/messages",   # POST — one path, two routes
        "/shop/api/catalog",
        "/shop/api/order/advance",
        "/shop/api/reset",
        "/shop/api/presenter/session",
        "/shop/api/presenter/participants",
        "/shop/api/presenter/reset-all",
    ])
    # no legacy router, no `/` static catch-all, no `/mcp`
    assert "/channels" not in _registered_paths(app)
    assert [r.path for r in app.routes if isinstance(r, Mount)] == []


def test_the_shop_mount_is_registered_inside_create_app_and_shadows_nothing(tmp_path):
    """`/` is a catch-all registered last and Starlette matches in registration
    order, so a mount added after `create_app` returns is unreachable — the
    mount therefore lives inside `create_app`, and this asserts it is there
    when the function returns rather than after a caller remembers to add it.
    """
    served = tmp_path / "dist"
    (served / "products").mkdir(parents=True)
    (served / "index.html").write_text("<!doctype html><title>shop</title>")

    app = _storefront_app(storefront_dir=served)

    # The mount is on the route table the moment `create_app` returns — which
    # is the claim, and the one a caller-side `app.mount("/shop", ...)` after
    # the fact would satisfy only by luck of registration order.
    assert [r.path for r in app.routes if isinstance(r, Mount)] == ["/shop"]

    # No lifespan: §4.9's readiness preflight is asserted in
    # `test_storefront_api.py` against a seeded workspace, and entering it here
    # would make this test about seeding rather than about mount ordering.
    client = TestClient(app)
    # `/shop` shadows neither the bare liveness probe nor the storefront's own
    assert client.get("/health").status_code in (200, 503)
    assert client.get("/shop/api/health").status_code == 200
    assert client.get("/shop/").status_code == 200


def test_create_app_never_pins_the_participant_id_generator(tmp_path):
    """S6's participant-id collision argument rests on **no caller ever pinning
    `id_gen`**, and `create_app` is the first caller.

    Asserted at the construction seam rather than by reading `app.py`: a source
    grep would pass if the parameter were set anywhere else, and would fail on
    a comment that merely mentions it.
    """
    import falkorchat.app as app_module
    from falkorchat.storefront import Storefront as RealStorefront

    seen: list[dict] = []

    class _Recording(RealStorefront):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs)
            super().__init__(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_module, "Storefront", _Recording)
        _storefront_app(storefront_dir=tmp_path)

    assert len(seen) == 1
    assert "id_gen" not in seen[0]
    # the positive control: the recorder really did see the construction, and
    # the parameters that *are* forwarded arrived
    assert seen[0]["storefront_dir"] == tmp_path
    assert seen[0]["ws"] == "test"


def test_the_default_deployment_is_untouched_by_the_storefront_parameters():
    """§4.9: "Consequence for the non-storefront deployment: none."

    `FALKORCHAT_STOREFRONT_ENABLED` is off by default, so the legacy router,
    `/mcp` and `web/index.html` at `/` all keep the shape that shipped — and
    none of S8's typed error handlers is registered on it.
    """
    from falkorchat import storefront_api

    plain = create_app(context_provider=CTX, mount_mcp=False)

    assert "/channels" in _registered_paths(plain)
    assert not [p for p in _registered_paths(plain) if p.startswith("/shop")]
    for exc_type in (*storefront_api.CROSS_CUTTING_HANDLERS,
                     storefront_api.StorefrontHTTPError):
        assert exc_type not in plain.exception_handlers
    # `ServiceError` is the one handler the storefront *re-shapes* rather than
    # adds, so the check on it is the other way round: the default deployment
    # keeps `app.py`'s, untouched (`salesperson-ui-impl.md` `## Pass 10`, P10-1)
    handler = plain.exception_handlers[ServiceError]
    assert handler.__module__ == "falkorchat.app"


def test_the_storefront_error_map_refuses_to_be_wired_before_the_app_wide_one():
    """The storefront's `ServiceError` handler re-shapes on `/shop/api` and
    **delegates** everywhere else, so it needs the inherited handler to already
    be there — `create_app` registers them in that order.

    Asserted as a loud refusal at wiring time rather than left as a comment:
    the alternative failure mode is a `TypeError` raised *inside* an exception
    handler, on the first `ServiceError` a participant provokes, in a
    deployment that came up green.
    """
    from falkorchat import storefront_api

    with pytest.raises(RuntimeError, match="must run after"):
        storefront_api.register_storefront_error_handlers(FastAPI())
