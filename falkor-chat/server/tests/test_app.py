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

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Mount

from falkorchat import config, db
from falkorchat.app import create_app
from falkorchat.config import CallContext
from falkorchat.repository import Repository
from falkorchat.services import MemberIdCollisionError, Services

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

    # WORKFLOW on: trigger wired (targets the agent + configured def), responder held
    # by the trigger (not passed to create_app → API schedules only the trigger).
    monkeypatch.setattr(app_mod.config, "WORKFLOW_ENABLED", True)
    monkeypatch.setattr(app_mod.config, "TRIGGER_DEF_KEY", "triage")
    monkeypatch.setattr(app_mod.config, "TRIGGER_DEF_VERSION", "v1")
    app_mod._build_default_app()
    assert captured.get("trigger") is not None
    assert captured.get("responder") is None
    trig = captured["trigger"]
    assert trig._agent_id == "assistant"
    assert trig._def_key == "triage"
    assert trig._responder is not None       # holds the responder for fall-through


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


def test_judge_prompt_survives_a_condition_with_no_evidence_at_all():
    # The degenerate case must still be a well-formed prompt, not a crash: the judge
    # then correctly biases to suspend (that behavior is Defect A's *symptom*, and is
    # the right answer when there genuinely is no evidence).
    from falkorchat.app import _render_judge_user

    user = _render_judge_user("enough info?", {}, [])

    assert user == "CONDITION: enough info?"
