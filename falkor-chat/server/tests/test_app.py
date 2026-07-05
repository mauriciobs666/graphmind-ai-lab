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
