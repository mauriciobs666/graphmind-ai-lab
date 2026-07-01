"""App-assembly tests: both front doors mounted on one FastAPI process.

Guards the python-sdk #1367 gotcha — if the MCP app's lifespan is not forwarded,
the session manager never initialises and requests to /mcp fail with a 500
("task group not initialized"). Here the endpoint returns a protocol-level
response instead, and REST keeps working alongside it.
"""

from __future__ import annotations

from starlette.routing import Mount

from fastapi.testclient import TestClient

from falkorchat.app import create_app
from falkorchat.config import CallContext
from falkorchat.repository import Repository
from falkorchat.services import Services

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
        # not a 404 (unmounted) or 500 (lifespan not forwarded).
        r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert r.status_code != 404
        assert r.status_code < 500
        # REST works on the same process
        assert c.get("/channels").status_code == 200


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
