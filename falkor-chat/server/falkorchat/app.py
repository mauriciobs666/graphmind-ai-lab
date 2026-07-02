"""One process, two front doors (plan §3.1, §6).

`create_app` builds the shared `Services` once and mounts both the REST router
(`api.py`) and the MCP Streamable-HTTP app (`mcp.py`) on a single FastAPI/uvicorn
process. The MCP app's lifespan MUST be forwarded to FastAPI or the MCP session
manager never initialises (python-sdk #1367).

Run:  uvicorn falkorchat.app:app   (agents connect at /mcp; REST under /)
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import api, config, db
from . import mcp as mcp_mod
from .config import CallContext
from .repository import Repository
from .services import (
    ChannelNotFoundError,
    Services,
    ServiceError,
    ThreadNotFoundError,
)


class _McpPathAlias:
    """ASGI shim: rewrite bare ``/mcp`` to ``/mcp/`` (QA DEF-1).

    Starlette's Mount only serves the sub-app under ``/mcp/``; a client POSTing
    to the documented ``/mcp`` got 405 and the MCP python client does not
    auto-append the slash. Rewriting the path serves both spellings without
    relying on clients following redirects.
    """

    def __init__(self, app) -> None:  # noqa: ANN001 - ASGI app
        self._app = app

    async def __call__(self, scope, receive, send):  # noqa: ANN001 - ASGI signature
        if scope["type"] == "http" and scope.get("path") == "/mcp":
            scope = dict(scope, path="/mcp/", raw_path=b"/mcp/")
        await self._app(scope, receive, send)

# The minimal browser client (DESIGN §14.5) lives at the repo root `web/`,
# a sibling of `server/`. Served from this process so there is no CORS seam.
_DEFAULT_WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _handle_service_error(_request, exc: ServiceError):  # noqa: ANN001
        not_found = isinstance(exc, (ChannelNotFoundError, ThreadNotFoundError))
        return JSONResponse(
            status_code=404 if not_found else 400,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )


def create_app(
    services: Services | None = None,
    *,
    context_provider: Callable[[], CallContext] | None = None,
    mount_mcp: bool = True,
    web_dir: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    `services`/`context_provider` are injectable for tests (target `ws:test`).
    `mount_mcp=False` skips the MCP mount — used by REST tests so the FastMCP
    session manager (run-once per instance) isn't started repeatedly.
    `web_dir` overrides where the static browser client is served from; it
    defaults to the repo-root `web/` and is skipped if that directory is absent.
    """
    if services is None:
        services = Services(Repository(db.connect()))

    provider = context_provider or config.get_context

    if mount_mcp:
        mcp_mod.configure(services, context_provider=context_provider)
        mcp_app = mcp_mod.mcp.streamable_http_app()
        # Forward the MCP app's lifespan or the session manager never inits
        # (python-sdk #1367). On this build the lifespan is exposed as the
        # Starlette router's `lifespan_context` (a callable taking the app).
        mcp_lifespan = mcp_app.router.lifespan_context
    else:
        mcp_lifespan = None

    @asynccontextmanager
    async def _lifespan(app_: FastAPI):
        # The §4 write paths anchor on the author node — ensure the configured
        # actor exists before the first write (deferred to startup, not import,
        # so building the app never requires a reachable FalkorDB).
        services.ensure_actor(provider())
        if mcp_lifespan is not None:
            async with mcp_lifespan(app_):
                yield
        else:
            yield

    app = FastAPI(lifespan=_lifespan)

    if context_provider is not None:
        app.dependency_overrides[api.get_context] = context_provider

    app.include_router(api.build_router(services))
    _register_error_handlers(app)

    if mount_mcp:
        app.mount("/mcp", mcp_app)
        # Serve the documented slash-less spelling too (QA DEF-1).
        app.add_middleware(_McpPathAlias)

    # Static UI mounts LAST: "/" is a catch-all, so it must sit behind the REST
    # routes and the /mcp mount (Starlette matches routes in registration order).
    web = _DEFAULT_WEB_DIR if web_dir is None else web_dir
    if web.is_dir():
        app.mount("/", StaticFiles(directory=str(web), html=True), name="web")

    return app


# Default ASGI app for `uvicorn falkorchat.app:app`.
app = create_app()
