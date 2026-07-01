"""One process, two front doors (plan §3.1, §6).

`create_app` builds the shared `Services` once and mounts both the REST router
(`api.py`) and the MCP Streamable-HTTP app (`mcp.py`) on a single FastAPI/uvicorn
process. The MCP app's lifespan MUST be forwarded to FastAPI or the MCP session
manager never initialises (python-sdk #1367).

Run:  uvicorn falkorchat.app:app   (agents connect at /mcp; REST under /)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import api, db
from . import mcp as mcp_mod
from .config import CallContext
from .repository import Repository
from .services import (
    ChannelNotFoundError,
    Services,
    ServiceError,
    ThreadNotFoundError,
)

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

    if mount_mcp:
        mcp_mod.configure(services, context_provider=context_provider)
        mcp_app = mcp_mod.mcp.streamable_http_app()
        # Forward the MCP app's lifespan or the session manager never inits
        # (python-sdk #1367). On this build the lifespan is exposed as the
        # Starlette router's `lifespan_context` (a callable taking the app).
        app = FastAPI(lifespan=mcp_app.router.lifespan_context)
    else:
        app = FastAPI()

    if context_provider is not None:
        app.dependency_overrides[api.get_context] = context_provider

    app.include_router(api.build_router(services))
    _register_error_handlers(app)

    if mount_mcp:
        app.mount("/mcp", mcp_app)

    # Static UI mounts LAST: "/" is a catch-all, so it must sit behind the REST
    # routes and the /mcp mount (Starlette matches routes in registration order).
    web = _DEFAULT_WEB_DIR if web_dir is None else web_dir
    if web.is_dir():
        app.mount("/", StaticFiles(directory=str(web), html=True), name="web")

    return app


# Default ASGI app for `uvicorn falkorchat.app:app`.
app = create_app()
