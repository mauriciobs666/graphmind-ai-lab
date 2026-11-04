"""Application factory: mounts every router under one prefix."""
from fastapi import FastAPI

from fx_svc.api import files, health, hooks, items, orders, reports, search

API_PREFIX = "/api"


def create_app():
    app = FastAPI(title="fx-svc")
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(files.router, prefix=API_PREFIX)
    app.include_router(hooks.router, prefix=API_PREFIX)
    app.include_router(items.router, prefix=API_PREFIX)
    app.include_router(orders.router, prefix=API_PREFIX)
    app.include_router(reports.router, prefix=API_PREFIX)
    app.include_router(search.router, prefix=API_PREFIX)
    return app
