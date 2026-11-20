"""Application factory: mounts every router under one prefix."""
from fastapi import FastAPI

from api import files, health, hooks, items, orders, reports, search

API_PREFIX = "/api"


def create_app():
    app = FastAPI(title="fx-svc")
    app.include_router(health.build_router(), prefix=API_PREFIX)
    app.include_router(files.build_router(), prefix=API_PREFIX)
    app.include_router(hooks.build_router(), prefix=API_PREFIX)
    app.include_router(items.build_router(), prefix=API_PREFIX + "/items")
    app.include_router(orders.build_router(), prefix=API_PREFIX)
    app.include_router(reports.build_router(), prefix=API_PREFIX)
    app.include_router(search.build_router(), prefix=API_PREFIX)
    return app
