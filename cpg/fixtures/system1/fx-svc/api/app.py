"""Application factory: mounts every router under one prefix."""
from fastapi import FastAPI

from fx_svc.api import files, health, hooks, items, orders, reports, search

API_PREFIX = "/api"


def create_app():
    app = FastAPI(title="fx-svc")
    for module in (health, files, hooks, items, orders, reports, search):
        app.include_router(module.router, prefix=API_PREFIX)
    return app
