"""Operator tool runner."""
from fastapi import FastAPI

app = FastAPI(title="fx-svc-tools")


@app.get("/tools/run")
def run():
    return {"ran": True}
