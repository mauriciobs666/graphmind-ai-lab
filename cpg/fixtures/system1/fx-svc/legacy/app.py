"""Legacy fee service."""
from fastapi import FastAPI

app = FastAPI(title="fx-svc-legacy")


@app.get("/legacy/fee")
def fee():
    return {"fee": 0}
