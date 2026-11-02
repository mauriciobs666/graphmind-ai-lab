"""HTTP surface of the job: trigger a run, read its status."""
from fastapi import FastAPI

app = FastAPI()
_state = {"running": False, "runs": 0}


@app.post("/job/trigger")
def trigger():
    _state["running"] = True
    _state["runs"] += 1
    return {"accepted": True}


@app.get("/job/status")
def status():
    return {"running": _state["running"], "runs": _state["runs"]}
