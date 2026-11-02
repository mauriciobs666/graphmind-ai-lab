"""Client for the job service."""
import os

import requests

JOB_URL = os.environ.get("FX_JOB_URL", "http://localhost:8300")

_REGISTRY: dict = {}
ROUTES = _REGISTRY.get("routes", {})


def get_base():
    return _REGISTRY["base"]


def trigger_job(payload):
    return requests.post(f"{JOB_URL}/job/trigger", json=payload, timeout=10)


def call_named(name):
    return requests.get(f"{get_base()}/{ROUTES[name]}", timeout=10)
