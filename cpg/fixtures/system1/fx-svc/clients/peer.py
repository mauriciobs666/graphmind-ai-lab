"""Client for the liveness endpoint of a peer instance."""
import os

import requests


def peer_health():
    return requests.get(os.environ.get("FX_PEER_BASE", "") + "/api/health", timeout=5)
