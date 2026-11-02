"""Thin HTTP client for the order service."""
import os

import requests


class ApiClient:
    def __init__(self):
        self.base = os.environ["FX_SVC_URL"]

    def post(self, path, body):
        return requests.post(self.base + path, json=body, timeout=10)
