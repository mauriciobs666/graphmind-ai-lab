"""Fixture tool module — imports only stdlib (plan §3.3's allowlist), and is actually executed by
`test_packs.py`'s `load_tool_module` test, so it cannot import `modelbench.tooling` (S2 ships that
module in a separate, concurrent unit; it does not exist yet)."""

import json
import os


def build_environment():
    return {"ok": True, "cwd": os.getcwd(), "json_module": json.__name__}
