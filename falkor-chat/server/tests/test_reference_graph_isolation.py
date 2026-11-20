"""GAL-I01-D0001 — the suites must never touch the live `reference` graph.

The reference-graph NAME is configurable (`config.REFERENCE_GRAPH`, env
`FALKORCHAT_REFERENCE_GRAPH`, default `reference`); `tests/conftest.py` points every
default pytest run at a throwaway name, and `scripts/test_queries.sh` does the same for
its own wipes. These tests fail if either suite would resolve the live name.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from falkorchat import config, db, querygen
from falkorchat.repository import Repository

LIVE_NAME = "reference"
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _config_value(env_value: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "FALKORCHAT_REFERENCE_GRAPH"}
    if env_value is not None:
        env["FALKORCHAT_REFERENCE_GRAPH"] = env_value
    out = subprocess.run(
        [sys.executable, "-c", "from falkorchat import config; print(config.REFERENCE_GRAPH)"],
        env=env, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_reference_graph_name_defaults_to_the_live_name_when_env_unset():
    assert _config_value(None) == LIVE_NAME


def test_reference_graph_name_is_overridable_by_env():
    assert _config_value("reference-xyz") == "reference-xyz"


class _Conn:
    def __init__(self) -> None:
        self.selected: list[str] = []

    def select_graph(self, name):
        self.selected.append(name)
        return name


def test_db_reference_graph_resolves_the_configured_name(monkeypatch):
    monkeypatch.setattr(config, "REFERENCE_GRAPH", "reference-probe")
    conn = _Conn()
    assert db.reference_graph(conn) == "reference-probe"
    assert Repository(conn)._reference() == "reference-probe"


# ── guards: the suites never resolve the live name ─────────────────────────────

def test_pytest_session_never_resolves_the_live_reference_graph():
    assert config.REFERENCE_GRAPH != LIVE_NAME
    conn = _Conn()
    db.reference_graph(conn)
    assert conn.selected == [config.REFERENCE_GRAPH]
    assert LIVE_NAME not in conn.selected


def test_catalog_dataset_targets_the_configured_reference_graph():
    assert querygen.CATALOG_SCHEMA.graph_key == config.REFERENCE_GRAPH


def test_test_queries_script_uses_a_throwaway_reference_graph():
    text = (_SCRIPTS / "test_queries.sh").read_text()
    match = re.search(r'^REF="([^"]+)"', text, re.M)
    assert match, "test_queries.sh must define REF"
    assert match.group(1) != LIVE_NAME
    assert "FALKORCHAT_REFERENCE_GRAPH" in text  # exported so bootstrap targets it too


def test_scripts_that_name_the_graph_honour_the_env_var():
    for name in ("bootstrap_schema.sh", "verify_catalog.sh"):
        text = (_SCRIPTS / name).read_text()
        assert "FALKORCHAT_REFERENCE_GRAPH" in text, name
