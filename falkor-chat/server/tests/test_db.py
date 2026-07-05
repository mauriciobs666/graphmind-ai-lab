"""Connection-layer tests (db.py) — DEF-2: fail fast, never hang.

`connect` must carry socket timeouts (config-resolved like FALKORDB_HOST/PORT)
so an unreachable FalkorDB — including a WSL2 blackholed dead port, which never
sends a RST — surfaces as a fast, clear error instead of a silent multi-minute
hang (the falkordb-py constructor issues an eager command, so the failure
belongs to `connect` itself).
"""

from __future__ import annotations

import time

import pytest

from falkorchat import config, db

# Non-routable address: SYNs blackhole (no refusal) — exactly the WSL2
# dead-port failure mode DEF-2 is about; only the connect timeout bounds it.
BLACKHOLE = "10.255.255.1"


def test_config_resolves_socket_timeout_defaults():
    # short sane defaults, overridable via FALKORDB_* env like host/port
    assert 0 < config.FALKORDB_CONNECT_TIMEOUT <= 30
    assert 0 < config.FALKORDB_SOCKET_TIMEOUT <= 60


def test_connect_passes_config_socket_timeouts(_schema, monkeypatch):
    monkeypatch.setattr(config, "FALKORDB_CONNECT_TIMEOUT", 1.5, raising=False)
    monkeypatch.setattr(config, "FALKORDB_SOCKET_TIMEOUT", 2.5, raising=False)

    conn = db.connect()

    kw = conn.connection.connection_pool.connection_kwargs
    assert kw["socket_connect_timeout"] == 1.5
    assert kw["socket_timeout"] == 2.5


def test_connect_unreachable_fails_within_timeout_budget_with_clear_error(monkeypatch):
    monkeypatch.setattr(config, "FALKORDB_HOST", BLACKHOLE)
    monkeypatch.setattr(config, "FALKORDB_PORT", 6399)
    monkeypatch.setattr(config, "FALKORDB_CONNECT_TIMEOUT", 1.0, raising=False)

    t0 = time.monotonic()
    with pytest.raises(db.FalkorDBUnreachableError) as exc:
        db.connect()
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0  # bounded by the connect timeout, not a ~90s+ hang
    assert f"{BLACKHOLE}:6399" in str(exc.value)  # actionable: names host:port
