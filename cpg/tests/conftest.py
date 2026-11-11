"""Test-mode safety net: no test may read or write the real credentials/key area (``~/.config/ccg``).

Every test runs with ``CCG_FORBID_REAL_CREDENTIALS=1`` (the marker lives in the environment, so child processes and ``importlib.reload`` keep
it) and with ``CCG_CREDENTIALS_DIR``/``CCG_CONFIG_DIR`` at a per-test temporary directory. The guard itself is
``ccg.atrest.assert_not_real_credentials``, the choke point every key-area and credentials operation passes through; nothing is patched here.
``tests/test_env_guard.py`` pins it.
"""
from __future__ import annotations

import pytest

from ccg import atrest


@pytest.fixture(autouse=True)
def _no_real_credentials_dir(monkeypatch, tmp_path_factory):
    env = tmp_path_factory.mktemp("ccgenv")
    monkeypatch.setenv(atrest.FORBID_REAL_ENV, "1")
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(env / "credentials"))
    monkeypatch.setenv("CCG_CONFIG_DIR", str(env / "config"))
