"""Shared fixtures. `fake_claude` puts a `claude` first on `PATH` that answers only `--version`
(the version floor, plan6 §3.3), so `driver install` and the runner's version check never reach
the real CLI. Its control directory is baked into the script, because the child environment is an
allowlist."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

SCRIPT = """#!/bin/sh
cat {version!r}
exit {code}
"""


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Returns `set_version(text, code=0)`: what `claude --version` prints and its exit code, or
    `None` for no binary."""
    bin_dir = tmp_path / "fake-claude-bin"
    bin_dir.mkdir()
    version = tmp_path / "fake-claude-version"
    binary = bin_dir / "claude"
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    def set_version(text: str | None, code: int = 0) -> None:
        if text is None:
            binary.unlink(missing_ok=True)
            return
        version.write_text(text, "utf-8")
        binary.write_text(SCRIPT.format(version=str(version), code=code), "utf-8")
        binary.chmod(0o755)

    set_version("2.1.316 (Claude Code)\n")
    return set_version
