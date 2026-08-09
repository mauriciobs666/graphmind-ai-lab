"""mcp-monitor — a generic MCP tool-result watcher that launches a command on match.

See `mcp-monitor/docs/plans/mcp-monitor.md` for the design and
`mcp-monitor/README.md` for usage. Package layout mirrors that plan's §10:

- `config.py`   — TOML load + validation -> `Config`/`ServerConfig`/`WatchConfig`
- `client.py`   — per-server `ClientSession` lifecycle, result flattening
- `watch.py`    — per-watch poll/match/dedupe loop
- `launcher.py` — subprocess launch: stdin JSON + env identifiers, exit-code logging
- `logging_setup.py` — per-watch `LoggerAdapter` stdout logging
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
