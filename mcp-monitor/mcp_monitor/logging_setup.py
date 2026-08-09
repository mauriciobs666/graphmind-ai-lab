"""Logging (plan §7) — one line per event, to stdout, `watch=`/`server=`/`tool=` tagged.

`grep watch=<name>` isolates one watch's whole history — that is what "an
operator can verify after the fact that a watch is working" (FR-11) means
concretely. Every watch gets its own `LoggerAdapter` carrying that context;
lines logged outside a watch's adapter (startup, shutdown) still go through
the same formatter, with `-` standing in for the missing fields.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = (
    "%(asctime)s %(levelname)-7s watch=%(watch)s server=%(server)s tool=%(tool)s %(message)s"
)

_CONTEXT_KEYS = ("watch", "server", "tool")


class _DefaultContextFilter(logging.Filter):
    """Fill in `watch`/`server`/`tool` with `-` for records logged without a `LoggerAdapter`."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key in _CONTEXT_KEYS:
            if not hasattr(record, key):
                setattr(record, key, "-")
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, at process startup (`__main__.py`).

    stdout, not a file — 12-factor-friendly, no rotation (out of scope's
    "production hardening"); operators redirect/tee as they see fit.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(_DefaultContextFilter())

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers = [handler]


def get_watch_logger(*, watch: str, server: str, tool: str) -> logging.LoggerAdapter:
    """A `LoggerAdapter` that stamps every line from one watch with its identity (FR-11)."""
    base = logging.getLogger("mcp_monitor.watch")
    return logging.LoggerAdapter(base, {"watch": watch, "server": server, "tool": tool})
