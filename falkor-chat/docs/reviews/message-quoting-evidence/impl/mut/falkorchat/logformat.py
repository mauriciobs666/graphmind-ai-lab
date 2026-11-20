"""Log formatters named by the shipped `config/logging.json` (K-069).

uvicorn's own formatter classes, with one change: `%(asctime)s` renders as ISO 8601 local time
with milliseconds **and** the UTC offset (`2026-10-17T19:53:14.394-03:00`). The stdlib cannot do
that from a `datefmt` alone: `time.strftime` has no milliseconds, and `%(msecs)` can only be
placed after the whole `asctime`, so the offset would land before the milliseconds. The offset
matters because a container runs in UTC while the dev host does not, and graph timestamps are
UTC.

Nothing in the app imports this module. Only `logging.config.dictConfig` loads it, through
uvicorn's `--log-config`. It must therefore stay import-light: no `falkorchat` module beyond the
package itself.
"""

from __future__ import annotations

import logging
from datetime import datetime

from uvicorn.logging import AccessFormatter as _UvicornAccessFormatter
from uvicorn.logging import DefaultFormatter as _UvicornDefaultFormatter


class _IsoOffsetTime(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        if datefmt:
            return super().formatTime(record, datefmt)
        return datetime.fromtimestamp(record.created).astimezone().isoformat(
            timespec="milliseconds"
        )


class DefaultFormatter(_IsoOffsetTime, _UvicornDefaultFormatter):
    """`uvicorn.logging.DefaultFormatter` with ISO 8601 + offset timestamps."""


class AccessFormatter(_IsoOffsetTime, _UvicornAccessFormatter):
    """`uvicorn.logging.AccessFormatter` with ISO 8601 + offset timestamps."""
