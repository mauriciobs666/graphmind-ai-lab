"""CLI entry point: `python -m mcp_monitor --config PATH [--log-level LEVEL]`.

Creates one shared `ServerConnection` per `[server.*]` block and one
`asyncio.Task` per `[[watch]]` (plan §3), all run concurrently for the process
lifetime. `SIGINT`/`SIGTERM` cancel the watch tasks; already-launched
subprocesses are left running/orphaned by design (out of scope: "what the
triggered command does once launched" — plan §3).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

from .client import ServerConnection
from .config import Config, ConfigError, load_config
from .logging_setup import configure_logging, get_watch_logger
from .watch import run_watch

_LOGGER = logging.getLogger("mcp_monitor")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mcp-monitor", description=__doc__)
    parser.add_argument("--config", required=True, help="path to the TOML config file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="root log level (default: INFO)",
    )
    return parser.parse_args(argv)


async def run(config: Config, *, stop: asyncio.Event | None = None) -> None:
    """Run every watch concurrently until `stop` is set (or, with none given, forever).

    `stop` is the test/embedding seam: production (`main()`) wires it to
    `SIGINT`/`SIGTERM`; a test can set it directly to bound a run.
    """
    connections = {
        name: ServerConnection(name, server_config)
        for name, server_config in config.servers.items()
    }
    tasks = [
        asyncio.create_task(
            run_watch(
                watch,
                connections[watch.server],
                server_name=watch.server,
                logger=get_watch_logger(watch=watch.name, server=watch.server, tool=watch.tool),
            ),
            name=f"mcp-monitor-watch:{watch.name}",
        )
        for watch in config.watches
    ]
    _LOGGER.info("started %d watch(es) over %d server(s)", len(tasks), len(connections))

    stop = stop if stop is not None else asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    try:
        await stop.wait()
    finally:
        _LOGGER.info("shutting down: cancelling %d watch task(s)", len(tasks))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for connection in connections.values():
            await connection.aclose()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        # Config errors are a hard startup failure — never a partially-running
        # process (plan §2).
        print(f"mcp-monitor: config error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not config.watches:
        print("mcp-monitor: config defines no [[watch]] entries — nothing to do", file=sys.stderr)
        raise SystemExit(1)

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
