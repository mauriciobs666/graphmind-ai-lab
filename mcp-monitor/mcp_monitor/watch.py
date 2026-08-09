"""Per-watch poll/match/dedupe loop (plan §3, §6).

One instance of `run_watch()` per `[[watch]]`, run as its own `asyncio.Task`
for the process lifetime. A watch's own polling is strictly sequential — poll,
sleep `interval_seconds`, poll again, never two polls in flight for the same
watch — but launched commands run fully in parallel across watches and across
matches within one watch (FR-9/AC-6): a match schedules its launch as a new,
un-awaited task and moves straight on to the next match / next poll.

`find_matches`/`filter_new` are pulled out as pure functions (no transport, no
asyncio) specifically so `tests/test_matcher.py` can exercise the regex/dedupe
logic in isolation, per the plan's own test-tier description (§11 tier 1).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable

from . import launcher
from .client import ServerConnection
from .config import WatchConfig


def find_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    """Every non-overlapping match's text (plan §3 step 3) — not just the first.

    A single poll can legitimately contain several new items (e.g. two
    messages posted between polls, both matching); each is a separate
    candidate for dedupe/launch.
    """
    return [m.group(0) for m in pattern.finditer(text)]


def filter_new(matches: Iterable[str], seen: set[str], *, repeat_trigger: bool) -> list[str]:
    """Which of `matches` should fire, given `repeat_trigger` and matches already `seen`.

    Dedupe key = the exact matched substring (plan §6) — generic across
    arbitrary tools, no assumption that a message id or any id is present.
    `repeat_trigger = True` fires every match every time and never touches
    `seen`; `repeat_trigger = False` fires a given matched substring at most
    once for the life of this set, mutating `seen` in place as it goes.
    """
    to_fire: list[str] = []
    for text in matches:
        if repeat_trigger:
            to_fire.append(text)
            continue
        if text in seen:
            continue
        seen.add(text)
        to_fire.append(text)
    return to_fire


async def run_watch(
    watch: WatchConfig,
    connection: ServerConnection,
    *,
    server_name: str,
    logger: logging.LoggerAdapter | logging.Logger,
    launch=launcher.launch,
    seen: set[str] | None = None,
) -> None:
    """Run `watch` forever: poll, match, dedupe, launch, sleep, repeat.

    Runs until cancelled (`__main__.py`'s signal handling cancels every watch
    task on shutdown) — this coroutine has no natural end. `seen` is
    injectable so tests can pre-seed or inspect dedupe state; production
    callers leave it `None` and get a fresh set per watch.
    """
    seen = set() if seen is None else seen
    while True:
        try:
            raw_text = await connection.call_tool(watch.tool, watch.args)
        except Exception as exc:  # noqa: BLE001 — a poll failure must never stop the watch
            # FR-10/AC-7: log and retry next cycle, never raise out of the loop.
            logger.warning("poll failed: %s: %s", type(exc).__name__, exc)
        else:
            matches = find_matches(watch.pattern, raw_text)
            to_fire = filter_new(matches, seen, repeat_trigger=watch.repeat_trigger)
            suppressed = len(matches) - len(to_fire)
            if suppressed:
                # DEBUG, not silence — a correctly-deduping watch should not
                # look silent under normal INFO-level operation (plan §7).
                logger.debug("suppressed %d repeat match(es)", suppressed)
            for matched_text in to_fire:
                logger.info("match found: %r", matched_text)
                # Not awaited: a slow/long-running launched process must never
                # block this watch's next poll (FR-9/AC-6).
                asyncio.create_task(
                    launch(watch, server_name, raw_text, matched_text, logger=logger),
                    name=f"mcp-monitor-launch:{watch.name}",
                )

        await asyncio.sleep(watch.interval_seconds)
