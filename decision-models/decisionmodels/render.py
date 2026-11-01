"""The guard decision point's input rendering (`docs/plans/typed-decision-models.md` v1.3 §3.2,
held equal across components by §3.2.2's golden vectors).

v1.3 §3.2's contract table: the guard's decision-model input is the judge user message "exactly
as today" — falkor-chat's `_render_judge_user(condition, understanding, normalized_turns)`. This
module implements that contract in the standard library (FR-8 forbids importing model-bench or
falkor-chat; v1.3 §3.2.2 rejects a shared library). It is pinned by a byte-copy of model-bench's
`render-guard.jsonl` under `tests/fixtures/decision-contract/`, whose expected strings were
produced by falkor-chat's real function, never by this code.

The rendered message is up to three blocks joined by a blank line, in this order:

1. `CONDITION: {condition}` — always present;
2. `CURRENT STATE:\\n{json}` — only when `understanding` is non-empty; the JSON is
   `json.dumps(understanding, indent=2, sort_keys=True, default=str)` (non-ASCII escaped);
3. `RECENT TURNS (context only):\\n` followed by one `{speaker}: {text}` line per turn, oldest
   first — only when at least one turn is kept.

The whole message is capped at `MAX_CHARS`. Turns are dropped oldest first until the message fits,
so the newest turn survives whenever it can fit; when not even the newest turn fits, the message
is the first two blocks alone, hard-truncated to `MAX_CHARS`.

The turns the renderer receives are already normalized (`normalize_turns`, falkor-chat's
`guards._recent_turns` rule). That rule is not covered by a cross-component vector file; its
tests here are derived by hand from the rule as v1.3 §3.2.1 and model-bench state it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

#: The judge user message's character cap.
MAX_CHARS = 6000

#: `normalize_turns`: how many valid turns are kept (the newest), and each turn's text cap.
RECENT_TURNS_KEPT = 6
TURN_TEXT_MAX_CHARS = 400

_BLOCK_SEPARATOR = "\n\n"
_TURNS_HEADER = "RECENT TURNS (context only):\n"
_DEFAULT_SPEAKER = "member"


def normalize_turns(raw_turns: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Thread rows to the renderer's turns. Rows whose `text` is not a non-empty string are
    dropped **first**, then the last `RECENT_TURNS_KEPT` remain (filtering after slicing would let
    a malformed tail row shrink the window). `speaker` is `displayName`, else `authorId`, else
    `"member"`; `role` is `role`, else `"user"`; `text` is cut to `TURN_TEXT_MAX_CHARS`."""
    turns: list[dict[str, str]] = []
    for row in raw_turns:
        text = row.get("text")
        if not isinstance(text, str) or not text:
            continue
        speaker = row.get("displayName") or row.get("authorId") or _DEFAULT_SPEAKER
        role = row.get("role") or "user"
        turns.append(
            {"speaker": str(speaker), "role": str(role), "text": text[:TURN_TEXT_MAX_CHARS]}
        )
    return turns[-RECENT_TURNS_KEPT:]


def render_guard_input(
    condition: str, understanding: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]
) -> str:
    """The guard judge's user message over already-normalized `turns` (module docstring). A
    turn's `role` is never read; a missing `speaker` renders `"member"`, a missing `text` empty."""
    blocks = [f"CONDITION: {condition}"]
    if understanding:
        state = json.dumps(understanding, indent=2, sort_keys=True, default=str)
        blocks.append(f"CURRENT STATE:\n{state}")
    base = _BLOCK_SEPARATOR.join(blocks)

    lines = [f"{turn.get('speaker', _DEFAULT_SPEAKER)}: {turn.get('text', '')}" for turn in turns]
    for start in range(len(lines)):
        kept = lines[start:]
        message = f"{base}{_BLOCK_SEPARATOR}{_TURNS_HEADER}" + "\n".join(kept)
        if len(message) <= MAX_CHARS:
            return message
    return base[:MAX_CHARS]


def render_guard_item(item: Mapping[str, Any]) -> str:
    """A guard item's decision-model input: its `condition`, its `understanding` (absent or empty
    renders no state block) and its raw `turns`, normalized first (v1.3 §3.2's
    `normalized_turns`)."""
    turns = normalize_turns(item.get("turns") or [])
    return render_guard_input(item["condition"], item.get("understanding") or {}, turns)
