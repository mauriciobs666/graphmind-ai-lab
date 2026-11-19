"""The hand-off reply parse (plan4 §3.11, §4.4 U6f.3a): what a hand-off model's raw reply means.

A standalone copy of model-bench's `extract_own_line_json_object` and verdict rules (FR-8: no
import), pinned by byte-copies of `handoff-parse-{guard,join-in,message-type}.jsonl`
(`tests/fixtures/decision-contract/`, asserted in `tests/test_handoff_parse.py`). Pure: no I/O.

Every parse is biased to find nothing: only a real bool `True` advances or replies, a label must
match the answer space exactly, and a reply with no single qualifying JSON object is a parse
failure. Each returns `(verdict, parse_failure)`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from decisionmodels.points import ROLES

#: The hand-off call's parity constants: fit and eval send the same budget as the bench pack
#: (`prompt.temperature`/`maxTokens`) and `decisions.DECISION_POINT_CHAT_TIMEOUT_SECONDS`,
#: pinned by `tests/test_handoff_parity.py`.
HANDOFF_TEMPERATURE = 0.0
HANDOFF_MAX_TOKENS = 256
HANDOFF_CHAT_TIMEOUT_SECONDS = 180.0

_FIT_DIR = Path(__file__).resolve().parents[1] / "fit"

GUARD_KEY = "decision"
JOIN_IN_KEY = "reply"
MESSAGE_TYPE_KEY = "type"

#: Deficiency assertions contradicting a `decision: true` when un-negated.
_NEGATION_CUES: tuple[str, ...] = (
    "not enough", "insufficient", "still missing", "still need", "cannot ", "unable",
    "unclear", "not yet", "more info", "need more",
)
#: A cue immediately preceded by one of these, in the same clause, affirms.
_NEGATORS: tuple[str, ...] = ("no ", "not ", "nothing ", "never ", "n't ")
#: How many chars before a cue a negator may start.
_NEGATOR_WINDOW = 12
#: A negator on the far side of one of these never reaches the cue.
_CLAUSE_BOUNDARY: frozenset[str] = frozenset(";.,")

#: A JSON object opening a line: optional indentation, then `{`.
_OWN_LINE_OBJECT_OPEN = re.compile(r"^[ \t]*\{", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    """Strip one surrounding ```/```json fence, if present."""
    if not text.startswith("```"):
        return text
    inner = text[3:]
    if inner[:4].lower() == "json":
        inner = inner[4:]
    return inner.rsplit("```", 1)[0].strip()


def _load_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _scan_balanced(text: str, open_idx: int) -> int | None:
    """The index after the `}` closing the `{` at `open_idx`, ignoring braces inside JSON string
    literals; `None` on unbalanced input."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_own_line_json_object(
    content: Any, *, require_key: str | None = None
) -> dict[str, Any] | None:
    """The conservative parse, biased to find nothing.

    Accepts, after stripping one surrounding code fence: a reply that is entirely one JSON object,
    or exactly one JSON object that owns its lines (its `{` is the first non-whitespace character
    of a line, nothing but whitespace follows its matching `}` on that line) and carries
    `require_key` when given. Anything else (an object quoted mid-sentence, two or more qualifying
    objects, nothing that parses, non-string content) is `None`."""
    if not isinstance(content, str):
        return None
    text = _strip_code_fence(content.strip())
    if not text:
        return None

    whole = _load_json_object(text)
    if whole is not None:
        return whole

    matches: list[dict[str, Any]] = []
    consumed = 0
    for match in _OWN_LINE_OBJECT_OPEN.finditer(text):
        if match.start() < consumed:
            continue  # nested inside an object already scanned
        after = _scan_balanced(text, match.end() - 1)
        if after is None:
            continue
        consumed = after
        if text[after:].split("\n", 1)[0].strip():
            continue  # trailing prose on the closing line: quoted, not asserted
        parsed = _load_json_object(text[match.end() - 1 : after])
        if parsed is None or (require_key is not None and require_key not in parsed):
            continue
        matches.append(parsed)

    return matches[0] if len(matches) == 1 else None


def _is_negated(low: str, cue_start: int) -> bool:
    """A negator in the `_NEGATOR_WINDOW` chars before the cue, after the last clause boundary in
    that window, negates it."""
    window = low[max(0, cue_start - _NEGATOR_WINDOW) : cue_start]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in _CLAUSE_BOUNDARY:
            window = window[i + 1 :]
            break
    return any(neg in window for neg in _NEGATORS)


def _rationale_contradicts(rationale: str) -> bool:
    """True when any occurrence of a `_NEGATION_CUES` cue in the lower-cased rationale is not
    negated."""
    low = rationale.lower()
    for cue in _NEGATION_CUES:
        start = low.find(cue)
        while start != -1:
            if not _is_negated(low, start):
                return True
            start = low.find(cue, start + 1)
    return False


def _coerce_verdict(raw: Mapping[str, Any]) -> bool:
    """Only a real bool `True` `decision` (not `"true"`, not `1`) with a rationale that does not
    contradict it advances; a non-string rationale reads as empty."""
    rationale = raw.get("rationale")
    rationale = rationale if isinstance(rationale, str) else ""
    if raw.get(GUARD_KEY) is not True:
        return False
    return not _rationale_contradicts(rationale)


def parse_guard_handoff(content: Any) -> tuple[bool, bool]:
    """`(advance, parse_failure)` for the guard's `{"decision": bool, "rationale": str}`."""
    parsed = extract_own_line_json_object(content, require_key=GUARD_KEY)
    if parsed is None:
        return False, True
    return _coerce_verdict(parsed), False


def parse_join_in_handoff(content: Any) -> tuple[bool, bool]:
    """`(reply, parse_failure)`; only a real bool `True` replies (fail-closed)."""
    parsed = extract_own_line_json_object(content, require_key=JOIN_IN_KEY)
    if parsed is None:
        return False, True
    return parsed.get(JOIN_IN_KEY) is True, False


def parse_message_type_handoff(
    content: Any, labels: Sequence[str]
) -> tuple[str | None, bool]:
    """`(label, parse_failure)`; the value must be one of `labels` exactly (case and whitespace
    included), else no label (`None`). `labels` is the point's answer space, never guessed."""
    parsed = extract_own_line_json_object(content, require_key=MESSAGE_TYPE_KEY)
    if parsed is None:
        return None, True
    value = parsed.get(MESSAGE_TYPE_KEY)
    return (value if isinstance(value, str) and value in labels else None), False


def handoff_messages(role: str, rendered_input: str) -> list[dict[str, str]]:
    """The hand-off chat messages, as the bench's `build_messages`: system = the role's
    `fit/<role>/handoff.md` verbatim, user = the rendering. An unknown role raises `ValueError`."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    system = (_FIT_DIR / role / "handoff.md").read_text(encoding="utf-8")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": rendered_input},
    ]
