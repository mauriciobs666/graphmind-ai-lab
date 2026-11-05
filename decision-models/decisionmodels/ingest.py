"""The answer ingester (plan `docs/plans/typed-decision-models4.md` §3.2 "Answer", §3.3's answer
rows and "Idempotence"; plan5 `typed-decision-models5.md` §3.5 "Schema first").

**An answer** is `exchange/ans/<opaque>.jsonl`, or parts `<opaque>.<k>.jsonl` (k = 1, 2, … with
no gap and no leading zero), concatenated in order (`read_answer_parts`). Line 1 is
`{"bodySha256": …}`, copied from the request header; then one line per key, in any order:
`{"key", "answer"}` (label: `answer` in the point's `expected`, type included, so `1` is not
`true`) or `{"key", "item"}` (generate: `item` exactly the point's input fields,
`points.input_field_errors`). A part that lacks a final newline ends its last line there. A CRLF
line ending reads as LF, so a `\r`-only line is a blank line (`malformed`) wherever it falls.

**Outcomes** of `ingest_answer`:
- `Ingested` — every key answered once, in space; `answers` maps key → answer or item.
- `Void` — run-attributable (plan4 §3.3), with one `reason` and every problem in `details`. The
  reason is the first that applies in this order: `answerMissing`, `malformed` (a part layout
  `read_answer_parts` cannot order, bytes that are not UTF-8, a line that is not one JSON object of
  the closed shape — a repeated name or a non-string key included — or a hash line after line 1),
  `bodyHashMissing`, `bodyHashMismatch`,
  `keyUnknown`, `keyDuplicate`, `outOfSpace`, `keyMissing`.
- `AlreadyIngested` — the request was ingested and these answer bytes hash the same: a no-op.
- `AnswerRefused` (raised) — the request was ingested and these bytes differ, valid or not: a
  second answer is refused and changes nothing.

`answer_sha256` is sha256 over the parts in order, each ended by a newline if it lacks one, so it
does not depend on how one answer is split into parts.

**Seams for later units.** The caller supplies what this module does not own: the request text
(moved to `store/` by U5.6's driver), the answer parts, and the ledger's record of an earlier
ingest (`ingested_answer_sha256`). Recording an outcome, moving files, re-issuing a void batch and
the transcript audit (U5.5), whose classes take precedence over these, are the caller's. A
generated item that passes here goes on to the screen and seed bindings (`screen.check_item`,
called from `Campaign.ingest`; plan5 §3.5).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exchange import RepeatedName, load_json_line, parse_request
from .points import Point, input_field_errors, load_point

RUN_ATTRIBUTABLE = "run-attributable"

ANSWER_MISSING = "answerMissing"
MALFORMED = "malformed"
BODY_HASH_MISSING = "bodyHashMissing"
BODY_HASH_MISMATCH = "bodyHashMismatch"
KEY_UNKNOWN = "keyUnknown"
KEY_DUPLICATE = "keyDuplicate"
OUT_OF_SPACE = "outOfSpace"
KEY_MISSING = "keyMissing"

#: The order in which one reason is chosen when several apply (module docstring).
REASON_ORDER = (
    ANSWER_MISSING, MALFORMED, BODY_HASH_MISSING, BODY_HASH_MISMATCH, KEY_UNKNOWN, KEY_DUPLICATE,
    OUT_OF_SPACE, KEY_MISSING,
)

_ANSWER_FIELD = {"label": "answer", "generate": "item"}


class AnswerRefused(ValueError):
    """A second, different answer to an ingested request."""


@dataclass(frozen=True)
class Ingested:
    answers: Mapping[str, Any]
    answer_sha256: str


@dataclass(frozen=True)
class AlreadyIngested:
    answer_sha256: str


@dataclass(frozen=True)
class Void:
    reason: str
    details: tuple[str, ...]
    answer_sha256: str
    void_class: str = field(default=RUN_ATTRIBUTABLE)


def read_answer_parts(ans_dir: Path, opaque: str) -> tuple[tuple[bytes, ...], str | None]:
    """The answer's parts in order, and a problem when their layout cannot be ordered: both a
    whole file and parts, a gap, a leading zero, or another `<opaque>.*` name. No file at all (or
    no directory) is `((), None)`."""
    if not ans_dir.is_dir():
        return (), None
    names = sorted(path.name for path in ans_dir.iterdir() if path.name.startswith(f"{opaque}."))
    if not names:
        return (), None
    part_pattern = re.compile(rf"{re.escape(opaque)}\.([1-9][0-9]*)\.jsonl")
    whole = f"{opaque}.jsonl"
    numbered: dict[int, str] = {}
    for name in names:
        match = part_pattern.fullmatch(name)
        if match:
            numbered[int(match.group(1))] = name
        elif name != whole:
            return (), f"unexpected answer file {name!r}"
    if whole in names:
        if numbered:
            return (), "both a whole answer and parts"
        return ((ans_dir / whole).read_bytes(),), None
    if sorted(numbered) != list(range(1, len(numbered) + 1)):
        return (), f"answer parts {sorted(numbered)} are not 1…{len(numbered)}"
    return tuple((ans_dir / numbered[k]).read_bytes() for k in sorted(numbered)), None


def answer_sha256(parts: Sequence[bytes]) -> str:
    """sha256 over the parts in order, each ended by a newline (module docstring)."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if part.endswith(b"\n") or not part else part + b"\n")
    return digest.hexdigest()


def in_answer_space(point: Point, answer: Any) -> bool:
    """Whether `answer` is one of the point's `expected` values, of the same JSON type."""
    return any(type(answer) is type(value) and answer == value for value in point.expected)


def _part_lines(parts: Sequence[bytes]) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    for number, part in enumerate(parts, 1):
        try:
            text = part.decode("utf-8")
        except UnicodeDecodeError:
            return [], [f"part {number} is not UTF-8"]
        if text.endswith("\n"):
            text = text[:-1]
        if text:
            lines += [line.removesuffix("\r") for line in text.split("\n")]
    return lines, []


def _check_entries(
    kind: str, point: Point, keys: Sequence[str], entries: Sequence[str]
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    answer_field = _ANSWER_FIELD[kind]
    answers: dict[str, Any] = {}
    problems: list[tuple[str, str]] = []
    known = set(keys)
    for number, raw in enumerate(entries, 2):
        where = f"answer line {number}"
        try:
            line = load_json_line(raw)
        except (ValueError, RecursionError) as exc:
            problems.append((MALFORMED, f"{where}: not one JSON object ({exc})"))
            continue
        if not isinstance(line, dict) or set(line) != {"key", answer_field}:
            problems.append((MALFORMED, f"{where}: not {{key, {answer_field}}}"))
            continue
        key, value = line["key"], line[answer_field]
        if not isinstance(key, str):
            problems.append((MALFORMED, f"{where}: key {key!r} is not a string"))
            continue
        if key not in known:
            problems.append((KEY_UNKNOWN, f"{where}: unknown key {key!r}"))
            continue
        if key in answers:
            problems.append((KEY_DUPLICATE, f"{where}: duplicate key {key!r}"))
            continue
        answers[key] = value
        if kind == "label":
            if not in_answer_space(point, value):
                problems.append((OUT_OF_SPACE, f"{where}: {value!r} is not in the answer space"))
        else:
            problems += [(OUT_OF_SPACE, f"{where}: {error}")
                         for error in input_field_errors(point, value)]
    problems += [(KEY_MISSING, f"key {key!r} is not answered") for key in keys
                 if key not in answers]
    return answers, problems


def _void(problems: Sequence[tuple[str, str]], sha: str) -> Void:
    reason = min((reason for reason, _ in problems), key=REASON_ORDER.index)
    return Void(reason=reason, details=tuple(detail for _, detail in problems), answer_sha256=sha)


def ingest_answer(
    request_text: str, parts: Sequence[bytes], *, ingested_answer_sha256: str | None
) -> Ingested | AlreadyIngested | Void:
    """Ingest `parts` against `request_text` (module docstring). `ingested_answer_sha256` is the
    ledger's record of an earlier ingest of this request, or `None`. An invalid request text is
    `exchange.RequestRefused`, never a void: the request is the driver's own file."""
    request = parse_request(request_text)
    sha = answer_sha256(parts)
    if ingested_answer_sha256 is not None:
        if sha == ingested_answer_sha256:
            return AlreadyIngested(answer_sha256=sha)
        raise AnswerRefused(
            f"the request is already ingested with answer {ingested_answer_sha256[:12]}…; a "
            f"second, different answer ({sha[:12]}…) is refused"
        )
    if not parts:
        return _void([(ANSWER_MISSING, "no answer file")], sha)
    lines, problems = _part_lines(parts)
    if problems:
        return _void([(MALFORMED, problem) for problem in problems], sha)
    if any(not line for line in lines):
        return _void([(MALFORMED, "a blank answer line")], sha)
    first = _first_line(lines[0]) if lines else None
    if first is None:
        return _void([(BODY_HASH_MISSING, "line 1 is not {bodySha256}")], sha)
    if isinstance(first, str):
        return _void([(MALFORMED, first)], sha)
    if first["bodySha256"] != request.body_sha256:
        return _void([(BODY_HASH_MISMATCH, "bodySha256 does not match the request")], sha)
    point = load_point(request.point)
    answers, problems = _check_entries(request.kind, point, request.keys, lines[1:])
    if problems:
        return _void(problems, sha)
    return Ingested(answers=answers, answer_sha256=sha)


def _first_line(raw: str) -> dict[str, Any] | str | None:
    """Line 1 as `{"bodySha256": str}`; `None` when it is not a hash line at all (the line is
    missing), or a problem text when it is one of the wrong shape."""
    try:
        line = load_json_line(raw)
    except RepeatedName as exc:
        return f"answer line 1: {exc}"
    except (ValueError, RecursionError):
        return None
    if not isinstance(line, dict) or "bodySha256" not in line:
        return None
    if set(line) != {"bodySha256"} or not isinstance(line["bodySha256"], str):
        return "answer line 1: not exactly {bodySha256: <string>}"
    return line


def ingest_answer_files(
    request_text: str, ans_dir: Path, opaque: str, *, ingested_answer_sha256: str | None
) -> Ingested | AlreadyIngested | Void:
    """`ingest_answer` over the parts `read_answer_parts` finds in `ans_dir`; a part layout it
    cannot order is a `malformed` void (or, for an ingested request, a refused second answer)."""
    parts, problem = read_answer_parts(ans_dir, opaque)
    if problem is not None and ingested_answer_sha256 is None:
        parse_request(request_text)
        return _void([(MALFORMED, problem)], answer_sha256(parts))
    return ingest_answer(request_text, parts, ingested_answer_sha256=ingested_answer_sha256)
