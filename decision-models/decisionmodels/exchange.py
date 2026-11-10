"""The request writer, the closed request schemas and the answer parser (plan
`docs/plans/typed-decision-models4.md` §3.2; plan5 `typed-decision-models5.md` §3.4; method note
`docs/plans/typed-decision-models-ml.md` §Q1.2 "Blind", "Batches").

**A request** is `exchange/req/<opaque>.jsonl`, JSON Lines so that Read can page it:

1. the header — `{"schema": "dm-request/1", "role": "label"|"generate", "point", "answerPath",
   "bodySha256", "pages"}`;
2. `{"rubric"}` (label: `Point.rubric`) or `{"instructions"}` (generate:
   `Point.generator_instructions`);
3. then one line per item, `{"key", "text"}` with `text` the item's rendering (`RENDERERS`), or one
   line per seed, `{"key", <the seed's text axes>}`.

Every line is a closed key set, checked by `request_errors`; the writer runs it before handing a
request out and again before writing one. **Generation lines are id-blind** (plan5 §3.4): the
writer projects every `*Id` key out of `Seed.axes` (the store keeps them, by request key, in
`BuiltRequest.keys`), and a `*Id` key on any line of a generation request is refused whatever
`points/` declares.

`bodySha256` is sha256 over the UTF-8 canonical JSON (`sort_keys`, `(",", ":")` separators,
non-ASCII raw) of the body `{"rubric"|"instructions": …, "items"|"seeds": [line, …]}`, so it does
not depend on the file layout: key order within a line, separators or paging.

**Paging.** Lines 2… are cut greedily into pages of at most `PAGE_BUDGET_CHARS` written
characters — each line's characters after JSON escaping plus its newline — listed in the header
as Read `[offset, limit]` pairs after the header's own `[1, 1]`. A batch is cut at its item cap or
at `P_MAX` pages (the header page included), whichever comes first; what does not fit is
`BuiltRequest.overflow`, for the next batch of that vote.

**Line breaks.** A line is written with `ensure_ascii=False`, which leaves U+0085, U+2028 and
U+2029 raw; a reader that breaks lines on them would misalign every page offset. The writer
escapes them (the parsed value, and so the body hash, is unchanged), and `request_errors` refuses
a raw one.

**Keys** are `KEY_LENGTH` base32 characters from `secrets`, minted per (item, vote) and per seed
against the caller's held set and redrawn on collision; a rerun gets fresh keys and a new opaque
name. Holding every key minted in a campaign is the caller's (U5.6's ledger).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from . import render
from .points import ROLES, Point, load_point
from .seeds import Seed

SCHEMA = "dm-request/1"

#: Written characters per page: about 20,100 tokens at the measured 2.0 characters per token,
#: under Read's 25,000-token cap (plan4 §2, §3.2).
PAGE_BUDGET_CHARS = 40_000

#: Pages per request, the header page included (plan4 §3.2).
P_MAX: Mapping[str, int] = MappingProxyType({"label": 10, "generate": 2})

KEY_ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"
KEY_LENGTH = 10
OPAQUE_LENGTH = 16
KEY_PATTERN = re.compile(f"[{KEY_ALPHABET}]{{{KEY_LENGTH}}}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

#: The label renderer per point.
RENDERERS: Mapping[str, Callable[[Mapping[str, Any]], str]] = MappingProxyType(
    {
        "guard-judge": render.render_guard_item,
        "join-in-judge": render.render_join_in_item,
        "message-type-classifier": render.render_message_type_item,
    }
)

HEADER_KEYS = ("schema", "role", "point", "answerPath", "bodySha256", "pages")
_LINE_2 = {"label": "rubric", "generate": "instructions"}
_ENTRIES = {"label": "items", "generate": "seeds"}
_ENTRY_KIND = {"label": "item", "generate": "seed"}

_RAW_LINE_BREAKS = {"\x85": "\\u0085", " ": "\\u2028", " ": "\\u2029"}
_ID_SUFFIX = "Id"


class RequestRefused(ValueError):
    """A request text that breaks its schema; `errors` lists every reason."""

    def __init__(self, errors: Sequence[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = tuple(errors)


@dataclass(frozen=True)
class BuiltRequest:
    """A written-to-be request. `keys` maps each request key to what the store keeps for it: the
    caller's item reference (label) or the whole `Seed`, `*Id` axes included (generate).
    `overflow` is what did not fit: `(reference, item)` pairs or seeds, in input order."""

    kind: str
    point: str
    opaque: str
    request_path: Path
    answer_path: Path
    text: str
    keys: Mapping[str, Any]
    overflow: tuple[Any, ...]

    @property
    def request_sha256(self) -> str:
        """sha256 of the request file's bytes (plan4 §3.7 `runs[].requestSha256`)."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Request:
    """A parsed, valid request: `entries` are lines 3… as read, `keys` their keys in order."""

    kind: str
    point: str
    answer_path: str
    body_sha256: str
    pages: tuple[tuple[int, int], ...]
    line_2: str
    entries: tuple[Mapping[str, Any], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(entry["key"] for entry in self.entries)


# --- keys ----------------------------------------------------------------------------------------


def _random_token(length: int) -> str:
    return "".join(secrets.choice(KEY_ALPHABET) for _ in range(length))


def mint_key(held: Collection[str]) -> str:
    """A fresh key not in `held`, redrawn on collision."""
    while True:
        key = _random_token(KEY_LENGTH)
        if key not in held:
            return key


def mint_opaque_name() -> str:
    """A request's opaque file stem: it names nothing about the batch."""
    return _random_token(OPAQUE_LENGTH)


# --- serialization -------------------------------------------------------------------------------


def dump_line(value: Mapping[str, Any]) -> str:
    """One request line, newline excluded: compact JSON, non-ASCII raw except the line breaks a
    reader could split on (module docstring)."""
    line = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    for raw, escaped in _RAW_LINE_BREAKS.items():
        line = line.replace(raw, escaped)
    return line


def body_sha256(kind: str, line_2: Any, entries: Sequence[Any]) -> str:
    """plan4 §3.2's body hash (module docstring)."""
    body = {_LINE_2[kind]: line_2, _ENTRIES[kind]: list(entries)}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cut_pages(
    line_lengths: Sequence[int], *, budget: int, max_pages: int
) -> tuple[list[list[int]], int]:
    """Greedy pages over lines 2… (`line_lengths` excludes newlines): the `[offset, limit]` list,
    the header's `[1, 1]` first, and how many of the lines fit within `max_pages` pages."""
    pages = [[1, 1]]
    used = 0
    for index, length in enumerate(line_lengths):
        written = length + 1
        if written > budget:
            raise ValueError(f"line {index + 2} alone holds {written} characters, over the "
                             f"page budget {budget}")
        if len(pages) > 1 and used + written <= budget:
            pages[-1][1] += 1
            used += written
            continue
        if len(pages) == max_pages:
            return pages, index
        pages.append([index + 2, 1])
        used = written
    return pages, len(line_lengths)


# --- the writer ----------------------------------------------------------------------------------


def seed_axis_names(point: Point) -> tuple[str, ...]:
    """Every seed axis the point's design declares, in design order."""
    design = point.design
    groups = ("stratified", "balanced", "conditional")
    return tuple(axis["axis"] for group in groups for axis in design[group])


def blind_axes(axes: Mapping[str, Any]) -> dict[str, Any]:
    """A seed's axes with every `*Id` key projected out (plan5 §3.4)."""
    return {name: value for name, value in axes.items() if not name.endswith(_ID_SUFFIX)}


def _build(
    kind: str,
    point: Point,
    line_2: str,
    capped: Sequence[tuple[Any, Any, dict[str, Any]]],
    rest: Sequence[Any],
    exchange_dir: Path,
    held_keys: Collection[str],
) -> BuiltRequest:
    """`capped` holds `(the input entry, what the store keeps, the line minus its key)` per entry
    within the cap; `rest` is past it already."""
    if not capped:
        raise ValueError(f"a request with no {_ENTRIES[kind]} costs a spawn for nothing")
    minted: set[str] = set()
    entries: list[dict[str, Any]] = []
    for _, _, fields in capped:
        key = mint_key(minted | set(held_keys))
        minted.add(key)
        entries.append({"key": key, **fields})
    lines = [dump_line({_LINE_2[kind]: line_2})] + [dump_line(entry) for entry in entries]
    pages, fitted = cut_pages(
        [len(line) for line in lines], budget=PAGE_BUDGET_CHARS, max_pages=P_MAX[kind]
    )
    kept = fitted - 1
    if kept < 1 and entries:
        raise ValueError("line 2 and the first entry do not fit in the request's pages")
    opaque = mint_opaque_name()
    answer_path = exchange_dir / "ans" / f"{opaque}.jsonl"
    header = {
        "schema": SCHEMA,
        "role": kind,
        "point": point.role,
        "answerPath": str(answer_path),
        "bodySha256": body_sha256(kind, line_2, entries[:kept]),
        "pages": pages,
    }
    text = "".join(line + "\n" for line in [dump_line(header), *lines[: 1 + kept]])
    validate_request(text, point=point)
    return BuiltRequest(
        kind=kind,
        point=point.role,
        opaque=opaque,
        request_path=exchange_dir / "req" / f"{opaque}.jsonl",
        answer_path=answer_path,
        text=text,
        keys=MappingProxyType(
            {entry["key"]: stored for entry, (_, stored, _) in zip(entries[:kept], capped)}
        ),
        overflow=tuple(original for original, _, _ in capped[kept:]) + tuple(rest),
    )


def _check_cap(cap: int) -> None:
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"cap must be a positive integer, not {cap!r}")


def build_label_request(
    point: Point,
    items: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    exchange_dir: Path,
    cap: int,
    held_keys: Collection[str],
) -> BuiltRequest:
    """A label request over `(reference, item)` pairs: at most `cap` of them and `P_MAX["label"]`
    pages; `keys` maps each key to its item's reference, `overflow` holds the pairs left out."""
    _check_cap(cap)
    renderer = RENDERERS.get(point.role)
    if renderer is None:
        raise ValueError(f"no label renderer for {point.role!r}")
    capped = [((ref, item), ref, {"text": renderer(item)}) for ref, item in items[:cap]]
    return _build("label", point, point.rubric, capped, items[cap:], exchange_dir, held_keys)


def build_generation_request(
    point: Point,
    seeds: Sequence[Seed],
    *,
    exchange_dir: Path,
    cap: int,
    held_keys: Collection[str],
) -> BuiltRequest:
    """A generation request over `seeds`: at most `cap` of them and `P_MAX["generate"]` pages,
    each line the seed's text axes only; `keys` maps each key to its whole `Seed`."""
    _check_cap(cap)
    capped = [(seed, seed, blind_axes(seed.axes)) for seed in seeds[:cap]]
    return _build(
        "generate", point, point.generator_instructions, capped, seeds[cap:], exchange_dir,
        held_keys,
    )


def write_request(built: BuiltRequest) -> Path:
    """Validate `built.text` and write it to `built.request_path`, never over an existing file."""
    validate_request(built.text)
    path = built.request_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(built.text, encoding="utf-8")
    os.replace(partial, path)
    return path


# --- the closed schemas --------------------------------------------------------------------------


def _key_errors(
    where: str, line: Mapping[str, Any], expected: Sequence[str], id_blind: bool
) -> list[str]:
    errors = []
    for name in line:
        if id_blind and name.endswith(_ID_SUFFIX):
            errors.append(f"{where}: unexpected key {name!r} (no *Id key reaches the generator)")
        elif name not in expected:
            errors.append(f"{where}: unexpected key {name!r}")
    errors += [f"{where}: missing key {name!r}" for name in expected if name not in line]
    return errors


def _split_lines(text: str) -> tuple[list[str], list[str]]:
    if not text.endswith("\n"):
        return [], ["the request does not end with a newline"]
    raw_lines = text[:-1].split("\n")
    errors = [
        f"line {number}: raw line separator U+{ord(char):04X}"
        for number, raw in enumerate(raw_lines, 1)
        for char in _RAW_LINE_BREAKS
        if char in raw
    ]
    return raw_lines, errors


class RepeatedName(ValueError):
    """A JSON object naming one member twice: not a closed key set (json keeps the last)."""


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    names = [name for name, _ in pairs]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise RepeatedName(f"repeated name {repeated[0]!r}")
    return dict(pairs)


def load_json_line(raw: str) -> Any:
    """One exchange line as JSON, a repeated name refused at any depth (`RepeatedName`). Other
    failures raise `ValueError` (a `json.JSONDecodeError`, or a too-long integer) or, for deep
    nesting, `RecursionError`."""
    return json.loads(raw, object_pairs_hook=_closed_object)


def _parse_lines(raw_lines: Sequence[str]) -> tuple[list[Any], list[str]]:
    lines, errors = [], []
    for number, raw in enumerate(raw_lines, 1):
        try:
            value = load_json_line(raw)
        except RepeatedName as exc:
            errors.append(f"line {number}: {exc}")
            continue
        except ValueError as exc:
            errors.append(f"line {number}: not JSON ({exc})")
            continue
        except RecursionError:
            errors.append(f"line {number}: nested too deeply")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {number}: not a JSON object")
            continue
        lines.append(value)
    return lines, errors


def _page_errors(
    pages: Any, kind: str, raw_lines: Sequence[str]
) -> list[str]:
    where = "line 1 (header)"
    if not isinstance(pages, list) or not all(
        isinstance(page, list) and len(page) == 2
        and all(isinstance(n, int) and not isinstance(n, bool) for n in page)
        for page in pages
    ):
        return [f"{where}: pages is not a list of [offset, limit] pairs"]
    errors = []
    if any(limit < 1 for _, limit in pages):
        errors.append(f"{where}: a page's limit is below 1")
    covered = [n for offset, limit in pages for n in range(offset, offset + limit)]
    if not pages or pages[0] != [1, 1] or covered != list(range(1, len(raw_lines) + 1)):
        errors.append(f"{where}: pages do not cover lines 1–{len(raw_lines)} exactly once, the "
                      "header alone first")
    if len(pages) > P_MAX[kind]:
        errors.append(f"{where}: {len(pages)} pages, over P_max {P_MAX[kind]}")
    for offset, limit in pages:
        written = sum(len(raw) + 1 for raw in raw_lines[offset - 1 : offset - 1 + limit])
        if written > PAGE_BUDGET_CHARS:
            errors.append(f"{where}: page [{offset}, {limit}] holds {written} characters, over "
                          f"{PAGE_BUDGET_CHARS}")
    return errors


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _header_value_errors(header: Mapping[str, Any], point: Point) -> list[str]:
    """The header's values; an absent key is already a missing-key error."""
    where = "line 1 (header)"
    errors = []
    if "schema" in header and header["schema"] != SCHEMA:
        errors.append(f"{where}: schema is not {SCHEMA!r}")
    if "point" in header and header["point"] != point.role:
        errors.append(f"{where}: point {header['point']!r} is not {point.role!r}")
    if "answerPath" in header and (
        not isinstance(header["answerPath"], str) or not header["answerPath"]
    ):
        errors.append(f"{where}: answerPath is not a non-empty string")
    if "bodySha256" in header and not _is_sha256(header["bodySha256"]):
        errors.append(f"{where}: bodySha256 is not a sha256 hex digest")
    return errors


def _entry_errors(
    kind: str, entries: Sequence[Mapping[str, Any]], point: Point
) -> list[str]:
    entry_kind = _ENTRY_KIND[kind]
    if kind == "label":
        expected: tuple[str, ...] = ("key", "text")
    else:
        expected = ("key",) + tuple(
            name for name in seed_axis_names(point) if not name.endswith(_ID_SUFFIX)
        )
    errors = []
    seen: set[str] = set()
    for number, entry in enumerate(entries, 3):
        where = f"line {number} ({entry_kind})"
        errors += _key_errors(where, entry, expected, kind == "generate")
        key = entry.get("key")
        if "key" in entry and (not isinstance(key, str) or not KEY_PATTERN.fullmatch(key)):
            errors.append(f"{where}: key {key!r} is not {KEY_LENGTH} base32 characters")
        elif key in seen:
            errors.append(f"{where}: duplicate key {key!r}")
        if isinstance(key, str):
            seen.add(key)
        if kind == "label" and "text" in entry and not isinstance(entry["text"], str):
            errors.append(f"{where}: text is not a string")
    return errors


def _check(text: str, point: Point | None) -> tuple[Request | None, list[str]]:
    raw_lines, errors = _split_lines(text)
    if errors and not raw_lines:
        return None, errors
    lines, parse_errors = _parse_lines(raw_lines)
    errors += parse_errors
    if parse_errors:
        return None, errors
    if len(lines) < 2:
        return None, errors + ["a request holds a header and line 2 at least"]
    header, line_2, *entries = lines
    if not entries:
        return None, errors + ["a request holds one entry at least"]
    kind = header.get("role")
    if kind not in P_MAX:
        return None, errors + [f"line 1 (header): role {kind!r} is not 'label' or 'generate'"]
    id_blind = kind == "generate"
    errors += _key_errors("line 1 (header)", header, HEADER_KEYS, id_blind)
    if point is None:
        if header.get("point") not in ROLES:
            return None, errors + [f"line 1 (header): point {header.get('point')!r} is unknown"]
        point = load_point(header["point"])
    errors += _key_errors(f"line 2 ({_LINE_2[kind]})", line_2, (_LINE_2[kind],), id_blind)
    errors += _entry_errors(kind, entries, point)
    errors += _header_value_errors(header, point)
    line_2_text = line_2.get(_LINE_2[kind])
    if not isinstance(line_2_text, str):
        errors.append(f"line 2 ({_LINE_2[kind]}): {_LINE_2[kind]} is not a string")
    elif _is_sha256(header.get("bodySha256")) and header["bodySha256"] != body_sha256(
        kind, line_2_text, entries
    ):
        errors.append("line 1 (header): bodySha256 does not match the body")
    if "pages" in header:
        errors += _page_errors(header["pages"], kind, raw_lines)
    if errors:
        return None, errors
    return Request(
        kind=kind,
        point=point.role,
        answer_path=header["answerPath"],
        body_sha256=header["bodySha256"],
        pages=tuple(tuple(page) for page in header["pages"]),
        line_2=line_2_text,
        entries=tuple(entries),
    ), []


def request_errors(text: str, *, point: Point | None = None) -> list[str]:
    """Every way `text` breaks its request schema (module docstring); empty means valid. `point`
    defaults to the header's."""
    return _check(text, point)[1]


def parse_request(text: str, *, point: Point | None = None) -> Request:
    """The parsed request, or `RequestRefused` naming every error."""
    request, errors = _check(text, point)
    if request is None:
        raise RequestRefused(errors)
    return request


def validate_request(text: str, *, point: Point | None = None) -> None:
    """`RequestRefused` unless `text` is a valid request."""
    parse_request(text, point=point)
