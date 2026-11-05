"""The generated-text screen and form obedience, checked per generated item at generation ingest
(method note `docs/plans/typed-decision-models-ml.md` §Q1.1 "Obedience is checked per item" and
§Q1.2 "Generated-text screen"; plan4 `typed-decision-models4.md` §3.6; plan5
`typed-decision-models5.md` §3.4 "Screen" and §3.5).

**`screen_hit(point, item)`** is pure: it returns the hit reason, or `None`. The text screened is
the item's *evidence* strings: every string the generator wrote, so not an enum value (`path`) and
not a field a seed fixes by equality (`condition`, which the seed's own text supplies and the
binding check already holds equal). The token lists are the point's (`points.screen_tokens`).

- **Hits anywhere** (`screen.anywhere`, compared case-insensitively as a substring: no prose uses
  them) are `token:<token>`; so is any `seeds.ID_PATTERN` match bounded by non-word characters
  (Unicode `\\w`), `id:<match>`. "pedido_123" and "opções_12" carry no such match, since a word
  character precedes the pattern; "(dom_07)" and "cue_01" do.
- **Hits in a meta position** (`screen.metaPosition`, whole words, case-insensitive) are
  `meta:<token>`. A meta position is a closed `(...)` or `[...]` group whose whole content is
  listed tokens joined by `/`, `,` or `|` (`(clear)`, `[true, false]`), or that holds a listed
  token followed by `:` or `=` (`(difficulty: hard)`); or a token directly after `label`,
  `answer`, `category` or `intent` and only separators (`label: complaint`, `answer - yes`). A
  stray closer (`:)`, `1)`) is ignored; an unclosed opener makes no group. So "had difficulty
  paying", a plain "complaint", "(order #4411)", "(other than login)", "(type A)" and "(no rush)"
  pass (note v1.9 §Q1.2 as amended 2026-11-04; the lists in `points/` are unchanged).

**`check_item(point, axes, item)`** adds the seed bindings. It returns an `ItemCheck` holding every
reason the item shows, in plan5 §3.5's precedence order: `screen`, then `bindingMismatch` (any
binding error not on `missingForm`), then `formMismatch` (an error on the `missingForm`
binding). The first is the item's one reason for the seed's 3-hit cap and its `dropped` count; the
round halts count the item under every reason it shows. An item with four binding errors (a
wrong-kind item) is still one `bindingMismatch`. Schema errors are not checked here: `ingest`
voids the run for them first.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .points import Point, screen_tokens, seed_binding_errors
from .seeds import ID_PATTERN

SCREEN = "screen"
BINDING_MISMATCH = "bindingMismatch"
FORM_MISMATCH = "formMismatch"
#: Plan5 §3.5's precedence: the first reason an item shows is its one.
REASONS = (SCREEN, BINDING_MISMATCH, FORM_MISMATCH)

#: The seed axis whose binding is a form (plan5 §3.5); any other binding is a binding mismatch.
FORM_AXIS = "missingForm"

_ID_TOKEN = re.compile(rf"(?<!\w)(?:{ID_PATTERN.pattern})(?!\w)")
_META_KEYWORD = re.compile(r"(?<!\w)(?:label|answer|category|intent)[\s:=>\-–—\"'*_`]*\Z")
_OPENERS, _CLOSERS = "([", ")]"
_BRACKET_JOINERS = re.compile(r"\s*[/,|]\s*")


@dataclass(frozen=True)
class ItemCheck:
    """What one generated item shows: `reasons` in precedence order (empty when it passes),
    `detail` the first screen reason or binding message, and `form` the seed's `missingForm`
    when `formMismatch` is among the reasons."""

    reasons: tuple[str, ...]
    detail: str | None
    form: str | None

    @property
    def primary(self) -> str | None:
        return self.reasons[0] if self.reasons else None


def evidence_strings(point: Point, item: Mapping[str, Any]) -> list[str]:
    """The strings the screen reads (module docstring)."""
    fixed = {b["field"] for b in point.definition["seedBindings"] if "equalsAxis" in b}
    found: list[str] = []
    for name, schema in point.definition["inputFields"].items():
        if name not in fixed and name in item:
            _collect(schema, item[name], found)
    return found


def _collect(schema: Mapping[str, Any], value: Any, found: list[str]) -> None:
    kind = schema["type"]
    if kind == "string" and isinstance(value, str):
        found.append(value)
    elif kind == "array" and isinstance(value, list):
        for entry in value:
            _collect(schema["items"], entry, found)
    elif kind == "object" and isinstance(value, dict):
        for name, field_schema in schema.get("fields", {}).items():
            if name in value:
                _collect(field_schema, value[name], found)


def _bracket_contents(text: str) -> list[str]:
    """The content of every closed `(...)` / `[...]` group, nested ones included. A closer with
    no matching opener before it (an emoticon `:)`, a list marker `1)`) is ignored and never
    cancels a later opener; an opener never closed is no group."""
    stack: list[tuple[str, int]] = []
    groups: list[str] = []
    for position, char in enumerate(text):
        if char in _OPENERS:
            stack.append((char, position))
        elif char in _CLOSERS and stack and _OPENERS.index(stack[-1][0]) == _CLOSERS.index(char):
            _, start = stack.pop()
            groups.append(text[start + 1:position])
    return groups


def _bracket_hit(content: str, tokens: Sequence[str]) -> str | None:
    """A listed token when the bracket's whole content is tokens joined by `/`, `,` or `|`, or
    when a listed token is followed by `:` or `=` inside it (note §Q1.2 "Hits")."""
    listed = {token.casefold() for token in tokens}
    parts = [part.strip() for part in _BRACKET_JOINERS.split(content)]
    if parts and all(part in listed for part in parts):
        return parts[0]
    for token in tokens:
        if re.search(rf"(?<!\w){re.escape(token.casefold())}\s*[:=]", content):
            return token
    return None


def _meta_hit(text: str, tokens: Sequence[str]) -> str | None:
    folded = text.casefold()
    for content in _bracket_contents(folded):
        found = _bracket_hit(content, tokens)
        if found is not None:
            return found
    for token in tokens:
        for match in re.finditer(rf"(?<!\w){re.escape(token.casefold())}(?!\w)", folded):
            if _META_KEYWORD.search(folded[:match.start()]):
                return token
    return None


def screen_hit(point: Point, item: Mapping[str, Any]) -> str | None:
    """The hit reason for `item`'s evidence strings, or `None` (module docstring)."""
    tokens = screen_tokens(point)
    for text in evidence_strings(point, item):
        folded = text.casefold()
        for token in tokens["anywhere"]:
            if token.casefold() in folded:
                return f"token:{token}"
        found = _ID_TOKEN.search(text)
        if found:
            return f"id:{found.group()}"
        meta = _meta_hit(text, tokens["metaPosition"])
        if meta is not None:
            return f"meta:{meta}"
    return None


def check_item(point: Point, axes: Mapping[str, Any], item: Mapping[str, Any]) -> ItemCheck:
    """Every reason `item` shows against its seed's `axes` (module docstring)."""
    errors = seed_binding_errors(point, item, axes)
    hit = screen_hit(point, item)
    form_errors = [e for e in errors if e.axis == FORM_AXIS]
    other_errors = [e for e in errors if e.axis != FORM_AXIS]
    shown = ((SCREEN, hit), (BINDING_MISMATCH, other_errors[0].message if other_errors else None),
             (FORM_MISMATCH, form_errors[0].message if form_errors else None))
    present = [(reason, detail) for reason, detail in shown if detail is not None]
    form = axes[FORM_AXIS] if form_errors else None
    return ItemCheck(tuple(reason for reason, _ in present),
                     present[0][1] if present else None, form)
