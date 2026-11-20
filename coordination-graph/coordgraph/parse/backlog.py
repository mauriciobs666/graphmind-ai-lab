"""Backlog files: the common item format (plan section 3.6; plan3 sections 3.3, 3.6).

An item is a `### <id> - <title>` heading followed by a field block of consecutive
`- **Label:** value` lines (one blank line allowed between). Anything else under a heading is
prose. Every ID string is stored **as written** and never shape-validated here: the heading
pattern only decides what is an item, `Depends on` tokens are split and kept verbatim, and
`Item.ref` is the heading ID for a scheme ID, `<scope>:<id>` for a legacy one. `scope` comes
from `.work-items/project.json` `legacy_paths` (`coordgraph/scope.py`).

An item heading without a recognised ID but with a `Status` field is an *id-less item*: it
gets the positional ref `<scope>:@<ord>` and a warning, and exists only until the one-time
reformat (`coordgraph migrate`) mints it an ID.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..keys import MAX_KEY_BYTES
from ..scope import scope_of
from .source import SourceDoc, base_fields

# plan3 section 3.6, verbatim. Shape plus minimum widths; `FR`/`AC` segments never match.
HEADING = re.compile(
    r"^### (?P<id>[A-Z]-\d{3,}|[A-Z]{2,6}(?:-(?:[IST]\d{2,}|[FDK]\d{4,}))+) — (?P<title>.+)$"
)
ANY_HEADING = re.compile(r"^(#{1,6})\s+\S")
FENCE = re.compile(r"^\s*(```|~~~)")
FIELD = re.compile(r"^- \*\*(?P<label>[^:*]+):\*\*[ \t]*(?P<value>.*)$")
_ITEM_TYPE = re.compile(r"-([A-Z])\d+$")
_DATE = r"(\d{4}-\d{2}-\d{2})"

STATUS_WORDS = {
    "proposed": "🔵", "in-progress": "🟡", "deferred": "⚪", "rejected": "⚪", "delivered": "✅",
}
PRIORITIES = ("high", "medium", "low")
FACT_MODES = ("changes", "relies")
TOUCH_PREFIXES = ("file:", "graph:", "fact:")


class BacklogParseError(ValueError):
    """The file cannot be mirrored (a duplicate item ID); the message names the lines."""


@dataclass
class Item:
    ref: str
    item_id: str | None
    scope: str | None
    backlog_path: str
    ord: int
    line: int  # 1-based heading line
    title: str
    item_hash: str
    item_type: str | None = None
    status_marker: str | None = None
    status: str | None = None  # the status word, canonical
    open: bool = True
    priority: str | None = None
    depends_on: list[str] | None = None  # raw tokens; None = not recorded, [] = none
    touches: list[str] | None = None
    milestone: str | None = None
    milestone_order: int | None = None
    milestone_approved: str | None = None
    milestone_started: str | None = None
    fields: dict[str, str] = field(default_factory=dict)  # every label, raw

    @property
    def id_less(self) -> bool:
        return self.item_id is None

    def props(self) -> dict:
        """The `:Item` properties a parse determines (graph note section 2.2, plan3 G4)."""
        p: dict = {
            "ref": self.ref, "backlogPath": self.backlog_path, "ord": self.ord,
            "title": self.title, "open": self.open, "itemHash": self.item_hash,
        }
        for name, value in (
            ("itemId", self.item_id), ("scope", self.scope), ("itemType", self.item_type),
            ("statusMarker", self.status_marker), ("priority", self.priority),
            ("dependsOn", self.depends_on), ("touches", self.touches),
            ("milestone", self.milestone), ("milestoneOrder", self.milestone_order),
            ("milestoneApproved", self.milestone_approved),
            ("milestoneStarted", self.milestone_started),
        ):
            if value is not None:
                p[name] = value
        return p


@dataclass
class BacklogDoc(SourceDoc):
    scope: str | None = None
    items: list[Item] = field(default_factory=list)

    def source_props(self) -> dict:
        props = super().source_props()
        if self.scope is not None:
            props["scope"] = self.scope
        props["warnings"] = list(self.warnings)
        props["itemCount"] = len(self.items)
        return props

    def item_props(self) -> list[dict]:
        return [i.props() for i in self.items]


def _split_tokens(value: str) -> list[str]:
    """`Depends on` value -> raw tokens (comma-split, trimmed); `none` -> []."""
    text = value.strip()
    if text.casefold() == "none":
        return []
    return [t.strip() for t in text.split(",") if t.strip()]


def _status(value: str, label: str, warnings: list[str]) -> tuple[str, str | None, bool]:
    """`(statusMarker, word, open)` from a `Status` value: keyed on the word, emoji must agree."""
    marker = value.split(" — ", 1)[0].strip()
    lowered = marker.casefold()
    word = next((w for w in STATUS_WORDS if w in lowered), None)
    if word is None:
        warnings.append(f"{label}: unrecognised Status {marker!r}")
        return marker, None, True
    emoji = [e for e in set(STATUS_WORDS.values()) if e in marker]
    if emoji and STATUS_WORDS[word] not in emoji:
        warnings.append(f"{label}: Status emoji does not match the word {word!r}: {marker!r}")
    return marker, word, word != "delivered"


def _milestone(value: str, label: str, item: Item, warnings: list[str]) -> None:
    parts = [p.strip() for p in value.split(" — ")]
    item.milestone = parts[0] or None
    for part in parts[1:]:
        if m := re.fullmatch(r"order (\d+)", part):
            item.milestone_order = int(m.group(1))
        elif m := re.fullmatch(rf"approved {_DATE}", part):
            item.milestone_approved = m.group(1)
        elif m := re.fullmatch(rf"started {_DATE}", part):
            item.milestone_started = m.group(1)
        else:
            warnings.append(f"{label}: unrecognised Milestone part {part!r}")


def field_block(lines: list[str], heading_idx: int) -> dict[str, str]:
    """`{label: value}` of the field block under a heading (continuations joined by a space)."""
    out: dict[str, str] = {}
    j = heading_idx + 1
    if j < len(lines) and not lines[j].strip():
        j += 1
    current: str | None = None
    while j < len(lines):
        line = lines[j]
        m = FIELD.match(line)
        if m:
            current = m.group("label").strip()
            out.setdefault(current, m.group("value").strip())
        elif current is not None and line.startswith("  ") and line.strip():
            out[current] = (out[current] + " " + line.strip()).strip()
        else:
            break
        j += 1
    return out


def _headings(lines: list[str]) -> list[tuple[int, int]]:
    """`(line index, level)` of every heading outside fenced code."""
    out, in_fence = [], False
    for i, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (m := ANY_HEADING.match(line)):
            out.append((i, len(m.group(1))))
    return out


def _hash(lines: list[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_backlog(path: str, raw: bytes, legacy_paths: Mapping[str, str] | None = None) -> BacklogDoc:
    """Parse one backlog (component `BACKLOG.md` or agent `plan.md`).

    `legacy_paths` is the project's `legacy_paths` mapping (`coordgraph.scope.load_legacy_paths`).
    A duplicate item ref raises `BacklogParseError`.
    """
    text, base = base_fields(path, "backlog", raw)
    scope = scope_of(path, legacy_paths or {})
    doc = BacklogDoc(**base, scope=scope)
    if scope is None:
        doc.warnings.append("no legacy_paths entry gives this backlog a scope")
    lines = text.splitlines()
    heads = _headings(lines)
    seen: dict[str, int] = {}
    for k, (idx, level) in enumerate(heads):
        if level != 3:
            continue
        end = next((i for i, lv in heads[k + 1:] if lv <= 3), len(lines))
        m = HEADING.match(lines[idx])
        fields = field_block(lines, idx)
        if m is None and "Status" not in fields:
            continue  # a section heading, not an item
        ord_ = len(doc.items) + 1
        if m:
            item_id, title = m.group("id"), m.group("title").strip()
            legacy = len(item_id.split("-")[0]) == 1
            ref = f"{scope or 'unscoped'}:{item_id}" if legacy else item_id
            label = item_id
            tm = None if legacy else _ITEM_TYPE.search(item_id)
            item_type = tm.group(1) if tm else None
        else:
            item_id, title = None, lines[idx][4:].strip()
            ref, label, item_type = f"{scope or 'unscoped'}:@{ord_}", f"item @{ord_}", None
            doc.warnings.append(f"line {idx + 1}: item without a work-item ID ({title!r}); positional ref {ref}")
        if ref in seen:
            raise BacklogParseError(
                f"{path}: duplicate item {item_id or ref} (lines {seen[ref]} and {idx + 1})"
            )
        seen[ref] = idx + 1
        if len(ref.encode("utf-8")) > MAX_KEY_BYTES:
            doc.warnings.append(f"{label}: ref over {MAX_KEY_BYTES} bytes")
        item = Item(
            ref=ref, item_id=item_id, scope=scope, backlog_path=path, ord=ord_, line=idx + 1,
            title=title, item_hash=_hash(lines[idx:end]), item_type=item_type, fields=fields,
        )
        if "Status" in fields:
            item.status_marker, item.status, item.open = _status(fields["Status"], label, doc.warnings)
        else:
            doc.warnings.append(f"{label}: no Status field")
        if "Priority" in fields:
            word = fields["Priority"].split(" — ", 1)[0].strip().casefold()
            if word in PRIORITIES:
                item.priority = word
            else:
                doc.warnings.append(f"{label}: unrecognised Priority {fields['Priority']!r}")
        if "Depends on" in fields:
            item.depends_on = _split_tokens(fields["Depends on"])
            value = fields["Depends on"].strip()
            if value.casefold() != "none" and any(not p.strip() for p in value.split(",")):
                doc.warnings.append(f"{label}: empty token in Depends on")
        if "Touches" in fields:
            value = fields["Touches"].strip()
            item.touches = [] if value.casefold() == "none" else [
                t.strip() for t in value.split(";") if t.strip()
            ]
            for t in item.touches:
                _check_touch(t, label, doc.warnings)
        if "Milestone" in fields:
            _milestone(fields["Milestone"], label, item, doc.warnings)
        doc.items.append(item)
    return doc


def _check_touch(entry: str, label: str, warnings: list[str]) -> None:
    if not entry.startswith(TOUCH_PREFIXES):
        warnings.append(f"{label}: Touches entry {entry!r} has no file:/graph:/fact: prefix")
    elif entry.startswith("fact:") and not re.search(r"\((changes|relies)\)\s*$", entry):
        warnings.append(f"{label}: fact Touches entry {entry!r} lacks its (changes|relies) mode")
