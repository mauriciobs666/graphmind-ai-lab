"""The one-line family-document header (work-item SPEC section 5; plan3 section 3.2).

Stdlib only, one function over **already-read text** (never a path), and no `wilib`
import: the sync path must not depend on the registry. It re-implements `wi`'s own header
reader (`wilib.rebuild.header_fields`/`tokens`) and a parity test pins the two together.

Grammar: the `> **Status:**` line immediately under the H1 (one blank line allowed);
` · `-separated `**Label:** value` fields; backticked tokens. Every field other than the
ones read here is tolerated and ignored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..keys import MAX_KEY_BYTES

SEP = " · "
_PART = re.compile(r"\*\*([^:*]+):\*\*\s*(.*)\Z")
_TICKS = re.compile(r"`([^`]+)`")
# The closed `Status:` set (root AGENTS.md); the canonical token is what comes first.
STATUS_TOKENS = ("Interviewing", "Ready for design", "active", "superseded", "archived")


@dataclass(frozen=True)
class Header:
    fields: dict[str, str]
    doc_status: str | None
    owner: str | None
    tracks: str | None
    item_tokens: list[str]
    alias_tokens: list[str]
    item_id: str | None = None  # first `Item:` token as written; None when absent or unusable
    warnings: list[str] = field(default_factory=list)

    @property
    def aliases(self) -> list[str]:
        """`Aliases:` tokens; `[]` for a header block without `Aliases:`."""
        return self.alias_tokens


def tokens(value: str) -> list[str]:
    """Backticked tokens of a field value, else the value itself (`—` and empty mean none)."""
    return _TICKS.findall(value) or ([value] if value and value != "—" else [])


def header_index(lines: list[str]) -> int | None:
    """Index of the header line: immediately under the H1, one blank line allowed."""
    if not lines or not lines[0].startswith("# "):
        return None
    i = 1
    if i < len(lines) and not lines[i].strip():
        i += 1
    return i if i < len(lines) and lines[i].startswith("> **Status:**") else None


def header_fields(text: str) -> dict[str, str] | None:
    """`{label: raw value}` of the header line, or None without a header block."""
    lines = text.splitlines()
    i = header_index(lines)
    if i is None:
        return None
    out: dict[str, str] = {}
    for part in lines[i].removeprefix("> ").split(SEP):
        m = _PART.match(part.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _status_token(value: str) -> str | None:
    """The canonical `Status:` token: the first thing after the label, free text after it ignored.

    A value outside the closed set keeps its leading word as written, so a mistyped status
    is visible in the graph instead of silently normalised away.
    """
    text = value.strip().lstrip("`*_").strip()
    lowered = text.casefold()
    for token in STATUS_TOKENS:
        if lowered.startswith(token.casefold()):
            rest = text[len(token):]
            if not rest or not (rest[0].isalnum() or rest[0] == "-"):
                return token
    word = re.match(r"[^\s`*_·—:,;.]+", text)
    return word.group(0) if word else None


def parse_header(text: str) -> Header | None:
    """The header block of a family document, or None when it has none."""
    fields = header_fields(text)
    if fields is None:
        return None
    warnings: list[str] = []
    item_tokens = tokens(fields.get("Item", ""))
    item_id: str | None = None
    if item_tokens:
        candidate = item_tokens[0].strip()
        if not candidate or len(candidate.encode("utf-8")) > MAX_KEY_BYTES:
            warnings.append("header Item: unusable (empty or over 512 bytes); itemId not stored")
        else:
            item_id = candidate
    return Header(
        fields=fields,
        doc_status=_status_token(fields["Status"]) if "Status" in fields else None,
        owner=fields.get("Owner"),
        tracks=fields.get("Tracks"),
        item_tokens=item_tokens,
        alias_tokens=tokens(fields.get("Aliases", "")),
        item_id=item_id,
        warnings=warnings,
    )
