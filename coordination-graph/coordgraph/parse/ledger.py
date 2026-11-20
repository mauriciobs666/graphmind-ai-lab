"""Coordination ledgers (plan sections 2.1, 3.4a; graph note sections 2.2, 2.5; plan3 section 3.4).

A ledger is a header block plus one or more pipe tables of units. Columns are mapped by
header name, cells are free text, and nothing in a row is ever dropped silently: a
status that maps to no teco value reads `unknown` with a warning, a malformed `Needs`
cell is recorded as `needsError`.

`statusToken` (graph note section 2.5) is one pure function, `status_token`. An archived
ledger keeps *minimal* units (`Unit.props(archived=True)`): the row is still parsed in full
to derive them and the two review lists, but no cell text, `needs` or `rowHash` is mirrored.
Positions (`tableIdx`, `rowIdx`) are 1-based.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field

from ..keys import MAX_KEY_BYTES, norm_value, safe_key
from .header import tokens
from .source import SourceDoc, base_fields
from .tables import find_tables

VOCABULARY = ("queued", "in-flight", "delivered", "gated", "accepted", "abandoned", "paused")
# Closed alias table, each entry justified by a real cell (graph note section 2.5 step 4).
ALIASES = {"relayed": "paused", "done": "delivered", "superseded": "abandoned"}
UNKNOWN = "unknown"
RESOLVING = ("accepted", "abandoned")

_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
_TWO_WORDS = re.compile(r"([a-z]+)\s+([a-z]+)")
_TICKS = re.compile(r"`([^`]+)`")
_REVIEW_PATH = re.compile(r"^(?P<p>[\w./\-]*docs/reviews/[\w./\-]+\.md)(?:#\S*)?$")
_REPO_PATH = re.compile(r"^[\w.\-]+(?:/[\w.\-]+)+\.\w{1,6}(?:#\S*)?$")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_QUOTED = re.compile(r"[\"“](.+?)[\"”]", re.DOTALL)
_PARSE_NAME = re.compile(r"[*`]")


def _strip_markup(cell: str) -> str:
    """Drop backticks, `*`, `_`, and emoji/pictographs (variation selectors, ZWJ included)."""
    out = []
    for ch in cell:
        if ch in "`*_\ufe0f\u200d":
            continue
        if unicodedata.category(ch) in ("So", "Sk"):
            continue
        out.append(ch)
    return "".join(out)


def status_token(cell: str) -> str:
    """Normalise a free-text `Status` cell to teco's vocabulary or `unknown` (graph note 2.5)."""
    text = _strip_markup(cell).strip().casefold()
    m = _WORD.match(text)
    if m is None:
        return UNKNOWN
    word = m.group(0)
    two = _TWO_WORDS.match(text)
    if word not in VOCABULARY and word not in ALIASES and two:
        joined = f"{two.group(1)}-{two.group(2)}"
        if joined in VOCABULARY:
            return joined
    if word in VOCABULARY:
        return word
    return ALIASES.get(word, UNKNOWN)


def _clean_name(name: str) -> str:
    return " ".join(_PARSE_NAME.sub("", name).casefold().split())


def map_columns(header: list[str]) -> dict[str, int]:
    """Column role -> index, by header name (first match wins; `Agent` is the owner only
    when there is no `Owner` column)."""
    names = [_clean_name(h) for h in header]
    cols: dict[str, int] = {}

    def put(role: str, i: int) -> None:
        cols.setdefault(role, i)

    for i, n in enumerate(names):
        if n.startswith("unit"):
            put("unit", i)
        elif n == "owner":
            put("owner", i)
        elif n.replace("-", " ") in ("agent id", "agentid", "agent ids"):
            put("agentId", i)
        elif n == "status":
            put("status", i)
        elif n.startswith("deliverable"):
            put("deliverable", i)
        elif n.startswith("gate"):
            put("gateVerdict", i)
        elif n == "cost":
            put("cost", i)
        elif n == "item":
            put("itemId", i)
        elif n == "needs":
            put("needs", i)
    if "owner" not in cols:
        for i, n in enumerate(names):
            if n == "agent" or n.startswith("agent ("):
                cols["owner"] = i
                break
    return cols


def unit_key_of(cell: str) -> str:
    """The Unit cell with emphasis/backticks stripped, cut at the first ` — `."""
    text = cell.replace("*", "").replace("`", "").replace("~~", "")
    return text.split(" — ", 1)[0].strip()


def _norm_cells(cells: list[str]) -> list[str]:
    return [" ".join(c.split()) for c in cells]


def _paths(cell: str) -> list[str]:
    return [t for t in _TICKS.findall(cell) if _REPO_PATH.match(t)]


def parse_needs(cell: str) -> tuple[list[str] | None, str | None]:
    """`(needs, error)` of a `Needs` cell: tokens raw (enclosing backticks dropped), `none` -> [].

    An empty or `—` cell is not recorded (`None`); an empty token between commas is malformed
    (`None`, error text).
    """
    text = cell.strip()
    if text in ("", "—"):
        return None, None
    if text.casefold() == "none":
        return [], None
    out = []
    for part in text.split(","):
        tok = part.strip()
        if len(tok) >= 2 and tok.startswith("`") and tok.endswith("`"):
            tok = tok[1:-1].strip()
        if not tok:
            return None, f"empty token in Needs cell {text!r}"
        out.append(tok)
    return out, None


def parse_paused(deliverable: str) -> tuple[str | None, str | None]:
    """`(question, askedOn)` from a paused row's Deliverable (teco.md step 2 shape:
    `⏸ "<question>" — relayed YYYY-MM-DD`)."""
    text = deliverable.strip()
    q = _QUOTED.search(text)
    question = " ".join(q.group(1).split()) if q else None
    if question is None:
        bare = re.split(r"\s+—\s+relayed\b", text, maxsplit=1)[0]
        bare = bare.replace("\u23f8", "").strip(" \ufe0f")
        question = " ".join(bare.split()) or None
    after = re.search(r"relayed\D{0,20}(\d{4}-\d{2}-\d{2})", text)
    date = after.group(1) if after else (_DATE.search(text).group(0) if _DATE.search(text) else None)
    return question, date


@dataclass
class Unit:
    unit: str
    unit_key: str  # without the `~n` suffix
    key: str  # `<ledgerPath>#<unitKey>[~n]`
    ledger_path: str
    slug: str
    component: str
    table_idx: int
    row_idx: int
    status_token: str
    status_text: str | None = None
    owner: str | None = None
    agent_id: str | None = None
    deliverable: str | None = None
    gate_verdict: str | None = None
    cost: str | None = None
    item_id: str | None = None
    needs: list[str] | None = None
    needs_error: str | None = None
    question: str | None = None
    asked_on: str | None = None
    cites: list[str] = field(default_factory=list)
    row_hash: str = ""

    def props(self, archived: bool = False) -> dict:
        """Properties as written to the graph; an archived ledger's unit is minimal."""
        p: dict = {
            "key": self.key, "unitKey": self.unit_key, "statusToken": self.status_token,
            "ledgerPath": self.ledger_path, "slug": self.slug, "component": self.component,
            "tableIdx": self.table_idx, "rowIdx": self.row_idx,
        }
        if self.item_id is not None:
            p["itemId"] = self.item_id
        if archived:
            return p
        p["unit"] = self.unit
        for name, value in (
            ("statusText", self.status_text), ("owner", self.owner), ("agentId", self.agent_id),
            ("deliverable", self.deliverable), ("gateVerdict", self.gate_verdict),
            ("cost", self.cost), ("needs", self.needs), ("needsError", self.needs_error),
            ("question", self.question), ("askedOn", self.asked_on),
        ):
            if value is not None:
                p[name] = value
        p["cites"] = list(self.cites)
        p["rowHash"] = self.row_hash
        return p


@dataclass
class LedgerDoc(SourceDoc):
    units: list[Unit] = field(default_factory=list)
    cited_reviews: list[str] = field(default_factory=list)
    resolved_reviews: list[str] = field(default_factory=list)

    @property
    def archived(self) -> bool:
        return self.header is not None and self.header.doc_status == "archived"

    def source_props(self) -> dict:
        props = super().source_props()
        props["citedReviews"] = list(self.cited_reviews)
        props["resolvedReviews"] = list(self.resolved_reviews)
        props["warnings"] = list(self.warnings)
        props["rowCount"] = len(self.units)
        return props

    def unit_props(self) -> list[dict]:
        return [u.props(archived=self.archived) for u in self.units]


def _resolve_review(path: str, component: str, review_paths: set[str] | None) -> str | None:
    """The covered review file a cited path denotes: as written, else under the component dir."""
    if review_paths is None:
        return path
    if path in review_paths:
        return path
    if component not in ("root", ""):
        alt = f"{component}/{path}"
        if alt in review_paths:
            return alt
    return None


def parse_ledger(path: str, raw: bytes, review_paths: set[str] | None = None) -> LedgerDoc:
    """Parse one ledger. `review_paths` is the set of covered review files (repo-relative);
    without it every cited review path is kept as written."""
    text, base = base_fields(path, "ledger", raw)
    doc = LedgerDoc(**base)
    archived = doc.archived
    seen: dict[str, int] = {}
    cited: set[str] = set()
    resolved: set[str] = set()
    for table in find_tables(text):
        cols = map_columns(table.header)
        if "unit" not in cols:
            continue
        if "status" not in cols:
            doc.warnings.append(f"table {table.idx}: no Status column")
        width = len(table.header)
        for row_i, (cells, line_no) in enumerate(zip(table.rows, table.row_lines), start=1):
            if not any(c.strip() for c in cells):
                continue
            if len(cells) > width:  # a stray pipe shifts every later column: say so
                doc.warnings.append(
                    f"table {table.idx} line {line_no}: {len(cells)} cells, header has {width}"
                )
            cells = (cells + [""] * width)[:width]  # trailing cells are optional

            def cell(role: str, cells: list[str] = cells) -> str | None:
                i = cols.get(role)
                return None if i is None else cells[i]

            unit_cell = cell("unit") or ""
            ukey = unit_key_of(unit_cell)
            if not ukey:
                doc.warnings.append(f"table {table.idx} line {line_no}: row without a Unit cell")
            n = seen.get(ukey, 0) + 1
            seen[ukey] = n
            full_key = ukey if n == 1 else f"{ukey}~{n}"
            if n > 1:
                doc.warnings.append(f"duplicate unit {ukey} → {full_key}")
            status_text = cell("status")
            token = status_token(status_text) if status_text is not None else UNKNOWN
            if token == UNKNOWN:
                doc.warnings.append(
                    f"unit {full_key}: unknown status {status_text!r}"
                    if status_text is not None else f"unit {full_key}: no status"
                )
            unit = Unit(
                unit=unit_cell, unit_key=ukey, key=safe_key(path, full_key),
                ledger_path=path, slug=doc.slug, component=doc.component,
                table_idx=table.idx, row_idx=row_i,
                status_token=token, status_text=status_text,
                owner=cell("owner"), agent_id=cell("agentId"), deliverable=cell("deliverable"),
                gate_verdict=cell("gateVerdict"), cost=cell("cost"),
            )
            item_cell = cell("itemId")
            if item_cell is not None:
                toks = tokens(item_cell.strip())
                cand = toks[0].strip() if toks else ""
                if cand and len(cand.encode("utf-8")) <= MAX_KEY_BYTES:
                    unit.item_id = norm_value(cand)
                elif item_cell.strip() not in ("", "—"):
                    doc.warnings.append(f"unit {full_key}: unusable Item cell; itemId not stored")
            needs_cell = cell("needs")
            if needs_cell is not None:
                unit.needs, unit.needs_error = parse_needs(needs_cell)
                if unit.needs_error and not archived:
                    doc.warnings.append(f"unit {full_key}: {unit.needs_error}")
            if token == "paused":
                unit.question, unit.asked_on = parse_paused(unit.deliverable or "")
                if unit.asked_on is None:
                    doc.warnings.append(f"unit {full_key}: paused without a relay date")
            citing = (unit.deliverable or "") + "\n" + (unit.gate_verdict or "")
            unit.cites = sorted({p.split("#", 1)[0] for c in (unit.deliverable, unit.gate_verdict)
                                 if c for p in _paths(c)})
            for tick in _TICKS.findall(citing):
                m = _REVIEW_PATH.match(tick)
                if not m:
                    continue
                review = _resolve_review(m.group("p"), doc.component, review_paths)
                if review is None:
                    doc.warnings.append(f"unit {full_key}: cited review {m.group('p')} not found")
                    continue
                cited.add(review)
                if token in RESOLVING:
                    resolved.add(review)
            unit.row_hash = hashlib.sha256(
                json.dumps(_norm_cells(cells), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            doc.units.append(unit)
    doc.cited_reviews = sorted(cited)
    doc.resolved_reviews = sorted(resolved)
    return doc
