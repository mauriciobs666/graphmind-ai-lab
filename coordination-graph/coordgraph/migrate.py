"""`coordgraph migrate`: the one-time backlog layout reformat (plan section 3.6; plan3 3.5).

Brings a backlog file to the common item format and gives every item without an ID one from
`wi new` - scheme IDs only, never a `K-NNN`/`C-NNN`, no "past the highest id" rule. Existing
IDs (legacy or scheme) are kept. What it does, in file order:

- every top-level bullet of a parking-lot or open-items section, and every id-less `###`
  heading under one, becomes a `###` item with a `Status` field (`🔵 proposed`, listed for the
  owner to confirm) and a minted ID; no text is dropped (the bold lead becomes the title, the
  rest the body);
- a bullet whose bold lead is `<ID> - <title>` (the cpg backlog) becomes a `###` item that
  keeps its ID, its status read from its emoji or its group heading;
- a `###` heading's trailing `(<emoji> <word> - ...)` moves into its `Status` field;
- an agent plan's `| ID | Added | Priority | Status | Summary |` table is folded into its
  items (`Priority`, `Status`, `Added`; a row with no heading becomes an item) and removed;
- a milestone map table (`| Milestone | ... | Items |`) puts `Milestone: <token>` on the items
  its rows name, and is wrapped in the milestone-map markers.

`Depends on`/`Touches` are never filled. `analyze` is pure; `render` substitutes IDs;
`apply_file` allocates (through an injected minter) and rewrites atomically. Dry-run mints
nothing: it renders placeholders and the exact `wi new` commands.
"""
from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .discover import component_of
from .lint import MAP_CLOSE, MAP_OPEN, default_wi_root
from .parse.backlog import ANY_HEADING, FENCE, FIELD, HEADING, field_block, parse_backlog
from .parse.tables import find_tables

EMOJI_WORD = {"🔵": "proposed", "🟡": "in-progress", "⚪": "deferred", "✅": "delivered"}
_CONTAINER = re.compile(r"^(open|parking lot|in progress|proposed|deferred|active)\b", re.I)
_GROUP = re.compile(r"^(in progress|proposed|deferred|open|active)\b", re.I)
_PAREN_STATUS = re.compile(
    r"^(?P<title>.*?) \((?P<emoji>[🔵🟡⚪✅]) (?P<word>proposed|in-progress|deferred|rejected|delivered)"
    r"(?P<rest>.*)\)\s*$"
)
_BULLET = re.compile(r"^[-*] ")
_LEAD = re.compile(r"^\*\*(?P<lead>.+?)\*\*", re.S)
_ID_LEAD = re.compile(
    r"^(?P<id>[A-Z]-\d{3,}|[A-Z]{2,6}(?:-(?:[IST]\d{2,}|[FDK]\d{4,}))+) — (?P<title>.+)$", re.S
)
_PLACEHOLDER = re.compile(r"\x00NEW(\d+)\x00")


@dataclass
class NewItem:
    index: int
    title: str
    wi_type: str
    home: str
    origin: str  # "parking-lot" | "open-items" | "open-heading"
    line: int

    @property
    def command(self) -> list[str]:
        return ["wi", "new", self.wi_type, "--home", self.home, "--title", self.title]


@dataclass
class Reformat:
    path: str
    old: str
    template: str  # new text with \x00NEW<i>\x00 placeholders
    new_items: list[NewItem] = field(default_factory=list)
    status_defaulted: list[str] = field(default_factory=list)  # titles/IDs given `🔵 proposed`
    notes: list[str] = field(default_factory=list)

    def render(self, ids: list[str]) -> str:
        return _PLACEHOLDER.sub(lambda m: ids[int(m.group(1))], self.template)

    @property
    def changed(self) -> bool:
        return self.template != self.old


def home_and_type(path: str) -> tuple[str, str]:
    """`wi new` type and `--home` for a backlog: agent plans are kaizen in `claude`, others features."""
    if len(PurePosixPath(path).parts) == 4 and path.startswith("claude/") and path.endswith("/kaizen/plan.md"):
        return "K", "claude"
    return "F", component_of(path)


def _clean_title(text: str) -> str:
    title = " ".join(text.replace("**", "").split()).rstrip(".:").strip()
    return title or "Untitled"


def _bullet_block(lines: list[str], i: int) -> int:
    """Index one past the last line of the top-level bullet starting at `i` (trailing blanks excluded)."""
    j, n = i + 1, len(lines)
    last = i
    while j < n:
        line = lines[j]
        if not line.strip():
            k = j
            while k < n and not lines[k].strip():
                k += 1
            if k < n and lines[k][:1] in (" ", "\t"):
                j = k
                continue
            break
        if line[:1] in (" ", "\t"):
            last = j
            j += 1
            continue
        break
    return last + 1


def _bullet_text(lines: list[str], i: int, j: int) -> str:
    first = lines[i][2:]
    rest = [ln[2:] if ln.startswith("  ") else ln.lstrip() for ln in lines[i + 1:j]]
    return "\n".join([first, *rest]).strip("\n")


def _split_lead(text: str) -> tuple[str, str]:
    """`(title, body)` of a bullet: its bold lead and the rest, else its first line and everything."""
    m = _LEAD.match(text)
    if m:
        return _clean_title(m.group("lead")), text[m.end():].lstrip(" ").strip("\n")
    first = " ".join(text.splitlines()[0].replace("**", "").split())
    if len(first) > 90:
        first = first[: first.rfind(" ", 0, 90) if " " in first[:90] else 90].rstrip(",;:") + "…"
    return _clean_title(first), text


def _status_field(emoji: str, note: str = "") -> str:
    return f"- **Status:** {emoji} {EMOJI_WORD[emoji]}" + (f" — {note}" if note else "")


def _insert_fields(block: list[str], new_fields: list[str]) -> list[str]:
    """Put `new_fields` at the start of an item's field block (heading is `block[0]`)."""
    if not new_fields:
        return block
    out = [block[0]]
    rest = block[1:]
    if rest and not rest[0].strip():
        out.append(rest[0])
        rest = rest[1:]
    else:
        out.append("")
    out.extend(new_fields)
    if rest and not FIELD.match(rest[0]):
        out.append("")
    return out + rest


def _heads(lines: list[str]) -> list[tuple[int, int]]:
    out, in_fence = [], False
    for i, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (m := ANY_HEADING.match(line)):
            out.append((i, len(m.group(1))))
    return out


@dataclass
class _Row:
    added: str
    priority: str
    emoji: str | None
    summary: str


def _tables(text: str, lines: list[str]):
    """Summary-table rows (`{id: _Row}` with their spans) and milestone-map spans."""
    summary: dict[str, _Row] = {}
    summary_spans: list[tuple[int, int]] = []
    map_spans: list[tuple[int, int]] = []
    milestone_of: dict[str, str] = {}
    for t in find_tables(text):
        if not t.rows:
            continue
        names = [h.strip().casefold() for h in t.header]
        start, end = t.row_lines[0] - 3, t.row_lines[-1]
        if names[:1] == ["id"] and "summary" in names:
            col = {n: names.index(n) for n in ("added", "priority", "status", "summary") if n in names}
            for cells in t.rows:
                def cell(n: str, cells=cells) -> str:
                    return cells[col[n]].strip() if n in col and col[n] < len(cells) else ""
                status = cell("status")
                emoji = next((e for e in EMOJI_WORD if e in status), None)
                summary[cells[0].strip().strip("`")] = _Row(cell("added"), cell("priority").casefold(),
                                                             emoji, cell("summary"))
            summary_spans.append((start, end))
        elif names[:1] == ["milestone"] and "items" in names:
            if start > 0 and lines[start - 1].strip() == MAP_OPEN:
                continue
            map_spans.append((start, end))
            for cells in t.rows:
                m = re.search(r"\b(?:ahead-of-)?M\d+(?:\.\d+)?\b", cells[0])
                if not m:
                    continue
                for ident in re.findall(r"\b[A-Z]-\d{3,}\b", " ".join(cells[names.index("items"):])):
                    milestone_of.setdefault(ident, m.group(0))
    return summary, summary_spans, map_spans, milestone_of


def analyze(path: str, text: str) -> Reformat:
    """Plan the reformat of one backlog file; pure."""
    wi_type, home = home_and_type(path)
    lines = text.splitlines()
    heads = _heads(lines)
    summary, summary_spans, map_spans, milestone_of = _tables(text, lines)
    headed_ids = {m.group("id") for i, _ in heads if (m := HEADING.match(lines[i]))}
    rf = Reformat(path=path, old=text, template="")
    out: list[str] = []
    n_lines = len(lines)

    def mint(title: str, origin: str, line: int) -> str:
        item = NewItem(len(rf.new_items), title, wi_type, home, origin, line)
        rf.new_items.append(item)
        return f"\x00NEW{item.index}\x00"

    def blank_before() -> None:
        if out and out[-1].strip():
            out.append("")

    def item_block(heading: str, fields: list[str], body: str) -> None:
        blank_before()
        out.extend([heading, "", *fields])
        if body.strip():
            out.extend(["", *body.strip("\n").splitlines()])
        out.append("")

    h2_container = False
    ctx = "other"  # what a top-level bullet now means: "container" | "group" | other
    i = 0
    in_fence = False
    while i < n_lines:
        line = lines[i]
        span = next((s for s in summary_spans if s[0] == i), None)
        if span:
            for ident, row in summary.items():
                if ident in headed_ids:
                    continue
                fields = [_status_field(row.emoji or "🔵")]
                if row.priority in ("high", "medium", "low"):
                    fields.append(f"- **Priority:** {row.priority}")
                if row.added:
                    fields.append(f"- **Added:** {row.added}")
                if not row.emoji:
                    rf.status_defaulted.append(ident)
                item_block(f"### {ident} — {row.summary or 'Untitled'}", fields, "")
                rf.notes.append(f"table row {ident} became an item")
            rf.notes.append("summary table folded into its items and removed")
            i = span[1]
            continue
        span = next((s for s in map_spans if s[0] == i), None)
        if span:
            blank_before()
            out.extend([MAP_OPEN, *lines[span[0]:span[1]], MAP_CLOSE])
            rf.notes.append("milestone map wrapped in markers")
            i = span[1]
            continue
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        hm = None if in_fence else ANY_HEADING.match(line)
        if hm:
            level = len(hm.group(1))
            if level == 2:
                h2_container = bool(_CONTAINER.match(re.sub(r"^[^\w]+", "", line[3:]).strip()))
                ctx = "container" if h2_container else "other"
                out.append(line)
                i += 1
                continue
            if level == 3:
                end = next((k for k, lv in heads if k > i and lv <= 3), n_lines)
                block = lines[i:end]
                tail = []
                while block and not block[-1].strip():
                    tail.append(block.pop())
                m = HEADING.match(line)
                if m:
                    out.extend(_reformat_known(block, m, field_block(lines, i), summary, milestone_of, rf))
                    ctx = "other"
                elif h2_container and _GROUP.match(re.sub(r"^[^\w]+", "", line[4:]).strip()):
                    out.append(line)
                    ctx = "group"
                    end = i + 1
                    tail = []
                elif h2_container:
                    title = _clean_title(line[4:])
                    token = mint(title, "open-heading", i + 1)
                    has_status = "Status" in field_block(lines, i)
                    if not has_status:
                        rf.status_defaulted.append(title)
                    body = block[1:]
                    fields = [] if has_status else [_status_field("🔵", "defaulted; owner confirms status")]
                    out.extend(_insert_fields([f"### {token} — {title}", *body], fields))
                    ctx = "other"
                else:
                    out.extend(block)
                    ctx = "other"
                out.extend(reversed(tail))
                i = end
                continue
            ctx = "other"
            out.append(line)
            i += 1
            continue
        if not in_fence and ctx in ("container", "group") and _BULLET.match(line):
            j = _bullet_block(lines, i)
            text_ = _bullet_text(lines, i, j)
            title, body = _split_lead(text_)
            idm = _ID_LEAD.match(title) if _LEAD.match(text_) else None
            if idm:
                emoji = next((e for e in EMOJI_WORD if e in body[:40]), None)
                if emoji:
                    body = body.replace(emoji, "", 1).lstrip(" ")
                else:
                    emoji = _group_emoji(out) or "🔵"
                    if _group_emoji(out) is None:
                        rf.status_defaulted.append(idm.group("id"))
                item_block(f"### {idm.group('id')} — {_clean_title(idm.group('title'))}",
                           [_status_field(emoji)], body)
                rf.notes.append(f"bullet {idm.group('id')} became a heading item")
            else:
                origin = "parking-lot" if _is_parking(out) else "open-items"
                token = mint(title, origin, i + 1)
                rf.status_defaulted.append(title)
                note = "parking-lot origin; owner confirms status" if origin == "parking-lot" \
                    else "defaulted; owner confirms status"
                item_block(f"### {token} — {title}", [_status_field("🔵", note)], body)
            i = j
            continue
        out.append(line)
        i += 1
    while out and not out[-1].strip():
        out.pop()
    rf.template = "\n".join(out) + ("\n" if text.endswith("\n") or out else "")
    return rf


def _is_parking(out: list[str]) -> bool:
    for line in reversed(out):
        if line.startswith("## "):
            return "parking lot" in line.casefold()
    return False


def _group_emoji(out: list[str]) -> str | None:
    for line in reversed(out):
        if line.startswith("### "):
            return next((e for e in EMOJI_WORD if e in line), None)
        if line.startswith("## "):
            return None
    return None


def _reformat_known(block: list[str], m: re.Match, fields: dict[str, str], summary: dict[str, _Row],
                    milestone_of: dict[str, str], rf: Reformat) -> list[str]:
    """An ID'd item: status paren -> field, table row and milestone map folded in."""
    ident = m.group("id")
    heading = block[0]
    new_fields: list[str] = []
    row = summary.get(ident)
    if "Status" not in fields:
        pm = _PAREN_STATUS.match(m.group("title"))
        if pm:
            heading = f"### {ident} — {pm.group('title').rstrip()}"
            rest = pm.group("rest").lstrip(" —·,-").strip()
            word = pm.group("word")
            marker = f"{pm.group('emoji')} {word}"
            new_fields.append(f"- **Status:** {marker}" + (f" — {rest}" if rest else ""))
            rf.notes.append(f"{ident}: heading status moved to the Status field")
        elif row is not None and row.emoji:
            new_fields.append(_status_field(row.emoji))
        else:
            new_fields.append(_status_field("🔵", "defaulted; owner confirms status"))
            rf.status_defaulted.append(ident)
    if row is not None:
        if "Priority" not in fields and row.priority in ("high", "medium", "low"):
            new_fields.append(f"- **Priority:** {row.priority}")
        if "Added" not in fields and row.added:
            new_fields.append(f"- **Added:** {row.added}")
    if "Milestone" not in fields and ident in milestone_of:
        new_fields.append(f"- **Milestone:** {milestone_of[ident]}")
        rf.notes.append(f"{ident}: Milestone {milestone_of[ident]} from the milestone table")
    return _insert_fields([heading, *block[1:]], new_fields)


# ----------------------------------------------------------------------------- reports

def placeholder_ids(items: list[NewItem]) -> list[str]:
    return [f"<new {i.wi_type} #{i.index + 1}>" for i in items]


def dry_run_report(rf: Reformat, legacy_paths: Mapping[str, str]) -> str:
    """The per-file report: counts, the `wi new` commands, and the unified diff (no allocation)."""
    after = rf.render(placeholder_ids(rf.new_items))
    before_n = len(parse_backlog(rf.path, rf.old.encode(), legacy_paths).items)
    after_n = len(parse_backlog(rf.path, after.encode(), legacy_paths).items)
    out = [f"== {rf.path}", f"   items: {before_n} now, {after_n} after the reformat",
           f"   new ids to mint: {len(rf.new_items)}"]
    for it in rf.new_items:
        out.append(f"   mint [{it.origin}, line {it.line}] {' '.join(_quote(a) for a in it.command)}")
    if rf.status_defaulted:
        out.append(f"   status defaulted to proposed (owner confirms): {len(rf.status_defaulted)}")
        out.extend(f"     - {t}" for t in rf.status_defaulted)
    out.extend(f"   {n}" for n in dict.fromkeys(rf.notes))
    out.extend(difflib.unified_diff(
        rf.old.splitlines(), after.splitlines(), f"a/{rf.path}", f"b/{rf.path}", lineterm=""))
    return "\n".join(out)


def _quote(arg: str) -> str:
    return repr(arg) if re.search(r"[\s'\"`$]", arg) else arg


# ----------------------------------------------------------------------------- apply

Minter = Callable[[NewItem], str]


class WiError(RuntimeError):
    pass


def wi_minter(repo_root: str | Path, wi_script: str | Path | None = None) -> Minter:
    """Mint through the `wi` CLI run in `repo_root` (the work-item registry it finds there)."""
    script = Path(wi_script) if wi_script else default_wi_root() / "wi.py"

    def mint(item: NewItem) -> str:
        proc = subprocess.run(
            [sys.executable, str(script), *item.command[1:]],
            cwd=repo_root, capture_output=True, text=True, check=False,
        )
        ident = proc.stdout.strip()
        if proc.returncode != 0 or not ident:
            raise WiError(f"`{' '.join(item.command)}` failed ({proc.returncode}): {proc.stderr.strip()}")
        return ident

    return mint


def wi_flush(repo_root: str | Path, wi_script: str | Path | None = None) -> None:
    script = Path(wi_script) if wi_script else default_wi_root() / "wi.py"
    proc = subprocess.run([sys.executable, str(script), "flush"], cwd=repo_root,
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise WiError(f"`wi flush` failed: {proc.stderr.strip()}")


def apply_file(root: str | Path, path: str, mint: Minter, flush: Callable[[], None] | None = None) -> Reformat:
    """Allocate every id-less item's ID, rewrite the file atomically; idempotent (no ids when done)."""
    target = Path(root) / path
    rf = analyze(path, target.read_text(encoding="utf-8"))
    ids = [mint(item) for item in rf.new_items]  # all allocation before any write
    if flush is not None and ids:
        flush()
    if rf.changed:
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".migrate-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(rf.render(ids))
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return rf
