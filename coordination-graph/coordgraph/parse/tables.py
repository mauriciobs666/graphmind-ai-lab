"""GitHub-flavoured pipe tables out of Markdown text (shared by the ledger parser)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Table:
    idx: int  # 1-based position among every table of the file
    header: list[str]
    rows: list[list[str]]
    row_lines: list[int]  # 1-based file line of each row


def split_row(line: str) -> list[str]:
    """Cells of one pipe row: outer pipes dropped, `\\|` kept as a literal pipe, cells stripped."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    cells, cur, i = [], [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if c == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    tail = "".join(cur).strip()
    if tail or not line.rstrip().endswith("|"):
        cells.append(tail)
    return cells


def find_tables(text: str) -> list[Table]:
    """Every pipe table outside fenced code, in file order."""
    lines = text.splitlines()
    tables: list[Table] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            in_fence = not in_fence
            i += 1
            continue
        if (
            not in_fence
            and line.lstrip().startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].lstrip().startswith(("|", "-", ":"))
            and _SEPARATOR.match(lines[i + 1])
            and "-" in lines[i + 1]
        ):
            header = split_row(line)
            rows, row_lines = [], []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(split_row(lines[j]))
                row_lines.append(j + 1)
                j += 1
            tables.append(Table(len(tables) + 1, header, rows, row_lines))
            i = j
            continue
        i += 1
    return tables
