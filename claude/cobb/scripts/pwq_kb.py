#!/usr/bin/env python3
"""pwq_kb.py -- tooling for the python-web-quirks knowledge-base migration.

Plan: `docs/plans/python-web-quirks-kb.md` (D1-D9); requirements:
`docs/requirements/python-web-quirks-kb.md`. This module is the S1 slice: everything that works on
the pinned source and the copy of record, with no network and no graph access.

Subcommands (stdlib only; the source is always read from `git show <explicit sha>:<path>`, never
from HEAD or the working tree):

  units [--chunks]       print every cross-reference hit of the pinned source with its ruling,
                         the derived units, header sizes and (with --chunks) the chunk table
  render-copy            render the copy of record (`claude/knowledge/python-web-quirks.md`)
                         to stdout, or to --out
  verify-copy --copy F   recompute every header and compare every body with the pinned source

Library surface (used by the selftest and by the later ingest/gate/diverge steps):
`read_pinned`, `parse_source`, `find_hits`, `RULINGS`, `build_units`, `render_header`,
`render_unit`, `render_copy`, `parse_copy`, `verify_copy`, `listing`, `listing_bytes`.

Conventions fixed here (the plan leaves them open): a unit id is `u` + the two-digit number of its
first member's source section; a merged unit's header groups its per-member `Versions[n]` /
`Verified[n]` / `Other-dates[n]` lines by member; `Members:` is followed by one `- <title>` line
per member; the migration date is the `Migrated: <date> ...` line in the `<!-- preamble-note -->`
block.

Run the checks: `python3 claude/cobb/scripts/pwq_kb_selftest.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = "skills/python-web-quirks/SKILL.md"
MIGRATED_ON = "2026-11-09"
TITLE_LIMIT = 200  # ingest document `title` limit
LISTING_BOUND = 10_000  # bytes, D6 item 3

BODY_DELIM = "<!-- body -->"
PREAMBLE_FENCE = "<!-- preamble -->"
NOTE_FENCE = "<!-- preamble-note -->"
UNIT_FENCE_RE = re.compile(r"^<!-- unit (\S+) -->$")
FENCE_SHAPED_RE = re.compile(r"^<!-- (unit |preamble|body)")
# D6 item 3: the title listing an agent runs on the copy (grep -n -E, same regex).
LISTING_RE = re.compile(r"^(<!-- (unit |preamble)|## )")

HEADING_RE = re.compile(r"^## (\S.*?)[ \t]*$", re.M)
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# --- D2: the coverage probe -------------------------------------------------------------------
NOUNS = "section|entry|gotcha|note|item|paragraph|rule|table|probe|half|sibling|companion"
DIRECTIONS = "above|below|earlier|previous|preceding|next|following|elsewhere"
PROBE_WINDOW = 40


def probe_regex(window: int = PROBE_WINDOW) -> re.Pattern[str]:
    n, d = rf"\b(?:{NOUNS})s?\b", rf"\b(?:{DIRECTIONS})\b"
    gap = rf"[^\n]{{0,{window}}}?"
    return re.compile(rf"{n}{gap}{d}|{d}{gap}{n}", re.I)


# Ruling table: (enclosing section heading prefix, lower-cased hit text) -> ruling.
#   ("referent", <heading prefix>, ...)  the sections the hit points at; they merge into one unit
#   ("intra",)                           points inside the same section
#   ("external",)                        points outside the file
# Keyed by text, not line number: "section above" occurs in four sections.
H_BT = "Starlette/FastAPI `BackgroundTasks` (sync callable)"
H_TC1 = "`starlette.testclient.TestClient`'s teardown"
H_TC2 = "`TestClient` re-raises"
H_TC3 = "Version rot ahead"
H_ST1 = "Anything the lifespan puts on `app.state`"
H_ST2 = "To force a `TestClient` response"
H_BT2 = "Starlette `BackgroundTasks` runs one request's"
H_PYC1 = "A same-second edit that keeps"
H_PYC2 = "`PYTHONPYCACHEPREFIX="
H_IN1 = "Under pytest's default capture, an in-test `input()`"
H_IN2 = "Under pytest's default capture, fd 0"
H_POOL = "Holding an application lock across"

RULINGS: dict[tuple[str, str], tuple[str, ...]] = {
    (H_TC2, "section above"): ("referent", H_TC1),
    (H_TC3, "sections above"): ("referent", H_TC1, H_TC2),
    (H_ST1, "probe above"): ("intra",),
    (H_ST2, "section above"): ("referent", H_ST1),
    (H_POOL, "table above"): ("intra",),
    (H_POOL, "rule above"): ("intra",),
    (H_BT2, "entry above"): ("referent", H_BT),
    (H_PYC2, "section above"): ("referent", H_PYC1),
    (H_IN2, "section above"): ("referent", H_IN1),
}


# --- source model ------------------------------------------------------------------------------
@dataclass
class Section:
    index: int  # 1-based, source order
    heading: str  # verbatim, without the "## "
    body: str  # verbatim, from the end of the heading line to the next "## " (or EOF)
    line: int  # 1-based line number of the heading


@dataclass
class Source:
    text: str
    front_matter: str
    preamble: str  # between the front matter and the first "## "
    sections: list[Section]


class KBError(Exception):
    """Any condition under which the build must fail rather than emit a copy."""


class UnruledHit(KBError):
    pass


def read_pinned(pin: str, path: str = SOURCE_PATH, repo: Path = REPO_ROOT) -> str:
    """`git show <pin>:<path>`. The pin must be an explicit hex sha, never a ref such as HEAD."""
    if not re.fullmatch(r"[0-9a-f]{7,40}", pin):
        raise KBError(f"--pin must be an explicit commit sha (7-40 hex chars), got {pin!r}")
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{pin}:{path}"], capture_output=True, check=False
    )
    if proc.returncode != 0:
        raise KBError(f"git show {pin}:{path} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout.decode("utf-8")


def parse_source(text: str) -> Source:
    front = ""
    rest_start = 0
    if text.startswith("---\n"):
        end = text.find("\n---\n", 3)
        if end == -1:
            raise KBError("front matter opened with '---' but never closed")
        rest_start = end + len("\n---\n")
        front = text[:rest_start]
    matches = list(HEADING_RE.finditer(text, rest_start))
    if not matches:
        raise KBError("no '## ' headings in the source")
    preamble = text[rest_start : matches[0].start()]
    sections: list[Section] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(Section(i + 1, m.group(1), text[m.end() : end], text.count("\n", 0, m.start()) + 1))
    return Source(text, front, preamble, sections)


def split_preamble(preamble: str) -> tuple[str, str]:
    """(verbatim part, waived blockquote). The blockquote is the contiguous `>` block; nothing may follow it."""
    lines = preamble.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.startswith(">")), None)
    if start is None:
        return preamble.strip("\n"), ""
    end = start
    while end < len(lines) and lines[end].startswith(">"):
        end += 1
    if "".join(lines[end:]).strip():
        raise KBError("text follows the preamble blockquote; it has no disposition")
    return "\n".join(lines[:start]).strip("\n"), "\n".join(lines[start:end])


# --- D2: hits, rulings, units -----------------------------------------------------------------
@dataclass
class Hit:
    line: int
    section: int  # 0 = before the first "## "
    heading: str
    text: str  # the matched span, as written
    context: str  # the full source line


def find_hits(text: str, window: int = PROBE_WINDOW) -> list[Hit]:
    rx = probe_regex(window)
    hits: list[Hit] = []
    section, heading = 0, "(preamble)"
    for no, ln in enumerate(text.split("\n"), 1):
        m = re.match(r"## (\S.*?)\s*$", ln)
        if m:
            section, heading = section + 1, m.group(1)
        for h in rx.finditer(ln):
            hits.append(Hit(no, section, heading, h.group(0), ln))
    return hits


def ruling_for(hit: Hit, rulings: dict) -> tuple[str, ...] | None:
    for (prefix, snippet), ruling in rulings.items():
        if hit.heading.startswith(prefix) and hit.text.lower() == snippet:
            return ruling
    return None


@dataclass
class Unit:
    id: str
    members: list[Section]

    @property
    def title(self) -> str:
        return self.members[0].heading


def _section_for(prefix: str, sections: list[Section]) -> Section:
    found = [s for s in sections if s.heading.startswith(prefix)]
    if len(found) != 1:
        raise KBError(f"heading prefix {prefix!r} matches {len(found)} sections, need exactly 1")
    return found[0]


def build_units(source: Source, rulings: dict = RULINGS) -> list[Unit]:
    """One unit per section, merged along every `referent` ruling. Fails on an unruled hit."""
    unruled = [h for h in find_hits(source.text) if ruling_for(h, rulings) is None]
    if unruled:
        raise UnruledHit(
            "unruled cross-reference hit(s): "
            + "; ".join(f"line {h.line} in {h.heading[:40]!r}: {h.text!r}" for h in unruled)
        )
    parent = {s.index: s.index for s in source.sections}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for h in find_hits(source.text):
        ruling = ruling_for(h, rulings)
        if ruling[0] != "referent":
            continue
        here = _section_for(h.heading, source.sections)
        for prefix in ruling[1:]:
            parent[find(_section_for(prefix, source.sections).index)] = find(here.index)
    groups: dict[int, list[Section]] = {}
    for s in source.sections:
        groups.setdefault(find(s.index), []).append(s)
    units = [Unit(f"u{g[0].index:02d}", g) for g in groups.values()]
    return sorted(units, key=lambda u: u.members[0].index)


# --- D3: header fields -------------------------------------------------------------------------
VERIFY_PHRASES = "verified|measured|confirmed|reproduced|re-run|checked|tested"
VERIFY_WINDOW = 100
VERSION_WINDOW = 40
# Libraries whose version a body may state. Precision over recall (D3 names the miss a cobb review
# judgement): a bare "in 0.2 ms" must never read as a version.
VERSION_LIBS = (
    "starlette|fastapi|anyio|uvicorn|pydantic|httpx2?|redis-py|redis|pytest|ruff|mcp|cpython|python"
)
VERSION_RE = re.compile(
    rf"\b(?P<lib>{VERSION_LIBS})\b[ \t]*(?:\*\*)?[ \t]*(?P<tok>\d+\.\d+(?:\.\d+)?(?:\.x)?)", re.I
)


def extract_versions(body: str) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for m in VERSION_RE.finditer(body):
        key = (m.group("lib").lower(), m.group("tok"))
        if key not in seen:
            seen.add(key)
            out.append((m.group("lib"), m.group("tok")))
    return out


def version_pair_ok(lib: str, tok: str, body: str, window: int = VERSION_WINDOW) -> bool:
    """T6: the token is a body substring with the lib name within `window` chars before it."""
    pos = body.find(tok)
    while pos != -1:
        if lib.lower() in body[max(0, pos - window - len(lib)) : pos].lower():
            return True
        pos = body.find(tok, pos + 1)
    return False


def header_version_pairs(header: str) -> list[tuple[str, str]]:
    """The `<lib> <token>` pairs named by every `Versions` line of a header."""
    pairs: list[tuple[str, str]] = []
    for ln in header.split("\n"):
        m = re.match(r"Versions(?:\[\d+\])?: (.*)$", ln)
        if m and m.group(1) != "none stated":
            pairs += [tuple(p.rsplit(" ", 1)) for p in m.group(1).split("; ")]
    return pairs


def extract_dates(body: str) -> tuple[list[str], list[str]]:
    """(verified, other), both ascending and unique. Verified = a phrase-qualified date."""
    phrase = re.compile(rf"\b(?:{VERIFY_PHRASES})\b", re.I)
    verified: set[str] = set()
    for m in phrase.finditer(body):
        for d in DATE_RE.finditer(body, m.end(), m.end() + VERIFY_WINDOW + 10):
            if d.start() - m.end() <= VERIFY_WINDOW:
                verified.add(d.group(0))
    every = set(DATE_RE.findall(body))
    return sorted(verified), sorted(every - verified)


def _member_lines(body: str, migrated_on: str) -> tuple[str, str, str | None]:
    pairs = extract_versions(body)
    versions = "; ".join(f"{lib} {tok}" for lib, tok in pairs) or "none stated"
    verified, other = extract_dates(body)
    ver = ", ".join(verified) or f"unverified, migrated {migrated_on}"
    return versions, ver, (", ".join(other) or None)


def render_header(unit: Unit, migrated_on: str = MIGRATED_ON) -> str:
    if len(unit.title) > TITLE_LIMIT:
        raise KBError(f"{unit.id}: title is {len(unit.title)} chars, limit {TITLE_LIMIT}")
    lines = [f"Gotcha: {unit.title}"]
    merged = len(unit.members) > 1
    if merged:
        lines.append("Members:")
        lines += [f"- {m.heading}" for m in unit.members]
    for n, m in enumerate(unit.members, 1):
        versions, verified, other = _member_lines(m.body, migrated_on)
        tag = f"[{n}]" if merged else ""
        lines.append(f"Versions{tag}: {versions}")
        lines.append(f"Verified{tag}: {verified}")
        if other:
            lines.append(f"Other-dates{tag}: {other}")
    return "\n".join(lines)


def check_body_safe(unit: Unit) -> None:
    for m in unit.members:
        for ln in m.body.split("\n"):
            if FENCE_SHAPED_RE.match(ln):
                raise KBError(f"{unit.id}: a body line collides with a copy delimiter: {ln!r}")


def render_unit(unit: Unit, migrated_on: str = MIGRATED_ON) -> str:
    """The unit's stored text: byte-identical to what is ingested (fence line excluded)."""
    check_body_safe(unit)
    parts = [render_header(unit, migrated_on), BODY_DELIM]
    text = "\n".join(parts) + "\n" + "".join(f"## {m.heading}{m.body}" for m in unit.members)
    return text.rstrip() + "\n"


# --- copy of record ----------------------------------------------------------------------------
NOTE_TEXT = (
    "This file is the source of record for the python-web-quirks gotchas. The `ws:agent-team`\n"
    "knowledge base holds a rebuildable index of it (one document per unit); new gotchas enter\n"
    "by raw capture and then `cobb` promotes them here. Lineage: distilled 2026-08-09 from the\n"
    "`analyst` agent's learnings inbox (`claude/analyst/kaizen/inbox.md`) via the\n"
    "`agent-maintenance` skill §5."
)


def render_copy(source: Source, units: list[Unit], migrated_on: str = MIGRATED_ON, pin: str = "") -> str:
    verbatim, _waived = split_preamble(source.preamble)
    note = f"{NOTE_TEXT}\nMigrated: {migrated_on} from {SOURCE_PATH} at {pin}"
    out = [PREAMBLE_FENCE, verbatim, NOTE_FENCE, note]
    for u in units:
        out += [f"<!-- unit {u.id} -->", render_unit(u, migrated_on).rstrip("\n")]
    return "\n".join(out) + "\n"


@dataclass
class CopyUnit:
    id: str
    text: str  # stored text (fence line excluded), rstripped + "\n"
    header: str = ""
    parts: list[tuple[str, str]] = field(default_factory=list)  # (heading, body)
    delim_count: int = 0


@dataclass
class Copy:
    preamble: str
    note: str
    units: list[CopyUnit]


def parse_copy(text: str) -> Copy:
    lines = text.split("\n")
    if not lines or lines[0] != PREAMBLE_FENCE:
        raise KBError(f"copy must open with {PREAMBLE_FENCE!r}")
    blocks: list[tuple[str, list[str]]] = []  # (kind, lines)
    for ln in lines:
        m = UNIT_FENCE_RE.match(ln)
        if ln == PREAMBLE_FENCE:
            blocks.append(("preamble", []))
        elif ln == NOTE_FENCE:
            blocks.append(("note", []))
        elif m:
            blocks.append((m.group(1), []))
        elif blocks:
            blocks[-1][1].append(ln)
    kinds = [k for k, _ in blocks]
    if kinds[:2] != ["preamble", "note"] or kinds.count("preamble") != 1 or kinds.count("note") != 1:
        raise KBError("copy must have exactly one preamble block followed by one preamble-note block")
    units: list[CopyUnit] = []
    for uid, body_lines in blocks[2:]:
        stored = "\n".join(body_lines).rstrip() + "\n"
        cu = CopyUnit(uid, stored)
        ls = stored.split("\n")
        cu.delim_count = sum(1 for ln in ls if ln == BODY_DELIM)
        if cu.delim_count:
            at = ls.index(BODY_DELIM)
            cu.header = "\n".join(ls[:at])
            rest = "\n".join(ls[at + 1 :])
            ms = list(HEADING_RE.finditer(rest))
            for i, m in enumerate(ms):
                end = ms[i + 1].start() if i + 1 < len(ms) else len(rest)
                cu.parts.append((m.group(1), rest[m.end() : end]))
        units.append(cu)
    return Copy("\n".join(blocks[0][1]).strip("\n"), "\n".join(blocks[1][1]).strip("\n"), units)


def listing(copy_text: str) -> str:
    """What `grep -n -E '^(<!-- (unit |preamble)|## )' <copy>` prints (D6 item 3)."""
    out = [f"{no}:{ln}\n" for no, ln in enumerate(copy_text.split("\n"), 1) if LISTING_RE.match(ln)]
    return "".join(out)


def listing_bytes(copy_text: str) -> int:
    return len(listing(copy_text).encode("utf-8"))


def verify_copy(copy_text: str, source_text: str, bound: int = LISTING_BOUND) -> list[str]:
    """Problems found (empty = clean). Recomputes every header; compares bodies with the source."""
    problems: list[str] = []
    try:
        copy = parse_copy(copy_text)
    except KBError as e:
        return [str(e)]
    source = parse_source(source_text)
    try:
        expected = build_units(source)
    except KBError as e:
        return [f"source does not build: {e}"]
    verbatim, _waived = split_preamble(source.preamble)
    if copy.preamble != verbatim:
        problems.append("preamble block differs from the verbatim source preamble")
    if re.search(r"^## ", copy.note, re.M) or any(FENCE_SHAPED_RE.match(n) for n in copy.note.split("\n")):
        problems.append("preamble-note holds a '## ' or fence-shaped line")
    m = re.search(r"^Migrated: (\d{4}-\d{2}-\d{2})\b", copy.note, re.M)
    if not m:
        problems.append("preamble-note has no 'Migrated: <date>' line")
    migrated_on = m.group(1) if m else MIGRATED_ON
    have = {u.id: u for u in copy.units}
    if [u.id for u in copy.units] != [u.id for u in expected]:
        problems.append(
            f"unit ids/order differ: copy {[u.id for u in copy.units]} vs source {[u.id for u in expected]}"
        )
    for exp in expected:
        cu = have.get(exp.id)
        if cu is None:
            problems.append(f"{exp.id}: unit missing from the copy ({exp.title[:60]!r})")
            continue
        if cu.delim_count != 1:
            problems.append(f"{exp.id}: body delimiter appears {cu.delim_count} times, need exactly 1")
            continue
        for lib, tok in header_version_pairs(cu.header):
            if not any(version_pair_ok(lib, tok, m.body) for m in exp.members):
                problems.append(f"{exp.id}: invented version {lib} {tok!r}: not in a body with the lib within {VERSION_WINDOW} chars")
        want = render_header(exp, migrated_on)
        if cu.header != want:
            diff = next(
                (f"copy {a!r} vs recomputed {b!r}" for a, b in zip(cu.header.split("\n"), want.split("\n")) if a != b),
                "line count differs",
            )
            problems.append(f"{exp.id}: header mismatch: {diff}")
        got_heads = [h for h, _ in cu.parts]
        want_heads = [s.heading for s in exp.members]
        if got_heads != want_heads:
            problems.append(f"{exp.id}: member headings/order differ: {got_heads} vs {want_heads}")
            continue
        for (h, body), sec in zip(cu.parts, exp.members):
            if body.rstrip() != sec.body.rstrip():
                problems.append(f"{exp.id}: body differs from the pin in section {h[:60]!r}")
    size = listing_bytes(copy_text)
    if size > bound:
        problems.append(f"title listing is {size} bytes, bound {bound}")
    return problems


# --- units subcommand --------------------------------------------------------------------------
def _load_chunker():
    path = REPO_ROOT / "falkor-chat" / "server" / "falkorchat" / "chunking.py"
    spec = importlib.util.spec_from_file_location("pwq_chunking", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pwq_chunking"] = mod
    spec.loader.exec_module(mod)
    return mod.split_into_chunks


def units_report(source: Source, chunks: bool = False, migrated_on: str = MIGRATED_ON) -> str:
    out: list[str] = []
    hits = find_hits(source.text)
    out.append(f"== cross-reference hits ({len(hits)}) ==")
    unruled = 0
    for h in hits:
        r = ruling_for(h, RULINGS)
        unruled += r is None
        label = "UNRULED" if r is None else (r[0] if r[0] != "referent" else "referent:" + " + ".join(p[:30] for p in r[1:]))
        out.append(f"line {h.line}\t§{h.section}\t{h.text!r}\t{label}\t{h.heading[:50]}")
    if unruled:
        out.append(f"BUILD FAILS: {unruled} unruled hit(s)")
        return "\n".join(out) + "\n"
    units = build_units(source)
    out.append(f"== units ({len(units)} units from {len(source.sections)} sections) ==")
    for u in units:
        out.append(f"{u.id}\tsections {','.join(str(m.index) for m in u.members)}\t{u.title[:70]}")
    sizes = [len(render_header(u, migrated_on)) + len(BODY_DELIM) + 2 for u in units]
    out.append(f"== header sizes: min {min(sizes)} max {max(sizes)} chars ==")
    if chunks:
        split = _load_chunker()
        out.append("== chunks (size 1000, overlap 150) ==")
        total = 0
        counts = []
        for u in units:
            n = len(split(render_unit(u, migrated_on)))
            counts.append(n)
            total += n
            out.append(f"{u.id}\t{n}\t{u.title[:60]}")
        out.append(f"total chunks {total}; units with >= 4 chunks: {sum(1 for c in counts if c >= 4)}")
    return "\n".join(out) + "\n"


# --- CLI ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--pin", required=True, help="explicit commit sha of the source (never a ref)")
        p.add_argument("--repo", default=str(REPO_ROOT), help="git repository holding the pin")

    p = sub.add_parser("units", help="print hits, rulings, units, header sizes")
    common(p)
    p.add_argument("--chunks", action="store_true", help="also print the per-unit chunk table")
    p = sub.add_parser("render-copy", help="render the copy of record")
    common(p)
    p.add_argument("--migrated-on", default=MIGRATED_ON)
    p.add_argument("--out", help="write here instead of stdout")
    p = sub.add_parser("verify-copy", help="recompute headers, compare bodies with the pin")
    common(p)
    p.add_argument("--copy", required=True, help="path of the copy to verify")
    args = ap.parse_args(argv)

    try:
        text = read_pinned(args.pin, repo=Path(args.repo))
        source = parse_source(text)
        if args.cmd == "units":
            report = units_report(source, args.chunks)
            sys.stdout.write(report)
            return 1 if "BUILD FAILS" in report else 0
        if args.cmd == "render-copy":
            copy = render_copy(source, build_units(source), args.migrated_on, args.pin)
            if args.out:
                Path(args.out).write_text(copy, encoding="utf-8")
            else:
                sys.stdout.write(copy)
            return 0
        problems = verify_copy(Path(args.copy).read_text(encoding="utf-8"), text)
        for pr in problems:
            print(f"FAIL: {pr}")
        print("verify-copy: " + ("clean" if not problems else f"{len(problems)} problem(s)"))
        return 1 if problems else 0
    except KBError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
