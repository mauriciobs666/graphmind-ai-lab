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
  eval                   S3b: the evaluation harness (D9/D11): --set A [--before], --set B --probe |
                         --sealed-once, --ac2, --sweep, --baseline bm25; scoring modes re-run the gate
  ingest / gate / diverge / replace / retire
                         S3a: the KB side. Every command names `--ws` and (for writes) `--base-url`;
                         the pair must match the committed TARGETS table. `replace`/`retire` refuse
                         production outright and print the stakeholder's `curl -X DELETE` line.

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
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
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
_VTOK = r"\d+\.\d+(?:\.\d+)?(?:\.x)?"
# `<lib> [**]<tok>`: the lib and its token may be separated by spaces, bold marks and ONE hard line wrap
# (never a paragraph break).
VERSION_RE = re.compile(
    rf"\b(?P<lib>{VERSION_LIBS})\b[ \t]*(?:\*\*)?[ \t]*\n?[ \t]*(?:\*\*)?[ \t]*(?P<tok>{_VTOK})", re.I
)
# `<lib> 1.3.1 (...) and **1.6.0**` / `<lib> 1.3.1, 1.4.0`: further tokens of the same lib after a
# separator word. A measurement ("and 0.5 s") is not a version.
_CONT_RE = re.compile(
    rf"(?:\*\*)?(?:\s*\([^()]{{0,120}}\))?\s*(?:,|\band\b|\bor\b)\s*(?:\*\*)?(?P<tok>{_VTOK})(?![\d.]*\d)"
    r"(?!(?:\*\*)?[ \t]*(?:ms|s|sec|seconds?|MB|KB|%)\b)",
    re.I,
)


def extract_versions(body: str) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []

    def add(lib: str, tok: str) -> None:
        key = (lib.lower(), tok)
        if key not in seen:
            seen.add(key)
            out.append((lib, tok))

    for m in VERSION_RE.finditer(body):
        add(m.group("lib"), m.group("tok"))
        pos = m.end()
        while (c := _CONT_RE.match(body, pos)):
            add(m.group("lib"), c.group("tok"))
            pos = c.end()
    return out


def version_pair_ok(lib: str, tok: str, body: str, window: int = VERSION_WINDOW) -> bool:
    """T6: the token is a body substring with the lib name within `window` chars before it, or the
    extractor itself reads the pair from the body (a list continuation can sit past the window)."""
    if (lib.lower(), tok) in {(l.lower(), t) for l, t in extract_versions(body)}:
        return True
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
    seen_id: set[str] = set()
    title_owner: dict[str, str] = {}
    for uid, body_lines in blocks[2:]:
        stored = "\n".join(body_lines).rstrip() + "\n"
        if uid in seen_id:
            raise KBError(f"duplicate unit id {uid}")
        seen_id.add(uid)
        first = stored.split("\n", 1)[0]
        title = first[len("Gotcha: "):] if first.startswith("Gotcha: ") else first
        if title in title_owner:
            raise KBError(f"duplicate title in {uid} and {title_owner[title]}: {title[:60]!r}")
        title_owner[title] = uid
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


_VERSIONS_LINE_RE = re.compile(r"^Versions(?:\[(\d+)\])?: (.*)$")


def _header_problems(exp: Unit, header: str, migrated_on: str) -> list[str]:
    """Header vs the recomputed one. Every line must match exactly except a `Versions` line, which
    must be a SUPERSET of the extracted pairs: cobb may add a pair he confirmed by hand (it must pass
    `version_pair_ok` against that member's own body) but may never drop an extracted one."""
    want = render_header(exp, migrated_on).split("\n")
    got = header.split("\n")
    if len(got) != len(want):
        return [f"{exp.id}: header mismatch: line count differs"]
    problems: list[str] = []
    for a, b in zip(got, want):
        ma, mb = _VERSIONS_LINE_RE.match(a), _VERSIONS_LINE_RE.match(b)
        if not (ma and mb and ma.group(1) == mb.group(1)):
            if a != b:
                problems.append(f"{exp.id}: header mismatch: copy {a!r} vs recomputed {b!r}")
            continue
        body = exp.members[int(ma.group(1) or 1) - 1].body
        parse = lambda v: [] if v == "none stated" else [tuple(p.rsplit(" ", 1)) if " " in p else (p, "") for p in v.split("; ")]
        have, need = parse(ma.group(2)), parse(mb.group(2))
        have_keys = {(lib.lower(), tok) for lib, tok in have}
        for lib, tok in need:
            if (lib.lower(), tok) not in have_keys:
                problems.append(f"{exp.id}: header omits extracted version {lib} {tok!r} ({mb.group(0).split(':')[0]})")
        need_keys = {(lib.lower(), tok) for lib, tok in need}
        for lib, tok in have:
            if (lib.lower(), tok) not in need_keys and not version_pair_ok(lib, tok, body):
                problems.append(
                    f"{exp.id}: invented version {lib} {tok!r}: not in this member's body with the lib within {VERSION_WINDOW} chars"
                )
    return problems


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
        problems += _header_problems(exp, cu.header, migrated_on)
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



# --- S3a: target guard, REST client, fetcher, manifest (D4, D8, D9) ---------------------------------
# The committed target table (D8). The REST client cannot ask a server which workspace it serves, so
# the guard keys on the URL actually written to: the (ws, base-url) pair must be a row of this table.
TARGETS = {"agent-team": "http://localhost:8200", "pwq-test": "http://localhost:8210"}
PRODUCTION_WS = "agent-team"
DEFAULT_COPY = REPO_ROOT / "claude" / "knowledge" / "python-web-quirks.md"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "pwq-kb-manifest.json"
PRODUCED_BY = "cobb"
GATE_TIMEOUT = 1800  # seconds PER DOCUMENT (RCA: a document's extract chain on one local model can take ~30 min)
GATE_POLL = 5
FAILED = "failed"  # Document.status of a document whose background processing failed


class TargetRefused(KBError):
    """The guard (or a production-delete refusal) stopped the run before any request was made."""


class PostReceiptMismatch(KBError):
    """A POST's receipt id is not visible in the target workspace: the write landed somewhere else (F6/F14)."""


@dataclass
class Target:
    ws: str
    base_url: str
    production: bool


def check_target(ws: str, base_url: str, targets: dict = TARGETS, confirm_production: bool = False) -> Target:
    if ws not in targets:
        raise TargetRefused(f"unknown workspace {ws!r}; the target table has {sorted(targets)}")
    url = base_url.rstrip("/")
    if url != targets[ws]:
        raise TargetRefused(f"--base-url {base_url!r} does not match the target table entry for --ws {ws} ({targets[ws]})")
    production = ws == PRODUCTION_WS
    if production and not confirm_production:
        raise TargetRefused("production ingest needs --confirm-production")
    return Target(ws, url, production)


def curl_delete_line(target: Target, document_id: str) -> str:
    return f"curl -X DELETE {target.base_url}/documents/{document_id}"


def urllib_transport(method: str, url: str, body: dict | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise KBError(f"{method} {url} failed: {e}") from e
    try:
        return status, json.loads(raw.decode("utf-8")) if raw else {}
    except ValueError:
        return status, {"raw": raw.decode("utf-8", "replace")}


class RestClient:
    """Writes to exactly one validated target. A production client has no delete at all."""

    def __init__(self, target: Target, transport=urllib_transport):
        self.target, self._transport = target, transport

    def post_document(self, body: dict) -> dict:
        status, resp = self._transport("POST", f"{self.target.base_url}/documents", body)
        if status != 201:
            raise KBError(f"POST /documents failed: HTTP {status} {resp}")
        return resp

    def delete_document(self, document_id: str) -> None:
        if self.target.production:
            raise TargetRefused(
                "refusing to delete in production; the stakeholder runs: " + curl_delete_line(self.target, document_id)
            )
        status, resp = self._transport("DELETE", f"{self.target.base_url}/documents/{document_id}", None)
        if status != 200:
            raise KBError(f"DELETE /documents/{document_id} failed: HTTP {status} {resp}")


# Reads over falkordb (cypher-mcp/.venv), like check_content_loss.py. stdin = one JSON request.
FETCH_SNIPPET = r"""
import json, os, sys
from falkordb import FalkorDB

req = json.loads(sys.stdin.read())
db = FalkorDB(host=os.environ.get("FALKORDB_HOST", "127.0.0.1"), port=int(os.environ.get("FALKORDB_PORT", "6379")))
g = db.select_graph("ws:" + req["ws"])
if req["op"] == "docs":
    res = g.ro_query(
        "MATCH (d:Document {currentVersion: true}) WHERE d.title IN $titles OR d.documentId IN $ids "
        "RETURN d.documentId, d.title, d.text, d.status",
        params={"titles": req["titles"], "ids": req["ids"]},
    )
    out = [{"documentId": r[0], "title": r[1], "text": r[2], "status": r[3]} for r in res.result_set]
else:
    out = {}
    for i in req["ids"]:
        rows = g.ro_query(
            "MATCH (d:Document {documentId: $id})-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN count(c), count(c.embedding), d.status", params={"id": i},
        ).result_set
        if rows and rows[0][0]:
            out[i] = {"chunks": rows[0][0], "embedded": rows[0][1], "status": rows[0][2]}
            continue
        st = g.ro_query("MATCH (d:Document {documentId: $id}) RETURN d.status", params={"id": i}).result_set
        out[i] = {"chunks": 0, "embedded": 0, "status": st[0][0]} if st else None
sys.stdout.write(json.dumps(out))
"""


class CypherFetcher:
    """Real fetcher: read-only queries through the cypher-mcp venv's `falkordb` client."""

    def _run(self, request: dict):
        py = os.environ.get("CYPHER_MCP_VENV_PYTHON") or str(REPO_ROOT / "cypher-mcp" / ".venv" / "bin" / "python3")
        if not Path(py).is_file():
            raise KBError(f"fetcher needs {py} (run cypher-mcp/setup.sh)")
        proc = subprocess.run([py, "-c", FETCH_SNIPPET], input=json.dumps(request).encode(), capture_output=True)
        if proc.returncode != 0:
            raise KBError(f"fetch failed: {proc.stderr.decode(errors='replace').strip()[-300:]}")
        return json.loads(proc.stdout.decode())

    def current_docs(self, ws: str, titles: list[str], ids: list[str]) -> list[dict]:
        return self._run({"op": "docs", "ws": ws, "titles": titles, "ids": ids})

    def embedding_counts(self, ws: str, ids: list[str]) -> dict:
        return self._run({"op": "counts", "ws": ws, "ids": ids})


def load_manifest(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise KBError(f"manifest {path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CopyEntry:
    id: str
    title: str
    text: str
    headings: list[str]


def copy_entries(copy_text: str) -> list[CopyEntry]:
    out = []
    for cu in parse_copy(copy_text).units:
        first = cu.text.split("\n", 1)[0]
        title = first[len("Gotcha: "):] if first.startswith("Gotcha: ") else first
        out.append(CopyEntry(cu.id, title, cu.text, [h for h, _ in cu.parts]))
    return out


def copy_note_fields(copy_text: str) -> tuple[str, str]:
    """(pin, migratedOn) from the preamble-note's `Migrated:` line."""
    m = re.search(r"^Migrated: (\d{4}-\d{2}-\d{2}) from \S+ at (\S+)\s*$", parse_copy(copy_text).note, re.M)
    if not m:
        raise KBError("the copy's preamble-note has no 'Migrated: <date> from <path> at <pin>' line")
    return m.group(2), m.group(1)


def _manifest_path(target: Target, path) -> Path:
    if path is None:
        if not target.production:
            raise TargetRefused("a non-production run needs an explicit --manifest (never the production one)")
        return DEFAULT_MANIFEST
    return Path(path)


@dataclass
class UnitResult:
    id: str
    action: str  # skip | adopt | ingest | refuse
    documentId: str = ""
    note: str = ""
    state: str = ""  # "" | failed | timeout: how the paced wait after a POST ended (any non-empty state makes the run exit non-zero)


def _terminal(c, strict: bool) -> bool:
    """A document's counts row is terminal. `strict` (pacing): every chunk embedded AND status not processing, or failed.
    Non-strict (the embedding gate): every chunk embedded, or failed, or no chunks, or the document is absent."""
    if c is None:
        return not strict
    if c["status"] == FAILED:
        return True
    complete = c["chunks"] > 0 and c["embedded"] >= c["chunks"]
    return complete and c["status"] != "processing" if strict else (complete or c["chunks"] == 0)


def wait_terminal(fetcher, ws: str, document_id: str, *, timeout: float, poll: float, sleep, monotonic, strict: bool):
    """Poll `ws` (read-only) until the document is terminal or `timeout` seconds pass: (counts row, timed_out)."""
    deadline = monotonic() + timeout
    while True:
        c = fetcher.embedding_counts(ws, [document_id]).get(document_id)
        if _terminal(c, strict):
            return c, False
        if monotonic() >= deadline:
            return c, True
        sleep(poll)


def _paced(fetcher, ws: str, unit_id: str, document_id: str, timeout, poll, sleep, monotonic) -> UnitResult:
    c, timed_out = wait_terminal(fetcher, ws, document_id, timeout=timeout, poll=poll, sleep=sleep, monotonic=monotonic, strict=True)
    if timed_out:
        return UnitResult(unit_id, "ingest", document_id, f"still processing after {timeout:g} s", "timeout")
    if c is not None and c["status"] == FAILED:
        why = "chunks embedded; entities missing" if c["chunks"] and c["embedded"] >= c["chunks"] else \
            f"{c['embedded']}/{c['chunks']} chunks embedded"
        return UnitResult(unit_id, "ingest", document_id, f"failed ({why})", "failed")
    return UnitResult(unit_id, "ingest", document_id)


def _entry_dict(e: CopyEntry, document_id: str) -> dict:
    return {"id": e.id, "title": e.title, "headings": e.headings, "documentId": document_id, "sha256": sha256_text(e.text)}


def _post_body(e: CopyEntry) -> dict:
    return {"text": e.text, "title": e.title, "sourceFormat": "markdown",
            "sourceLabel": f"python-web-quirks {e.id}", "producedBy": PRODUCED_BY}


def ingest(copy_text: str, *, ws: str, base_url: str, manifest_path, transport=urllib_transport, fetcher=None,
           confirm_production: bool = False, targets: dict = TARGETS, read_source=read_pinned,
           pace_timeout: float = GATE_TIMEOUT, poll: float = GATE_POLL, sleep=time.sleep, monotonic=time.monotonic) -> list[UnitResult]:
    """Idempotent ingest of every copy unit (D8). Never posts over a different-text current document. Paced (RCA): after each
    POST the next one waits until that document is terminal (`pace_timeout` per document); a failed or timed-out document
    is reported in its result's `state` and the batch continues."""
    target = check_target(ws, base_url, targets, confirm_production)
    mpath = _manifest_path(target, manifest_path)
    fetcher = fetcher or CypherFetcher()
    client = RestClient(target, transport)
    pin, migrated_on = copy_note_fields(copy_text)
    waived = split_preamble(parse_source(read_source(pin)).preamble)[1]
    old = {u["id"]: u for u in (load_manifest(mpath)["units"] if mpath.is_file() else [])}
    entries = copy_entries(copy_text)
    results: list[UnitResult] = []
    new_entries: dict[str, dict] = {}
    for e in entries:
        reg = old.get(e.id)
        docs = fetcher.current_docs(ws, [e.title], [reg["documentId"]] if reg else [])
        docs = list({d["documentId"]: d for d in docs}.values())
        differing = [d for d in docs if d["text"] != e.text]
        identical = [d for d in docs if d["text"] == e.text]
        good = [d for d in identical if d["status"] != FAILED]
        bad = [d for d in identical if d["status"] == FAILED]
        if differing:
            note = "current document(s) with different text; the stakeholder deletes first: " + "; ".join(
                curl_delete_line(target, d["documentId"]) for d in differing)
            results.append(UnitResult(e.id, "refuse", note=note))
            if reg:
                new_entries[e.id] = reg
            continue
        by_reg = [d for d in good if reg and d["documentId"] == reg["documentId"]]
        if by_reg:
            results.append(UnitResult(e.id, "skip", by_reg[0]["documentId"]))
            new_entries[e.id] = _entry_dict(e, by_reg[0]["documentId"])
        elif good:
            results.append(UnitResult(e.id, "adopt", good[0]["documentId"]))
            new_entries[e.id] = _entry_dict(e, good[0]["documentId"])
        elif bad and target.production:
            note = "identical current document has status failed; the stakeholder deletes it: " + "; ".join(
                curl_delete_line(target, d["documentId"]) for d in bad)
            results.append(UnitResult(e.id, "refuse", note=note))
            if reg:
                new_entries[e.id] = reg
        else:
            receipt = client.post_document(_post_body(e))
            if not fetcher.current_docs(ws, [], [receipt["documentId"]]):  # F6/F14: visible in the TARGET workspace, not elsewhere
                raise PostReceiptMismatch(
                    f"document {receipt['documentId']} (unit {e.id}) is not visible in the target workspace {ws} after the POST to "
                    f"{target.base_url}: it landed somewhere else; stopping. Find and delete the stray by that id")
            results.append(_paced(fetcher, ws, e.id, receipt["documentId"], pace_timeout, poll, sleep, monotonic))
            new_entries[e.id] = _entry_dict(e, receipt["documentId"])
    in_copy = {e.id for e in entries}
    units = [new_entries[e.id] for e in entries if e.id in new_entries]
    units += [u for i, u in old.items() if i not in in_copy]  # keep orphans registered until retired
    save_manifest(mpath, {"pin": pin, "migratedOn": migrated_on, "units": units, "waivedSpans": [waived] if waived else []})
    return results


def _test_only_unit(copy_text: str, unit: str, ws: str, base_url: str, manifest_path, targets: dict):
    target = check_target(ws, base_url, targets, confirm_production=True)  # identification only; never a licence
    mpath = _manifest_path(target, manifest_path)
    manifest = load_manifest(mpath)
    reg = next((u for u in manifest["units"] if u["id"] == unit), None)
    if target.production:
        hint = curl_delete_line(target, reg["documentId"]) if reg else f"(unit {unit} has no manifest entry)"
        raise TargetRefused(f"replace/retire never act on production; the stakeholder runs: {hint}")
    if reg is None:
        raise KBError(f"unit {unit} has no manifest entry in {mpath}")
    return target, mpath, manifest, reg


def retire(copy_text: str, *, unit: str, ws: str, base_url: str, manifest_path, transport=urllib_transport, fetcher=None,
           confirm_production: bool = False, targets: dict = TARGETS, read_source=read_pinned) -> None:
    target, mpath, manifest, reg = _test_only_unit(copy_text, unit, ws, base_url, manifest_path, targets)
    RestClient(target, transport).delete_document(reg["documentId"])
    manifest["units"] = [u for u in manifest["units"] if u["id"] != unit]
    save_manifest(mpath, manifest)


def replace(copy_text: str, *, unit: str, ws: str, base_url: str, manifest_path, transport=urllib_transport, fetcher=None,
            confirm_production: bool = False, targets: dict = TARGETS, read_source=read_pinned,
            pace_timeout: float = GATE_TIMEOUT, poll: float = GATE_POLL, sleep=time.sleep, monotonic=time.monotonic) -> UnitResult:
    target, mpath, manifest, reg = _test_only_unit(copy_text, unit, ws, base_url, manifest_path, targets)
    entry = next((e for e in copy_entries(copy_text) if e.id == unit), None)
    if entry is None:
        raise KBError(f"unit {unit} is not in the copy")
    client = RestClient(target, transport)
    client.delete_document(reg["documentId"])
    receipt = client.post_document(_post_body(entry))
    manifest["units"] = [_entry_dict(entry, receipt["documentId"]) if u["id"] == unit else u for u in manifest["units"]]
    save_manifest(mpath, manifest)
    return _paced(fetcher or CypherFetcher(), ws, unit, receipt["documentId"], pace_timeout, poll, sleep, monotonic)


# --- gate (K-102 embedding hard gate) ----------------------------------------------------------
def expected_unit_ids(copy_path) -> list[str]:
    """The unit ids the copy of record holds: what a complete manifest must register."""
    p = Path(copy_path)
    if not p.is_file():
        raise KBError(f"copy {p} not found: the gate cannot tell a partial manifest without it")
    return [e.id for e in copy_entries(p.read_text(encoding="utf-8"))]


def expected_chunk_counts(copy_text: str) -> dict[str, int]:
    """unit id -> the chunk count the server's chunker derives from the unit's stored text (the copy entry), the same
    derivation `units --chunks` prints. A chunker that cannot be loaded or run raises: the check is never skipped."""
    split = _load_chunker()
    return {e.id: len(split(e.text)) for e in copy_entries(copy_text)}


def run_gate(fetcher, ws: str, manifest: dict, *, timeout: float = GATE_TIMEOUT, poll: float = GATE_POLL,
             sleep=time.sleep, monotonic=time.monotonic, expected_ids=None, expected_chunks=None) -> tuple[bool, str]:
    """`expected_ids` (the copy's unit ids): one the manifest does not register is an ABSENT row, so a
    unit that `ingest` refused cannot leave the gate clean on a partial manifest. `expected_chunks` (unit id -> count,
    `expected_chunk_counts`): a unit whose stored chunk count differs, or that has no expected count, fails (a deleted
    chunk leaves the rest embedded, so embedding alone cannot see it)."""
    ids = [u["documentId"] for u in manifest["units"]]
    registered = {u["id"] for u in manifest["units"]}
    absent = [i for i in (expected_ids or []) if i not in registered]
    for i in ids:  # each document gets its own `timeout` window, in manifest order
        wait_terminal(fetcher, ws, i, timeout=timeout, poll=poll, sleep=sleep, monotonic=monotonic, strict=False)
    counts = fetcher.embedding_counts(ws, ids)
    rows = []
    for u in manifest["units"]:
        c = counts.get(u["documentId"])
        if c is None:
            rows.append((u["id"], "-", "-", "MISSING", False))
        elif c["status"] == FAILED:
            extra = " (chunks embedded; entities missing)" if c["chunks"] and c["embedded"] >= c["chunks"] else ""
            rows.append((u["id"], c["chunks"], c["embedded"], "FAILED" + extra, False))
        elif c["chunks"] == 0:
            rows.append((u["id"], 0, 0, "NO CHUNKS", False))
        elif c["embedded"] < c["chunks"]:
            rows.append((u["id"], c["chunks"], c["embedded"], f"UNEMBEDDED {c['chunks'] - c['embedded']}/{c['chunks']}", True))
        elif expected_chunks is not None and u["id"] not in expected_chunks:
            rows.append((u["id"], c["chunks"], c["embedded"], "no expected chunk count (unit not in the copy)", False))
        elif expected_chunks is not None and c["chunks"] != expected_chunks[u["id"]]:
            rows.append((u["id"], c["chunks"], c["embedded"], f"chunks {c['chunks']} != expected {expected_chunks[u['id']]}", False))
        else:
            rows.append((u["id"], c["chunks"], c["embedded"], "ok", False))
    rows += [(i, "-", "-", "ABSENT", False) for i in absent]
    table = "unit chunks embedded status\n" + "".join(f"{a} {b} {c} {d}\n" for a, b, c, d, _ in rows)
    return all(r[3] == "ok" for r in rows), table


# --- diverge -----------------------------------------------------------------------------------
def diverge(copy_text: str, *, ws: str, manifest: dict, fetcher) -> list[tuple[str, str, str]]:
    """(kind, unit id, title) for every way the KB differs from the copy; empty = faithful."""
    entries = copy_entries(copy_text)
    reg = {u["id"]: u for u in manifest["units"]}
    docs = fetcher.current_docs(ws, [e.title for e in entries] + [u["title"] for u in manifest["units"]],
                                [u["documentId"] for u in manifest["units"]])
    by_id = {d["documentId"]: d for d in docs}
    out: list[tuple[str, str, str]] = []
    # the manifest's pin/migratedOn/sha256 are written by ingest and read only here: a registry built
    # from another copy (or an older text of this one) is stale even when the KB matches the copy
    try:
        pin, migrated_on = copy_note_fields(copy_text)
        for key, want in (("pin", pin), ("migratedOn", migrated_on)):
            if manifest.get(key) != want:
                out.append(("REGISTRY-STALE", "-", f"manifest {key} {manifest.get(key)!r} != the copy note's {want!r}"))
    except KBError as e:
        out.append(("REGISTRY-STALE", "-", f"the copy's note cannot be read: {e}"))
    for e in entries:
        if e.id in reg and reg[e.id].get("sha256") != sha256_text(e.text):
            out.append(("REGISTRY-STALE", e.id, e.title))
        mine = by_id.get(reg[e.id]["documentId"]) if e.id in reg else None
        titled = [d for d in docs if d["title"] == e.title]
        if mine is None and not titled:
            out.append(("MISSING", e.id, e.title))
            continue
        if mine is not None:
            if sha256_text(mine["text"]) != sha256_text(e.text):
                out.append(("STALE", e.id, e.title))
            if mine["title"] != e.title:
                out.append(("TITLE-DRIFT", e.id, e.title))
        elif not any(d["text"] == e.text for d in titled):
            out.append(("STALE", e.id, e.title))
        if len(titled) > 1:
            out.append(("DUPLICATE", e.id, e.title))
    in_copy = {e.id for e in entries}
    for u in manifest["units"]:
        if u["id"] not in in_copy and u["documentId"] in by_id:
            out.append(("ORPHAN", u["id"], u["title"]))
    # a unit the listing cannot show (no fence, or a `###` heading) still has its `Gotcha:` header line
    listed = [e.title for e in entries]
    for m in re.finditer(r"^Gotcha: (.*)$", copy_text, re.M):
        if m.group(1) in listed:
            listed.remove(m.group(1))
        else:
            out.append(("UNLISTED", "-", m.group(1)))
    return out



# --- S3b: eval (D9, D11) -----------------------------------------------------------------------
EVAL_WINDOW = 50  # chunks requested per query; the convention's limit=5 is deliberately exceeded (D11)
EVAL_TOPK = 5
SWEEP_FLOORS = [round(0.38 + 0.01 * i, 2) for i in range(13)]  # 0.38 .. 0.50 (ML §6)
LEAK_SCORE = 0.25  # a Set B probe whose rank-1 score is below this is a leak sign (D11 ruling i)
BM25_K1, BM25_B = 1.5, 0.75
DEFAULT_SKILL = REPO_ROOT / "skills" / "agent-kb-retrieval" / "SKILL.md"
DEFAULT_SET_A = Path(__file__).resolve().parent / "fixtures" / "pwq-golden-set-a.json"


class GateRefused(KBError):
    pass


class SealedRefused(KBError):
    pass


def extract_prefix(skill_text: str) -> str:
    """The query template (with `{situation}`) from the retrieval skill's fenced literal.

    Pure string extraction, never eval/exec: the literal is `f"...\\n...{situation}"`; only the `\\n`
    escape is understood and anything else aborts.
    """
    lines = skill_text.split("\n")
    head = next((i for i, ln in enumerate(lines) if ln.startswith("#") and "query-instruction prefix" in ln), None)
    if head is None:
        raise KBError("retrieval skill: no 'query-instruction prefix' heading")
    fence = [i for i in range(head + 1, len(lines)) if lines[i].startswith("```")]
    if len(fence) < 2:
        raise KBError("retrieval skill: the prefix fence is missing")
    body = [ln for ln in lines[fence[0] + 1 : fence[1]] if ln.strip()]
    if len(body) != 1:
        raise KBError("retrieval skill: the prefix fence must hold exactly one line")
    lit = body[0].strip()
    if not (lit.startswith('f"') and lit.endswith('"')):
        raise KBError("retrieval skill: the prefix literal is not an f-string literal")
    inner = lit[2:-1].replace("\\n", "\n")
    if "\\" in inner or '"' in inner:
        raise KBError("retrieval skill: the prefix literal uses an escape other than \\n")
    if inner.count("{situation}") != 1 or "\n" not in inner:
        raise KBError("retrieval skill: the prefix literal needs one {situation} placeholder and a newline")
    return inner


def build_query(template: str, situation: str) -> str:
    return template.replace("{situation}", situation, 1)


def load_set(path: Path) -> dict:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise KBError(f"cannot read set {path}: {e}") from e
    pos, neg = [], []
    for p in raw.get("positives", []):
        if not (isinstance(p.get("section"), int) and isinstance(p.get("query"), str) and p["query"].strip() and p.get("id")):
            raise KBError(f"bad positive {p.get('id')!r}: needs id, integer section, non-empty query")
        pos.append({"id": p["id"], "section": p["section"], "query": p["query"], "tail": bool(p.get("tail", False))})
    for n in raw.get("negatives", []):
        if not (n.get("kind") in ("near", "off") and isinstance(n.get("query"), str) and n["query"].strip() and n.get("id")):
            raise KBError(f"bad negative {n.get('id')!r}: needs id, kind near|off, non-empty query")
        neg.append({"id": n["id"], "kind": n["kind"], "query": n["query"]})
    return {"positives": pos, "negatives": neg}


SET_B_SECTIONS = 42  # a constant of the spec (ML §1), never derived from the set under test
SET_B_PER_SECTION = 2
SET_B_NEAR = 15
SET_B_OFF = 15
# Set B2 (docs/plans/python-web-quirks-kb-b2-ml.md section 4): spec constants, never derived from the set under test
SET_B2_PER_SECTION = 3
SET_B2_NEAR = 30
SET_B2_OFF = 30
B2_MIN_RECALL = 116  # of 126 floor-applied positives
B2_MAX_NEAR_FA = 6  # of 30
B2_MAX_OFF_FA = 0  # of 30
B2_MAX_OVERALL_FA = 6  # of 60
SET_B_RECORDED = (73, 84)  # Set B's failed score (hits, positives): pooled with B2 as informational text only
SET_PROFILES = {"B": (SET_B_PER_SECTION, SET_B_NEAR, SET_B_OFF), "B2": (SET_B2_PER_SECTION, SET_B2_NEAR, SET_B2_OFF)}
TAIL_MIN_CHUNKS = 4  # ML section 5 rule 4: a unit with this many chunks or more needs a tail probe


def tail_unit_sections(source: Source, split, migrated_on: str = MIGRATED_ON) -> list[tuple[str, list[int]]]:
    """(unit id, member section numbers) of every unit that renders to >= TAIL_MIN_CHUNKS chunks: the units
    ML section 5 rule 4 wants a tail probe for. `split` is the chunker (`_load_chunker()`), the same count
    `units --chunks` prints."""
    return [(u.id, [m.index for m in u.members]) for u in build_units(source)
            if len(split(render_unit(u, migrated_on))) >= TAIL_MIN_CHUNKS]


def validate_set_b(data: dict, tail_units: list[tuple[str, list[int]]] | None = None, profile: str = "B") -> None:
    """ML §1's sealed-set shape: 42 sections x 2 positives = 84, and 15 near + 15 off negatives (profile "B");
    profile "B2": 3 x 42 = 126 positives, 30 near + 30 off.

    Run before anything burns the one-shot seal: the coverage clause of AC-3 divides by
    SET_B_SECTIONS, not by however many sections the loaded set happens to mention.
    `tail_units` (from `tail_unit_sections`, when the caller can derive it at the pin) adds ML section 5
    rule 4: each such unit needs a `tail`-flagged positive on one of its sections."""
    pos, neg = data["positives"], data["negatives"]
    per_section, want_near, want_off = SET_PROFILES[profile]
    label = f"Set {profile} shape"
    want_pos = SET_B_SECTIONS * per_section
    if len(pos) != want_pos:
        raise KBError(f"{label}: {len(pos)} positives, need {want_pos} ({per_section} per section)")
    per: dict[int, int] = {}
    for p in pos:
        per[p["section"]] = per.get(p["section"], 0) + 1
    if set(per) != set(range(1, SET_B_SECTIONS + 1)) or any(n != per_section for n in per.values()):
        bad = sorted(f"{s}:{n}" for s, n in per.items() if s not in range(1, SET_B_SECTIONS + 1) or n != per_section)
        missing = sorted(set(range(1, SET_B_SECTIONS + 1)) - set(per))
        raise KBError(f"{label}: every section 1-{SET_B_SECTIONS} needs exactly {per_section} positives "
                      f"(section:count off {bad}, missing {missing})")
    kinds = {"near": 0, "off": 0}
    for n in neg:
        kinds[n["kind"]] += 1
    if len(neg) != want_near + want_off:
        raise KBError(f"{label}: {len(neg)} negatives, need {want_near + want_off}")
    if (kinds["near"], kinds["off"]) != (want_near, want_off):
        raise KBError(f"{label}: negative split near/off is {kinds['near']}/{kinds['off']}, need {want_near}/{want_off}")
    ids = [i["id"] for i in pos + neg]
    if len(set(ids)) != len(ids):
        raise KBError(f"{label}: duplicate ids " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    flagged = {p["section"] for p in pos if p.get("tail")}
    bare = [f"{uid} (sections {','.join(map(str, secs))})" for uid, secs in tail_units or [] if not flagged & set(secs)]
    if bare:
        raise KBError(f"{label}: no tail-flagged positive for unit {'; '.join(bare)} "
                      f"(each unit of >= {TAIL_MIN_CHUNKS} chunks needs one)")


SET_A_SECTIONS = 42  # a constant of the spec (ML section 3), never derived from the set under test
SET_A_NEAR = 7
SET_A_OFF = 7


def validate_set_a(data: dict) -> None:
    """ML section 3's Set A shape: one positive per section 1-42, and 7 near + 7 off negatives, unique ids.

    Run before every Set A scoring mode (not `--before`): coverage divides by SET_A_SECTIONS and the floor
    (ML section 6) rests on the false-accept counts, so a partial set must not report "ok"."""
    pos, neg = data["positives"], data["negatives"]
    if len(pos) != SET_A_SECTIONS:
        raise KBError(f"Set A shape: {len(pos)} positives, need {SET_A_SECTIONS} (one per section)")
    secs = sorted(p["section"] for p in pos)
    if secs != list(range(1, SET_A_SECTIONS + 1)):
        dup = sorted({n for n in secs if secs.count(n) > 1})
        missing = sorted(set(range(1, SET_A_SECTIONS + 1)) - set(secs))
        raise KBError(f"Set A shape: every section 1-{SET_A_SECTIONS} needs exactly one positive "
                      f"(repeated or out of range {dup or sorted(set(secs) - set(range(1, SET_A_SECTIONS + 1)))}, missing {missing})")
    if len(neg) != SET_A_NEAR + SET_A_OFF:
        raise KBError(f"Set A shape: {len(neg)} negatives, need {SET_A_NEAR + SET_A_OFF}")
    near = sum(1 for n in neg if n["kind"] == "near")
    if (near, len(neg) - near) != (SET_A_NEAR, SET_A_OFF):
        raise KBError(f"Set A shape: negative split near/off is {near}/{len(neg) - near}, need {SET_A_NEAR}/{SET_A_OFF}")
    ids = [i["id"] for i in pos + neg]
    if len(set(ids)) != len(ids):
        raise KBError("Set A shape: duplicate ids " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))


def resolve_set_b_path(arg, env=os.environ, kind: str = "B") -> Path:
    """Set B's path comes only from --set-b-path or PWQ_SET_B, and must be outside the repo (D11). Set B2 has its own
    pair, --set-b2-path and PWQ_SET_B2: B's variable never stands in for it."""
    flag, var = ("--set-b-path", "PWQ_SET_B") if kind == "B" else ("--set-b2-path", "PWQ_SET_B2")
    chosen = arg or env.get(var)
    if not chosen:
        raise KBError(f"Set {kind} path: pass {flag} or set {var} (never read from the source)")
    path = Path(chosen)
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return path
    raise KBError(f"Set {kind} must live outside the repo, got {path}")


def check_expected_sha256(expected, path: Path) -> None:
    """The sealed scoring's freeze check (ML section 5 rule 6): `expected` is the lower-case 64-hex sha256 recorded when
    Set B was frozen, compared exactly (no case folding, no prefix) with the file's own."""
    if expected is None:
        raise KBError("--sealed-once requires --expect-sha256 <sha256 of Set B recorded at freeze>")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise KBError("--expect-sha256 must be 64 lower-case hex characters")
    try:
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as e:
        raise KBError(f"cannot read Set B {path}: {e}") from e
    if actual != expected:
        raise KBError(f"--expect-sha256 mismatch: Set B {path} hashes to {actual}, expected {expected}; "
                      "it was edited after freezing")


class RestSearcher:
    """`search_documents` / `get_document` over the validated target's REST routes."""

    def __init__(self, target: Target, transport=urllib_transport):
        self.target, self._transport = target, transport

    def search(self, ws: str, query: str, limit: int) -> list[dict]:
        qs = urllib.parse.urlencode({"q": query, "limit": limit})
        status, body = self._transport("GET", f"{self.target.base_url}/documents/search?{qs}", None)
        if status != 200 or not isinstance(body, list):
            raise KBError(f"search failed: HTTP {status} {str(body)[:200]}")
        return body

    def get_document(self, ws: str, document_id: str) -> dict | None:
        status, body = self._transport("GET", f"{self.target.base_url}/documents/{document_id}", None)
        if status == 404:
            return None
        if status != 200:
            raise KBError(f"get_document failed: HTTP {status}")
        return body


def window_chunks(searcher, ws: str, query: str, excluded=frozenset(), window: int = EVAL_WINDOW, k: int = EVAL_TOPK):
    return scored_chunks(searcher.search(ws, query, window), excluded, k)


def scored_chunks(fetched: list[dict], excluded=frozenset(), k: int = EVAL_TOPK) -> list[dict]:
    """Drop excluded documents' chunks from the fetched window, then keep the first `k`."""
    return [c for c in fetched if c["documentId"] not in excluded][:k]


def name_hit(searcher, ws: str, chunks: list[dict]):
    """(documentId, title, score) of the best-scoring distinct document among `chunks`, titled via get_document."""
    if not chunks:
        return None
    best = min(chunks, key=lambda c: c["score"])
    doc = searcher.get_document(ws, best["documentId"])
    return best["documentId"], (doc or {}).get("title"), best["score"]


def displacement(pos: list[dict], windows: list[list[dict]], *, unit_doc: dict, feature_docs: set) -> dict:
    """F11 diagnostic: per positive, the 1-based rank of its own unit's first chunk in the (exclusion-filtered, untruncated)
    window, and the non-unit documents holding a chunk that scores strictly better. With no own-unit chunk in the window
    every non-unit chunk counts as ahead. Aggregate keeps no query text."""
    rows = []
    for p, win in zip(pos, windows):
        own = unit_doc.get(p["section"])
        hit = next(((i, c) for i, c in enumerate(win) if c["documentId"] == own), None)
        ahead = [c["documentId"] for c in win if c["documentId"] not in feature_docs and (hit is None or c["score"] < hit[1]["score"])]
        rows.append({"id": p["id"], "rank": hit[0] + 1 if hit else None, "ahead": list(dict.fromkeys(ahead))})
    return {"rows": rows, "n": len(rows), "unit_in_window": sum(r["rank"] is not None for r in rows),
            "outranked": sum(bool(r["ahead"]) for r in rows), "documents": sorted({d for r in rows for d in r["ahead"]})}


def evaluate(pos: list[dict], neg: list[dict], *, floor, unit_doc: dict, feature_docs: set) -> dict:
    """Metrics over already-fetched, already-truncated chunks. `floor=None`: no floor (BM25)."""
    under = (lambda c: True) if floor is None else (lambda c: c["score"] <= floor)
    raw = fl = 0
    rr_sum = 0.0
    by_section: dict[int, bool] = {}
    by_section_raw: dict[int, bool] = {}
    head = [0, 0]
    tail = [0, 0]
    distinct = []  # positive queries only: ML §7 risk 2 is slot crowding where a gotcha is wanted
    for p in pos:
        want = unit_doc.get(p["section"])
        ranks = [i for i, c in enumerate(p["chunks"], 1) if c["documentId"] == want]
        hit_raw = bool(ranks)
        hit_fl = any(c["documentId"] == want and under(c) for c in p["chunks"])
        raw += hit_raw
        fl += hit_fl
        rr_sum += 1 / ranks[0] if ranks else 0.0
        by_section[p["section"]] = by_section.get(p["section"], False) or hit_fl
        by_section_raw[p["section"]] = by_section_raw.get(p["section"], False) or hit_raw
        bucket = tail if p.get("tail") else head
        bucket[0] += 1
        bucket[1] += hit_fl
        distinct.append(len({c["documentId"] for c in p["chunks"]}))
    n_neg = {"off": 0, "near": 0}
    accepts = {"off": 0, "near": 0}
    for n in neg:
        n_neg[n["kind"]] += 1
        accepts[n["kind"]] += any(c["documentId"] in feature_docs and under(c) for c in n["chunks"])
    n = len(pos)
    return {
        "n_pos": n, "raw_hits": raw, "floor_hits": fl,
        "recall_raw": raw / n if n else 0.0, "recall_floor": fl / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
        "distinct_docs_mean": sum(distinct) / len(distinct) if distinct else 0.0,
        "n_sections": len(by_section), "covered_sections": sum(by_section.values()),
        "covered_sections_raw": sum(by_section_raw.values()),
        "n_head": head[0], "n_tail": tail[0],
        "head_recall": head[1] / head[0] if head[0] else None, "tail_recall": tail[1] / tail[0] if tail[0] else None,
        "false_accepts": None if floor is None else accepts, "n_neg": n_neg,
    }


def ac3_verdict(m: dict) -> dict:
    """ML §1 gate, integer arithmetic so the boundaries are exact: recall >= 0.90, every section
    covered, off-domain rejected 100%, near-domain rejected >= 80% (12/15)."""
    near_n, off_n = m["n_neg"]["near"], m["n_neg"]["off"]
    near_rej, off_rej = near_n - m["false_accepts"]["near"], off_n - m["false_accepts"]["off"]
    parts = {
        "recall": m["floor_hits"] * 10 >= m["n_pos"] * 9,
        "coverage": m["covered_sections"] == m["n_sections"],
        "off_domain": off_rej == off_n,
        "near_domain": near_rej * 5 >= near_n * 4,
    }
    return {**parts, "pass": all(parts.values())}


def b2_verdict(m: dict) -> dict:
    """The Set B2 decision rule (docs/plans/python-web-quirks-kb-b2-ml.md section 4), integer arithmetic against the spec's
    constants: floor-applied recall >= 116/126, every one of the 42 sections hit, near false accepts <= 6/30, off 0/30,
    overall <= 6/60. Binary: no indeterminate band."""
    fa = m["false_accepts"]
    parts = {
        "recall": m["floor_hits"] >= B2_MIN_RECALL,
        "coverage": m["covered_sections"] == SET_B_SECTIONS,
        "near_domain": fa["near"] <= B2_MAX_NEAR_FA,
        "off_domain": fa["off"] <= B2_MAX_OFF_FA,
        "overall": fa["near"] + fa["off"] <= B2_MAX_OVERALL_FA,
    }
    return {**parts, "pass": all(parts.values())}


def sweep(pos: list[dict], neg: list[dict], *, unit_doc: dict, feature_docs: set, keep: float) -> list[dict]:
    """One row per floor 0.38-0.50, with ML §6 step 2's adequacy criteria evaluated at that floor."""
    best = []  # best score of each positive's expected document (None when it is not among the chunks)
    for p in pos:
        scores = [c["score"] for c in p["chunks"] if c["documentId"] == unit_doc.get(p["section"])]
        best.append(min(scores) if scores else None)
    rows = []
    for f in SWEEP_FLOORS:
        found = [b for b in best if b is not None and b <= f]  # found at this floor
        worst = max(found) if found else None
        m = evaluate(pos, neg, floor=f, unit_doc=unit_doc, feature_docs=feature_docs)
        fa = m["false_accepts"]
        ok = (
            (m["raw_hits"] - m["floor_hits"]) * 100 <= 3 * m["n_pos"]
            and fa["off"] == 0
            and fa["near"] * 7 <= 2 * m["n_neg"]["near"]
            and (worst is None or round(f - worst, 6) >= 0.025)
        )
        rows.append({"floor": f, "recall_floor": m["recall_floor"], "recall_raw": m["recall_raw"], "near_fa": fa["near"],
                     "off_fa": fa["off"], "adequate": ok, "current": f == keep})
    return rows


# BM25 over the copy's chunks: the offline lexical baseline (ML §4.3).
def bm25_rank(corpus: list[tuple[str, str]], query: str, k1: float = BM25_K1, b: float = BM25_B) -> list[tuple[str, float]]:
    import math

    tok = lambda t: re.findall(r"[a-z0-9_]+", t.lower())
    docs = {k: tok(t) for k, t in corpus}
    n = len(docs)
    avg = sum(len(d) for d in docs.values()) / n if n else 0.0
    df: dict[str, int] = {}
    for d in docs.values():
        for term in set(d):
            df[term] = df.get(term, 0) + 1
    scores: dict[str, float] = {}
    for key, d in docs.items():
        s = 0.0
        for term in sorted(set(tok(query))):
            tf = d.count(term)
            if tf:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg))
        scores[key] = s
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def bm25_top(corpus: list[tuple[str, str]], query: str, k: int = EVAL_TOPK) -> list[dict]:
    """The first `k` chunks with a positive BM25 score, shaped like search results (`key` = `<unit>:<seq>`)."""
    ranked = [(key, sc) for key, sc in bm25_rank(corpus, query) if sc > 0][:k]
    return [{"chunkId": key, "documentId": key.split(":")[0], "seq": 0, "score": sc, "text": ""} for key, sc in ranked]


def fmt_metrics(m: dict) -> str:
    f = lambda x: "n/a" if x is None else f"{x:.4f}"
    fa = m["false_accepts"]
    parts = [f"n_pos={m['n_pos']}", f"raw_hits={m['raw_hits']}", f"floor_hits={m['floor_hits']}",
             f"recall_raw={f(m['recall_raw'])}", f"recall_floor={f(m['recall_floor'])}", f"mrr={f(m['mrr'])}",
             f"distinct_docs_mean={m['distinct_docs_mean']:.2f}",
             f"covered_sections={m['covered_sections']}/{m['n_sections']}",
             f"covered_sections_raw={m['covered_sections_raw']}/{m['n_sections']}",
             f"head_recall={f(m['head_recall'])} tail_recall={f(m['tail_recall'])}"]
    if fa is not None:
        parts.append(f"false_accepts near={fa['near']}/{m['n_neg']['near']} off={fa['off']}/{m['n_neg']['off']}")
    return " ".join(parts)


def _section_maps(manifest: dict, read_source=read_pinned) -> tuple[dict, dict, set]:
    """(section -> documentId, section -> unit id, feature document ids), from the manifest's pin."""
    units_ = build_units(parse_source(read_source(manifest["pin"])))
    doc_of_unit = {u["id"]: u["documentId"] for u in manifest["units"]}
    sec_unit = {s.index: u.id for u in units_ for s in u.members}
    return {n: doc_of_unit.get(uid) for n, uid in sec_unit.items()}, sec_unit, set(doc_of_unit.values())


def eval_gate(fetcher, ws: str, manifest: dict, timeout: float, poll: float, expected_ids=None) -> None:
    ok, table = run_gate(fetcher, ws, manifest, timeout=timeout, poll=poll, expected_ids=expected_ids)
    if not ok:
        raise GateRefused("embedding gate failed; refusing to score:\n" + table)


def run_eval(args, *, searcher, fetcher, target: Target, mpath: Path, env=os.environ, out=print) -> int:
    ws = args.ws
    skill = Path(args.skill).read_text(encoding="utf-8")
    template = extract_prefix(skill)
    excluded = frozenset(args.exclude or [])
    which = args.set if args.set in ("B", "B2") else "A"
    sealed_set = which != "A"

    def set_path() -> Path:
        return resolve_set_b_path(args.set_b2_path if which == "B2" else args.set_b_path, env, kind=which)
    if sealed_set and not (args.probe or args.sealed_once or args.baseline):
        raise KBError(f"--set {which} needs --probe or --sealed-once")
    if args.probe and not sealed_set:
        raise KBError("--probe is a Set B mode")
    if args.sealed_once and not sealed_set:
        raise KBError("--sealed-once is a Set B mode")
    if args.before and args.set != "A":
        raise KBError("--before is a Set A mode")
    chosen = [bool(args.probe), bool(args.sealed_once), bool(args.before), bool(args.ac2), bool(args.sweep), bool(args.baseline)]
    if sum(chosen) > 1 or (not any(chosen) and args.set is None):
        raise KBError("choose one mode: --set A [--before] | --set B --probe|--sealed-once | --ac2 | --sweep | --baseline bm25")
    if args.sealed_once:  # before the sentinel pre-check, the gate and any search: a B edited after freezing must not score
        check_expected_sha256(args.expect_sha256, set_path())
    elif args.expect_sha256 is not None:
        raise KBError("--expect-sha256 belongs to --sealed-once only")
    if not args.probe and not args.baseline and args.floor is None:
        raise KBError("--floor is required (it is recorded in every report)")

    def load_which() -> dict:
        return load_set(set_path() if sealed_set else Path(args.set_a_path))

    if sealed_set:  # shape first: nothing below may spend the seal on a malformed set
        tails = None
        if args.sealed_once:  # the manifest's pin gives the units; probe and baseline run before/without that tie
            pinned = load_manifest(mpath)
            tails = tail_unit_sections(parse_source(read_pinned(pinned["pin"])), _load_chunker(), pinned["migratedOn"])
        validate_set_b(load_which(), tails, which)
    elif not args.before:  # every Set A scoring mode; --before only lists coverage and tallies nothing
        validate_set_a(load_which())

    def fetch_all(items):
        return [searcher.search(ws, build_query(template, it["query"]), EVAL_WINDOW) for it in items]

    # ---- gate-exempt modes
    if args.probe:
        bp = set_path()
        data = load_set(bp)
        out(f"set={which} mode=probe window={EVAL_WINDOW}")
        for it in data["positives"] + data["negatives"]:
            five = scored_chunks(searcher.search(ws, build_query(template, it["query"]), EVAL_WINDOW), excluded)
            named = name_hit(searcher, ws, five)
            if named is None:
                out(f"{it['id']} (no chunks)")
                continue
            did, title, score = named
            out(f"{it['id']} {did} {title or '?'} {score:.4f}" + (" LEAK?" if score < LEAK_SCORE else ""))
        return 0
    if args.before:
        data = load_which()
        out(f"set=A mode=before floor={args.floor:g} window={EVAL_WINDOW}")
        items = data["positives"] + data["negatives"]
        n_hit = 0
        for it, fetched in zip(items, fetch_all(items)):
            five = scored_chunks(fetched, excluded)
            under = [c for c in five if c["score"] <= args.floor]
            if under:
                n_hit += 1
                best = min(under, key=lambda c: c["score"])
                if it in data["positives"]:
                    out(f"DUPLICATE-COVERAGE {it['id']} {best['documentId']} {best['score']:.4f}")
        out(f"queries_with_a_hit_under_floor={n_hit}/{len(items)}")
        return 0

    # ---- scoring modes: sentinel check, then the gate, then the queries
    sentinel = None
    if args.sealed_once:
        sentinel = Path(str(set_path()) + ".scored")
        if sentinel.exists():
            raise SealedRefused(f"Set B already scored ({sentinel}); a second scoring is refused (write B2 instead)")
    manifest = load_manifest(mpath)
    eval_gate(fetcher, ws, manifest, args.timeout, args.poll, expected_unit_ids(args.copy))
    unit_doc, sec_unit, feature_docs = _section_maps(manifest)
    header = f"set={which} floor={args.floor if args.floor is None else format(args.floor, 'g')} window={EVAL_WINDOW} topk={EVAL_TOPK}"

    if args.ac2:
        return _ac2(args, searcher, ws, template, excluded, manifest, unit_doc, sec_unit, out, header)
    data = load_which()
    pos, neg = data["positives"], data["negatives"]

    if args.baseline:
        if args.baseline != "bm25":
            raise KBError("--baseline supports only bm25")
        ents = copy_entries(Path(args.copy).read_text(encoding="utf-8"))
        split = _load_chunker()
        corpus = [(f"{e.id}:{i:03d}", c) for e in ents for i, c in enumerate(split(e.text))]
        by_unit = {n: uid for n, uid in sec_unit.items()}

        pr = [{**p, "chunks": bm25_top(corpus, p["query"])} for p in pos]
        nr = [{**n_, "chunks": bm25_top(corpus, n_["query"])} for n_ in neg]
        m = evaluate(pr, nr, floor=None, unit_doc=by_unit, feature_docs=set(by_unit.values()))
        out(f"baseline=bm25 set={which} chunks={len(corpus)} topk={EVAL_TOPK}")
        out("metrics: " + fmt_metrics(m))
        return 0

    fetched_pos, fetched_neg = fetch_all(pos), fetch_all(neg)

    def build(exclude):
        return ([{**p, "chunks": scored_chunks(f, exclude)} for p, f in zip(pos, fetched_pos)],
                [{**n_, "chunks": scored_chunks(f, exclude)} for n_, f in zip(neg, fetched_neg)])

    pr, nr = build(excluded)
    if args.sweep:
        out(header)
        if excluded:
            out("excluded: " + " ".join(sorted(excluded)))

        def sweep_rows(prefix, p_, n_):
            for r in sweep(p_, n_, unit_doc=unit_doc, feature_docs=feature_docs, keep=args.floor):
                out(f"{prefix}floor {r['floor']:g} recall_floor={r['recall_floor']:g} recall_raw={r['recall_raw']:g} "
                    f"near_fa={r['near_fa']} off_fa={r['off_fa']} adequate={r['adequate']}"
                    + (" <- current" if r["current"] else ""))
        sweep_rows("", pr, nr)
        if excluded:  # ML 4.2 / D11: the filtered headline sits beside the unfiltered numbers
            sweep_rows("unfiltered ", *build(frozenset()))
        return 0
    m = evaluate(pr, nr, floor=args.floor, unit_doc=unit_doc, feature_docs=feature_docs)
    if sentinel is not None:  # burnt only now: results exist, and nothing has been shown yet
        try:
            fd = os.open(sentinel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as e:
            raise SealedRefused(f"Set B already scored ({sentinel})") from e
        with os.fdopen(fd, "w") as fh:
            json.dump({"floor": args.floor, "setSha256": hashlib.sha256(set_path().read_bytes()).hexdigest(),
                       "scoredAt": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh)
    out(header)
    if excluded:
        out("excluded: " + " ".join(sorted(excluded)))
        out(f"note: limit=5 deliberately exceeded for this measurement (window={EVAL_WINDOW} chunks requested, "
            "excluded documents' chunks dropped, the first five scored)")
        ur, un = build(frozenset())
        out("filtered: " + fmt_metrics(m))
        out("unfiltered: " + fmt_metrics(evaluate(ur, un, floor=args.floor, unit_doc=unit_doc, feature_docs=feature_docs)))
    else:
        out("metrics: " + fmt_metrics(m))
    if which == "B2":  # B2's own rule replaces B's ac3 gate; the pooled figure is informational only
        out("b2: " + " ".join(f"{k}={v}" for k, v in b2_verdict(m).items()))
        hits, n = SET_B_RECORDED
        out(f"pooled with Set B (informational, not part of the verdict): ({hits}+{m['floor_hits']})/{n + m['n_pos']} = "
            f"{(hits + m['floor_hits']) / (n + m['n_pos']):.4f}")
    else:
        out("ac3: " + " ".join(f"{k}={v}" for k, v in ac3_verdict(m).items()))
    diag = displacement(pos, [scored_chunks(f, excluded, k=len(f)) for f in fetched_pos], unit_doc=unit_doc, feature_docs=feature_docs)
    out(f"displacement: positives={diag['n']} unit_in_window={diag['unit_in_window']} "
        f"non_unit_outranks_unit={diag['outranked']} documents={','.join(diag['documents']) or '-'}")
    if which == "A":  # per-query lines only where failures may be inspected; Set B stays aggregate-only
        for r in diag["rows"]:
            out(f"RANK {r['id']} {r['rank'] or '-'} ahead={','.join(r['ahead']) or '-'}")
    if which == "A":  # Set A is the only set whose failures may be inspected
        for p in pr:
            if not any(c["documentId"] == unit_doc.get(p["section"]) and c["score"] <= args.floor for c in p["chunks"]):
                named = name_hit(searcher, ws, p["chunks"])
                out(f"MISS {p['id']} " + ("(no chunks)" if named is None else f"{named[0]} {named[1] or '?'} {named[2]:.4f}"))
        for n_ in nr:
            bad = [c for c in n_["chunks"] if c["documentId"] in feature_docs and c["score"] <= args.floor]
            if bad:
                out(f"FALSE-ACCEPT {n_['id']} ({n_['kind']}) {bad[0]['documentId']} {bad[0]['score']:.4f}")
    return 0


def _ac2(args, searcher, ws, template, excluded, manifest, unit_doc, sec_unit, out, header) -> int:
    ents = copy_entries(Path(args.copy).read_text(encoding="utf-8"))
    doc_of_unit = {u["id"]: u["documentId"] for u in manifest["units"]}
    a_pos = load_set(Path(args.set_a_path))["positives"]
    out(header)

    def returns(query, doc):
        five = window_chunks(searcher, ws, build_query(template, query), excluded)
        return any(c["documentId"] == doc and c["score"] <= args.floor for c in five)

    passed = 0
    for e in ents:
        doc = doc_of_unit.get(e.id)
        sit = [p["query"] for p in a_pos if sec_unit.get(p["section"]) == e.id]
        res = {
            "title": returns(e.title, doc),
            "situation": bool(sit) and all(returns(q, doc) for q in sit),
            "tail": returns(e.text.rstrip().split("\n\n")[-1], doc),
        }
        got = searcher.get_document(ws, doc)
        res["name"] = bool(got) and got.get("title") == e.title
        ok = all(res.values())
        passed += ok
        out(f"{e.id} title={'ok' if res['title'] else 'MISS'} situation={'ok' if res['situation'] else 'MISS'} "
            f"tail={'ok' if res['tail'] else 'MISS'} name={'ok' if res['name'] else 'WRONG'}")
    out(f"ac2: {passed}/{len(ents)} units pass")
    return 0 if passed == len(ents) else 1


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
def _kb_main(args, transport, fetcher, targets, searcher=None) -> int:
    copy_path = Path(args.copy)
    if args.cmd in ("ingest", "replace", "retire"):
        text = copy_path.read_text(encoding="utf-8")
        kw = dict(ws=args.ws, base_url=args.base_url, manifest_path=args.manifest, transport=transport, fetcher=fetcher,
                  confirm_production=args.confirm_production, targets=targets)
        if args.cmd == "ingest":
            results = ingest(text, pace_timeout=args.timeout, poll=args.poll, **kw)
            for r in results:
                print(f"{r.id} {r.action} {r.documentId} {r.note}".rstrip())
            return 1 if any(r.action == "refuse" or r.state for r in results) else 0
        if args.cmd == "replace":
            r = replace(text, unit=args.unit, pace_timeout=args.timeout, poll=args.poll, **kw)
            print(f"replace {args.unit}: " + (r.note if r.state else "done"))
            return 1 if r.state else 0
        retire(text, unit=args.unit, **kw)
        print(f"retire {args.unit}: done")
        return 0
    target = check_target(args.ws, args.base_url or targets.get(args.ws, ""), targets, confirm_production=True)
    mpath = _manifest_path(target, args.manifest)
    fetcher = fetcher or CypherFetcher()
    if args.cmd == "eval":
        searcher = searcher or RestSearcher(target, transport)
        return run_eval(args, searcher=searcher, fetcher=fetcher, target=target, mpath=mpath)
    manifest = load_manifest(mpath)
    if args.cmd == "gate":
        ok, table = run_gate(fetcher, args.ws, manifest, timeout=args.timeout, poll=args.poll,
                             expected_ids=expected_unit_ids(copy_path),
                             expected_chunks=expected_chunk_counts(copy_path.read_text(encoding="utf-8")))
        print(table, end="")
        print("gate: " + ("clean" if ok else "FAILED"))
        return 0 if ok else 1
    found = diverge(copy_path.read_text(encoding="utf-8"), ws=args.ws, manifest=manifest, fetcher=fetcher)
    for kind, uid, title in found:
        print(f"{kind} {uid} {title}")
    print("diverge: " + ("clean" if not found else f"{len(found)} finding(s)"))
    return 1 if found else 0


def main(argv: list[str] | None = None, *, transport=None, fetcher=None, targets: dict | None = None, searcher=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--pin", required=True, help="explicit commit sha of the source (never a ref)")
        p.add_argument("--repo", default=str(REPO_ROOT), help="git repository holding the pin")

    def kbside(p, base_required, unit=False, flag=False):
        p.add_argument("--copy", default=str(DEFAULT_COPY), help="the copy of record")
        p.add_argument("--ws", required=True, help="workspace name; with --base-url must be a row of the target table")
        p.add_argument("--base-url", required=base_required)
        p.add_argument("--manifest", help="manifest path (default only for production)")
        if unit:
            p.add_argument("--unit", required=True)
        if flag:
            p.add_argument("--confirm-production", action="store_true")

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
    for name, help_, unit in (("ingest", "idempotent, paced ingest of every copy unit", False),
                              ("replace", "TEST target only: delete then re-post one unit (paced)", True)):
        p = sub.add_parser(name, help=help_)
        kbside(p, True, unit=unit, flag=True)
        p.add_argument("--timeout", type=float, default=GATE_TIMEOUT, help="seconds to wait per document to leave processing")
        p.add_argument("--poll", type=float, default=GATE_POLL)
    kbside(sub.add_parser("retire", help="TEST target only: delete one unit's document"), True, unit=True, flag=True)
    p = sub.add_parser("gate", help="K-102 embedding gate over the manifest's documents")
    kbside(p, False)
    p.add_argument("--timeout", type=float, default=GATE_TIMEOUT)
    p.add_argument("--poll", type=float, default=GATE_POLL)
    kbside(sub.add_parser("diverge", help="compare the KB with the copy (read-only)"), False)
    p = sub.add_parser("eval", help="evaluation harness (D9/D11)")
    kbside(p, True)
    p.add_argument("--set", choices=["A", "B", "B2"])
    p.add_argument("--before", action="store_true", help="Set A before ingestion (gate-exempt)")
    p.add_argument("--probe", action="store_true", help="Set B: rank-1 source and score only (gate-exempt, no sentinel)")
    p.add_argument("--sealed-once", action="store_true", help="Set B: the one scoring; writes the sentinel")
    p.add_argument("--expect-sha256", help="--sealed-once only (required there): the sha256 of Set B recorded at freeze")
    p.add_argument("--ac2", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--baseline", choices=["bm25"])
    p.add_argument("--floor", type=float, help="score floor, recorded in every report (required except --probe)")
    p.add_argument("--exclude", action="append", help="documentId whose chunks are dropped from the window")
    p.add_argument("--set-a-path", default=str(DEFAULT_SET_A))
    p.add_argument("--set-b-path", help="Set B file (or PWQ_SET_B); must be outside the repo")
    p.add_argument("--set-b2-path", help="Set B2 file (or PWQ_SET_B2); must be outside the repo")
    p.add_argument("--skill", default=str(DEFAULT_SKILL), help="retrieval skill holding the fenced prefix literal")
    p.add_argument("--timeout", type=float, default=GATE_TIMEOUT)
    p.add_argument("--poll", type=float, default=GATE_POLL)
    args = ap.parse_args(argv)
    targets = TARGETS if targets is None else targets

    try:
        if args.cmd in ("ingest", "replace", "retire", "gate", "diverge", "eval"):
            return _kb_main(args, transport or urllib_transport, fetcher, targets, searcher)
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
    except TargetRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    except GateRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 4
    except SealedRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 5
    except KBError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
