"""KB files: `##` sections, split modes, claim hashes (plan section 3.5; graph note section 2.4).

A KB file is mirrored into `ws:agent-team` one document per `##` heading (mode `whole`) or,
for a heading carrying `<!-- kb-split: bullets -->` (on the heading line or alone right
under it), one document per top-level bullet. This module only parses: it yields each
section's hash and its claims; the REST writes and the sync state live in `kbsync.py` (S6).

`claim_hash` is `sha256(normalize_text(claim text))` with falkor-chat's own normaliser
(whitespace-collapse, strip, `casefold`; `falkorchat/update_detection.py`), directly
comparable to `Document.textNormalizedHash`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..keys import safe_key
from .source import decode

SPLIT_MARKER = re.compile(r"<!--\s*kb-split:\s*(\w+)\s*-->")
_H2 = re.compile(r"^##\s+(?P<h>.*?)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_BULLET = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_WS = re.compile(r"\s+")
MODES = ("whole", "bullets")


def normalize_text(text: str) -> str:
    """falkor-chat's normaliser, replicated (graph note section 2.4)."""
    return _WS.sub(" ", text).strip().casefold()


def claim_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Claim:
    text: str
    title: str  # the heading (whole) or a short bullet lead (bullets); S6 composes the final title
    claim_hash: str
    key: str  # `whole`, or `b-<12 hex of the bullet hash>`


@dataclass
class Section:
    heading: str
    heading_key: str
    ord: int
    mode: str
    body: str
    section_hash: str
    claims: list[Claim] = field(default_factory=list)


@dataclass
class KbDoc:
    path: str
    sha256: str
    sections: list[Section] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _slug(heading: str) -> str:
    out = re.sub(r"[^\w\s-]", "", heading.casefold())
    return re.sub(r"[\s_]+", "-", out).strip("-") or "section"


def _split_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _H2.match(line)
        if m:
            sections.append((m.group("h"), []))
        elif sections:
            sections[-1][1].append(line)
    return sections


def _bullets(lines: list[str]) -> tuple[list[str], int]:
    """`(bullet texts, count of non-blank lines belonging to no bullet)`."""
    bullets: list[list[str]] = []
    stray = 0
    in_fence = False
    for line in lines:
        if _FENCE.match(line.lstrip()):
            in_fence = not in_fence
        if not in_fence and _BULLET.match(line):
            bullets.append([line])
        elif bullets and (not line.strip() or line[:1].isspace() or in_fence):
            bullets[-1].append(line)
        elif line.strip():
            stray += 1
    return ["\n".join(b).strip() for b in bullets], stray


def parse_kb(path: str, raw: bytes) -> KbDoc:
    text = decode(raw)
    doc = KbDoc(path=path, sha256=hashlib.sha256(raw).hexdigest())
    seen: dict[str, int] = {}
    for ord_, (heading, lines) in enumerate(_split_sections(text)):
        mode = "whole"
        marker = SPLIT_MARKER.search(heading)
        if marker:
            heading = SPLIT_MARKER.sub("", heading).strip()
        else:
            first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
            if first is not None and SPLIT_MARKER.fullmatch(lines[first].strip()):
                marker = SPLIT_MARKER.fullmatch(lines[first].strip())
                lines = lines[:first] + lines[first + 1:]
        if marker:
            if marker.group(1) in MODES:
                mode = marker.group(1)
            else:
                doc.warnings.append(f"## {heading}: unknown kb-split mode {marker.group(1)!r}")
        body = "\n".join(lines).strip()
        base_key = _slug(heading)
        n = seen.get(base_key, 0) + 1
        seen[base_key] = n
        if n > 1:
            doc.warnings.append(f"duplicate heading {heading!r} → key suffix ~{n}")
        section = Section(
            heading=heading,
            heading_key=safe_key(base_key if n == 1 else f"{base_key}~{n}"),
            ord=ord_, mode=mode, body=body,
            section_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        )
        if mode == "whole":
            if body:
                section.claims.append(Claim(body, heading, claim_hash(body), "whole"))
            else:
                doc.warnings.append(f"## {heading}: empty section")
        else:
            texts, stray = _bullets(lines)
            if stray:
                doc.warnings.append(
                    f"## {heading}: {stray} non-bullet line(s) in a bullets section are not mirrored"
                )
            for t in texts:
                h = claim_hash(t)
                lead = _BULLET.sub("", t.splitlines()[0]).strip()
                section.claims.append(Claim(t, lead[:80], h, f"b-{h[:12]}"))
        doc.sections.append(section)
    return doc


def duplicate_claims(docs: list[KbDoc]) -> dict[str, list[tuple[str, str]]]:
    """`claim_hash -> [(path, heading), ...]` for every claim text two sections share
    (falkor-chat would auto-supersede one with the other; `kb-sync` refuses to ingest)."""
    seen: dict[str, list[tuple[str, str]]] = {}
    for doc in docs:
        for s in doc.sections:
            for c in s.claims:
                seen.setdefault(c.claim_hash, []).append((doc.path, s.heading))
    return {h: where for h, where in seen.items() if len(where) > 1}
