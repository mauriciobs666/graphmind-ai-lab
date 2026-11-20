"""Review documents: header status plus the latest verdict (plan section 2.1; graph note 2.2).

A review's verdict is spelled many ways (`**Verdict: approve.**`, `Verdict (Pass 1): needs
changes`, `## Verdict: **needs changes**`, `NEEDS CHANGES`, a `## Verdict` heading with the
verdict on the next line). The latest verdict is the one in the last `## Pass N` section; a
review without pass sections has its single verdict (the highest `(Pass N)`-marked one when
it carries several). `verdictClass` is the leading verdict phrase's normalisation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .source import SourceDoc, base_fields

UNKNOWN = "unknown"
CLASSES = ("needs-changes", "approve", "approve-with-suggestions", "reject", UNKNOWN)
_PASS_HEADING = re.compile(r"^##\s+Pass\s+(\d+)\b", re.MULTILINE)
_NEEDS = r"needs?[\s-]+(?:(?:one|two|three|a|more|further|some)[\s-]+)?changes?"
_MARK = r"[*_`]*"
_VERDICT = re.compile(
    r"\b(?:updated\s+|overall\s+|current\s+)?verdicts?\b" + _MARK
    + r"\s*(?:\(\s*pass\s*(?P<mark>\d+)[^)]*\))?\s*" + _MARK + r"\s*[:\u2014-]?\s*" + _MARK
    + r"\s*(?P<v>" + _NEEDS + r"|approved?(?:\s+with\s+suggestions?)?|reject(?:ed)?|pass(?:es)?\b)",
    re.IGNORECASE,
)
_PHRASES = (
    ("needs-changes", re.compile(_NEEDS + r"|changes?\s+(?:are\s+)?(?:needed|required)")),
    ("approve-with-suggestions", re.compile(r"approved?\s+with\s+suggestions?")),
    ("approve", re.compile(r"approved?\b|\bpass(?:es|ed)?\b(?!\s+with)|\blgtm\b")),
    ("reject", re.compile(r"reject(?:ed)?\b")),
)


def classify_verdict(text: str) -> str:
    """The class of the earliest verdict phrase in `text`, else `unknown`."""
    plain = re.sub(r"[*_`]", "", text).casefold()
    best: tuple[int, str] | None = None
    for cls, pattern in _PHRASES:
        m = pattern.search(plain)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), cls)
    return best[1] if best else UNKNOWN


def _verdict_in(section: str) -> tuple[str, int] | None:
    """`(verdict text, pass marker)` of the best verdict statement in `section`.

    A statement is the word `verdict` followed (markup, whitespace including line breaks and an
    optional `(Pass N)` aside in between) by a verdict phrase, so `## Verdict` over a phrase
    on the next line counts too. Marker 0 when unmarked. The text kept is the phrase's
    sentence, up to 300 characters.
    """
    found: list[tuple[str, int]] = []
    for m in _VERDICT.finditer(section):
        end = section.find("\n", m.end())
        text = section[m.start("v"):end if end != -1 else len(section)].strip(" *_`")
        found.append((text[:300], int(m.group("mark")) if m.group("mark") else 0))
    if not found:
        return None
    marked = [f for f in found if f[1]]
    return max(marked, key=lambda f: f[1]) if marked else found[0]


@dataclass
class ReviewDoc(SourceDoc):
    latest_verdict: str | None = None
    latest_pass: int = 1
    verdict_class: str = UNKNOWN

    def source_props(self) -> dict:
        props = super().source_props()
        if self.latest_verdict is not None:
            props["latestVerdict"] = self.latest_verdict
        props["latestPass"] = self.latest_pass
        props["verdictClass"] = self.verdict_class
        props["warnings"] = list(self.warnings)
        return props


def parse_review(path: str, raw: bytes) -> ReviewDoc:
    text, base = base_fields(path, "review", raw)
    doc = ReviewDoc(**base)
    passes = list(_PASS_HEADING.finditer(text))
    if passes:
        last = passes[-1]
        doc.latest_pass = int(last.group(1))
        found = _verdict_in(text[last.end():])
    else:
        found = _verdict_in(text)
        if found and found[1]:
            doc.latest_pass = found[1]
    if found is None:
        doc.warnings.append("no verdict found")
        return doc
    doc.latest_verdict = found[0]
    doc.verdict_class = classify_verdict(found[0])
    return doc
