"""The prompt-bearing Markdown glob grammar: validator (``check_block``), ``Globs`` and the matcher (C-1001 P1-03a).

Lives under ``ccg`` so that ``ccg.manifest`` can validate a system manifest's ``promptBearingMarkdown`` block without
importing ``extractors`` (only the orchestration modules may); ``extractors.markdown.globs`` imports from here.
A failure is a ``ManifestError`` that never carries a glob's text.

Matching is by hand, not by regex: ``*`` and ``?`` act inside one path segment, a whole ``**`` segment matches zero or
more segments, and both matchers are polynomial in the (capped) pattern and path sizes, so a hostile path cannot cause
backtracking blow-up.
"""
from __future__ import annotations

from dataclasses import dataclass
KLASSES = ("prompt-md", "doc")

_MAX_PATTERNS = 64
_MAX_PATTERN_LEN = 200
_MAX_SEGMENTS = 64
_MAX_SEGMENT_LEN = 255
_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-*?/")


class ManifestError(ValueError):
    """The manifest is absent or its ``promptBearingMarkdown`` block is not the expected shape."""


def _check_patterns(name: str, value: object, *, non_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (non_empty and not value) or len(value) > _MAX_PATTERNS:
        raise ManifestError(f"promptBearingMarkdown.{name} must be a list of 1..{_MAX_PATTERNS} globs"
                            if non_empty else f"promptBearingMarkdown.{name} must be a list of <= {_MAX_PATTERNS} globs")
    out = []
    for i, p in enumerate(value):
        if (not isinstance(p, str) or not p or len(p) > _MAX_PATTERN_LEN or not set(p) <= _ALPHABET
                or any(s in ("", ".", "..") for s in p.split("/"))
                or any("**" in s and s != "**" for s in p.split("/"))):
            raise ManifestError(f"promptBearingMarkdown.{name}[{i}]: unsupported glob")
        out.append(p)
    return tuple(out)


def check_block(block: object) -> "Globs":
    """Validate a ``promptBearingMarkdown`` value (exactly ``include``, 1..64 globs, and ``except``, 0..64 globs; a
    manifest may omit ``except``) and return its ``Globs``. ``ManifestError`` otherwise, value-free."""
    if not isinstance(block, dict) or "include" not in block or not set(block) <= {"include", "except"}:
        raise ManifestError("promptBearingMarkdown must be a mapping with the key include and optionally except")
    return Globs(include=_check_patterns("include", block["include"], non_empty=True),
                 exclude=_check_patterns("except", block.get("except", []), non_empty=False))


@dataclass(frozen=True)
class Globs:
    include: tuple[str, ...]
    exclude: tuple[str, ...]

    def classify(self, path: str) -> str:
        """``prompt-md`` or ``doc`` for a repository-relative POSIX path of a Markdown file."""
        if any(_match(p, path) for p in self.include) and not any(_match(p, path) for p in self.exclude):
            return "prompt-md"
        return "doc"


def _seg_match(pat: str, text: str) -> bool:
    """``*`` (any run) and ``?`` (one char) over one segment; the greedy two-pointer form, O(len(pat) * len(text))."""
    p = t = 0
    star = mark = -1
    while t < len(text):
        if p < len(pat) and pat[p] in ("?", text[t]) and pat[p] != "*":
            p += 1
            t += 1
        elif p < len(pat) and pat[p] == "*":
            star, mark = p, t
            p += 1
        elif star != -1:
            p = star + 1
            mark += 1
            t = mark
        else:
            return False
    while p < len(pat) and pat[p] == "*":
        p += 1
    return p == len(pat)


def _match(pattern: str, path: str) -> bool:
    segs = path.split("/")
    pats = pattern.split("/")
    if len(segs) > _MAX_SEGMENTS or any(len(s) > _MAX_SEGMENT_LEN for s in segs):
        return False
    # memoised over (pattern index, path index): ``**`` consumes zero or more whole segments
    seen: dict[tuple[int, int], bool] = {}

    def go(i: int, j: int) -> bool:
        key = (i, j)
        if key in seen:
            return seen[key]
        if i == len(pats):
            r = j == len(segs)
        elif pats[i] == "**":
            r = go(i + 1, j) or (j < len(segs) and go(i, j + 1))
        else:
            r = j < len(segs) and _seg_match(pats[i], segs[j]) and go(i + 1, j + 1)
        seen[key] = r
        return r

    return go(0, 0)


