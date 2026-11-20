"""History files: delivery-shaped headings and ID tokens found by shape (plan section 3.7
step 2; plan3 sections 3.3, 3.6).

History is never mirrored (FR-1); it is read from disk. A heading is *delivery-shaped* when
an ID (optionally followed by `/`-joined sibling tokens) is immediately followed by
`closed`/`delivered`/`fixed`/`shipped`/`done`, or when it carries both `✅` and the ID. A comma
list (`K-070, K-071 delivered`) does not match: only the token run that directly precedes the
verb counts, and a run that follows a comma is a list member, not the subject. ID tokens are
*found* by shape (`coordgraph.shapes`), stored as written, and compared through a candidate
function so the registry-backed one of S1d plugs in without this module knowing.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .scope import scope_of
from .shapes import Candidates, classify, offline_candidates

VERBS = ("closed", "delivered", "fixed", "shipped", "done")
_HEADING = re.compile(r"^#{2,4}\s")
_TOKEN = r"[A-Za-z0-9][A-Za-z0-9._:-]*"
_RUN = re.compile(
    rf"(?<![A-Za-z0-9._:/-])(?P<run>{_TOKEN}(?:/{_TOKEN})*)\s+(?:{'|'.join(VERBS)})\b"
)
_TOKEN_RE = re.compile(rf"(?<![A-Za-z0-9._:/-]){_TOKEN}")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class DeliveryHeading:
    line: int  # 1-based
    text: str
    tokens: tuple[str, ...]  # ID-shaped tokens as written


def find_id_tokens(text: str) -> list[str]:
    """ID-shaped tokens (`shapes.classify`) of a text, as written, in order."""
    out = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).rstrip(".:")
        if classify(tok):
            out.append(tok)
    return out


def _run_tokens(run: str) -> list[str]:
    return [t.rstrip(".:") for t in run.split("/") if classify(t.rstrip(".:"))]


def delivery_headings(text: str) -> list[DeliveryHeading]:
    """Every delivery-shaped heading of a history file, with the ID tokens it delivers."""
    out: list[DeliveryHeading] = []
    in_fence = False
    for n, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not _HEADING.match(line):
            continue
        tokens: list[str] = []
        for m in _RUN.finditer(line):
            if line[: m.start()].rstrip().endswith(","):
                continue  # a list member, not a subject (P3-n2)
            tokens.extend(_run_tokens(m.group("run")))
        if "✅" in line:
            tokens.extend(find_id_tokens(line))
        if tokens:
            out.append(DeliveryHeading(n, line.strip(), tuple(dict.fromkeys(tokens))))
    return out


def mentioned_tokens(text: str) -> list[str]:
    """Distinct work-item ID tokens mentioned anywhere in a history file (no FR/AC citations)."""
    seen: dict[str, None] = {}
    for tok in find_id_tokens(text):
        if classify(tok) == "scheme" and re.search(r"-(FR|AC)\d", tok):
            continue
        seen.setdefault(tok, None)
    return list(seen)


def history_path_for(backlog_path: str) -> str | None:
    """The history file that owns a backlog's deliveries."""
    p = PurePosixPath(backlog_path)
    if p.name == "BACKLOG.md" and p.parent.name == "docs":
        return str(p.with_name("HISTORY.md"))
    if p.name == "plan.md" and p.parent.name == "kaizen":
        return str(p.with_name("history.md"))
    return None


def backlog_path_for(history_path: str) -> str | None:
    p = PurePosixPath(history_path)
    if p.name == "HISTORY.md" and p.parent.name == "docs":
        return str(p.with_name("BACKLOG.md"))
    if p.name == "history.md" and p.parent.name == "kaizen":
        return str(p.with_name("plan.md"))
    return None


@dataclass
class HistoryIndex:
    """Delivery headings of every history file, keyed by the scope that owns the file."""

    by_scope: dict[str, list[tuple[str, list[DeliveryHeading]]]] = field(default_factory=dict)

    @classmethod
    def from_repo(cls, root: str | Path, backlog_paths: Iterable[str],
                  legacy_paths: Mapping[str, str]) -> HistoryIndex:
        idx = cls()
        for bpath in backlog_paths:
            hpath = history_path_for(bpath)
            scope = scope_of(bpath, legacy_paths)
            if not hpath or scope is None or not (Path(root) / hpath).is_file():
                continue
            text = (Path(root) / hpath).read_text(encoding="utf-8", errors="replace")
            idx.by_scope.setdefault(scope, []).append((hpath, delivery_headings(text)))
        return idx

    def delivered(self, token: str, referrer_scope: str,
                  candidates: Candidates = offline_candidates) -> tuple[str, DeliveryHeading] | None:
        """The delivery heading that satisfies a dependency token, as `(history path, heading)`.

        A scheme ref searches every history file; a legacy ref only its owning backlog's
        (`<home>:K-1` names it, a bare one is the referrer's). A heading delivers the ref when
        any of its tokens' candidates intersect the dependency's.
        """
        kind = classify(token)
        if kind == "legacy-qualified":
            scopes = [token.split(":", 1)[0]]
        elif kind == "legacy-bare":
            scopes = [referrer_scope]
        elif kind == "scheme":
            scopes = list(self.by_scope)
        else:
            return None
        want = candidates(token, referrer_scope)
        for scope in scopes:
            for hpath, headings in self.by_scope.get(scope, []):
                for h in headings:
                    if any(want & candidates(t, scope) for t in h.tokens):
                        return hpath, h
        return None


@dataclass(frozen=True)
class Coverage:
    path: str
    mentioned: int  # distinct ID tokens mentioned in the file
    still_open: int  # of those, open in some backlog (not deliveries)
    matched: list[str]
    unmatched: list[str]


def coverage(root: str | Path, backlog_paths: Iterable[str], legacy_paths: Mapping[str, str],
             candidates: Candidates = offline_candidates) -> list[Coverage]:
    """The coverage probe (plan section 3.7 step 2, P2-M4): per history file, each ID it mentions
    that is open in no backlog, split into matched / unmatched by a delivery-shaped heading."""
    from .parse.backlog import parse_backlog  # noqa: PLC0415 - keeps this module's import light

    root = Path(root)
    backlog_paths = list(backlog_paths)
    open_refs: set[str] = set()
    for bpath in backlog_paths:
        for item in parse_backlog(bpath, (root / bpath).read_bytes(), legacy_paths).items:
            open_refs.add(item.ref)
    index = HistoryIndex.from_repo(root, backlog_paths, legacy_paths)
    rows: list[Coverage] = []
    for bpath in backlog_paths:
        hpath, scope = history_path_for(bpath), scope_of(bpath, legacy_paths)
        if not hpath or scope is None or not (root / hpath).is_file():
            continue
        text = (root / hpath).read_text(encoding="utf-8", errors="replace")
        matched: list[str] = []
        unmatched: list[str] = []
        still_open = 0
        tokens = mentioned_tokens(text)
        for tok in tokens:
            if candidates(tok, scope) & open_refs:
                still_open += 1
            elif index.delivered(tok, scope, candidates) is not None:
                matched.append(tok)
            else:
                unmatched.append(tok)
        rows.append(Coverage(hpath, len(tokens), still_open, matched, unmatched))
    return rows
