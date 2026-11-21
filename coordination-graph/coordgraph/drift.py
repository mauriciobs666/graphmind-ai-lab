"""The read-time staleness check and the sync header (plan section 3.3).

`check` labels, never repairs: each covered file is compared with its `Source.sha256` (*changed*,
*unsynced*, *gone*) and each changed file is re-parsed and diffed against the stored children by row key
(*changed* / *added in file* / *removed from file*). `render_header` is the block every read subcommand
prints first. Imports no `wilib`: the registry line is handed in by the caller that loaded `identity`.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import store
from .parse.backlog import BacklogParseError
from .parse.source import PARSER_VERSION
from .sync import Parser, canon_hash, child_key, sha_of

CHILD_QUERY = "MATCH (s:Source {path: $path})-[]->(c) RETURN properties(c)"
WIN = "→ file wins; run coordgraph sync"


@dataclass
class SourceDrift:
    path: str
    state: str  # changed | unsynced | gone | stale-parser
    rows: list[tuple[str, str]] = field(default_factory=list)  # (child key, changed|added in file|removed from file)
    note: str = ""


@dataclass
class DriftReport:
    source_count: int  # sources in the graph the check covered
    oldest_ms: int | None
    newest_ms: int | None
    drifts: list[SourceDrift]

    @property
    def clean(self) -> bool:
        return not self.drifts

    def flagged_rows(self) -> set[tuple[str, str]]:
        """`(path, child key)` of every row the body should mark `⚠ drift`."""
        return {(d.path, k) for d in self.drifts for k, _ in d.rows}


def _stamp(ms: int | None) -> str:
    if ms is None:
        return "never"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check(graph, parser: Parser, paths: Iterable[str] | None = None) -> DriftReport:
    """Drift over `paths` (default: every covered file and every stored source)."""
    want = None if paths is None else set(paths)
    rows = graph.ro_query("MATCH (s:Source) RETURN s.path, s.sha256, s.syncedAt, s.parserVersion").result_set
    stored = {r[0]: (r[1], r[2], r[3]) for r in rows if want is None or r[0] in want}
    candidates = {p for p in parser.found if want is None or p in want}
    drifts: list[SourceDrift] = []
    for path in sorted(candidates | set(stored)):
        file = parser.root / path
        on_disk = path in parser.found and file.is_file()
        if path not in stored:
            drifts.append(SourceDrift(path, "unsynced"))
        elif not on_disk:
            drifts.append(SourceDrift(path, "gone"))
        elif sha_of(file) != stored[path][0]:
            drifts.append(_changed(graph, parser, path))
        elif stored[path][2] != PARSER_VERSION:  # same bytes, but `sync --all` would rewrite the source
            drifts.append(SourceDrift(path, "stale-parser",
                                      note=f"parserVersion {stored[path][2]}, current {PARSER_VERSION}"))
    times = [t for _, t, _ in stored.values() if isinstance(t, int)]
    return DriftReport(len(stored), min(times, default=None), max(times, default=None), drifts)


def _changed(graph, parser: Parser, path: str) -> SourceDrift:
    try:
        kind, meta, rows = parser.parse(path)
        _, now = store.write_form(kind, meta, rows)
    except (BacklogParseError, store.StoreError) as exc:
        return SourceDrift(path, "changed", note=f"does not parse: {exc}")
    have = {child_key(p): canon_hash(p) for (p,) in graph.ro_query(CHILD_QUERY, {"path": path}).result_set}
    new = {child_key(p): canon_hash(p) for p in now}
    out = [(k, "changed") for k in new if k in have and new[k] != have[k]]
    out += [(k, "added in file") for k in new if k not in have]
    out += [(k, "removed from file") for k in have if k not in new]
    return SourceDrift(path, "changed", sorted(out))


def render_header(report: DriftReport, hook_state: str, registry_line: str | None = None) -> str:
    lines = [f"Synced: {_stamp(report.oldest_ms)} – {_stamp(report.newest_ms)} ({report.source_count} sources)"
             f" · Hook: {hook_state}"]
    if registry_line:
        lines.append(registry_line)
    if report.clean:
        lines.append("Drift: none")
        return "\n".join(lines)
    lines.append(f"Drift: {len(report.drifts)} source(s)")
    for d in report.drifts:
        lines.append(f"  {d.state}\t{d.path}\t{WIN}" + (f"\t{d.note}" if d.note else ""))
        lines += [f"    ⚠ drift\t{key}\t{what}" for key, what in d.rows]
    return "\n".join(lines)
