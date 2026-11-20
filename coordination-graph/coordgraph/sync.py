"""`sync`, `rebuild` and the parity proof (plan section 3.1/3.2, graph note v4 sections 5.1-5.3).

A source's subgraph is a pure function of its file, so a sync is: parse, then one atomic
`store.replace_source`, re-hashing the file afterwards (`store.write_converged`). **This module
never imports `wilib` or `identity`** (plan3 section 3.1, test 12): the hook path stays offline.

* `sync_paths` replaces each named source whose sha256 or `parserVersion` differs from the stored one
  (or unconditionally with `force`); a covered path whose file is gone is deleted from the graph.
* `sync_all` is `sync_paths` over every discovered file plus the prune of sources no longer on disk.
* `rebuild` is the unconditional form: bootstrap, force-sync everything, prune, orphan sweep, stamp
  `SyncState`, then `parity` — a fresh parse compared with a graph export, never the graph with itself.

Everything here takes a `graph` (a `falkordb` graph or a fake with `query`); the caller connects, so
the ephemeral-endpoint guard of `store.connect` stays in one place.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import store
from .discover import Found, discover
from .parse.backlog import BacklogParseError, parse_backlog
from .parse.ledger import parse_ledger
from .parse.requirements import parse_requirements
from .parse.review import parse_review
from .parse.source import PARSER_VERSION
from .scope import load_legacy_paths

VOLATILE = frozenset({"syncedAt", "mtimeNs"})  # written by the store, not derived from the file

ORPHAN_SWEEPS = (
    ("Unit", "HAS_UNIT",
     "MATCH (u:Unit) OPTIONAL MATCH (s:Source)-[:HAS_UNIT]->(u) WITH u, s WHERE s IS NULL DELETE u RETURN count(u)"),
    ("Item", "HAS_ITEM",
     "MATCH (u:Item) OPTIONAL MATCH (s:Source)-[:HAS_ITEM]->(u) WITH u, s WHERE s IS NULL DELETE u RETURN count(u)"),
)
EXPORT = ("MATCH (s:Source) OPTIONAL MATCH (s)-[]->(c) "
          "RETURN s.path AS path, properties(s) AS meta, collect(properties(c)) AS kids")


# --- parsing one source -------------------------------------------------------------------------

@dataclass
class Parser:
    """What parsing any covered file needs, computed once per run."""

    root: Path
    found: dict[str, str]  # covered path -> kind
    reviews: frozenset[str]
    legacy_paths: Mapping[str, str]

    @classmethod
    def for_root(cls, root: str | Path) -> Parser:
        root = Path(root)
        found: list[Found] = discover(root)
        return cls(root, {f.path: f.kind for f in found},
                   frozenset(f.path for f in found if f.kind == "review"), load_legacy_paths(root))

    def parse_bytes(self, path: str, raw: bytes) -> tuple[str, dict, list[dict]]:
        """`(kind, meta, rows)` for a covered path's bytes; `BacklogParseError` for an unparseable backlog."""
        kind = self.found[path]
        if kind == "ledger":
            doc = parse_ledger(path, raw, set(self.reviews))
            return kind, doc.source_props(), doc.unit_props()
        if kind == "backlog":
            doc = parse_backlog(path, raw, self.legacy_paths)
            return kind, doc.source_props(), doc.item_props()
        doc = parse_review(path, raw) if kind == "review" else parse_requirements(path, raw)
        return kind, doc.source_props(), []

    def parse(self, path: str) -> tuple[str, dict, list[dict]]:
        return self.parse_bytes(path, (self.root / path).read_bytes())


def sha_of(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


# --- sync -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Outcome:
    path: str
    status: str  # synced | unchanged | deleted | failed | skipped
    detail: str = ""

    def render(self) -> str:
        return f"{self.status}\t{self.path}" + (f"\t{self.detail}" if self.detail else "")


def stored_sources(graph) -> dict[str, tuple[str | None, int | None]]:
    """`{path: (sha256, parserVersion)}` of every `:Source` in the graph."""
    rows = graph.query("MATCH (s:Source) RETURN s.path, s.sha256, s.parserVersion").result_set
    return {r[0]: (r[1], r[2]) for r in rows}


def sync_paths(graph, parser: Parser, paths: list[str], *, force: bool = False,
               stored: dict | None = None) -> list[Outcome]:
    """Sync each of `paths` (repo-relative). A failure on one path never stops the others."""
    stored = stored_sources(graph) if stored is None else stored
    out: list[Outcome] = []
    for path in paths:
        file = parser.root / path
        if path not in parser.found:
            if not file.is_file() and path in stored:
                out.append(_delete(graph, path))
            else:
                out.append(Outcome(path, "skipped", "not a covered file"))
            continue
        try:
            sha = sha_of(file)
            if not force and stored.get(path) == (sha, PARSER_VERSION):
                out.append(Outcome(path, "unchanged"))
                continue
            rounds = store.write_converged(graph, lambda p=path: parser.parse(p), lambda f=file: sha_of(f))
            out.append(Outcome(path, "synced", "" if rounds == 1 else f"{rounds} rounds"))
        except (BacklogParseError, store.StoreError, OSError) as exc:
            out.append(Outcome(path, "failed", str(exc)))
    return out


def _delete(graph, path: str) -> Outcome:
    try:
        store.delete_source(graph, path)
    except store.StoreError as exc:
        return Outcome(path, "failed", str(exc))
    return Outcome(path, "deleted", "file is gone")


def prune(graph, parser: Parser, stored: dict) -> list[Outcome]:
    """Delete every stored source whose path is not a covered file on disk (mirror kinds only)."""
    return [_delete(graph, p) for p in sorted(stored) if p not in parser.found]


def orphan_sweep(graph) -> dict[str, int]:
    """Children no `:Source` owns. Must be 0 everywhere; non-zero is a bug signal to report."""
    return {label: graph.query(q).result_set[0][0] for label, _rel, q in ORPHAN_SWEEPS}


def sync_changed(graph, parser: Parser) -> list[Outcome]:
    """Every covered file whose hash or parser version differs from the graph's; no prune."""
    return sync_paths(graph, parser, sorted(parser.found))


def sync_all(graph, parser: Parser, *, force: bool = False) -> list[Outcome]:
    """`sync_paths` over every covered file, then prune what left the disk (the routine crash repair)."""
    stored = stored_sources(graph)
    return sync_paths(graph, parser, sorted(parser.found), force=force, stored=stored) + prune(graph, parser, stored)


# --- parity (AC-2) -------------------------------------------------------------------------------

def canon_hash(props: Mapping) -> str:
    """sha256 over the sorted JSON of a property map; keeps absent distinct from `[]` and `''`."""
    return hashlib.sha256(json.dumps(props, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def child_key(props: Mapping) -> str:
    return str(props.get("key") if "key" in props else props.get("ref"))


@dataclass(frozen=True)
class Snapshot:
    """One source as parity compares it."""

    kind: str
    doc_status: str | None
    sha256: str
    row_count: int
    meta_hash: str
    children: dict[str, str]


def snapshot(meta: Mapping, kids: list[Mapping]) -> Snapshot:
    meta = {k: v for k, v in meta.items() if k not in VOLATILE}
    return Snapshot(str(meta.get("kind")), meta.get("docStatus"), str(meta.get("sha256")),
                    meta.get("rowCount", 0), canon_hash(meta), {child_key(k): canon_hash(k) for k in kids})


def expected_snapshots(parser: Parser) -> tuple[dict[str, Snapshot], list[Outcome]]:
    """The fresh parse in the form the store writes it, plus the files that did not parse."""
    out: dict[str, Snapshot] = {}
    failed: list[Outcome] = []
    for path, kind in parser.found.items():
        try:
            kind, meta, rows = parser.parse(path)
        except BacklogParseError as exc:
            failed.append(Outcome(path, "failed", str(exc)))
            continue
        written_meta, written_rows = store.write_form(kind, meta, rows)
        out[path] = snapshot(written_meta, written_rows)
    return out, failed


def actual_snapshots(graph) -> dict[str, Snapshot]:
    """One read of the whole mirror; each child hash is recomputed here, never read from `rowHash`."""
    rows = graph.query(EXPORT).result_set
    total = graph.query("MATCH (s:Source) RETURN count(s)").result_set[0][0]
    if len(rows) != total:  # a capped reply looks complete
        raise store.StoreError(f"export returned {len(rows)} rows for {total} sources")
    return {r[0]: snapshot(r[1], list(r[2])) for r in rows}


@dataclass
class Parity:
    expected: int
    actual: int
    diffs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.diffs

    def render(self) -> str:
        out = [f"parity: {self.expected} expected, {self.actual} in graph"]
        out += [f"DIFF {d}" for d in self.diffs]
        out.append("parity: " + ("ok" if self.ok else f"{len(self.diffs)} difference(s)"))
        return "\n".join(out)


def compare(expected: Mapping[str, Snapshot], actual: Mapping[str, Snapshot],
            unparsed: list[Outcome] = ()) -> Parity:
    diffs = [f"{o.path}: does not parse: {o.detail}" for o in unparsed]
    for path in sorted(set(expected) | set(actual)):
        want, got = expected.get(path), actual.get(path)
        if got is None:
            diffs.append(f"{path}: missing in graph")
        elif want is None:
            if path not in {o.path for o in unparsed}:
                diffs.append(f"{path}: in graph, not on disk")
        else:
            diffs += _diff_source(path, want, got)
    return Parity(len(expected), len(actual), diffs)


def _diff_source(path: str, want: Snapshot, got: Snapshot) -> list[str]:
    out = [f"{path}: {name} {getattr(got, attr)!r}, expected {getattr(want, attr)!r}"
           for name, attr in (("kind", "kind"), ("docStatus", "doc_status"), ("sha256", "sha256"),
                              ("rowCount", "row_count"))
           if getattr(want, attr) != getattr(got, attr)]
    if want.meta_hash != got.meta_hash and not out:
        out.append(f"{path}: source properties differ")
    for key in sorted(set(want.children) | set(got.children)):
        if key not in got.children:
            out.append(f"{path}: child {key!r} missing in graph")
        elif key not in want.children:
            out.append(f"{path}: child {key!r} in graph, not in the file")
        elif want.children[key] != got.children[key]:
            out.append(f"{path}: child {key!r} differs")
    return out


def parity(graph, parser: Parser) -> Parity:
    expected, unparsed = expected_snapshots(parser)
    return compare(expected, actual_snapshots(graph), unparsed)


# --- rebuild ----------------------------------------------------------------------------------------

@dataclass
class RebuildResult:
    outcomes: list[Outcome]
    orphans: dict[str, int]
    parity: Parity

    @property
    def ok(self) -> bool:
        return self.parity.ok and not any(o.status == "failed" for o in self.outcomes) \
            and not any(self.orphans.values())

    def render(self) -> str:
        counts: dict[str, int] = {}
        for o in self.outcomes:
            counts[o.status] = counts.get(o.status, 0) + 1
        out = [o.render() for o in self.outcomes if o.status in ("failed", "deleted")]
        out.append("rebuild: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
        out += [f"ORPHANS {label}: {n} deleted (a bug signal)" for label, n in self.orphans.items() if n]
        out.append(self.parity.render())
        return "\n".join(out)


def rebuild(db, graph, key: str, parser: Parser, *, now_ms: Callable[[], int] | None = None) -> RebuildResult:
    """Bootstrap, unconditional per-file replace, prune, orphan sweep, `SyncState`, parity (note 5.1)."""
    store.bootstrap(db, graph, key)
    outcomes = sync_all(graph, parser, force=True)
    orphans = orphan_sweep(graph)
    stamp = now_ms() if now_ms else int(time.time() * 1000)
    graph.query("MERGE (m:SyncState {id:'mirror'}) SET m.lastRebuildAt = $now, m.parserVersion = $pv",
                {"now": stamp, "pv": PARSER_VERSION})
    return RebuildResult(outcomes, orphans, parity(graph, parser))
