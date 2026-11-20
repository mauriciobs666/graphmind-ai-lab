"""Plain command functions for the store-backed subcommands, wired into the CLI.

Each takes an `argparse.Namespace`-like object and returns an exit code. Optional attributes
(read with `getattr`): `repo_root`, `graph` (overrides `COORDGRAPH_KEY`), `manifest` (a path to
write the `(path, sha256)` manifest to). Every connection goes through `store.connect`, so only an
explicitly announced ephemeral FalkorDB is ever reached. `sync` and `rebuild` import no `wilib`; the
read commands (`drift`, `query`) load `identity` lazily for the header's registry line.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import drift, dryrun, store, sync
from .config import Config
from .scope import load_legacy_paths


def bootstrap(args) -> int:
    """`coordgraph bootstrap`: indexes, UNIQUE constraints, `SyncState`; idempotent. Ephemeral targets only."""
    cfg = Config.from_env()
    key = getattr(args, "graph", None) or cfg.key
    try:
        db, graph = store.connect(cfg.falkordb_host, cfg.falkordb_port, key)
        result = store.bootstrap(db, graph, key)
    except store.StoreError as exc:
        print(f"bootstrap: {exc}", file=sys.stderr)
        return 1
    for label, prop in result.created_indexes:
        print(f"created index {label}.{prop}")
    for label, prop in result.created_constraints:
        print(f"created UNIQUE constraint {label}.{prop}")
    print("bootstrap: " + ("done" if result.changed else "nothing to do"))
    return 0


def rebuild_dry_run(args) -> int:
    """`coordgraph rebuild --dry-run`: offline validation (note section 5.5 step 1); no connection."""
    root = Path(getattr(args, "repo_root", None) or Config.from_env().repo_root)
    report = dryrun.rebuild_dry_run(root, load_legacy_paths(root))
    print(report.render())
    manifest = getattr(args, "manifest", None)
    if manifest:
        Path(manifest).write_text("".join(f"{sha}  {path}\n" for path, sha in report.manifest), encoding="utf-8")
    return 0 if report.ok else 1


def _target(args):
    """`(cfg, key, root, db, graph)`; raises `store.StoreError` (incl. `RefusedEndpoint`) or a connection error."""
    cfg = Config.from_env()
    key = getattr(args, "graph", None) or cfg.key
    root = Path(getattr(args, "repo_root", None) or cfg.repo_root)
    db, graph = store.connect(cfg.falkordb_host, cfg.falkordb_port, key)
    return cfg, key, root, db, graph


def _fail(name: str, exc: Exception) -> int:
    print(f"{name}: {exc}", file=sys.stderr)
    return 1


def _relative(root: Path, raw: str) -> str:
    p = Path(raw)
    return p.resolve().relative_to(root.resolve()).as_posix() if p.is_absolute() else p.as_posix()


def sync_cmd(args) -> int:
    """`coordgraph sync [PATH...] | --changed | --all`: parse and replace each source whose hash moved."""
    try:
        cfg, _key, root, _db, graph = _target(args)
        parser = sync.Parser.for_root(root)
        if args.all:
            outcomes = sync.sync_all(graph, parser)
        elif args.changed:
            outcomes = sync.sync_changed(graph, parser)
        else:
            outcomes = sync.sync_paths(graph, parser, [_relative(root, p) for p in args.paths])
    except Exception as exc:  # noqa: BLE001 - a refused or unreachable endpoint is one clear line
        return _fail("sync", exc)
    for o in outcomes:
        if args.verbose or o.status not in ("unchanged",):
            print(o.render())
    failed = sum(o.status == "failed" for o in outcomes)
    print(f"sync: {len(outcomes)} path(s), {failed} failed", file=sys.stderr)
    return 1 if failed else 0


def rebuild_cmd(args) -> int:
    """`coordgraph rebuild [--dry-run]`: unconditional per-file replace, prune, parity (note section 5.1)."""
    if getattr(args, "dry_run", False):
        return rebuild_dry_run(args)
    try:
        cfg, key, root, db, graph = _target(args)
        result = sync.rebuild(db, graph, key, sync.Parser.for_root(root))
    except Exception as exc:  # noqa: BLE001
        return _fail("rebuild", exc)
    print(result.render())
    return 0 if result.ok else 1


def _hook_active(cfg: Config) -> bool:
    return not (cfg.hook_disabled or (cfg.state_dir / "disabled").exists())


def _registry_line(root: Path) -> str:
    from .identity import load_identity  # noqa: PLC0415 - read side only, deliberately lazy

    return load_identity(root).registry_line()


def drift_cmd(args) -> int:
    """`coordgraph drift [PATH...]`: the sync header; exit 1 when any covered file differs from the graph."""
    try:
        cfg, _key, root, _db, graph = _target(args)
        paths = [_relative(root, p) for p in args.paths] or None
        report = drift.check(graph, sync.Parser.for_root(root), paths)
    except Exception as exc:  # noqa: BLE001
        return _fail("drift", exc)
    print(drift.render_header(report, _hook_active(cfg), _registry_line(root)))
    return 0 if report.clean else 1


def query_cmd(args) -> int:
    """`coordgraph query '<cypher>' [--covers PATH...]`: `GRAPH.RO_QUERY` under the sync header."""
    try:
        cfg, _key, root, _db, graph = _target(args)
        covers = [_relative(root, p) for p in args.covers] or None
        report = drift.check(graph, sync.Parser.for_root(root), covers)
        result = graph.ro_query(args.cypher)
    except Exception as exc:  # noqa: BLE001 - includes a write attempt, which RO_QUERY refuses
        return _fail("query", exc)
    print(drift.render_header(report, _hook_active(cfg), _registry_line(root)))
    print("\t".join(str(h[1] if isinstance(h, (list, tuple)) else h) for h in result.header))
    for row in result.result_set:
        print("\t".join(str(v) for v in row))
    return 0
