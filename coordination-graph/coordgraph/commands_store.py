"""Plain command functions for the store-backed subcommands, wired into the CLI.

Each takes an `argparse.Namespace`-like object and returns an exit code. Optional attributes
(read with `getattr`): `repo_root`, `graph` (overrides `COORDGRAPH_KEY`), `manifest` (a path to
write the `(path, sha256)` manifest to), `confirm` (`--confirm GRAPHKEY`). Every connection goes through
`store.open_graph`: an explicitly announced ephemeral FalkorDB by default, the shared instance only when a
confirmation equal to the graph key is given (`--confirm`, or the standing `COORDGRAPH_SHARED_CONFIRM` for the
routine commands; `bootstrap`, `rebuild` and `kb-reconcile` accept the flag only). `sync` and `rebuild` import no
`wilib`; the read commands (`drift`, `query`) load `identity` lazily for the header's registry line.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import drift, dryrun, store, sync
from .config import Config
from .scope import load_legacy_paths


def confirmation(args, cfg: Config, *, flag_only: bool = False) -> str | None:
    """The shared-instance confirmation for this run, or `None` (ephemeral mode).

    The command-line flag always wins. The standing env value (`COORDGRAPH_SHARED_CONFIRM`, what the
    non-interactive hook sets) authorizes only the routine commands: a `flag_only` command (the first load,
    the destructive rebuild, the KB re-seed) refuses it rather than falling back to ephemeral mode.
    """
    flag = getattr(args, "confirm", None)
    if flag is not None:
        return flag
    if cfg.shared_confirm is not None and flag_only:
        raise store.RefusedEndpoint(
            "refusing to act on COORDGRAPH_SHARED_CONFIRM alone: this command needs "
            f"--confirm {cfg.shared_confirm!r} on the command line"
        )
    return cfg.shared_confirm


def bootstrap(args) -> int:
    """`coordgraph bootstrap`: indexes, UNIQUE constraints, `SyncState`; idempotent. Shared only with `--confirm`."""
    cfg = Config.from_env()
    key = getattr(args, "graph", None) or cfg.key
    try:
        confirm = confirmation(args, cfg, flag_only=True)
        db, graph = store.open_graph(cfg.falkordb_host, cfg.falkordb_port, key, confirm)
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


def _target(args, *, flag_only: bool = False):
    """`(cfg, key, root, db, graph)`; raises `store.StoreError` (incl. `RefusedEndpoint`) or a connection error."""
    cfg = Config.from_env()
    key = getattr(args, "graph", None) or cfg.key
    root = Path(getattr(args, "repo_root", None) or cfg.repo_root)
    confirm = confirmation(args, cfg, flag_only=flag_only)
    db, graph = store.open_graph(cfg.falkordb_host, cfg.falkordb_port, key, confirm)
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
        cfg, key, root, db, graph = _target(args, flag_only=True)
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
