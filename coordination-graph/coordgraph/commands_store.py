"""Plain command functions for the store-backed subcommands (wired into the CLI by S3b).

Each takes an `argparse.Namespace`-like object and returns an exit code. Optional attributes
(read with `getattr`): `repo_root`, `graph` (overrides `COORDGRAPH_KEY`), `manifest` (a path to
write the `(path, sha256)` manifest to).
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import dryrun, store
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
