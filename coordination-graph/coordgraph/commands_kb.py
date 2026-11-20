"""Command functions for the KB mirror subcommands (`kb-sync`, `kb-status`, `kb-reconcile`, `kb-finish`).

Exit codes, the same for all four commands: 0 everything settled, 2 not complete (a section pending, refused or
failed, or for `kb-status` anything flagged: the hook turns this into its "say so" message), 1 the command itself
could not run. The mirror graph goes through `store.connect` (ephemeral targets only) and falkor-chat's REST
endpoint must be a loopback, non-shared one too (`store.require_ephemeral_kb_url`); its graph is read by read-only
queries on `$COORDGRAPH_KB_GRAPH` through the same connection. Imports no `wilib`.
"""
from __future__ import annotations

import fcntl
import subprocess
import sys
from pathlib import Path

from . import kbsync, store
from .config import Config
from .discover import kb_files
from .parse.kb import KbDoc, parse_kb


def _build(args):
    """`(cfg, root, db, graph)` for the mirror; raises `store.StoreError` (incl. `RefusedEndpoint`)."""
    cfg = Config.from_env()
    key = getattr(args, "graph", None) or cfg.key
    root = Path(getattr(args, "repo_root", None) or cfg.repo_root)
    db, graph = store.connect(cfg.falkordb_host, cfg.falkordb_port, key)
    return cfg, root, db, graph


def _engine(cfg: Config, db, graph) -> kbsync.KbSync:
    store.require_ephemeral_kb_url(cfg.kb_url)  # the mirror is ephemeral-only: so must its falkor-chat be
    http = kbsync.KbHttp(cfg.kb_url, cfg.kb_http_timeout_s)
    return kbsync.KbSync(graph, http, kbsync.Observer(db.select_graph(cfg.kb_graph)), budget_s=cfg.kb_budget_s)


def load_docs(root: Path) -> tuple[list[KbDoc], list[kbsync.KbOutcome]]:
    """Every registered KB file, parsed. An unreadable one becomes a `failed` outcome, not an abort."""
    docs, problems = [], []
    for rel in kb_files(root):
        try:
            docs.append(parse_kb(rel, (root / rel).read_bytes()))
        except OSError as exc:
            problems.append(kbsync.KbOutcome(rel, "(file)", "failed", f"unreadable: {exc}"))
    return docs, problems


def _fail(name: str, exc: Exception) -> int:
    print(f"{name}: {exc}", file=sys.stderr)
    return 1


def _relative(root: Path, raw: str) -> str:
    p = Path(raw)
    return p.resolve().relative_to(root.resolve()).as_posix() if p.is_absolute() else p.as_posix()


def _report(outcomes: list[kbsync.KbOutcome], name: str, verbose: bool = False) -> int:
    for o in outcomes:
        if verbose or o.status != "unchanged":
            print(o.render())
    bad = [o for o in outcomes if o.incomplete]
    print(f"{name}: {len(outcomes)} section(s), {len(bad)} not complete", file=sys.stderr)
    if bad:
        print(f"{name}: this update is not complete — say so; it is completed by `coordgraph kb-finish` or "
              "retried by the next `kb-sync`", file=sys.stderr)
    return 2 if bad else 0


def detach_finish(cfg: Config, popen=subprocess.Popen) -> None:
    """Launch `coordgraph kb-finish` detached, never holding the caller's pipes (plan 3.2, P2-M1)."""
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    log = open(cfg.state_dir / "hook.log", "ab")  # noqa: SIM115 - handed to the child
    popen([sys.executable, "-m", "coordgraph", "kb-finish"], start_new_session=True, stdin=subprocess.DEVNULL,
          stdout=log, stderr=log)


def kb_sync_cmd(args) -> int:
    """`coordgraph kb-sync [PATH...]`: every registered KB file, or the named ones."""
    try:
        cfg, root, db, graph = _build(args)
        engine = _engine(cfg, db, graph)
        docs, problems = load_docs(root)
        rows, _ = engine.read_all()
        ctx = kbsync.Context.build(docs, rows)
        wanted = {_relative(root, p) for p in args.paths}
        outcomes = [o for o in problems if not wanted or o.path in wanted]
        unknown = wanted - {d.path for d in docs}
        outcomes += [kbsync.KbOutcome(p, "(file)", "failed", "not a registered KB file") for p in sorted(unknown)]
    except Exception as exc:  # noqa: BLE001 - a refused or unreachable endpoint is one clear line
        return _fail("kb-sync", exc)
    for doc in docs:
        if not wanted or doc.path in wanted:
            try:
                outcomes += engine.sync_file(doc, ctx, rows)
            except Exception as exc:  # noqa: BLE001 - one file's unforeseen error must not lose the others' outcomes
                outcomes.append(kbsync.KbOutcome(doc.path, "(file)", "failed", f"{type(exc).__name__}: {exc}"))
    code = _report(outcomes, "kb-sync", getattr(args, "verbose", False))
    if code and getattr(args, "detach_finish", False) and any(o.status == "pending" for o in outcomes):
        detach_finish(cfg)
    return code


def kb_status_cmd(args) -> int:
    """`coordgraph kb-status [--ids ID...]`: last sync time and what is stale, pending, unmirrored or duplicated."""
    try:
        cfg, root, db, graph = _build(args)
        engine = _engine(cfg, db, graph)
        docs, problems = load_docs(root)
        rows, _ = engine.read_all()
        files = engine._q(kbsync.READ_FILES)  # noqa: SLF001 - one read, no reason for a second API
        last = max((int(r[2]) for r in files if r[2] is not None), default=None)
        report = kbsync.status(docs, rows, last, args.ids or None)
    except Exception as exc:  # noqa: BLE001
        return _fail("kb-status", exc)
    print(report.header())
    for p in problems:
        print(p.render())
    for line in report.lines:
        print(line.render())
    return 0 if report.clean and not problems else 2


def kb_reconcile_cmd(args) -> int:
    """`coordgraph kb-reconcile`: mapping vs falkor-chat (re-ingest, adopt, report non-current and orphans)."""
    try:
        cfg, root, db, graph = _build(args)
        engine = _engine(cfg, db, graph)
        docs, outcomes = load_docs(root)
        outcomes += engine.reconcile(docs)
    except Exception as exc:  # noqa: BLE001
        return _fail("kb-reconcile", exc)
    for o in outcomes:
        if o.status != "unchanged":
            print(o.render())
    reported = [o for o in outcomes if o.status in ("pending", "refused", "failed", "orphan", "non-current")]
    print(f"kb-reconcile: {len(outcomes)} result(s), {len(reported)} to review", file=sys.stderr)
    return 2 if reported else 0


def kb_finish_cmd(args) -> int:
    """`coordgraph kb-finish [--once]`: complete or give up every pending section; one instance at a time."""
    try:
        cfg, _root, db, graph = _build(args)
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.state_dir / "kb-finish.lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("kb-finish: another kb-finish is running", file=sys.stderr)
                return 0
            outcomes = _engine(cfg, db, graph).finish(once=getattr(args, "once", False))
    except Exception as exc:  # noqa: BLE001
        return _fail("kb-finish", exc)
    return _report(outcomes, "kb-finish")
